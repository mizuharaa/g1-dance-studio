"""Upload-time video quality gate: score a reference dance clip on a 1-10 rubric so the
operator knows, BEFORE spending GPU on extract/train, whether the footage is good enough
and how hard the dance will be to reproduce on the G1.

Everything here is CPU-only (imageio-ffmpeg + numpy + scipy) — runs on the laptop in a few
seconds. Dimensions (each scored 1-10, higher = better, except `difficulty`):

  framerate            - smooth capture; low/odd/VFR fps hurts pose extraction
  resolution           - detail for the pose estimator; a small subject = bad keypoints
  lighting             - brightness / contrast / no clipping / no flicker
  sharpness_snappy     - crisp frames vs motion blur ("snappy"); blur wrecks pose tracking
  movement_feasibility - proxy for the robot's hard limits: how much the dancer TRAVELS
                         (>2 m area gets vet-rejected) and how FAST they move (joint-vel limit)
  difficulty (1-10)    - how hard the DANCE is (speed + burstiness + range); NOT a quality score

Plus an overall_score (1-10, mean of the quality dims) and a verdict + blockers/warnings.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ---- pure scoring helpers (unit-tested; no video needed) --------------------------------
def _lerp_score(x: float, lo: float, hi: float, invert: bool = False) -> float:
    """Map x in [lo, hi] -> 1..10 (clamped). invert: smaller x scores higher."""
    if hi == lo:
        return 5.0
    t = (x - lo) / (hi - lo)
    t = min(1.0, max(0.0, t))
    if invert:
        t = 1.0 - t
    return round(1.0 + 9.0 * t, 1)


def score_framerate(fps: float) -> dict:
    flag = None
    if fps and abs(fps - round(fps)) > 0.2:
        flag = f"odd/variable rate ({fps:.1f} fps) — may be VFR; pose extraction assumes CFR"
    s = _lerp_score(fps, 15, 30)
    if fps >= 30:
        s = 10.0
    note = "smooth" if fps >= 30 else "usable" if fps >= 24 else "low — choppy capture hurts extraction"
    return {"score": s, "value": f"{fps:.1f} fps", "note": note, "flag": flag}


def score_resolution(w: int, h: int) -> dict:
    md = min(w, h)
    s = _lerp_score(md, 360, 1080)
    if md >= 1080:
        s = 10.0
    note = ("plenty of detail" if md >= 720 else "low — the dancer may be too few pixels for good keypoints"
            if md < 480 else "acceptable")
    return {"score": s, "value": f"{w}x{h}", "note": note,
            "flag": None if md >= 360 else "too low-res for reliable pose extraction"}


def score_lighting(brightness: float, contrast: float, clip_lo: float, clip_hi: float, flicker: float) -> dict:
    # ideal mean luma ~90-160; contrast (std) higher is better up to ~60; penalise clipping + flicker
    bright_pen = min(abs(brightness - 125) / 90, 1.0)             # 0 good .. 1 bad
    contrast_s = min(contrast / 55, 1.0)                          # 0 flat .. 1 good
    clip_pen = min((clip_lo + clip_hi) / 0.25, 1.0)              # >25% clipped = worst
    flicker_pen = min(flicker / 25, 1.0)
    quality = contrast_s * (1 - bright_pen) * (1 - 0.6 * clip_pen) * (1 - 0.5 * flicker_pen)
    s = round(1.0 + 9.0 * min(1.0, max(0.0, quality)), 1)
    notes = []
    if brightness < 70: notes.append("underexposed")
    elif brightness > 185: notes.append("overexposed")
    if contrast < 25: notes.append("flat/low contrast")
    if clip_lo + clip_hi > 0.12: notes.append("clipped shadows/highlights")
    if flicker > 18: notes.append("inconsistent lighting/flicker")
    return {"score": s, "value": f"luma {brightness:.0f}, contrast {contrast:.0f}",
            "note": ", ".join(notes) or "well lit", "flag": None}


def score_sharpness(sharpness: float, motion: float) -> dict:
    s = _lerp_score(sharpness, 40, 400)
    blur = sharpness < 120 and motion > 12
    note = "crisp / snappy" if sharpness >= 250 else "soft" if sharpness < 120 else "acceptable"
    if blur:
        note = "motion blur on the fast moves — will soften the robot's tracking"
    return {"score": s, "value": f"sharpness {sharpness:.0f}", "note": note,
            "flag": "heavy motion blur" if sharpness < 70 else None}


def score_feasibility(travel: float, move_pct: float) -> dict:
    """ESTIMATE only (pixel motion, no pose). travel = drift of the moving region across the
    frame (dancer walking around); move_pct = % of frame changing during active moments. The
    authoritative feasibility check is the retarget vet gate (real joint vel + excursion)."""
    travel_pen = min(travel / 0.30, 1.0)                  # big frame-travel => may exceed the 2 m area
    fast_pen = min(max(move_pct - 12, 0) / 15, 1.0)       # very fast => joint-velocity-limit risk
    s = round(1.0 + 9.0 * (1 - max(travel_pen, fast_pen)), 1)
    notes = []
    if travel_pen > 0.55: notes.append("dancer moves around the frame — may exceed the ~2 m in-place area")
    if fast_pen > 0.55: notes.append("very fast footage — watch the joint-velocity limit")
    return {"score": s, "value": f"travel {travel:.2f}, motion {move_pct:.0f}%",
            "note": (", ".join(notes) + " (estimate — vet gate is authoritative)") if notes
                    else "appears to stay roughly in place (estimate — vet gate is authoritative)",
            "flag": None}


def score_difficulty(move_pct: float, burstiness: float, move_range: float) -> dict:
    """1-10 difficulty of the DANCE (not a quality score). Estimated from footage motion:
    more/ burstier/ wider-range motion = harder to retarget, train, and keep balanced."""
    energy = min(move_pct / 14, 1.0)
    complexity = min(burstiness / 1.2, 1.0)
    rng = min(move_range / 10, 1.0)
    d = 0.5 * energy + 0.3 * complexity + 0.2 * rng
    score = round(1.0 + 9.0 * min(1.0, d), 1)
    band = ("gentle" if score < 3.5 else "moderate" if score < 6 else "hard" if score < 8 else "extreme")
    return {"score": score, "value": band,
            "note": f"{band} — estimated from footage motion; faster/complex dances are harder to "
                    "retarget, train, and balance"}


def summarize(dims: dict, difficulty: dict) -> dict:
    quality_keys = ["framerate", "resolution", "lighting", "sharpness_snappy", "movement_feasibility"]
    overall = round(sum(dims[k]["score"] for k in quality_keys) / len(quality_keys), 1)
    # a BLOCKER is a dimension scored so low the clip is unusable; a FLAG is advisory only
    # (e.g. an odd/VFR framerate on an otherwise great clip) and must NOT demote the verdict.
    blockers = [f"{k.replace('_', ' ')}: {dims[k]['note']}" for k in quality_keys if dims[k]["score"] < 3]
    flags = [f"{k.replace('_', ' ')}: {dims[k]['flag']}" for k in quality_keys if dims[k].get("flag")]
    warnings = [f"{k.replace('_', ' ')} {dims[k]['score']}/10 — {dims[k]['note']}"
                for k in quality_keys if 3 <= dims[k]["score"] < 5]
    verdict = ("poor" if (overall < 5 or blockers) else
               "good" if overall >= 7 else "acceptable")
    rec = {"good": "Good to proceed with extract/train.",
           "acceptable": "Usable, but the flagged dimensions will cap fidelity — review before spending GPU.",
           "poor": "Below bar — a better clip (fix the blockers) will train much better."}[verdict]
    return {"overall_score": overall, "verdict": verdict, "recommendation": rec,
            "blockers": blockers, "flags": flags, "warnings": warnings}


# ---- video reading + feature extraction -------------------------------------------------
def _gray_small(frame: np.ndarray, target_w: int = 256) -> np.ndarray:
    if frame.ndim == 3:
        g = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
    else:
        g = frame.astype(float)
    k = max(1, g.shape[1] // target_w)
    return g[::k, ::k].astype(np.float32)


def _laplace_var(g: np.ndarray) -> float:
    from scipy import ndimage
    return float(ndimage.laplace(g).var())


def analyze(path: str | Path, n_static: int = 36) -> dict:
    """Full rubric for a video. Never raises: on an unreadable file returns a 'unreadable'
    verdict so the upload flow can surface it instead of crashing."""
    import imageio.v2 as imageio
    from collections import deque
    path = str(path)
    try:                                     # native meta (true fps/resolution/duration)
        mr = imageio.get_reader(path)
        meta = mr.get_meta_data(); mr.close()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "verdict": "unreadable", "overall_score": 0,
                "recommendation": f"could not open the video ({type(e).__name__}) — is it a valid clip?",
                "blockers": ["file is not a readable video"], "warnings": [], "dimensions": {}}

    fps = float(meta.get("fps") or 0) or 30.0
    w, h = (meta.get("size") or (0, 0))
    dur = float(meta.get("duration") or 0)

    # Decode DOWNSCALED (huge frames are ~50x slower) — fine for brightness/sharpness/motion stats.
    try:
        reader = imageio.get_reader(path, "ffmpeg", size=(256, 256))
    except Exception:  # noqa: BLE001 — size unsupported on some builds
        reader = imageio.get_reader(path)

    bright, contrast, sharp, clip_lo, clip_hi, frame_means = [], [], [], [], [], []
    fracs, cx, cy = [], [], []
    stride = max(1, int(round(fps / 8)))      # ~0.12 s gap: accumulates real motion vs per-frame noise
    MAXF = 3000
    buf: deque = deque(maxlen=stride + 1)
    i = 0
    for frame in reader:
        if i >= MAXF:
            break
        g = _gray_small(frame)
        if i % 8 == 0:                        # static metrics (evenly across the clip)
            bright.append(float(g.mean())); contrast.append(float(g.std()))
            sharp.append(_laplace_var(g)); frame_means.append(float(g.mean()))
            clip_lo.append(float((g < 16).mean())); clip_hi.append(float((g > 240).mean()))
        buf.append(g)
        if len(buf) == stride + 1 and buf[0].shape == g.shape and i % 2 == 0:
            d = np.abs(g - buf[0])
            fracs.append(float((d > 12).mean()) * 100.0)     # % of frame in motion
            ys, xs = np.nonzero(d > 25)
            if xs.size:
                cx.append(xs.mean() / d.shape[1]); cy.append(ys.mean() / d.shape[0])
        i += 1
    try:
        reader.close()
    except Exception:  # noqa: BLE001
        pass

    if not bright:
        return {"ok": False, "verdict": "unreadable", "overall_score": 0,
                "recommendation": "no frames could be decoded", "blockers": ["no decodable frames"],
                "warnings": [], "dimensions": {}}

    brightness = float(np.mean(bright)); contrast_v = float(np.mean(contrast))
    sharpness = float(np.mean(sharp)); flicker = float(np.std(frame_means))
    clo, chi = float(np.mean(clip_lo)), float(np.mean(clip_hi))
    fa = np.array(fracs) if fracs else np.array([0.0])
    move_active = float(np.percentile(fa, 90))                # motion during the busier moments
    burstiness = float(fa.std() / (fa.mean() + 1e-6))
    move_range = float(np.percentile(fa, 90) - np.median(fa))
    travel = float(np.std(cx) + np.std(cy)) if len(cx) > 3 else 0.0

    dims = {
        "framerate": score_framerate(fps),
        "resolution": score_resolution(int(w), int(h)),
        "lighting": score_lighting(brightness, contrast_v, clo, chi, flicker),
        "sharpness_snappy": score_sharpness(sharpness, move_active),
        "movement_feasibility": score_feasibility(travel, move_active),
    }
    difficulty = score_difficulty(move_active, burstiness, move_range)
    out = {"ok": True, "duration_s": round(dur, 1), "dimensions": dims, "difficulty": difficulty}
    out.update(summarize(dims, difficulty))
    return out
