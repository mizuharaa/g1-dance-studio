"""Audit: does clean_motion() reject/attenuate GENUINE dance moves?

User concern (2026-07-20): the cleaning stage (accel-spike rejection + SG
smoothing) might be silently deleting essential choreography — and nobody can
eye-check every joint of every dance. Three legs, each with ground truth:

  A. SYNTHETIC bench (known truth):
     A1 sine sweep 0.5-8 Hz         -> amplitude retention vs frequency
     A2 raised-cosine "hits" 80-400ms -> peak retention vs hit duration
     A3 injected 1-frame glitches on a moving base -> removal vs collateral
  B. REAL Thriller (raw retarget thriller_g1.csv):
     - every frame the spike rejector fires on, classified by glitch signature:
       glitch = the sample deviates from the neighbour-spline and RETURNS
       (net displacement across the flagged run ~ 0); a genuine fast move
       CONTINUES. Score = |raw - spline_from_neighbours| / local move scale.
     - per-beat-window, per-joint-group amplitude + peak-velocity retention
       cleaned vs raw (the metric that would catch a washed-out handswing).
  C. ROBOT-BANDWIDTH cross-check: Welch PSD of the real measured q from the
     live native-speed run (robot executed the UNFILTERED motion, 2026-07-08
     era policies) vs raw vs cleaned reference. If the robot's own rolloff is
     below the filter's, the filter cannot be destroying anything the hardware
     could physically express.

Output: audit_result.json + retention_curves.png (committed, raw).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.signal import welch  # noqa: E402

from pipeline.motion_io import load_motion_csv  # noqa: E402
from tools.motion_quality import (FPS, SG_WINDOW, _spike_mask,  # noqa: E402
                                  clean_motion)

OUT = Path(__file__).resolve().parent
RAW = ROOT / "data/motions/thriller/thriller_g1.csv"
TEL = ROOT / "data/telemetry/20260710-145111_ground-run-legodom.npz"
FPS_TEL = 50.0

# native-time hard beats (falls / peak demand) + a handswing showcase section
WINDOWS = [(13.0, 18.0), (25.0, 36.0), (2.0, 8.0), (40.0, 48.0)]
GROUPS = {
    "legs": list(range(12)),
    "waist": [12, 13, 14],
    "arms": list(range(15, 29)),
}


def wrap36(dof: np.ndarray) -> np.ndarray:
    """Embed a (N,29)-or-fewer dof track in a static (N,36) motion so
    clean_motion can run on it."""
    n, d = dof.shape
    m = np.zeros((n, 36))
    m[:, 6] = 1.0  # unit quat w (xyzw col 3:7 -> w at col 6)
    m[:, 2] = 0.79
    m[:, 7:7 + d] = dof
    return m


# ---------------- A. synthetic ------------------------------------------------
def bench_sines() -> dict:
    t = np.arange(0, 20, 1 / FPS)
    freqs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    dof = np.zeros((len(t), 29))
    for j, f in enumerate(freqs):
        dof[:, j] = 0.5 * np.sin(2 * np.pi * f * t)
    cleaned, _ = clean_motion(wrap36(dof))
    ret = {}
    for j, f in enumerate(freqs):
        ret[f"{f}Hz"] = round(float(
            np.ptp(cleaned[:, 7 + j]) / np.ptp(dof[:, j])), 3)
    return ret


def bench_hits() -> dict:
    """Raised-cosine pulse = an isolated dance 'hit' (punch/snap). Peak
    retention after cleaning, by pulse full-width."""
    t = np.arange(0, 6, 1 / FPS)
    widths_ms = [80, 133, 200, 267, 400]
    dof = np.zeros((len(t), 29))
    for j, w in enumerate(widths_ms):
        w_s = w / 1000.0
        c = 3.0
        s = np.clip((t - (c - w_s / 2)) / w_s, 0, 1)
        dof[:, j] = 0.8 * 0.5 * (1 - np.cos(2 * np.pi * s)) * ((t > c - w_s / 2) & (t < c + w_s / 2))
    cleaned, info = clean_motion(wrap36(dof))
    return {
        "outliers_fired_on_pure_hits": info["outlier_frames_replaced"],
        "peak_retention": {
            f"{w}ms": round(float(cleaned[:, 7 + j].max() / dof[:, j].max()), 3)
            for j, w in enumerate(widths_ms)},
    }


def bench_glitch() -> dict:
    """1-frame flips injected on a MOVING base (2 Hz dance) — removal rate and
    collateral damage away from the glitches."""
    rng = np.random.default_rng(7)
    t = np.arange(0, 20, 1 / FPS)
    dof = np.zeros((len(t), 29))
    base = 0.5 * np.sin(2 * np.pi * 2.0 * t)
    for j in range(8):
        dof[:, j] = base
    g_frames = rng.integers(30, len(t) - 30, size=12)
    for k, f in enumerate(g_frames):
        dof[f, k % 8] += (0.4 + 0.5 * rng.random()) * (1 if k % 2 else -1)
    cleaned, info = clean_motion(wrap36(dof))
    resid = np.abs(cleaned[g_frames, 7 + np.arange(len(g_frames)) % 8]
                   - base[g_frames])
    off = np.ones(len(t), bool)
    for f in g_frames:
        off[max(0, f - 3):f + 4] = False
    collateral = float(np.sqrt(np.mean(
        (cleaned[off][:, 7:15] - dof[off][:, :8]) ** 2)))
    return {
        "glitches_injected": int(len(g_frames)),
        "outlier_frames_detected": info["outlier_frames_replaced"],
        "residual_at_glitch_rad": {
            "mean": round(float(resid.mean()), 4),
            "max": round(float(resid.max()), 4)},
        "collateral_rms_off_glitch_rad": round(collateral, 4),
    }


# ---------------- B. real Thriller -------------------------------------------
def real_motion_audit() -> dict:
    m = load_motion_csv(RAW)
    cleaned, info = clean_motion(m)
    dof = m[:, 7:]
    t = np.arange(len(m)) / FPS

    # classify every flagged sample: glitch vs genuine-move
    cols = np.concatenate([m[:, 0:3], dof], axis=1)
    mask = _spike_mask(cols, FPS)[:, 3:]          # joints only
    idx = np.arange(len(m))
    flagged = []
    for d in np.flatnonzero(mask.any(axis=0)):
        bad = mask[:, d]
        good = ~bad
        spl = CubicSpline(idx[good], dof[good, d])(idx[bad])
        dev = np.abs(dof[bad, d] - spl)
        # local motion scale: p95 |frame delta| of that joint in +-1 s
        for f, dv in zip(idx[bad], dev):
            lo, hi = max(0, f - 30), min(len(m), f + 30)
            scale = np.percentile(np.abs(np.diff(dof[lo:hi, d])), 95) + 1e-6
            flagged.append({
                "t_s": round(float(f / FPS), 2), "dof": int(d),
                "dev_rad": round(float(dv), 3),
                "dev_over_local_scale": round(float(dv / scale), 1),
            })
    # a rejection is SUSPICIOUS if the removed deviation is small relative to
    # local motion (could be genuine texture) — glitches are many x local scale
    suspicious = [f for f in flagged if f["dev_over_local_scale"] < 3.0]

    # per-window retention
    ret = {}
    for (w0, w1) in WINDOWS:
        s = (t >= w0) & (t <= w1)
        row = {}
        for g, jj in GROUPS.items():
            a_raw = [np.ptp(dof[s, j]) for j in jj]
            a_cln = [np.ptp(cleaned[s, 7 + j]) for j in jj]
            keep = [i for i, a in enumerate(a_raw) if a > 0.05]
            amp = (np.mean([a_cln[i] / a_raw[i] for i in keep])
                   if keep else float("nan"))
            v_raw = np.abs(np.diff(dof[s][:, jj], axis=0)).max() * FPS
            v_cln = np.abs(np.diff(cleaned[s][:, [7 + j for j in jj]],
                                   axis=0)).max() * FPS
            row[g] = {"amp_retention": round(float(amp), 3),
                      "peakvel_retention": round(float(v_cln / v_raw), 3)}
        ret[f"{w0}-{w1}s"] = row
    return {
        "file": str(RAW), "clean_info": info,
        "flagged_samples": len(flagged),
        "suspicious_rejections(dev<3x_local_scale)": suspicious,
        "window_retention": ret,
    }


# ---------------- C. robot bandwidth ------------------------------------------
def robot_bandwidth() -> dict:
    m = load_motion_csv(RAW)
    cleaned, _ = clean_motion(m)
    d = np.load(TEL, allow_pickle=True)
    q = d["q"]
    out = {}
    # joints: left shoulder pitch (handswing), left hip pitch, left ankle pitch
    for name, j in [("l_shoulder_pitch", 15), ("l_hip_pitch", 0),
                    ("l_ankle_pitch", 4)]:
        f_r, P_r = welch(m[:, 7 + j], fs=FPS, nperseg=256)
        f_c, P_c = welch(cleaned[:, 7 + j], fs=FPS, nperseg=256)
        f_q, P_q = welch(q[:, j], fs=FPS_TEL, nperseg=512)

        def bw95(f, P):
            c = np.cumsum(P)
            return float(f[np.searchsorted(c, 0.95 * c[-1])])
        out[name] = {
            "bw95_raw_ref_hz": round(bw95(f_r, P_r), 2),
            "bw95_cleaned_ref_hz": round(bw95(f_c, P_c), 2),
            "bw95_robot_measured_hz": round(bw95(f_q, P_q), 2),
        }
    return out


def main() -> None:
    res = {
        "A_synthetic": {
            "sine_amplitude_retention": bench_sines(),
            "hit_peak_retention": bench_hits(),
            "glitch_removal": bench_glitch(),
        },
        "B_real_thriller": real_motion_audit(),
        "C_robot_bandwidth": robot_bandwidth(),
    }
    (OUT / "audit_result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
