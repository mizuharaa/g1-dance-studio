"""Music-sync-preserving choreography difficulty editor for G1 dance CSVs.

Purpose: if a trained policy struggles in specific dance sections on hardware,
reduce the PHYSICAL difficulty of those time windows without changing timing
anywhere — frame count, fps, and everything outside the edited joints are
preserved exactly, so the motion stays on-beat with the music.

Two independent edits:

1. SECTION EDIT (--sections + --leg-scale): within each section [t0, t1] the
   LEG joints (hips/knees/ankles, identified by joint name from the MuJoCo
   model) and root z are blended toward the "stance interpolation" — the linear
   interpolation between the section's own boundary frames:
       q_new = leg_scale * q_orig + (1 - leg_scale) * q_interp
   Arms, waist, root XY and root orientation are untouched. The blend weight is
   cosine-ramped over --edge-blend-s at both edges so there is no velocity
   discontinuity where the edit meets the untouched motion.

2. LEAN CAP (--lean-cap-nm, global, optional): quasi-static ankle-load proxy =
       total_mass * g * (sagittal distance of whole-body CoM from the mid-ankle
       point, projected on the root's heading)
   (same formula as the physical analysis: mj_kinematics + mj_comPos,
   subtree_com[1] vs mean ankle_roll_link xpos). Frames whose |proxy| exceeds
   the cap get their hip_pitch / ankle_pitch / waist_pitch deviations from the
   motion's own median pose scaled down iteratively (<= 3 passes, step 0.85),
   with the per-frame scale factor smoothed over +-0.2 s so it never steps.
   GUARD (validated on the Thriller lean): the proxy is per-frame independent
   under FK, and on frames whose lean lives in the ROOT orientation (which this
   tool never touches) pulling the pitch joints toward the median moves the
   feet AWAY from the CoM and makes the load WORSE — so each pass re-evaluates
   the proxy and REVERTS the scaling on any frame it made worse; a pass (and
   the whole cap edit) is kept only if it is a net improvement. Whatever
   remains over the cap after the passes is reported honestly.

Validation (all must pass, or the tool prints FAILED and exits non-zero):
  (a) proxy recomputed before/after, per-section + global stats reported;
  (b) FK foot-height check — edited ankle_roll_link minimum heights must not
      go below the original's by more than 5 mm (no new ground penetration);
      pipeline/grounding.py is applied ONLY when the edit dug new penetration
      below the input's own floor reference (an edit must never re-reference
      the untouched frames — the input's floor is where the policy trained);
  (c) finite-difference joint velocities must not exceed the original's global
      max (no new spikes);
  (d) pipeline/vet_motion.py (the tiered motion gate) runs on the output and
      its verdict is included.

CSV convention (project-wide, LAFAN1 @ 30 fps): 36 cols =
    0:3 root xyz | 3:7 root quat (XYZW) | 7:36 the 29 joint angles
(MuJoCo wants the quat WXYZ — reorder when setting qpos.)

Usage:
    python tools/edit_choreography.py --csv IN.csv --out OUT.csv \
        --report OUT.json --sections "13.0-17.5,43.0-47.0" \
        --leg-scale 0.6 --lean-cap-nm 35 --edge-blend-s 0.3
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_XML = ROOT / "third_party/mujoco_menagerie/unitree_g1/g1.xml"
VET_SCRIPT = ROOT / "pipeline/vet_motion.py"

CSV_FPS = 30.0
GRAVITY = 9.81

# lean-cap tuning (spec'd, not user-facing)
CAP_MAX_PASSES = 3
CAP_SCALE_STEP = 0.85
CAP_SMOOTH_S = 0.2

# grounding is applied to the edited motion only when it "applies cleanly":
# a shift bigger than this means something is structurally off — keep the raw
# motion and let the foot check / vet fail loudly instead of hiding it.
GROUNDING_CLEAN_M = 0.03
GROUNDING_MIN_M = 1e-4          # below this the shift is noise — skip it

FOOT_DROP_TOL_M = 0.005         # (b) max allowed new penetration
VEL_TOL = 1e-6                  # (c) numerical slack on the velocity max

LEG_KEYS = ("hip_", "knee_", "ankle_")
PITCH_CAP_JOINTS = {
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "waist_pitch_joint",
}


# --------------------------------------------------------------------------- model

def load_model():
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    if model.nq != 36:
        raise RuntimeError(
            f"{MODEL_XML.name} has nq={model.nq}, expected 36 "
            "(freejoint + 29 joints) — wrong model variant")
    return model


def joint_names(model) -> list[str]:
    """Names of the 29 actuated joints, in qpos (== CSV column) order."""
    return [model.joint(i).name for i in range(1, model.njnt)]


def leg_columns(model) -> list[int]:
    """CSV column indices of the leg joints (hips/knees/ankles, by name)."""
    return [7 + j for j, n in enumerate(joint_names(model))
            if any(k in n for k in LEG_KEYS)]


def pitch_cap_columns(model) -> list[int]:
    """CSV column indices of hip_pitch/ankle_pitch/waist_pitch (lean-cap set)."""
    return [7 + j for j, n in enumerate(joint_names(model))
            if n in PITCH_CAP_JOINTS]


def fk_metrics(motion: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
    """One FK pass: (signed quasi-static ankle-load proxy [Nm] per frame,
    ankle_roll_link heights (n, 2) [left, right])."""
    import mujoco
    data = mujoco.MjData(model)
    lf = model.body("left_ankle_roll_link").id
    rf = model.body("right_ankle_roll_link").id
    total_mass = float(model.body_subtreemass[1])   # pelvis subtree = whole robot
    n = len(motion)
    proxy = np.empty(n)
    foot_z = np.empty((n, 2))
    for i, row in enumerate(motion):
        data.qpos[:3] = row[:3]
        data.qpos[3:7] = row[[6, 3, 4, 5]]          # CSV xyzw -> mujoco wxyz
        data.qpos[7:] = row[7:]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        com = data.subtree_com[1]
        mid = 0.5 * (data.xpos[lf] + data.xpos[rf])
        # root heading = body x-axis projected on the ground plane
        w, x, y, z = row[6], row[3], row[4], row[5]
        hx = 1.0 - 2.0 * (y * y + z * z)
        hy = 2.0 * (x * y + w * z)
        norm = math.hypot(hx, hy) or 1.0
        sag = ((com[0] - mid[0]) * hx + (com[1] - mid[1]) * hy) / norm
        proxy[i] = total_mass * GRAVITY * sag
        foot_z[i, 0] = data.xpos[lf][2]
        foot_z[i, 1] = data.xpos[rf][2]
    return proxy, foot_z


# --------------------------------------------------------------------------- edits

def parse_sections(spec: str) -> list[tuple[float, float]]:
    """'13.0-17.5,43.0-47.0' -> [(13.0, 17.5), (43.0, 47.0)] (sorted)."""
    if not spec or not spec.strip():
        return []
    out = []
    for part in spec.split(","):
        part = part.strip()
        try:
            a, b = part.split("-")
            t0, t1 = float(a), float(b)
        except ValueError as e:
            raise ValueError(f"bad section '{part}' — expected 't0-t1'") from e
        if not (t1 > t0 >= 0):
            raise ValueError(f"bad section '{part}' — need 0 <= t0 < t1")
        out.append((t0, t1))
    out.sort()
    return out


def sections_to_frames(sections: list[tuple[float, float]], n_frames: int,
                       fps: float = CSV_FPS) -> list[tuple[int, int]]:
    """Seconds -> inclusive frame ranges, clamped; error on overlap/degenerate."""
    frames = []
    for t0, t1 in sections:
        s = max(0, int(round(t0 * fps)))
        e = min(n_frames - 1, int(round(t1 * fps)))
        if e - s < 2:
            raise ValueError(
                f"section {t0}-{t1}s maps to frames {s}..{e} — too short/out of "
                f"range for a {n_frames}-frame motion")
        frames.append((s, e))
    for (s0, e0), (s1, e1) in zip(frames, frames[1:]):
        if s1 <= e0:
            raise ValueError(
                f"sections overlap after frame conversion ({s0}..{e0} vs "
                f"{s1}..{e1}) — merge them")
    return frames


def _edge_weight(length: int, blend_frames: int) -> np.ndarray:
    """Cosine-ramped blend weight over a section of `length` frames: 0 at both
    boundary frames, 1 in the interior, zero slope at the edges (C1 with the
    untouched motion outside)."""
    i = np.arange(length, dtype=float)
    if blend_frames <= 0:
        w = np.ones(length)
        w[0] = w[-1] = 0.0
        return w
    u0 = np.clip(i / blend_frames, 0.0, 1.0)                 # ramp in
    u1 = np.clip((length - 1 - i) / blend_frames, 0.0, 1.0)  # ramp out
    ramp = lambda u: 0.5 - 0.5 * np.cos(np.pi * u)           # noqa: E731
    return ramp(u0) * ramp(u1)


def apply_section_edits(motion: np.ndarray, frame_sections: list[tuple[int, int]],
                        leg_scale: float, edge_blend_frames: int,
                        cols: list[int]) -> np.ndarray:
    """Blend `cols` (leg joints + root z) toward the per-section stance
    interpolation. Frame count and everything outside `cols` are untouched."""
    out = motion.copy()
    for s, e in frame_sections:
        length = e - s + 1
        alpha = np.linspace(0.0, 1.0, length)[:, None]
        q0 = motion[s, cols][None, :]
        q1 = motion[e, cols][None, :]
        interp = (1.0 - alpha) * q0 + alpha * q1
        w = _edge_weight(length, edge_blend_frames)[:, None]
        orig = motion[s:e + 1, cols]
        out[s:e + 1, cols] = orig + w * (1.0 - leg_scale) * (interp - orig)
    return out


def _smooth_factor_field(tf: np.ndarray, smooth_frames: int) -> np.ndarray:
    """Expand per-frame target scale factors into a step-free applied field:
    each reduced frame's factor extends to +-smooth_frames neighbours with a
    cosine falloff, combined by min — so a reduced frame gets at least its own
    full reduction, and the field is continuous (never steps)."""
    n = len(tf)
    applied = np.ones(n)
    w = max(1, smooth_frames)
    for j in np.flatnonzero(tf < 1.0):
        lo, hi = max(0, j - w), min(n, j + w + 1)
        d = np.abs(np.arange(lo, hi) - j)
        prof = 0.5 * (1.0 + np.cos(np.pi * d / w))
        applied[lo:hi] = np.minimum(applied[lo:hi], 1.0 - (1.0 - tf[j]) * prof)
    return applied


def apply_lean_cap(motion: np.ndarray, model, cap_nm: float,
                   cols: list[int], fps: float = CSV_FPS,
                   max_passes: int = CAP_MAX_PASSES,
                   step: float = CAP_SCALE_STEP,
                   smooth_s: float = CAP_SMOOTH_S) -> tuple[np.ndarray, dict]:
    """Iteratively scale hip/ankle/waist-pitch deviations from the motion's own
    median pose down on frames whose |proxy| exceeds cap_nm (<= max_passes
    passes, factor `step` per pass, smoothed over +-smooth_s so it never
    steps). Guarded: scaling is kept per frame only where it actually reduces
    the proxy (frames whose lean is carried by the untouched root orientation
    get WORSE under median-pull and are reverted + reported), and the whole
    edit is dropped if it is not a net improvement."""
    base = motion.copy()                       # deviations always taken from here
    n = len(motion)
    median = np.median(base[:, cols], axis=0)
    smooth_f = max(1, int(round(smooth_s * fps)))

    def rescale(applied):
        out = base.copy()
        out[:, cols] = median + applied[:, None] * (base[:, cols] - median)
        return out

    proxy0, _ = fk_metrics(base, model)
    start_abs = np.abs(proxy0)
    start_over = int((start_abs > cap_nm).sum())
    start_max = float(start_abs.max())

    tf = np.ones(n)                            # per-frame target factor
    hopeless = np.zeros(n, dtype=bool)         # frames where scaling backfires
    cur, prev_abs = base, start_abs
    passes = []
    for _ in range(max_passes):
        flag = (prev_abs > cap_nm) & ~hopeless
        if not flag.any():
            break
        tf_new = tf.copy()
        tf_new[flag] *= step
        cand = rescale(_smooth_factor_field(tf_new, smooth_f))
        cand_abs = np.abs(fk_metrics(cand, model)[0])
        # per-frame guard: the proxy is per-frame independent under FK, so a
        # frame the scaling made worse can be reverted exactly
        worse = flag & (cand_abs > prev_abs + 0.1)
        if worse.any():
            tf_new[worse] = tf[worse]
            hopeless |= worse
            cand = rescale(_smooth_factor_field(tf_new, smooth_f))
            cand_abs = np.abs(fk_metrics(cand, model)[0])
        accepted = ((cand_abs > cap_nm).sum() <= (prev_abs > cap_nm).sum()
                    and cand_abs.max() <= prev_abs.max() + 0.1)
        passes.append({
            "flagged": int(flag.sum()),
            "reverted_counterproductive": int(worse.sum()),
            "frames_over_cap_after_pass": int((cand_abs > cap_nm).sum()),
            "max_abs_proxy_nm_after_pass": round(float(cand_abs.max()), 2),
            "accepted": bool(accepted)})
        if not accepted:
            break
        tf, cur, prev_abs = tf_new, cand, cand_abs

    end_over = int((prev_abs > cap_nm).sum())
    end_max = float(prev_abs.max())
    # net-improvement gate: a cap edit that doesn't help must not touch the motion
    effective = (cur is not base
                 and (end_over < start_over or end_max < start_max - 0.1))
    if not effective:
        cur, end_over, end_max = base, start_over, start_max
        tf = np.ones(n)
    residual = np.flatnonzero(prev_abs > cap_nm) if effective else \
        np.flatnonzero(start_abs > cap_nm)
    info = {
        "cap_nm": cap_nm,
        "cap_joint_columns": cols,
        "applied": bool(effective),
        "passes": passes,
        "passes_run": len(passes),
        "frames_over_cap_before": start_over,
        "frames_over_cap_after": end_over,
        "max_abs_proxy_before_nm": round(start_max, 2),
        "max_abs_proxy_after_nm": round(end_max, 2),
        "min_scale_factor": round(float(tf.min()), 4),
        "counterproductive_frames": int(hopeless.sum()),
        "residual_over_cap_frames": residual[:50].tolist(),
        "note": ("counterproductive frames (lean carried by the untouched root "
                 "orientation) were reverted and remain over cap"
                 if hopeless.any() else ""),
    }
    if not effective:
        info["note"] = ("lean-cap scaling gave no net improvement on this "
                        "motion — cap edit NOT applied; " + info["note"]).strip()
    return cur, info


# ---------------------------------------------------------------------- validation

def _stats(proxy_abs: np.ndarray) -> dict:
    return {"mean_abs_nm": round(float(proxy_abs.mean()), 2),
            "p95_abs_nm": round(float(np.percentile(proxy_abs, 95)), 2),
            "max_abs_nm": round(float(proxy_abs.max()), 2)}


def proxy_report(proxy_before: np.ndarray, proxy_after: np.ndarray,
                 frame_sections: list[tuple[int, int]], cap_nm: float) -> dict:
    rep = {"global": {"before": _stats(np.abs(proxy_before)),
                      "after": _stats(np.abs(proxy_after))},
           "sections": []}
    for s, e in frame_sections:
        rep["sections"].append({
            "frames": [s, e],
            "seconds": [round(s / CSV_FPS, 3), round(e / CSV_FPS, 3)],
            "before": _stats(np.abs(proxy_before[s:e + 1])),
            "after": _stats(np.abs(proxy_after[s:e + 1]))})
    if cap_nm > 0:
        rep["global"]["frames_over_cap_before"] = int(
            (np.abs(proxy_before) > cap_nm).sum())
        rep["global"]["frames_over_cap_after"] = int(
            (np.abs(proxy_after) > cap_nm).sum())
    return rep


def check_feet(foot_z_before: np.ndarray, foot_z_after: np.ndarray) -> dict:
    """(b) edited per-foot minimum ankle_roll_link heights must not go below the
    original's by more than FOOT_DROP_TOL_M."""
    res = {"tolerance_m": FOOT_DROP_TOL_M, "feet": {}}
    ok = True
    for k, name in enumerate(("left", "right")):
        mn_b = float(foot_z_before[:, k].min())
        mn_a = float(foot_z_after[:, k].min())
        drop = mn_b - mn_a
        foot_ok = drop <= FOOT_DROP_TOL_M
        ok &= foot_ok
        res["feet"][name] = {"min_m_before": round(mn_b, 4),
                             "min_m_after": round(mn_a, 4),
                             "new_penetration_m": round(max(0.0, drop), 4),
                             "pass": foot_ok}
    res["pass"] = bool(ok)
    return res


def check_velocity(orig: np.ndarray, edited: np.ndarray,
                   fps: float = CSV_FPS) -> dict:
    """(c) finite-difference joint velocities: edited global max must not exceed
    the original's (no new spikes)."""
    v_orig = float(np.abs(np.diff(orig[:, 7:], axis=0) * fps).max())
    v_edit = float(np.abs(np.diff(edited[:, 7:], axis=0) * fps).max())
    return {"orig_max_rad_s": round(v_orig, 3),
            "edited_max_rad_s": round(v_edit, 3),
            "pass": bool(v_edit <= v_orig + VEL_TOL)}


def run_vet(csv_path: Path) -> dict:
    """(d) run the tiered motion gate on the output CSV; return its JSON report
    (with 'pass'), or a failed stub if the gate itself crashes."""
    proc = subprocess.run(
        [sys.executable, str(VET_SCRIPT), str(csv_path), "--json"],
        capture_output=True, text=True, timeout=600, cwd=ROOT)
    try:
        return json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return {"pass": False,
                "error": f"vet_motion produced no JSON (rc={proc.returncode}): "
                         f"{proc.stderr[-400:]}"}


def validate(orig: np.ndarray, edited: np.ndarray, model,
             frame_sections: list[tuple[int, int]] | None = None,
             cap_nm: float = 0.0, vet_csv: Path | None = None) -> dict:
    """Full validation of an edit (checks a-d). vet_csv=None skips the vet gate
    (direct/unit-test use); the CLI always runs it."""
    frame_sections = frame_sections or []
    proxy_b, foot_b = fk_metrics(orig, model)
    proxy_a, foot_a = fk_metrics(edited, model)
    rep = {
        "proxy": proxy_report(proxy_b, proxy_a, frame_sections, cap_nm),
        "foot_height": check_feet(foot_b, foot_a),
        "joint_velocity": check_velocity(orig, edited),
        "frame_count": {"before": len(orig), "after": len(edited),
                        "pass": len(orig) == len(edited)},
        "frame0_max_delta_rad": round(
            float(np.abs(edited[0] - orig[0]).max()), 6),
    }
    if vet_csv is not None:
        rep["vet"] = run_vet(vet_csv)
    checks = [rep["foot_height"]["pass"], rep["joint_velocity"]["pass"],
              rep["frame_count"]["pass"]]
    if "vet" in rep:
        checks.append(bool(rep["vet"].get("pass", False)))
    rep["pass"] = bool(all(checks))
    return rep


# ----------------------------------------------------------------------------- run

def run(args: argparse.Namespace, _corrupt=None) -> tuple[dict, bool]:
    """Execute the edit + validation; returns (report, ok). `_corrupt` is a
    test-only hook: a callable applied to the edited motion before validation,
    used to prove the validators catch a broken edit."""
    from pipeline.motion_io import load_motion_csv

    model = load_model()
    orig = load_motion_csv(args.csv)
    n = len(orig)

    sections = parse_sections(args.sections)
    frame_sections = sections_to_frames(sections, n)
    edge_blend_frames = max(0, int(round(args.edge_blend_s * CSV_FPS)))

    edited = orig.copy()
    edit_applied = False
    if frame_sections:
        cols = leg_columns(model) + [2]         # legs + root z
        edited = apply_section_edits(edited, frame_sections, args.leg_scale,
                                     edge_blend_frames, cols)
        edit_applied = True

    cap_info = None
    if args.lean_cap_nm > 0:
        edited, cap_info = apply_lean_cap(edited, model, args.lean_cap_nm,
                                          pitch_cap_columns(model))
        edit_applied = True

    # Grounding policy for EDITS: never re-reference the untouched frames (the
    # input's z reference is what the policy trained on). Only if the edit dug
    # NEW penetration below the input's own floor reference does grounding
    # "apply cleanly": lift the motion so the lowest contact sits on z=0.
    grounding = {"applied": False, "shift_m": 0.0}
    if edit_applied:
        from pipeline.grounding import (ground_motion, have_model,
                                        min_contact_height)
        if have_model():
            zmin_orig = min_contact_height(orig)
            zmin_edit = min_contact_height(edited)
            grounding["min_contact_m_input"] = round(float(zmin_orig), 5)
            grounding["min_contact_m_edited"] = round(float(zmin_edit), 5)
            floor_ref = min(zmin_orig, 0.0)
            if zmin_edit < floor_ref - GROUNDING_MIN_M:
                if abs(zmin_edit) <= GROUNDING_CLEAN_M:
                    edited, shift = ground_motion(edited)
                    grounding["applied"] = True
                    grounding["shift_m"] = round(float(shift), 5)
                    grounding["note"] = (
                        "edit dug new penetration — motion lifted so the "
                        "lowest contact sits on z=0 (z reference changed)")
                else:
                    grounding["note"] = (
                        f"edit dug {-zmin_edit:.3f} m of penetration — beyond "
                        f"the clean-apply bound {GROUNDING_CLEAN_M} m, NOT "
                        "grounded (expect foot/vet failures)")
            else:
                grounding["note"] = ("no new penetration vs the input's floor "
                                     "reference — z reference preserved")
        else:
            grounding["note"] = "mujoco model missing — grounding skipped"

    if _corrupt is not None:
        edited = _corrupt(edited)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # %.18e round-trips float64 exactly: untouched columns stay bit-identical.
    np.savetxt(out, edited, delimiter=",", fmt="%.18e")

    report = {
        "tool": "edit_choreography/v1",
        "cli_args": {
            "csv": str(args.csv), "out": str(out),
            "report": str(args.report), "sections": args.sections,
            "leg_scale": args.leg_scale, "lean_cap_nm": args.lean_cap_nm,
            "edge_blend_s": args.edge_blend_s,
        },
        "argv": sys.argv,
        "fps": CSV_FPS,
        "frames": n,
        "duration_s": round(n / CSV_FPS, 3),
        "sections_frames": [list(fs) for fs in frame_sections],
        "edge_blend_frames": edge_blend_frames,
        "leg_joint_columns": leg_columns(model),
        "edit_applied": edit_applied,
        "grounding": grounding,
        "lean_cap": cap_info,
    }
    report["validation"] = validate(orig, edited, model, frame_sections,
                                    args.lean_cap_nm, vet_csv=out)
    ok = bool(report["validation"]["pass"])
    report["pass"] = ok

    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2) + "\n")
    return report, ok


def _print_summary(report: dict, ok: bool) -> None:
    v = report["validation"]
    g = v["proxy"]["global"]
    print(f"{report['cli_args']['csv']}: {report['frames']} frames "
          f"({report['duration_s']}s @ {report['fps']:.0f} fps)")
    print(f"  proxy global  before {g['before']}  after {g['after']}")
    for s in v["proxy"]["sections"]:
        print(f"  proxy section {s['seconds'][0]}-{s['seconds'][1]}s  "
              f"before {s['before']}  after {s['after']}")
    if report["lean_cap"]:
        lc = report["lean_cap"]
        print(f"  lean-cap {lc['cap_nm']} Nm "
              f"({'applied' if lc['applied'] else 'NOT applied'}): "
              f"{lc['passes_run']} pass(es), over-cap "
              f"{lc['frames_over_cap_before']} -> {lc['frames_over_cap_after']} "
              f"frames, max {lc['max_abs_proxy_before_nm']} -> "
              f"{lc['max_abs_proxy_after_nm']} Nm, "
              f"min scale {lc['min_scale_factor']}, "
              f"{lc['counterproductive_frames']} counterproductive frame(s)")
        if lc.get("note"):
            print(f"    note: {lc['note']}")
    fh = v["foot_height"]
    print(f"  [{'PASS' if fh['pass'] else 'FAIL'}] foot height: {fh['feet']}")
    jv = v["joint_velocity"]
    print(f"  [{'PASS' if jv['pass'] else 'FAIL'}] joint velocity: "
          f"orig max {jv['orig_max_rad_s']} rad/s, "
          f"edited max {jv['edited_max_rad_s']} rad/s")
    if "vet" in v:
        print(f"  [{'PASS' if v['vet'].get('pass') else 'FAIL'}] vet_motion gate")
    print(f"  grounding: {report['grounding']}")
    print(f"  frame0 max delta: {v['frame0_max_delta_rad']} rad")
    if ok:
        print("OVERALL: PASS — edited motion written to",
              report["cli_args"]["out"])
    else:
        print("\n*** FAILED *** — the edited motion did NOT pass validation. "
              "Do not use it. See the report JSON for details.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Music-sync-preserving choreography difficulty editor "
                    "(see module docstring)")
    ap.add_argument("--csv", required=True, help="input 36-col motion CSV @30fps")
    ap.add_argument("--out", required=True, help="edited CSV output path")
    ap.add_argument("--report", required=True, help="report JSON output path")
    ap.add_argument("--sections", default="",
                    help='seconds, e.g. "13.0-17.5,43.0-47.0" (empty = none)')
    ap.add_argument("--leg-scale", type=float, default=0.6,
                    help="per-section leg amplitude blend factor (1 = no change)")
    ap.add_argument("--lean-cap-nm", type=float, default=0.0,
                    help="global quasi-static proxy cap in Nm (0 = disabled)")
    ap.add_argument("--edge-blend-s", type=float, default=0.3,
                    help="cosine edge blend duration in seconds")
    args = ap.parse_args(argv)

    if not (0.0 <= args.leg_scale <= 1.0):
        ap.error("--leg-scale must be in [0, 1]")
    if args.lean_cap_nm < 0:
        ap.error("--lean-cap-nm must be >= 0 (0 disables)")
    if args.edge_blend_s < 0:
        ap.error("--edge-blend-s must be >= 0")

    report, ok = run(args)
    _print_summary(report, ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
