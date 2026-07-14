"""Automated vetting gate for G1 motion CSVs (LAFAN1 convention, 30 fps).

Two tiers (see docs/architecture.md section 5):
  HARD (gate fails):
    1. Root XY excursion from start <= 1.5 m  (2 m-radius dance area, drift margin)
    2. Joint angles within model position limits
    3. No floorwork in v1: pelvis never below 0.35 m
  ADVISORY (warnings only -- the RL tracking reward smooths infeasible references;
  Unitree's own LAFAN1 retargets exceed motor velocity limits on ~30% of frames):
    4. Joint velocity stats vs the ~3*pi rad/s motor class limit
    5. Foot-skate: stance-foot (low + vertically still) horizontal speed

Exit code 0 = PASS, 1 = FAIL (hard checks only). --json for the UI hook.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_XML = ROOT / "third_party/mujoco_menagerie/unitree_g1/scene.xml"
CSV_FPS = 30.0

# Max root excursion (footprint radius) the venue allows. Fallback 1.5 m (the
# 2 m-radius home area minus a 0.5 m margin). Resolution order (see _excursion_limit):
# G1_MAX_EXCURSION_M env override (subprocess one-offs) > the ACTIVE venue's limit
# (pipeline.venue registry, app-selectable) > this fallback.
MAX_ROOT_EXCURSION_M = float(os.environ.get("G1_MAX_EXCURSION_M", 1.5))


def _excursion_limit() -> float:
    """The excursion limit to enforce NOW: env override wins; else the active venue's
    limit; else the 1.5 m fallback. Resolved per-vet so a venue switch takes effect
    without a reimport, and a bad venue registry never breaks the safety gate."""
    if "G1_MAX_EXCURSION_M" in os.environ:
        return float(os.environ["G1_MAX_EXCURSION_M"])
    try:
        from pipeline.venue import active_max_excursion_m
        return float(active_max_excursion_m())
    except Exception:  # noqa: BLE001 — registry issue must not disable the gate
        return MAX_ROOT_EXCURSION_M
MIN_PELVIS_HEIGHT_M = 0.35
FOOT_SKATE_SPEED = 0.30      # m/s tolerated horizontal foot speed during stance
FOOT_CONTACT_HEIGHT = 0.045  # ankle_roll_link origin: grounded sole ~0.023-0.04 m
FOOT_CONTACT_VZ = 0.20       # m/s max vertical speed to count as stance
MOTOR_VEL_LIMIT = 3 * np.pi  # rad/s, G1 motor class limit (advisory)
# Smoothness gate thresholds (2026-07-10 measurements,
# data/telemetry/motion_quality_20260710): cleaned motions sit at jerk_peak
# 7.5-14k rad/s^3 and <1.3% spike frames; raw GVHMR/GMR glitchy ones at 37-68k
# and 7-13% — a glitchy motion trips BOTH margins comfortably.
MAX_JERK_PEAK = 20000.0      # rad/s^3
MAX_SPIKE_FRAMES_PCT = 2.0   # % of frames with a robust accel spike
# SEVERE tier (HARD gate — the "don't pay GPU on garbage" backstop). Set well
# ABOVE the advisory limits so a cleaned motion (jerk 7.5-14k, <1.3% spikes) and
# even a marginal one pass, but raw un-de-glitched GVHMR/GMR (37-68k, 7-13%) is
# BLOCKED before any cloud spend. Same data basis as the advisory thresholds
# (data/telemetry/motion_quality_20260710).
SEVERE_JERK_PEAK = 30000.0      # rad/s^3  (between the 20k advisory and 37k raw floor)
SEVERE_SPIKE_FRAMES_PCT = 5.0   # %        (between 2% advisory and 7% raw floor)
SEVERE_VEL_OVER_PCT = 70.0      # % frames past the motor limit = grossly infeasible


def severe_after_clean(csv_path: str) -> list:
    """Post-clean HARD backstop (importable): reasons a CLEANED motion is STILL
    clearly broken — jerk/spike above the severe floor, or grossly infeasible.
    Empty list = safe to train. This is the 'don't pay GPU on garbage that even
    de-glitch couldn't fix' gate; verified NOT to trip on the cleaned Thriller
    (7.6k jerk, 0.1% spike) — only a raw un-cleaned source (63k) trips it."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from pipeline.motion_io import load_motion_csv
    from tools.motion_quality import analyze as mq_analyze
    m = load_motion_csv(csv_path)
    mq = mq_analyze(m, CSV_FPS)
    spike_pct = 100.0 * mq["spike_frame_count"] / max(len(m), 1)
    jvel = np.abs(np.diff(m[:, 7:], axis=0) * CSV_FPS)
    over = float((jvel > MOTOR_VEL_LIMIT).any(axis=1).mean())
    reasons = []
    if mq["jerk_peak_rad_s3"] > SEVERE_JERK_PEAK:
        reasons.append(f"jerk_peak {mq['jerk_peak_rad_s3']:.0f}>{SEVERE_JERK_PEAK:.0f}")
    if spike_pct > SEVERE_SPIKE_FRAMES_PCT:
        reasons.append(f"spike {spike_pct:.1f}%>{SEVERE_SPIKE_FRAMES_PCT:.0f}%")
    if 100 * over > SEVERE_VEL_OVER_PCT:
        reasons.append(f"vel_over {100*over:.0f}%>{SEVERE_VEL_OVER_PCT:.0f}%")
    return reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # This module runs both as a package import AND as a standalone subprocess
    # (the retarget stage calls `python pipeline/vet_motion.py`), so import via the
    # absolute package with ROOT on sys.path — a relative import breaks the script.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from pipeline.grounding import UNGROUNDED_FLAG_M, ground_motion
    from pipeline.motion_io import load_motion_csv
    from pipeline.venue import footprint

    m = load_motion_csv(args.csv)  # clear error on malformed CSV, not a traceback
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)

    # Ground-reference before ANY absolute-z test (audit HIGH safety-gate bug):
    # HARD-3 (no floorwork) and foot-skate compare against z=0, so the motion's
    # lowest robot geom must sit on the floor first. Idempotent — a motion that
    # is already grounded shifts by ~0.
    m, ground_shift = ground_motion(m, model)

    qpos = np.empty_like(m)
    qpos[:, 0:3] = m[:, 0:3]
    qpos[:, 3] = m[:, 6]
    qpos[:, 4:7] = m[:, 3:6]
    qpos[:, 7:] = m[:, 7:]

    res = {"file": args.csv, "frames": len(m), "seconds": len(m) / CSV_FPS,
           "ground_shift_m": round(float(ground_shift), 4)}
    hard, advisory = {}, {}

    # HARD 1: spatial footprint = minimal enclosing circle of the root-XY path.
    # This is the smallest venue that holds the whole dance if the robot is placed
    # at the circle centre — translation-invariant, so a dance isn't penalised for
    # starting off-centre. The old metric (max distance from the first frame)
    # over-counted an off-centre-but-compact dance. Placement note: the guarantee
    # holds only if the robot is positioned at footprint_center_xy at deploy time.
    fcenter, fradius = footprint(m[:, 0:2])
    excursion_limit = _excursion_limit()
    hard["root_excursion"] = {
        "max_m": round(float(fradius), 3),          # footprint radius (kept key)
        "footprint_radius_m": round(float(fradius), 3),
        "footprint_center_xy": [round(float(fcenter[0]), 3),
                                round(float(fcenter[1]), 3)],
        "limit": round(excursion_limit, 3),
        "pass": bool(fradius <= excursion_limit + 1e-6)}

    # HARD 2: joint position limits (model ranges, joints are qpos 7..35 = jnt 1..29)
    lo = model.jnt_range[1:, 0]
    hi = model.jnt_range[1:, 1]
    joints = m[:, 7:]
    viol = np.clip(lo - joints, 0, None) + np.clip(joints - hi, 0, None)
    worst = float(viol.max())
    hard["joint_limits"] = {"worst_violation_rad": round(worst, 4),
                            "pass": worst < 0.02}

    # HARD 3: pelvis height (no floorwork in v1)
    pelvis_min = float(m[:, 2].min())
    hard["pelvis_height"] = {"min_m": round(pelvis_min, 3),
                             "limit": MIN_PELVIS_HEIGHT_M,
                             "pass": pelvis_min >= MIN_PELVIS_HEIGHT_M}

    # ADVISORY 1: joint velocity stats
    jvel = np.abs(np.diff(joints, axis=0) * CSV_FPS)
    over = float((jvel > MOTOR_VEL_LIMIT).any(axis=1).mean())
    advisory["joint_velocity"] = {
        "peak_rad_s": round(float(jvel.max()), 2),
        "p99_rad_s": round(float(np.percentile(jvel, 99)), 2),
        "frames_over_limit_pct": round(100 * over, 1),
        "limit": round(MOTOR_VEL_LIMIT, 2),
        "ok": over < 0.40}

    # ADVISORY 2: foot skate (FK pass; stance = low AND vertically still)
    lfoot = model.body("left_ankle_roll_link").id
    rfoot = model.body("right_ankle_roll_link").id
    fpos = np.empty((len(qpos), 2, 3))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        fpos[i, 0] = data.xpos[lfoot]
        fpos[i, 1] = data.xpos[rfoot]
    fvel_xy = np.linalg.norm(np.diff(fpos[:, :, 0:2], axis=0), axis=2) * CSV_FPS
    fvel_z = np.abs(np.diff(fpos[:, :, 2], axis=0)) * CSV_FPS
    stance = (fpos[:-1, :, 2] < FOOT_CONTACT_HEIGHT) & (fvel_z < FOOT_CONTACT_VZ)
    skate_p95 = float(np.percentile(fvel_xy[stance], 95)) if stance.any() else 0.0
    advisory["foot_skate"] = {"p95_stance_speed_m_s": round(skate_p95, 3),
                              "limit": FOOT_SKATE_SPEED,
                              "ok": skate_p95 <= FOOT_SKATE_SPEED}

    # ADVISORY 3: smoothness — accel/jerk spikes are the preview "twitch" and
    # become jerky commands on hardware (2026-07-10 lane-B fix). A motion that
    # went through prep_motion's clean stage passes; a raw glitchy one warns.
    from tools.motion_quality import analyze as mq_analyze
    mq = mq_analyze(m, CSV_FPS)
    spike_pct = 100.0 * mq["spike_frame_count"] / max(len(m), 1)
    advisory["smoothness"] = {
        "jerk_peak_rad_s3": mq["jerk_peak_rad_s3"],
        "jerk_p99_rad_s3": mq["jerk_p99_rad_s3"],
        "spike_frames": mq["spike_frame_count"],
        "spike_frames_pct": round(spike_pct, 1),
        "limit_jerk_peak": MAX_JERK_PEAK,
        "limit_spike_pct": MAX_SPIKE_FRAMES_PCT,
        "ok": mq["jerk_peak_rad_s3"] <= MAX_JERK_PEAK
              and spike_pct <= MAX_SPIKE_FRAMES_PCT}

    # ADVISORY 4: intake grounding — flag a motion that arrived un-grounded, so a
    # raw (offset_to_ground=False) retarget can't silently rely on grounding here.
    advisory["grounding"] = {
        "input_contact_offset_m": round(float(ground_shift), 4),
        "limit": UNGROUNDED_FLAG_M,
        "ok": abs(ground_shift) <= UNGROUNDED_FLAG_M}

    # ADVISORY 5: severe-quality signal. This gate runs on the RAW pre-clean
    # motion (prep_motion.clean_motion de-glitches downstream), so it must NOT be
    # a hard gate here — a raw glitchy source is expected and gets cleaned. It is
    # surfaced loudly so the operator knows the SOURCE is bad; the actual hard
    # backstop is applied POST-clean, before GPU spend (see local_motion + the
    # train stage: severe_after_clean()).
    severe_reasons = []
    if mq["jerk_peak_rad_s3"] > SEVERE_JERK_PEAK:
        severe_reasons.append(f"jerk_peak {mq['jerk_peak_rad_s3']:.0f}>{SEVERE_JERK_PEAK:.0f}")
    if spike_pct > SEVERE_SPIKE_FRAMES_PCT:
        severe_reasons.append(f"spike {spike_pct:.1f}%>{SEVERE_SPIKE_FRAMES_PCT:.0f}%")
    if 100 * over > SEVERE_VEL_OVER_PCT:
        severe_reasons.append(f"vel_over {100*over:.0f}%>{SEVERE_VEL_OVER_PCT:.0f}%")
    advisory["severe_quality"] = {
        "reasons": severe_reasons,
        "jerk_peak_rad_s3": mq["jerk_peak_rad_s3"],
        "spike_frames_pct": round(spike_pct, 1),
        "vel_over_limit_pct": round(100 * over, 1),
        "note": "raw source severely glitchy — clean stage must tame it before training"
                if severe_reasons else "ok",
        "ok": not severe_reasons}

    res["hard"] = hard
    res["advisory"] = advisory
    res["pass"] = all(c["pass"] for c in hard.values())

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"{res['file']}: {res['frames']} frames, {res['seconds']:.1f}s")
        for name, c in hard.items():
            status = "PASS" if c["pass"] else "FAIL"
            detail = {k: v for k, v in c.items() if k != "pass"}
            print(f"  [{status}] {name}: {detail}")
        for name, c in advisory.items():
            status = "ok" if c["ok"] else "WARN"
            detail = {k: v for k, v in c.items() if k != "ok"}
            print(f"  [{status}] {name}: {detail}")
        print("OVERALL:", "PASS" if res["pass"] else "FAIL")
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
