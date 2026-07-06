#!/usr/bin/env python
"""Acro-reference sanity gate (the acro replacement for pipeline/vet_motion.py).

pipeline/vet_motion.py encodes show-dance assumptions (pelvis >= 0.35 m, foot-
skate/stance heuristics, 3*pi rad/s advisory) that a backflip legitimately
violates — acro references BYPASS the show vet (documented in cloud/
dynamic_skills_task.py) and must pass THIS gate instead before training:

  HARD (exit 1 on failure):
    1. flip rotation actually present: the HORIZONTAL component of the summed
       per-frame world rotation vectors is 5.0..8.0 rad (a single full flip
       ~2*pi). Decomposed, not axis-of-the-sum: mocap yaw (turn-arounds,
       GVHMR drift) is reported separately and does not mask the flip;
    2. airborne phase present: both ankle_roll links > their grounded baseline
       +0.20 m (same rule as the task's flight-grace mask) for >= 0.15 s;
    3. joint angles inside model limits (worst violation < 0.05 rad — RSI and
       csv_to_npz clip to soft limits, but a gross violation means bad joint
       order or units);
    4. ends recoverable: final root z >= 0.55 m and final torso within ~37 deg
       of upright (projected gravity z < -0.8) — landing-success eval needs an
       upright reference ending;
    5. root quaternions normalized (|q|-1 < 0.01 pre-normalization is checked
       by the converter; here we re-check the CSV as written).
  REPORT (printed, no gate — physics feasibility is the RL policy's job):
    peak/p99 joint velocities vs the 20-37 rad/s true motor limits, flight
    apex/launch velocity, per-phase frame ranges.

Usage:
  ~/miniconda3/envs/g1dance/bin/python tools/check_acro_reference.py \
      data/acro_refs/converted/acro_backflip.csv [--json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CSV_FPS = 30.0
FEET_AERIAL_RISE = 0.20  # matches cloud/dynamic_skills_task.py


def quat_to_rotvec_delta(q: np.ndarray) -> np.ndarray:
    """Per-frame world-frame rotation vectors between consecutive quats (wxyz)."""
    w0, v0 = q[:-1, :1], q[:-1, 1:]
    w1, v1 = q[1:, :1], q[1:, 1:]
    # q_rel = q1 * conj(q0)
    w = w1 * w0 + np.sum(v1 * v0, axis=1, keepdims=True)
    v = w1 * v0 * -1 + w0 * v1 + np.cross(v1, v0 * -1)
    # shortest arc
    sign = np.where(w < 0, -1.0, 1.0)
    w, v = w * sign, v * sign
    ang = 2.0 * np.arctan2(np.linalg.norm(v, axis=1), w[:, 0])
    axis = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    return axis * ang[:, None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    import mujoco

    from pipeline.motion_io import load_motion_csv

    m = load_motion_csv(args.csv)
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "third_party/mujoco_menagerie/unitree_g1/scene.xml")
    )
    data = mujoco.MjData(model)

    qpos = np.empty_like(m)
    qpos[:, 0:3] = m[:, 0:3]
    qpos[:, 3] = m[:, 6]
    qpos[:, 4:7] = m[:, 3:6]
    qpos[:, 7:] = m[:, 7:]

    hard, report = {}, {}
    T = len(m)

    # FK pass: feet + torso orientation
    lf = model.body("left_ankle_roll_link").id
    rf = model.body("right_ankle_roll_link").id
    fz = np.empty((T, 2))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        fz[i, 0] = data.xpos[lf][2]
        fz[i, 1] = data.xpos[rf][2]

    # HARD 1: flip rotation present — HORIZONTAL component of the summed world
    # rotation vectors (the flip axis); yaw (z) reported separately so a mocap
    # turn-around/drift can't mask or fake a flip.
    quat_wxyz = qpos[:, 3:7]
    dr = quat_to_rotvec_delta(quat_wxyz)
    total = dr.sum(axis=0)
    flip_rad = float(np.linalg.norm(total[:2]))
    yaw_rad = float(total[2])
    flip_axis = total[:2] / max(flip_rad, 1e-12)
    hard["flip_rotation"] = {
        "flip_rad_horizontal": round(flip_rad, 3),
        "flip_rev": round(flip_rad / (2 * np.pi), 3),
        "flip_axis_xy_w": [round(float(a), 3) for a in flip_axis],
        "yaw_rad_reported": round(yaw_rad, 3),
        "pass": bool(5.0 <= flip_rad <= 8.0),
    }

    # HARD 2: airborne phase (same relative rule as the task grace mask)
    baseline = np.percentile(fz, 5, axis=0)
    aerial = (fz > baseline + FEET_AERIAL_RISE).all(axis=1)
    flight_s = float(aerial.sum() / CSV_FPS)
    idx = np.where(aerial)[0]
    hard["airborne_phase"] = {
        "flight_s": round(flight_s, 3),
        "frames": [int(idx[0]), int(idx[-1])] if idx.size else None,
        "feet_baseline_m": [round(float(b), 3) for b in baseline],
        "max_foot_rise_m": round(float((fz - baseline).max()), 3),
        "pass": bool(flight_s >= 0.15),
    }

    # HARD 3: joint limits
    lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
    joints = m[:, 7:]
    viol = np.clip(lo - joints, 0, None) + np.clip(joints - hi, 0, None)
    hard["joint_limits"] = {"worst_violation_rad": round(float(viol.max()), 4),
                            "pass": bool(viol.max() < 0.05)}

    # HARD 4: recoverable ending (upright + standing height)
    qe = quat_wxyz[-1]
    # projected gravity z in body frame = R^T * (0,0,-1) -> z component
    w, x, y, z = qe
    gz = -(1.0 - 2.0 * (x * x + y * y))
    hard["end_state"] = {"root_z_m": round(float(m[-1, 2]), 3),
                         "gravity_z_body": round(float(gz), 3),
                         "pass": bool(m[-1, 2] >= 0.55 and gz < -0.8)}

    # HARD 5: quat normalization as-written
    qn = np.abs(np.linalg.norm(m[:, 3:7], axis=1) - 1.0).max()
    hard["quat_norm"] = {"worst_abs_err": round(float(qn), 5),
                         "pass": bool(qn < 0.01)}

    # REPORT: velocities & flight kinematics
    jvel = np.abs(np.diff(joints, axis=0) * CSV_FPS)
    report["joint_velocity_rad_s"] = {
        "peak": round(float(jvel.max()), 2),
        "p99": round(float(np.percentile(jvel, 99)), 2),
        "n_joints_over_20": int((jvel.max(axis=0) > 20.0).sum()),
        "note": "true G1 motor limits 20-37 rad/s (docs/retarget_fidelity.md)",
    }
    root_vz = np.diff(m[:, 2]) * CSV_FPS
    report["flight"] = {
        "apex_root_z_m": round(float(m[:, 2].max()), 3),
        "launch_root_vz_m_s": round(float(root_vz[max(idx[0] - 2, 0)]), 2) if idx.size else None,
        "min_root_z_m": round(float(m[:, 2].min()), 3),
    }
    inv = np.where(1.0 - 2.0 * (quat_wxyz[:, 1] ** 2 + quat_wxyz[:, 2] ** 2) < -0.5)[0]
    report["inverted_frames"] = [int(inv[0]), int(inv[-1])] if inv.size else None

    res = {"file": args.csv, "frames": T, "seconds": round(T / CSV_FPS, 2),
           "hard": hard, "report": report,
           "pass": all(c["pass"] for c in hard.values())}
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"{args.csv}: {T} frames, {T / CSV_FPS:.2f}s")
        for name, c in hard.items():
            print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {name}: "
                  f"{ {k: v for k, v in c.items() if k != 'pass'} }")
        for name, c in report.items():
            print(f"  [info] {name}: {c}")
        print("OVERALL:", "PASS" if res["pass"] else "FAIL")
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
