"""Automated vetting gate for G1 motion CSVs (LAFAN1 convention, 30 fps).

Checks (see docs/architecture.md section 5):
  1. Root XY excursion from start <= 1.5 m  (2 m-radius dance area, drift margin)
  2. Joint angles within model limits; joint velocities within model limits
  3. Foot-skate heuristic: stance-foot horizontal speed while at ground height
  4. Root height sanity (no floorwork in v1: pelvis never below 0.35 m)

Exit code 0 = PASS, 1 = FAIL. Use --json for machine-readable output (UI hook).
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_XML = ROOT / "third_party/mujoco_menagerie/unitree_g1/scene.xml"
CSV_FPS = 30.0

MAX_ROOT_EXCURSION_M = 1.5
MIN_PELVIS_HEIGHT_M = 0.35
FOOT_SKATE_SPEED = 0.30      # m/s tolerated horizontal foot speed during contact
FOOT_CONTACT_HEIGHT = 0.08   # foot site below this height counts as stance
VEL_LIMIT_FRACTION = 1.0     # fraction of model joint velocity limit allowed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    m = np.loadtxt(args.csv, delimiter=",")
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)

    qpos = np.empty_like(m)
    qpos[:, 0:3] = m[:, 0:3]
    qpos[:, 3] = m[:, 6]
    qpos[:, 4:7] = m[:, 3:6]
    qpos[:, 7:] = m[:, 7:]

    res = {"file": args.csv, "frames": len(m), "seconds": len(m) / CSV_FPS}
    checks = {}

    # 1. root excursion
    xy = m[:, 0:2] - m[0, 0:2]
    exc = float(np.linalg.norm(xy, axis=1).max())
    checks["root_excursion"] = {"max_m": round(exc, 3), "limit": MAX_ROOT_EXCURSION_M,
                                "pass": exc <= MAX_ROOT_EXCURSION_M}

    # 2a. joint position limits (model ranges, joints are qpos 7..35 = jnt 1..29)
    lo = model.jnt_range[1:, 0]
    hi = model.jnt_range[1:, 1]
    joints = m[:, 7:]
    viol = np.clip(lo - joints, 0, None) + np.clip(joints - hi, 0, None)
    worst = float(viol.max())
    checks["joint_limits"] = {"worst_violation_rad": round(worst, 4),
                              "pass": worst < 0.02}

    # 2b. joint velocities vs actuator/model limits where defined
    jvel = np.diff(joints, axis=0) * CSV_FPS
    peak = float(np.abs(jvel).max())
    checks["joint_velocity"] = {"peak_rad_s": round(peak, 2),
                                "limit_note": "G1 motor limit ~ 3*pi=9.42",
                                "pass": peak <= 3 * np.pi * VEL_LIMIT_FRACTION}

    # 3+4. FK pass: foot skate + pelvis height
    lfoot = model.body("left_ankle_roll_link").id
    rfoot = model.body("right_ankle_roll_link").id
    fpos = np.empty((len(qpos), 2, 3))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        fpos[i, 0] = data.xpos[lfoot]
        fpos[i, 1] = data.xpos[rfoot]
    fvel = np.linalg.norm(np.diff(fpos[:, :, 0:2], axis=0), axis=2) * CSV_FPS
    stance = fpos[:-1, :, 2] < FOOT_CONTACT_HEIGHT
    skate = float(fvel[stance].max()) if stance.any() else 0.0
    checks["foot_skate"] = {"max_stance_speed_m_s": round(skate, 3),
                            "limit": FOOT_SKATE_SPEED, "pass": skate <= FOOT_SKATE_SPEED}

    pelvis_min = float(m[:, 2].min())
    checks["pelvis_height"] = {"min_m": round(pelvis_min, 3),
                               "limit": MIN_PELVIS_HEIGHT_M,
                               "pass": pelvis_min >= MIN_PELVIS_HEIGHT_M}

    res["checks"] = checks
    res["pass"] = all(c["pass"] for c in checks.values())

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"{res['file']}: {res['frames']} frames, {res['seconds']:.1f}s")
        for name, c in checks.items():
            status = "PASS" if c["pass"] else "FAIL"
            detail = {k: v for k, v in c.items() if k != "pass"}
            print(f"  [{status}] {name}: {detail}")
        print("OVERALL:", "PASS" if res["pass"] else "FAIL")
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
