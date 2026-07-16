#!/usr/bin/env python
"""Measure the per-frame foot-contact grounding fix (§3.3 'floaty feet' defect).

Before/after on the Thriller reference:
  * float:        % frames the support (lower) foot is >0.10 m off the floor
                  (the exact axis motion_triage / motion_dynamics flag).
  * penetration:  lowest robot geom below z=0 (must not regress).
  * root height:  root-above-lower-foot (what the tracking policy targets) must
                  be preserved EXACTLY by grounding (a pure vertical translation).
  * jitter:       peak root-z jerk must not spike.

Writes a grounded output CSV with a NEW name (never overwrites the source) plus a
raw JSON report, so the load-bearing numbers have durable provenance
(measurement-discipline rule). Numpy + mujoco only (no scipy needed).

Usage:
  python experiments/grounding_fix/measure_grounding_fix.py \
      [--source data/motions/thriller/thriller_g1_clean.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from pipeline import g1_limits as L  # noqa: E402
from pipeline import grounding  # noqa: E402
from pipeline.motion_io import load_motion_csv  # noqa: E402

FPS = 30.0
FLOATY_Z = 0.10   # motion_dynamics.FLOATY_FOOT_Z — lower foot above this = floaty


def _measure(m: np.ndarray, gmodel, umodel, lf: int, rf: int) -> dict:
    """Metrics on an already-grounded motion `m`."""
    n = len(m)
    contact = grounding.per_contact_height(m, gmodel)   # menagerie lowest geom / frame
    lower = np.empty(n)                                  # unitree lower ankle-roll origin
    root_above_foot = np.empty(n)
    ud = mujoco.MjData(umodel)
    for i, row in enumerate(m):
        q = np.empty(36)
        q[:3] = row[:3]; q[3] = row[6]; q[4:7] = row[3:6]; q[7:] = row[7:]
        ud.qpos[:] = q
        mujoco.mj_forward(umodel, ud)
        lower[i] = min(ud.xpos[lf, 2], ud.xpos[rf, 2])
        root_above_foot[i] = row[2] - lower[i]
    root_jerk = float(np.abs(np.diff(m[:, 2], n=3)).max()) * FPS ** 3
    return {
        "floaty_feet_pct": round(100.0 * float((lower > FLOATY_Z).mean()), 2),
        "lower_foot_z_max_m": round(float(lower.max()), 4),
        "lower_foot_z_mean_m": round(float(lower.mean()), 4),
        "penetration_min_contact_m": round(float(contact.min()), 4),
        "penetration_frames_below_-2mm_pct": round(100.0 * float((contact < -0.002).mean()), 2),
        "root_above_lower_foot_mean_m": round(float(root_above_foot.mean()), 4),
        "root_z_mean_m": round(float(m[:, 2].mean()), 4),
        "root_z_peak_jerk_rad_s3": round(root_jerk, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path,
                    default=ROOT / "data/motions/thriller/thriller_g1_clean.csv")
    ap.add_argument("--out-csv", type=Path,
                    default=ROOT / "data/motions/thriller/thriller_g1_grounded.csv")
    ap.add_argument("--out-json", type=Path,
                    default=Path(__file__).resolve().parent / "grounding_before_after.json")
    args = ap.parse_args()

    m = load_motion_csv(args.source)
    gmodel = grounding._model()
    umodel = L.build_model()
    lf = mujoco.mj_name2id(umodel, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    rf = mujoco.mj_name2id(umodel, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

    # BEFORE: the current global (single-offset) grounding — what the pipeline did.
    before_m, before_shift = grounding.ground_motion(m, gmodel)
    before = _measure(before_m, gmodel, umodel, lf, rf)

    # AFTER: the new per-frame grounding.
    after_m, ginfo = grounding.ground_motion_per_frame(m, gmodel, fps=FPS)
    after = _measure(after_m, gmodel, umodel, lf, rf)

    # write the grounded output (NEW name — source is never overwritten)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out_csv, after_m, delimiter=",", fmt="%.6f")

    report = {
        "source": str(args.source.relative_to(ROOT)),
        "grounded_output": str(args.out_csv.relative_to(ROOT)),
        "frames": int(len(m)),
        "global_grounding_shift_m": round(float(before_shift), 4),
        "per_frame_grounding_info": ginfo,
        "before_global_grounding": before,
        "after_per_frame_grounding": after,
        "checks": {
            "float_reduced": after["floaty_feet_pct"] < before["floaty_feet_pct"],
            "float_target_met_<2pct": after["floaty_feet_pct"] < 2.0,
            "no_new_penetration": after["penetration_min_contact_m"] >= before["penetration_min_contact_m"] - 1e-4
                                  or after["penetration_min_contact_m"] >= -1e-4,
            "root_above_foot_preserved": abs(after["root_above_lower_foot_mean_m"]
                                             - before["root_above_lower_foot_mean_m"]) < 1e-3,
            "no_jitter_spike": after["root_z_peak_jerk_rad_s3"] <= before["root_z_peak_jerk_rad_s3"] * 1.10,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print("\nwrote grounded CSV ->", args.out_csv)
    print("wrote report       ->", args.out_json)
    if not all(report["checks"].values()):
        print("\nWARNING: not all checks passed:", report["checks"], file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
