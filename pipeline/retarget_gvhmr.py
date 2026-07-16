#!/usr/bin/env python
"""Headless GVHMR -> G1 retarget (GMR), producing our standard 30 fps CSV.

Equivalent to third_party/GMR/scripts/gvhmr_to_robot.py + batch_gmr_pkl_to_csv.py
but with no MuJoCo viewer (runs on the laptop, no display needed) and a single
output convention: CSV rows = [root_pos(3), root_rot xyzw(4), dof_pos(29)],
30 fps — the same LAFAN1-style layout the vet/window/preview stages consume.

Usage:
  python -m pipeline.retarget_gvhmr --pred hmr4d_results.pt --out data/motions/thriller.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GMR_DIR = ROOT / "third_party" / "GMR"


def _import_gmr():
    """Lazy GMR import — kept out of module scope so the pure-numpy helpers
    (dof_aware_postprocess) are importable in a bare env without GMR's heavy deps."""
    if str(GMR_DIR) not in sys.path:
        sys.path.insert(0, str(GMR_DIR))
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import (
        get_gvhmr_data_offline_fast,
        load_gvhmr_pred_file,
    )
    return GMR, get_gvhmr_data_offline_fast, load_gvhmr_pred_file


def dof_aware_postprocess(motion, soft_frac: float = 0.9, protect_legs: bool = True):
    """DOF-aware retarget cleanup (PROMPT B task 5). The human has DoF/ranges the
    G1 lacks; copying joints 1:1 pushes G1 joints past their limits and the motion
    reads worse, not better. This scales each joint's excursion GLOBALLY (a single
    per-joint factor, NOT per-frame — per-frame scaling injects jitter) so its peak
    stays within ``soft_frac`` of the model range, measured from the default pose.

    Priorities (what makes the dance READ): leg joints carry balance/end-effector
    (foot) placement, so by default they are PROTECTED from scaling (protect_legs);
    over-range there should be fixed in retarget/grounding, not squashed. Arms,
    waist and wrists — where over-range is cosmetic — are scaled to fit. Missing
    human DoF (spine articulation) should be mapped onto waist+hip in the retarget
    itself; this post-process only guarantees the emitted targets are in-range.

    Returns (scaled_motion, info). Near-identity on a motion that already fits
    (e.g. the current Thriller retarget), active only on over-range raw retargets.
    """
    import sys as _sys
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))
    from pipeline import g1_limits as L

    m = np.asarray(motion, dtype=float).copy()
    if L.POS_LO is None:
        return m, {"scaled": False, "reason": "no model ranges available"}
    j = m[:, 7:]
    default = L.DEFAULT_JOINT_POS
    dev = j - default
    lo_room = (default - L.POS_LO) * soft_frac        # allowed downward excursion
    hi_room = (L.POS_HI - default) * soft_frac        # allowed upward excursion
    factors = np.ones(L.N_JOINTS)
    changed = []
    protect = set(L.LEG_IDX.tolist()) if protect_legs else set()
    for k in range(L.N_JOINTS):
        if k in protect:
            continue
        pos_peak = max(dev[:, k].max(), 0.0)
        neg_peak = max(-dev[:, k].min(), 0.0)
        f = 1.0
        if pos_peak > hi_room[k] > 0:
            f = min(f, hi_room[k] / pos_peak)
        if neg_peak > lo_room[k] > 0:
            f = min(f, lo_room[k] / neg_peak)
        if f < 0.999:
            factors[k] = f
            changed.append({"joint": L.JOINT_ORDER[k], "factor": round(float(f), 3)})
    m[:, 7:] = default + dev * factors
    return m, {"scaled": bool(changed), "soft_frac": soft_frac,
               "protect_legs": protect_legs, "joints_scaled": changed}


def retarget(
    pred_file: Path,
    out_csv: Path,
    robot: str = "unitree_g1",
    use_velocity_limit: bool = False,
    dof_aware: bool = False,
) -> dict:
    GMR, get_gvhmr_data_offline_fast, load_gvhmr_pred_file = _import_gmr()
    smplx_folder = GMR_DIR / "assets" / "body_models"
    smplx_data, body_model, smplx_output, human_height = load_gvhmr_pred_file(
        str(pred_file), smplx_folder
    )
    frames, fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )
    rt = GMR(
        actual_human_height=human_height,
        src_human="smplx",
        tgt_robot=robot,
        use_velocity_limit=use_velocity_limit,
    )

    qpos_list = []
    for i, frame in enumerate(frames):
        qpos_list.append(rt.retarget(frame))
        if i % 200 == 0:
            print(f"retarget {i}/{len(frames)}", flush=True)

    q = np.asarray(qpos_list, dtype=np.float32)
    motion = np.zeros((q.shape[0], q.shape[1]), dtype=np.float32)
    motion[:, :3] = q[:, :3]
    motion[:, 3:7] = q[:, 3:7][:, [1, 2, 3, 0]]  # wxyz -> xyzw (CSV convention)
    motion[:, 7:] = q[:, 7:]

    dof_info = {"scaled": False}
    if dof_aware:
        motion, dof_info = dof_aware_postprocess(motion)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, motion, delimiter=",")
    info = {
        "frames": int(motion.shape[0]),
        "fps": int(fps),
        "dof": int(motion.shape[1] - 7),
        "seconds": round(motion.shape[0] / fps, 1),
        "human_height_m": round(float(human_height), 3),
        "dof_aware": dof_info,
        "out": str(out_csv),
    }
    print(info)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path, help="hmr4d_results.pt from GVHMR")
    ap.add_argument("--out", required=True, type=Path, help="output CSV path")
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument(
        "--velocity-limit",
        action="store_true",
        help="clamp retargeted joint velocities inside GMR (recipe doc, section 2)",
    )
    ap.add_argument(
        "--dof-aware",
        action="store_true",
        help="scale over-range joint excursions globally into the G1 workspace "
             "(legs protected); PROMPT B task 5",
    )
    args = ap.parse_args()
    retarget(args.pred, args.out, args.robot, use_velocity_limit=args.velocity_limit,
             dof_aware=args.dof_aware)
