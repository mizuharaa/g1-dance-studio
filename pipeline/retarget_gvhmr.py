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
sys.path.insert(0, str(GMR_DIR))

from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402
from general_motion_retargeting.utils.smpl import (  # noqa: E402
    get_gvhmr_data_offline_fast,
    load_gvhmr_pred_file,
)


def retarget(pred_file: Path, out_csv: Path, robot: str = "unitree_g1") -> dict:
    smplx_folder = GMR_DIR / "assets" / "body_models"
    smplx_data, body_model, smplx_output, human_height = load_gvhmr_pred_file(
        str(pred_file), smplx_folder
    )
    frames, fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )
    rt = GMR(actual_human_height=human_height, src_human="smplx", tgt_robot=robot)

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

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, motion, delimiter=",")
    info = {
        "frames": int(motion.shape[0]),
        "fps": int(fps),
        "dof": int(motion.shape[1] - 7),
        "seconds": round(motion.shape[0] / fps, 1),
        "human_height_m": round(float(human_height), 3),
        "out": str(out_csv),
    }
    print(info)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path, help="hmr4d_results.pt from GVHMR")
    ap.add_argument("--out", required=True, type=Path, help="output CSV path")
    ap.add_argument("--robot", default="unitree_g1")
    args = ap.parse_args()
    retarget(args.pred, args.out, args.robot)
