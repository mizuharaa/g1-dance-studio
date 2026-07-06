#!/usr/bin/env python
"""Activation ramp for deployable motions: <name>_show.csv -> <name>_deploy.csv.

The trained policy activates while the robot stands at the mjlab standby pose
(policy_meta default_joint_pos). A prepped show motion starts from the menagerie
'stand' keyframe, which differs by up to ~0.68 rad (elbows/knees) — activating on
the raw clip lurches (ACTIVATION_HAZARD, 2026-07-04). The deployable motion
therefore prepends a 2.5 s cosine ramp from default_joint_pos to the show
motion's first frame, holding the root (xyz + quat) at frame-0 values.

This reproduces exactly how thriller_deploy.csv was generated from
thriller_show.csv (75 frames @30 fps, per-row blend s_i = (1-cos(pi*i/(n-1)))/2):
frame 0 == default_joint_pos, last ramp row == show frame 0, root constant.

Usage:
  python -m pipeline.deploy_ramp --in <name>_show.csv --out <name>_deploy.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.config import PROJECT_ROOT

# Canonical policy interface (identical to data/policies/thriller/policy_meta.json;
# tracked in git so a fresh checkout can still build deploy motions).
POLICY_INTERFACE = PROJECT_ROOT / "docs" / "mjlab_policy_interface.json"

FPS = 30
RAMP_S = 2.5
RAMP_FRAMES = round(RAMP_S * FPS)  # 75


def default_joint_pos(meta_path: Path = POLICY_INTERFACE) -> np.ndarray:
    meta = json.loads(Path(meta_path).read_text())
    dj = np.asarray(meta["default_joint_pos_rad"], dtype=float)
    if dj.shape != (29,):
        raise ValueError(f"default_joint_pos_rad must have 29 entries, got {dj.shape}")
    return dj


def add_activation_ramp(motion: np.ndarray, dj: np.ndarray,
                        n: int = RAMP_FRAMES) -> np.ndarray:
    """Prepend n ramp frames (default 2.5 s @30 fps): joints cosine-ease from the
    standby pose `dj` to motion[0]'s joints; root held at motion[0]'s root."""
    if motion.ndim != 2 or motion.shape[1] != 36:
        raise ValueError(f"expected 36-col motion CSV array, got {motion.shape}")
    first = motion[0]
    ramp = np.tile(first, (n, 1))
    # s_i = 0 at i=0 (pure standby joints), 1 at i=n-1 (== show frame 0) — matches
    # the validated thriller_deploy generation bit-for-bit at the endpoints.
    s = (1.0 - np.cos(np.pi * np.arange(n) / (n - 1))) / 2.0
    ramp[:, 7:] = dj[None, :] + s[:, None] * (first[7:] - dj)[None, :]
    return np.vstack([ramp, motion])


def make_deploy_csv(show_csv: Path, deploy_csv: Path,
                    meta_path: Path = POLICY_INTERFACE) -> dict:
    motion = np.loadtxt(show_csv, delimiter=",")
    full = add_activation_ramp(motion, default_joint_pos(meta_path))
    deploy_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(deploy_csv, full, delimiter=",")
    info = {
        "in_frames": int(motion.shape[0]),
        "out_frames": int(full.shape[0]),
        "ramp_s": RAMP_S,
        "seconds": round(full.shape[0] / FPS, 1),
        "frame0_max_delta_rad": float(np.abs(full[0, 7:] -
                                             default_joint_pos(meta_path)).max()),
        "out": str(deploy_csv),
    }
    print(json.dumps(info))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, type=Path)
    ap.add_argument("--out", dest="out_csv", required=True, type=Path)
    ap.add_argument("--meta", type=Path, default=POLICY_INTERFACE)
    args = ap.parse_args()
    make_deploy_csv(args.in_csv, args.out_csv, args.meta)
