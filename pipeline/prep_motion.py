#!/usr/bin/env python
"""Show-prep for a retargeted motion CSV (recipe doc, sections 2 & 4).

Steps, in order:
  0. Outlier rejection + temporal smoothing (tools/motion_quality.clean_motion):
     hampel + Savitzky-Golay on joints & root pos, slerp-aware SG on the root
     quat. GVHMR is per-frame, so raw retargets carry single-frame limb flips —
     this removes them BEFORE the velocity clamp, which otherwise drags an
     outlier across multiple frames and snaps back (itself a glitch).
  1. Residual joint-velocity clamp: cap per-frame deltas at LIMIT_FRACTION of
     3π rad/s (GMR's use_velocity_limit handles most of it; this catches leftovers),
     then a 3-frame moving average ONLY on frames that were touched.
  2. Per-frame FK ground correction: remove the retarget's slow vertical drift so
     the SUPPORT (lower) foot sits on z≈0 in every frame — not just the single
     lowest instant (the old trajectory-wide offset left the foot floating >0.10 m
     in most frames: the §3.3 'floaty feet' defect). See grounding.ground_motion_per_frame.
  3. Prepend: PAD_IN seconds of static standing, then BLEND_IN seconds
     standing -> first dance pose (linear joints, slerp base quat, base z ramp).
  4. Append: BLEND_OUT seconds last pose -> standing, then HOLD_OUT seconds of
     static standing (recipe: the final pose is never held during training
     otherwise — no clean finish for a show).

Standing pose comes from the menagerie G1 'home' keyframe (same model the vet
and preview stages use), with the dance's first/last-frame yaw and xy preserved
so the robot doesn't teleport or spin at the seams.

Usage:
  python -m pipeline.prep_motion --in data/motions/thriller/thriller_vlim.csv \
      --out data/motions/thriller/thriller_show.csv
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pipeline.config import PROJECT_ROOT
from tools.motion_quality import clean_motion

MODEL_XML = PROJECT_ROOT / "third_party/mujoco_menagerie/unitree_g1/scene.xml"

FPS = 30
DT = 1.0 / FPS
VEL_LIMIT = 3.0 * np.pi          # rad/s, retarget-quality bound from the recipe
LIMIT_FRACTION = 0.90
PAD_IN_S = 1.0
BLEND_IN_S = 0.5
BLEND_OUT_S = 1.0
HOLD_OUT_S = 2.5


def _clamp_joint_velocities(dof: np.ndarray) -> tuple[np.ndarray, int]:
    """Cap frame-to-frame joint deltas; returns (cleaned, n_frames_touched)."""
    limit = VEL_LIMIT * LIMIT_FRACTION * DT
    out = dof.copy()
    touched = np.zeros(len(out), dtype=bool)
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        over = np.abs(delta) > limit
        if over.any():
            out[i, over] = out[i - 1, over] + np.clip(delta[over], -limit, limit)
            touched[i] = True
    # light smoothing only around modified frames
    idx = np.flatnonzero(touched)
    for i in idx:
        lo, hi = max(1, i - 1), min(len(out) - 2, i + 1)
        out[lo : hi + 1] = (out[lo - 1 : hi] + out[lo : hi + 1] + out[lo + 1 : hi + 2]) / 3.0
    return out, int(touched.sum())


def _min_height_fk(motion: np.ndarray, model: mujoco.MjModel) -> float:
    """Lowest z of any ROBOT geom over the trajectory (world/floor geoms excluded).

    Thin wrapper over the shared grounding helper so prep and the vet/window gate
    ground motion identically (audit HIGH: grounding was orphaned here)."""
    from .grounding import min_contact_height
    return min_contact_height(motion, model)


def _standing_row(model: mujoco.MjModel, like: np.ndarray) -> np.ndarray:
    """Standing pose row in CSV convention, xy+yaw taken from `like`."""
    key = model.key("stand") if model.nkey else None
    if key is None:
        raise SystemExit("G1 model has no 'stand' keyframe — cannot build standing pose")
    qpos = np.array(key.qpos)
    row = np.zeros_like(like)
    row[:2] = like[:2]                      # keep xy
    row[2] = qpos[2]                        # standing height
    yaw = Rotation.from_quat(like[3:7]).as_euler("zyx")[0]
    row[3:7] = Rotation.from_euler("z", yaw).as_quat()  # upright, same heading
    row[7:] = qpos[7:]
    return row


def _blend(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """n intermediate rows easing a -> b (cosine ease, slerp for the base quat)."""
    t = (1 - np.cos(np.linspace(0, np.pi, n + 2)[1:-1])) / 2
    rows = a[None, :] + t[:, None] * (b - a)[None, :]
    slerp = Slerp([0, 1], Rotation.from_quat(np.stack([a[3:7], b[3:7]])))
    rows[:, 3:7] = slerp(t).as_quat()
    return rows


def prep(in_csv: Path, out_csv: Path) -> dict:
    motion = np.loadtxt(in_csv, delimiter=",")
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))

    motion, clean_info = clean_motion(motion)  # de-glitch BEFORE the clamp

    dof, touched = _clamp_joint_velocities(motion[:, 7:])
    motion[:, 7:] = dof

    # Per-frame foot-contact grounding (support foot on z≈0 every frame), replacing
    # the old single trajectory-wide offset that left the foot floating (§3.3).
    from .grounding import ground_motion_per_frame
    motion, ground_info = ground_motion_per_frame(motion, model)

    stand_in = _standing_row(model, motion[0])
    stand_out = _standing_row(model, motion[-1])
    parts = [
        np.tile(stand_in, (round(PAD_IN_S * FPS), 1)),
        _blend(stand_in, motion[0], round(BLEND_IN_S * FPS)),
        motion,
        _blend(motion[-1], stand_out, round(BLEND_OUT_S * FPS)),
        np.tile(stand_out, (round(HOLD_OUT_S * FPS), 1)),
    ]
    full = np.vstack(parts)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, full, delimiter=",")
    info = {
        "in_frames": int(motion.shape[0]),
        "out_frames": int(full.shape[0]),
        "seconds": round(full.shape[0] / FPS, 1),
        "vel_clamped_frames": touched,
        "motion_quality": clean_info,
        "ground_shift_m": ground_info["mean_shift_m"],   # back-compat key (mean per-frame shift)
        "grounding": ground_info,
        "out": str(out_csv),
    }
    print(info)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, type=Path)
    ap.add_argument("--out", dest="out_csv", required=True, type=Path)
    args = ap.parse_args()
    prep(args.in_csv, args.out_csv)
