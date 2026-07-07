#!/usr/bin/env python
"""Append a return-to-standing tail to a deploy motion so `--exit stand` can engage.

WHY: pipeline/deploy_runtime.py's `--exit stand` (opt-in) hands the robot back to
onboard balance while STANDING instead of the ramp-to-damping catch-step — but it
GUARDS on the motion's final frame being within 0.15 rad of default_joint_pos. The
promoted Thriller deploy motion ends mid-dance (~39 deg off default at elbows/knees),
so the guard correctly refuses and falls back to damp. This tool authors a candidate
motion that ENDS STANDING, so the stand handoff can be validated.

HOW: mirror pipeline/deploy_ramp.add_activation_ramp (which the policy already tracks
at the START). Append:
  * a RETURN_S cosine blend of the 29 joints from the final dance pose -> default_joint_pos
  * a HOLD_S hold at default_joint_pos
The torso anchor (body_pos_w / body_quat_w at the torso index) is held at the final
dance frame's value (the dance ends ~in place; the robot returns to standing without
translating). Body velocities go to zero over the tail. Only joint_pos/joint_vel and the
torso anchor are read by the runtime; the other 29 body slots are held for array shape.

This blend is WITHIN the policy's training distribution (standing + the same cosine
easing it tracks on activation), but the resulting motion is UNVERIFIED end-to-end:
the box (mjlab held-out exam) is gone, so this candidate is for TETHERED validation
with the user present ONLY — NOT show-ready. It never overwrites data/policies/thriller/.

Usage:
  ~/miniconda3/envs/tv/bin/python tools/make_stand_tail.py \
    [--src data/policies/thriller] [--out data/policies/thriller_standtail_candidate]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TORSO_IDX = 15          # matches deploy_runtime.TORSO_NPZ_IDX
FPS = 30.0
RETURN_S = 2.5          # cosine ease dance-end -> default (mirrors the 2.5s activation ramp)
HOLD_S = 1.5            # stand still at default before the handoff
GUARD_TOL_RAD = 0.15    # must match deploy_runtime.STAND_GUARD_TOL_RAD


def default_joint_pos() -> np.ndarray:
    meta = json.load(open(ROOT / "docs" / "mjlab_policy_interface.json"))
    dj = np.asarray(meta["default_joint_pos_rad"], float)
    assert dj.shape == (29,), dj.shape
    return dj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data/policies/thriller")
    ap.add_argument("--out", type=Path, default=ROOT / "data/policies/thriller_standtail_candidate")
    ap.add_argument("--return-s", type=float, default=RETURN_S)
    ap.add_argument("--hold-s", type=float, default=HOLD_S)
    args = ap.parse_args()

    d = dict(np.load(args.src / "thriller_deploy.npz"))
    jp = d["joint_pos"]                       # [T,29]
    dj = default_joint_pos()
    n_ret = int(round(args.return_s * FPS))
    n_hold = int(round(args.hold_s * FPS))

    # cosine ease from the final dance joints to default (s: 0 -> 1)
    last = jp[-1]
    s = (1.0 - np.cos(np.pi * np.arange(1, n_ret + 1) / n_ret)) / 2.0    # ends at 1.0
    ret = last[None, :] + s[:, None] * (dj - last)[None, :]
    hold = np.tile(dj, (n_hold, 1))
    jp_tail = np.vstack([ret, hold])                                     # [n_ret+n_hold, 29]

    # joint_vel: finite-difference within the tail (0 across the hold), continuous-ish
    jv_tail = np.zeros_like(jp_tail)
    jv_tail[1:] = (jp_tail[1:] - jp_tail[:-1]) * FPS
    jv_tail[0] = (jp_tail[0] - jp[-1]) * FPS

    n_tail = jp_tail.shape[0]
    out = {"fps": d["fps"], "joint_pos": np.vstack([jp, jp_tail]),
           "joint_vel": np.vstack([d["joint_vel"], jv_tail])}
    # body arrays: hold the final dance frame (torso anchor stationary; robot stands in
    # place). Velocities ramp to zero over the tail.
    for k in ("body_pos_w", "body_quat_w"):
        tail = np.tile(d[k][-1:], (n_tail, 1, 1))
        out[k] = np.concatenate([d[k], tail], axis=0)
    for k in ("body_lin_vel_w", "body_ang_vel_w"):
        vtail = np.zeros((n_tail,) + d[k].shape[1:], d[k].dtype)
        out[k] = np.concatenate([d[k], vtail], axis=0)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "thriller_deploy.npz", **out)
    for f in ("policy.onnx", "policy_meta.json"):
        shutil.copyfile(args.src / f, args.out / f)

    # verify the guard will PASS on this candidate
    final_delta = float(np.abs(out["joint_pos"][-1] - dj).max())
    ok = final_delta <= GUARD_TOL_RAD
    (args.out / "STANDTAIL.txt").write_text(
        f"CANDIDATE — return-to-standing tail for --exit stand validation (TETHERED ONLY).\n"
        f"Built from {args.src}/thriller_deploy.npz + {args.return_s}s cosine return + "
        f"{args.hold_s}s hold at default.\n"
        f"frames: {jp.shape[0]} dance -> {out['joint_pos'].shape[0]} total "
        f"({out['joint_pos'].shape[0]/FPS:.1f}s).\n"
        f"final-frame max |q-default| = {final_delta:.4f} rad "
        f"(guard tol {GUARD_TOL_RAD}) -> stand-guard {'PASSES' if ok else 'FAILS'}.\n"
        f"NOT show-ready: unverified end-to-end (mjlab box deleted). Validate on the tether\n"
        f"with the user present + damping remote, using EXIT_MODE=stand.\n")
    print(f"wrote {args.out}/thriller_deploy.npz  final|q-default|={final_delta:.4f}rad  "
          f"stand-guard={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
