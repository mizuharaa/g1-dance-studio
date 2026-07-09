#!/usr/bin/env python3
"""Kinematically render a deploy-motion npz to an mp4 sim panel.

This is the FAITHFUL sim panel for the show side-by-side: it plays back the exact
per-frame pose the robot is commanded to hold (deploy npz: root pose from
body_pos_w[:,0]/body_quat_w[:,0], joints from joint_pos in joint_order_29dof).
No policy inference, no physics drift — the panel == what the robot does.

Usage:
  render_deploy_sim.py --npz <deploy.npz> --meta <policy_meta.json> --out <out.mp4>
                       [--fps 50] [--start-frame 0] [--width 640] [--height 720]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import mujoco  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
XML = ROOT / "third_party/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
FFMPEG = str(Path.home() / "miniconda3/envs/g1dance/bin/ffmpeg")
if not Path(FFMPEG).exists():
    import shutil
    FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xml", default=str(XML))
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=-1)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--smoke", type=int, default=0, help="render only N frames for a smoke test")
    args = ap.parse_args()

    import json
    meta = json.load(open(args.meta))
    jorder = meta["joint_order_29dof"]

    d = np.load(args.npz)
    jpos = d["joint_pos"]                 # (T, 29) in joint_order_29dof
    root_pos = d["body_pos_w"][:, 0, :]   # (T, 3)   pelvis/root world pos
    root_quat = d["body_quat_w"][:, 0, :] # (T, 4)   wxyz
    T = jpos.shape[0]
    src_fps = float(d["fps"][0]) if "fps" in d.files else 50.0

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # name -> qpos address for the 29 actuated joints (robust to XML ordering)
    jadr = {}
    for name in jorder:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            print(f"WARN: joint {name} not in XML", file=sys.stderr)
            continue
        jadr[name] = model.jnt_qposadr[jid]

    # free base is joint 0 -> qpos[0:7] = [x,y,z, qw,qx,qy,qz]
    has_free = model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 135, -15, 3.2
    cam.lookat[:] = [0, 0, 0.8]
    opt = mujoco.MjvOption()

    lo = args.start_frame
    hi = T if args.end_frame < 0 else min(args.end_frame, T)
    if args.smoke:
        hi = min(lo + args.smoke, hi)

    tmp = Path(args.out).with_suffix(".frames")
    tmp.mkdir(exist_ok=True)
    n = 0
    for t in range(lo, hi):
        if has_free:
            data.qpos[0:3] = root_pos[t]
            data.qpos[3:7] = root_quat[t]   # wxyz matches mujoco
        for i, name in enumerate(jorder):
            if name in jadr:
                data.qpos[jadr[name]] = jpos[t, i]
        mujoco.mj_forward(model, data)
        # keep camera tracking the pelvis horizontally so the dancer stays centered
        cam.lookat[0] = float(root_pos[t, 0])
        cam.lookat[1] = float(root_pos[t, 1])
        renderer.update_scene(data, cam, opt)
        px = renderer.render()
        # write raw ppm-ish via numpy -> png through ffmpeg later; use imageio if present
        from PIL import Image
        Image.fromarray(px).save(tmp / f"f{n:05d}.png")
        n += 1
    renderer.close()
    print(f"rendered {n} frames ({lo}..{hi}) at src_fps={src_fps}")

    if args.smoke:
        print(f"smoke frames in {tmp}")
        return 0

    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-framerate", str(args.fps),
        "-i", str(tmp / "f%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        args.out,
    ]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("ffmpeg encode failed", file=sys.stderr)
        return 1
    # cleanup frames
    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
