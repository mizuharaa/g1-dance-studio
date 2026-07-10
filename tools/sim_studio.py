#!/usr/bin/env python
"""Sim 'dance studio' — side-by-side preview of the robot dancing (AGENT D, phase 2).

Renders two frame-synced panels with a live state overlay, so you can SEE the gap between
what you designed and what the robot does — and, once a retrain lands, before vs after.

Panel sources:
  * REFERENCE (kinematic): the INTENDED motion — reference joint angles played straight
    (a perfect tracker). This is the "design intent" the 3D preview shows.
  * POLICY (dynamic sandbox): what the robot ACTUALLY does — the real policy.onnx run in
    MuJoCo via the exact deploy contract (tools/sim_sandbox), optional latency/tether.

Default:  REFERENCE | POLICY            — the honest fidelity preview (intent vs reality).
--dance-b: POLICY(A) | POLICY(B)         — before vs after a retrain (A=before, B=after).

Overlay per panel: label + the achieved-fraction / tracking state.

Usage:
  python -m tools.sim_studio --dance data/policies/thriller_csv_ankle_penalty --out studio.mp4
  python -m tools.sim_studio --dance <before> --dance-b <after> --out before_after.mp4
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402
import mujoco  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import pipeline.deploy_runtime as D  # noqa: E402
from tools.sim_sandbox import run_sandbox, tracking_report, SCENE  # noqa: E402

H, W = 480, 380


def _kinematic_reference(dance: Path, steps: int) -> dict:
    """The INTENDED dance: reference joint_pos played kinematically (a perfect tracker)."""
    meta = D.Meta(dance / "policy_meta.json")
    ref = D.Reference(next(dance.glob("*_deploy.npz")))
    n = min(steps, ref.T)
    q = np.array([ref.jp[t] for t in range(n)])
    base = np.array([[ref.apos[t][0], ref.apos[t][1], 0.79] for t in range(n)])
    return {"q": q, "base_pos": base, "meta": meta, "achieved": 1.0, "kind": "REFERENCE (intended)"}


def _policy_rollout(dance: Path, steps: int, latency: float, tether: float, label: str) -> dict:
    out, _, _ = run_sandbox(dance, steps=steps, latency_ms=latency, tether_kp=tether)
    rep = tracking_report(out)
    out["achieved"] = rep["amplitude_ratio_overall"]   # honest: how much it DANCES (not error)
    out["kind"] = label
    out["fell_at"] = rep["fell_at_tick"]
    return out


def _qadr(model, meta) -> np.ndarray:
    a = np.zeros(meta.n, int)
    for i, name in enumerate(meta.joint_order):
        a[i] = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    return a


def render_studio(left: dict, right: dict, out_path: Path, meta):
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    qadr = _qadr(model, meta)
    dl, dr = mujoco.MjData(model), mujoco.MjData(model)
    rL = mujoco.Renderer(model, height=H, width=W)
    rR = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = 135, -15, 3.2
    opt = mujoco.MjvOption()
    n = min(len(left["q"]), len(right["q"]))
    tmp = Path(str(out_path) + ".frames"); tmp.mkdir(exist_ok=True)

    def panel(data, r, rec, k):
        data.qpos[:] = 0
        data.qpos[0:3] = rec["base_pos"][k]
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qpos[qadr] = rec["q"][k]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [rec["base_pos"][k][0], rec["base_pos"][k][1], 0.8]
        r.update_scene(data, cam, opt)
        return r.render()

    for k in range(n):
        imgL = panel(dl, rL, left, k)
        imgR = panel(dr, rR, right, k)
        combo = np.concatenate([imgL, imgR], axis=1)
        im = Image.fromarray(combo); dr_ = ImageDraw.Draw(im)
        for x0, rec in ((6, left), (W + 6, right)):
            dr_.rectangle([x0 - 2, 2, x0 + W - 12, 34], fill=(0, 0, 0))
            dr_.text((x0, 4), rec["kind"], fill=(255, 255, 255))
            dr_.text((x0, 18), f"dances {rec['achieved']*100:.0f}% of the motion"
                     + (f"  FELL@{rec['fell_at']}" if rec.get("fell_at") else ""),
                     fill=(120, 220, 120) if rec["achieved"] > 0.85 else (240, 200, 90))
        # uncalibrated-sim caveat, bottom strip
        dr_.rectangle([0, 2 * H - 16, 2 * W, 2 * H], fill=(0, 0, 0))
        dr_.text((6, 2 * H - 14),
                 "SIM NOT YET CALIBRATED to the training model — under-represents hardware "
                 "(reference | policy)", fill=(230, 170, 90))
        im.save(tmp / f"f{k:05d}.png")
    rL.close(); rR.close()

    ff = str(Path.home() / "miniconda3/envs/g1dance/bin/ffmpeg")
    ff = ff if Path(ff).exists() else (shutil.which("ffmpeg") or "ffmpeg")
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "50",
                    "-i", str(tmp / "f%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(out_path)])
    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dance", type=Path, required=True, help="policy dir (left = its reference)")
    ap.add_argument("--dance-b", type=Path, default=None, help="second policy dir -> before|after")
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--latency-ms", type=float, default=0.0)
    ap.add_argument("--tether-kp", type=float, default=0.0,
                    help="keep 0 — a high tether pins the base and SUPPRESSES the dance")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None, help="write a small json summary")
    args = ap.parse_args()

    meta = D.Meta(args.dance / "policy_meta.json")
    if args.dance_b:                              # POLICY(before) | POLICY(after)
        left = _policy_rollout(args.dance, args.steps, args.latency_ms, args.tether_kp, "BEFORE")
        right = _policy_rollout(args.dance_b, args.steps, args.latency_ms, args.tether_kp, "AFTER")
    else:                                         # REFERENCE(intended) | POLICY(actual)
        left = _kinematic_reference(args.dance, args.steps)
        right = _policy_rollout(args.dance, args.steps, args.latency_ms, args.tether_kp,
                                "POLICY (sim — uncalibrated)")
    print(f"left  {left['kind']}: achieved {left['achieved']*100:.0f}%")
    print(f"right {right['kind']}: achieved {right['achieved']*100:.0f}%"
          + (f"  fell@{right.get('fell_at')}" if right.get('fell_at') else ""))
    render_studio(left, right, args.out, meta)
    print(f"wrote {args.out}")
    if args.report:
        import json
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "left_kind": left["kind"], "left_achieved": float(left["achieved"]),
            "right_kind": right["kind"], "right_achieved": float(right["achieved"]),
            "right_fell_at": right.get("fell_at"),
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
