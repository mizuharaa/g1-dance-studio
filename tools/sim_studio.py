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
from tools.sim_sandbox import run_sandbox, tracking_report, SCENE, is_faithful  # noqa: E402


def _banner(model_path):
    """Model-aware honest caveat for the preview footer. Softer (blue-grey) on the
    faithful training model; loud amber on the non-faithful menagerie model."""
    if is_faithful(model_path):
        return ("PREVIEW on the mjlab TRAINING model — armatures + gains matched to training. "
                "Faithful to what was trained; still not the real robot.", (150, 190, 235))
    return ("SIM NOT ON THE TRAINING MODEL (menagerie) — under-represents the trained policy "
            "(washed-out / frozen).", (230, 170, 90))

H, W = 480, 380


def _ffmpeg_exe() -> str:
    """Encoder resolver. There is NO system ffmpeg on this laptop, so prefer the
    imageio-ffmpeg bundle (per handoff), then the conda env's ffmpeg, then PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    conda = Path.home() / "miniconda3/envs/g1dance/bin/ffmpeg"
    if conda.exists():
        return str(conda)
    return shutil.which("ffmpeg") or "ffmpeg"


def _encode(frames_dir: Path, out_path: Path) -> None:
    """Encode a directory of f%05d.png frames (50 fps) to an H.264 mp4, then clean up."""
    subprocess.run([_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", "50",
                    "-i", str(frames_dir / "f%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(out_path)], check=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()
    frames_dir.rmdir()


def _kinematic_reference(dance: Path, steps: int) -> dict:
    """The INTENDED dance: reference joint_pos played kinematically (a perfect tracker)."""
    meta = D.Meta(dance / "policy_meta.json")
    ref = D.Reference(next(dance.glob("*_deploy.npz")))
    n = min(steps, ref.T)
    q = np.array([ref.jp[t] for t in range(n)])
    base = np.array([[ref.apos[t][0], ref.apos[t][1], 0.79] for t in range(n)])
    return {"q": q, "base_pos": base, "meta": meta, "achieved": 1.0, "kind": "REFERENCE (intended)"}


def _policy_rollout(dance: Path, steps: int, latency: float, tether: float, label: str,
                    model_path: Path = SCENE) -> dict:
    out, _, _ = run_sandbox(dance, steps=steps, latency_ms=latency, xml=model_path,
                            tether_kp=tether)
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


def render_studio(left: dict, right: dict, out_path: Path, meta, model_path: Path = SCENE):
    model = mujoco.MjModel.from_xml_path(str(model_path))
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
        # model-fidelity caveat, bottom strip
        _msg, _col = _banner(model_path)
        dr_.rectangle([0, 2 * H - 16, 2 * W, 2 * H], fill=(0, 0, 0))
        dr_.text((6, 2 * H - 14), _msg + "  (reference | policy)", fill=_col)
        im.save(tmp / f"f{k:05d}.png")
    rL.close(); rR.close()
    _encode(tmp, out_path)


# ---- OVERLAY: reference (green ghost) + policy (blue) in ONE shared scene ----------
OW, OH = 760, 540
_REF_RGBA = np.array([0.28, 0.85, 0.42, 1.0], np.float32)   # translucent green ghost
_POL_RGBA = np.array([0.24, 0.48, 0.98, 1.0], np.float32)   # solid blue


def _tint_dynamic(scene, rgba) -> None:
    """Recolor only the robot (dynamic-category) geoms; leaves floor/sky/world alone."""
    for i in range(scene.ngeom):
        g = scene.geoms[i]
        if g.category == mujoco.mjtCatBit.mjCAT_DYNAMIC:
            g.rgba[:] = rgba


def render_overlay(left: dict, right: dict, out_path: Path, meta, model_path: Path = SCENE):
    """Render BOTH skeletons in the SAME mujoco scene, color-coded, into one mp4.

    Reference (intended) = translucent green ghost; policy (actual robot) = solid blue.
    Divergence between intent and reality is directly visible in a single view.
    model_path is a parameter so a faithful mjlab model can be swapped in later.
    """
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, OW)
    model.vis.global_.offheight = max(model.vis.global_.offheight, OH)
    qadr = _qadr(model, meta)
    dL, dR = mujoco.MjData(model), mujoco.MjData(model)
    r = mujoco.Renderer(model, height=OH, width=OW)
    cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = 135, -15, 3.6
    opt = mujoco.MjvOption()
    n = min(len(left["q"]), len(right["q"]))
    tmp = Path(str(out_path) + ".frames"); tmp.mkdir(exist_ok=True)

    def pose(data, rec, k):
        data.qpos[:] = 0
        data.qpos[0:3] = rec["base_pos"][k]
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qpos[qadr] = rec["q"][k]
        mujoco.mj_forward(model, data)

    for k in range(n):
        cx = (left["base_pos"][k][0] + right["base_pos"][k][0]) / 2
        cy = (left["base_pos"][k][1] + right["base_pos"][k][1]) / 2
        cam.lookat[:] = [cx, cy, 0.8]
        pose(dL, left, k)
        r.update_scene(dL, cam, opt); _tint_dynamic(r.scene, _REF_RGBA)
        img_ref = r.render().astype(np.float32)
        pose(dR, right, k)
        r.update_scene(dR, cam, opt); _tint_dynamic(r.scene, _POL_RGBA)
        img_pol = r.render().astype(np.float32)
        # Blend: reference as a faint ghost, policy stronger. Identical background in
        # both frames blends back to ~itself; the two tinted robots stay distinct.
        combo = np.clip(img_ref * 0.5 + img_pol * 0.72, 0, 255).astype(np.uint8)
        im = Image.fromarray(combo); d = ImageDraw.Draw(im)
        d.rectangle([0, 0, OW, 40], fill=(0, 0, 0))
        d.rectangle([8, 10, 26, 28], fill=(72, 217, 107))
        d.text((32, 14), "REFERENCE (intended dance)", fill=(210, 245, 215))
        d.rectangle([OW - 250, 10, OW - 232, 28], fill=(61, 122, 250))
        d.text((OW - 226, 14), "POLICY (actual robot)", fill=(200, 220, 255))
        fell = right.get("fell_at")
        if fell and k >= fell:
            d.text((OW // 2 - 60, 14), f"POLICY FELL @ {fell}", fill=(250, 130, 90))
        _msg, _col = _banner(model_path)
        d.rectangle([0, OH - 18, OW, OH], fill=(0, 0, 0))
        d.text((6, OH - 15), "OVERLAY - same scene, color-coded. " + _msg, fill=_col)
        im.save(tmp / f"f{k:05d}.png")
    r.close()
    _encode(tmp, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dance", type=Path, required=True, help="policy dir (left = its reference)")
    ap.add_argument("--dance-b", type=Path, default=None, help="second policy dir -> before|after")
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--latency-ms", type=float, default=0.0)
    ap.add_argument("--tether-kp", type=float, default=0.0,
                    help="keep 0 — a high tether pins the base and SUPPRESSES the dance")
    ap.add_argument("--out", type=Path, required=True, help="side-by-side mp4 (reference | policy)")
    ap.add_argument("--overlay-out", type=Path, default=None,
                    help="also render the SAME-SCENE color-coded overlay to this mp4")
    ap.add_argument("--model", type=Path, default=SCENE,
                    help="MuJoCo scene xml (default = faithful mjlab-aligned training model)")
    ap.add_argument("--menagerie", action="store_true",
                    help="use the (non-faithful) menagerie model instead of the faithful one")
    ap.add_argument("--report", type=Path, default=None, help="write a small json summary")
    args = ap.parse_args()
    from tools.sim_sandbox import MENAGERIE
    args.model = MENAGERIE if args.menagerie else args.model

    meta = D.Meta(args.dance / "policy_meta.json")
    if args.dance_b:                              # POLICY(before) | POLICY(after)
        left = _policy_rollout(args.dance, args.steps, args.latency_ms, args.tether_kp, "BEFORE",
                               model_path=args.model)
        right = _policy_rollout(args.dance_b, args.steps, args.latency_ms, args.tether_kp, "AFTER",
                                model_path=args.model)
    else:                                         # REFERENCE(intended) | POLICY(actual)
        left = _kinematic_reference(args.dance, args.steps)
        _plabel = "POLICY (mjlab training model)" if is_faithful(args.model) else "POLICY (menagerie — uncalibrated)"
        right = _policy_rollout(args.dance, args.steps, args.latency_ms, args.tether_kp,
                                _plabel, model_path=args.model)
    print(f"left  {left['kind']}: achieved {left['achieved']*100:.0f}%")
    print(f"right {right['kind']}: achieved {right['achieved']*100:.0f}%"
          + (f"  fell@{right.get('fell_at')}" if right.get('fell_at') else ""))
    render_studio(left, right, args.out, meta, model_path=args.model)
    print(f"wrote {args.out}")
    if args.overlay_out:
        render_overlay(left, right, args.overlay_out, meta, model_path=args.model)
        print(f"wrote {args.overlay_out}")
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
