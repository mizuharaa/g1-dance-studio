#!/usr/bin/env python
"""Policy-in-the-loop simulation sandbox — the HONEST preview (AGENT D, 2026-07-10).

The 3D preview plays the REFERENCE motion (design intent). The robot runs an RL POLICY
that only approximately tracks it, so subtle/fast moves wash out or get skipped (tester:
~60-70% on hardware). This runs the ACTUAL policy.onnx in a dynamic MuJoCo sim using the
EXACT deploy contract, so we see + measure what the robot really does BEFORE hardware.

Faithfulness: the obs builder, inference, action->target and PD are IMPORTED from
pipeline/deploy_runtime.py (not re-implemented) — the sandbox and the real robot run the
same code. Optional --latency-ms injects the measured 40-80 ms sensorimotor delay
(data/telemetry/latency_diag_20260709/DIAGNOSIS.md) so the twin matches hardware, not
ideal sim.

Usage:
  python -m tools.sim_sandbox --dance data/policies/thriller_csv_ankle_penalty \
      --steps 400 --latency-ms 0   --out /tmp/rollout_ideal.mp4
  python -m tools.sim_sandbox --dance ... --latency-ms 60 --out /tmp/rollout_hw.mp4
"""
from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402
import mujoco  # noqa: E402
import onnxruntime as ort  # noqa: E402

import pipeline.deploy_runtime as D  # obs builder / inference / target / PD contract  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "third_party/mujoco_menagerie/unitree_g1/scene.xml"
CONTROL_HZ = 50.0
SIM_DT = 0.005
DECIM = 4                       # 50 Hz control over 200 Hz sim


def _wxyz(q):  # mujoco stores base quat as wxyz already
    return np.asarray(q, float)


def run_sandbox(dance: Path, steps: int, latency_ms: float, xml: Path = SCENE,
                tether_kp: float = 0.0):
    meta = D.Meta(dance / "policy_meta.json")
    npz = next(dance.glob("*_deploy.npz"))
    ref = D.Reference(npz)
    sess = ort.InferenceSession(str(dance / "policy.onnx"),
                                providers=["CPUExecutionProvider"])

    model = mujoco.MjModel.from_xml_path(str(xml))
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)

    # name-based joint map (policy order -> mujoco qpos/dof addresses); robust to XML order
    qadr, dadr = np.zeros(meta.n, int), np.zeros(meta.n, int)
    for i, name in enumerate(meta.joint_order):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise SystemExit(f"joint {name} not in {xml.name}")
        qadr[i], dadr[i] = model.jnt_qposadr[jid], model.jnt_dofadr[jid]

    # initial pose: standing keyframe if present, else default joints at a nominal height
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[qadr] = meta.default
    if data.qpos[2] < 0.4:
        data.qpos[2] = 0.79
    data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    # settle 0.3 s holding default with the deploy PD so it stands before the policy starts
    for _ in range(int(0.3 / SIM_DT)):
        q = data.qpos[qadr]; dq = data.qvel[dadr]
        data.qfrc_applied[dadr] = np.clip(meta.kp * (meta.default - q) - meta.kd * dq,
                                          -meta.effort, meta.effort)
        mujoco.mj_step(model, data)
    data.qvel[:] = 0

    # yaw-align the reference to the sim heading (same call the deploy makes at policy start)
    ref.align_yaw(D._anchor_quat(meta, data.qpos[qadr], _wxyz(data.qpos[3:7])))
    base_pos0 = data.qpos[0:3].copy()

    lat_ticks = int(round((latency_ms / 1000.0) * CONTROL_HZ))
    obs_delay = deque(maxlen=lat_ticks + 1)      # delayed OBSERVATION (sensor lag)
    tgt_delay = deque(maxlen=lat_ticks + 1)      # delayed ACTION (actuation lag)

    last_action = np.zeros(meta.n)
    rec = {k: [] for k in ("q", "target", "ref_jp", "base_pos", "base_up", "action", "fell")}
    target = meta.default.copy()
    fell_at = None

    for tick in range(steps):
        q = data.qpos[qadr].copy(); dq = data.qvel[dadr].copy()
        imu_quat = _wxyz(data.qpos[3:7]); gyro = data.qvel[3:6].copy()
        v_world = data.qvel[0:3].copy(); disp = data.qpos[0:3] - base_pos0

        # build obs from the (optionally delayed) sensed state — sensor/estimation lag
        state = (q, dq, imu_quat, gyro, v_world, disp)
        obs_delay.append(state)
        sq, sdq, squat, sgyro, sv, sdisp = obs_delay[0]   # oldest within the window
        obs, _ = D.build_obs_odom(meta, ref, sq, sdq, squat, sgyro, last_action, tick, sdisp, sv)

        if not np.isfinite(obs).all():
            fell_at = tick; break
        action = D.run_policy(sess, obs, tick)
        last_action = action
        new_target = D.action_to_target(meta, action)
        # actuation lag: the target the motors chase is the delayed one
        tgt_delay.append(new_target)
        target = tgt_delay[0]

        # up-vector of the torso (z of body frame) — fall if it drops below 0.5
        up = 1 - 2 * (imu_quat[1] ** 2 + imu_quat[2] ** 2)
        for r, val in zip(("q", "target", "ref_jp", "base_pos", "base_up", "action"),
                          (q, target, ref.at(tick)[0], data.qpos[0:3].copy(), up, action)):
            rec[r].append(val)
        rec["fell"].append(up < 0.5 or data.qpos[2] < 0.4)
        if (up < 0.5 or data.qpos[2] < 0.4) and fell_at is None:
            fell_at = tick

        # apply the deploy PD toward `target` for DECIM sim steps
        for _ in range(DECIM):
            qn = data.qpos[qadr]; dqn = data.qvel[dadr]
            data.qfrc_applied[dadr] = np.clip(meta.kp * (target - qn) - meta.kd * dqn,
                                              -meta.effort, meta.effort)
            # optional COMPLIANT TETHER: soft base station-keeping (XY) + sag support (Z),
            # so the FULL dance plays under the same "not 360-free" constraint the operator
            # runs with. Low gain — catches drift/sag without masking real balance loss.
            if tether_kp > 0:
                data.qfrc_applied[0:2] = (-tether_kp * (data.qpos[0:2] - base_pos0[0:2])
                                          - 0.2 * tether_kp * data.qvel[0:2])
                sag = base_pos0[2] - data.qpos[2]
                data.qfrc_applied[2] = tether_kp * max(sag, 0.0)
            mujoco.mj_step(model, data)

    out = {k: np.array(v) for k, v in rec.items()}
    out["fell_at_tick"] = fell_at
    out["meta"] = meta
    out["qadr"] = qadr
    return out, model, meta


def tracking_report(out) -> dict:
    """Reference-vs-achieved per joint: which moves the policy drops (the 60-70%)."""
    q, ref_jp = out["q"], out["ref_jp"]
    n = min(len(q), len(ref_jp))
    err = ref_jp[:n] - q[:n]                                  # (T,29) tracking error
    ref_rng = np.ptp(ref_jp[:n], axis=0) + 1e-6               # per-joint reference range
    ach = 1.0 - np.clip(np.abs(err).mean(axis=0) / ref_rng, 0, 1)   # achieved fraction
    return {
        "rms_err_rad": float(np.sqrt((err ** 2).mean())),
        "achieved_fraction_overall": float(ach.mean()),
        "worst_tracked_dofs": [int(i) for i in np.argsort(ach)[:6]],
        "per_dof_achieved": ach.round(3).tolist(),
        "fell_at_tick": out["fell_at_tick"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dance", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--latency-ms", type=float, default=0.0)
    ap.add_argument("--tether-kp", type=float, default=0.0,
                    help="compliant tether stiffness (N/m). ~150 mimics the operator's tether")
    ap.add_argument("--out", type=Path, default=None, help="render mp4 (optional)")
    ap.add_argument("--report", type=Path, default=None, help="write tracking report json")
    args = ap.parse_args()

    out, model, meta = run_sandbox(args.dance, args.steps, args.latency_ms,
                                   tether_kp=args.tether_kp)
    rep = tracking_report(out)
    print(f"== sandbox {args.dance.name}  latency={args.latency_ms}ms  steps={len(out['q'])} ==")
    print(f"  fell_at_tick: {rep['fell_at_tick']}  (None = stayed up)")
    print(f"  tracking RMS err: {rep['rms_err_rad']:.3f} rad")
    print(f"  ACHIEVED fraction (ref range the robot reproduces): {rep['achieved_fraction_overall']*100:.1f}%")
    print(f"  worst-tracked dof indices: {rep['worst_tracked_dofs']}")
    if args.report:
        import json
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, indent=2))
        print(f"  wrote {args.report}")
    if args.out:
        _render(out, model, args.out)
        print(f"  wrote {args.out}")


def _render(out, model, path):
    import subprocess, shutil
    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, height=480, width=420)
    cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = 135, -15, 3.2
    opt = mujoco.MjvOption()
    tmp = Path(str(path) + ".frames"); tmp.mkdir(exist_ok=True)
    from PIL import Image
    qadr = out["qadr"]
    for k in range(len(out["q"])):
        data.qpos[0:3] = out["base_pos"][k]
        data.qpos[qadr] = out["q"][k]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [out["base_pos"][k][0], out["base_pos"][k][1], 0.8]
        r.update_scene(data, cam, opt)
        Image.fromarray(r.render()).save(tmp / f"f{k:05d}.png")
    r.close()
    ff = str(Path.home() / "miniconda3/envs/g1dance/bin/ffmpeg")
    ff = ff if Path(ff).exists() else (shutil.which("ffmpeg") or "ffmpeg")
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "50",
                    "-i", str(tmp / "f%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(path)])
    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
