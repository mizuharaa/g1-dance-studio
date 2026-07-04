"""Offline obs-frame sensitivity test (audit experiment #2 — no robot, no GPU).

Question: how much did the two deploy obs-frame defects corrupt the policy's actions
on hardware, and does the fix restore sim-equivalent obs?
  (a) YAW: the npz reference world yaw (t=0: 90.3 deg) was never aligned to the IMU
      world frame -> sweep a simulated boot-heading offset 0..180 deg and measure
      action divergence, old (unaligned) vs new (align_yaw) code path.
  (b) TORSO vs PELVIS anchor: training anchors on torso_link; deploy used the pelvis
      IMU quat -> replay the dance's own waist trajectory and measure the action
      divergence between the two anchor conventions.

Method: perfect-tracking replay (the robot exactly follows the reference in a world
rotated by the boot offset), obs built by deploy_runtime.build_obs_odom — the actual
deploy code — fed to the actual ONNX policy. Divergence is measured against the
baseline (offset 0, aligned). No dynamics: this isolates the OBS pathway.

Run:  ~/miniconda3/envs/g1dance/bin/python tools/obs_frame_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import deploy_runtime as dr  # noqa: E402

import onnxruntime as ort  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "reports" / "obs_frame_sensitivity.json"


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def run_rollout(meta, session, yaw_deg: float, align: bool, torso_anchor: bool,
                n_ticks: int = 1500):
    """Perfect-tracking replay with the robot's world rotated by yaw_deg vs the npz
    frame. Returns the action trajectory produced through the deploy obs path."""
    dr.TORSO_ANCHOR = torso_anchor
    ref = dr.Reference(dr.DEFAULT_MOTION)   # fresh (align mutates)
    dyaw = np.deg2rad(yaw_deg)
    Rz = rot_z(dyaw)
    qz = dr.quat_axis_angle((0.0, 0.0, 1.0), dyaw)

    # Robot state in ITS world (npz world rotated by dyaw), tracking perfectly.
    apos0 = ref.apos[0].copy()
    robot_quat = np.array([dr.quat_mul_wxyz(qz, qq) for qq in ref.aquat])
    robot_disp = (ref.apos - apos0) @ Rz.T
    v_world_npz = np.gradient(ref.apos, 1.0 / 50.0, axis=0)
    robot_v_world = v_world_npz @ Rz.T

    if align:
        dr.YAW_ALIGN = True
        dr._align_reference(meta, ref, ref.jp[0], robot_quat[0])

    n = min(n_ticks, ref.T)
    last_action = np.zeros(29)
    actions = np.zeros((n, 29))
    for t in range(n):
        obs, _ = dr.build_obs_odom(meta, ref, ref.jp[t], ref.jv[t], robot_quat[t],
                                   np.zeros(3), last_action, t,
                                   robot_disp[t], robot_v_world[t])
        a = session.run(["actions"], {"obs": obs[None].astype(np.float32),
                                      "time_step": np.array([[float(t)]], np.float32)})[0][0]
        actions[t] = a
        last_action = a.astype(np.float64)
    return actions


def main():
    meta = dr.Meta(dr.DEFAULT_META)
    session = ort.InferenceSession(str(dr.DEFAULT_POLICY), providers=["CPUExecutionProvider"])
    n_ticks = 1500  # 30 s — covers the ramp, clean dancing, and the 13-17 s stepping

    print("baseline: yaw 0, aligned, torso anchor")
    base = run_rollout(meta, session, 0.0, align=True, torso_anchor=True, n_ticks=n_ticks)

    results = {"n_ticks": n_ticks, "action_rms_baseline": float(np.sqrt((base ** 2).mean())),
               "yaw_sweep": [], "torso_vs_pelvis": None}

    for yaw in (15.0, 30.0, 60.0, 90.3, 135.0, 180.0):
        old = run_rollout(meta, session, yaw, align=False, torso_anchor=True, n_ticks=n_ticks)
        new = run_rollout(meta, session, yaw, align=True, torso_anchor=True, n_ticks=n_ticks)
        d_old = np.abs(old - base)
        d_new = np.abs(new - base)
        row = {
            "yaw_deg": yaw,
            "old_unaligned": {"mean": float(d_old.mean()), "p95": float(np.quantile(d_old, 0.95)),
                              "max": float(d_old.max())},
            "new_aligned": {"mean": float(d_new.mean()), "p95": float(np.quantile(d_new, 0.95)),
                            "max": float(d_new.max())},
        }
        results["yaw_sweep"].append(row)
        print(f"yaw {yaw:6.1f}deg  OLD |da| mean {row['old_unaligned']['mean']:.3f} "
              f"p95 {row['old_unaligned']['p95']:.3f} max {row['old_unaligned']['max']:.3f}   "
              f"NEW mean {row['new_aligned']['mean']:.4f} max {row['new_aligned']['max']:.4f}")

    # (b) torso vs pelvis anchor, aligned, yaw 0 (dance's own waist trajectory)
    pelvis = run_rollout(meta, session, 0.0, align=True, torso_anchor=False, n_ticks=n_ticks)
    d = np.abs(pelvis - base)
    waist_amp = float(np.degrees(np.abs(
        dr.Reference(dr.DEFAULT_MOTION).jp[:n_ticks,
            [meta.waist_idx[n] for n in ("waist_yaw_joint", "waist_roll_joint",
                                         "waist_pitch_joint")]]).max()))
    results["torso_vs_pelvis"] = {"mean": float(d.mean()), "p95": float(np.quantile(d, 0.95)),
                                  "max": float(d.max()), "max_waist_angle_deg": waist_amp}
    print(f"torso-vs-pelvis anchor: |da| mean {d.mean():.4f} p95 {np.quantile(d,0.95):.4f} "
          f"max {d.max():.4f}  (max waist angle in window: {waist_amp:.1f} deg)")
    print(f"action RMS baseline: {results['action_rms_baseline']:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
