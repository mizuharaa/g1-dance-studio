#!/usr/bin/env python
"""Acro skill eval: survival + landing success + rotation-completed checker.

Runs the trained policy from frame 0 over the full acro reference in N
randomized envs (train-task DR + obs noise, RSI forced to frame 0, no pushes)
and reports, per env:

  * survived      — never hit a failure termination before the motion end;
  * rotation_ok   — cumulative torso rotation about the reference's dominant
                    flip axis matches the reference total (|err| <= max(0.6 rad,
                    20% of the reference total)) — i.e. the flip actually
                    happened, no under- or over-rotation;
  * upright_ok    — final projected gravity z < -0.85 (within ~30 deg of
                    upright) AND final torso z within 0.25 m of the reference;
  * LANDED        — all three.

Plus the hardware-risk numbers: per-joint-group peak |qfrc_actuator| vs the
model's actuator rating, peak joint velocities, peak downward base deceleration
around touchdown (landing-impact proxy), and the reference flight window.

Usage (box):
  ./envs/mjlab/bin/python cloud/acro_eval.py \
      --checkpoint <model.pt> --motion-file motions/<acro>.npz \
      --num-envs 64 --output-file exports/acro1/<tag>/acro_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import dynamic_skills_task  # noqa: F401  registers the acro task

from dataclasses import asdict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
)

JOINT_GROUPS = {
  "hip": ".*_hip_.*_joint",
  "knee": ".*_knee_joint",
  "ankle": ".*_ankle_.*_joint",
  "waist": "waist_.*_joint",
  "shoulder": ".*_shoulder_.*_joint",
  "elbow": ".*_elbow_joint",
  "wrist": ".*_wrist_.*_joint",
}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--motion-file", required=True)
  ap.add_argument("--task", default=dynamic_skills_task.TASK_ID)
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--seed", type=int, default=94001)
  ap.add_argument("--output-file", default="acro_eval.json")
  args = ap.parse_args()
  device = "cuda:0"

  data = np.load(args.motion_file)
  fps = float(np.array(data["fps"]).reshape(-1)[0]) if "fps" in data else 50.0
  n_frames = int(data["joint_pos"].shape[0])
  duration_s = n_frames / fps

  env_cfg = load_env_cfg(args.task, play=False)  # train law: DR + noise, no pushes
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.commands["motion"].motion_file = args.motion_file
  env_cfg.commands["motion"].sampling_mode = "start"  # every env starts at frame 0
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.episode_length_s = duration_s + 2.0  # time_out never fires during eval
  env_cfg.seed = args.seed

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint, map_location=device)
  policy = runner.get_inference_policy(device=device)

  asset = env.unwrapped.scene["robot"]
  command = env.unwrapped.command_manager.get_term("motion")
  n = args.num_envs

  # joint bookkeeping
  all_ids, all_names = asset.find_joints((".*",))
  group_cols = {}
  for g, expr in JOINT_GROUPS.items():
    ids, names = asset.find_joints((expr,))
    group_cols[g] = [all_names.index(nm) for nm in names]
  jids = torch.tensor(all_ids, device=device, dtype=torch.long)

  # actuator ratings from the compiled model (joint-space, direct drive)
  ratings = {}
  try:
    mj_model = env.unwrapped.sim.mj_model
    for i in range(mj_model.nu):
      act_name = mj_model.actuator(i).name.split("/")[-1]
      fr = mj_model.actuator_forcerange[i]
      ratings[act_name] = float(max(abs(fr[0]), abs(fr[1])))
  except Exception as e:  # noqa: BLE001
    print(f"[warn] could not read actuator ratings: {e}")

  # reference flight window (from the tracked feet in the command's motion)
  feet = [i for i, nm in enumerate(command.cfg.body_names)
          if nm.endswith("_ankle_roll_link")]
  ref_feet_z = command.motion.body_pos_w[:, feet, 2]
  feet_baseline = ref_feet_z.quantile(0.05, dim=0)  # per-foot grounded link z
  aerial = (ref_feet_z > feet_baseline + 0.20).all(dim=1)  # matches task profile
  flight_frames = int(aerial.sum().item())

  # reference total rotation (dominant flip axis), from anchor quats
  ref_q = command.motion.body_quat_w[:, command.motion_anchor_body_index]  # (T,4)
  dq = quat_mul(ref_q[1:], quat_conjugate(ref_q[:-1]))
  ref_rotvec = axis_angle_from_quat(dq).sum(dim=0)  # (3,) world frame
  ref_total = float(ref_rotvec.norm().item())
  rot_axis = (ref_rotvec / max(ref_total, 1e-6)).to(device)

  # ---- rollout -----------------------------------------------------------------
  steps = n_frames - 2  # stop before the command wraps/resamples
  done_envs = torch.zeros(n, dtype=torch.bool, device=device)
  failed = torch.zeros(n, dtype=torch.bool, device=device)
  rot_acc = torch.zeros(n, device=device)
  peak_tau = torch.zeros(n, len(all_ids), device=device)
  peak_vel = torch.zeros(n, len(all_ids), device=device)
  peak_decel = torch.zeros(n, device=device)
  min_pelvis_z = torch.full((n,), 1e9, device=device)
  prev_q = command.robot_anchor_quat_w.clone()
  prev_vz = asset.data.body_link_lin_vel_w[:, 0, 2].clone()
  final_gz = torch.zeros(n, device=device)
  final_z_err = torch.zeros(n, device=device)
  gvec = torch.tensor([0.0, 0.0, -1.0], device=device).expand(n, 3)
  step_dt = float(env.unwrapped.step_dt)

  obs = env.get_observations()
  with torch.inference_mode():
    for step in range(steps):
      actions = policy(obs)
      obs, _, dones, _ = env.step(actions)

      active = ~done_envs
      q = command.robot_anchor_quat_w
      dqr = quat_mul(q, quat_conjugate(prev_q))
      step_rot = (axis_angle_from_quat(dqr) * rot_axis).sum(dim=1)
      rot_acc += torch.where(active, step_rot, torch.zeros_like(step_rot))
      prev_q = q.clone()

      tau = asset.data.qfrc_actuator[:, jids].abs()
      vel = asset.data.joint_vel.abs()
      peak_tau = torch.where(active.unsqueeze(1), torch.maximum(peak_tau, tau), peak_tau)
      peak_vel = torch.where(active.unsqueeze(1), torch.maximum(peak_vel, vel), peak_vel)

      vz = asset.data.body_link_lin_vel_w[:, 0, 2]
      decel = ((vz - prev_vz) / step_dt).clamp(min=0.0)  # upward accel = impact
      # only count as impact if we were falling
      decel = torch.where(prev_vz < -0.5, decel, torch.zeros_like(decel))
      peak_decel = torch.where(active, torch.maximum(peak_decel, decel), peak_decel)
      prev_vz = vz.clone()

      pz = asset.data.body_link_pos_w[:, 0, 2]
      min_pelvis_z = torch.where(active, torch.minimum(min_pelvis_z, pz), min_pelvis_z)

      terminated = env.unwrapped.termination_manager.terminated
      newly = dones.bool() & ~done_envs
      if newly.any():
        failed |= newly & terminated
        done_envs |= newly

      if step == steps - 1:
        gz = quat_apply_inverse(q, gvec)[:, 2]
        final_gz = gz
        final_z_err = (command.robot_anchor_pos_w[:, 2]
                       - command.anchor_pos_w[:, 2]).abs()
  env.close()

  survived = ~failed
  rot_tol = max(0.6, 0.2 * ref_total)
  rotation_ok = survived & ((rot_acc - ref_total).abs() <= rot_tol)
  upright_ok = survived & (final_gz < -0.85) & (final_z_err < 0.25)
  landed = survived & rotation_ok & upright_ok

  def _group_stats(peaks: torch.Tensor) -> dict:
    out = {}
    for g, cols in group_cols.items():
      if not cols:
        continue
      grp = peaks[:, cols]
      names = [all_names[c] for c in cols]
      rated = max((ratings.get(nm, 0.0) for nm in names), default=0.0)
      out[g] = {
        "max": round(float(grp.max().item()), 2),
        "p95_env_peak": round(float(grp.max(dim=1).values.quantile(0.95).item()), 2),
        "rating": round(rated, 2) if rated else None,
      }
    return out

  result = {
    "checkpoint": args.checkpoint,
    "motion_file": args.motion_file,
    "task": args.task,
    "num_envs": n,
    "frames": n_frames,
    "fps": fps,
    "duration_s": round(duration_s, 2),
    "reference": {
      "total_rotation_rad": round(ref_total, 3),
      "total_rotation_rev": round(ref_total / (2 * np.pi), 3),
      "rotation_axis_w": [round(float(x), 3) for x in rot_axis.cpu()],
      "flight_frames": flight_frames,
      "flight_s": round(flight_frames / fps, 3),
    },
    "success": {
      "survived": int(survived.sum().item()),
      "rotation_ok": int(rotation_ok.sum().item()),
      "upright_ok": int(upright_ok.sum().item()),
      "landed": int(landed.sum().item()),
      "landed_rate": round(float(landed.float().mean().item()), 4),
      "rotation_tolerance_rad": round(rot_tol, 3),
    },
    "rotation_achieved_rad": {
      "mean": round(float(rot_acc.mean().item()), 3),
      "min": round(float(rot_acc.min().item()), 3),
      "max": round(float(rot_acc.max().item()), 3),
    },
    "final_gravity_z": {
      "mean": round(float(final_gz.mean().item()), 3),
      "worst": round(float(final_gz.max().item()), 3),
    },
    "min_pelvis_z_m": round(float(min_pelvis_z.min().item()), 3),
    "peak_torque_nm_by_group": _group_stats(peak_tau),
    "peak_joint_vel_rad_s_by_group": _group_stats(peak_vel),
    "landing_impact": {
      "peak_base_decel_m_s2": round(float(peak_decel.max().item()), 1),
      "median_env_peak_m_s2": round(float(peak_decel.median().item()), 1),
      "note": "upward base accel while falling >0.5 m/s; F ~ m*(g+a) across both legs",
    },
  }

  Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
  with open(args.output_file, "w") as f:
    json.dump(result, f, indent=2)
  print(json.dumps(result["success"], indent=2))
  print(f"LANDED {int(landed.sum().item())}/{n}  "
        f"(survived {int(survived.sum().item())}, rotation {int(rotation_ok.sum().item())}, "
        f"upright {int(upright_ok.sum().item())})")
  print("WROTE", args.output_file)


if __name__ == "__main__":
  main()
