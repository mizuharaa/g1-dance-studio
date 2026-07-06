#!/usr/bin/env python
"""Dump a 1-env nominal full-motion rollout trace for fluidity forensics.

Writes sim_trace.npz with, per 50 Hz step: actions (raw policy output, the
same convention deploy telemetry records), q (robot joint pos), ref_q
(reference joint pos from the motion command), base_ang_vel (pelvis gyro,
body frame), ref_pelvis_ang_vel_w — plus joint_names, fps, steps, terminated.

cloud/fluidity_sim_metrics.py consumes this to produce the 2-10 Hz leg action
band / amplitude-ratio / wobble numbers of the v3/v4 decision table.

Usage:
  ./envs/mjlab/bin/python cloud/sim_trace_dump.py \
      --checkpoint <model.pt> --motion-file motions/thriller_deploy.npz \
      [--task Mjlab-Tracking-Flat-Unitree-G1] --out exports/.../sim_trace.npz
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))


@dataclass(frozen=True)
class Cfg:
  checkpoint: str
  motion_file: str
  out: str
  task: str = "Mjlab-Tracking-Flat-Unitree-G1"
  seed: int = 91001
  device: str | None = None


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import sim2real_task_v3  # noqa: F401  V3 ids (incl. GAPEVAL for v3b traces)
  try:
    import sim2real_task_v4  # noqa: F401
  except Exception as e:  # noqa: BLE001 — v4 module optional for old traces
    print(f"[warn] v4 task not registered: {e}")

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.tasks.tracking.mdp.commands import MotionCommand
  from mjlab.utils.torch import configure_torch_backends

  cfg = tyro.cli(Cfg)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  torch.manual_seed(cfg.seed)

  mot = np.load(cfg.motion_file, allow_pickle=True)
  fps = float(np.array(mot["fps"]).reshape(-1)[0]) if "fps" in mot else 50.0
  T = int(mot["joint_pos"].shape[0])
  episode_length_s = T / fps + 0.2
  max_steps = int(episode_length_s * 50) + 50

  env_cfg = load_env_cfg(cfg.task, play=False)
  agent_cfg = load_rl_cfg(cfg.task)
  env_cfg.commands["motion"].motion_file = cfg.motion_file
  env_cfg.commands["motion"].sampling_mode = "start"
  env_cfg.episode_length_s = episode_length_s
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = cfg.seed

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  from dataclasses import asdict, is_dataclass
  runner = runner_cls(env, asdict(agent_cfg) if is_dataclass(agent_cfg) else dict(agent_cfg),
                      device=device)
  runner.load(cfg.checkpoint, map_location=device)
  policy = runner.get_inference_policy(device=device)

  uenv = env.unwrapped
  asset = uenv.scene["robot"]
  joint_names = list(asset.data.joint_names) if hasattr(asset.data, "joint_names") \
    else list(asset.joint_names)
  command = cast(MotionCommand, uenv.command_manager.get_term("motion"))

  # Ordering guard: command.robot_joint_pos must be the asset joint vector in
  # the same order as joint_names, or every per-joint number would be mislabeled.
  jd = (command.robot_joint_pos[0] - asset.data.joint_pos[0]).abs().max().item()
  if jd > 1e-5:
    raise RuntimeError(f"command joint ordering != asset ordering (max diff {jd})")

  def gyro():
    try:
      from mjlab.tasks.tracking import mdp as tracking_mdp
      return tracking_mdp.builtin_sensor(uenv, "robot/imu_gyro")[0]
    except Exception:
      return asset.data.root_link_ang_vel_b[0]

  rows = {k: [] for k in ("action", "q", "ref_q", "base_ang_vel",
                          "ref_pelvis_ang_vel_w")}
  obs = env.get_observations()
  terminated_at = -1
  step = 0
  while step < max_steps:
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    rows["action"].append(actions[0].detach().cpu().numpy().copy())
    rows["q"].append(asset.data.joint_pos[0].cpu().numpy().copy())
    rows["ref_q"].append(command.joint_pos[0].cpu().numpy().copy())
    rows["base_ang_vel"].append(np.asarray(gyro().cpu().numpy(), float).copy())
    rows["ref_pelvis_ang_vel_w"].append(
      command.body_ang_vel_w[0, 0].cpu().numpy().copy())
    if bool(dones[0]):
      term = bool(uenv.termination_manager.terminated[0])
      if term:
        terminated_at = step
      break
    step += 1

  out = Path(cfg.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    out,
    **{k: np.asarray(v) for k, v in rows.items()},
    joint_names=np.array(joint_names),
    fps=np.array([50.0]),
    checkpoint=np.array(str(cfg.checkpoint)),
    task=np.array(cfg.task),
    motion_file=np.array(cfg.motion_file),
    terminated_at=np.array([terminated_at]),
  )
  print(f"[sim_trace] wrote {out} steps={len(rows['action'])} "
        f"terminated_at={terminated_at} (motion {T} frames)")
  env.close()


if __name__ == "__main__":
  main()
