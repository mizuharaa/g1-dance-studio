#!/usr/bin/env python
"""Joint-space ARM tracking metric in sim — the dance-quality yardstick.

The 2026-07-06 hardware runs showed the promoted s2r-b policy's arms track the
reference at ~2x the sim error (arm RMS 13.2 deg on hardware). A v3 candidate
must first BEAT s2r-b's *sim* arm tracking (this metric), then close the
hardware gap via the arm-plant work (V3B / ARM_GROUND_KP_SCALE).

Rolls the policy out over the FULL motion under nominal conditions (no obs
noise, no pushes, no injected delay — same 'nominal' definition as
cloud/sim_gap_check.py) and reports per-joint RMS / p95 of
|robot joint pos - reference joint pos| in DEGREES, with arm-group rollups.

Usage (on the box):
  ./envs/mjlab/bin/python cloud/arm_tracking_eval.py \
      --checkpoint <model.pt> --motion-file motions/thriller_deploy.npz \
      [--task Mjlab-Tracking-Flat-Unitree-G1-S2R-V3B-GAPEVAL] \
      --num-envs 64 --output-file reports/arm_tracking_<tag>.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.torch import configure_torch_backends

GROUP_PATTERNS = {
  "shoulder": ("shoulder",),
  "elbow": ("elbow",),
  "wrist": ("wrist",),
  "arms_all": ("shoulder", "elbow", "wrist"),
  "legs_all": ("hip", "knee", "ankle"),
  "waist_all": ("waist",),
}


@dataclass(frozen=True)
class Cfg:
  checkpoint: str
  motion_file: str
  task: str = "Mjlab-Tracking-Flat-Unitree-G1"
  num_envs: int = 64
  seed: int = 91001
  device: str | None = None
  output_file: str = "arm_tracking.json"
  episode_length_s: float = 0.0  # 0 = derive from the motion file


def _motion_duration_s(motion_file: str) -> float:
  data = np.load(motion_file, allow_pickle=True)
  fps = float(np.array(data["fps"]).reshape(-1)[0]) if "fps" in data else 50.0
  n = int(data["joint_pos"].shape[0])
  return n / fps


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import sim2real_task_v3  # noqa: F401  registers the V3 task ids (incl. GAPEVAL)

  cfg = tyro.cli(Cfg)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  torch.manual_seed(cfg.seed)

  episode_length_s = cfg.episode_length_s or (_motion_duration_s(cfg.motion_file) + 0.2)
  max_steps = int(episode_length_s * 50) + 100

  env_cfg = load_env_cfg(cfg.task, play=False)
  agent_cfg = load_rl_cfg(cfg.task)
  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(f"{cfg.task} is not a tracking task")
  motion_cmd.motion_file = cfg.motion_file
  motion_cmd.sampling_mode = "start"
  env_cfg.episode_length_s = episode_length_s
  env_cfg.observations["actor"].enable_corruption = False  # nominal
  env_cfg.events.pop("push_robot", None)                   # nominal
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, _asdict(agent_cfg), device=device)
  runner.load(cfg.checkpoint, map_location=device)
  policy = runner.get_inference_policy(device=device)

  asset = env.unwrapped.scene["robot"]
  joint_names = list(asset.data.joint_names) if hasattr(asset.data, "joint_names") \
    else list(asset.joint_names)
  command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
  n_j = int(command.joint_pos.shape[1])
  if len(joint_names) != n_j:
    raise RuntimeError(f"joint name count {len(joint_names)} != command dofs {n_j}")

  n = cfg.num_envs
  done_envs = torch.zeros(n, dtype=torch.bool, device=device)
  success = torch.zeros(n, dtype=torch.bool, device=device)
  sq_sum = torch.zeros(n_j, dtype=torch.float64, device=device)  # sum err^2
  n_samp = 0
  err_frames = []  # for p95: (steps, n, J) |err|, subsampled envs to bound memory

  obs = env.get_observations()
  step = 0
  while not done_envs.all() and step < max_steps:
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    active = ~done_envs
    err = (command.joint_pos - command.robot_joint_pos).abs()  # (n, J), rad
    err = torch.where(active.unsqueeze(1), err, torch.zeros_like(err))
    sq_sum += err.square().sum(dim=0).double()
    n_samp += int(active.sum().item())
    err_frames.append(err[: min(n, 8)].cpu())  # p95 on first 8 envs
    terminated = env.unwrapped.termination_manager.terminated
    truncated = env.unwrapped.termination_manager.time_outs
    newly = dones.bool() & ~done_envs
    if newly.any():
      success = success | (newly & truncated & ~terminated)
      done_envs = done_envs | newly
    step += 1

  rms_rad = (sq_sum / max(n_samp, 1)).sqrt().cpu().numpy()
  rms_deg = np.degrees(rms_rad)
  err_all = torch.stack(err_frames, 0).numpy()  # (T, 8, J)
  p95_deg = np.degrees(np.percentile(err_all, 95, axis=(0, 1)))

  per_joint = {jn: {"rms_deg": float(rms_deg[i]), "p95_deg": float(p95_deg[i])}
               for i, jn in enumerate(joint_names)}
  groups = {}
  for g, pats in GROUP_PATTERNS.items():
    idx = [i for i, jn in enumerate(joint_names) if any(p in jn for p in pats)]
    groups[g] = {
      "n_joints": len(idx),
      "rms_deg": float(np.sqrt(np.mean(rms_deg[idx] ** 2))),
      "worst_joint": joint_names[idx[int(np.argmax(rms_deg[idx]))]],
      "worst_rms_deg": float(np.max(rms_deg[idx])),
    }

  out = {
    "checkpoint": cfg.checkpoint,
    "task": cfg.task,
    "motion_file": cfg.motion_file,
    "num_envs": n,
    "success_rate": success.float().mean().item(),
    "steps_run": step,
    "groups": groups,
    "per_joint": per_joint,
  }
  Path(cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
  with open(cfg.output_file, "w") as f:
    json.dump(out, f, indent=2)

  print(f"[arm_tracking] task={cfg.task} survival={out['success_rate']:.3f}")
  for g in ("arms_all", "shoulder", "elbow", "wrist", "legs_all", "waist_all"):
    s = groups[g]
    print(f"  {g:<10} RMS {s['rms_deg']:6.2f} deg   worst {s['worst_joint']} "
          f"{s['worst_rms_deg']:.2f} deg")
  print(f"ARM_RMS_DEG={groups['arms_all']['rms_deg']:.3f}")
  print(f"[INFO] wrote {cfg.output_file}")


def _asdict(agent_cfg) -> dict:
  from dataclasses import asdict, is_dataclass

  return asdict(agent_cfg) if is_dataclass(agent_cfg) else dict(agent_cfg)


if __name__ == "__main__":
  main()
