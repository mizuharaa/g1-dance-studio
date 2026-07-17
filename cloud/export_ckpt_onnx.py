#!/usr/bin/env python3
"""Export an ARBITRARY mjlab tracking checkpoint to a deploy ONNX.

Why this exists: mjlab exports one ONNX per run, at training end — i.e. always the
LAST checkpoint. cloud/export_policy.py just copies that file, so whenever
pick_checkpoint selects a non-final winner (the whole point of the picker), the
staged policy.onnx silently disagrees with the gated checkpoint. This script loads
the winner .pt into the task's runner and re-runs the runner's own ONNX exporter
(motion baked in + metadata attached), so gap.json and policy.onnx describe the
SAME network.

Usage (box):
  $PY cloud/export_ckpt_onnx.py --checkpoint <model_N.pt> --motion-file <motion.npz> \
      --out-dir <exports> --task Mjlab-Tracking-Flat-Unitree-G1-S2R-V8 \
      --task-module sim2real_task_v8
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, is_dataclass

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class Cfg:
  checkpoint: str
  motion_file: str
  out_dir: str
  filename: str = "policy.onnx"
  task: str = "Mjlab-Tracking-Flat-Unitree-G1"
  task_module: str = ""  # module registering custom tasks (e.g. sim2real_task_v8)
  num_envs: int = 2  # runner needs a live env; keep it tiny
  device: str | None = None


def main() -> None:
  import mjlab.tasks  # noqa: F401

  cfg = tyro.cli(Cfg)
  if cfg.task_module:
    importlib.import_module(cfg.task_module)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task, play=False)
  agent_cfg = load_rl_cfg(cfg.task)
  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(f"{cfg.task} is not a tracking task")
  motion_cmd.motion_file = cfg.motion_file
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  agent_dict = asdict(agent_cfg) if is_dataclass(agent_cfg) else dict(agent_cfg)
  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, agent_dict, device=device)
  runner.load(cfg.checkpoint, map_location=device)
  runner.export_policy_to_onnx(cfg.out_dir, filename=cfg.filename)
  print(f"EXPORTED {cfg.checkpoint} -> {cfg.out_dir}/{cfg.filename}")


if __name__ == "__main__":
  main()
