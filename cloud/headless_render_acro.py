#!/usr/bin/env python
"""Headless checkpoint -> MP4 with the acro task registered.

Same shape as cloud/headless_render_v3.py (unedited, history) but registers
cloud/dynamic_skills_task.py so acro candidates render on the acro task
(flip-aware terminations; play cfg starts at frame 0, no DR/noise).

Usage: headless_render_acro.py <checkpoint.pt> <motion.npz> <out.mp4> [steps] [task]
"""
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import mjlab.tasks  # noqa: F401
import dynamic_skills_task  # noqa: F401  registers Mjlab-Tracking-Flat-Unitree-G1-Acro

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.wrappers import VideoRecorder

ckpt, motion, out_mp4 = sys.argv[1], sys.argv[2], sys.argv[3]
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 500
TASK = sys.argv[5] if len(sys.argv) > 5 else dynamic_skills_task.TASK_ID
device = "cuda:0"

env_cfg = load_env_cfg(TASK)
agent_cfg = load_rl_cfg(TASK)
env_cfg.scene.num_envs = 1
env_cfg.commands["motion"].motion_file = motion
env_cfg.viewer.height, env_cfg.viewer.width = 480, 640

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
out_dir = Path(out_mp4).parent
out_dir.mkdir(parents=True, exist_ok=True)
env = VideoRecorder(env, video_folder=str(out_dir / "_vid"),
                    step_trigger=lambda s: s == 0, video_length=steps,
                    disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), device=device)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

obs = env.get_observations()
with torch.inference_mode():
    for _ in range(steps + 5):
        act = policy(obs)
        obs, _, _, _ = env.step(act)
env.close()
vids = list((out_dir / "_vid").glob("*.mp4"))
if vids:
    vids[0].rename(out_mp4)
    print("WROTE", out_mp4)
else:
    print("NO_VIDEO"); sys.exit(1)
