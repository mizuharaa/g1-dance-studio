#!/usr/bin/env python
"""Smoke test for the acro task (run on the box, tiny env, ~1 min).

Checks BEFORE burning GPU-hours on train-acro-1:
  1. task registers; stock task stays uncontaminated (thresholds, pushes);
  2. train cfg: flip-aware termination classes wired with relaxed thresholds,
     push_robot removed, RSI kernel/uniform-ratio set for short skills;
  3. play cfg: flip-aware terminations present, sampling_mode start;
  4. an 8-env env builds and steps 30x with random actions; the flight-grace
     mask prints its coverage (0% on a ground dance npz is CORRECT — grace
     only opens where the reference itself is airborne).

Usage: MUJOCO_GL=egl ./envs/mjlab/bin/python cloud/smoke_acro.py <motion.npz>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import mjlab.tasks  # noqa: F401
import dynamic_skills_task as ds

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

MOTION = sys.argv[1]
ok = True


def check(name, cond, detail=""):
  global ok
  status = "PASS" if cond else "FAIL"
  print(f"[{status}] {name} {detail}")
  ok = ok and cond


train_cfg = load_env_cfg(ds.TASK_ID, play=False)
play_cfg = load_env_cfg(ds.TASK_ID, play=True)
stock_cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=False)

# 1. stock uncontaminated
check("stock anchor_pos threshold 0.25",
      stock_cfg.terminations["anchor_pos"].params["threshold"] == 0.25)
check("stock anchor_ori threshold 0.8",
      stock_cfg.terminations["anchor_ori"].params["threshold"] == 0.8)
check("stock still has push_robot", "push_robot" in stock_cfg.events)
check("stock adaptive_kernel_size 1",
      stock_cfg.commands["motion"].adaptive_kernel_size == 1)

# 2. acro train cfg
t = train_cfg.terminations
check("acro anchor_pos flip-aware cls",
      t["anchor_pos"].func is ds.anchor_pos_z_flip_aware,
      f"threshold={t['anchor_pos'].params['threshold']}")
check("acro anchor_pos threshold", t["anchor_pos"].params["threshold"] == 0.45)
check("acro anchor_ori flip-aware cls",
      t["anchor_ori"].func is ds.anchor_ori_flip_aware,
      f"threshold={t['anchor_ori'].params['threshold']}")
check("acro anchor_ori threshold", t["anchor_ori"].params["threshold"] == 1.4)
check("acro ee_body_pos flip-aware cls",
      t["ee_body_pos"].func is ds.ee_body_pos_z_flip_aware,
      f"threshold={t['ee_body_pos'].params['threshold']}")
check("acro no push_robot", "push_robot" not in train_cfg.events)
check("acro RSI kernel 4", train_cfg.commands["motion"].adaptive_kernel_size == 4)
check("acro RSI uniform 0.2",
      abs(train_cfg.commands["motion"].adaptive_uniform_ratio - 0.2) < 1e-9)
check("acro no torque penalties",
      all("torque" not in k for k in train_cfg.rewards))
check("acro keeps self_collisions", "self_collisions" in train_cfg.rewards)

# 3. play cfg
check("play flip-aware anchor_pos",
      play_cfg.terminations["anchor_pos"].func is ds.anchor_pos_z_flip_aware)
check("play sampling start", play_cfg.commands["motion"].sampling_mode == "start")

# 4. build + step
train_cfg.scene.num_envs = 8
train_cfg.commands["motion"].motion_file = MOTION
env = ManagerBasedRlEnv(cfg=train_cfg, device="cuda:0")
obs, _ = env.reset()
act_dim = env.action_manager.total_action_dim
print(f"obs actor shape: {obs['actor'].shape}, action dim: {act_dim}")
check("obs 160-dim", obs["actor"].shape == (8, 160))
for i in range(30):
  a = 0.1 * torch.randn(8, act_dim, device="cuda:0")
  env.step(a)
term = env.termination_manager
print("termination terms:", term.active_terms)
env.close()
check("30 steps ran", True)

print("SMOKE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
