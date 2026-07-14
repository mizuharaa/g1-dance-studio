#!/usr/bin/env python
"""Roll out the v6 policy in REAL mjlab physics and dump the qpos trajectory to a
LAFAN-order CSV — NO rendering (the GreenNode compute image has no headless GL).
Play the CSV back on the laptop (pipeline/playback_csv.py) for an HONEST video of
what the policy actually does, in the dynamics it trained on (not the mismatched
menagerie sandbox that makes every policy fall).

Usage: dump_v6_traj.py <checkpoint.pt> <motion.npz> <out.csv> [max_steps]
"""
import sys
sys.path.insert(0, "/workspace/notebook-data/cloud")
from dataclasses import asdict, is_dataclass

import numpy as np
import torch

import mjlab.tasks  # noqa: F401  populate stock registry
import sim2real_task_v6  # noqa: F401  register V6
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK = "Mjlab-Tracking-Flat-Unitree-G1-S2R-V6"
CKPT, MOTION, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MAXN = int(sys.argv[4]) if len(sys.argv) > 4 else 2600

configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

env_cfg = load_env_cfg(TASK, play=True)   # nominal: no DR / no random pushes (deploy-like)
env_cfg.scene.num_envs = 1
mc = env_cfg.commands["motion"]
mc.motion_file = MOTION
mc.sampling_mode = "start"          # deterministic frame-0 start (heldout got 100% here)
agent_cfg = load_rl_cfg(TASK)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg) if is_dataclass(agent_cfg) else dict(agent_cfg), device=device)
runner.load(CKPT, map_location=device)
policy = runner.get_inference_policy(device=device)

robot = env.unwrapped.scene["robot"]
rows = []
obs = env.get_observations()
for step in range(MAXN):
    with torch.no_grad():
        actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    p = robot.data.root_link_pos_w[0].detach().cpu().numpy()    # (3)
    w = robot.data.root_link_quat_w[0].detach().cpu().numpy()   # (4) wxyz
    j = robot.data.joint_pos[0].detach().cpu().numpy()          # (29) LAFAN order
    # LAFAN CSV: pos(3), quat xyzw(4), dof(29)
    rows.append([p[0], p[1], p[2], w[1], w[2], w[3], w[0], *j])
    if bool(dones[0].item()):                        # time_out or fall -> stop (honest)
        print(f"[dump] episode ended at step {step} (done)")
        break

arr = np.asarray(rows, dtype=np.float32)
np.savetxt(OUT, arr, delimiter=",", fmt="%.6f")
print(f"[dump] wrote {OUT}: {len(arr)} frames x {arr.shape[1]} cols "
      f"(root drift end={np.hypot(arr[-1,0]-arr[0,0], arr[-1,1]-arr[0,1]):.2f} m)")
env.close()
