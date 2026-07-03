"""Held-out robustness eval of a trained mjlab tracking policy.

Adapted from mjlab's tasks/tracking/scripts/evaluate.py, but:
  * loads a LOCAL checkpoint + LOCAL motion (no W&B),
  * uses a HELD-OUT seed disjoint from training,
  * runs two conditions — nominal (obs corruption on, no push) and
    push-on (obs corruption + external shoves) — so we measure both clean
    generalization and shove-recovery,
  * writes a JSON the laptop turns into a signed verdict.

This is same-ENGINE held-out verification (mjlab), NOT a different-simulator
sim2sim check — the plain-MuJoCo model isn't dynamically faithful. It catches a
policy that overfits training seeds or can't take a shove; it does NOT catch
mjlab-specific physics exploitation. Robot-day (gantry-first) is the real gate.

Run on the box:
  envs/mjlab/bin/python cloud/heldout_eval.py Mjlab-Tracking-Flat-Unitree-G1 \
    --checkpoint <model.pt> --motion-file <motion.npz> --num-envs 256 \
    --seed 90001 --output-file <out.json>
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.metrics import (
    compute_ee_orientation_error,
    compute_ee_position_error,
    compute_mpkpe,
    compute_root_relative_mpkpe,
)
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class Cfg:
    checkpoint: str
    motion_file: str
    num_envs: int = 256
    seed: int = 90001
    device: str | None = None
    output_file: str = "heldout_eval.json"


def _run_condition(task_id: str, cfg: Cfg, device: str, push: bool) -> dict:
    env_cfg = load_env_cfg(task_id, play=False)
    agent_cfg = load_rl_cfg(task_id)

    motion_cmd = env_cfg.commands.get("motion")
    if not isinstance(motion_cmd, MotionCommandCfg):
        raise ValueError(f"{task_id} is not a tracking task")
    motion_cmd.motion_file = cfg.motion_file
    motion_cmd.sampling_mode = "start"  # every episode starts at motion frame 0

    env_cfg.observations["actor"].enable_corruption = True  # held-out sensor noise
    if not push:
        env_cfg.events.pop("push_robot", None)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed + (1 if push else 0)  # held-out, disjoint per condition

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, _as_dict(agent_cfg), device=device)
    runner.load(cfg.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)

    command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
    ee_body_names = env_cfg.terminations["ee_body_pos"].params["body_names"]

    n = cfg.num_envs
    done_envs = torch.zeros(n, dtype=torch.bool, device=device)
    success = torch.zeros(n, dtype=torch.bool, device=device)
    mpkpe_acc, active_acc, ee_pos_acc = [], [], []

    obs = env.get_observations()
    step = 0
    max_steps = 4000
    while not done_envs.all() and step < max_steps:
        ref = SimpleNamespace(
            num_envs=command.num_envs, device=command.device, cfg=command.cfg,
            body_pos_w=command.body_pos_w.clone(),
            body_pos_relative_w=command.body_pos_relative_w.clone(),
            body_quat_relative_w=command.body_quat_relative_w.clone(),
            joint_vel=command.joint_vel.clone(),
        )
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        ref.robot_body_pos_w = command.robot_body_pos_w
        ref.robot_body_quat_w = command.robot_body_quat_w
        ref.robot_joint_vel = command.robot_joint_vel
        rc = cast(MotionCommand, ref)

        active = ~done_envs
        active_acc.append(active.float())
        mpkpe_acc.append(torch.where(active, compute_mpkpe(rc), 0.0))
        ee_pos_acc.append(torch.where(active, compute_ee_position_error(rc, ee_body_names), 0.0))

        terminated = env.unwrapped.termination_manager.terminated
        truncated = env.unwrapped.termination_manager.time_outs
        newly = dones.bool() & ~done_envs
        if newly.any():
            success = success | (newly & truncated & ~terminated)
            done_envs = done_envs | newly
        step += 1

    active_steps = torch.stack(active_acc, 0).sum(0).clamp(min=1)
    mpkpe = (torch.stack(mpkpe_acc, 0).sum(0) / active_steps).mean().item()
    ee_pos = (torch.stack(ee_pos_acc, 0).sum(0) / active_steps).mean().item()
    out = {
        "condition": "push" if push else "nominal",
        "num_episodes": n,
        "success_rate": success.float().mean().item(),
        "n_success": int(success.sum().item()),
        "mpkpe_m": mpkpe,
        "ee_pos_error_m": ee_pos,
        "seed": env_cfg.seed,
        "push_enabled": push,
    }
    env.close()
    return out


def _as_dict(agent_cfg) -> dict:
    from dataclasses import asdict, is_dataclass
    return asdict(agent_cfg) if is_dataclass(agent_cfg) else dict(agent_cfg)


def main() -> None:
    import mjlab.tasks  # noqa: F401
    task_id = "Mjlab-Tracking-Flat-Unitree-G1"
    argv = [a for a in sys.argv[1:] if a != task_id]
    cfg = tyro.cli(Cfg, args=argv)
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    results = {}
    for push in (False, True):
        cond = _run_condition(task_id, cfg, device, push)
        results[cond["condition"]] = cond
        print(f"[{cond['condition']}] success={cond['success_rate']:.3f} "
              f"({cond['n_success']}/{cond['num_episodes']}) mpkpe={cond['mpkpe_m']:.4f}m")

    Path(cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_file, "w") as f:
        json.dump({"task": task_id, "checkpoint": cfg.checkpoint,
                   "motion_file": cfg.motion_file, "conditions": results}, f, indent=2)
    print(f"[INFO] wrote {cfg.output_file}")


if __name__ == "__main__":
    main()
