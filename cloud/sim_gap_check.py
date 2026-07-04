"""Sim2real gap check: full-motion eval of a tracking policy under injected
real-world conditions (command latency, pushes, obs noise), measuring survival
AND leg-joint torques.

Two uses:
  * PRE-retrain, on the currently deployed policy: does injected latency/DR
    reproduce the hardware signature in sim (ankle torque rising from ~0
    toward the ~15 Nm measured on the robot, and/or falls)? This validates the
    mechanism hypothesis BEFORE spending GPU hours.
  * POST-retrain gate: the retrained policy must keep ankle torque low and
    survive the same injected conditions.

Differences vs heldout_eval.py (which this is derived from):
  * FULL MOTION: episode_length_s is set to the motion's true duration
    (heldout_eval inherited the training cfg's episode_length_s=10.0, so its
    "success" only certified the first 10 s of the dance).
  * Joint-space torque telemetry via data.qfrc_actuator (plus a one-time
    cross-check against the actuator_force joint-name indexing that
    sim_ankle.py used).
  * A conditions matrix with constant injected command delay.

Run on the box:
  NB=/workspace/notebook-data
  $NB/envs/mjlab/bin/python $NB/cloud/sim_gap_check.py \
    --checkpoint <model.pt> --motion-file $NB/motions/thriller_deploy.npz \
    --num-envs 64 --output-file $NB/reports/sim_gap_check_<tag>.json
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.metrics import compute_mpkpe
from mjlab.utils.torch import configure_torch_backends

LEG_JOINTS = (
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
)
ANKLE_PITCH = ("left_ankle_pitch_joint", "right_ankle_pitch_joint")

# name, constant command delay in physics steps (5 ms each), push, obs noise
CONDITIONS = [
  ("nominal", 0, False, False),
  ("noise", 0, False, True),
  ("delay10ms", 2, False, True),
  ("delay20ms", 4, False, True),
  ("delay40ms", 8, False, True),
  ("delay20ms_push", 4, True, True),
  ("delay40ms_push", 8, True, True),
]

GATE = {
  "survival_min": 0.99,  # worst condition
  "ankle_mean_abs_max_nm": 5.0,  # worst condition, ankle pitch
  "ankle_p95_abs_max_nm": 15.0,  # worst condition, ankle pitch
  # Deployed a2 baseline measures 0.307 on THIS harness (full motion from
  # start, train-cfg RSI perturbations) vs 0.221 on the old 10 s protocol —
  # bound is baseline parity + small DR allowance, final call is visual.
  "mpkpe_nominal_max_m": 0.33,
}


@dataclass(frozen=True)
class Cfg:
  checkpoint: str
  motion_file: str
  task: str = "Mjlab-Tracking-Flat-Unitree-G1"
  num_envs: int = 64
  seed: int = 91001
  device: str | None = None
  output_file: str = "sim_gap_check.json"
  episode_length_s: float = 0.0  # 0 = derive from the motion file
  quick: bool = False  # smoke test: 8 envs, 2 conditions, 300 steps


def _motion_duration_s(motion_file: str) -> float:
  data = np.load(motion_file, allow_pickle=True)
  fps = float(np.array(data["fps"]).reshape(-1)[0]) if "fps" in data else 50.0
  n = 0
  for key in ("joint_pos", "joint_positions", "dof_pos", "body_pos_w"):
    if key in data:
      n = int(data[key].shape[0])
      break
  if n == 0:
    arrs = [data[k] for k in data.files if hasattr(data[k], "shape") and data[k].ndim >= 2]
    n = int(max(a.shape[0] for a in arrs)) if arrs else 0
  if n == 0:
    raise ValueError(
      f"could not infer motion length from {motion_file}; pass --episode-length-s"
    )
  return n / fps


def _as_dict(agent_cfg) -> dict:
  from dataclasses import asdict, is_dataclass

  return asdict(agent_cfg) if is_dataclass(agent_cfg) else dict(agent_cfg)


def _run_condition(
  cfg: Cfg,
  device: str,
  name: str,
  delay_lag: int,
  push: bool,
  noise: bool,
  episode_length_s: float,
  cond_index: int,
  max_steps: int,
) -> dict:
  env_cfg = load_env_cfg(cfg.task, play=False)
  agent_cfg = load_rl_cfg(cfg.task)

  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(f"{cfg.task} is not a tracking task")
  motion_cmd.motion_file = cfg.motion_file
  motion_cmd.sampling_mode = "start"

  env_cfg.episode_length_s = episode_length_s
  env_cfg.observations["actor"].enable_corruption = noise
  if not push:
    env_cfg.events.pop("push_robot", None)

  if delay_lag > 0:
    # The robot EntityCfg shares module-level actuator cfg objects across all
    # tasks in this process — deep-copy before mutating delay fields.
    robot = copy.deepcopy(env_cfg.scene.entities["robot"])
    for act in robot.articulation.actuators:
      act.delay_min_lag = delay_lag
      act.delay_max_lag = delay_lag
      act.delay_hold_prob = 0.0
      act.delay_update_period = 0
    env_cfg.scene.entities["robot"] = robot

  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed + cond_index

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, _as_dict(agent_cfg), device=device)
  runner.load(cfg.checkpoint, map_location=device)
  policy = runner.get_inference_policy(device=device)

  asset = env.unwrapped.scene["robot"]
  leg_ids_list, leg_names = asset.find_joints(LEG_JOINTS)
  leg_ids = torch.tensor(leg_ids_list, device=device, dtype=torch.long)
  name_to_col = {n: i for i, n in enumerate(leg_names)}

  command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))

  n = cfg.num_envs
  done_envs = torch.zeros(n, dtype=torch.bool, device=device)
  success = torch.zeros(n, dtype=torch.bool, device=device)
  mpkpe_acc, active_acc = [], []
  tau_frames = []  # per-step (n, len(LEG_JOINTS)) |tau|, masked to active envs
  crosscheck = None

  obs = env.get_observations()
  step = 0
  while not done_envs.all() and step < max_steps:
    ref = SimpleNamespace(
      num_envs=command.num_envs,
      device=command.device,
      cfg=command.cfg,
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

    tau = asset.data.qfrc_actuator[:, leg_ids].abs()
    tau_frames.append(torch.where(active.unsqueeze(1), tau, torch.nan))

    if step == 200 and crosscheck is None:
      # H1 cross-check: does the actuator_force[joint_name_index] convention
      # (what sim_ankle.py measured) agree with joint-space qfrc_actuator?
      crosscheck = {}
      try:
        jn = list(asset.data.joint_names) if hasattr(asset.data, "joint_names") else list(asset.joint_names)
        for jname in ANKLE_PITCH:
          q = tau[:, name_to_col[jname]].nanmean().item()
          af = asset.data.actuator_force[:, jn.index(jname)].abs().mean().item()
          entry = {"qfrc_actuator_mean_abs": q, "actuator_force_jn_idx_mean_abs": af}
          if hasattr(asset.data, "applied_torque"):
            entry["applied_torque_jn_idx_mean_abs"] = (
              asset.data.applied_torque[:, jn.index(jname)].abs().mean().item()
            )
          crosscheck[jname] = entry
      except Exception as e:  # cross-check is best-effort diagnostics
        crosscheck = {"error": repr(e)}

    terminated = env.unwrapped.termination_manager.terminated
    truncated = env.unwrapped.termination_manager.time_outs
    newly = dones.bool() & ~done_envs
    if newly.any():
      success = success | (newly & truncated & ~terminated)
      done_envs = done_envs | newly
    step += 1

  active_steps = torch.stack(active_acc, 0).sum(0).clamp(min=1)
  mpkpe = (torch.stack(mpkpe_acc, 0).sum(0) / active_steps).mean().item()

  tau_all = torch.stack(tau_frames, 0)  # (T, n, J)
  torques = {}
  for jname in LEG_JOINTS:
    col = tau_all[:, :, name_to_col[jname]]
    vals = col[~torch.isnan(col)]
    if vals.numel() == 0:
      torques[jname] = {"mean_abs": None, "p95_abs": None, "max_abs": None}
      continue
    torques[jname] = {
      "mean_abs": vals.mean().item(),
      "p95_abs": vals.quantile(0.95).item(),
      "max_abs": vals.max().item(),
    }

  ankle_cols = [name_to_col[j] for j in ANKLE_PITCH]
  ankle_vals = tau_all[:, :, ankle_cols]
  ankle_vals = ankle_vals[~torch.isnan(ankle_vals)]
  ankle_pitch_stats = {
    "mean_abs": ankle_vals.mean().item() if ankle_vals.numel() else None,
    "p95_abs": ankle_vals.quantile(0.95).item() if ankle_vals.numel() else None,
    "max_abs": ankle_vals.max().item() if ankle_vals.numel() else None,
  }

  out = {
    "condition": name,
    "delay_ms": delay_lag * 5,
    "push": push,
    "obs_noise": noise,
    "num_episodes": n,
    "success_rate": success.float().mean().item(),
    "n_success": int(success.sum().item()),
    "mpkpe_m": mpkpe,
    "steps_run": step,
    "ankle_pitch": ankle_pitch_stats,
    "torques_nm": torques,
    "crosscheck_step200": crosscheck,
    "seed": env_cfg.seed,
  }
  env.close()
  return out


def main() -> None:
  import mjlab.tasks  # noqa: F401

  known_tasks = {"Mjlab-Tracking-Flat-Unitree-G1"}
  argv = [a for a in sys.argv[1:] if a not in known_tasks]
  cfg = tyro.cli(Cfg, args=argv)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  torch.manual_seed(cfg.seed)

  episode_length_s = cfg.episode_length_s or (_motion_duration_s(cfg.motion_file) + 0.2)
  max_steps = int(episode_length_s * 50) + 100

  conditions = CONDITIONS
  num_envs = cfg.num_envs
  if cfg.quick:
    conditions = [CONDITIONS[0], CONDITIONS[4]]
    cfg = Cfg(**{**cfg.__dict__, "num_envs": 8})
    num_envs = 8
    max_steps = 300

  print(
    f"[INFO] sim_gap_check: {len(conditions)} conditions x {num_envs} envs, "
    f"episode {episode_length_s:.1f}s ({max_steps} max steps)",
    flush=True,
  )

  results = {}
  for i, (name, delay, push, noise) in enumerate(conditions):
    cond = _run_condition(
      cfg, device, name, delay, push, noise, episode_length_s, i, max_steps
    )
    results[name] = cond
    ap = cond["ankle_pitch"]
    print(
      f"[{name}] survival={cond['success_rate']:.3f} "
      f"({cond['n_success']}/{cond['num_episodes']}) "
      f"mpkpe={cond['mpkpe_m']:.3f}m "
      f"ankle_pitch |tau| mean={ap['mean_abs']:.2f} p95={ap['p95_abs']:.2f} "
      f"max={ap['max_abs']:.2f} Nm",
      flush=True,
    )

  # Gate: judged on the worst injected condition present.
  gate = None
  worst_names = [n for n in ("delay40ms_push", "delay20ms_push", "delay40ms") if n in results]
  if worst_names and "nominal" in results:
    worst = min(worst_names, key=lambda k: results[k]["success_rate"])
    w = results[worst]
    checks = {
      f"survival>={GATE['survival_min']} [{worst}]": w["success_rate"] >= GATE["survival_min"],
      f"ankle_mean<={GATE['ankle_mean_abs_max_nm']}Nm [{worst}]": (
        w["ankle_pitch"]["mean_abs"] is not None
        and w["ankle_pitch"]["mean_abs"] <= GATE["ankle_mean_abs_max_nm"]
      ),
      f"ankle_p95<={GATE['ankle_p95_abs_max_nm']}Nm [{worst}]": (
        w["ankle_pitch"]["p95_abs"] is not None
        and w["ankle_pitch"]["p95_abs"] <= GATE["ankle_p95_abs_max_nm"]
      ),
      f"mpkpe<={GATE['mpkpe_nominal_max_m']}m [nominal]": (
        results["nominal"]["mpkpe_m"] <= GATE["mpkpe_nominal_max_m"]
      ),
    }
    gate = {"checks": checks, "pass": all(checks.values()), "worst_condition": worst}
    print("\n=== SIM2REAL GATE ===")
    for k, v in checks.items():
      print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"SIM2REAL_GATE={'PASS' if gate['pass'] else 'FAIL'}")
    print(
      "(pre-retrain policy: FAIL under delay conditions is EXPECTED and "
      "validates the latency hypothesis; the retrained policy must PASS)"
    )

  Path(cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
  with open(cfg.output_file, "w") as f:
    json.dump(
      {
        "task": cfg.task,
        "checkpoint": cfg.checkpoint,
        "motion_file": cfg.motion_file,
        "episode_length_s": episode_length_s,
        "gate": gate,
        "conditions": results,
      },
      f,
      indent=2,
    )
  print(f"[INFO] wrote {cfg.output_file}")


if __name__ == "__main__":
  main()
