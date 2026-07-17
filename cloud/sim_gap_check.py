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
from mjlab.tasks.tracking.mdp.metrics import compute_mpkpe, compute_root_relative_mpkpe
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
# 2026-07-10: extended past 40 ms after the hardware fall. The deployed robot's measured
# effective command->response latency was 40-80 ms (telemetry cross-correlation,
# data/telemetry/latency_diag_20260709/); 60/80 ms conditions make that regime VISIBLE
# and the 40 ms conditions are now GATED (see worst_names below) so a policy that would
# fall at hardware latency can no longer pass verification.
CONDITIONS = [
  ("nominal", 0, False, False),
  ("noise", 0, False, True),
  ("delay10ms", 2, False, True),
  ("delay20ms", 4, False, True),
  ("delay40ms", 8, False, True),
  ("delay60ms", 12, False, True),
  ("delay80ms", 16, False, True),
  ("delay20ms_push", 4, True, True),
  ("delay40ms_push", 8, True, True),
  ("delay60ms_push", 12, True, True),
  ("delay80ms_push", 16, True, True),
]

# Gates per the first-principles audit (docs/first_principles_audit.md §3):
# the true ankle floor is ~5-7 Nm/ankle (pose + choreography), so <=5 worst-case is
# unpassable without tracking sacrifice; thermal is gated on RMS (heat ~ tau^2):
# predicted rate 22.5*(RMS/20)^2 <= ~8 C/min  =>  RMS <= ~12 Nm.
GATE = {
  "survival_nominal_min": 0.99,
  "survival_worst_min": 0.95,
  "ankle_mean_nominal_max_nm": 6.0,
  "ankle_mean_worst_max_nm": 8.0,
  "ankle_p95_nominal_max_nm": 15.0,
  "ankle_p95_worst_max_nm": 20.0,
  "ankle_rms_worst_max_nm": 12.0,  # thermal projection
  # Quality bar is ROOT-RELATIVE mpkpe (baseline a2 = 0.089 on this harness): the
  # 2026-07-05 s2r eval showed global mpkpe conflates stage DRIFT with dance
  # quality (s2r: global 0.52 but root-relative 0.084 — crisper than baseline).
  # Global mpkpe is reported as info; drift gets its own stage-keeping bar
  # (2 m-radius area, 1.5 m excursion vet limit).
  "rr_mpkpe_nominal_max_m": 0.10,
  "drift_nominal_max_m": 1.0,  # max XY anchor error over the full dance
}

# Per-section reporting (seconds in thriller_deploy time): the known 14-16 s brace
# window, the mid-dance lean cluster, and the worst lean cluster (43-47 s) from the
# quasi-static analysis; plus a sliding worst-5s ankle RMS.
SECTIONS = [(0.0, 10.0), (13.0, 17.5), (25.0, 36.0), (40.0, 49.5)]


@dataclass(frozen=True)
class Cfg:
  motion_file: str
  checkpoint: str = ""  # rsl_rl .pt (mutually exclusive with --onnx)
  onnx: str = ""  # deploy-exported policy.onnx (obs[N,160-ish] + time_step inputs)
  task: str = "Mjlab-Tracking-Flat-Unitree-G1"
  task_module: str = ""  # python module to import first (registers custom tasks, e.g. sim2real_task_v8)
  num_envs: int = 64
  seed: int = 91001
  device: str | None = None
  output_file: str = "sim_gap_check.json"
  episode_length_s: float = 0.0  # 0 = derive from the motion file
  quick: bool = False  # smoke test: 8 envs, 2 conditions, 300 steps
  only: str = ""  # comma-separated condition names to run (e.g. "nominal")


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


def _extract_actor_obs(obs) -> torch.Tensor:
  """Pull the flat actor-obs tensor out of whatever the env wrapper returns."""
  if torch.is_tensor(obs):
    return obs
  if isinstance(obs, (tuple, list)):
    return _extract_actor_obs(obs[0])
  for key in ("actor", "policy"):
    try:
      v = obs[key]
      if torch.is_tensor(v):
        return v
    except (KeyError, TypeError, IndexError):
      pass
  if hasattr(obs, "policy") and torch.is_tensor(obs.policy):
    return obs.policy
  raise ValueError(f"cannot extract actor obs from {type(obs)}")


class _OnnxPolicy:
  """Rollout policy backed by a deploy-exported policy.onnx (same contract as
  pipeline/deploy_runtime.py: inputs obs float32 + time_step float32 [[tick]],
  output 'actions'). Lets the gate score OLD promoted policies whose training
  checkpoints no longer exist — the Agent A calibration path."""

  def __init__(self, path: str, device: str):
    import onnxruntime as ort

    self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    self.input_names = [i.name for i in self.sess.get_inputs()]
    self.device = device
    self.t = 0

  def __call__(self, obs) -> torch.Tensor:
    x = _extract_actor_obs(obs).detach().cpu().numpy().astype(np.float32)
    feeds = {}
    for nm in self.input_names:
      if nm == "obs":
        feeds[nm] = x
      elif nm == "time_step":
        feeds[nm] = np.full((x.shape[0], 1), float(self.t), np.float32)
      else:
        raise ValueError(f"onnx policy has unexpected input '{nm}'")
    try:
      out = self.sess.run(["actions"], feeds)[0]
    except Exception:
      # graph may pin batch=1 — fall back to a per-env loop
      outs = [
        self.sess.run(["actions"], {nm: v[i : i + 1] for nm, v in feeds.items()})[0]
        for i in range(x.shape[0])
      ]
      out = np.concatenate(outs, 0)
    self.t += 1
    return torch.as_tensor(out, dtype=torch.float32, device=self.device)


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

  if cfg.onnx:
    policy = _OnnxPolicy(cfg.onnx, device)
  else:
    runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, _as_dict(agent_cfg), device=device)
    runner.load(cfg.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)

  asset = env.unwrapped.scene["robot"]
  leg_ids_list, leg_names = asset.find_joints(LEG_JOINTS)
  leg_ids = torch.tensor(leg_ids_list, device=device, dtype=torch.long)
  name_to_col = {n: i for i, n in enumerate(leg_names)}

  command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
  try:
    anchor_i = list(command.cfg.body_names).index(command.cfg.anchor_body_name)
  except (ValueError, AttributeError):
    anchor_i = 7  # torso_link position in the G1 tracking body list

  n = cfg.num_envs
  done_envs = torch.zeros(n, dtype=torch.bool, device=device)
  success = torch.zeros(n, dtype=torch.bool, device=device)
  done_step = torch.full((n,), -1, dtype=torch.long, device=device)
  mpkpe_acc, rr_mpkpe_acc, active_acc, drift_acc = [], [], [], []
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
    rr_mpkpe_acc.append(torch.where(active, compute_root_relative_mpkpe(rc), 0.0))
    xy_err = (command.robot_body_pos_w[:, anchor_i, :2]
              - command.body_pos_w[:, anchor_i, :2]).norm(dim=1)
    drift_acc.append(torch.where(active, xy_err, torch.nan))

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
      done_step = torch.where(newly & terminated, torch.full_like(done_step, step), done_step)
      done_envs = done_envs | newly
    step += 1

  active_steps = torch.stack(active_acc, 0).sum(0).clamp(min=1)
  mpkpe = (torch.stack(mpkpe_acc, 0).sum(0) / active_steps).mean().item()
  rr_mpkpe = (torch.stack(rr_mpkpe_acc, 0).sum(0) / active_steps).mean().item()
  drift_all = torch.stack(drift_acc, 0)
  drift_vals = drift_all[~torch.isnan(drift_all)]
  drift = {
    "mean_m": drift_vals.mean().item() if drift_vals.numel() else None,
    "max_m": drift_vals.max().item() if drift_vals.numel() else None,
  }

  def _stats(vals):
    if vals.numel() == 0:
      return {"mean_abs": None, "p95_abs": None, "max_abs": None, "rms_abs": None}
    return {
      "mean_abs": vals.mean().item(),
      "p95_abs": vals.quantile(0.95).item(),
      "max_abs": vals.max().item(),
      "rms_abs": vals.square().mean().sqrt().item(),
    }

  tau_all = torch.stack(tau_frames, 0)  # (T, n, J)
  torques = {}
  for jname in LEG_JOINTS:
    col = tau_all[:, :, name_to_col[jname]]
    torques[jname] = _stats(col[~torch.isnan(col)])

  ankle_cols = [name_to_col[j] for j in ANKLE_PITCH]
  ankle_all = tau_all[:, :, ankle_cols]  # (T, n, 2)
  ankle_pitch_stats = _stats(ankle_all[~torch.isnan(ankle_all)])

  # Per-section ankle stats + where terminations (falls) happened.
  ds = done_step.cpu().numpy()
  sections = {}
  for lo_s, hi_s in SECTIONS:
    lo, hi = int(lo_s * 50), min(int(hi_s * 50), ankle_all.shape[0])
    if lo >= ankle_all.shape[0]:
      continue
    seg = ankle_all[lo:hi]
    falls = int(((ds >= lo) & (ds < hi)).sum())
    sections[f"{lo_s:.0f}-{hi_s:.0f}s"] = {
      **_stats(seg[~torch.isnan(seg)]), "falls": falls,
    }
  # sliding worst-5s ankle RMS (250 steps)
  per_step_ms = ankle_all.square().nanmean(dim=(1, 2))  # (T,)
  win = 250
  worst5 = None
  if per_step_ms.shape[0] >= win:
    kernel = torch.ones(win, device=per_step_ms.device) / win
    valid = torch.nan_to_num(per_step_ms, nan=0.0)
    roll = torch.nn.functional.conv1d(valid.view(1, 1, -1), kernel.view(1, 1, -1)).view(-1)
    k = int(roll.argmax().item())
    worst5 = {"start_s": k / 50.0, "rms": float(roll[k].sqrt().item())}
  sections["worst_5s_window"] = worst5

  out = {
    "condition": name,
    "delay_ms": delay_lag * 5,
    "push": push,
    "obs_noise": noise,
    "num_episodes": n,
    "success_rate": success.float().mean().item(),
    "n_success": int(success.sum().item()),
    "mpkpe_m": mpkpe,
    "mpkpe_root_rel_m": rr_mpkpe,
    "drift": drift,
    "steps_run": step,
    "ankle_pitch": ankle_pitch_stats,
    "torques_nm": torques,
    "sections": sections,
    "crosscheck_step200": crosscheck,
    "seed": env_cfg.seed,
  }
  env.close()
  return out


def main() -> None:
  import mjlab.tasks  # noqa: F401

  # Tolerate a bare positional task id (legacy callers), but never strip the
  # value of an explicit --task flag — that made "--task <stock>" unparseable.
  known_tasks = {"Mjlab-Tracking-Flat-Unitree-G1"}
  raw = sys.argv[1:]
  argv = [a for i, a in enumerate(raw)
          if a not in known_tasks or (i > 0 and raw[i - 1] == "--task")]
  cfg = tyro.cli(Cfg, args=argv)
  if bool(cfg.checkpoint) == bool(cfg.onnx):
    raise SystemExit("pass exactly one of --checkpoint or --onnx")
  if cfg.task_module:
    # registers custom tasks (e.g. sim2real_task_v8 -> ...-S2R-V8); the cloud/
    # dir is sys.path[0] when this script is run by path.
    import importlib

    importlib.import_module(cfg.task_module)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  torch.manual_seed(cfg.seed)

  episode_length_s = cfg.episode_length_s or (_motion_duration_s(cfg.motion_file) + 0.2)
  max_steps = int(episode_length_s * 50) + 100

  conditions = CONDITIONS
  if cfg.only:
    names = {n.strip() for n in cfg.only.split(",")}
    conditions = [c for c in CONDITIONS if c[0] in names]
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
      f"mpkpe={cond['mpkpe_m']:.3f}m rr={cond['mpkpe_root_rel_m']:.3f}m "
      f"drift_max={cond['drift']['max_m']:.2f}m "
      f"ankle_pitch |tau| mean={ap['mean_abs']:.2f} rms={ap['rms_abs']:.2f} "
      f"p95={ap['p95_abs']:.2f} max={ap['max_abs']:.2f} Nm",
      flush=True,
    )
    for sname, s in (cond.get("sections") or {}).items():
      if s is None:
        continue
      if sname == "worst_5s_window":
        print(f"    worst-5s ankle RMS: {s['rms']:.2f} Nm @ {s['start_s']:.1f}s", flush=True)
      else:
        print(f"    [{sname}] mean={s['mean_abs']:.2f} rms={s['rms_abs']:.2f} "
              f"p95={s['p95_abs']:.2f} falls={s['falls']}", flush=True)

  # Gate: nominal bars + worst-injected-condition bars (audit §3 numbers).
  gate = None
  # Worst GATED condition = 40 ms + push. REVISED 2026-07-10 after the hardware fall:
  # the old 20 ms gate was WRONG. The DDS/comms staleness is ~zero (p95 1.78 ms wired),
  # but the *effective command->response latency* — actuation + leg-odometry estimation —
  # measured 40-80 ms on hardware (telemetry cross-correlation, DIAGNOSIS.md). A policy
  # that fell at ~45 s passed the old 20 ms gate. Gating at 40 ms + push (a firm lower
  # bound on the real added latency) makes that failure un-passable. 60/80 ms stay in the
  # matrix as informational stress lines (they also fold in mechanical PD lag, so they
  # over-state pure added latency — reported, not gated).
  worst_names = [n for n in ("delay40ms_push", "delay40ms", "delay20ms_push", "delay20ms")
                 if n in results]
  if worst_names and "nominal" in results:
    worst = min(worst_names, key=lambda k: results[k]["success_rate"])
    w, nom = results[worst], results["nominal"]

    def _le(v, bound):
      return v is not None and v <= bound

    checks = {
      f"survival>={GATE['survival_nominal_min']} [nominal]":
        nom["success_rate"] >= GATE["survival_nominal_min"],
      f"survival>={GATE['survival_worst_min']} [{worst}]":
        w["success_rate"] >= GATE["survival_worst_min"],
      f"ankle_mean<={GATE['ankle_mean_nominal_max_nm']}Nm [nominal]":
        _le(nom["ankle_pitch"]["mean_abs"], GATE["ankle_mean_nominal_max_nm"]),
      f"ankle_mean<={GATE['ankle_mean_worst_max_nm']}Nm [{worst}]":
        _le(w["ankle_pitch"]["mean_abs"], GATE["ankle_mean_worst_max_nm"]),
      f"ankle_p95<={GATE['ankle_p95_nominal_max_nm']}Nm [nominal]":
        _le(nom["ankle_pitch"]["p95_abs"], GATE["ankle_p95_nominal_max_nm"]),
      f"ankle_p95<={GATE['ankle_p95_worst_max_nm']}Nm [{worst}]":
        _le(w["ankle_pitch"]["p95_abs"], GATE["ankle_p95_worst_max_nm"]),
      f"ankle_RMS<={GATE['ankle_rms_worst_max_nm']}Nm(thermal) [{worst}]":
        _le(w["ankle_pitch"]["rms_abs"], GATE["ankle_rms_worst_max_nm"]),
      f"rr_mpkpe<={GATE['rr_mpkpe_nominal_max_m']}m [nominal]":
        nom.get("mpkpe_root_rel_m") is not None
        and nom["mpkpe_root_rel_m"] <= GATE["rr_mpkpe_nominal_max_m"],
      f"drift_max<={GATE['drift_nominal_max_m']}m [nominal]":
        nom.get("drift") is not None
        and nom["drift"]["max_m"] <= GATE["drift_nominal_max_m"],
    }
    gate = {"checks": checks, "pass": all(checks.values()), "worst_condition": worst}
    print("\n=== SIM2REAL GATE ===")
    for k, v in checks.items():
      print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"SIM2REAL_GATE={'PASS' if gate['pass'] else 'FAIL'}")
    print("(baseline pre-retrain policy is expected to FAIL the delay/torque bars; "
          "the retrained policy must PASS)")

  Path(cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
  with open(cfg.output_file, "w") as f:
    json.dump(
      {
        "task": cfg.task,
        "checkpoint": cfg.checkpoint,
        "onnx": cfg.onnx,
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
