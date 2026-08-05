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

Harness v2 (2026-07-21, audit F4):
  * Every condition is built EXPLICITLY from the task's PLAY cfg by
    make_condition_cfg (no DR/RSI/delay inherited from the train cfg); the
    train cfg is only a donor for the exact DR/push event definitions on rows
    that name them. `clean` provably contains nothing injected.
  * Rows: clean, one-factor (dr_nominal, noise, push, cmd_delay*, obs_delay*)
    and honestly-named composites (dr_delay40ms_push, ...). v1 names remain
    valid --only selectors (mapped, see LEGACY_ROW_MAP) but are NOT emitted:
    v1 rows silently carried DR + RSI + obs delay, so no v1 name's semantics
    survive. Cross-version comparison must check gap.json harness_version.
  * Paired seeds: the SAME seed for every row (common random numbers);
    --seeds gives a repetition list, recorded per row.
  * Per-condition `realized` block (CONVENTIONS §3.4) written from the FINAL
    cfg values, not intentions.
  * EXACT horizon: episode_length_s = frames/fps (no +0.2 padding). Stepping
    past the last frame makes pinned mjlab teleport survivors to frame zero
    and rescore the start. Success = reached T. Entry/exit handoff is
    explicitly OUT OF SCOPE (separate future scenario).

Run on the box:
  NB=/workspace/notebook-data
  $NB/envs/mjlab/bin/python $NB/cloud/sim_gap_check.py \
    --checkpoint <model.pt> --motion-file $NB/motions/thriller_deploy.npz \
    --num-envs 64 --output-file $NB/reports/sim_gap_check_<tag>.json
"""

from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import torch

# mjlab/tyro exist only in the box training env. The pure harness pieces
# (condition table, cfg builder, realized-block extraction, horizon math) must
# import on the laptop for tests/test_eval_harness.py, so the heavy imports are
# guarded; main() refuses to run without them.
try:
  import tyro
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.tasks.tracking.mdp import MotionCommandCfg
  from mjlab.tasks.tracking.mdp.commands import MotionCommand
  from mjlab.tasks.tracking.mdp.metrics import compute_mpkpe, compute_root_relative_mpkpe
  from mjlab.utils.torch import configure_torch_backends

  _EVAL_IMPORT_ERROR: Exception | None = None
except ImportError as _e:  # laptop / CI: pure functions only
  tyro = None
  ManagerBasedRlEnv = MjlabOnPolicyRunner = RslRlVecEnvWrapper = None
  load_env_cfg = load_rl_cfg = load_runner_cls = None
  MotionCommandCfg = MotionCommand = None
  compute_mpkpe = compute_root_relative_mpkpe = None
  configure_torch_backends = None
  _EVAL_IMPORT_ERROR = _e

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

# ---------------------------------------------------------------------------
# Harness v2 condition table (audit F4). Delay rationale unchanged from v1:
# the deployed robot's measured effective command->response latency was
# 40-80 ms (telemetry cross-correlation, data/telemetry/latency_diag_20260709/);
# 60/80 ms rows make that regime VISIBLE and 40 ms rows are GATED (worst_names
# below) so a policy that would fall at hardware latency can't pass.
HARNESS_VERSION = 2

# Deploy-measured obs terms that get observation delay (must mirror
# cloud/sim2real_task.py DELAYED_OBS_TERMS; duplicated so this module imports
# without mjlab — sim2real_task pulls in mjlab at module level).
DELAYED_OBS_TERMS = (
  "motion_anchor_pos_b",
  "motion_anchor_ori_b",
  "base_lin_vel",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
)


@dataclass(frozen=True)
class ConditionSpec:
  """One eval row: exactly the knobs its name declares, nothing inherited."""

  name: str
  cmd_delay_steps: tuple[int, int] = (0, 0)  # physics steps, 5 ms each
  obs_delay_steps: tuple[int, int] = (0, 0)  # control steps, 20 ms each
  noise: bool = False
  push: bool = False
  startup_dr: bool = False


CONDITIONS_V2: tuple[ConditionSpec, ...] = (
  # Clean deterministic baseline: NO DR, RSI, delay, noise or push. With
  # everything zeroed all envs are identical — this row is a (near-)
  # deterministic reference rollout, not a distribution.
  ConditionSpec("clean"),
  # One-factor rows: exactly one knob each (vs clean).
  ConditionSpec("dr_nominal", startup_dr=True),
  ConditionSpec("noise", noise=True),
  ConditionSpec("push", push=True),
  ConditionSpec("cmd_delay20ms", cmd_delay_steps=(4, 4)),
  ConditionSpec("cmd_delay40ms", cmd_delay_steps=(8, 8)),
  ConditionSpec("cmd_delay60ms", cmd_delay_steps=(12, 12)),
  ConditionSpec("cmd_delay80ms", cmd_delay_steps=(16, 16)),
  ConditionSpec("obs_delay20ms", obs_delay_steps=(1, 1)),
  ConditionSpec("obs_delay80ms", obs_delay_steps=(4, 4)),
  # Composite robustness rows (the old *_push campaign, honestly named). Each
  # ALSO carries obs noise and the trained 0-1 control-step obs-delay band —
  # the full deploy-realism stack; the per-row `realized` block records it.
  ConditionSpec("dr_delay20ms_push", cmd_delay_steps=(4, 4),
                obs_delay_steps=(0, 1), noise=True, push=True, startup_dr=True),
  ConditionSpec("dr_delay40ms_push", cmd_delay_steps=(8, 8),
                obs_delay_steps=(0, 1), noise=True, push=True, startup_dr=True),
  ConditionSpec("dr_delay60ms_push", cmd_delay_steps=(12, 12),
                obs_delay_steps=(0, 1), noise=True, push=True, startup_dr=True),
  ConditionSpec("dr_delay80ms_push", cmd_delay_steps=(16, 16),
                obs_delay_steps=(0, 1), noise=True, push=True, startup_dr=True),
)

CONDITION_BY_NAME = {c.name: c for c in CONDITIONS_V2}

# harness-v1 name -> closest v2 row, for --only selectors in existing shell
# pipelines ONLY. No v1 name is emitted as an output alias: v1 rows silently
# carried startup DR, RSI and obs delay (even "nominal"), so no v1 name's
# semantics actually match a v2 row. Cross-version comparisons must check
# gap.json's harness_version.
LEGACY_ROW_MAP = {
  "nominal": "clean",
  "delay10ms": "cmd_delay20ms",  # nearest v2 rung; v1's 10 ms line was dropped
  "delay20ms": "cmd_delay20ms",
  "delay40ms": "cmd_delay40ms",
  "delay60ms": "cmd_delay60ms",
  "delay80ms": "cmd_delay80ms",
  "delay20ms_push": "dr_delay20ms_push",
  "delay40ms_push": "dr_delay40ms_push",
  "delay60ms_push": "dr_delay60ms_push",
  "delay80ms_push": "dr_delay80ms_push",
}


def resolve_only(only: str) -> list[str]:
  """--only selector -> canonical v2 row names (legacy v1 names mapped)."""
  names: list[str] = []
  for raw in only.split(","):
    n = raw.strip()
    if not n:
      continue
    if n in CONDITION_BY_NAME:
      mapped = n
    elif n in LEGACY_ROW_MAP:
      mapped = LEGACY_ROW_MAP[n]
      print(f"[WARN] --only '{n}' is a harness-v1 row name -> running v2 row "
            f"'{mapped}' (semantics differ; see LEGACY_ROW_MAP)", flush=True)
    else:
      raise SystemExit(f"unknown condition '{n}' (valid: "
                       f"{', '.join(CONDITION_BY_NAME)})")
    if mapped not in names:
      names.append(mapped)
  if not names:
    raise SystemExit(f"--only '{only}' selected no conditions")
  return names


def make_condition_cfg(
  base_play_cfg,
  *,
  seed: int,
  cmd_delay_steps: tuple[int, int] = (0, 0),
  obs_delay_steps: tuple[int, int] = (0, 0),
  noise: bool = False,
  push: bool = False,
  startup_dr: bool = False,
  donor_train_cfg=None,
):
  """Build one eval env cfg from the task's PLAY cfg, adding ONLY what the row
  names. Everything else is EXPLICITLY zeroed — never inherited: the play cfg
  itself still carries startup DR (base_com/encoder_bias/foot_friction in the
  stock task) and a custom play cfg could leak delays, so zeroing is done here
  by construction, not assumed. `donor_train_cfg` (the play=False cfg) is only
  read for the exact DR/push event definitions the policy trained under.
  Pure config surgery — no mjlab imports; caller still sets motion_file,
  episode_length_s and num_envs."""
  cfg = copy.deepcopy(base_play_cfg)

  # RSI off ALWAYS: fixed frame-0 start, no reset pose/velocity randomization.
  try:
    motion = cfg.commands["motion"]
  except (KeyError, TypeError) as e:
    raise ValueError(f"cfg has no 'motion' command: {e}") from e
  motion.pose_range = {}
  motion.velocity_range = {}
  motion.sampling_mode = "start"

  # Events: start from NOTHING (the play cfg still carries startup DR).
  events = {}
  if startup_dr:
    if donor_train_cfg is None:
      raise ValueError("startup_dr=True needs donor_train_cfg (play=False cfg)")
    for ev_name, ev in donor_train_cfg.events.items():
      if getattr(ev, "mode", None) == "startup":
        events[ev_name] = copy.deepcopy(ev)
    if not events:
      raise ValueError("donor_train_cfg has no startup events to copy")
  if push:
    if donor_train_cfg is None or "push_robot" not in donor_train_cfg.events:
      raise ValueError("push=True needs donor_train_cfg with a push_robot event")
    events["push_robot"] = copy.deepcopy(donor_train_cfg.events["push_robot"])
  cfg.events = events

  # Command (actuator) delay: explicitly SET on every actuator — zero unless
  # the row names it. Constant rows use lo == hi (no per-env variation).
  lo, hi = int(cmd_delay_steps[0]), int(cmd_delay_steps[1])
  if not (0 <= lo <= hi):
    raise ValueError(f"bad cmd_delay_steps {cmd_delay_steps}")
  for act in cfg.scene.entities["robot"].articulation.actuators:
    act.delay_min_lag = lo
    act.delay_max_lag = hi
    act.delay_hold_prob = 0.0
    act.delay_update_period = 0
    if hasattr(act, "delay_per_env_phase"):
      act.delay_per_env_phase = hi > lo

  # Observation delay: explicitly zero EVERY actor term, then apply the row's
  # band to the deploy-measured terms only (mirrors training's delayed set).
  olo, ohi = int(obs_delay_steps[0]), int(obs_delay_steps[1])
  if not (0 <= olo <= ohi):
    raise ValueError(f"bad obs_delay_steps {obs_delay_steps}")
  group = cfg.observations["actor"]
  for term in group.terms.values():
    term.delay_min_lag = 0
    term.delay_max_lag = 0
  if ohi > 0:
    applied = 0
    for term_name in DELAYED_OBS_TERMS:
      term = group.terms.get(term_name)
      if term is None:
        continue
      term.delay_min_lag = olo
      term.delay_max_lag = ohi
      if hasattr(term, "delay_per_env"):
        term.delay_per_env = True
      applied += 1
    if applied == 0:
      raise ValueError("obs delay requested but no deploy-measured obs terms "
                       f"({DELAYED_OBS_TERMS}) exist in this task's actor group")

  group.enable_corruption = bool(noise)
  cfg.seed = int(seed)
  return cfg


def extract_realized(env_cfg) -> dict:
  """CONVENTIONS §3.4 `realized` block, read back from the FINAL cfg (what will
  actually run) — never from the row's intent."""
  acts = list(env_cfg.scene.entities["robot"].articulation.actuators)
  cmd_lo = min((int(a.delay_min_lag) for a in acts), default=0)
  cmd_hi = max((int(a.delay_max_lag) for a in acts), default=0)
  terms = env_cfg.observations["actor"].terms
  delayed = [t for t in terms.values()
             if int(getattr(t, "delay_max_lag", 0) or 0) > 0]
  obs_hi = max((int(t.delay_max_lag) for t in delayed), default=0)
  obs_lo = min((int(t.delay_min_lag) for t in delayed), default=0)
  motion = env_cfg.commands["motion"]
  play_mode = (getattr(motion, "sampling_mode", None) == "start"
               and not getattr(motion, "pose_range", None)
               and not getattr(motion, "velocity_range", None))
  return {
    "seed": int(env_cfg.seed),
    "play_mode": bool(play_mode),
    "startup_dr": any(getattr(ev, "mode", None) == "startup"
                      for ev in env_cfg.events.values()),
    "cmd_delay_steps": [cmd_lo, cmd_hi],
    "obs_delay_steps": [obs_lo, obs_hi],
    "noise": bool(env_cfg.observations["actor"].enable_corruption),
    "push": "push_robot" in env_cfg.events,
  }


def exact_horizon(frames: int, fps: float, control_hz: float = 50.0) -> tuple[float, int]:
  """EXACT eval horizon: episode_length_s = frames/fps, T = that in control
  steps. NO padding — stepping past the last reference frame makes pinned
  mjlab wrap survivors to frame zero (teleport) and rescore the start."""
  if frames <= 0 or fps <= 0:
    raise ValueError(f"bad horizon inputs frames={frames} fps={fps}")
  episode_length_s = frames / float(fps)
  return episode_length_s, int(round(episode_length_s * control_hz))


def gap_payload(
  *,
  task: str,
  checkpoint: str,
  onnx: str,
  motion_file: str,
  episode_length_s: float,
  horizon_steps: int,
  seeds: list[int],
  gate,
  conditions: dict,
  repeats: dict | None = None,
) -> dict:
  """Assemble the gap.json payload (pure; unit-tested for the v2 stamp)."""
  payload = {
    "harness_version": HARNESS_VERSION,
    "task": task,
    "checkpoint": checkpoint,
    "onnx": onnx,
    "motion_file": motion_file,
    "episode_length_s": episode_length_s,
    "horizon_steps": horizon_steps,
    "seeds": list(seeds),
    "gate": gate,
    "conditions": conditions,
  }
  if repeats:
    payload["repeats"] = repeats
  return payload

# Gates per the first-principles audit (docs/first_principles_audit.md §3):
# the true ankle floor is ~5-7 Nm/ankle (pose + choreography), so <=5 worst-case is
# unpassable without tracking sacrifice; thermal is gated on RMS (heat ~ tau^2):
# predicted rate 22.5*(RMS/20)^2 <= ~8 C/min  =>  RMS <= ~12 Nm.
GATE = {
  "survival_nominal_min": 0.99,
  "survival_worst_min": 0.95,
  "ankle_mean_nominal_max_nm": 6.0,
  "ankle_mean_worst_max_nm": 8.0,
  # Ankle p95 bars are env-overridable (defaults unchanged for v5-v9 reruns).
  # v10 exports 22/25: the real robot measured ankle p95 15-19 Nm at NATIVE tempo
  # (decision log 2026-07-20, experiments/torque_crosscheck_20260720) — the old
  # 15 Nm nominal bar sat BELOW physical reality.
  "ankle_p95_nominal_max_nm": float(os.environ.get("G1_GATE_ANKLE_P95_NOMINAL_NM", "15.0")),
  "ankle_p95_worst_max_nm": float(os.environ.get("G1_GATE_ANKLE_P95_WORST_NM", "20.0")),
  "ankle_rms_worst_max_nm": 12.0,  # thermal projection
  # Quality bar is ROOT-RELATIVE mpkpe (baseline a2 = 0.089 on this harness): the
  # 2026-07-05 s2r eval showed global mpkpe conflates stage DRIFT with dance
  # quality (s2r: global 0.52 but root-relative 0.084 — crisper than baseline).
  # Global mpkpe is reported as info; drift gets its own stage-keeping bar
  # (2 m-radius area, 1.5 m excursion vet limit).
  "rr_mpkpe_nominal_max_m": 0.10,
  # Gate on the 95th-percentile per-episode worst point, NOT the single-worst
  # timestep. Bar 1.5 m = the vet's root-excursion limit on the 2 m-radius stage
  # (a show that keeps 95% of runs within 1.5 m of the anchor stays on stage; the
  # operator re-centres between pieces). max_m is still REPORTED as a stress read.
  # Env-overridable. Rationale: experiments/drift_rootcause_20260720 (clean drift
  # 0.87 m, friction-independent; the 3.5 m max was an unlucky-init tail outlier).
  "drift_nominal_p95_max_m": float(os.environ.get("G1_GATE_DRIFT_P95_M", "1.5")),
  # 2026-08-05 hardware audit bars. The real robot measured 40-60 ms latency and
  # v12 collapsed past 60 (trained DR topped at 60 ms) — gate survival AT the top
  # of the real band. Leg amplitude: hardware achieved 58% (v12) / 64% (anchor)
  # of commanded — the show-quality gap no other bar saw. Default 0 = report-only
  # (old runs stay comparable); run_attempt11 sets real bars.
  "delay60_survival_min": float(os.environ.get("G1_GATE_DELAY60_SURVIVAL_MIN", "0.95")),
  "leg_amp_ratio_min": float(os.environ.get("G1_GATE_LEG_AMP_MIN", "0")),
}

def _reference_joint_pos(motion_file):
  """Reference joint trajectory [T,29] from the motion npz (None if unreadable)."""
  try:
    import numpy as _np
    d = _np.load(motion_file, allow_pickle=True)
    for k in ("joint_pos", "dof_pos", "jp"):
      if k in getattr(d, "files", []):
        return _np.asarray(d[k], dtype=float)
  except Exception:
    pass
  return None


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
  episode_length_s: float = 0.0  # 0 = derive EXACT horizon from the motion file
  quick: bool = False  # smoke test: 8 envs, 2 conditions, <=300 steps
  only: str = ""  # comma-separated condition names to run (e.g. "clean"; v1 names mapped)
  seeds: str = ""  # comma-separated repetition seeds; empty = [seed]. Each seed
                   # runs the WHOLE matrix (paired rows); extras land in "repeats".


def _motion_frames_fps(motion_file: str) -> tuple[int, float]:
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
  return n, fps


def _motion_duration_s(motion_file: str) -> float:
  n, fps = _motion_frames_fps(motion_file)
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
  spec: ConditionSpec,
  seed: int,
  base_play_cfg,
  donor_train_cfg,
  agent_cfg,
  episode_length_s: float,
  max_steps: int,
) -> dict:
  # Explicit condition construction (harness v2): start from the PLAY cfg and
  # add ONLY what the row names; the same seed is used for every row (paired
  # comparisons / common random numbers).
  env_cfg = make_condition_cfg(
    base_play_cfg,
    seed=seed,
    cmd_delay_steps=spec.cmd_delay_steps,
    obs_delay_steps=spec.obs_delay_steps,
    noise=spec.noise,
    push=spec.push,
    startup_dr=spec.startup_dr,
    donor_train_cfg=donor_train_cfg,
  )

  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(f"{cfg.task} is not a tracking task")
  motion_cmd.motion_file = cfg.motion_file
  env_cfg.episode_length_s = episode_length_s
  env_cfg.scene.num_envs = cfg.num_envs

  # §3.4 realized block: read back from the FINAL cfg right before the env is
  # built — records what actually runs, not what the row intended.
  realized = extract_realized(env_cfg)

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
  # Joint-amplitude audit (2026-08-05): collect q so the gate can score achieved
  # vs reference amplitude — the metric that matched hardware (legs 58-64%
  # achieved while every gate bar passed). CPU accumulation, ~40 MB per condition.
  q_acc = []
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

    q_acc.append(asset.data.joint_pos.detach().cpu())
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

  # EXACT horizon: never step past T (audit F4 — past the last reference frame
  # pinned mjlab teleports survivors to frame zero and rescores the start).
  assert step <= max_steps, f"steps_run {step} > T {max_steps}"
  # success = reached T. Envs still alive when the loop hits T survived the
  # whole motion even if their time_out flag would land on the same step as
  # the motion wrap. Entry/exit handoff is a separate future scenario.
  success = success | ~done_envs

  # Achieved/reference amplitude per joint group. Reference amp comes from the
  # motion the env tracks; achieved from the collected q trace (surviving envs).
  amp_ratios = {}
  try:
    q_trace = torch.stack(q_acc, 0)                     # [T, envs, 29]
    jn_all = list(asset.data.joint_names) if hasattr(asset.data, "joint_names") else list(asset.joint_names)
    ref_jp = _reference_joint_pos(cfg.motion_file)       # [T_ref, 29] numpy, env joint order
    if ref_jp is not None:
      import numpy as _np
      ach = (q_trace - q_trace.mean(dim=0, keepdim=True)).abs().quantile(0.95, dim=0)  # [envs,29]
      ach = ach.mean(dim=0).numpy()                      # [29]
      ref = _np.percentile(_np.abs(ref_jp - ref_jp.mean(axis=0)), 95, axis=0)
      ratio = ach / _np.maximum(ref, 1e-3)
      def _grp(pats):
        ids = [i for i, n in enumerate(jn_all) if any(pt in n for pt in pats)]
        return float(ratio[ids].mean()) if ids else None
      amp_ratios = {
        "leg_amp_ratio": _grp(("hip", "knee", "ankle")),
        "arm_amp_ratio": _grp(("shoulder", "elbow", "wrist")),
      }
  except Exception as e:  # diagnostics must never kill the gate run
    amp_ratios = {"error": repr(e)}

  active_steps = torch.stack(active_acc, 0).sum(0).clamp(min=1)
  mpkpe = (torch.stack(mpkpe_acc, 0).sum(0) / active_steps).mean().item()
  rr_mpkpe = (torch.stack(rr_mpkpe_acc, 0).sum(0) / active_steps).mean().item()
  drift_all = torch.stack(drift_acc, 0)            # [T, num_envs] XY anchor error
  drift_vals = drift_all[~torch.isnan(drift_all)]
  # Per-EPISODE worst point (how far each run ever strays from the anchor). The
  # p95 over episodes is the honest "typical worst-case per show" on the 2 m stage;
  # the old headline (max over T*num_envs = the single worst timestep of the worst
  # of 128 heavily-perturbed episodes) is a statistical outlier, not a quality
  # metric — decisively so per experiments/drift_rootcause_20260720: a CLEAN
  # deterministic rollout drifts only 0.87 m (friction-independent 0.79-0.94 m over
  # mu 0.3-1.3), while this max read hit 3.5 m purely from unlucky init in the tail.
  ep_max = torch.nan_to_num(drift_all, nan=0.0).max(dim=0).values  # [num_envs]
  ep_max = ep_max[ep_max > 0] if (ep_max > 0).any() else ep_max
  drift = {
    "mean_m": drift_vals.mean().item() if drift_vals.numel() else None,
    "max_m": drift_vals.max().item() if drift_vals.numel() else None,
    "p95_m": drift_vals.quantile(0.95).item() if drift_vals.numel() else None,
    "episode_max_median_m": ep_max.median().item() if ep_max.numel() else None,
    "episode_max_p95_m": ep_max.quantile(0.95).item() if ep_max.numel() else None,
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
    "condition": spec.name,
    "realized": realized,  # CONVENTIONS §3.4 — from the FINAL cfg
    "cmd_delay_ms": [realized["cmd_delay_steps"][0] * 5,
                     realized["cmd_delay_steps"][1] * 5],
    "obs_delay_ms": [realized["obs_delay_steps"][0] * 20,
                     realized["obs_delay_steps"][1] * 20],
    "num_episodes": n,
    "success_rate": success.float().mean().item(),
    "n_success": int(success.sum().item()),
    "mpkpe_m": mpkpe,
    "mpkpe_root_rel_m": rr_mpkpe,
    "drift": drift,
    "steps_run": step,
    "ankle_pitch": ankle_pitch_stats,
    "amp_ratios": amp_ratios,
    "torques_nm": torques,
    "sections": sections,
    "crosscheck_step200": crosscheck,
    "seed": env_cfg.seed,
  }
  env.close()
  return out


def main() -> None:
  if _EVAL_IMPORT_ERROR is not None:
    raise SystemExit(f"sim_gap_check needs the mjlab box env: {_EVAL_IMPORT_ERROR}")
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

  if cfg.episode_length_s:
    episode_length_s = cfg.episode_length_s
    max_steps = int(round(episode_length_s * 50))
  else:
    # EXACT horizon (audit F4): T = frames/fps, no +0.2 padding, no +100 slack.
    frames, fps = _motion_frames_fps(cfg.motion_file)
    episode_length_s, max_steps = exact_horizon(frames, fps)

  conditions = list(CONDITIONS_V2)
  if cfg.only:
    conditions = [CONDITION_BY_NAME[n] for n in resolve_only(cfg.only)]
  num_envs = cfg.num_envs
  if cfg.quick:
    conditions = [CONDITION_BY_NAME["clean"], CONDITION_BY_NAME["dr_delay40ms_push"]]
    cfg = Cfg(**{**cfg.__dict__, "num_envs": 8})
    num_envs = 8
    max_steps = min(max_steps, 300)

  # Paired seeds: the SAME seed for every row; --seeds adds whole-matrix
  # repetitions (recorded; extras land under "repeats" in the json).
  seeds = ([int(s) for s in cfg.seeds.split(",") if s.strip()]
           if cfg.seeds else [cfg.seed])

  # Load base cfgs ONCE: every row builds from the PLAY cfg; the train cfg is
  # only the donor for DR/push event definitions (rows that name them).
  base_play_cfg = load_env_cfg(cfg.task, play=True)
  donor_train_cfg = load_env_cfg(cfg.task, play=False)
  agent_cfg = load_rl_cfg(cfg.task)

  print(
    f"[INFO] sim_gap_check harness v{HARNESS_VERSION}: {len(conditions)} conditions "
    f"x {num_envs} envs, episode {episode_length_s:.2f}s (T={max_steps} steps, exact), "
    f"seeds={seeds}",
    flush=True,
  )

  results = {}
  repeats: dict[str, dict] = {}
  for seed_i, seed in enumerate(seeds):
    dest = results if seed_i == 0 else repeats.setdefault(str(seed), {})
    for spec in conditions:
      cond = _run_condition(
        cfg, device, spec, seed, base_play_cfg, donor_train_cfg, agent_cfg,
        episode_length_s, max_steps
      )
      dest[spec.name] = cond
      ap = cond["ankle_pitch"]
      tag = spec.name if len(seeds) == 1 else f"{spec.name}@seed{seed}"
      print(
        f"[{tag}] survival={cond['success_rate']:.3f} "
        f"({cond['n_success']}/{cond['num_episodes']}) "
        f"mpkpe={cond['mpkpe_m']:.3f}m rr={cond['mpkpe_root_rel_m']:.3f}m "
        f"drift p95={cond['drift'].get('episode_max_p95_m') or float('nan'):.2f}m "
        f"(max={cond['drift']['max_m']:.2f}m) "
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
  # Harness v2: the "nominal" bars now apply to the CLEAN row (true baseline —
  # v1's "nominal" silently carried DR + delays); numbers re-baseline, which is
  # the point (rerun the incumbent under v2 before comparing).
  worst_names = [n for n in ("dr_delay40ms_push", "cmd_delay40ms",
                             "dr_delay20ms_push", "cmd_delay20ms")
                 if n in results]
  if worst_names and "clean" in results:
    worst = min(worst_names, key=lambda k: results[k]["success_rate"])
    w, nom = results[worst], results["clean"]

    def _le(v, bound):
      return v is not None and v <= bound

    checks = {
      f"survival>={GATE['survival_nominal_min']} [clean]":
        nom["success_rate"] >= GATE["survival_nominal_min"],
      f"survival>={GATE['survival_worst_min']} [{worst}]":
        w["success_rate"] >= GATE["survival_worst_min"],
      **({f"survival>={GATE['delay60_survival_min']} [cmd_delay60ms]":
            results["cmd_delay60ms"]["success_rate"] >= GATE["delay60_survival_min"]}
         if "cmd_delay60ms" in results else {}),
      **({f"leg_amp>={GATE['leg_amp_ratio_min']} [clean]":
            (nom.get("amp_ratios") or {}).get("leg_amp_ratio", 0) is not None
            and (nom.get("amp_ratios") or {}).get("leg_amp_ratio", 0) >= GATE["leg_amp_ratio_min"]}
         if GATE["leg_amp_ratio_min"] > 0 else {}),
      f"ankle_mean<={GATE['ankle_mean_nominal_max_nm']}Nm [clean]":
        _le(nom["ankle_pitch"]["mean_abs"], GATE["ankle_mean_nominal_max_nm"]),
      f"ankle_mean<={GATE['ankle_mean_worst_max_nm']}Nm [{worst}]":
        _le(w["ankle_pitch"]["mean_abs"], GATE["ankle_mean_worst_max_nm"]),
      f"ankle_p95<={GATE['ankle_p95_nominal_max_nm']}Nm [clean]":
        _le(nom["ankle_pitch"]["p95_abs"], GATE["ankle_p95_nominal_max_nm"]),
      f"ankle_p95<={GATE['ankle_p95_worst_max_nm']}Nm [{worst}]":
        _le(w["ankle_pitch"]["p95_abs"], GATE["ankle_p95_worst_max_nm"]),
      f"ankle_RMS<={GATE['ankle_rms_worst_max_nm']}Nm(thermal) [{worst}]":
        _le(w["ankle_pitch"]["rms_abs"], GATE["ankle_rms_worst_max_nm"]),
      f"rr_mpkpe<={GATE['rr_mpkpe_nominal_max_m']}m [clean]":
        nom.get("mpkpe_root_rel_m") is not None
        and nom["mpkpe_root_rel_m"] <= GATE["rr_mpkpe_nominal_max_m"],
      f"drift_p95<={GATE['drift_nominal_p95_max_m']}m [clean]":
        nom.get("drift") is not None
        and nom["drift"].get("episode_max_p95_m") is not None
        and nom["drift"]["episode_max_p95_m"] <= GATE["drift_nominal_p95_max_m"],
    }
    # Record the ACTUAL numeric bars used this run (env overrides resolved) so
    # cross-version comparison can compare raw p95 Nm / drift m against one fixed
    # bar instead of a per-run PASS flag whose embedded threshold moved (audit J:
    # ankle-p95 nominal/worst went 15/20 -> 22/25 via env override in v10/v11).
    bars = {
      "survival_nominal_min": GATE["survival_nominal_min"],
      "survival_worst_min": GATE["survival_worst_min"],
      "ankle_mean_nominal_max_nm": GATE["ankle_mean_nominal_max_nm"],
      "ankle_mean_worst_max_nm": GATE["ankle_mean_worst_max_nm"],
      "ankle_p95_nominal_max_nm": GATE["ankle_p95_nominal_max_nm"],
      "ankle_p95_worst_max_nm": GATE["ankle_p95_worst_max_nm"],
      "ankle_rms_worst_max_nm": GATE["ankle_rms_worst_max_nm"],
      "rr_mpkpe_nominal_max_m": GATE["rr_mpkpe_nominal_max_m"],
      "drift_nominal_p95_max_m": GATE["drift_nominal_p95_max_m"],
    }
    gate = {"checks": checks, "pass": all(checks.values()),
            "worst_condition": worst, "bars": bars}
    print("\n=== SIM2REAL GATE ===")
    for k, v in checks.items():
      print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"SIM2REAL_GATE={'PASS' if gate['pass'] else 'FAIL'}")
    print("(baseline pre-retrain policy is expected to FAIL the delay/torque bars; "
          "the retrained policy must PASS)")

  Path(cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
  payload = gap_payload(
    task=cfg.task,
    checkpoint=cfg.checkpoint,
    onnx=cfg.onnx,
    motion_file=cfg.motion_file,
    episode_length_s=episode_length_s,
    horizon_steps=max_steps,
    seeds=seeds,
    gate=gate,
    conditions=results,
    repeats=repeats or None,
  )
  with open(cfg.output_file, "w") as f:
    json.dump(payload, f, indent=2)
  print(f"[INFO] wrote {cfg.output_file}")


if __name__ == "__main__":
  main()
