"""Sim2real retrain task: Mjlab-Tracking-Flat-Unitree-G1-Sim2Real (recipe v2).

Recipe re-ranked per the first-principles audit (docs/first_principles_audit.md,
2026-07-05 — verdict CRITICAL-MISTAKE-FOUND: the original "sim ankle 0 Nm vs real
15 Nm, prime suspect latency" was a measurement artifact; correctly measured the
policy is ankle-hungry even in clean sim (~6–8 Nm mean, transients to the 50 Nm
clamp) and the real excess is ~2x with a STATIC signature):

  1. TORQUE PENALTY + POSTURE — HEADLINE.
     - joint_torques_l2 (all actuators) -2e-5.
     - ankle_torque_l2 (custom, qfrc_actuator, order-safe) -4e-4: ~0 at healthy
       torques, ~-0.2/step at the 15 Nm hardware level. Success is gated on the
       ankle actually unloading (sim_gap_check: mean<=6/8 Nm, RMS<=12 Nm thermal).
  2. SYSTEM-ID-INFORMED MASS/CoM (nominal shift, DR around it).
     - Real robot ~35 kg vs 33.34 kg model: hand payload +0.40–0.70 kg per wrist
       (Inspire hands), torso mass scale 1.00–1.12 (battery/covers; never lighter
       than the model — the real robot is heavier).
     - base_com x-offset widened ±0.025 → ±0.05 m.
     - ankle joint zero-offset ±0.08 rad (parallel-ankle calibration; BeyondMimic
       table — the stock ±0.01 was 3–5x too narrow), general encoder bias ±0.02.
  3. ACTUATOR-RESPONSE DR (modest; same PD law as firmware — no "bandwidth" item).
     - pd_gains scale 0.85–1.15, effort_limits 0.80–1.00,
       frictionloss 0–0.4 Nm, armature scale 0.9–1.4 (4-bar ankle armature is a
       documented guess in g1_constants.py).
  4. OBS DYNAMICS matching the deploy estimator (not wider white noise).
     - base_lin_vel through a leg-odometry-like sensor model: first-order lag
       30–80 ms + slew limit + episodic stance-break bias (LegOdometry's measured
       error modes). Stock white-noise bands kept (leg-odom is 97.8–99% inside
       ±0.5 m/s).
     - Observation delay 0–1 control steps (0–20 ms) on the six measured terms.
  5. LATENCY DR — hygiene, not headline.
     - Command delay 0–4 physics steps (0–20 ms, hold_prob 0.8). The static walls
       are latency-free; 40 ms remains an EVAL-ONLY condition in sim_gap_check
       (baseline policy falls there; 10–40 ms training exceeded published practice).

  Plus: action_rate_l2 -0.2 (attempt-2's winning delta), episode_length_s 10→20
  (stock 10 s episodes + pushes every 1–3 s rarely train long unperturbed stance —
  the thermal-relevant behavior).

Obs dims are unchanged (160) — the deploy runtime needs no changes.

Launch (on the box):
  NB=/workspace/notebook-data
  $NB/envs/mjlab/bin/python $NB/cloud/train_sim2real.py \
      Mjlab-Tracking-Flat-Unitree-G1-Sim2Real \
      --env.scene.num-envs 4096 \
      --env.commands.motion.motion-file $NB/motions/thriller_deploy.npz \
      [usual train.py args]

Gate BEFORE hardware: cloud/sim_gap_check.py (full motion, held-out seed,
7 conditions incl. 40 ms delay + pushes; gates in that file).
"""

from __future__ import annotations

import copy

import torch

from mjlab.envs.mdp import dr
from mjlab.envs.mdp.rewards import joint_torques_l2
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1-Sim2Real"

# 1 physics step = 5 ms (mujoco timestep 0.005, decimation 4 -> 50 Hz control).
# 2026-07-09 REVISION (was 0-20 ms): the 20 ms cap was wrong for this hardware. The
# 2026-07-09 ground run drifted and fell at ~45 s; four independent signals (sim
# gap_check falls at 40 ms, hardware tilt/knee-buckle, telemetry command->response
# cross-correlation = 80 ms leg median / 60-100 ms on light arm joints, comms ruled out
# at 0.16 ms wired) put the real actuation+leg-odometry latency at 40-80 ms — OUTSIDE the
# old trained band, exactly where the policy collapsed. See
# data/telemetry/latency_diag_20260709/DIAGNOSIS.md. Randomizing the FULL 0-80 ms band
# (not just the high end) keeps sharpness on low-delay episodes while teaching robustness.
CMD_DELAY_MIN_LAG = 0
CMD_DELAY_MAX_LAG = 16  # 0-80 ms (was 4 = 20 ms)
CMD_DELAY_HOLD_PROB = 0.8

OBS_DELAY_MAX_LAG = 4  # control steps -> 0-80 ms (was 1 = 20 ms)
DELAYED_OBS_TERMS = (
  "motion_anchor_pos_b",
  "motion_anchor_ori_b",
  "base_lin_vel",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
)

ANKLE_JOINT_NAMES = (
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
)


class ankle_torque_l2:
  """L2 penalty on actuated ankle joint torque (qfrc_actuator, joint space).

  Class-based term (mirrors mjlab's electrical_power_cost pattern) so joint
  ids resolve once at manager init. qfrc_actuator is joint-space and immune
  to any actuator-ordering ambiguity in data.actuator_force.
  """

  def __init__(self, cfg, env):
    asset = env.scene[cfg.params["asset_cfg"].name]
    ids, _ = asset.find_joints(ANKLE_JOINT_NAMES)
    self._ids = torch.tensor(ids, device=env.device, dtype=torch.long)

  def __call__(self, env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    tau = asset.data.qfrc_actuator[:, self._ids]
    return torch.sum(torch.square(tau), dim=1)


class legodom_like_base_lin_vel:
  """base_lin_vel through a leg-odometry-like sensor model (audit recipe item 4).

  The deploy estimator (pipeline/leg_odometry.py) is not white noise: its measured
  error modes are (a) a first-order lag from the planted-foot blend + EMA smoother,
  (b) a slew limit (VEL_MAX_STEP 0.30 m/s per 20 ms tick), and (c) sustained bias
  episodes (~0.4 s, up to ~0.15 m/s) when feet break contact during steps. This
  term reproduces those on top of the true sensor; the ObservationTermCfg's own
  white noise/clip/delay still apply afterwards (compute -> noise -> ... -> delay).
  """

  LAG_TAU_RANGE = (0.03, 0.08)   # s, per-env first-order lag
  SLEW = 0.30                    # m/s per control step (matches VEL_MAX_STEP)
  BIAS_P_ENTER = 0.02            # per step ~ once per second at 50 Hz
  BIAS_STEPS = 20                # ~0.4 s episodes
  BIAS_MAG = 0.15                # m/s, per-axis uniform

  def __init__(self, cfg, env):
    n, dev = env.num_envs, env.device
    dt = float(getattr(env, "step_dt", 0.02))
    tau = torch.empty(n, 1, device=dev).uniform_(*self.LAG_TAU_RANGE)
    self._alpha = dt / (tau + dt)
    self._state = torch.zeros(n, 3, device=dev)
    self._bias = torch.zeros(n, 3, device=dev)
    self._bias_left = torch.zeros(n, device=dev)
    self._init = torch.zeros(n, dtype=torch.bool, device=dev)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self._state[ids] = 0.0
    self._bias[ids] = 0.0
    self._bias_left[ids] = 0.0
    self._init[ids] = False

  def __call__(self, env, sensor_name):
    from mjlab.tasks.tracking import mdp as tracking_mdp

    v_true = tracking_mdp.builtin_sensor(env, sensor_name)
    fresh = ~self._init
    if fresh.any():
      self._state[fresh] = v_true[fresh]
      self._init |= True
    # first-order lag toward the true value, slew-limited (per control step)
    step = self._alpha * (v_true - self._state)
    self._state = self._state + torch.clamp(step, -self.SLEW, self.SLEW)
    # episodic stance-break bias
    active = self._bias_left > 0
    enter = (~active) & (torch.rand_like(self._bias_left) < self.BIAS_P_ENTER)
    if enter.any():
      self._bias[enter] = (torch.rand(int(enter.sum()), 3, device=v_true.device) * 2 - 1) * self.BIAS_MAG
      self._bias_left[enter] = self.BIAS_STEPS
    self._bias_left = torch.clamp(self._bias_left - 1, min=0)
    bias = torch.where((self._bias_left > 0).unsqueeze(1), self._bias,
                       torch.zeros_like(self._bias))
    return self._state + bias


def _apply_sim2real(cfg, train: bool):
  """Mutate a freshly built G1 tracking env cfg with the sim2real deltas.

  train=False (play/export cfg) keeps the reward changes but applies NO delay
  and NO DR events, so play/export stay deterministic. sim_gap_check.py
  injects eval-time delay itself.
  """
  # get_g1_robot_cfg() shares the module-level G1_ARTICULATION instance across
  # every task in the process — deep-copy before touching actuator cfgs, or
  # the stock task registered from the same registry would inherit our delays.
  robot = copy.deepcopy(cfg.scene.entities["robot"])
  cfg.scene.entities["robot"] = robot

  # --- 3. torque/energy penalties (both modes: harmless at play, logged) ---
  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=joint_torques_l2,
    weight=-2e-5,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.rewards["ankle_torque_l2"] = RewardTermCfg(
    func=ankle_torque_l2,
    weight=-4e-4,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  # attempt-2's winning smoothness delta.
  cfg.rewards["action_rate_l2"].weight = -0.2

  if not train:
    return cfg

  # Longer episodes: 10 s + pushes every 1-3 s never trains long unperturbed stance,
  # which is exactly the thermal-relevant standing behavior (audit item 8).
  cfg.episode_length_s = 20.0

  # --- 4. obs dynamics matching the deploy estimator (leg-odometry) ---
  blv = cfg.observations["actor"].terms["base_lin_vel"]
  blv.func = legodom_like_base_lin_vel
  # (params stay {"sensor_name": "robot/imu_lin_vel"}; noise/clip/delay unchanged)

  # --- 5(latency, demoted). command-bus delay randomization ---
  for act in robot.articulation.actuators:
    act.delay_min_lag = CMD_DELAY_MIN_LAG
    act.delay_max_lag = CMD_DELAY_MAX_LAG
    act.delay_hold_prob = CMD_DELAY_HOLD_PROB
    act.delay_update_period = 0
    act.delay_per_env_phase = True

  for term_name in DELAYED_OBS_TERMS:
    term = cfg.observations["actor"].terms[term_name]
    term.delay_min_lag = 0
    term.delay_max_lag = OBS_DELAY_MAX_LAG
    term.delay_per_env = True

  # --- 2. actuator-response DR (startup: one draw per env, no reset cost) ---
  cfg.events["dr_pd_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "kp_range": (0.85, 1.15),
      "kd_range": (0.85, 1.15),
      "operation": "scale",
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.events["dr_effort_limits"] = EventTermCfg(
    mode="startup",
    func=dr.effort_limits,
    params={
      "effort_limit_range": (0.80, 1.00),
      "operation": "scale",
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.events["dr_joint_friction"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "ranges": (0.0, 0.4),
      "operation": "abs",
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )
  cfg.events["dr_joint_armature"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "ranges": (0.9, 1.4),
      "operation": "scale",
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )

  # --- 2. system-ID-informed mass / CoM / calibration DR ---
  # Real robot ~35 kg vs 33.34 kg model -> nominal shift UP with DR around it
  # (never lighter than the model; the hardware is heavier).
  cfg.events["base_com"].params["ranges"][0] = (-0.05, 0.05)  # widen x
  cfg.events["encoder_bias"].params["bias_range"] = (-0.02, 0.02)
  cfg.events["dr_ankle_zero_offset"] = EventTermCfg(
    mode="startup",
    func=dr.encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_ankle_.*_joint",)),
      "bias_range": (-0.08, 0.08),  # parallel-ankle calibration, BeyondMimic table
    },
  )
  cfg.events["dr_torso_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "ranges": (1.00, 1.12),  # battery/covers: +0 to ~1.2 kg on the ~10 kg torso
      "operation": "scale",
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.events["dr_hand_payload"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "ranges": (0.40, 0.70),  # Inspire hands ~0.55 kg each, +-DR
      "operation": "add",
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=("left_wrist_yaw_link", "right_wrist_yaw_link")
      ),
    },
  )

  return cfg


def _make(train: bool, play: bool):
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  return _apply_sim2real(cfg, train=train)


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=_make(train=True, play=False),
  play_env_cfg=_make(train=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
