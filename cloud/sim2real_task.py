"""Sim2real retrain task: Mjlab-Tracking-Flat-Unitree-G1-Sim2Real.

Implements the 5-item retrain plan (HANDOVER.md / PROJECT_STATE 2026-07-05 00:15)
on top of the stock G1 flat tracking task, using only native mjlab features —
no mjlab source edits:

  1. LATENCY RANDOMIZATION (prime suspect)
     - Actuator command delay: 0–8 physics steps = 0–40 ms @ dt=5 ms, sampled
       per env each step with hold_prob=0.8 (temporally correlated jitter, one
       shared command bus for all joints — mjlab fuses matching configs into a
       single DelayBuffer).
     - Observation delay: 0–1 control steps = 0–20 ms on the six MEASURED actor
       terms (anchor pos/ori, base lin/ang vel, joint pos/vel). The reference
       `command` term and `actions` are generated locally at deploy → no delay.
  2. ACTUATOR-RESPONSE DR (startup, one draw per env)
     - pd_gains scale kp/kd 0.85–1.15 (firmware gain interpretation / torque
       constant error, first order).
     - effort_limits scale 0.80–1.00 (weaker-than-spec motors, derating).
     - joint frictionloss abs 0.0–0.4 Nm (stiction the sim lacks).
     - joint armature scale 0.9–1.4 (ankle/waist 4-bar armature is explicitly
       "unknown geometry, nominal assumption" in g1_constants.py).
  3. TORQUE/ENERGY PENALTY (cool by design)
     - joint_torques_l2 over all actuators, weight -2e-5.
     - ankle_torque_l2 (custom, qfrc_actuator on ankle pitch+roll), weight
       -4e-4: ~0 cost at healthy ankle torques, ~-0.2/step at the observed
       15 Nm hardware level → policy learns to keep CoP over the ankles.
  4. OBS NOISE MATCHING LEG-ODOM
     - Stock actor noise already brackets measured leg-odom error
       (base_lin_vel ±0.5 vs measured 99% within ±0.5; anchor ±0.25 vs 0.18
       worst stepping-phase height error) → kept as is. The temporal error
       mode leg-odom actually exhibits (sustained degradation during steps)
       is covered by the obs delay above.
  5. MASS/GAIN/PUSH DR
     - base_com x-offset widened ±0.025 → ±0.05 m (static CoM error moves
       ankle torque directly: 350 N × 5 cm ≈ 17.5 Nm split across ankles).
     - torso mass scale 0.95–1.15 (covers/battery not in MJCF).
     - wrist payload add 0.0–0.6 kg per hand (Inspire hands).
     - encoder_bias widened ±0.01 → ±0.02 rad.
     - stock push_robot kept (interval 1–3 s, ±0.5 m/s).
     - gains covered by pd_gains above.

  Plus: action_rate_l2 weight -0.2 (attempt-2's winning stability delta).

Obs dims are unchanged (160) — the deploy runtime needs no changes.

Launch (on the box):
  NB=/workspace/notebook-data
  $NB/envs/mjlab/bin/python $NB/cloud/train_sim2real.py \
      Mjlab-Tracking-Flat-Unitree-G1-Sim2Real \
      --env.scene.num-envs 4096 \
      --env.commands.motion.motion-file $NB/motions/thriller_deploy.npz \
      [usual train.py args]

Gate BEFORE hardware (cloud/sim_gap_check.py, full motion, held-out seed):
  survival >= 99% AND ankle mean|tau| <= 5 Nm AND p95 <= 15 Nm under
  40 ms constant delay + pushes + obs noise.
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
CMD_DELAY_MIN_LAG = 0
CMD_DELAY_MAX_LAG = 8  # 40 ms
CMD_DELAY_HOLD_PROB = 0.8

OBS_DELAY_MAX_LAG = 1  # control steps -> 0-20 ms
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

  # --- 1. latency randomization ---
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

  # --- 5. mass / CoM / payload DR ---
  cfg.events["base_com"].params["ranges"][0] = (-0.05, 0.05)  # widen x
  cfg.events["encoder_bias"].params["bias_range"] = (-0.02, 0.02)
  cfg.events["dr_torso_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "ranges": (0.95, 1.15),
      "operation": "scale",
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )
  cfg.events["dr_hand_payload"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "ranges": (0.0, 0.6),
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
