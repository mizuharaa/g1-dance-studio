"""Sim2real recipe v4 "calm-legs" — fluidity-forensics-directed variant, 2026-07-06.

Why (docs/fluidity_forensics.md, data/reports/fluidity_forensics.json): the leg
clunkiness has a measured mechanism that NONE of v3a-d touches —
  * legs execute only 0.35-0.44x of the reference amplitude,
  * 3.1x intrinsic 2-10 Hz leg action chatter (present in CLEAN-obs replay, so
    policy-intrinsic — the estimator was cleared),
  * leg plant lag 80-105 ms vs the 0-20 ms the policy was trained to expect.
And v3a/c RELAXED action_rate_l2 (-0.2 -> -0.1) globally, which may raise the
2-10 Hz chatter further. v4 splits the difference by GROUP.

V4 = V3A precision weights (motion_body_pos/ori 1.5, torque penalties, all v2
DR, legodom obs model, 20 s episodes), trained on the SHARP reference
(motions/thriller_deploy_v2_sharp.npz — pass at launch), PLUS:

  (a) PER-GROUP action-rate penalty (replaces the single global action_rate_l2
      to avoid double-counting): legs+waist weight -0.3 (calm the chatter where
      the wobble lives), arms -0.1 (keep the crispness v3a bought). Groups
      resolve BY JOINT NAME at manager init (class-based term, same pattern as
      sim2real_task.ankle_torque_l2). The stock action term covers all 29
      joints with (".*",) so action columns == entity joint ids (asserted).
  (b) motion_body_ang_vel weight 1.0 -> 2.0 — direct reward pressure against
      the incoherent pelvis/body sway (the 2-10 Hz gyro wobble is 3.5-5.6x the
      choreography's own rotation demand).
  (c) LEG-group actuator command delay DR stretched to 0-16 physics steps
      (0-80 ms, hold_prob 0.8) approximating the MEASURED leg plant lag
      (80-105 ms), so the policy learns not to fight its own echo. Applies to
      the three leg-dominant actuator groups (7520_14 = hip pitch/yaw +
      waist_yaw rides along, 7520_22 = hip roll/knee, ANKLE); arms and the
      waist-pitch/roll group keep the v2 hygiene 0-4 (0-20 ms).
  (d) everything else identical to V3A.

Obs stays 160-dim -> deploy runtime unchanged. Eval: stock task harness +
the SHARP npz (same as v3d), baseline reports/arm_tracking_s2rb_baseline_sharp
+ reports/fluidity_s2rb_baseline.json.

Launch (queued behind a free training slot — see cloud/V3_PROGRAM.md):
  ./envs/mjlab/bin/python cloud/train_sim2real_v4.py \
      Mjlab-Tracking-Flat-Unitree-G1-S2R-V4 \
      --env.commands.motion.motion-file motions/thriller_deploy_v2_sharp.npz \
      --env.scene.num-envs 4096 --agent.max-iterations 5000 ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sim2real_task_v3 as v3  # v3a recipe builder (also pulls in v2)

from mjlab.tasks.registry import register_mjlab_task
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

TASK_V4 = "Mjlab-Tracking-Flat-Unitree-G1-S2R-V4"

LEGS_WAIST_EXPRS = (".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint",
                    "waist_.*_joint")
ARM_EXPRS = (".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint")
W_LEGS_WAIST = -0.3
W_ARMS = -0.1

LEG_DELAY_MAX_LAG = 16   # physics steps = 80 ms (measured leg plant lag 80-105)
LEG_GROUP_PATTERNS = ("hip", "knee", "ankle", "waist_yaw")  # 7520_14/7520_22/ANKLE


class group_action_rate_l2:
  """L2 action-rate penalty over a named joint group.

  Class-based (mirrors sim2real_task.ankle_torque_l2) so the group's action
  columns resolve ONCE at manager init, by name. Valid because the tracking
  action term spans all joints with (".*",): action column == entity joint id
  (asserted at init).
  """

  def __init__(self, cfg, env):
    asset = env.scene[cfg.params["asset_cfg"].name]
    ids, names = asset.find_joints(cfg.params["joint_exprs"])
    n_joints = len(asset.find_joints((".*",))[0])
    if env.action_manager.total_action_dim != n_joints:
      raise RuntimeError(
        f"action dim {env.action_manager.total_action_dim} != joint count "
        f"{n_joints}; group action-rate indexing would be wrong")
    self._ids = torch.tensor(ids, device=env.device, dtype=torch.long)

  def __call__(self, env, asset_cfg, joint_exprs):
    am = env.action_manager
    da = am.action[:, self._ids] - am.prev_action[:, self._ids]
    return torch.sum(torch.square(da), dim=1)


def _apply_v4(cfg, train: bool):
  cfg = v3._apply_v3a(cfg, train=train)

  # (a) per-group action-rate: REPLACE the global term (no double-counting).
  del cfg.rewards["action_rate_l2"]
  cfg.rewards["action_rate_legs_waist_l2"] = RewardTermCfg(
    func=group_action_rate_l2,
    weight=W_LEGS_WAIST,
    params={"asset_cfg": SceneEntityCfg("robot"), "joint_exprs": LEGS_WAIST_EXPRS},
  )
  cfg.rewards["action_rate_arms_l2"] = RewardTermCfg(
    func=group_action_rate_l2,
    weight=W_ARMS,
    params={"asset_cfg": SceneEntityCfg("robot"), "joint_exprs": ARM_EXPRS},
  )

  # (b) body angular-velocity tracking pressure against the incoherent sway.
  cfg.rewards["motion_body_ang_vel"].weight = 2.0

  if not train:
    return cfg

  # (c) stretch the LEG actuator groups' command delay to the measured plant
  # lag. v3a already set every group to 0-4 (hold_prob 0.8, per-env phase);
  # only max_lag changes here, on the leg-dominant groups.
  robot = cfg.scene.entities["robot"]  # already deep-copied by the v2 apply
  stretched = []
  for act in robot.articulation.actuators:
    names = tuple(getattr(act, "target_names_expr", ()) or ())
    if names and all(any(p in n for p in LEG_GROUP_PATTERNS) for n in names):
      act.delay_max_lag = LEG_DELAY_MAX_LAG
      stretched.append(names)
  if len(stretched) != 3:
    raise RuntimeError(
      f"expected 3 leg-dominant actuator groups (7520_14/7520_22/ANKLE), "
      f"got {stretched}")
  return cfg


def _make(train: bool, play: bool):
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  return _apply_v4(cfg, train=train)


register_mjlab_task(
  task_id=TASK_V4,
  env_cfg=_make(train=True, play=False),
  play_env_cfg=_make(train=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
