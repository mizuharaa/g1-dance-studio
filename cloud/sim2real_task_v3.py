"""Sim2real recipe v3 — DANCE-QUALITY (arm crispness) program, 2026-07-06.

Problem being solved: the PROMOTED s2r-b policy dances the full Thriller on
hardware (2589/2589 ticks, ankles cool) but the arms are visibly less crisp
than the sim rollout. Measured on hardware (telemetry 20260706-11{4445,5004,
5456} vs thriller_deploy.npz): arm RMS tracking error 13.2 deg / wrist 15.2 deg
(p95 ~27-30 deg), ~2x the sim level; two wrist joints lag 100-160 ms; several
arm joints at 0.78-0.92 amplitude ratio.

SYSTEM-ID (tools/system_id_arms.py -> data/reports/system_id_20260706.json,
3 full-dance runs):
  * arm plant lag vs the COMMANDED target: shoulders 114-141 ms, elbows 101 ms,
    wrists 81-101 ms (this is PLANT lag at kp 14.3-16.8, not obs latency —
    sensing staleness is p95 1.78 ms);
  * amplitude ratio vs command: shoulder_pitch/roll 0.83-0.92 (worst);
  * Coulomb friction fit 0.08-0.53 Nm; stiction (PD torque while stuck under a
    moving command) median 0.6-2.8 Nm on shoulders, ~0.9 elbows, 0.2-0.35 wrists;
  * torque delivery: wrist_roll 0.24-0.37 (its ~0.5 Nm command IS the friction
    floor), shoulder_yaw 0.76, elbow 0.79 — the rest 0.83-0.98.
  => the real arm plant has friction/lag sim's clean actuator lacks, and the
     trained arm gains (kp 14.25/16.78 vs the teleop-proven 80/40 on the SAME
     motors) are too soft to punch through it.

THREE VARIANTS (all keep obs 160-dim -> deploy runtime unchanged):

  V3A  "precision": the s2r (v2) recipe with the tracking/smoothness trade
       moved back toward crispness — action_rate_l2 -0.2 -> -0.1 (stock value;
       -0.2 was attempt-2's anti-jerk delta and plausibly costs sharpness),
       motion_body_pos & motion_body_ori weights 1.0 -> 1.5. Both torque
       penalties and ALL v2 DR kept.

  V3B  "arm-plant realism": V3A + train against the measured arm plant so
       training matches deploy:
         - arm-joint frictionloss DR 0.1-0.6 Nm abs (covers the measured
           Coulomb 0.08-0.53 with stiction headroom); body joints keep 0-0.4.
           The v2 single all-joint friction event is REPLACED by two disjoint
           events (body / arms) so the result can't depend on event ordering.
         - arm actuator stiffness AND damping x2.5 (both arm actuator group
           cfgs: G1_ACTUATOR_5020 = shoulders+elbow+wrist_roll, G1_ACTUATOR_4010
           = wrist_pitch/yaw — together = the four deploy gain groups).
           Why 2.5: lag ~ 1/sqrt(kp) -> 114-141 ms shoulder lag drops to
           ~70-90 ms and PD authority rises 2.5x over the friction floor;
           resulting kp 35.6 (shoulder/elbow/wrist_roll) and 41.9 (wrist p/y)
           stay INSIDE the teleop-proven envelope on these motors (80 / 40).
           Damping scales by the SAME factor: zeta = kd/(2*sqrt(kp*J)) then
           RISES by sqrt(2.5) — strictly more overdamped, no oscillation risk.
       !!! DEPLOY CONTRACT: a V3B policy is trained expecting arm kp/kd x2.5.
       It MUST be deployed with ARM_GROUND_KP_SCALE=2.5 (pipeline/
       deploy_runtime.py, mode ground-run-legodom) or an equivalently scaled
       policy_meta.json. Deploying it at the unscaled meta gains reproduces
       exactly the soft-arm gap it was trained to avoid. The exported
       policy_meta must carry a note; the autopilot writes it into RESULT.txt.

  V3C  "converge longer": V3A env, trained 10000 iterations instead of 5000
       (registered under its own id so logs/gap-checks stay unambiguous).

  V3B-GAPEVAL (not a training task): the STOCK task + the x2.5 arm gains ONLY.
       sim_gap_check evaluates every candidate on the stock harness for
       comparability with the a2/s2r baselines; a V3B policy must be evaluated
       on the plant it will actually get at deploy, so its gap check / arm
       tracking / render use this task instead of the stock one.

Launch (on the box) — see cloud/V3_PROGRAM.md for the exact job commands:
  ./envs/mjlab/bin/python cloud/train_sim2real_v3.py <TASK_ID> \
      --env.commands.motion.motion-file motions/thriller_deploy.npz \
      --env.scene.num-envs 4096 --agent.max-iterations {5000|10000} ...
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The v2 recipe module (history — imported, never edited). Importing it also
# registers the v2 task id, which is harmless.
import sim2real_task as s2r

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

TASK_V3A = "Mjlab-Tracking-Flat-Unitree-G1-S2R-V3A"
TASK_V3B = "Mjlab-Tracking-Flat-Unitree-G1-S2R-V3B"
TASK_V3C = "Mjlab-Tracking-Flat-Unitree-G1-S2R-V3C"
TASK_V3B_GAPEVAL = "Mjlab-Tracking-Flat-Unitree-G1-S2R-V3B-GAPEVAL"

# --- V3B arm-plant numbers (from data/reports/system_id_20260706.json) --------
ARM_GAIN_SCALE = 2.5              # stiffness AND damping, both arm actuator cfgs
ARM_FRICTION_RANGE = (0.1, 0.6)   # Nm abs DR, arm joints (measured 0.08-0.53)
BODY_FRICTION_RANGE = (0.0, 0.4)  # unchanged v2 range for legs/waist
ARM_PATTERNS = ("shoulder", "elbow", "wrist")
ARM_JOINT_EXPRS = (".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint")
BODY_JOINT_EXPRS = (".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint",
                    "waist_.*_joint")


def _apply_v3a(cfg, train: bool):
  """v2 recipe + the precision deltas (rewards apply to train AND play cfgs,
  mirroring v2's convention — harmless at play, logged)."""
  cfg = s2r._apply_sim2real(cfg, train=train)
  cfg.rewards["action_rate_l2"].weight = -0.1     # back to stock (v2 had -0.2)
  cfg.rewards["motion_body_pos"].weight = 1.5     # stock 1.0
  cfg.rewards["motion_body_ori"].weight = 1.5     # stock 1.0
  return cfg


def _scale_arm_actuators(cfg, scale: float):
  """Multiply stiffness+damping of every actuator cfg whose targets are ALL arm
  joints. On the G1 that is exactly G1_ACTUATOR_5020 (elbow, shoulder x3,
  wrist_roll) and G1_ACTUATOR_4010 (wrist pitch/yaw) — no leg/waist leakage.
  cfg.scene.entities['robot'] must already be a deep copy (module-level
  G1_ARTICULATION is shared process-wide)."""
  robot = cfg.scene.entities["robot"]
  hit = []
  for act in robot.articulation.actuators:
    names = tuple(getattr(act, "target_names_expr", ()) or ())
    if names and all(any(p in n for p in ARM_PATTERNS) for n in names):
      act.stiffness = act.stiffness * scale
      act.damping = act.damping * scale
      hit.append(names)
  if len(hit) != 2:
    raise RuntimeError(f"expected exactly 2 all-arm actuator groups, got {hit}")
  return cfg


def _apply_v3b(cfg, train: bool):
  cfg = _apply_v3a(cfg, train)
  # Train AND play get the boosted arm plant — play/export/eval must see the
  # same actuator the deploy contract requires (ARM_GROUND_KP_SCALE=2.5).
  _scale_arm_actuators(cfg, ARM_GAIN_SCALE)
  if not train:
    return cfg
  # Replace the v2 all-joint friction event with two DISJOINT events so arm
  # friction can't silently depend on event iteration order.
  cfg.events.pop("dr_joint_friction")
  cfg.events["dr_joint_friction_body"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "ranges": BODY_FRICTION_RANGE,
      "operation": "abs",
      "asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINT_EXPRS),
    },
  )
  cfg.events["dr_joint_friction_arms"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "ranges": ARM_FRICTION_RANGE,
      "operation": "abs",
      "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_EXPRS),
    },
  )
  return cfg


def _make(apply_fn, train: bool, play: bool):
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  return apply_fn(cfg, train=train)


def _make_gapeval(play: bool):
  """STOCK task + x2.5 arm gains only — the comparable eval harness for V3B."""
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  cfg.scene.entities["robot"] = copy.deepcopy(cfg.scene.entities["robot"])
  return _scale_arm_actuators(cfg, ARM_GAIN_SCALE)


register_mjlab_task(
  task_id=TASK_V3A,
  env_cfg=_make(_apply_v3a, train=True, play=False),
  play_env_cfg=_make(_apply_v3a, train=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
register_mjlab_task(
  task_id=TASK_V3B,
  env_cfg=_make(_apply_v3b, train=True, play=False),
  play_env_cfg=_make(_apply_v3b, train=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
# V3C is the V3A env trained longer — own id so runs/gap-checks stay unambiguous.
register_mjlab_task(
  task_id=TASK_V3C,
  env_cfg=_make(_apply_v3a, train=True, play=False),
  play_env_cfg=_make(_apply_v3a, train=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
register_mjlab_task(
  task_id=TASK_V3B_GAPEVAL,
  env_cfg=_make_gapeval(play=False),
  play_env_cfg=_make_gapeval(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
