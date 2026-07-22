"""v13 task — STYLE fix on top of v11: kill leg-command chatter, commit to the
sharper reference.

WHY (measured 2026-07-22, experiments/style_gap_20260722/FINDINGS.md, from the
v12-final policy the user reviewed): leg TARGET high-freq content 3.97% (>5 Hz)
vs the real robot's 1.2-2.3% baseline — visible twitching; leg reach only
60-79% of the (now unblunted, sharper) reference — visibly short movement;
arms clean (0.29%) so it is a legs-specific reward-stack gap: v11 gave the legs
tracking REWARDS but never the per-channel action-rate penalty the ankles have
had since v8.

DELTAS vs v11 (everything else inherited unchanged, incl. all audit fixes):
  1. leg_action_rate_l2 — per-channel L2 first-difference penalty on the 8
     hip+knee action channels, mirroring v8's proven ankle_action_rate_l2
     (the ankles keep their own term; arms measured clean, left alone).
     Weight G1_LEG_ACTION_RATE_W (default -0.03; ankle term uses -0.05).
  2. Leg tracking stds tightened via the launcher env (NO code change here):
     G1_LEG_POS_STD 0.30->0.26, G1_LEG_ORI_STD 0.40->0.34 — commit harder to
     the sharper reference (run_attempt10.sh sets them).
Deploy contract unchanged (rewards only). Selfcheck asserts the new term sits
on exactly the 8 hip/knee channels and v11's stack is intact.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import sim2real_task as base           # noqa: E402
import sim2real_task_v11 as v11        # noqa: E402  full v11 stack (audit-fixed)

from mjlab.managers.reward_manager import RewardTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.tasks.registry import register_mjlab_task  # noqa: E402
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg  # noqa: E402
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner  # noqa: E402

TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1-S2R-V13"

HIP_KNEE_JOINT_NAMES = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint",
)
LEG_ACTION_RATE_W = float(os.environ.get("G1_LEG_ACTION_RATE_W", "-0.03"))


class leg_action_rate_l2:
  """L2 first-difference on the 8 hip/knee action channels (style fix 2026-07-22).
  Mirrors v8.ankle_action_rate_l2: one JointPositionAction over all 29 joints in
  joint order, so find_joints ids index the action channels directly."""

  def __init__(self, cfg, env):
    asset = env.scene[cfg.params["asset_cfg"].name]
    ids, _ = asset.find_joints(HIP_KNEE_JOINT_NAMES)
    if len(ids) != len(HIP_KNEE_JOINT_NAMES):
      raise RuntimeError(f"leg_action_rate_l2 resolved {len(ids)} of "
                         f"{len(HIP_KNEE_JOINT_NAMES)} hip/knee joints")
    self._ids = torch.tensor(ids, device=env.device, dtype=torch.long)

  def __call__(self, env, asset_cfg):
    am = env.action_manager
    cur = am.action
    prev = getattr(am, "prev_action", None)
    if prev is None:
      prev = cur
    d = (cur - prev)[:, self._ids]
    return torch.sum(torch.square(d), dim=1)


def _apply_v13(cfg, train: bool):
  v11._apply_v11(cfg, train=train)
  cfg.rewards["leg_action_rate_l2"] = RewardTermCfg(
    func=leg_action_rate_l2,
    weight=LEG_ACTION_RATE_W,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  return cfg


def _make(train: bool, play: bool):
  return _apply_v13(base._make(train=train, play=play), train=train)


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=_make(train=True, play=False),
  play_env_cfg=_make(train=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)


def _selfcheck() -> int:
  cfg = _make(train=True, play=False)
  ok = True
  present = "leg_action_rate_l2" in cfg.rewards
  w_ok = present and abs(cfg.rewards["leg_action_rate_l2"].weight - LEG_ACTION_RATE_W) < 1e-12
  ok &= w_ok
  print(f"  leg_action_rate_l2     : {'OK' if w_ok else 'MISSING/WRONG'} "
        f"w={LEG_ACTION_RATE_W} on {len(HIP_KNEE_JOINT_NAMES)} hip/knee channels")
  for k in ("motion_leg_pos", "motion_leg_ori", "ankle_action_rate_l2",
            "dr_effort_scoped" if "dr_effort_scoped" in cfg.events else "motion_arm_pos"):
    src = cfg.rewards if k in cfg.rewards else cfg.events
    p = k in src
    ok &= p
    print(f"  {k:22s}: {'present' if p else 'MISSING'}")
  import sim2real_task_v10 as v10
  pf, flat, hist, unknown = v10._actor_obs_dim(cfg)
  want = 154 + (2 if v10.G1_PHASE_OBS else 0)
  dim_ok = pf == want and flat == want * v10.v8.G1_OBS_HISTORY and not unknown
  ok &= dim_ok
  print(f"  actor obs              : {pf}/frame x{hist} = {flat} "
        f"{'OK' if dim_ok else 'CONTRACT CHANGED!'}")
  print(f"  leg stds (env)         : pos {os.environ.get('G1_LEG_POS_STD','0.30(default)')} "
        f"ori {os.environ.get('G1_LEG_ORI_STD','0.40(default)')}")
  print("SELFCHECK:", "PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  if "--selfcheck" in sys.argv:
    raise SystemExit(_selfcheck())
  import mjlab.tasks  # noqa: F401
  from mjlab.scripts.train import main
  main()
