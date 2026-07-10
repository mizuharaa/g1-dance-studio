"""Sim2real recipe v5 "fidelity" — Lane E retrain (tasks/AGENT_E_FIDELITY_RETRAIN.md).

Fixes BOTH 2026-07-10 findings in one recipe:
  * the tester's 60-70% gap (subtle arm moves washed out), and
  * the lat80 retrain failure (0-80 ms delay DR from step 0 -> drift 2-7 m,
    survival 0.000; data/telemetry/latency_retrain_20260710/RESULT.md).

Base = sim2real_task.py recipe v2 (the config that produced the policy which ran
44 s on hardware — proven except for latency). v5 deltas:

  1. SUBTLE-MOVE FIDELITY: two NEW tracking terms scoped to the 6 arm bodies the
     motion command already tracks (shoulder_roll/elbow/wrist_yaw links, both
     sides): motion_arm_pos (w 1.0, std 0.25) + motion_arm_ori (w 1.0, std 0.35).
     Arms get ~2x reward pressure at tighter tolerance than the whole-body terms
     (std 0.3/0.4), so small gestures stop being traded away against balance.
  2. STATION-KEEPING: motion_global_root_pos weight 0.5 -> 1.0 (the part of the
     lat80 attempt that was RIGHT — kept).
  3. LATENCY = CURRICULUM VIA STAGED RESUME, not a blunt band from step 0.
     Delay caps are env-var-configurable and read at import:
       G1_CMD_DELAY_MAX_LAG  physics steps (5 ms each), default 4  (0-20 ms)
       G1_OBS_DELAY_MAX_LAG  control steps (20 ms each), default 1 (0-20 ms)
     cloud/train_v5_curriculum.sh drives the stages: 0-20 ms (learn the dance +
     station-keeping first) -> 0-50 ms -> 0-60 ms, resuming the checkpoint each
     time. 60 ms cap, not 80: sim PD already models mechanical lag, so 80 ms
     double-counted it (DIAGNOSIS.md) — that over-randomization is what traded
     away station-keeping. ~10k iters total (5k was too few for the harder task).

Obs stays 160-dim -> deploy runtime unchanged.

Gate BEFORE hardware (both must pass):
  * cloud/sim_gap_check.py — survival at 40 ms + push (hard gate), 60/80 ms lines.
  * nominal root drift < 1 m (the lat80 failure mode).
Then Lane-D sandbox fidelity report, THEN hardware.

Launch: see cloud/train_v5_curriculum.sh (stages, resume flags, verify chain).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sim2real_task as base  # recipe v2 builder; also registers the base task

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1-S2R-V5"

# Curriculum knobs (stage script sets these; defaults = stage-1 hygiene band).
# base._apply_sim2real reads the module globals at call time, so overriding them
# BEFORE building the cfg is sufficient — no mjlab internals touched.
base.CMD_DELAY_MAX_LAG = int(os.environ.get("G1_CMD_DELAY_MAX_LAG", "4"))
base.OBS_DELAY_MAX_LAG = int(os.environ.get("G1_OBS_DELAY_MAX_LAG", "1"))

# The 6 arm bodies already in motion_cmd.body_names (g1 env cfg) — no new sites.
ARM_BODY_NAMES = (
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)


def _apply_v5(cfg):
  # 2. station-keeping pressure (kept from the lat80 attempt — that part worked:
  #    rr_mpkpe was 0.079; only the delay schedule was wrong).
  cfg.rewards["motion_global_root_pos"].weight = 1.0

  # 1. subtle-move fidelity: arm-scoped tracking at tighter std, on top of the
  #    whole-body terms (mean over the 6 bodies; funcs accept body_names).
  cfg.rewards["motion_arm_pos"] = RewardTermCfg(
    func=mdp.motion_relative_body_position_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.25, "body_names": ARM_BODY_NAMES},
  )
  cfg.rewards["motion_arm_ori"] = RewardTermCfg(
    func=mdp.motion_relative_body_orientation_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.35, "body_names": ARM_BODY_NAMES},
  )
  return cfg


def _make(train: bool, play: bool):
  return _apply_v5(base._make(train=train, play=play))


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=_make(train=True, play=False),
  play_env_cfg=_make(train=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

if __name__ == "__main__":
  print(f"{TASK_ID}: cmd delay 0-{base.CMD_DELAY_MAX_LAG * 5} ms, "
        f"obs delay 0-{base.OBS_DELAY_MAX_LAG * 20} ms, "
        f"arm terms on {len(ARM_BODY_NAMES)} bodies")
