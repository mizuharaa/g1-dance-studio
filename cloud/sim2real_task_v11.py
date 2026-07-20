"""v11 task — FIX THE LEG UNDER-REACH. Native tempo, same as v10, plus a
dedicated LEG-tracking reward and de-suppressed lower body.

WHY (measured 2026-07-20, from the v10 rollout in tools.sim_sandbox):
  In the 10-25 s arm-swing section the v10 policy achieved only 43-59% of the
  reference KNEE/HIP amplitude while the ARMS hit 77-137%. Root cause is a
  STRUCTURAL reward asymmetry, not a capability limit:
    * ARMS  are rewarded TWICE — the whole-body motion_body_pos/ori term AND a
      dedicated motion_arm_pos/ori term (v5, weight 1.0, tight std 0.25/0.35 on
      6 arm bodies).
    * LEGS  are rewarded ONCE (only the whole-body term) and PENALISED five ways
      (ankle_torque_barrier, ankle_action_rate_l2, stance_foot_lin_vel -0.5,
      stance_foot_yaw_rate -0.1, stance_foot_flat -0.5). The lower body is taxed
      for balance and never given a fidelity incentive, so the policy trades the
      stepping away first. The user saw exactly this ("~0 leg movement during the
      arm swings").

FIX (this file, all training-only — ZERO deploy-contract change):
  1. ADD motion_leg_pos/ori — a dedicated leg-fidelity term mirroring the arm
     term, on the 6 leg bodies already in the tracking command body_names
     (hip_roll / knee / ankle_roll links, both sides). std 0.30/0.40 — slightly
     looser than the arms because the legs also carry weight, so the term rewards
     COMMITTING to the stepping without demanding sub-cm foot placement.
  2. SOFTEN the stance-foot linear-velocity penalty -0.5 -> -0.20 (env default
     here). It exists to kill SLIP, but at -0.5 it also suppressed legitimate
     stepping when the policy's phase lagged the reference; slip is still
     penalised, just not hard enough to freeze the feet. yaw/flat unchanged.
  3. The launcher (run_attempt8 / train_v11) LOOSENS the drift termination in the
     late stages (G1_DRIFT_TERM_M 0.8/0.6/0.6/0.8 vs v10's 0.8/0.6/0.5/0.4): the
     actor is position-BLIND by design (no base_lin_vel / anchor_pos on the real
     robot), so a hard 0.4 m band on a 2 m stage over-constrains it into freezing
     the legs and getting brittle, while buying little real-world benefit — the
     calibration history shows deployable-quality policies drift metres in this
     same sim yet perform acceptably tethered (drift is partly a sim/friction
     artifact; stance-foot SLIP, penalised above, is the position-free proxy that
     actually transfers). Drift is still MEASURED (999) at the gate.

Everything else = v10 verbatim (native-tempo speed curriculum, obs history,
ankle barrier, saturation penalty, phase-obs hook, calibrated gate bars 22/25).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sim2real_task as base           # noqa: E402
import sim2real_task_v10 as v10        # noqa: E402  the full v10 stack

from mjlab.managers.reward_manager import RewardTermCfg  # noqa: E402
from mjlab.tasks.registry import register_mjlab_task  # noqa: E402
from mjlab.tasks.tracking import mdp  # noqa: E402
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg  # noqa: E402
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner  # noqa: E402

TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1-S2R-V11"

# The 6 leg bodies already in motion_cmd.body_names (g1 env cfg) — no new sites,
# mirrors ARM_BODY_NAMES.
LEG_BODY_NAMES = (
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
)
LEG_POS_STD = float(os.environ.get("G1_LEG_POS_STD", "0.30"))
LEG_ORI_STD = float(os.environ.get("G1_LEG_ORI_STD", "0.40"))
LEG_POS_W = float(os.environ.get("G1_LEG_POS_W", "1.0"))
LEG_ORI_W = float(os.environ.get("G1_LEG_ORI_W", "1.0"))

# v11 softens the stance linvel penalty (still a slip guard, no longer a step-
# freezer — the under-sway finding shows the lower body was over-suppressed).
# Applied by OVERRIDING the weight on the built cfg in _apply_v11, NOT via env:
# v10 captures G1_STANCE_LINVEL_W at its own import (before this module runs), so
# a late setdefault here would silently no-op (caught by the box selfcheck).
STANCE_LINVEL_SOFT = float(os.environ.get("G1_STANCE_LINVEL_SOFT", "-0.20"))


def _apply_v11(cfg, train: bool):
  # full v10 stack (which applies v8: obs history, ankle barrier + clamp + DR,
  # waist slack, drift termination, latency DR; then v10 stance + saturation).
  v10._apply_v10(cfg, train=train)

  # soften the stance-foot linear-velocity penalty on the BUILT cfg (robust to
  # v10's import-time capture of the env var — see STANCE_LINVEL_SOFT note).
  if "stance_foot_lin_vel" in cfg.rewards:
    cfg.rewards["stance_foot_lin_vel"].weight = STANCE_LINVEL_SOFT

  # dedicated leg-fidelity tracking, mirroring v5's arm terms. Mean over the 6
  # leg bodies; sits ON TOP of the whole-body motion_body_pos/ori so the lower
  # body finally has the same fidelity incentive the arms have had since v5.
  cfg.rewards["motion_leg_pos"] = RewardTermCfg(
    func=mdp.motion_relative_body_position_error_exp,
    weight=LEG_POS_W,
    params={"command_name": "motion", "std": LEG_POS_STD,
            "body_names": LEG_BODY_NAMES},
  )
  cfg.rewards["motion_leg_ori"] = RewardTermCfg(
    func=mdp.motion_relative_body_orientation_error_exp,
    weight=LEG_ORI_W,
    params={"command_name": "motion", "std": LEG_ORI_STD,
            "body_names": LEG_BODY_NAMES},
  )
  return cfg


def _make(train: bool, play: bool):
  return _apply_v11(base._make(train=train, play=play), train=train)


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=_make(train=True, play=False),
  play_env_cfg=_make(train=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)


def _selfcheck() -> int:
  """Assert the two new leg terms registered on real leg bodies, the stance
  linvel penalty is the softened value, and the actor obs dim is UNCHANGED vs
  v10 (rewards must never touch the deploy contract)."""
  cfg = _make(train=True, play=False)
  ok = True

  for k in ("motion_leg_pos", "motion_leg_ori"):
    present = k in cfg.rewards
    bodies = cfg.rewards[k].params.get("body_names") if present else None
    good = present and bodies == LEG_BODY_NAMES
    ok &= good
    print(f"  {k:22s}: {'OK' if good else 'MISSING/WRONG'}  bodies={bodies}")

  # arm terms must still be there (we ADD legs, not replace arms)
  for k in ("motion_arm_pos", "motion_arm_ori", "motion_body_pos"):
    present = k in cfg.rewards
    ok &= present
    print(f"  {k:22s}: {'present' if present else 'MISSING'}")

  lv = cfg.rewards["stance_foot_lin_vel"].weight if "stance_foot_lin_vel" in cfg.rewards else None
  lv_ok = lv is not None and abs(lv - STANCE_LINVEL_SOFT) < 1e-9
  ok &= lv_ok
  print(f"  stance_foot_lin_vel W : {lv} (want {STANCE_LINVEL_SOFT})  "
        f"{'OK (softened)' if lv_ok else 'UNEXPECTED'}")

  # deploy contract unchanged vs v10
  pf, flat, hist, unknown = v10._actor_obs_dim(cfg)
  want_pf = 154 + (2 if v10.G1_PHASE_OBS else 0)
  dim_ok = (pf == want_pf) and (flat == want_pf * v10.v8.G1_OBS_HISTORY) and not unknown
  ok &= dim_ok
  print(f"  actor obs per-frame   : {pf} (want {want_pf}), flat {flat}, "
        f"hist {hist}, unknown {unknown}  {'OK' if dim_ok else 'CONTRACT CHANGED!'}")

  print("SELFCHECK:", "PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  if "--selfcheck" in sys.argv:
    raise SystemExit(_selfcheck())
  print(f"{TASK_ID}: v10 + dedicated leg tracking on {len(LEG_BODY_NAMES)} bodies "
        f"(pos std {LEG_POS_STD}, ori std {LEG_ORI_STD}), stance linvel softened to "
        f"{os.environ['G1_STANCE_LINVEL_W']}")
