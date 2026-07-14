"""Sim2real recipe v7 — attempt 4. v6 (station-keeping) + targeted fixes for the
TWO checks v6 still missed, WITHOUT altering the choreography:

  v6 result (gap.json): drift SOLVED (nominal mean 0.15 m, clean-rollout 0.02 m over
  the full dance; was 4.56 m in v5), latency/push robust, held-out survival 100%.
  Still failed: nominal survival 92.2% (need >=99%) and ankle p95 17.7/21.8 Nm
  (need <=15/20). Those two are coupled — on the hardest segments the policy drives
  the ankles to the 50 Nm effort cap to hold position, saturates, and ~8% of episodes
  tip over.

v7 deltas on top of v6:
  1. ANKLE SMOOTHNESS: raise the global action-rate penalty (-0.20 -> -0.28). v6's
     ankle_torque_l2 bump alone didn't clear p95; penalising the RATE of action change
     smooths the target stream, cutting the torque spikes that drive p95 AND the
     saturation that precedes the falls. Arm-fidelity terms stay weight 1.0 so the
     modest action-rate bump doesn't dull the gestures.
  2. MORE ITERS FOR THE TAIL: the curriculum extends stage 3 (0-60 ms, drift<0.5 m)
     from +3000 to +5000 (10k -> 12k total) — more time on the hardest band to firm
     up the survival tail the extra reward pressure exposes.

Everything else = v6 verbatim (drift-XY termination @0.5 m, arm terms, station-keeping
reward, latency curriculum, obs 160-dim -> deploy runtime unchanged).

PREFLIGHT: python cloud/sim2real_task_v7.py --selfcheck   (asserts the keys before the run)
Launch:    cloud/train_v7_curriculum.sh  (via cloud/run_attempt4.sh)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sim2real_task as base       # recipe v2 builder; registers the base task
import sim2real_task_v6 as v6      # drift termination + ankle L2 + arm + station-keeping

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1-S2R-V7"

# global action-rate penalty (smoother targets -> lower ankle p95 + less saturation)
ACTION_RATE_W = float(os.environ.get("G1_ACTION_RATE_W", "-0.28"))


def _apply_v7(cfg):
  v6._apply_v6(cfg)                                    # all v6 deltas
  cfg.rewards["action_rate_l2"].weight = ACTION_RATE_W  # v7: was -0.20
  return cfg


def _make(train: bool, play: bool):
  return _apply_v7(base._make(train=train, play=play))


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
  for k in ("action_rate_l2", "ankle_torque_l2", "motion_global_root_pos",
            "motion_arm_pos", "motion_arm_ori"):
    present = k in cfg.rewards
    ok &= present
    w = f"w={cfg.rewards[k].weight}" if present else ""
    print(f"  reward   {k:<24} {'OK' if present else 'MISSING'}  {w}")
  present = "anchor_drift_xy" in cfg.terminations
  ok &= present
  print(f"  termin.  {'anchor_drift_xy':<24} {'OK' if present else 'MISSING'}")
  print(f"  action_rate_l2 w : {cfg.rewards['action_rate_l2'].weight}  (v6 was -0.20)")
  print(f"  drift threshold  : {v6.DRIFT_TERM_M} m")
  print("SELFCHECK", "PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  if "--selfcheck" in sys.argv:
    raise SystemExit(_selfcheck())
  import mjlab.tasks  # noqa: F401
  from mjlab.scripts.train import main
  main()
