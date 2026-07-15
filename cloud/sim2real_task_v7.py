"""Sim2real recipe v7 — attempt 4 (user extended the <=3 budget by one). v6
station-keeping + a targeted fix for the TWO checks v6 still missed, WITHOUT
touching the choreography.

v6 result (exports/train-thriller_v6sk-0714/gap.json), read per-section:
  * drift moved a LOT (v5 4.56 m -> v6 1.67 m) via the new XY drift termination,
    but STILL failed the <=1.0 m gate. It is a TAIL: nominal mean drift is only
    0.15 m — the 1.67 m max is a handful of near-fall episodes, not a global bias.
  * nominal survival 92.2% (need >=99%): 6 of 10 falls sit in 25-36s, the rest
    scattered; the first to collapse under latency stress is 13-18s.
  * ankle p95 17.7/21.8 Nm (need <=15/20): the ankles SATURATE (hit the 50 Nm
    effort cap) at those SAME 13-18s / 25-36s passages.
  => all three failures are ONE root cause: on the 1-2 most dynamic passages
     (verified feasible — fastest joints ~8.4-8.5 rad/s, under the 9.4 limit) the
     policy drives the ankles to the cap to hold station, saturates, ~8% of full
     49 s rollouts tip, and the tippers are also the drifters. Fix the saturation
     on those passages and survival + ankle + the drift tail move together.

v7 deltas on top of v6 (each evidence-backed, not a guess):
  1. ANKLE PENALTY -6e-4 -> -1e-3  AND  action-rate -0.20 -> -0.25.  This is the
     EXACT pair the 2026-07-08 ankle-penalty policy (96da66) used, which measured
     ankle p95 = 10.7 Nm (well under 15) at mpkpe 0.154 m and 100% survival — i.e.
     proven to cut saturation WITHOUT over-smoothing the gestures. v6's -6e-4 alone
     was too timid (17.7 Nm). Arm-fidelity terms stay weight 1.0 to keep the arms crisp.
  2. (launcher) stage-3 drift band 0.5 -> 0.4 m and +5000 iters (12k total) — more
     time on the hardest band with a slightly tighter reset to pull in the drift tail.
  3. (launcher) BEST-checkpoint selection, not blind-last. v6 auto-exported model_9997
     whose mean episode length (388) was a low point in an oscillating late reward; a
     neighbouring checkpoint survives better. train_v7_curriculum.sh screens the last
     ~6 checkpoints with a cheap gap_check and exports the winner. Pure eval, zero
     training risk — the single highest-leverage change for the survival gate.

Everything else = v6/v5 verbatim (XY drift termination, arm terms, station-keeping
reward, latency curriculum, obs 160-dim -> deploy runtime unchanged). The gate stays
on the STOCK task (no drift termination) = the honest drift measurement.

PREFLIGHT: python cloud/sim2real_task_v7.py --selfcheck   (asserts keys + weights)
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

# The 2026-07-08-proven ankle pair (measured ankle p95 10.7 Nm together).
ANKLE_TORQUE_L2_W = float(os.environ.get("G1_ANKLE_TORQUE_L2_W", "-1e-3"))
ACTION_RATE_W = float(os.environ.get("G1_ACTION_RATE_W", "-0.25"))


def _apply_v7(cfg):
  v6._apply_v6(cfg)                                       # all v6 deltas
  cfg.rewards["ankle_torque_l2"].weight = ANKLE_TORQUE_L2_W   # v6 was -6e-4 (too timid)
  cfg.rewards["action_rate_l2"].weight = ACTION_RATE_W        # v6 was -0.20
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
  """Assert attempt 4's wiring on THIS mjlab. Cheap preflight (seconds, not 6 h)."""
  cfg = _make(train=True, play=False)
  ok = True
  for k in ("ankle_torque_l2", "action_rate_l2", "motion_global_root_pos",
            "motion_arm_pos", "motion_arm_ori"):
    present = k in cfg.rewards
    ok &= present
    w = f"w={cfg.rewards[k].weight}" if present else ""
    print(f"  reward   {k:<24} {'OK' if present else 'MISSING'}  {w}")
  present = "anchor_drift_xy" in cfg.terminations
  ok &= present
  print(f"  termin.  {'anchor_drift_xy':<24} {'OK' if present else 'MISSING'}")
  # assert the proven values actually landed (a silent inherit of v6's -6e-4 would
  # quietly re-run the failing recipe).
  ankle_ok = abs(cfg.rewards["ankle_torque_l2"].weight - (-1e-3)) < 1e-9
  rate_ok = abs(cfg.rewards["action_rate_l2"].weight - (-0.25)) < 1e-9
  ok &= ankle_ok and rate_ok
  print(f"  ankle_torque_l2 w : {cfg.rewards['ankle_torque_l2'].weight}  "
        f"{'OK (-1e-3 proven)' if ankle_ok else '!! expected -1e-3'}")
  print(f"  action_rate_l2 w  : {cfg.rewards['action_rate_l2'].weight}  "
        f"{'OK (-0.25 proven)' if rate_ok else '!! expected -0.25'}  (v6 was -0.20)")
  print(f"  drift threshold   : {v6.DRIFT_TERM_M} m")
  print("SELFCHECK", "PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  if "--selfcheck" in sys.argv:
    raise SystemExit(_selfcheck())
  import mjlab.tasks  # noqa: F401
  from mjlab.scripts.train import main
  main()
