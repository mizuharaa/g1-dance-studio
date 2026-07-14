"""Sim2real recipe v6 "station-keeping" — attempt 3 retrain (of a <=3 budget).

WHY v6 (root-cause of the v5 borderline result, from exports/.../gap.json):
  v5 finished the full 10k curriculum but FAILED the gate on:
    * drift_max 4.56 m   (gate <= 1.0 m)      <- the headline failure
    * nominal survival 92.2%  (gate >= 99%)   <- the falls ARE the drift tail
    * ankle_pitch p95 16.4/21.5 Nm (gate <= 15/20) — marginal saturation spikes
  It tracked the dance tightly (rr_mpkpe 0.08 m) and was latency/push robust
  (99.2% under 40 ms + push). So the dance and the latency hardening WORK; the
  one unsolved failure mode is horizontal ROOT DRIFT — the robot does the right
  poses but slides across the floor (world mpkpe 0.50 m vs root-rel 0.08 m).

ROOT CAUSE (code, not a guess): the stock tracking env terminates on VERTICAL
  anchor error only — `anchor_pos` uses `bad_anchor_pos_z_only` (threshold 0.25).
  There is NO horizontal (XY) drift termination wired in, so an episode keeps
  running while the robot moonwalks metres sideways; the only thing resisting it
  is the soft `motion_global_root_pos` reward. v5 tried to fix drift by bumping
  that reward 0.5->1.0 — necessary but NOT sufficient: a soft reward can't pin a
  free-sliding DoF. The fix is a TERMINATION.

v6 = v5 (kept verbatim) PLUS:
  1. XY ANCHOR-DRIFT TERMINATION — new `anchor_drift_xy` term, full horizontal
     norm of (motion anchor - robot anchor), threshold curriculum'd via env
     G1_DRIFT_TERM_M (0.8 m stage1 -> 0.6 -> 0.5). Half the 1.0 m gate at the end,
     so any episode approaching the failure resets instead of accumulating drift
     (which also cleans the PPO experience — the falls stop being fed back).
  2. ANKLE SATURATION — ankle_torque_l2 weight -4e-4 -> -6e-4 to shave the p95
     spikes that hit the 50 Nm effort limit, without over-penalising balance.

Everything else is v5 verbatim (arm-fidelity terms, station-keeping reward,
latency curriculum, obs 160-dim -> deploy runtime unchanged).

PREFLIGHT (cheap, run BEFORE the 5 h train — see cloud/run_attempt3.sh):
    python cloud/sim2real_task_v6.py --selfcheck
  imports + builds the cfg and asserts every reward/termination key this recipe
  depends on actually registered on THIS mjlab. If an mjlab API name differs it
  fails in seconds, not after a wasted GPU run.

Launch: cloud/train_v6_curriculum.sh (stages, drift curriculum, verify chain).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import sim2real_task as base  # recipe v2 builder; registers the base task
import sim2real_task_v5 as v5  # arm-fidelity + station-keeping deltas (reused)

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1-S2R-V6"

# XY drift-termination radius (m). The curriculum script tightens this per stage;
# 0.5 default = the final, strictest band (half the 1.0 m gate).
DRIFT_TERM_M = float(os.environ.get("G1_DRIFT_TERM_M", "0.5"))


def _bad_anchor_pos_xy(env, command_name: str, threshold: float) -> torch.Tensor:
  """Terminate when the robot's anchor drifts past `threshold` metres in the
  HORIZONTAL plane from the reference motion's anchor. mjlab ships only the
  z-only variant; this is the XY-norm version — the direct fix for v5's 4.56 m
  slide. Mirrors the stock `bad_anchor_pos_z_only` shape but on components x,y."""
  command = env.command_manager.get_term(command_name)
  err_xy = command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2]
  return torch.norm(err_xy, dim=1) > threshold


def _drift_term_func():
  """Prefer mjlab's own full-norm termination if it ships one (it derives from a
  codebase that defines `bad_anchor_pos`); else fall back to the XY function
  above. Either way --selfcheck confirms the term registered before we train."""
  return getattr(mdp, "bad_anchor_pos", _bad_anchor_pos_xy)


def _termination_cfg_cls():
  """TerminationTermCfg under whichever module name this mjlab exposes."""
  from mjlab.managers.termination_manager import TerminationTermCfg
  return TerminationTermCfg


def _apply_v6(cfg):
  # keep every v5 delta verbatim (arm fidelity terms + station-keeping reward +
  # the env-var latency caps read at import by the base builder).
  v5._apply_v5(cfg)

  # 1. THE fix: bound horizontal drift with a hard termination (curriculum'd).
  TerminationTermCfg = _termination_cfg_cls()
  cfg.terminations["anchor_drift_xy"] = TerminationTermCfg(
    func=_drift_term_func(),
    params={"command_name": "motion", "threshold": DRIFT_TERM_M},
  )

  # 2. shave ankle p95 saturation (16.4/21.5 -> aim <=15/20 Nm).
  cfg.rewards["ankle_torque_l2"].weight = -6e-4
  return cfg


def _make(train: bool, play: bool):
  return _apply_v6(base._make(train=train, play=play))


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=_make(train=True, play=False),
  play_env_cfg=_make(train=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)


def _selfcheck() -> int:
  """Assert the recipe wired what attempt 3 depends on. Cheap preflight — run on
  the box before the real train so an mjlab API mismatch costs seconds, not 5 h."""
  cfg = _make(train=True, play=False)
  need_rewards = ("motion_global_root_pos", "ankle_torque_l2",
                  "motion_arm_pos", "motion_arm_ori")
  need_terms = ("anchor_drift_xy",)
  ok = True
  for k in need_rewards:
    present = k in cfg.rewards
    ok &= present
    print(f"  reward   {k:<26} {'OK' if present else 'MISSING'}")
  for k in need_terms:
    present = k in cfg.terminations
    ok &= present
    print(f"  termin.  {k:<26} {'OK' if present else 'MISSING'}")
  drift_fn = _drift_term_func()
  print(f"  drift termination fn : {getattr(drift_fn, '__name__', drift_fn)}")
  print(f"  drift threshold (m)  : {DRIFT_TERM_M}")
  print(f"  ankle_torque_l2 w    : {cfg.rewards['ankle_torque_l2'].weight}")
  print(f"  cmd/obs delay caps   : {base.CMD_DELAY_MAX_LAG*5} ms / "
        f"{base.OBS_DELAY_MAX_LAG*20} ms")
  print("SELFCHECK", "PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  if "--selfcheck" in sys.argv:
    raise SystemExit(_selfcheck())
  # otherwise behave like the train entry (registry already populated above)
  import mjlab.tasks  # noqa: F401
  from mjlab.scripts.train import main
  main()
