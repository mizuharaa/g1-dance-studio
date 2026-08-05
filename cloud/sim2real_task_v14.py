"""v14 task — LEAN SHAPING + KEYPOINT TERMINATION (HoloSoma-informed style fix).

WHY (2026-08-05, experiments/holosoma_eval_20260805.md): amazon-far/holosoma
dances a real G1 29-DoF on the SAME BeyondMimic-lineage recipe with (a) almost
no shaping — no ankle-torque terms, no stance terms, no saturation terms; DR +
training-time pushes do the robustness work — and (b) a full-body KEYPOINT
termination (feet + wrists must stay within 0.25 m of the reference), which
makes leg under-reach fatal instead of tradeable. Our v12 symptoms (sway 3x
too stiff, legs 60-79% reach, perpetual near-tipping look) map onto exactly
the terms holosoma doesn't have and the termination we don't have. Our audit
(2026-07-21) already found much of our ankle shaping inert or inverted.

DELTAS vs v13 (everything else inherited: v13 leg anti-chatter, v11 leg
tracking + softened stance weight base, v10/v8 obs history, effort clamp DR,
waist slack, latency DR, audit fixes):
  1. LEAN (G1_V14_LEAN=1 default): DELETE the shaping rewards
       ankle_torque_barrier   (audit: gradient dead/in-band-arguable; holosoma: absent)
       stance_foot_lin_vel    (drift study: policy already under-sways)
       stance_foot_yaw_rate   (same family)
       stance_foot_flat       (audit H: fought leg-ori tracking even as residual)
       torque_saturation_dur  (audit I: inert on legs at realistic thresholds)
     KEPT deliberately: ankle_action_rate_l2 + leg_action_rate_l2 (chatter is
     MEASURED on our hardware baseline), motion_leg_pos/ori (reach is measured),
     the ~40 Nm velocity-derated ankle effort CLAMP + scoped effort DR (that is
     actuator truth, not shaping), waist slack, latency DR 0-80 ms (our
     hardware measures 40-80 ms; holosoma's delay DR is weaker — we keep ours).
  2. KEYPOINT TERMINATION `bad_keypoint_pos`: episode ends when any of the 4
     keypoints (left/right ankle_roll_link, left/right wrist_yaw_link) deviates
     more than G1_KEYPOINT_TERM_M (default 0.25 m, holosoma's number) from the
     reference, measured ROOT-RELATIVE so it disciplines reach without double-
     charging global drift (anchor_drift_xy already owns that). Grace period
     G1_KEYPOINT_TERM_GRACE_S (default 1.0 s) covers the activation ramp.
Deploy contract unchanged (terminations + reward deletions only; selfcheck
asserts the 154-dim/frame actor obs is untouched).

A/B guidance: v14 vs v12-final on the same v12_bundle motion isolates
{lean+termination}. To isolate further, G1_V14_LEAN=0 keeps the shaping and
tests the termination alone.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import sim2real_task as base            # noqa: E402
import sim2real_task_v13 as v13         # noqa: E402  full v13 stack (audit-fixed)

from mjlab.managers.termination_manager import TerminationTermCfg  # noqa: E402
from mjlab.tasks.registry import register_mjlab_task  # noqa: E402
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg  # noqa: E402
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner  # noqa: E402

TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1-S2R-V14"

V14_LEAN = os.environ.get("G1_V14_LEAN", "1") == "1"
LEAN_DELETE_REWARDS = (
  "ankle_torque_barrier",
  "stance_foot_lin_vel",
  "stance_foot_yaw_rate",
  "stance_foot_flat",
  "torque_saturation_dur",
)
# Terms that must SURVIVE the lean pass (regression guard in selfcheck).
LEAN_KEEP_REWARDS = (
  "ankle_action_rate_l2", "leg_action_rate_l2",
  "motion_leg_pos", "motion_leg_ori",
  "motion_arm_pos", "motion_arm_ori",
  "motion_body_pos", "motion_body_ori",
)

KEYPOINT_BODY_NAMES = (
  "left_ankle_roll_link", "right_ankle_roll_link",
  "left_wrist_yaw_link", "right_wrist_yaw_link",
)
KEYPOINT_TERM_M = float(os.environ.get("G1_KEYPOINT_TERM_M", "0.25"))
KEYPOINT_GRACE_S = float(os.environ.get("G1_KEYPOINT_TERM_GRACE_S", "1.0"))

# Candidate (reference, robot) body-position attribute pairs on the mjlab
# MotionCommand term, preferred order: root/anchor-RELATIVE first (disciplines
# reach without double-charging drift), world-frame as fallback (still correct,
# just overlaps anchor_drift_xy's job). Resolved once at first call; the box
# selfcheck prints which pair resolved so a rename costs seconds, not a run.
_KEYPOINT_ATTR_PAIRS = (
  ("body_pos_relative_w", "robot_body_pos_relative_w"),
  ("body_pos_b", "robot_body_pos_b"),
  ("body_pos_w", "robot_body_pos_w"),
)
_keypoint_cache: dict = {}


def _resolve_keypoints(cmd):
  """Map KEYPOINT_BODY_NAMES to column indices of the command's tracked-body
  tensors and pick the first attribute pair the command actually has."""
  key = id(cmd)
  hit = _keypoint_cache.get(key)
  if hit is not None:
    return hit
  names = list(getattr(cmd.cfg, "body_names", ()) or ())
  if not names:
    raise RuntimeError("bad_keypoint_pos: motion command cfg has no body_names")
  missing = [n for n in KEYPOINT_BODY_NAMES if n not in names]
  if missing:
    raise RuntimeError(f"bad_keypoint_pos: keypoints not tracked by command: {missing}")
  idx = [names.index(n) for n in KEYPOINT_BODY_NAMES]
  for ref_attr, rob_attr in _KEYPOINT_ATTR_PAIRS:
    if hasattr(cmd, ref_attr) and hasattr(cmd, rob_attr):
      hit = (idx, ref_attr, rob_attr)
      _keypoint_cache[key] = hit
      return hit
  raise RuntimeError(
    "bad_keypoint_pos: none of the candidate body-position attribute pairs "
    f"{_KEYPOINT_ATTR_PAIRS} exist on the motion command "
    f"(has: {[a for a in dir(cmd) if 'body_pos' in a]})")


def _bad_keypoint_pos(env, command_name: str, threshold: float, grace_steps: int):
  """Terminate when any keypoint (feet/wrists) strays > threshold metres from
  its reference position. holosoma-style full-body tracking discipline: leg
  under-reach becomes fatal instead of a reward trade."""
  cmd = env.command_manager.get_term(command_name)
  idx, ref_attr, rob_attr = _resolve_keypoints(cmd)
  ref = getattr(cmd, ref_attr)[:, idx, :]
  rob = getattr(cmd, rob_attr)[:, idx, :]
  err = torch.norm(ref - rob, dim=-1)          # (num_envs, 4)
  bad = torch.any(err > threshold, dim=1)
  if grace_steps > 0:
    bad &= env.episode_length_buf > grace_steps
  return bad


def _apply_v14(cfg, train: bool):
  v13._apply_v13(cfg, train=train)

  if V14_LEAN:
    for k in LEAN_DELETE_REWARDS:
      cfg.rewards.pop(k, None)

  step_dt = getattr(cfg.sim, "dt", 0.005) * getattr(cfg, "decimation", 4)
  cfg.terminations["bad_keypoint_pos"] = TerminationTermCfg(
    func=_bad_keypoint_pos,
    params={"command_name": "motion",
            "threshold": KEYPOINT_TERM_M,
            "grace_steps": int(round(KEYPOINT_GRACE_S / step_dt))},
  )
  return cfg


def _make(train: bool, play: bool):
  return _apply_v14(base._make(train=train, play=play), train=train)


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

  if V14_LEAN:
    for k in LEAN_DELETE_REWARDS:
      gone = k not in cfg.rewards
      ok &= gone
      print(f"  lean: {k:22s}: {'DELETED (ok)' if gone else 'STILL PRESENT!'}")
  else:
    print("  lean: DISABLED (G1_V14_LEAN=0) — termination-only A/B arm")
  for k in LEAN_KEEP_REWARDS:
    present = k in cfg.rewards
    ok &= present
    print(f"  keep: {k:22s}: {'present' if present else 'MISSING!'}")

  t = cfg.terminations.get("bad_keypoint_pos")
  t_ok = t is not None and t.params["threshold"] == KEYPOINT_TERM_M
  ok &= t_ok
  print(f"  bad_keypoint_pos       : {'OK' if t_ok else 'MISSING/WRONG'} "
        f"thr={KEYPOINT_TERM_M} m grace={KEYPOINT_GRACE_S}s "
        f"bodies={KEYPOINT_BODY_NAMES}")
  drift_ok = "anchor_drift_xy" in cfg.terminations
  ok &= drift_ok
  print(f"  anchor_drift_xy        : {'present' if drift_ok else 'MISSING!'}")

  # keypoint plumbing resolves against the real command term? Only checkable
  # with a live env; here we at least confirm the tracked-body list covers us.
  cmd_cfg = cfg.commands.get("motion") if hasattr(cfg, "commands") else None
  names = list(getattr(cmd_cfg, "body_names", ()) or ()) if cmd_cfg else []
  if names:
    missing = [n for n in KEYPOINT_BODY_NAMES if n not in names]
    cov_ok = not missing
    ok &= cov_ok
    print(f"  keypoints in command   : {'OK' if cov_ok else f'MISSING {missing}'}")
  else:
    print("  keypoints in command   : (command cfg not inspectable here — box env check)")

  import sim2real_task_v10 as v10
  pf, flat, hist, unknown = v10._actor_obs_dim(cfg)
  want = 154 + (2 if v10.G1_PHASE_OBS else 0)
  dim_ok = pf == want and flat == want * v10.v8.G1_OBS_HISTORY and not unknown
  ok &= dim_ok
  print(f"  actor obs              : {pf}/frame x{hist} = {flat} "
        f"{'OK' if dim_ok else 'CONTRACT CHANGED!'}")
  print("SELFCHECK:", "PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  if "--selfcheck" in sys.argv:
    raise SystemExit(_selfcheck())
  import mjlab.tasks  # noqa: F401
  from mjlab.scripts.train import main
  main()
