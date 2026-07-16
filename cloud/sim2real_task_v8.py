"""Sim2real recipe v8 — attempt 5. The v-chain revamp run.

This layers TWO committed design memos onto the proven v7 recipe, WITHOUT touching
the choreography-independent v7 machinery that already works (drift termination was
SOLVED in v6/v7 — it is kept verbatim here).

CITED MEMOS (read these before editing this file):
  * experiments/upstream_alignment_report.md  (Agent 0, upstream-alignment audit)
      -> the OBSERVATION-CONTRACT change: our actor carried two PRIVILEGED terms that
         upstream's `Unitree-G1-Tracking-No-State-Estimation` task keeps CRITIC-ONLY:
             base_lin_vel (3)         — needs a base-velocity estimate, unmeasurable
             motion_anchor_pos_b (3)  — needs a base-position estimate, unmeasurable
         Asymmetric actor-critic: the critic still sees them from clean sim truth during
         training; the deployed actor never observes them, so there is nothing to fake
         on the robot (kills the leg-odometry sim/real divergence on the actor's input).
  * experiments/actuation_design_v8.md  (Agent D, actuation/control) — CANDIDATE A:
      the feasible-reference + hip-strategy-shaping bet. Train on the 1.8x repaired
      reference and add the actuation deltas that induce the policy to move balance
      load off the ankles onto the hips/torso.

============================ DELTAS vs v7 (each one) ============================

OBS (Agent 0):
  0. DROP base_lin_vel + motion_anchor_pos_b from the ACTOR observation group; leave
     them in the CRITIC group (the base mjlab tracking task exposes SEPARATE "actor"
     and "critic" ObservationGroupCfgs — verified in
     third_party/mjlab_mdp_ref/tracking_env_cfg.py, and the g1 env cfg even ships a
     `has_state_estimation=False` switch that drops exactly these two). This is a CLEAN
     asymmetric split on the engine we already run — no base-task edit required.
     ACTOR OBS DIM: 160 - 6 = 154  (NOT 160, NOT ~155 — see note below).

ACTUATION (Agent D, CANDIDATE A §2 shared-prereq + reward deltas):
  1. Train on the 1.8x REPAIRED + GROUNDED reference. G1_SLOWDOWN (default 1.8;
     fallbacks 2.0 / 2.5) selects the motion; the launcher regenerates the npz
     (ground -> repair 1.8x -> csv_to_npz). This recipe file only READS G1_SLOWDOWN to
     scale the waist-slack windows to the slowed clock and to print/verify the factor.
  2. VELOCITY-HONEST ANKLE EFFORT CLAMP (Agent 0's T-N-curve finding): lower the ankle
     effort_limit_sim from the optimistic flat 50 Nm to 40 Nm AND widen the effort DR
     downward (0.80-1.00 -> 0.65-0.95) on the 4 ANKLE channels ONLY. Non-ankle joints
     keep the stock 0.80-1.00 DR. Net trained ankle authority ~ 50*0.80*U(0.65,0.95)
     = 26-38 Nm, so the policy never learns to rely on ankle torque the hardware can't
     deliver at speed. Train-only; deploy already clamps at the true motor limit.
  3. REPLACE the global `ankle_torque_l2` (-1e-3) with a SOFT-BARRIER
     `ankle_torque_barrier` = sum_ankles relu(|tau| - 35)^2, weight -5e-3. ~0 below
     35 Nm, bites only near saturation, where hip strategy should take over (does not
     over-smooth the gentle passages the global L2 penalised).
  4. PER-CHANNEL ankle action-rate: `ankle_action_rate_l2` = L2 on the first difference
     of the 4 ankle action channels, weight -0.05 (on top of the global action_rate
     -0.25). Damps the oscillatory ankle saturation buzz without a kd change.
  5. WAIST-TRACKING SLACK: multiply the torso_link ("waist") component of the
     motion_body_pos/ori tracking rewards by 0.5 inside 13-18 s and 25-36 s (windows
     SCALED by G1_SLOWDOWN to the slowed clock); every other body and all other times
     stay 1.0. Frees the trunk flywheel to inject the counter-rotation (hip strategy)
     that unloads the ankle. Arms/shoulders (v5 motion_arm_* terms) stay 1.0 — gesture
     fidelity is spent last.

KEPT VERBATIM from v7/v6/v5:
  * anchor_drift_xy XY drift termination + its curriculum (drift was SOLVED — no regress)
  * motion_arm_pos/ori arm-fidelity terms (weight 1.0), station-keeping reward
  * action_rate_l2 -0.25 global, latency curriculum, mass/CoM/calibration DR
  * kp/kd UNTOUCHED (Agent 0: they match upstream and ARE the deploy gains; raising kp
    raises peak ankle torque = the v7 failure).

--------------------------- NOTE: 154 vs "~155" ---------------------------
Agent 0's audit reports upstream's No-State-Estimation actor as 155-dim. That 155 = our
154 + 1, because UPSTREAM's motion command carries one extra element (a phase/time
scalar): 160 - 6 (dropped) + 1 (command 58->59) = 155. OUR MotionCommand emits a 58-dim
command, and neither memo authorises changing the command manager, so we do NOT fabricate
that +1. The correct target for OUR contract is 154, which is exactly what
pipeline/deploy_runtime.py already expects for the estimator-free ("ground") layout and
what the existing No-State-Estimation ground task exports. --selfcheck asserts 154.

------------------- DEPLOY-WAVE TODO (do NOT do it here) -------------------
Once a v8 policy trains on the 154-dim actor and is signed, the deploy wave should DELETE
the now-dead honest-odometry path (Agent 0 P0 item): pipeline/leg_odometry.py and
deploy_runtime.build_obs_odom() (+ the base_lin_vel/motion_anchor_pos_b entries in
ESTIMATOR_DEPENDENT_TERMS). The actor no longer consumes base_lin_vel, so faking it is
obsolete. This file does NOT touch deploy code — flagged for the deploy wave only.

PREFLIGHT:  python cloud/sim2real_task_v8.py --selfcheck   (asserts keys + 154-dim actor)
Launch:     cloud/train_v8_curriculum.sh   (via cloud/run_attempt5.sh)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import sim2real_task as base       # recipe v2 builder; registers the base task
import sim2real_task_v7 as v7      # ankle pair + drift termination + arm/station-keeping

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.utils.lab_api.math import quat_error_magnitude

TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1-S2R-V8"

# ---- tunables (env-overridable so the launcher can walk them without a rewrite) ----
# Slowdown factor for the repaired reference. 1.8 = the design target (Agent D §1.2);
# 2.0 / 2.5 are the documented fallbacks if the policy under-uses hips on the box.
G1_SLOWDOWN = float(os.environ.get("G1_SLOWDOWN", "1.8"))

# Velocity-honest ankle effort clamp (Agent D §2.2). Stock sim ankle effort_limit_sim
# is a flat 50 Nm; we clamp the trained envelope to 40 and widen the ankle effort DR
# downward. Expressed as multiplicative startup DR (the SAME proven dr.effort_limits
# call the base uses) so no fragile actuator-cfg mutation is needed.
STOCK_ANKLE_EFFORT_NM = float(os.environ.get("G1_STOCK_ANKLE_EFFORT_NM", "50.0"))
ANKLE_EFFORT_LIMIT_NM = float(os.environ.get("G1_ANKLE_EFFORT_NM", "40.0"))
ANKLE_CLAMP_SCALE = ANKLE_EFFORT_LIMIT_NM / STOCK_ANKLE_EFFORT_NM          # 40/50 = 0.80
ANKLE_EFFORT_DR = (
  float(os.environ.get("G1_ANKLE_EFFORT_DR_LO", "0.65")),
  float(os.environ.get("G1_ANKLE_EFFORT_DR_HI", "0.95")),
)

# Ankle soft-barrier (Agent D §2, delta 3): relu(|tau| - TAU_SOFT)^2, weight BARRIER_W.
ANKLE_BARRIER_TAU_SOFT = float(os.environ.get("G1_ANKLE_BARRIER_TAU", "35.0"))
ANKLE_BARRIER_W = float(os.environ.get("G1_ANKLE_BARRIER_W", "-5e-3"))

# Per-channel ankle action-rate (Agent D §2, delta 4).
ANKLE_ACTION_RATE_W = float(os.environ.get("G1_ANKLE_ACTION_RATE_W", "-0.05"))

# Waist-tracking slack (Agent D §2, delta 5). Windows in ORIGINAL (1.0x) motion seconds;
# scaled by G1_SLOWDOWN at build time to the slowed clock the motion npz runs on.
WAIST_BODY_NAME = "torso_link"
WAIST_SLACK = float(os.environ.get("G1_WAIST_SLACK", "0.5"))
WAIST_SLACK_WINDOWS_S = ((13.0, 18.0), (25.0, 36.0))

# The two privileged terms Agent 0 moves to critic-only.
PRIVILEGED_ACTOR_TERMS = ("base_lin_vel", "motion_anchor_pos_b")

# Non-ankle joint regexes (every G1 non-ankle joint contains exactly one of these
# substrings; no ankle joint contains any). Used to re-scope the global effort DR off
# the ankle channels so ankles get ONLY the widened band.
NON_ANKLE_JOINT_PATTERNS = (
  ".*hip.*", ".*knee.*", ".*waist.*", ".*shoulder.*", ".*elbow.*", ".*wrist.*",
)

# Per-term actor obs dims (from Agent 0 audit table, upstream_alignment_report.md §2).
# Used by --selfcheck to compute the actor dim WITHOUT instantiating the (GPU) env.
TERM_DIMS = {
  "command": 58,
  "motion_anchor_pos_b": 3,
  "motion_anchor_ori_b": 6,
  "base_lin_vel": 3,
  "base_ang_vel": 3,
  "joint_pos": 29,
  "joint_vel": 29,
  "actions": 29,
}


class ankle_torque_barrier:
  """Soft-barrier penalty on ankle actuator torque (Agent D §2, delta 3).

  penalty = sum_ankles relu(|tau| - TAU_SOFT)^2 . ~0 below TAU_SOFT (35 Nm), rises
  steeply toward the 40 Nm clamp, so it bites ONLY near saturation — exactly where hip
  strategy should take over. Mirrors base.ankle_torque_l2: reads qfrc_actuator (joint
  space, order-safe) on the 4 ankle ids resolved once at manager init.
  """

  def __init__(self, cfg, env):
    asset = env.scene[cfg.params["asset_cfg"].name]
    ids, _ = asset.find_joints(base.ANKLE_JOINT_NAMES)
    self._ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    self._tau_soft = float(cfg.params.get("tau_soft", ANKLE_BARRIER_TAU_SOFT))

  def __call__(self, env, asset_cfg, tau_soft=ANKLE_BARRIER_TAU_SOFT):
    asset = env.scene[asset_cfg.name]
    tau = asset.data.qfrc_actuator[:, self._ids]
    over = torch.clamp(torch.abs(tau) - self._tau_soft, min=0.0)
    return torch.sum(torch.square(over), dim=1)


class ankle_action_rate_l2:
  """L2 on the first difference of the 4 ANKLE action channels (Agent D §2, delta 4).

  Scoped version of the stock action_rate_l2 (env.action_manager.action -
  prev_action), restricted to the ankle channels. All 29 G1 joints are actuated by a
  single JointPositionAction over actuator_names=(".*",), so the action vector is in
  joint order and the ankle joint ids (from find_joints) index the ankle action
  channels. (Box smoke test validates the shape; a mis-map would penalise the wrong 4
  channels, not crash.)
  """

  def __init__(self, cfg, env):
    asset = env.scene[cfg.params["asset_cfg"].name]
    ids, _ = asset.find_joints(base.ANKLE_JOINT_NAMES)
    self._ids = torch.tensor(ids, device=env.device, dtype=torch.long)

  def __call__(self, env, asset_cfg):
    am = env.action_manager
    cur = am.action
    prev = getattr(am, "prev_action", None)
    if prev is None:
      prev = cur  # first step / older mjlab: no penalty rather than a crash
    d = (cur - prev)[:, self._ids]
    return torch.sum(torch.square(d), dim=1)


class WaistGatedBodyTracking:
  """Time-gated waist-slack version of motion_body_{pos,ori} (Agent D §2, delta 5).

  Reproduces mjlab's motion_relative_body_{position,orientation}_error_exp EXACTLY
  (exp of the mean per-body error over all command bodies), EXCEPT the torso_link
  ("waist") body's squared error is multiplied by WAIST_SLACK (0.5) while that env's
  motion clock is inside a hard beat. Windows are pre-scaled by G1_SLOWDOWN to the
  slowed clock the motion npz runs on. Outside the windows every body weight is 1.0 ->
  numerically identical to the stock v7 term (no regression). Arms/shoulders always 1.0.

  Motion time per env = command.time_steps * step_dt (the motion npz is at the 50 Hz
  control rate; see mjlab MotionCommand.time_steps / bin_count).
  """

  def __init__(self, cfg, env):
    self._dt = float(getattr(env, "step_dt", 0.02))
    self._waist_idx = None  # resolved lazily against command.cfg.body_names

  def __call__(self, env, command_name, std, kind, waist_body_name, windows_s):
    command = env.command_manager.get_term(command_name)
    if self._waist_idx is None:
      names = list(command.cfg.body_names)
      self._waist_idx = names.index(waist_body_name) if waist_body_name in names else -1

    if kind == "pos":
      per_body = torch.sum(
        torch.square(command.body_pos_relative_w - command.robot_body_pos_w), dim=-1
      )
    else:  # "ori"
      per_body = (
        quat_error_magnitude(command.body_quat_relative_w, command.robot_body_quat_w) ** 2
      )

    weight = torch.ones_like(per_body)  # [num_envs, n_body]
    if self._waist_idx >= 0:
      t = command.time_steps.to(per_body.dtype) * self._dt  # motion seconds, [num_envs]
      in_window = torch.zeros_like(t, dtype=torch.bool)
      for w0, w1 in windows_s:
        in_window |= (t >= w0) & (t <= w1)
      weight[in_window, self._waist_idx] = WAIST_SLACK

    error = (weight * per_body).mean(dim=-1)
    return torch.exp(-error / std**2)


def _scaled_windows():
  return tuple((a * G1_SLOWDOWN, b * G1_SLOWDOWN) for a, b in WAIST_SLACK_WINDOWS_S)


def _drop_privileged_actor_terms(cfg) -> None:
  """Asymmetric actor-critic (Agent 0): drop the two privileged terms from the ACTOR
  group; leave them in the CRITIC group. Runs in BOTH train and play so the exported
  policy is 154-dim. Idempotent + guarded so a future mjlab that already dropped them
  does not error."""
  actor = cfg.observations["actor"]
  for name in PRIVILEGED_ACTOR_TERMS:
    if name in actor.terms:
      del actor.terms[name]


def _apply_v8(cfg, train: bool):
  # 1. inherit every v7 delta (which inherits v6 drift termination + v5 arm/station).
  v7._apply_v7(cfg)

  # 3. replace the global ankle_torque_l2 with the soft-barrier at 35 Nm.
  cfg.rewards.pop("ankle_torque_l2", None)
  cfg.rewards["ankle_torque_barrier"] = RewardTermCfg(
    func=ankle_torque_barrier,
    weight=ANKLE_BARRIER_W,
    params={"asset_cfg": SceneEntityCfg("robot"), "tau_soft": ANKLE_BARRIER_TAU_SOFT},
  )

  # 4. per-channel ankle action-rate (on top of the global action_rate_l2 -0.25).
  cfg.rewards["ankle_action_rate_l2"] = RewardTermCfg(
    func=ankle_action_rate_l2,
    weight=ANKLE_ACTION_RATE_W,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )

  # 5. waist-tracking slack: REPLACE the global body pos/ori terms with the gated
  #    versions (identical outside the beats; torso weight 0.5 inside them).
  win = _scaled_windows()
  cfg.rewards["motion_body_pos"] = RewardTermCfg(
    func=WaistGatedBodyTracking,
    weight=1.0,
    params={"command_name": "motion", "std": 0.3, "kind": "pos",
            "waist_body_name": WAIST_BODY_NAME, "windows_s": win},
  )
  cfg.rewards["motion_body_ori"] = RewardTermCfg(
    func=WaistGatedBodyTracking,
    weight=1.0,
    params={"command_name": "motion", "std": 0.4, "kind": "ori",
            "waist_body_name": WAIST_BODY_NAME, "windows_s": win},
  )

  # 2. velocity-honest ankle effort clamp + widened downward DR (train only — the DR
  #    events only exist in train mode; deploy/play clamp at the true motor limit).
  if train and "dr_effort_limits" in cfg.events:
    # re-scope the stock global effort DR OFF the ankle channels (keep 0.80-1.00 there)
    cfg.events["dr_effort_limits"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=NON_ANKLE_JOINT_PATTERNS
    )
    # deterministic clamp 50 -> 40 Nm on the 4 ankle channels
    cfg.events["dr_ankle_effort_clamp"] = EventTermCfg(
      mode="startup",
      func=dr.effort_limits,
      params={
        "effort_limit_range": (ANKLE_CLAMP_SCALE, ANKLE_CLAMP_SCALE),
        "operation": "scale",
        "asset_cfg": SceneEntityCfg("robot", joint_names=base.ANKLE_JOINT_NAMES),
      },
    )
    # widened downward effort DR (0.65-0.95) on ankles only -> trained envelope 26-38 Nm
    cfg.events["dr_effort_limits_ankle"] = EventTermCfg(
      mode="startup",
      func=dr.effort_limits,
      params={
        "effort_limit_range": ANKLE_EFFORT_DR,
        "operation": "scale",
        "asset_cfg": SceneEntityCfg("robot", joint_names=base.ANKLE_JOINT_NAMES),
      },
    )

  # 0. asymmetric actor-critic obs (drop privileged terms from the actor). LAST, so the
  #    v7/base wiring that touched actor base_lin_vel is cleanly superseded.
  _drop_privileged_actor_terms(cfg)
  return cfg


def _make(train: bool, play: bool):
  return _apply_v8(base._make(train=train, play=play), train=train)


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=_make(train=True, play=False),
  play_env_cfg=_make(train=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)


def _actor_obs_dim(cfg):
  """Sum the per-term dims of the ACTOR group from TERM_DIMS. Returns (dim, unknown)
  where unknown is the list of actor terms with no dim in the table (so nothing is
  silently miscounted)."""
  actor = cfg.observations["actor"]
  dim, unknown = 0, []
  for k in actor.terms.keys():
    if k in TERM_DIMS:
      dim += TERM_DIMS[k]
    else:
      unknown.append(k)
  return dim, unknown


def _selfcheck() -> int:
  """Cheap CPU preflight (seconds, not hours). Asserts every v8 reward/obs/term key is
  registered, that the asymmetric actor/critic split is REAL on this mjlab, prints the
  actor obs dim (must be 154, NOT 160), the ankle effort clamp, and the slowdown factor.
  LOUDLY flags anything the base task could not supply rather than passing silently."""
  cfg = _make(train=True, play=False)
  ok = True

  print("== rewards ==")
  need = ("ankle_torque_barrier", "ankle_action_rate_l2", "motion_body_pos",
          "motion_body_ori", "motion_arm_pos", "motion_arm_ori",
          "action_rate_l2", "motion_global_root_pos")
  for k in need:
    present = k in cfg.rewards
    ok &= present
    w = f"w={cfg.rewards[k].weight}" if present else ""
    print(f"  {k:<24} {'OK' if present else '!! MISSING'}  {w}")
  # the global L2 must be GONE (replaced by the barrier); a silent inherit re-runs v7.
  removed = "ankle_torque_l2" not in cfg.rewards
  ok &= removed
  print(f"  ankle_torque_l2 removed  {'OK (-> barrier)' if removed else '!! still present'}")
  bar_ok = abs(cfg.rewards['ankle_torque_barrier'].weight - (-5e-3)) < 1e-12
  ar_ok = abs(cfg.rewards['ankle_action_rate_l2'].weight - (-0.05)) < 1e-12
  ok &= bar_ok and ar_ok
  print(f"  ankle_torque_barrier w   : {cfg.rewards['ankle_torque_barrier'].weight}  "
        f"tau_soft={ANKLE_BARRIER_TAU_SOFT}  {'OK' if bar_ok else '!! expected -5e-3'}")
  print(f"  ankle_action_rate_l2 w   : {cfg.rewards['ankle_action_rate_l2'].weight}  "
        f"{'OK' if ar_ok else '!! expected -0.05'}")

  print("== terminations (drift curriculum kept from v6/v7) ==")
  drift_ok = "anchor_drift_xy" in cfg.terminations
  ok &= drift_ok
  print(f"  anchor_drift_xy          {'OK' if drift_ok else '!! MISSING (drift regressed!)'}")

  print("== asymmetric actor-critic obs (Agent 0) ==")
  obs = cfg.observations
  has_critic = "critic" in obs
  if not has_critic:
    ok = False
    print("  !! BASE-TASK CHANGE STILL NEEDED: this mjlab exposes NO 'critic' obs group.")
    print("  !! Cannot do an asymmetric split — the base task must expose separate")
    print("  !! actor/critic ObservationGroupCfgs before the privileged terms can be")
    print("  !! moved to critic-only. (Expected present per mjlab_mdp_ref.)")
  else:
    actor_terms = set(obs["actor"].terms.keys())
    critic_terms = set(obs["critic"].terms.keys())
    for name in PRIVILEGED_ACTOR_TERMS:
      dropped = name not in actor_terms
      kept = name in critic_terms
      ok &= dropped and kept
      print(f"  {name:<22} actor:{'DROPPED OK' if dropped else '!! STILL IN ACTOR'}  "
            f"critic:{'KEPT OK' if kept else '!! MISSING FROM CRITIC'}")

  print("== actor obs dim ==")
  adim, unknown = _actor_obs_dim(cfg)
  print(f"  actor terms  : {sorted(cfg.observations['actor'].terms.keys())}")
  print(f"  actor obs dim: {adim}   (expected 154; upstream '155' adds a +1 command "
        f"element we do NOT fabricate)")
  if unknown:
    ok = False
    print(f"  !! actor terms with UNKNOWN dim (not counted): {unknown}")
  dim_ok = (adim == 154) and (adim != 160)
  ok &= dim_ok
  print(f"  dim check    : {'OK (154, NOT 160)' if dim_ok else f'!! got {adim}, expected 154'}")

  print("== velocity-honest ankle effort clamp (Agent D) ==")
  clamp_ok = ("dr_ankle_effort_clamp" in cfg.events
              and "dr_effort_limits_ankle" in cfg.events)
  ok &= clamp_ok
  lo, hi = ANKLE_EFFORT_DR
  print(f"  ankle effort clamp       : {ANKLE_EFFORT_LIMIT_NM} Nm "
        f"(scale {ANKLE_CLAMP_SCALE:.3f} off {STOCK_ANKLE_EFFORT_NM} Nm)  "
        f"{'OK' if 'dr_ankle_effort_clamp' in cfg.events else '!! clamp event MISSING'}")
  print(f"  ankle effort DR          : {lo}-{hi}  -> trained envelope "
        f"~{ANKLE_EFFORT_LIMIT_NM*lo:.0f}-{ANKLE_EFFORT_LIMIT_NM*hi:.0f} Nm  "
        f"{'OK' if 'dr_effort_limits_ankle' in cfg.events else '!! ankle DR event MISSING'}")

  print("== slowdown + waist slack ==")
  print(f"  G1_SLOWDOWN              : {G1_SLOWDOWN}x")
  print(f"  waist-slack windows (s)  : {_scaled_windows()}  (orig {WAIST_SLACK_WINDOWS_S} "
        f"x {G1_SLOWDOWN})  factor {WAIST_SLACK}")

  print("SELFCHECK", "PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  if "--selfcheck" in sys.argv:
    raise SystemExit(_selfcheck())
  import mjlab.tasks  # noqa: F401
  from mjlab.scripts.train import main
  main()
