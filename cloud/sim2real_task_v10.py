"""Sim2real recipe v10 — attempt 7. NATIVE-TEMPO speed curriculum + stance-foot
contact shaping + saturation-duration torque shaping, layered on the proven v8
architecture (which v9 reused unchanged).

WHY v10 EXISTS (decision log 2026-07-20, torque_crosscheck_20260720):
The torque-demand model that forced v8's 1.8x / v9's 1.53x slowdown was FALSIFIED
against real robot telemetry — real ankle-pitch tau p95 measured 15.3-18.8 Nm at
NATIVE speed on the live runs vs the model's predicted p95 114 Nm (6-10x inflation
from double-support mis-attribution + extraction jitter through the double
derivative). v10 therefore targets the NATIVE (1.0x) tempo, reached through a
SPEED CURRICULUM rather than a permanently slowed motion.

=========================== THE FOUR v10 DELTAS vs v8 ===========================

1. SPEED CURRICULUM (mechanism lives in the LAUNCHER, not this file).
   mjlab 1.5.0's MotionCommand advances the motion npz exactly one frame per
   50 Hz control step (`self.time_steps += 1`, integer index — verified in
   third_party/mjlab_mdp_ref/mdp/commands.py). MotionCommandCfg exposes NO
   playback-rate / phase-speed hook, so tempo is a property of the npz itself.
   cloud/retime_motion.py generates 4 tempo-resampled variants of the SAME
   native motion (0.60x / 0.75x / 0.90x / 1.00x tempo; cubic interp + quaternion
   slerp, velocities recomputed by csv_to_npz — never frame duplication), and
   cloud/train_v10_curriculum.sh resumes training on the next-faster npz each
   stage. This file only reads G1_SLOWDOWN (= 1/tempo: 1.667, 1.333, 1.111, 1.0)
   to scale the v8 waist-slack windows to each stage's motion clock — the exact
   v8 mechanism, reused verbatim.

   SPEED RANDOMIZATION (+/-5%) — SKIPPED, infeasible without forking mjlab:
   fractional per-env playback rate would require replacing MotionCommand's
   integer `time_steps += 1` stepping with a float phase accumulator (a base-task
   fork). Faking it via the physics dt is explicitly forbidden (it would change
   the dynamics, not just the tempo). The curriculum's 4 discrete tempos already
   expose the policy to the same choreography at different speeds, which covers
   most of what mild speed DR would buy.

2. STANCE-FOOT CONTACT REWARDS (training-only shaping; ZERO deploy impact —
   rewards touch neither the obs contract nor the action space).
   A per-frame stance schedule is precomputed (lazily, on first reward call)
   from the REFERENCE motion already loaded by the command term: a foot is
   "stance" at frame t when its reference foot height is within
   G1_STANCE_HEIGHT_EPS (3 cm) of the lower foot AND its reference foot speed is
   < G1_STANCE_SPEED_MAX (0.2 m/s). Both feet can be stance (double support —
   83% of Thriller). For the CURRENT stance foot/feet we penalize:
     * stance_foot_lin_vel  — squared linear-velocity RESIDUAL of the stance
       foot body vs the reference foot velocity (residual, not absolute, so the
       term never fights the tracking objective; ref stance speed is <0.2 m/s
       anyway). Targets foot skating == drift (v8 4.31 m / v9 3.26 m vs 1.0 bar).
     * stance_foot_yaw_rate — squared yaw-rate residual of the stance foot
       (pivot-scrubbing, the other skating mode).
     * stance_foot_flat     — foot-flat shaping: sin^2 of the sole's tilt angle
       (from the foot body quaternion) while in stance.
   All plumbing already exists: MotionCommand exposes robot_body_lin_vel_w /
   robot_body_ang_vel_w / robot_body_quat_w and motion.body_pos_w /
   motion.body_lin_vel_w for the command body set, which includes both
   *_ankle_roll_link foot bodies (g1 env cfg body_names). Nothing was skipped.

3. TORQUE SHAPING: v8's ankle soft-barrier relu(|tau|-35)^2 and per-channel
   ankle action-rate are KEPT verbatim (inherited). NEW: torque_saturation_dur —
   a per-step count of joints whose |tau| exceeds a per-joint ABSOLUTE threshold
   (SAT_THRESHOLD_NM); summed over time this penalizes saturation DURATION,
   complementing the ankle barrier's magnitude shaping and covering all 29 joints.
   The thresholds are the term's OWN constants: legs/ankles at ~1.5x the measured
   native-tempo p95 dance torque so the term actually counts genuine leg/ankle
   over-exertion, and the upper body pinned at its effort limit so it counts only
   true saturation and does not tax the arm/wrist motion-fidelity rewards
   (finding I). Deliberately NOT tied to v8.ANKLE_EFFORT_LIMIT_NM (which drives the
   real actuator clamp + effort DR and must not move). Torque remains a SOFT
   penalty + the hard actuator clamp — NEVER a termination.

4. PHASE CONDITIONING — investigated, default OFF.
   The base actor obs already carries dense reference-relative conditioning:
   `command` (58 = reference joint_pos + joint_vel of the CURRENT target frame)
   and `motion_anchor_ori_b`. Together with the 5-frame history these serve as a
   de-facto phase signal (the network sees where in the choreography it is from
   what it is being asked to do). An explicit sin/cos phase obs is therefore NOT
   required and is added ONLY behind G1_PHASE_OBS=1 (default OFF).

   ############################  WARNING  ############################
   # G1_PHASE_OBS=1 CHANGES THE DEPLOY OBS CONTRACT: the actor grows #
   # a 7th term `phase_sincos` (2 dims, appended AFTER `actions`),   #
   # so per-frame becomes 156 and the flattened input 156*5 = 780.   #
   # pipeline/deploy_runtime.py MUST be updated to append            #
   # [sin(2*pi*t/T), cos(2*pi*t/T)] per frame BEFORE any deploy of a #
   # phase-obs policy. Leave OFF unless the deploy wave signs it.    #
   ####################################################################

EVERYTHING ELSE IS v8 VERBATIM (via v8._apply_v8): teacher-student split with
154-dim/frame actor + 5-frame history (770 flat), ankle 40 Nm velocity-honest
clamp + widened ankle effort DR, waist slack windows, drift termination
(G1_DRIFT_TERM_M) + latency DR (G1_CMD/OBS_DELAY_MAX_LAG) curricula, arm
fidelity, station-keeping, mass/CoM/calibration DR.

GATE NOTE (launcher, not this file): the ankle p95 gate bar moves 15 -> 22 Nm.
The real robot measured 15-19 Nm ankle p95 at native tempo on the live runs
(decision log 2026-07-20) — the old 15 bar sat BELOW physical reality. The bars
are exported as env overrides (G1_GATE_ANKLE_P95_NOMINAL_NM=22) by
train_v10_curriculum.sh; sim_gap_check.py / pick_checkpoint.py defaults are
unchanged for older recipes.

PREFLIGHT:  python cloud/sim2real_task_v10.py --selfcheck
Launch:     cloud/train_v10_curriculum.sh   (via cloud/run_attempt7.sh)
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import sim2real_task as base       # recipe v2 builder; registers the base task
import sim2real_task_v8 as v8      # the full v8 stack (obs history, barrier, DR, ...)

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1-S2R-V10"

# ---- tunables (env-overridable so the launcher can walk them without a rewrite) ----

# Stance-schedule extraction thresholds (delta 2).
STANCE_HEIGHT_EPS_M = float(os.environ.get("G1_STANCE_HEIGHT_EPS", "0.03"))
STANCE_SPEED_MAX = float(os.environ.get("G1_STANCE_SPEED_MAX", "0.2"))

# Stance reward weights (all penalties -> negative).
STANCE_LINVEL_W = float(os.environ.get("G1_STANCE_LINVEL_W", "-0.5"))
STANCE_YAWRATE_W = float(os.environ.get("G1_STANCE_YAWRATE_W", "-0.1"))
STANCE_FLAT_W = float(os.environ.get("G1_STANCE_FLAT_W", "-0.5"))

# Saturation-duration penalty (delta 3): count of joints above their per-joint
# absolute threshold (SAT_THRESHOLD_NM), per control step. -0.02 * 1 saturated
# joint-step ~ the same order as the ankle barrier at the 40 Nm clamp
# (25 * 5e-3 = 0.125) without dwarfing it.
SAT_DURATION_W = float(os.environ.get("G1_SAT_DURATION_W", "-0.02"))

# Optional explicit phase obs (delta 4) — default OFF; see WARNING in the header.
G1_PHASE_OBS = os.environ.get("G1_PHASE_OBS", "0") == "1"

# The two foot bodies. Both are in the g1 tracking command body_names
# (third_party/mjlab_mdp_ref/g1_config/env_cfgs.py), so reference AND robot
# kinematics for them are already on-device in the command term.
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")

# Per-joint ABSOLUTE saturation thresholds [Nm] for torque_saturation_duration.
# These are the term's OWN independent constants — deliberately NOT the effort
# limits and NOT v8.ANKLE_EFFORT_LIMIT_NM (that symbol also drives the real
# actuator clamp + effort DR, so it must not move). Audit finding I: at 0.90x the
# per-joint effort limit the leg/ankle bars (knee 125, hip 79-125, ankle 36 Nm)
# sat 3-8x above the measured native-tempo dance torques, so the term NEVER fired
# on a leg and instead only reached the 5 Nm wrists / 25 Nm arms — taxing the
# w=1.0 arm/wrist motion-fidelity rewards while delivering zero leg shaping.
# Legs/ankles are now set to ~1.5x the measured p95 dance torque so the term
# counts genuine over-exertion (not the whole operating band); the upper body is
# pinned at its effort limit so it only counts TRUE saturation and no longer taxes
# arm crispness. Measured p95 (decision log 2026-07-20): knee 31-34, ankle 15-19.
SAT_THRESHOLD_NM = {
  ".*_hip_pitch_joint": 40.0,    # ~1.5x measured hip envelope
  ".*_hip_roll_joint": 40.0,
  ".*_hip_yaw_joint": 40.0,
  ".*_knee_joint": 50.0,         # ~1.5x knee p95 (31-34)
  ".*_ankle_pitch_joint": 30.0,  # ~1.5x ankle p95 (15-19); INDEPENDENT of the
  ".*_ankle_roll_joint": 30.0,   # 40 Nm clamp (v8.ANKLE_EFFORT_LIMIT_NM), untouched
  "waist_yaw_joint": 88.0,       # upper body pinned AT the effort limit: counts
  "waist_roll_joint": 50.0,      # only true saturation, no spurious arm/wrist tax
  "waist_pitch_joint": 50.0,
  ".*_shoulder_pitch_joint": 25.0,
  ".*_shoulder_roll_joint": 25.0,
  ".*_shoulder_yaw_joint": 25.0,
  ".*_elbow_joint": 25.0,
  ".*_wrist_roll_joint": 25.0,
  ".*_wrist_pitch_joint": 5.0,
  ".*_wrist_yaw_joint": 5.0,
}

# v10 actor term dims = v8's + the optional phase term (selfcheck arithmetic).
TERM_DIMS = dict(v8.TERM_DIMS, phase_sincos=2)


def motion_phase_sincos(env, command_name: str) -> torch.Tensor:
  """[sin, cos] of 2*pi * (motion frame / motion length). Optional actor term
  (G1_PHASE_OBS=1) — see the deploy-contract WARNING in the module header."""
  command = env.command_manager.get_term(command_name)
  total = max(int(command.motion.time_step_total) - 1, 1)
  ang = 2.0 * math.pi * (command.time_steps.to(torch.float32) / float(total))
  return torch.stack([torch.sin(ang), torch.cos(ang)], dim=1)


class StanceFootPenalty:
  """Stance-gated foot-contact shaping (v10 delta 2). kind selects the metric:

    lin_vel  — sum over stance feet of || v_foot_robot - v_foot_ref ||^2
    yaw_rate — sum over stance feet of ( wz_foot_robot - wz_foot_ref )^2
    tilt     — sum over stance feet of sin^2(sole tilt)   (foot-flat shaping)

  The per-frame stance schedule is built ONCE (lazily) from the reference motion
  already resident on the GPU in the command term's MotionLoader: foot f is
  stance at frame t iff ref_height_f(t) - min(ref heights)(t) <= 3 cm AND
  ||ref foot velocity||(t) < 0.2 m/s. Everything is indexed by the command's
  own time_steps, so the schedule follows each env's motion clock exactly
  (works unchanged on every tempo variant — the schedule is rebuilt per process
  from whatever npz that stage trains on).
  """

  def __init__(self, cfg, env):
    self._mask = None       # [T, 2] bool, lazy
    self._foot_cols = None  # indices into the command body set

  def _build(self, command) -> None:
    names = list(command.cfg.body_names)
    cols = [names.index(n) for n in FOOT_BODY_NAMES]  # raises if absent — loud
    self._foot_cols = torch.tensor(cols, device=command.device, dtype=torch.long)
    foot_z = command.motion.body_pos_w[:, self._foot_cols, 2]            # [T,2]
    foot_speed = command.motion.body_lin_vel_w[:, self._foot_cols].norm(dim=-1)
    lower = foot_z.min(dim=1, keepdim=True).values
    self._mask = ((foot_z - lower) <= STANCE_HEIGHT_EPS_M) & (
      foot_speed < STANCE_SPEED_MAX
    )

  def __call__(self, env, command_name, kind):
    command = env.command_manager.get_term(command_name)
    if self._mask is None:
      self._build(command)
    stance = self._mask[command.time_steps].to(torch.float32)  # [n_envs, 2]

    if kind == "lin_vel":
      dv = (
        command.robot_body_lin_vel_w[:, self._foot_cols]
        - command.body_lin_vel_w[:, self._foot_cols]
      )
      val = dv.square().sum(dim=-1)                                # [n, 2]
    elif kind == "yaw_rate":
      dw = (
        command.robot_body_ang_vel_w[:, self._foot_cols, 2]
        - command.body_ang_vel_w[:, self._foot_cols, 2]
      )
      val = dw.square()                                            # [n, 2]
    else:  # "tilt" — RESIDUAL of sole tilt vs the REFERENCE sole (a foot-flat
           # shaping that, like its lin_vel/yaw_rate siblings, is a reference
           # residual so it never fights the reference weight-shifts / edge-rolls
           # that v11 motion_leg_ori is paid to track). tilt vector = world-up
           # expressed in the foot frame; xy components = R[2,0], R[2,1] (wxyz).
      q = command.robot_body_quat_w[:, self._foot_cols]            # robot  [n,2,4]
      w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
      a = 2.0 * (x * z - w * y)
      b = 2.0 * (y * z + w * x)
      qr = command.body_quat_w[:, self._foot_cols]                 # ref    [n,2,4]
      wr, xr, yr, zr = qr[..., 0], qr[..., 1], qr[..., 2], qr[..., 3]
      a_ref = 2.0 * (xr * zr - wr * yr)
      b_ref = 2.0 * (yr * zr + wr * xr)
      val = (a - a_ref).square() + (b - b_ref).square()            # [n, 2]

    return (stance * val).sum(dim=1)


class torque_saturation_duration:
  """Count of joints with |tau| > per-joint threshold, per control step
  (v10 delta 3). Integrated by the reward sum over time == time spent saturated.
  Thresholds resolve per joint NAME from SAT_THRESHOLD_NM (the term's OWN absolute
  Nm bars — see finding I) at manager init (order-safe, same qfrc_actuator source
  as the ankle barrier). SOFT shaping only — torque never terminates an episode.
  """

  def __init__(self, cfg, env):
    asset = env.scene[cfg.params["asset_cfg"].name]
    ids_all: list[int] = []
    thresh: list[float] = []
    for pattern, limit in SAT_THRESHOLD_NM.items():
      ids, _ = asset.find_joints((pattern,))
      ids_all.extend(ids)
      thresh.extend([float(limit)] * len(ids))
    if len(set(ids_all)) != len(ids_all):
      raise ValueError("SAT_THRESHOLD_NM patterns overlap — joints double-counted")
    self._ids = torch.tensor(ids_all, device=env.device, dtype=torch.long)
    self._thresh = torch.tensor(thresh, device=env.device, dtype=torch.float32)

  def __call__(self, env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    tau = asset.data.qfrc_actuator[:, self._ids].abs()
    return (tau > self._thresh).to(torch.float32).sum(dim=1)


def _apply_v10(cfg, train: bool):
  # 1. the full v8 stack (obs drop + history, ankle barrier + clamp + DR, waist
  #    slack scaled by G1_SLOWDOWN, drift termination, latency DR, v7/v6/v5 terms).
  v8._apply_v8(cfg, train=train)

  # 2. stance-foot contact shaping (training-only effect; registered in both
  #    modes like every other reward — rewards never touch the deploy contract).
  cfg.rewards["stance_foot_lin_vel"] = RewardTermCfg(
    func=StanceFootPenalty,
    weight=STANCE_LINVEL_W,
    params={"command_name": "motion", "kind": "lin_vel"},
  )
  cfg.rewards["stance_foot_yaw_rate"] = RewardTermCfg(
    func=StanceFootPenalty,
    weight=STANCE_YAWRATE_W,
    params={"command_name": "motion", "kind": "yaw_rate"},
  )
  cfg.rewards["stance_foot_flat"] = RewardTermCfg(
    func=StanceFootPenalty,
    weight=STANCE_FLAT_W,
    params={"command_name": "motion", "kind": "tilt"},
  )

  # 3. saturation-duration penalty (all 29 joints, 90% of per-joint limit).
  cfg.rewards["torque_saturation_dur"] = RewardTermCfg(
    func=torque_saturation_duration,
    weight=SAT_DURATION_W,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )

  # 4. optional explicit phase obs — appended LAST so the existing 154-dim
  #    per-frame layout is untouched when enabled (phase block lands after the
  #    actions history block). Default OFF; see the deploy-contract WARNING.
  if G1_PHASE_OBS:
    cfg.observations["actor"].terms["phase_sincos"] = ObservationTermCfg(
      func=motion_phase_sincos, params={"command_name": "motion"}
    )

  return cfg


def _make(train: bool, play: bool):
  return _apply_v10(base._make(train=train, play=play), train=train)


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=_make(train=True, play=False),
  play_env_cfg=_make(train=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)


def _actor_obs_dim(cfg):
  """v8's arithmetic with the v10 TERM_DIMS (covers phase_sincos)."""
  actor = cfg.observations["actor"]
  per_frame, unknown = 0, []
  for k in actor.terms.keys():
    if k in TERM_DIMS:
      per_frame += TERM_DIMS[k]
    else:
      unknown.append(k)
  history = int(getattr(actor, "history_length", 0) or 0)
  return per_frame, per_frame * (history if history > 0 else 1), history, unknown


def _selfcheck() -> int:
  """Cheap CPU preflight. Asserts the v8 inheritance is intact PLUS every v10
  delta is registered, and that the actor obs contract matches G1_PHASE_OBS."""
  cfg = _make(train=True, play=False)
  ok = True

  print("== v8 inheritance (must be intact) ==")
  need_v8 = ("ankle_torque_barrier", "ankle_action_rate_l2", "motion_body_pos",
             "motion_body_ori", "motion_arm_pos", "motion_arm_ori",
             "action_rate_l2", "motion_global_root_pos")
  for k in need_v8:
    present = k in cfg.rewards
    ok &= present
    w = f"w={cfg.rewards[k].weight}" if present else ""
    print(f"  {k:<24} {'OK' if present else '!! MISSING'}  {w}")
  removed = "ankle_torque_l2" not in cfg.rewards
  ok &= removed
  print(f"  ankle_torque_l2 removed  {'OK (-> barrier)' if removed else '!! still present'}")
  drift_ok = "anchor_drift_xy" in cfg.terminations
  ok &= drift_ok
  print(f"  anchor_drift_xy          {'OK' if drift_ok else '!! MISSING (drift regressed!)'}")
  clamp_ok = ("dr_ankle_effort_clamp" in cfg.events
              and "dr_effort_limits_ankle" in cfg.events)
  ok &= clamp_ok
  print(f"  ankle 40Nm clamp + DR    {'OK' if clamp_ok else '!! MISSING'}")

  print("== v10 rewards ==")
  v10_terms = {
    "stance_foot_lin_vel": STANCE_LINVEL_W,
    "stance_foot_yaw_rate": STANCE_YAWRATE_W,
    "stance_foot_flat": STANCE_FLAT_W,
    "torque_saturation_dur": SAT_DURATION_W,
  }
  for k, want_w in v10_terms.items():
    present = k in cfg.rewards
    w_ok = present and abs(cfg.rewards[k].weight - want_w) < 1e-12
    neg_ok = present and cfg.rewards[k].weight < 0
    ok &= present and w_ok and neg_ok
    print(f"  {k:<24} {'OK' if present else '!! MISSING'}  "
          f"w={cfg.rewards[k].weight if present else '-'} "
          f"{'OK' if w_ok and neg_ok else '!! bad weight (must be the env default, negative)'}")
  print(f"  stance thresholds        : height eps {STANCE_HEIGHT_EPS_M} m, "
        f"speed < {STANCE_SPEED_MAX} m/s")
  print(f"  saturation thresholds    : absolute Nm per joint (knee {SAT_THRESHOLD_NM['.*_knee_joint']}, "
        f"ankle {SAT_THRESHOLD_NM['.*_ankle_roll_joint']}, arm 25); "
        f"INDEPENDENT of the {v8.ANKLE_EFFORT_LIMIT_NM} Nm ankle clamp")
  # no torque termination may exist (soft-penalty rule)
  torque_terms = [k for k in cfg.terminations if "torque" in k or "saturat" in k]
  ok &= not torque_terms
  print(f"  torque terminations      : {'NONE OK' if not torque_terms else '!! ' + str(torque_terms)}")

  print("== actor obs contract ==")
  actor = cfg.observations["actor"]
  hist = int(getattr(actor, "history_length", 0) or 0)
  hist_ok = hist == v8.G1_OBS_HISTORY and hist > 1
  ok &= hist_ok
  print(f"  history_length           : {hist}  {'OK' if hist_ok else '!! not set'}")
  phase_present = "phase_sincos" in actor.terms
  phase_ok = phase_present == G1_PHASE_OBS
  ok &= phase_ok
  print(f"  G1_PHASE_OBS             : {int(G1_PHASE_OBS)}  term present: {phase_present}  "
        f"{'OK' if phase_ok else '!! obs term does not match env var'}")
  if phase_present:
    last = list(actor.terms.keys())[-1]
    last_ok = last == "phase_sincos"
    ok &= last_ok
    print(f"  phase term position      : {'LAST OK' if last_ok else '!! must be appended LAST, got ' + last}")
    print("  !! DEPLOY CONTRACT CHANGED: per-frame 156 / flat 780 — deploy_runtime.py")
    print("  !! must append [sin,cos](2*pi*t/T) per frame BEFORE deploying this policy.")
  want_pf = 154 + (2 if G1_PHASE_OBS else 0)
  per_frame, flat, _, unknown = _actor_obs_dim(cfg)
  pf_ok = per_frame == want_pf
  flat_ok = flat == want_pf * v8.G1_OBS_HISTORY
  ok &= pf_ok and flat_ok and not unknown
  print(f"  per-frame dim            : {per_frame}  (expected {want_pf})  {'OK' if pf_ok else '!!'}")
  print(f"  flattened dim            : {flat}  (expected {want_pf * v8.G1_OBS_HISTORY})  "
        f"{'OK' if flat_ok else '!!'}")
  if unknown:
    print(f"  !! actor terms with UNKNOWN dim: {unknown}")
  for name in v8.PRIVILEGED_ACTOR_TERMS:
    dropped = name not in actor.terms
    ok &= dropped
    print(f"  {name:<22} {'DROPPED OK' if dropped else '!! STILL IN ACTOR'}")

  print("== speed-curriculum clock ==")
  print(f"  G1_SLOWDOWN              : {v8.G1_SLOWDOWN}x  (launcher sets 1/tempo per stage:"
        f" 1.667 / 1.333 / 1.111 / 1.0)")
  print(f"  waist-slack windows (s)  : {v8._scaled_windows()}")
  print("  speed DR +/-5%           : SKIPPED (no playback-rate hook in mjlab 1.5.0"
        " MotionCommand; would need a base-task fork — see header)")

  print("SELFCHECK", "PASS" if ok else "FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  if "--selfcheck" in sys.argv:
    raise SystemExit(_selfcheck())
  import mjlab.tasks  # noqa: F401
  from mjlab.scripts.train import main
  main()
