#!/usr/bin/env python3
"""Laptop-side deploy runtime: run a trained mjlab ONNX policy on the REAL Unitree G1
over Ethernet (like the existing teleop), no Docker/onboard controller needed.

WHY laptop-side: the robot's onboard Docker controller belongs to a colleague (off
limits) and the BeyondMimic image isn't present. This drives the robot the same way
~/robot's teleop does — unitree_sdk2py + CycloneDDS over enp0s31f6.

Run in the `tv` conda env (has unitree_sdk2py + CycloneDDS + numpy + onnxruntime).

    conda activate tv
    python -m pipeline.deploy_runtime --mode read      # SAFE default: reads + prints, sends NOTHING

SAFETY — this can move a 35 kg robot. Non-negotiable:
  * --mode read is the default and sends NOTHING. Use it to sanity-check the policy.
  * --mode move-to-default and --mode run COMMAND MOTORS. They refuse to run unless BOTH
    `--i-will-watch-the-robot` is passed AND env CONFIRMED_BY_HUMAN=alois. Gantry-first,
    feet off ground, remote (damping) in hand.
  * Any NaN/inf/out-of-range policy output -> immediate damping + exit.
  * Targets clamped to joint limits; torque clamped to effort_limit. 50 Hz loop with a
    watchdog: a cycle overrun -> damping.

OBS FIDELITY NOTE (read this before trusting anything on the GROUND):
  148 of the 160 obs dims (command 58, joint_pos 29, joint_vel 29, base_ang_vel 3,
  actions 29) are built EXACTLY from LowState + the reference motion. The remaining 12
  (motion_anchor_pos_b 3, motion_anchor_ori_b 6, base_lin_vel 3) need the torso's world
  pose/velocity, which the real robot can't measure without a state estimator. On the
  GANTRY the base barely translates so base_lin_vel~=0 (inside the policy's training
  noise) and anchor_pos_b is approximated as the reference's displacement-from-start in
  the IMU frame (~=0 at t=0). This is fine for gantry sanity + tracking, but ground-free
  use needs a real torso-pose estimator (DLIO) feeding these terms. Flagged, not hidden.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

# Set by a motion mode once the LowCmd publisher is up: (pub, low_cmd, crc, mode_machine,
# meta). The signal handler + clean-exit use it to GUARANTEE the robot ends DAMPED (soft)
# on ANY exit path — normal end, Ctrl-C, or an external SIGTERM/kill.
_DAMP_CTX = None
# Active Telemetry recorder (set by motion modes). Saved best-effort in _finalize_and_exit
# AFTER damping — safety never waits on telemetry.
_TELEM = None

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_META = ROOT / "data/policies/thriller/policy_meta.json"
DEFAULT_MOTION = ROOT / "data/policies/thriller/thriller_deploy.npz"
DEFAULT_POLICY = ROOT / "data/policies/thriller/policy.onnx"
# Robot control interface (CycloneDDS). Default = the wired ethernet the teleop uses. Override
# with ROBOT_IFACE (env) or --iface for a WIRELESS show — but note real-time control over wifi
# risks jitter/dropout on the 50 Hz balance loop (the read_state comms-loss deadman damps on a
# stale read; validate wifi latency/jitter first — see docs/WIRELESS_SHOW.md).
IFACE = os.environ.get("ROBOT_IFACE", "enp0s31f6")
TORSO_NPZ_IDX = 15          # torso_link: mjlab body 16 minus the dropped world body
CONTROL_HZ = 50.0
# Firm approach gains for reaching the ready pose (verified on hardware 2026-07-05):
# scale BOTH kp and kd so the joint stays overdamped. Env-overridable.
APPROACH_KP_SCALE = float(os.environ.get("APPROACH_KP_SCALE", "2.0"))
# |action|>this -> damp. 12.0 is hardware-validated: at 8.0 free-swinging gantry legs grazed
# the cap and the safety damped a healthy run; 12 completed the full dance twice (2026-07-05).
MAX_ACTION = float(os.environ.get("MAX_ACTION", "12.0"))
# On exit, re-activate the onboard motion service we released, so the robot is handed back
# to onboard control and the REMOTE/app can pair again. Leaving it released strands the
# robot (remote can't reconnect — learned the hard way 2026-07-04). "" disables.
RESTORE_MOTION_MODE = os.environ.get("RESTORE_MOTION_MODE", "ai")
# END-OF-RUN EXIT behaviour (OPT-IN, clean full-completion ONLY — see --exit).
#   --exit damp  (DEFAULT): the proven smooth ramp-to-damping handoff. Byte-for-byte the
#                current behaviour; the frozen demo path never sets --exit so it is untouched.
#   --exit stand: after the LAST dance tick, keep actively commanding the motion's FINAL
#                (standing) reference pose at the SAME holding gains the policy just used for
#                HANDOFF_HOLD_S, THEN restore the onboard motion service (SelectMode) and only
#                AFTER that returns stop publishing lowcmd — the vendor controller takes over a
#                still-balanced STANDING robot instead of catching a damped collapse (removes
#                the end-of-run catch-step). This is UNVALIDATED on hardware (see the runtime
#                docstring); every ABORT/FAULT path still damps immediately regardless of --exit.
HANDOFF_HOLD_S = float(os.environ.get("HANDOFF_HOLD_S", "2.0"))
# Optional OVERLAP: after SelectMode('ai') is re-asserted at the handoff, keep commanding
# the SAME standing pose for this long so the robot is never briefly unheld if onboard's
# takeover has latency (bridges a gap-induced catch-step). Commands only the pose the robot
# is already in -> no new-pose fall risk; a harmless no-op if lowcmd is ignored once 'ai'
# owns the actuators. Default 0.5s: VALIDATED on hardware 2026-07-07 (tethered, 2 replications)
# — 0.5s overlap shrank the onboard-takeover catch-step to negligible/gone, where 0.0 left a
# small shift. The residual step was thus a brief unheld GAP at takeover, not (only) the ~18 deg
# pose mismatch to onboard's neutral. Set 0.0 to restore the no-overlap behavior.
HANDOFF_OVERLAP_S = float(os.environ.get("HANDOFF_OVERLAP_S", "0.5"))
# ENTRY handoff (mirror of the exit overlap): the onboard->policy takeover has an unheld
# window while the motion service releases — nothing for feet-off gantry, but a FALL RISK
# untethered on the ground. Fix: pre-arm our publisher + safety spine BEFORE releasing (zero
# setup latency), then HOLD the robot's CURRENT pose for ENTRY_CATCH_S the instant onboard
# lets go, so our controller grabs it exactly where it stands before easing to the ready pose.
# Default 0.5s. Set 0.0 to restore the old release->ramp entry. Untethered use MUST be tether-
# validated first (watch the feet at the onboard->policy handoff).
ENTRY_CATCH_S = float(os.environ.get("ENTRY_CATCH_S", "0.5"))
# GUARD for --exit stand: the motion's final frame must be within this many rad of the
# default (standing) pose on EVERY joint, else the handoff would start from a non-standing
# pose and could topple the robot -> refuse --exit stand and fall back to damp.
STAND_GUARD_TOL_RAD = 0.15
# Policy-phase LEG gain boost. The trained gains stand/balance the robot in SIM, but are
# too soft to bear its weight on the real hardware — it sags into a crouch and dances from
# there instead of standing (observed 2026-07-04). Scale ONLY the leg joints (hips/knees/
# ankles, idx 0-11) so the legs can hold standing under load; arms keep their trained gains
# so the dance tracks. Tune UP on the tether while watching for oscillation. Env-overridable.
GROUND_LEG_KP_SCALE = float(os.environ.get("GROUND_LEG_KP_SCALE", "1.0"))
# Boost ONLY the SAGITTAL weight-bearing joints (hip pitch, knee, ankle pitch). The ROLL
# joints (hip_roll idx 1/7, ankle_roll idx 5/11) are what the policy uses for SIDEWAYS
# balance — stiffening them fought the policy and the robot fell sideways into the tether
# (observed 2026-07-04, 5s run). Leave roll/yaw at trained gains so lateral balance is free.
LEG_JOINT_IDX = [0, 3, 4, 6, 9, 10]  # L/R hip_pitch, knee, ankle_pitch
# Policy-phase ARM gain boost (ground-run-legodom ONLY; dance-quality program 2026-07-06).
# System-ID (tools/system_id_arms.py -> data/reports/system_id_20260706.json, 3 full-dance
# telemetry runs): the arm plant lags its COMMANDED target 81-141 ms with Coulomb friction
# 0.1-0.5 Nm; at the trained arm gains (kp 14.25/16.78) the PD command sits at the friction
# floor (wrist_roll delivers only 0.24-0.37x) and shoulders under-swing (amp 0.83-0.92) —
# the visible "arms less crisp than sim" gap. Scales kp AND kd of the 14 arm joints
# (identified BY NAME from meta.joint_order, never positional) by the SAME factor:
# damping ratio zeta = kd/(2*sqrt(kp*J)) then RISES by sqrt(scale) — never less damped
# than trained (no oscillation risk; real friction adds damping sim lacked) — and it
# exactly reproduces the V3B retrain's train-time actuator scaling, so one knob serves
# both the deploy-side boost experiment and a V3B-policy deploy (REQUIRED 2.5 there).
# At 2.5x the resulting kp (35.6 shoulder/elbow/wrist_roll, 41.9 wrist_pitch/yaw) stays
# inside the teleop-proven envelope on these motors (kp 80 shoulder/elbow, 40 wrist).
ARM_GROUND_KP_SCALE = float(os.environ.get("ARM_GROUND_KP_SCALE", "1.0"))
ARM_GROUND_KP_SCALE_MAX = 3.0
# Feedforward gravity compensation (EXPERIMENTAL, default OFF). NOTE (audit 2026-07-05):
# the earlier rationale ("sim's position actuator implicitly provides gravity-hold torque")
# was wrong physics — sim and firmware run the SAME PD law; sim stands because the POLICY
# balances, not because the actuator hides gravity. Also the 2026-07-04 FF hardware test
# computed HANGING (fixed-base) torques, i.e. ~zero ankle FF, so it falsified nothing.
# A standing-support (feet-loaded) gravity FF remains an untested deploy-side lever.
GRAVITY_FF = os.environ.get("GRAVITY_FF", "0") == "1"
GRAVITY_FF_SCALE = float(os.environ.get("GRAVITY_FF_SCALE", "1.0"))  # ramp-in / trim knob
# Yaw-align the reference world frame to the robot's actual heading at policy start.
# Training expresses robot and reference in ONE world frame; on hardware the IMU yaw is
# arbitrary (boot heading) and the show workflow WALKS the robot into place first. The
# thriller_deploy reference's t=0 torso yaw is 90.3 deg in its npz world frame, so without
# this alignment motion_anchor_ori_b carries a permanent yaw error far outside anything
# training saw (RSI yaw range ±0.2 rad) — the policy fights it all dance (audit 2026-07-05).
# '0' restores the old unaligned behaviour.
YAW_ALIGN = os.environ.get("YAW_ALIGN", "1") != "0"
# Use the TORSO orientation (pelvis IMU quat composed with waist-joint FK) for the anchor
# obs terms — training's anchor body is torso_link, the IMU sits at the pelvis. '1' enables
# after the offline sensitivity test quantifies it; '0' = pelvis approximation (old behaviour).
TORSO_ANCHOR = os.environ.get("TORSO_ANCHOR", "1") != "0"
# Per-tick run telemetry (q/dq/tau_est/temps/IMU/action/target -> data/telemetry/*.npz).
# The '15 Nm ankle' finding came from an ad-hoc uncommitted capture (audit: unauditable);
# every motion run now records automatically so hardware numbers have provenance.
TELEMETRY = os.environ.get("TELEMETRY", "1") != "0"
TELEMETRY_DIR = Path(os.environ.get("TELEMETRY_DIR", str(ROOT / "data" / "telemetry")))
# See read_state: drain is a no-op with the verified depth-1 DDS QoS and spams SDK
# error prints at sub-ms timeouts — opt-in only.
DRAIN_READS = os.environ.get("DRAIN_READS", "0") == "1"
# Constant ankle_pitch trim (DEGREES) applied to the STAND-HOLD targets ONLY — enables
# the audit's ±3 deg posture sweep (posture -> ankle torque -> heat mapping; first-
# principles audit §4 exp #6) without code edits. Clamped to ±6 deg, loudly printed
# when nonzero, recorded in the run telemetry. No other mode reads this.
ANKLE_TRIM_DEG = float(os.environ.get("ANKLE_TRIM_DEG", "0"))
ANKLE_TRIM_MAX_DEG = 6.0
ANKLE_PITCH_IDX = [4, 10]  # left/right ankle_pitch in the 29-joint order

# FALL DETECTOR: PHYSICAL-state triggers alongside the action-based ones (bad-action, NaN,
# cycle-overrun, comms-loss). Two signals, both DEBOUNCED over FALL_CONFIRM_TICKS consecutive
# ticks so one spurious IMU/odom sample can NEVER damp a healthy robot (a single-tick trip that
# damped-to-limp would itself induce the fall it claims to catch — adversarial review 2026-07-07):
#   (1) TOPPLE: pelvis uprightness R_pelvis[2,2] (the IMU sits at the pelvis; +1 upright, 0
#       horizontal, <0 inverted) falls below FALL_UPRIGHT_MIN (~70 deg tilt). The absolute
#       threshold is valid because the vet gate forbids floorwork / requires pelvis height
#       >=0.35 m (upright in-place dances only) — no supported choreography tilts the pelvis
#       past 70 deg. Cross-checked vs ALL 26 legodom hardware runs (38.6k ticks): worst real
#       lean 35.7 deg / uprightness 0.812 — a 0.46 margin to the 0.35 trip.
#   (2) HEIGHT COLLAPSE: the torso sits FALL_HEIGHT_DROP_M below where the CHOREOGRAPHY expects
#       it — (h_est - h0) minus the reference height change. Catches a leg-buckle / vertical sag
#       that keeps the pelvis upright (which the topple signal MISSES) — the likely real ground
#       fall. Choreography-relative so an intentional squat doesn't trip. Cross-checked: clean
#       dances never sink >2.5 cm below the choreographed height, so 0.15 m has ~6x margin.
# On a CONFIRMED trip the mode's except/finally damps immediately and hands the SOFT motors back
# to onboard 'ai' so the operator's remote / vendor controller take over for recovery. Damp
# softens the impact; it does NOT arrest a committed fall, and autonomous get-up is UNVERIFIED
# future work (needs the GPU box; see docs/FALL_RECOVERY.md). Scope: the ground policy loop only.
# Env knobs: FALL_UPRIGHT_MIN=0 disables topple; FALL_HEIGHT_DROP_M=0 disables the height signal.
FALL_UPRIGHT_MIN = float(os.environ.get("FALL_UPRIGHT_MIN", "0.35"))
FALL_HEIGHT_DROP_M = float(os.environ.get("FALL_HEIGHT_DROP_M", "0.15"))
FALL_CONFIRM_TICKS = int(os.environ.get("FALL_CONFIRM_TICKS", "3"))  # 3 ticks @50Hz = 60 ms
# START-POSE GUARD: refuse to start a ground run if the robot is not standing roughly upright.
# A near-horizontal start makes move-to-default + the policy begin from a losing pose the robot
# can't recover (near-fall observed 2026-07-07 free run 3 — the robot had been left leaned over).
# Checked BEFORE releasing onboard, so a refusal leaves the robot safely self-balanced. Default
# 0.85 (~32 deg tilt); a proper upright start is ~1.0 (a few deg). START_UPRIGHT_MIN=0 disables.
START_UPRIGHT_MIN = float(os.environ.get("START_UPRIGHT_MIN", "0.85"))

# obs term order + widths (mjlab tracking, sums to 160) — authoritative layout.
OBS_LAYOUT = [
    ("command", 58), ("motion_anchor_pos_b", 3), ("motion_anchor_ori_b", 6),
    ("base_lin_vel", 3), ("base_ang_vel", 3), ("joint_pos", 29),
    ("joint_vel", 29), ("actions", 29),
]

# ---- GROUND (obs-restricted) deployment --------------------------------------
# The gantry policy's obs needs base_lin_vel + motion_anchor_pos_b, both of which
# require a torso-position/velocity state estimator we do NOT have on the robot
# (BeyondMimic arXiv 2508.08241 §obs). On the gantry we fed zeros/approximations —
# fine when the robot hangs, but on the GROUND those wrong terms would drive a fall.
# The ground retrain drops exactly those two terms -> a 154-dim estimator-free obs.
GROUND_META = ROOT / "data/policies/thriller_ground/policy_meta.json"
GROUND_POLICY = ROOT / "data/policies/thriller_ground/policy.onnx"
GROUND_MOTION = ROOT / "data/policies/thriller_ground/thriller_deploy.npz"
# Falls are unforgiving on the ground. The gantry policy's real action range is ~8.5, so
# the old default 6.0 false-tripped ~4% of ticks (runbook Stage B-ODOM); 10.0 is the
# measured-need default (audit item 7c). Re-measure for any retrained policy. Env-overridable.
GROUND_MAX_ACTION = float(os.environ.get("GROUND_MAX_ACTION", "10.0"))
# The action cap is a RUNAWAY tripwire, but action units are per-joint (action_scale
# 0.074 wrists vs 0.35 knees): Thriller's claw choreography legitimately rides the
# wrist at 10-12 units (= ~0.9 rad on a 5 Nm motor) while the legs never exceeded
# |a| 3.4 on hardware (2026-07-06 runs). A uniform cap conflates the two — so ARM
# joints (shoulder/elbow/wrist) get cap*ARM_ACTION_CAP_SCALE, legs/waist keep the
# tight tripwire that actually protects balance.
# Default 2.2 (was 1.6): the v3-family arm actions ride up to ~17.1 in the sim envelope,
# so 1.6 (arm cap = 10.0*1.6 = 16) tripped a BENIGN wrist cap on hardware 2026-07-07; 2.2
# clears the sim envelope max while legs/waist keep the tight balance tripwire. Env override.
ARM_ACTION_CAP_SCALE = float(os.environ.get("ARM_ACTION_CAP_SCALE", "2.2"))
_ARM_NAME_TOKENS = ("shoulder", "elbow", "wrist")


def action_cap_vector(meta, base_cap):
    """Per-joint |action| tripwire: base_cap for legs/waist, scaled for arm joints."""
    cap = np.full(meta.n, float(base_cap))
    for i, name in enumerate(meta.joint_order):
        if any(tok in name for tok in _ARM_NAME_TOKENS):
            cap[i] = base_cap * ARM_ACTION_CAP_SCALE
    return cap
# Per-term widths, so build_obs_ground can validate ANY layout the ground meta
# declares (it reads the order from the meta rather than hard-coding it).
TERM_WIDTHS = {
    "command": 58, "motion_anchor_pos_b": 3, "motion_anchor_ori_b": 6,
    "base_lin_vel": 3, "base_ang_vel": 3, "joint_pos": 29, "joint_vel": 29,
    "actions": 29,
}
# Terms that CANNOT be produced on the robot without a state estimator. If a
# "ground" policy still lists either, it is NOT estimator-free -> refuse to run it.
ESTIMATOR_DEPENDENT_TERMS = {"base_lin_vel", "motion_anchor_pos_b"}
# The expected estimator-free layout (154). Used as the fallback order if the meta
# does not carry an explicit obs-term list.
GROUND_OBS_LAYOUT = [
    ("command", 58), ("motion_anchor_ori_b", 6), ("base_ang_vel", 3),
    ("joint_pos", 29), ("joint_vel", 29), ("actions", 29),
]


# ---- small math helpers (match pipeline/sim_exam.py conventions) --------------
def quat_wxyz_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mat_first_two_cols_b(q_ref_wxyz, q_rob_wxyz):
    """First two columns of R_robot^T @ R_ref (6-D), as mjlab's anchor_ori_b."""
    r_ref = quat_wxyz_to_mat(q_ref_wxyz)
    r_rob = quat_wxyz_to_mat(q_rob_wxyz)
    m = r_rob.T @ r_ref
    return m[:, :2].reshape(-1)  # columns 0,1 -> 6 values


def quat_mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_axis_angle(axis, angle):
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def yaw_of_quat_wxyz(q):
    R = quat_wxyz_to_mat(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))


# Waist chain pelvis->torso in the G1 MJCF: yaw about z, then roll about x, then pitch
# about y (menagerie g1.xml joint axes; validated against MuJoCo FK in tests).
_WAIST_CHAIN = (
    ("waist_yaw_joint", (0.0, 0.0, 1.0)),
    ("waist_roll_joint", (1.0, 0.0, 0.0)),
    ("waist_pitch_joint", (0.0, 1.0, 0.0)),
)


def _anchor_quat(meta, q, imu_quat):
    """Orientation used for the anchor obs terms. Training anchors on torso_link; the
    IMU sits at the pelvis, so compose the waist joints on top of the IMU quat.
    TORSO_ANCHOR=0 falls back to the raw pelvis quat (pre-2026-07-05 behaviour)."""
    if not TORSO_ANCHOR:
        return np.asarray(imu_quat, float)
    qt = np.asarray(imu_quat, float)
    for name, axis in _WAIST_CHAIN:
        i = meta.waist_idx.get(name)
        if i is not None:
            qt = quat_mul_wxyz(qt, quat_axis_angle(axis, float(q[i])))
    return qt


def _align_reference(meta, ref, q, imu_quat):
    """Yaw-align the reference world to the robot's heading NOW (call ONCE, at policy
    start, robot at the ready pose). Returns the applied offset in radians."""
    if not YAW_ALIGN:
        print("YAW_ALIGN=0 — reference kept in its own world yaw frame (npz frame).")
        return 0.0
    dyaw = ref.align_yaw(_anchor_quat(meta, q, imu_quat))
    print(f"reference yaw-aligned to robot heading (offset {np.degrees(dyaw):+.1f} deg).")
    return dyaw


class Meta:
    def __init__(self, path: Path):
        m = json.loads(path.read_text())
        self.default = np.asarray(m["default_joint_pos_rad"], float)
        self.kp = np.asarray(m["kp_stiffness"], float)
        self.kd = np.asarray(m["kd_damping"], float)
        self.effort = np.asarray(m["effort_limit_nm"], float)
        self.action_scale = np.asarray(m["action_scale_per_joint"], float)
        self.joint_order = list(m["joint_order_29dof"])
        self.n = len(self.joint_order)
        assert self.n == 29, f"expected 29 joints, got {self.n}"
        # Waist joint indices for the pelvis->torso anchor FK (None if absent).
        self.waist_idx = {
            name: (self.joint_order.index(name) if name in self.joint_order else None)
            for name, _axis in _WAIST_CHAIN
        }
        # Optional: the exact obs term order this policy was trained with. The ground
        # (obs-restricted) exporter writes it; when present, build_obs_ground trusts it
        # over any hard-coded layout so the runtime auto-adapts to the trained obs.
        self.obs_terms = (m.get("actor_obs_terms_in_order") or m.get("obs_terms_in_order")
                          or m.get("obs_terms"))
        # joint position limits from the model would be ideal; use a safe default band
        # around the reference range if not present.
        self.q_lo = self.default - np.deg2rad(140)
        self.q_hi = self.default + np.deg2rad(140)


class Reference:
    def __init__(self, npz: Path):
        d = np.load(npz)
        self.jp = d["joint_pos"]           # [T,29]
        self.jv = d["joint_vel"]           # [T,29]
        self.apos = d["body_pos_w"][:, TORSO_NPZ_IDX, :]     # [T,3] torso world pos
        self.aquat = d["body_quat_w"][:, TORSO_NPZ_IDX, :]   # [T,4] torso world quat (wxyz)
        self.T = self.jp.shape[0]
        # sanity: torso height should be ~0.6-0.8 m at t=0
        h = float(self.apos[0, 2])
        if not (0.3 < h < 1.2):
            print(f"WARN: reference torso height at t=0 = {h:.2f} m — TORSO_NPZ_IDX may be wrong")

    def at(self, tick):
        i = min(tick, self.T - 1)
        return self.jp[i], self.jv[i], self.apos[i], self.aquat[i]

    def align_yaw(self, anchor_quat_wxyz):
        """Rotate the reference world about +z so its frame-0 heading matches the
        robot's actual heading. Training compares robot and reference in ONE world
        frame; the IMU world yaw is arbitrary (boot heading), so without this the
        anchor obs carry a permanent yaw error (ref t=0 yaw is 90.3 deg in the npz
        frame). Rotates positions about the frame-0 origin (displacement math is
        unchanged in magnitude; heights untouched). Call once per run."""
        dyaw = yaw_of_quat_wxyz(anchor_quat_wxyz) - yaw_of_quat_wxyz(self.aquat[0])
        c, s = np.cos(dyaw), np.sin(dyaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.apos = (self.apos - self.apos[0]) @ Rz.T + self.apos[0]
        qz = quat_axis_angle((0.0, 0.0, 1.0), dyaw)
        self.aquat = np.array([quat_mul_wxyz(qz, qq) for qq in self.aquat])
        self.yaw_offset = dyaw
        return dyaw


class Telemetry:
    """Auditable per-tick run capture -> data/telemetry/<stamp>_<mode>.npz.

    Records q, dq, tau_est, temperatures, IMU quat/gyro, action, target each tick.
    The '15 Nm ankle' hardware number came from an ad-hoc uncommitted script (audit
    2026-07-05: unauditable, window contaminated by the approach ramp) — with this,
    every run's numbers are reproducible, stage-tagged, and survive the session.
    Design rules: add() never raises into the control loop and does NO disk I/O
    (pure appends); save() runs at exit AFTER the robot is damped."""

    def __init__(self, mode, meta, extra=None):
        self.mode, self.meta = mode, meta
        self.extra = extra or {}
        keys = ("tick", "t", "stage", "q", "dq", "tau_est", "temp",
                "imu_quat", "gyro", "action", "target")
        self.rows = {k: [] for k in keys}
        self.path = TELEMETRY_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{mode}.npz"

    def add(self, tick, q, dq, msg, imu_quat, gyro, action, target, stage=1):
        try:
            ms = msg.motor_state
            tau = [float(ms[i].tau_est) for i in range(29)]
            temp = [float(np.atleast_1d(np.asarray(ms[i].temperature, dtype=float))[0])
                    for i in range(29)]
            r = self.rows
            r["tick"].append(int(tick)); r["t"].append(time.time()); r["stage"].append(int(stage))
            r["q"].append(np.asarray(q, float)); r["dq"].append(np.asarray(dq, float))
            r["tau_est"].append(tau); r["temp"].append(temp)
            r["imu_quat"].append(np.asarray(imu_quat, float))
            r["gyro"].append(np.asarray(gyro, float))
            r["action"].append(np.asarray(action, float))
            r["target"].append(np.asarray(target, float))
        except Exception:
            pass  # telemetry must never disturb the control loop

    def save(self, quiet=False):
        if not TELEMETRY or not self.rows["tick"]:
            return
        try:
            TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
            arrays = {k: np.asarray(v) for k, v in self.rows.items()}
            arrays["joint_order"] = np.array(self.meta.joint_order)
            arrays["kp"], arrays["kd"] = self.meta.kp, self.meta.kd
            arrays["run_meta_json"] = np.array(json.dumps({
                "mode": self.mode, "approach_kp_scale": APPROACH_KP_SCALE,
                "ground_leg_kp_scale": GROUND_LEG_KP_SCALE,
                "arm_ground_kp_scale": ARM_GROUND_KP_SCALE, "gravity_ff": GRAVITY_FF,
                "yaw_align": YAW_ALIGN, "torso_anchor": TORSO_ANCHOR,
                "max_action": MAX_ACTION, "ground_max_action": GROUND_MAX_ACTION,
                **self.extra}))
            np.savez_compressed(self.path, **arrays)
            if not quiet:
                print(f"telemetry saved: {self.path} ({len(self.rows['tick'])} ticks)")
        except Exception as e:  # noqa: BLE001 - best-effort by design
            if not quiet:
                print(f"telemetry save failed (non-fatal): {e}")


# ---- LowState reading (unitree_sdk2py, like ~/robot teleop) --------------------
def make_dds(iface):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(0, iface)


def lowstate_subscriber():
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()
    return sub


def read_state(sub, timeout_s=2.0):
    # ChannelSubscriber.Read takes SECONDS (cyclonedds duration) — the old code passed
    # int(timeout_s*1000), so failure paths waited ~1000x too long before the NO-GO.
    msg = sub.Read(timeout_s)
    if msg is None:
        raise SystemExit(f"no LowState within {timeout_s}s — robot off / wrong iface / LAN down. NO-GO.")
    # Optional drain to the LATEST queued sample. With default DDS QoS (KEEP_LAST depth 1
    # — verified on hardware: staleness p95 1.75-1.78 ms across two sessions) the reader
    # already returns the newest sample, so draining is a no-op; worse, the sub-ms timeout
    # reads make the SDK print "[Reader] take sample error" every tick (observed
    # 2026-07-06). Default OFF; DRAIN_READS=1 only if a deep-queue subscriber is ever used.
    if DRAIN_READS:
        for _ in range(8):
            newer = sub.Read(0.0005)
            if newer is None:
                break
            msg = newer
    q = np.array([msg.motor_state[i].q for i in range(29)], float)
    dq = np.array([msg.motor_state[i].dq for i in range(29)], float)
    imu = msg.imu_state
    quat = np.array(list(imu.quaternion), float)       # wxyz
    gyro = np.array(list(imu.gyroscope), float)         # rad/s, body frame
    return q, dq, quat, gyro, msg


# ---- ONBOARD state estimate (rt/odommodestate) --------------------------------
# The robot's onboard EKF publishes a base pose+velocity estimate at ~184 Hz on
# rt/odommodestate (SportModeState_). This is the state estimator the estimator-free
# path lacked — with it we can build the FULL 160-D obs HONESTLY on the ground and
# deploy the PROVEN gantry policy, instead of feeding zeros/approximations.
# position = base position in a fixed odom world frame (accumulates); velocity[3];
# body height. Confirmed live 2026-07-04 (184 Hz, clean ~0 vel at rest).
ODOM_TOPIC = "rt/odommodestate"
# base_lin_vel needs BODY-frame velocity. The odom velocity FIELD's frame (body vs
# world) is a Unitree convention we have NOT yet confirmed on hardware in motion, so
# the SAFE default derives world velocity from position differencing (frame-unambiguous)
# and rotates it into the body frame with the IMU orientation. Switch to "field" only
# after a supervised sway test confirms the field's frame. Env-overridable.
ODOM_VEL_SOURCE = os.environ.get("ODOM_VEL_SOURCE", "diff")  # "diff" | "field"


def odom_subscriber():
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    sub = ChannelSubscriber(ODOM_TOPIC, SportModeState_)
    sub.Init()
    return sub


def read_odom(sub, timeout_s=0.5):
    """Return (pos[3], vel_field[3], stamp_s) from rt/odommodestate, or None if not received.

    stamp_s is the message timestamp (sec) — used to detect a FROZEN estimate (topic still
    publishing but the estimator stalled), which would fly the policy blind. Non-fatal by
    design: the caller decides. A ground run that needs odometry must treat None as NO-GO
    (never fall back to fabricated terms mid-run)."""
    # Read() takes SECONDS (same units-bug class as read_state; fixed 2026-07-05).
    msg = sub.Read(timeout_s)
    if msg is None:
        return None
    st = getattr(msg, "stamp", None)
    stamp_s = float(getattr(st, "sec", 0)) + float(getattr(st, "nanosec", 0)) * 1e-9 if st else 0.0
    return np.array(list(msg.position), float), np.array(list(msg.velocity), float), stamp_s


# ---- observation builder (real robot state -> 160-D mjlab obs) -----------------
def build_obs(meta: Meta, ref: Reference, q, dq, imu_quat, gyro, last_action, tick):
    ref_jp, ref_jv, ref_apos, ref_aquat = ref.at(tick)
    ref_apos0 = ref.apos[0]
    # Anchor terms use the TORSO orientation (training's anchor body); pelvis IMU quat
    # composed with waist FK. base_ang_vel stays the raw pelvis gyro (training reads the
    # pelvis IMU site). Reference must be yaw-aligned first (_align_reference).
    anchor_q = _anchor_quat(meta, q, imu_quat)
    R_anchor = quat_wxyz_to_mat(anchor_q)
    terms = {
        "command": np.concatenate([ref_jp, ref_jv]),                    # 58
        # gantry approx: robot torso pos ~= reference start -> displacement of ref in robot frame
        "motion_anchor_pos_b": R_anchor.T @ (ref_apos - ref_apos0),     # 3
        "motion_anchor_ori_b": mat_first_two_cols_b(ref_aquat, anchor_q),  # 6
        "base_lin_vel": np.zeros(3),                                     # 3 (gantry ~= 0)
        "base_ang_vel": gyro,                                            # 3 (IMU gyro)
        "joint_pos": q - meta.default,                                  # 29
        "joint_vel": dq,                                                # 29
        "actions": last_action,                                         # 29
    }
    parts, widths_ok = [], True
    for name, w in OBS_LAYOUT:
        v = np.asarray(terms[name], float).reshape(-1)
        if v.shape[0] != w:
            widths_ok = False
            print(f"  !! term {name}: width {v.shape[0]} != expected {w}")
        parts.append(v)
    obs = np.concatenate(parts)
    assert obs.shape[0] == 160 and widths_ok, f"obs dim {obs.shape[0]} != 160"
    return obs, terms


def build_obs_odom(meta: Meta, ref: Reference, q, dq, imu_quat, gyro, last_action,
                   tick, robot_disp, v_world):
    """HONEST 160-D obs for GROUND deploy of the PROVEN full-obs gantry policy.

    Same as build_obs, but the two terms build_obs fakes are computed from the onboard
    estimate (rt/odommodestate) instead:
      * motion_anchor_pos_b = R_robᵀ · (ref_disp − robot_disp) — the reference-vs-robot
        torso position error in the body frame. Both displacements are measured FROM the
        policy-start pose (re-anchoring), so absolute-origin and slow XY drift cancel, and
        at t=0 (robot at the reference) the term is ~0, matching training.
      * base_lin_vel = R_robᵀ · v_world — torso linear velocity in the body frame.
    `robot_disp` = odom_pos − odom_pos0 (world frame). `v_world` = world-frame base
    velocity (from position differencing by default; see ODOM_VEL_SOURCE).
    """
    ref_jp, ref_jv, ref_apos, ref_aquat = ref.at(tick)
    ref_disp = ref_apos - ref.apos[0]
    R_rob = quat_wxyz_to_mat(imu_quat)   # pelvis frame — training's base_lin_vel site
    anchor_q = _anchor_quat(meta, q, imu_quat)   # torso frame — training's anchor body
    R_anchor = quat_wxyz_to_mat(anchor_q)
    terms = {
        "command": np.concatenate([ref_jp, ref_jv]),                    # 58
        "motion_anchor_pos_b": R_anchor.T @ (ref_disp - robot_disp),    # 3 (HONEST)
        "motion_anchor_ori_b": mat_first_two_cols_b(ref_aquat, anchor_q),  # 6
        "base_lin_vel": R_rob.T @ v_world,                              # 3 (HONEST)
        "base_ang_vel": gyro,                                            # 3
        "joint_pos": q - meta.default,                                  # 29
        "joint_vel": dq,                                                # 29
        "actions": last_action,                                         # 29
    }
    parts, widths_ok = [], True
    for name, w in OBS_LAYOUT:
        v = np.asarray(terms[name], float).reshape(-1)
        if v.shape[0] != w:
            widths_ok = False
            print(f"  !! term {name}: width {v.shape[0]} != expected {w}")
        parts.append(v)
    obs = np.concatenate(parts)
    assert obs.shape[0] == 160 and widths_ok, f"obs dim {obs.shape[0]} != 160"
    return obs, terms


def _ground_obs_order(meta: Meta):
    """Resolve the ground obs term order and REFUSE anything estimator-dependent.

    Prefer the order the ground meta declares (auto-adapts to the trained policy);
    fall back to the documented 154-dim layout. Either way, if the order contains a
    term we cannot honestly produce on the robot (base_lin_vel / motion_anchor_pos_b),
    this is not an estimator-free policy — hard-refuse rather than feed it a lie that
    drives a fall on the ground.
    """
    if meta.obs_terms:
        order = [(name, TERM_WIDTHS[name]) for name in meta.obs_terms if name in TERM_WIDTHS]
        unknown = [n for n in meta.obs_terms if n not in TERM_WIDTHS]
        if unknown:
            raise SystemExit(f"REFUSED: ground meta lists unknown obs term(s) {unknown}")
    else:
        order = list(GROUND_OBS_LAYOUT)
    bad = [n for n, _ in order if n in ESTIMATOR_DEPENDENT_TERMS]
    if bad:
        raise SystemExit(
            "REFUSED: 'ground' policy obs still needs estimator-only term(s) "
            f"{bad} — we have no torso state estimator, so on the GROUND these would "
            "be fabricated and drive a fall. The retrain must drop them (154-dim obs).")
    return order


def build_obs_ground(meta: Meta, ref: Reference, q, dq, imu_quat, gyro,
                     last_action, tick, order):
    """Estimator-free obs (default 154) for GROUND deployment.

    Identical to build_obs for the shared terms, but NEVER includes base_lin_vel or
    motion_anchor_pos_b — no fabricated estimator quantities. `order` comes from
    _ground_obs_order (already validated estimator-free)."""
    ref_jp, ref_jv, _ref_apos, ref_aquat = ref.at(tick)
    anchor_q = _anchor_quat(meta, q, imu_quat)
    terms = {
        "command": np.concatenate([ref_jp, ref_jv]),                       # 58
        "motion_anchor_ori_b": mat_first_two_cols_b(ref_aquat, anchor_q),  # 6 (IMU+waist FK)
        "base_ang_vel": gyro,                                              # 3 (IMU gyro)
        "joint_pos": q - meta.default,                                     # 29
        "joint_vel": dq,                                                   # 29
        "actions": last_action,                                            # 29
    }
    parts, widths_ok = [], True
    for name, w in order:
        v = np.asarray(terms[name], float).reshape(-1)
        if v.shape[0] != w:
            widths_ok = False
            print(f"  !! term {name}: width {v.shape[0]} != expected {w}")
        parts.append(v)
    obs = np.concatenate(parts)
    expected = sum(w for _, w in order)
    assert obs.shape[0] == expected and widths_ok, f"ground obs dim {obs.shape[0]} != {expected}"
    return obs, terms


def run_policy(session, obs, tick):
    out = session.run(["actions"], {
        "obs": obs[None].astype(np.float32),
        "time_step": np.array([[float(tick)]], np.float32),
    })
    return out[0][0].astype(np.float64)


def action_to_target(meta: Meta, action):
    return meta.default + meta.action_scale * action


# ---- MODE: read (SAFE, default) ------------------------------------------------
def mode_read(meta, ref, session, iface, timeout_s):
    make_dds(iface)
    sub = lowstate_subscriber()
    print(f"reading LowState on {iface} ...")
    q, dq, imu_quat, gyro, _ = read_state(sub, timeout_s)
    # Align exactly as a motion run would, and REPORT the offset — a large value here
    # means the robot is facing away from the npz frame and the old (unaligned) obs
    # would have been far out of distribution.
    _align_reference(meta, ref, q, imu_quat)
    last_action = np.zeros(meta.n)
    obs, terms = build_obs(meta, ref, q, dq, imu_quat, gyro, last_action, tick=0)

    # ODOM CHECK (read-only): is the onboard estimate ground-run-odom depends on live?
    print("\n=== ONBOARD ESTIMATE (rt/odommodestate) ===")
    try:
        odom = odom_subscriber()
        samples = [read_odom(odom, timeout_s=0.3) for _ in range(5)]
        samples = [s for s in samples if s is not None]
        if not samples:
            print("  NOT PUBLISHED — ground-run-odom would REFUSE (NO-GO). Estimator-free "
                  "ground-run or Stage A stand-hold only until this is live.")
        else:
            p, v, _ = samples[-1]
            moved = np.linalg.norm(samples[-1][0] - samples[0][0])
            print(f"  LIVE ({len(samples)}/5 reads): pos={np.round(p,3).tolist()} "
                  f"vel={np.round(v,3).tolist()} height={p[2]:+.3f}m")
            print(f"  drift across reads: {moved*1000:.1f} mm (should be ~0 at rest). "
                  f"vel_src={ODOM_VEL_SOURCE}. ground-run-odom OK to attempt (tethered).")
    except Exception as e:  # noqa: BLE001 - diagnostic only, never fatal
        print(f"  odom read error (non-fatal): {e}")

    # obs sanity
    bad = (~np.isfinite(obs)).sum()
    big = int((np.abs(obs) > 50).sum())
    print("\n=== OBS SANITY (t=0) ===")
    print(f"  dim: {obs.shape[0]}  non-finite: {bad}  |values|>50: {big}  "
          f"range: [{obs.min():.2f}, {obs.max():.2f}]")
    for name, w in OBS_LAYOUT:
        v = np.asarray(terms[name], float).reshape(-1)
        tag = "  (EXACT)" if name in ("command", "base_ang_vel", "joint_pos", "joint_vel", "actions") \
            else "  (gantry-approx)"
        print(f"  {name:<22} n={w:<3} range[{v.min():+.3f},{v.max():+.3f}]{tag}")

    if bad:
        print("\nNO-GO: obs has non-finite values — do not run the policy.")
        return 2

    action = run_policy(session, obs, tick=0)
    target = action_to_target(meta, action)
    delta = target - q  # how far each joint would be commanded to move from NOW

    print("\n=== POLICY OUTPUT (t=0) ===")
    a_bad = (~np.isfinite(action)).sum()
    print(f"  actions: non-finite {a_bad}  range [{action.min():+.3f}, {action.max():+.3f}]")
    print(f"  {'joint':<26}{'now(deg)':>9}{'target(deg)':>12}{'move(deg)':>10}")
    worst = 0.0
    for name, qn, tg, dl in zip(meta.joint_order, q, target, delta):
        worst = max(worst, abs(dl))
        flag = "  <-- big" if abs(dl) > np.deg2rad(45) else ""
        print(f"  {name:<26}{np.degrees(qn):>9.1f}{np.degrees(tg):>12.1f}{np.degrees(dl):>10.1f}{flag}")
    print("-" * 60)
    print(f"  worst commanded move-from-now: {np.degrees(worst):.1f} deg")
    print("\nNOTE: big move-from-now values are EXPECTED here — the robot is limp/hanging,")
    print("not at the ready pose. That's why deployment must move-to-default FIRST, then run.")
    print("What matters for GO: actions are finite and bounded (range within ~[-3,3]).")
    if a_bad or not np.all(np.abs(action) < 6):
        print("CAUTION: actions look unbounded/odd — investigate before any motion.")
        return 2
    print("READ-ONLY sanity: PASS — policy produced finite, bounded actions on the real robot.")
    return 0


# ---- MODES: motion (GATED — human-supervised only) -----------------------------
def _require_human(mode):
    if not (os.environ.get("CONFIRMED_BY_HUMAN") == "alois"):
        raise SystemExit(f"REFUSED: --mode {mode} needs env CONFIRMED_BY_HUMAN=alois")
    print(f"[{mode}] human-confirmed. Damping is your remote's job if anything looks wrong.")


PR_MODE = 0  # Mode.PR (series control for pitch/roll) — matches h1_2 low_level example


def _release_motion_service():
    """Release the built-in sport/balance service so rt/lowcmd is accepted for the LEGS.
    Matches h1_2_low_level_example Init(). WARNING: disables onboard balance — only safe
    with feet OFF the ground on the gantry, remote (damping) in hand."""
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    print("!! RELEASING onboard motion/balance service — the robot will NOT self-balance.\n"
          "   Feet OFF ground, remote in hand, ready to damp.")
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    # CheckMode can transiently return None right after Init (raced the switcher) — do NOT
    # assume a dict, or we crash BEFORE taking control and leave the run half-initialized.
    for _ in range(15):
        status, result = msc.CheckMode()
        if result is None:                 # transient — retry, don't crash
            time.sleep(0.5)
            continue
        if not result.get("name"):         # no active mode -> released; rt/lowcmd accepted
            print("   motion service released — rt/lowcmd accepted for full-body.")
            return
        msc.ReleaseMode()
        time.sleep(1.0)
    raise SystemExit("could not release motion service after 15 tries — abort "
                     "(robot still under onboard control, safe)")


def _lowcmd_setup():
    """Publisher + ONE reusable LowCmd from the factory (pre-allocates motor_cmd[]) + CRC.
    Reuse the object every tick (mutate fields) — matches the h1_2 example."""
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    low_cmd = unitree_hg_msg_dds__LowCmd_()   # factory: motor_cmd[] pre-allocated
    return pub, low_cmd, CRC()


def _send_cmd(pub, low_cmd, crc, mode_machine, targets, kp, kd, meta, damping=False, tau_ff=None):
    """Mutate the REUSED low_cmd and publish. damping=True -> hold, kp=0, small kd.

    tau_ff (optional 29-vec): FEEDFORWARD torque added at each motor (the real robot's cmd is
    tau = kp*(q_des-q) + kd*(dq_des-dq) + tau_ff). We use it for GRAVITY COMPENSATION so the
    legs hold the commanded pose at the TRAINED gains instead of a gain boost — the sim's
    position actuator provides this implicitly; pure PD on hardware does not, which caused the
    sag + the ankle thermal wall. Clamped to +/- effort_limit. Never applied when damping."""
    low_cmd.mode_pr = PR_MODE
    low_cmd.mode_machine = mode_machine
    for i in range(29):
        mc = low_cmd.motor_cmd[i]
        mc.mode = 1          # enable
        mc.dq = 0.0
        if damping:
            mc.q = 0.0
            mc.kp = 0.0
            mc.kd = 2.0
            mc.tau = 0.0
        else:
            mc.q = float(np.clip(targets[i], meta.q_lo[i], meta.q_hi[i]))
            mc.kp = float(kp[i])
            mc.kd = float(kd[i])
            mc.tau = 0.0 if tau_ff is None else float(np.clip(tau_ff[i], -meta.effort[i], meta.effort[i]))
    low_cmd.crc = crc.Crc(low_cmd)
    pub.Write(low_cmd)


def _damp_burst(reps=30):
    """Send a burst of damping cmds (kp=0, kd=2) from _DAMP_CTX so the robot goes SOFT.
    Best-effort, no exceptions escape — used on every exit path incl. signal handlers."""
    ctx = _DAMP_CTX
    if not ctx:
        return
    pub, low_cmd, crc, mode_machine, meta = ctx
    for _ in range(reps):
        try:
            _send_cmd(pub, low_cmd, crc, mode_machine, meta.default, meta.kp, meta.kd, meta, damping=True)
        except Exception:
            break
        time.sleep(0.01)   # ~let DDS transmit; 30*10ms ~= 0.3s of damping


def _restore_motion_service():
    """Best-effort: re-activate the onboard motion service we released, so the robot is
    handed back to onboard control and the remote/app can pair again. NEVER let a failure
    here block the exit — the robot is already soft (damped) before this runs. Verified on
    hardware that SelectMode on a limp robot does not lurch. Set RESTORE_MOTION_MODE="" off."""
    if not RESTORE_MOTION_MODE:
        return
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        msc = MotionSwitcherClient()
        msc.SetTimeout(3.0)
        msc.Init()
        msc.SelectMode(RESTORE_MOTION_MODE)
        print(f"   restored onboard motion service ('{RESTORE_MOTION_MODE}') — remote/app can pair.",
              flush=True)
    except Exception as e:  # noqa: BLE001 - restore is best-effort, exit anyway
        print(f"   WARN: could not restore motion service ({e}); if the remote won't pair, run "
              f"SelectMode('{RESTORE_MOTION_MODE}') manually or reboot the robot.", flush=True)


def _finalize_and_exit(code=0):
    """Guarantee soft robot, hand control back to onboard (so the remote can pair), then exit
    PROMPTLY (DDS teardown can hang -> os._exit). Damp happens FIRST, so safety never waits
    on the restore. Telemetry (if any) is flushed AFTER the robot is soft."""
    _damp_burst(30)
    _restore_motion_service()
    try:
        if _TELEM is not None:
            _TELEM.save()
    except Exception:
        pass
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(code)


# ---- OPT-IN clean-completion "stand" exit (removes the end-of-run catch-step) ----
# SAFETY: this replaces ONLY the smooth ramp-to-damping on a CLEAN full completion. Every
# abort/fault path (bad action, NaN, comms loss, signal, exception) still damps immediately
# via the motion modes' except/finally -> _finalize_and_exit, regardless of --exit.
def _final_pose_is_standing(meta: "Meta", ref: "Reference", tol: float = STAND_GUARD_TOL_RAD):
    """True iff the motion's FINAL reference joint pose is within `tol` rad of the default
    (standing) pose on EVERY joint. --exit stand hands the robot back to the onboard
    controller from this pose, so it MUST end standing or the handoff could topple it."""
    dev = np.abs(np.asarray(ref.jp[-1], float) - meta.default)
    return bool(np.all(dev <= tol))


def _resolve_exit_mode(requested: str, meta: "Meta", ref: "Reference"):
    """Return the EFFECTIVE end-of-run exit mode. --exit stand is honored ONLY when the
    loaded motion ends standing (the guard); otherwise it prints a clear refusal and falls
    back to the proven 'damp'. Anything other than 'stand' -> 'damp' (the safe default)."""
    if requested != "stand":
        return "damp"
    if _final_pose_is_standing(meta, ref):
        print(f"--exit stand: motion ends within {STAND_GUARD_TOL_RAD:.2f} rad of the default "
              f"standing pose on all joints -> on a CLEAN finish the robot will HOLD that "
              f"standing pose {HANDOFF_HOLD_S:.1f}s, then hand back to onboard balance "
              f"(no end-of-run damping). ABORTS still damp.")
        return "stand"
    dev = np.abs(np.asarray(ref.jp[-1], float) - meta.default)
    j = int(dev.argmax())
    print(f"REFUSED --exit stand: motion final frame is NOT a standing pose — joint "
          f"'{meta.joint_order[j]}' sits {np.degrees(dev[j]):.1f} deg ({dev[j]:.3f} rad > "
          f"{STAND_GUARD_TOL_RAD:.2f} rad) from default. Handing off from a non-standing "
          f"pose could topple the robot -> FALLING BACK to --exit damp (proven "
          f"ramp-to-damping).")
    return "damp"


def _stand_handoff_and_exit(pub, low_cmd, crc, mode_machine, meta, ref, kp=None, kd=None):
    """CLEAN-COMPLETION 'stand' exit (OPT-IN; the guard in _resolve_exit_mode has already
    confirmed the motion ends standing). Keep actively commanding the motion's FINAL
    reference pose at the SAME holding gains the policy just used (kp/kd; default meta's
    trained gains) for HANDOFF_HOLD_S, THEN restore the onboard motion service, and ONLY
    AFTER restore returns stop publishing lowcmd — so the robot is held the whole time and
    the vendor controller takes over a still-balanced STANDING robot (no damping, no
    catch-step). Ends via os._exit so the caller's `finally: _damp(...)` does NOT run; any
    signal arriving mid-handoff still routes through _finalize_and_exit -> damping."""
    kp = meta.kp if kp is None else kp
    kd = meta.kd if kd is None else kd
    final_pose = np.asarray(ref.jp[-1], float)
    print(f"dance complete — STAND handoff: holding the final standing pose "
          f"{HANDOFF_HOLD_S:.1f}s at holding gains, then restoring onboard balance "
          f"(NO damping).", flush=True)
    for _ in range(max(1, int(HANDOFF_HOLD_S * CONTROL_HZ))):
        _send_cmd(pub, low_cmd, crc, mode_machine, final_pose, kp, kd, meta)
        time.sleep(1.0 / CONTROL_HZ)
    # Robot is standing, actively held. Hand back to onboard control BEFORE we stop
    # publishing, so it is never left unheld between us and the vendor controller.
    _restore_motion_service()
    # Optional overlap hold (HANDOFF_OVERLAP_S): keep sending the same standing pose so a
    # latent onboard takeover never leaves the robot unheld. Same-pose command only.
    for _ in range(int(HANDOFF_OVERLAP_S * CONTROL_HZ)):
        _send_cmd(pub, low_cmd, crc, mode_machine, final_pose, kp, kd, meta)
        time.sleep(1.0 / CONTROL_HZ)
    print("   onboard balance restored while STANDING — handoff complete (no catch-step).",
          flush=True)
    try:
        if _TELEM is not None:
            _TELEM.save()
    except Exception:
        pass
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(0)


def _install_damp_on_signals():
    """SIGTERM (external kill, e.g. `timeout`) and SIGINT (Ctrl-C): damp then exit.
    Default SIGTERM would terminate WITHOUT damping -> robot left energized. Not allowed."""
    def handler(signum, _frame):
        try:
            print(f"\n[signal {signum}] -> emergency damping, then exit", flush=True)
        except Exception:
            pass
        _finalize_and_exit(0)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _ramp_to_pose(pub, low_cmd, crc, mode_machine, q_start, q_end, secs, kp, kd, meta):
    """Cosine-interpolate joint targets q_start -> q_end over `secs`, streaming LowCmd at
    CONTROL_HZ with the given (firm) gains. Non-finite target -> raise (caller damps)."""
    steps = max(1, int(secs * CONTROL_HZ))
    for s in range(steps + 1):
        a = 0.5 - 0.5 * np.cos(np.pi * s / steps)   # 0 -> 1
        target = (1 - a) * q_start + a * q_end
        if not np.all(np.isfinite(target)):
            raise RuntimeError("non-finite target in ramp")
        _send_cmd(pub, low_cmd, crc, mode_machine, target, kp, kd, meta)
        time.sleep(1.0 / CONTROL_HZ)


def _hold(pub, low_cmd, crc, mode_machine, q, secs, kp, kd, meta):
    for _ in range(max(1, int(secs * CONTROL_HZ))):
        _send_cmd(pub, low_cmd, crc, mode_machine, q, kp, kd, meta)
        time.sleep(1.0 / CONTROL_HZ)


def _damp(pub, low_cmd, crc, mode_machine, meta, secs=1.0):
    for _ in range(max(1, int(secs * CONTROL_HZ))):
        _send_cmd(pub, low_cmd, crc, mode_machine, meta.default, meta.kp, meta.kd, meta, damping=True)
        time.sleep(1.0 / CONTROL_HZ)


def _check_start_upright(quat):
    """Refuse (SystemExit) if the robot is not standing roughly upright at the start of a ground
    run — call BEFORE releasing onboard so a refusal leaves the robot safely self-balanced. A
    near-horizontal start can't be recovered by move-to-default + the policy. Disabled at 0."""
    if START_UPRIGHT_MIN <= 0:
        return
    up = float(quat_wxyz_to_mat(quat)[2, 2])
    if up < START_UPRIGHT_MIN:
        tilt = float(np.degrees(np.arccos(np.clip(up, -1.0, 1.0))))
        raise SystemExit(
            f"REFUSED: robot is not upright at start ({tilt:.0f} deg tilt, uprightness {up:.2f} "
            f"< {START_UPRIGHT_MIN}) — stand it up on its feet first, then re-run. Onboard "
            f"balance untouched (nothing was released).")


def _fall_signal(R_base, h_est, h0, ref_dz):
    """(is_fall_this_tick, reason) — the raw per-tick fall condition, NOT yet debounced.
    Topple: pelvis uprightness R_base[2,2] < FALL_UPRIGHT_MIN. Height collapse: the torso sits
    FALL_HEIGHT_DROP_M below the choreographed height ((h_est-h0) vs the reference change ref_dz).
    Either fires the condition; the caller confirms it over FALL_CONFIRM_TICKS before damping."""
    if FALL_UPRIGHT_MIN > 0:
        upright = float(R_base[2, 2])
        if upright < FALL_UPRIGHT_MIN:
            tilt = float(np.degrees(np.arccos(np.clip(upright, -1.0, 1.0))))
            return True, (f"pelvis {tilt:.0f} deg from vertical "
                          f"(uprightness {upright:.2f} < {FALL_UPRIGHT_MIN})")
    if FALL_HEIGHT_DROP_M > 0 and h_est is not None:
        height_err = (float(h_est) - float(h0)) - float(ref_dz)   # actual minus choreographed
        if height_err < -FALL_HEIGHT_DROP_M:
            return True, (f"torso {-height_err:.2f} m below the choreographed height "
                          f"(drop > {FALL_HEIGHT_DROP_M} m)")
    return False, ""


def _check_fall(run_ticks, R_base, h_est, h0, ref_dz, tick):
    """DEBOUNCED fall check: returns the updated consecutive-fall-tick counter, and RAISES only
    after the condition holds FALL_CONFIRM_TICKS ticks in a row (so one spurious sample can't
    damp a healthy robot). On raise, the mode's except/finally damps + hands back to onboard."""
    fall, reason = _fall_signal(R_base, h_est, h0, ref_dz)
    run_ticks = run_ticks + 1 if fall else 0
    if run_ticks >= FALL_CONFIRM_TICKS:
        raise RuntimeError(
            f"FALL DETECTED at tick {tick}: {reason} for {run_ticks} ticks "
            f"-> damping + soft handoff to onboard")
    return run_ticks


def mode_move_to_default(meta, session, ref, iface, secs, watch):
    if not watch:
        raise SystemExit("REFUSED: pass --i-will-watch-the-robot")
    _require_human("move-to-default")
    make_dds(iface)
    sub = lowstate_subscriber()
    q0, _, _, _, msg0 = read_state(sub)
    mode_machine = int(msg0.mode_machine)
    _release_motion_service()
    pub, low_cmd, crc = _lowcmd_setup()
    global _DAMP_CTX
    _DAMP_CTX = (pub, low_cmd, crc, mode_machine, meta)
    _install_damp_on_signals()
    kp, kd = meta.kp * APPROACH_KP_SCALE, meta.kd * APPROACH_KP_SCALE  # firm, both scaled -> overdamped
    print(f"moving to default over {secs:.1f}s at {CONTROL_HZ:.0f}Hz "
          f"(approach gains {APPROACH_KP_SCALE:.1f}x)...")
    try:
        _ramp_to_pose(pub, low_cmd, crc, mode_machine, q0, meta.default, secs, kp, kd, meta)
        print("at default pose. Holding (damping).")
    finally:
        _damp(pub, low_cmd, crc, mode_machine, meta)
    _finalize_and_exit(0)   # guarantee soft + exit promptly (DDS teardown can hang)


def mode_run(meta, session, ref, iface, watch, max_secs=None, exit_mode="damp"):
    """Stage 1: firm move-to-default (no damping gap). Stage 2: policy loop from default.
    Mirrors the h1_2 example's posture->behavior pattern. exit_mode: 'damp' (default,
    proven ramp-to-damping) or 'stand' (opt-in clean-completion handoff, guarded upstream)."""
    if not watch:
        raise SystemExit("REFUSED: pass --i-will-watch-the-robot")
    _require_human("run")
    make_dds(iface)
    sub = lowstate_subscriber()
    q0, _, _, _, msg0 = read_state(sub)
    mode_machine = int(msg0.mode_machine)
    _release_motion_service()
    pub, low_cmd, crc = _lowcmd_setup()
    global _DAMP_CTX
    _DAMP_CTX = (pub, low_cmd, crc, mode_machine, meta)
    _install_damp_on_signals()
    dt = 1.0 / CONTROL_HZ
    kp_a, kd_a = meta.kp * APPROACH_KP_SCALE, meta.kd * APPROACH_KP_SCALE
    n_ticks = ref.T if not max_secs else min(ref.T, int(max_secs * CONTROL_HZ))
    print(f"RUN: stage-1 firm move-to-default (4s) + hold, then policy {n_ticks}/{ref.T} ticks "
          f"@ {CONTROL_HZ:.0f}Hz{' [--max-secs %.1f]' % max_secs if max_secs else ''}. "
          f"Ctrl-C / remote-damp to stop.")
    global _TELEM
    telem = _TELEM = Telemetry("run", meta)
    _acap = action_cap_vector(meta, MAX_ACTION)
    last_action = np.zeros(meta.n)
    last_target = meta.default.copy()
    try:
        # STAGE 1 — reach the ready pose at firm gains, seamlessly (no damping gap).
        _ramp_to_pose(pub, low_cmd, crc, mode_machine, q0, meta.default, 4.0, kp_a, kd_a, meta)
        _hold(pub, low_cmd, crc, mode_machine, meta.default, 0.6, kp_a, kd_a, meta)
        # Robot is at the ready pose — align the reference world yaw to its heading NOW.
        q, dq, imu_quat, gyro, _ = read_state(sub, timeout_s=0.5)
        _align_reference(meta, ref, q, imu_quat)
        print("at default — starting policy. (Legs may look odd on the gantry: the policy "
              "trained with ground contact. Watch for fault/violence; arms should track.)")
        # STAGE 2 — policy loop at TRAINED gains. Robot is already AT default and the
        # ramped motion (thriller_deploy) starts from default -> no lurch on entry.
        for tick in range(n_ticks):
            t0 = time.time()
            q, dq, imu_quat, gyro, msg = read_state(sub, timeout_s=0.5)
            obs, _ = build_obs(meta, ref, q, dq, imu_quat, gyro, last_action, tick)
            if not np.all(np.isfinite(obs)):
                raise RuntimeError(f"non-finite obs at tick {tick}")
            action = run_policy(session, obs, tick)
            if not np.all(np.isfinite(action)) or np.any(np.abs(action) > _acap):
                j = int(np.abs(np.asarray(action) / _acap).argmax())
                raise RuntimeError(f"bad action at tick {tick} ({meta.joint_order[j]} "
                                   f"|a|={abs(action[j]):.2f} > cap {_acap[j]:.1f})")
            last_action = action
            last_target = action_to_target(meta, action)
            _send_cmd(pub, low_cmd, crc, mode_machine, last_target, meta.kp, meta.kd, meta)
            telem.add(tick, q, dq, msg, imu_quat, gyro, action, last_target)
            elapsed = time.time() - t0
            if elapsed > 2 * dt:
                raise RuntimeError(f"cycle overrun {elapsed*1000:.0f}ms -> damp")
            time.sleep(max(0.0, dt - elapsed))
        # CLEAN full completion. OPT-IN: hand off STANDING instead of damping (guarded).
        if exit_mode == "stand":
            _stand_handoff_and_exit(pub, low_cmd, crc, mode_machine, meta, ref)
        # NORMAL end (or --max-secs reached): smooth ramp-down — fade kp->0 holding pose.
        print("policy segment done — smooth ramp to damping.")
        steps = int(0.6 * CONTROL_HZ)
        for s in range(steps + 1):
            f = 1.0 - s / steps
            _send_cmd(pub, low_cmd, crc, mode_machine, last_target, meta.kp * f, meta.kd, meta)
            time.sleep(dt)
    except BaseException as e:  # noqa: BLE001 - ANY failure -> immediate damp
        print(f"\nSTOP: {e} -> damping")
    finally:
        _damp(pub, low_cmd, crc, mode_machine, meta, secs=1.0)
    _finalize_and_exit(0)   # guarantee soft + exit promptly (DDS teardown can hang)


def _stand_hold_targets(meta, trim_deg=None):
    """Ready-pose hold targets with a constant ANKLE_TRIM_DEG bias on BOTH ankle_pitch
    joints (stand-hold ONLY — the audit §4 exp #6 posture->torque->heat sweep).
    Returns (targets[29], applied_trim_deg); trim clamped to ±ANKLE_TRIM_MAX_DEG."""
    trim = ANKLE_TRIM_DEG if trim_deg is None else float(trim_deg)
    trim = float(np.clip(trim, -ANKLE_TRIM_MAX_DEG, ANKLE_TRIM_MAX_DEG))
    tgt = meta.default.astype(float).copy()
    for i in ANKLE_PITCH_IDX:
        tgt[i] += np.deg2rad(trim)
    return tgt, trim


def mode_stand_hold(meta, iface, watch, secs):
    """GROUND stage A (no policy): firm move-to-default, then HOLD the ready pose
    standing (tethered) indefinitely until Ctrl-C / remote-damp. Pure PD — proves the
    robot can hold the stance and lets the human judge stability before any policy runs.
    Always ends soft (Ctrl-C/SIGTERM/crash -> damp)."""
    if not watch:
        raise SystemExit("REFUSED: pass --i-will-watch-the-robot")
    _require_human("stand-hold")
    make_dds(iface)
    sub = lowstate_subscriber()
    q0, _, _, _, msg0 = read_state(sub)
    mode_machine = int(msg0.mode_machine)
    _release_motion_service()
    pub, low_cmd, crc = _lowcmd_setup()
    global _DAMP_CTX
    _DAMP_CTX = (pub, low_cmd, crc, mode_machine, meta)
    _install_damp_on_signals()
    kp, kd = meta.kp * APPROACH_KP_SCALE, meta.kd * APPROACH_KP_SCALE  # firm, overdamped
    hold_target, trim_deg = _stand_hold_targets(meta)
    if trim_deg != 0.0:
        print("!" * 72)
        print(f"!! ANKLE_TRIM_DEG={trim_deg:+.1f} deg — BOTH ankle_pitch hold targets biased "
              f"(clamp ±{ANKLE_TRIM_MAX_DEG:.0f}).")
        print("!! Posture->torque->heat sweep (audit §4 exp #6). Ramp AND hold use the "
              "trimmed pose.")
        print("!" * 72)
    print(f"STAND-HOLD: firm move-to-default over {secs:.1f}s, then hold indefinitely "
          f"(approach gains {APPROACH_KP_SCALE:.1f}x). Ctrl-C / remote-damp to stop.")
    global _TELEM
    telem = _TELEM = Telemetry("stand-hold", meta, extra={"ankle_trim_deg": trim_deg})
    try:
        _ramp_to_pose(pub, low_cmd, crc, mode_machine, q0, hold_target, secs, kp, kd, meta)
        print("at default — HOLDING. Watch stance; damp when done.")
        tick = 0
        while True:  # SIGINT/SIGTERM handler damps + exits; this loop just streams the hold
            _send_cmd(pub, low_cmd, crc, mode_machine, hold_target, kp, kd, meta)
            # Record the hold (tau_est/temps/pose): this is the exact test class that
            # produced the unauditable '20 Nm continuous' number — now it's captured.
            q, dq, imu_quat, gyro, msg = read_state(sub, timeout_s=0.5)
            telem.add(tick, q, dq, msg, imu_quat, gyro, np.zeros(meta.n), hold_target)
            tick += 1
            time.sleep(1.0 / CONTROL_HZ)
    except BaseException as e:  # noqa: BLE001 - ANY failure -> immediate damp
        print(f"\nSTOP: {e} -> damping")
    finally:
        _damp(pub, low_cmd, crc, mode_machine, meta, secs=1.0)
    _finalize_and_exit(0)


def mode_ground_run(meta, session, ref, iface, watch, max_secs, obs_order, exit_mode="damp"):
    """GROUND stage B: firm move-to-default, then run the ESTIMATOR-FREE ground policy
    for a short capped segment. Same safety spine as mode_run but with build_obs_ground
    (no fabricated estimator terms) and the conservative GROUND_MAX_ACTION cap.
    Requires --max-secs (no unbounded ground runs while bringing this up). exit_mode:
    'damp' (default) or 'stand' (opt-in clean-completion handoff, guarded upstream)."""
    if not watch:
        raise SystemExit("REFUSED: pass --i-will-watch-the-robot")
    if not max_secs or max_secs <= 0:
        raise SystemExit("REFUSED: ground-run requires --max-secs > 0 (cautious capped segment)")
    _require_human("ground-run")
    make_dds(iface)
    sub = lowstate_subscriber()
    q0, _, quat0, _, msg0 = read_state(sub)
    mode_machine = int(msg0.mode_machine)
    _check_start_upright(quat0)   # refuse a non-upright start BEFORE releasing onboard (stays safe)
    # ENTRY HANDOFF: pre-arm the publisher + damp context + signal handler BEFORE releasing
    # onboard, so there is zero setup latency in the unheld release window (fall risk untethered).
    pub, low_cmd, crc = _lowcmd_setup()
    global _DAMP_CTX
    _DAMP_CTX = (pub, low_cmd, crc, mode_machine, meta)
    _install_damp_on_signals()
    dt = 1.0 / CONTROL_HZ
    kp_a, kd_a = meta.kp * APPROACH_KP_SCALE, meta.kd * APPROACH_KP_SCALE
    _release_motion_service()
    # Catch the CURRENT pose the instant onboard lets go (no unheld sag), then the ramp below
    # eases from that same pose to the ready pose. Holds q0 at firm approach gains.
    if ENTRY_CATCH_S > 0:
        print(f"   entry catch: holding current pose {ENTRY_CATCH_S:.1f}s so the robot is never "
              f"unheld at the onboard->policy handoff, then easing to the ready pose.")
        _hold(pub, low_cmd, crc, mode_machine, q0, ENTRY_CATCH_S, kp_a, kd_a, meta)
    n_ticks = min(ref.T, int(max_secs * CONTROL_HZ))
    obs_dim = sum(w for _, w in obs_order)
    print(f"GROUND-RUN: stage-1 firm move-to-default (4s)+hold, then estimator-free policy "
          f"({obs_dim}-dim obs) {n_ticks}/{ref.T} ticks @ {CONTROL_HZ:.0f}Hz "
          f"[--max-secs {max_secs:.1f}], action cap {GROUND_MAX_ACTION:.1f}. "
          f"Tethered. Ctrl-C / remote-damp to stop.")
    global _TELEM
    telem = _TELEM = Telemetry("ground-run", meta)
    _acap = action_cap_vector(meta, GROUND_MAX_ACTION)
    last_action = np.zeros(meta.n)
    last_target = meta.default.copy()
    try:
        _ramp_to_pose(pub, low_cmd, crc, mode_machine, q0, meta.default, 4.0, kp_a, kd_a, meta)
        _hold(pub, low_cmd, crc, mode_machine, meta.default, 0.6, kp_a, kd_a, meta)
        # Robot is at the ready pose — align the reference world yaw to its heading NOW.
        q, dq, imu_quat, gyro, _ = read_state(sub, timeout_s=0.5)
        _align_reference(meta, ref, q, imu_quat)
        print("at default — starting ground policy. Keep tension on the tether; damp at "
              "the first sign of a fault, lurch, or lean.")
        for tick in range(n_ticks):
            t0 = time.time()
            q, dq, imu_quat, gyro, msg = read_state(sub, timeout_s=0.5)
            obs, _ = build_obs_ground(meta, ref, q, dq, imu_quat, gyro, last_action, tick, obs_order)
            if not np.all(np.isfinite(obs)):
                raise RuntimeError(f"non-finite obs at tick {tick}")
            action = run_policy(session, obs, tick)
            if not np.all(np.isfinite(action)) or np.any(np.abs(action) > _acap):
                j = int(np.abs(np.asarray(action) / _acap).argmax())
                raise RuntimeError(f"bad action at tick {tick} ({meta.joint_order[j]} "
                                   f"|a|={abs(action[j]):.2f} > cap {_acap[j]:.1f})")
            last_action = action
            last_target = action_to_target(meta, action)
            _send_cmd(pub, low_cmd, crc, mode_machine, last_target, meta.kp, meta.kd, meta)
            telem.add(tick, q, dq, msg, imu_quat, gyro, action, last_target)
            elapsed = time.time() - t0
            if elapsed > 2 * dt:
                raise RuntimeError(f"cycle overrun {elapsed*1000:.0f}ms -> damp")
            time.sleep(max(0.0, dt - elapsed))
        # CLEAN full completion. OPT-IN: hand off STANDING instead of damping (guarded).
        if exit_mode == "stand":
            _stand_handoff_and_exit(pub, low_cmd, crc, mode_machine, meta, ref)
        print("ground segment done — smooth ramp to damping.")
        steps = int(0.6 * CONTROL_HZ)
        for s in range(steps + 1):
            f = 1.0 - s / steps
            _send_cmd(pub, low_cmd, crc, mode_machine, last_target, meta.kp * f, meta.kd, meta)
            time.sleep(dt)
    except BaseException as e:  # noqa: BLE001 - ANY failure -> immediate damp
        print(f"\nSTOP: {e} -> damping")
    finally:
        _damp(pub, low_cmd, crc, mode_machine, meta, secs=1.0)
    _finalize_and_exit(0)


def mode_ground_run_odom(meta, session, ref, iface, watch, max_secs, exit_mode="damp"):
    """GROUND stage B (ODOMETRY-FED): run the PROVEN full-obs gantry policy on the
    ground, building the two estimator-dependent obs terms HONESTLY from the onboard
    estimate (rt/odommodestate) instead of faking them. Same safety spine + conservative
    GROUND_MAX_ACTION cap as mode_ground_run; --max-secs required; always ends soft.

    This is the path the odometry finding (2026-07-04) opened: the gantry policy is 100%
    in sim, and the only reason it could not go to the ground was the missing torso
    pose/velocity — which the robot publishes. No estimator-free retrain needed.
    """
    if not watch:
        raise SystemExit("REFUSED: pass --i-will-watch-the-robot")
    if not max_secs or max_secs <= 0:
        raise SystemExit("REFUSED: ground-run-odom requires --max-secs > 0 (cautious capped segment)")
    _require_human("ground-run-odom")
    make_dds(iface)
    sub = lowstate_subscriber()
    odom = odom_subscriber()
    # NO-GO if the estimate is not actually flowing — never fall back to fabricated terms.
    o0 = read_odom(odom, timeout_s=1.0)
    if o0 is None:
        raise SystemExit(f"REFUSED: no {ODOM_TOPIC} within 1s — the onboard estimate this "
                         "mode depends on is not being published. NO-GO.")
    q0, _, _, _, msg0 = read_state(sub)
    mode_machine = int(msg0.mode_machine)
    _release_motion_service()
    pub, low_cmd, crc = _lowcmd_setup()
    global _DAMP_CTX
    _DAMP_CTX = (pub, low_cmd, crc, mode_machine, meta)
    _install_damp_on_signals()
    dt = 1.0 / CONTROL_HZ
    kp_a, kd_a = meta.kp * APPROACH_KP_SCALE, meta.kd * APPROACH_KP_SCALE
    n_ticks = min(ref.T, int(max_secs * CONTROL_HZ))
    print(f"GROUND-RUN-ODOM: stage-1 firm move-to-default (4s)+hold, then PROVEN gantry "
          f"policy (160-dim HONEST obs from {ODOM_TOPIC}, vel_src={ODOM_VEL_SOURCE}) "
          f"{n_ticks}/{ref.T} ticks @ {CONTROL_HZ:.0f}Hz [--max-secs {max_secs:.1f}], "
          f"action cap {GROUND_MAX_ACTION:.1f}. Tethered. Ctrl-C / remote-damp to stop.")
    global _TELEM
    telem = _TELEM = Telemetry("ground-run-odom", meta)
    _acap = action_cap_vector(meta, GROUND_MAX_ACTION)
    last_action = np.zeros(meta.n)
    last_target = meta.default.copy()
    odom_pos0 = None
    prev_pos, prev_t = None, None
    try:
        _ramp_to_pose(pub, low_cmd, crc, mode_machine, q0, meta.default, 4.0, kp_a, kd_a, meta)
        _hold(pub, low_cmd, crc, mode_machine, meta.default, 0.6, kp_a, kd_a, meta)
        # Robot is at the ready pose — align the reference world yaw to its heading NOW.
        q, dq, imu_quat, gyro, _ = read_state(sub, timeout_s=0.5)
        _align_reference(meta, ref, q, imu_quat)
        # Capture the re-anchor origin at policy start (robot is now at the reference pose).
        o = read_odom(odom, timeout_s=0.5)
        if o is None:
            raise RuntimeError("lost odom at policy start -> damp")
        odom_pos0 = o[0].copy()
        prev_pos, prev_t = o[0].copy(), time.time()
        prev_stamp, stale = o[2], 0
        print("at default — starting odometry-fed policy. Keep tension on the tether; "
              "damp at the first sign of a fault, lurch, or lean.")
        for tick in range(n_ticks):
            t0 = time.time()
            q, dq, imu_quat, gyro, msg = read_state(sub, timeout_s=0.5)
            o = read_odom(odom, timeout_s=0.5)
            if o is None:
                raise RuntimeError(f"lost {ODOM_TOPIC} at tick {tick} -> damp")
            pos, vel_field, stamp = o
            # FROZEN-estimate guard: topic still arriving but the estimator stalled (stamp
            # not advancing) -> the obs would go stale and fly blind. Damp after ~5 ticks.
            if stamp <= prev_stamp:
                stale += 1
                if stale >= 5:
                    raise RuntimeError(f"{ODOM_TOPIC} stamp frozen {stale} ticks "
                                       f"(estimator stalled) -> damp")
            else:
                stale = 0
            prev_stamp = stamp
            robot_disp = pos - odom_pos0
            if ODOM_VEL_SOURCE == "field":
                v_world = vel_field   # ASSUMES field is world-frame; validate before use
            else:  # "diff" — frame-unambiguous world velocity from position differencing
                pdt = max(1e-3, t0 - prev_t)
                v_world = (pos - prev_pos) / pdt
            prev_pos, prev_t = pos.copy(), t0
            obs, _ = build_obs_odom(meta, ref, q, dq, imu_quat, gyro, last_action, tick,
                                    robot_disp, v_world)
            if not np.all(np.isfinite(obs)):
                raise RuntimeError(f"non-finite obs at tick {tick}")
            action = run_policy(session, obs, tick)
            if not np.all(np.isfinite(action)) or np.any(np.abs(action) > _acap):
                j = int(np.abs(np.asarray(action) / _acap).argmax())
                raise RuntimeError(f"bad action at tick {tick} ({meta.joint_order[j]} "
                                   f"|a|={abs(action[j]):.2f} > cap {_acap[j]:.1f})")
            last_action = action
            last_target = action_to_target(meta, action)
            _send_cmd(pub, low_cmd, crc, mode_machine, last_target, meta.kp, meta.kd, meta)
            telem.add(tick, q, dq, msg, imu_quat, gyro, action, last_target)
            elapsed = time.time() - t0
            if elapsed > 2 * dt:
                raise RuntimeError(f"cycle overrun {elapsed*1000:.0f}ms -> damp")
            time.sleep(max(0.0, dt - elapsed))
        # CLEAN full completion. OPT-IN: hand off STANDING instead of damping (guarded).
        if exit_mode == "stand":
            _stand_handoff_and_exit(pub, low_cmd, crc, mode_machine, meta, ref)
        print("ground segment done — smooth ramp to damping.")
        steps = int(0.6 * CONTROL_HZ)
        for s in range(steps + 1):
            f = 1.0 - s / steps
            _send_cmd(pub, low_cmd, crc, mode_machine, last_target, meta.kp * f, meta.kd, meta)
            time.sleep(dt)
    except BaseException as e:  # noqa: BLE001 - ANY failure -> immediate damp
        print(f"\nSTOP: {e} -> damping")
    finally:
        _damp(pub, low_cmd, crc, mode_machine, meta, secs=1.0)
    _finalize_and_exit(0)


def _arm_joint_indices(joint_order):
    """The 14 arm joints, matched BY NAME (shoulder/elbow/wrist) — never positional."""
    return [i for i, n in enumerate(joint_order)
            if ("shoulder" in n) or ("elbow" in n) or ("wrist" in n)]


def _arm_boost_gains(meta, kp, kd, scale=None):
    """Apply the ARM_GROUND_KP_SCALE boost to policy-phase gains (copies).

    Returns (kp, kd, arm_idx); arm_idx is [] when no boost is applied. kd scales
    by the SAME factor as kp (see the knob comment: overdamped-ness rises with
    sqrt(scale), and it matches the V3B retrain's train-time actuator scaling).
    Refuses scales outside [1.0, ARM_GROUND_KP_SCALE_MAX] — above ~3x the wrist
    kp leaves the teleop-proven envelope on these motors.
    """
    s = ARM_GROUND_KP_SCALE if scale is None else float(scale)
    if s == 1.0:
        return kp, kd, []
    if not (1.0 <= s <= ARM_GROUND_KP_SCALE_MAX):
        raise SystemExit(
            f"REFUSED: ARM_GROUND_KP_SCALE={s:g} outside [1.0, "
            f"{ARM_GROUND_KP_SCALE_MAX:g}] (teleop-proven arm envelope).")
    idx = _arm_joint_indices(meta.joint_order)
    if len(idx) != 14:
        raise SystemExit(
            f"REFUSED: expected 14 arm joints in joint_order, found {len(idx)}")
    kp, kd = np.asarray(kp, float).copy(), np.asarray(kd, float).copy()
    kp[idx] *= s
    kd[idx] *= s
    return kp, kd, idx


def mode_ground_run_legodom(meta, session, ref, iface, watch, max_secs, exit_mode="damp"):
    """GROUND stage B (KINEMATIC ODOMETRY): run the PROVEN gantry policy on the ground with
    base_lin_vel + torso-height feedback estimated from the LEGS (joint q/dq + IMU), which
    is fully under our control — unlike rt/odommodestate, which FREEZES when the motion
    service is released (confirmed on hardware 2026-07-04). Validated offline: leg-odom
    base_lin_vel is within the policy's ±0.5 m/s trained band on 97.8% of frames.

    obs: real base_lin_vel (leg odom); motion_anchor_pos_b = real height error (leg-odom
    Z change) with XY under the tracking assumption (drift-free, correct for an IN-PLACE
    dance). Same safety spine + GROUND_MAX_ACTION cap; --max-secs required; always soft.
    """
    if not watch:
        raise SystemExit("REFUSED: pass --i-will-watch-the-robot")
    if not max_secs or max_secs <= 0:
        raise SystemExit("REFUSED: ground-run-legodom requires --max-secs > 0")
    _require_human("ground-run-legodom")
    from pipeline.leg_odometry import LegOdometry
    legodom = LegOdometry(list(meta.joint_order))
    legodom.reset_filter()  # clear the velocity smoother so no stale value leaks in
    make_dds(iface)
    sub = lowstate_subscriber()
    q0, _, quat0, _, msg0 = read_state(sub)
    mode_machine = int(msg0.mode_machine)
    _check_start_upright(quat0)   # refuse a non-upright start BEFORE releasing onboard (stays safe)
    # ENTRY HANDOFF: pre-arm the publisher + damp context + signal handler BEFORE releasing
    # onboard, so there is zero setup latency in the unheld release window (fall risk untethered).
    pub, low_cmd, crc = _lowcmd_setup()
    global _DAMP_CTX
    _DAMP_CTX = (pub, low_cmd, crc, mode_machine, meta)
    _install_damp_on_signals()
    dt = 1.0 / CONTROL_HZ
    kp_a, kd_a = meta.kp * APPROACH_KP_SCALE, meta.kd * APPROACH_KP_SCALE
    _release_motion_service()
    # Catch the CURRENT pose the instant onboard lets go (no unheld sag), then the ramp below
    # eases from that same pose to the ready pose. Holds q0 at firm approach gains.
    if ENTRY_CATCH_S > 0:
        print(f"   entry catch: holding current pose {ENTRY_CATCH_S:.1f}s so the robot is never "
              f"unheld at the onboard->policy handoff, then easing to the ready pose.")
        _hold(pub, low_cmd, crc, mode_machine, q0, ENTRY_CATCH_S, kp_a, kd_a, meta)
    n_ticks = min(ref.T, int(max_secs * CONTROL_HZ))
    print(f"GROUND-RUN-LEGODOM: stage-1 firm move-to-default (4s)+hold, then PROVEN gantry "
          f"policy (160-dim obs, base_lin_vel+height from LEG kinematics — service-independent) "
          f"{n_ticks}/{ref.T} ticks @ {CONTROL_HZ:.0f}Hz [--max-secs {max_secs:.1f}], "
          f"action cap {GROUND_MAX_ACTION:.1f}. Tethered. Ctrl-C / remote-damp to stop.")
    last_action = np.zeros(meta.n)
    last_target = meta.default.copy()
    fall_ticks = 0                     # consecutive-tick fall-condition counter (debounce)
    # Policy-phase gains: boost the LEGS so they can bear weight and hold standing while the
    # arms dance at their trained gains. GROUND_LEG_KP_SCALE=1.0 -> unchanged (old behaviour).
    kp_pol, kd_pol = meta.kp.astype(float).copy(), meta.kd.astype(float).copy()
    kp_pol[LEG_JOINT_IDX] *= GROUND_LEG_KP_SCALE
    kd_pol[LEG_JOINT_IDX] *= GROUND_LEG_KP_SCALE
    if GROUND_LEG_KP_SCALE != 1.0:
        print(f"   LEG gains x{GROUND_LEG_KP_SCALE:.1f} during policy (weight-bearing); arms unchanged.")
    kp_pol, kd_pol, _arm_idx = _arm_boost_gains(meta, kp_pol, kd_pol)
    if _arm_idx:
        print(f"   *** ARM gains x{ARM_GROUND_KP_SCALE:.2f} during policy (kp AND kd, "
              f"{len(_arm_idx)} joints by name): shoulder/elbow kp -> "
              f"{kp_pol[_arm_idx[0]]:.1f}, wrist_pitch/yaw kp -> {kp_pol[_arm_idx[-1]]:.1f} "
              f"(teleop-proven 80/40). Legs/waist untouched. "
              f"A V3B-retrained policy REQUIRES 2.5 here. ***")
    if GRAVITY_FF:
        print(f"   GRAVITY FEEDFORWARD on (x{GRAVITY_FF_SCALE:.2f}) — EXPERIMENTAL "
              f"(see audit note at the GRAVITY_FF flag).")
    global _TELEM
    telem = _TELEM = Telemetry("ground-run-legodom", meta)
    _acap = action_cap_vector(meta, GROUND_MAX_ACTION)
    try:
        _ramp_to_pose(pub, low_cmd, crc, mode_machine, q0, meta.default, 4.0, kp_a, kd_a, meta)
        _hold(pub, low_cmd, crc, mode_machine, meta.default, 0.6, kp_a, kd_a, meta)
        q, dq, imu_quat, gyro, _ = read_state(sub, timeout_s=0.5)
        # Robot is at the ready pose — align the reference world yaw to its heading NOW.
        _align_reference(meta, ref, q, imu_quat)
        h0 = legodom.estimate(q, dq, quat_wxyz_to_mat(imu_quat), gyro)[1]  # start height datum
        print("at default — starting leg-odometry policy. Keep tension on the tether; "
              "damp at the first sign of a fault, lurch, or lean.")
        for tick in range(n_ticks):
            t0 = time.time()
            q, dq, imu_quat, gyro, msg = read_state(sub, timeout_s=0.5)
            R_base = quat_wxyz_to_mat(imu_quat)
            v_body, h_est, _ = legodom.estimate(q, dq, R_base, gyro)
            v_world = R_base @ v_body
            ref_disp = ref.at(tick)[2] - ref.apos[0]
            # debounced physical-state fall trigger (topple OR choreography-relative height
            # collapse) -> on a confirmed trip, damp + soft handoff to onboard
            fall_ticks = _check_fall(fall_ticks, R_base, h_est, h0, ref_disp[2], tick)
            # XY: tracking assumption (in-place dance) -> anchor XY ~0, drift-free.
            # Z: real height change from leg odom (bias cancels in the displacement).
            robot_disp = np.array([ref_disp[0], ref_disp[1], h_est - h0])
            obs, _ = build_obs_odom(meta, ref, q, dq, imu_quat, gyro, last_action, tick,
                                    robot_disp, v_world)
            if not np.all(np.isfinite(obs)):
                raise RuntimeError(f"non-finite obs at tick {tick}")
            action = run_policy(session, obs, tick)
            if not np.all(np.isfinite(action)) or np.any(np.abs(action) > _acap):
                j = int(np.abs(np.asarray(action) / _acap).argmax())
                raise RuntimeError(f"bad action at tick {tick} ({meta.joint_order[j]} "
                                   f"|a|={abs(action[j]):.2f} > cap {_acap[j]:.1f})")
            last_action = action
            last_target = action_to_target(meta, action)
            # Gravity-comp feedforward at the COMMANDED pose, in the current torso frame (IMU),
            # so the legs hold the pose at trained gains without the ankle-cooking gain boost.
            tau_ff = (GRAVITY_FF_SCALE * legodom.gravity_comp(last_target, R_base)
                      if GRAVITY_FF else None)
            _send_cmd(pub, low_cmd, crc, mode_machine, last_target, kp_pol, kd_pol, meta, tau_ff=tau_ff)
            telem.add(tick, q, dq, msg, imu_quat, gyro, action, last_target)
            elapsed = time.time() - t0
            if elapsed > 2 * dt:
                raise RuntimeError(f"cycle overrun {elapsed*1000:.0f}ms -> damp")
            time.sleep(max(0.0, dt - elapsed))
        # CLEAN full completion. OPT-IN: hand off STANDING instead of damping. Hold at the
        # SAME boosted policy-phase gains (kp_pol/kd_pol) just used so there is no gain
        # discontinuity/sag during the handoff. Guarded upstream (motion ends standing).
        if exit_mode == "stand":
            _stand_handoff_and_exit(pub, low_cmd, crc, mode_machine, meta, ref, kp_pol, kd_pol)
        print("ground segment done — smooth ramp to damping.")
        steps = int(0.6 * CONTROL_HZ)
        for s in range(steps + 1):
            f = 1.0 - s / steps
            _send_cmd(pub, low_cmd, crc, mode_machine, last_target, meta.kp * f, meta.kd, meta)
            time.sleep(dt)
    except BaseException as e:  # noqa: BLE001 - ANY failure -> immediate damp
        print(f"\nSTOP: {e} -> damping")
    finally:
        _damp(pub, low_cmd, crc, mode_machine, meta, secs=1.0)
    _finalize_and_exit(0)


def _build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["read", "move-to-default", "run", "stand-hold", "ground-run",
                             "ground-run-odom", "ground-run-legodom"],
                    default="read")
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--motion-npz", default=str(DEFAULT_MOTION))
    ap.add_argument("--policy", default=str(DEFAULT_POLICY))
    ap.add_argument("--iface", default=IFACE)
    ap.add_argument("--timeout-s", type=float, default=2.0)
    ap.add_argument("--secs", type=float, default=4.0, help="move-to-default duration")
    ap.add_argument("--max-secs", type=float, default=None,
                    help="run/ground-run: cap the policy segment to this many seconds "
                         "(cautious test), then smooth-ramp to damping. Required for ground-run.")
    ap.add_argument("--exit", dest="exit_mode", choices=["damp", "stand"], default="damp",
                    help="end-of-run handoff on a CLEAN full completion: 'damp' (DEFAULT, "
                         "proven smooth ramp-to-damping) or 'stand' (OPT-IN: hold the final "
                         "standing pose, hand back to onboard balance while balanced — removes "
                         "the catch-step; guarded to standing motions; UNVALIDATED on hardware). "
                         "Every abort/fault still damps immediately regardless of this flag.")
    ap.add_argument("--ground-meta", default=str(GROUND_META))
    ap.add_argument("--ground-policy", default=str(GROUND_POLICY))
    ap.add_argument("--ground-motion", default=str(GROUND_MOTION))
    ap.add_argument("--i-will-watch-the-robot", action="store_true",
                    help="required for any motion mode; you are watching, remote in hand")
    return ap


def main():
    a = _build_parser().parse_args()

    import onnxruntime as ort

    # stand-hold is pure PD — no policy needed. Uses the standard meta (default pose).
    if a.mode == "stand-hold":
        return mode_stand_hold(Meta(Path(a.meta)), a.iface, a.i_will_watch_the_robot, a.secs) or 0

    # ground-run uses the ESTIMATOR-FREE ground policy. Fail-safe if it isn't there yet.
    if a.mode == "ground-run":
        gm, gp = Path(a.ground_meta), Path(a.ground_policy)
        gmot = Path(a.ground_motion)
        missing = [str(p) for p in (gm, gp) if not p.exists()]
        if missing:
            raise SystemExit(
                "REFUSED: ground policy artifacts not found: " + ", ".join(missing) +
                "\nThe obs-restricted retrain has not landed. Do NOT substitute the "
                "gantry policy — its obs needs a state estimator and would fall on the "
                "ground. Wait for data/policies/thriller_ground/.")
        if not gmot.exists():   # motion may be shared with the gantry export
            gmot = Path(a.motion_npz)
        gmeta = Meta(gm)
        obs_order = _ground_obs_order(gmeta)   # raises if not estimator-free
        gref = Reference(gmot)
        gsession = ort.InferenceSession(str(gp), providers=["CPUExecutionProvider"])
        # --exit stand honored only if this ground motion ends standing (else -> damp).
        gexit = _resolve_exit_mode(a.exit_mode, gmeta, gref)
        return mode_ground_run(gmeta, gsession, gref, a.iface, a.i_will_watch_the_robot,
                               a.max_secs, obs_order, gexit) or 0

    meta = Meta(Path(a.meta))
    ref = Reference(Path(a.motion_npz))
    session = ort.InferenceSession(a.policy, providers=["CPUExecutionProvider"])

    if a.mode == "read":
        return mode_read(meta, ref, session, a.iface, a.timeout_s)
    if a.mode == "move-to-default":
        return mode_move_to_default(meta, session, ref, a.iface, a.secs, a.i_will_watch_the_robot) or 0
    # Resolve --exit ONCE for the policy motion modes: 'stand' is honored only when the
    # loaded motion ends standing (the guard), otherwise it falls back to 'damp'.
    exit_mode = _resolve_exit_mode(a.exit_mode, meta, ref)
    if a.mode == "run":
        return mode_run(meta, session, ref, a.iface, a.i_will_watch_the_robot, a.max_secs,
                        exit_mode) or 0
    if a.mode == "ground-run-odom":
        # PROVEN gantry policy (meta/ref/session above) + honest odometry-fed obs.
        return mode_ground_run_odom(meta, session, ref, a.iface, a.i_will_watch_the_robot,
                                    a.max_secs, exit_mode) or 0
    if a.mode == "ground-run-legodom":
        # PROVEN gantry policy + base_lin_vel/height from LEG kinematics (service-independent).
        return mode_ground_run_legodom(meta, session, ref, a.iface, a.i_will_watch_the_robot,
                                       a.max_secs, exit_mode) or 0


if __name__ == "__main__":
    sys.exit(main() or 0)
