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

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_META = ROOT / "data/policies/thriller/policy_meta.json"
DEFAULT_MOTION = ROOT / "data/policies/thriller/thriller_deploy.npz"
DEFAULT_POLICY = ROOT / "data/policies/thriller/policy.onnx"
IFACE = "enp0s31f6"
TORSO_NPZ_IDX = 15          # torso_link: mjlab body 16 minus the dropped world body
CONTROL_HZ = 50.0
# Firm approach gains for reaching the ready pose (verified on hardware 2026-07-05):
# scale BOTH kp and kd so the joint stays overdamped. Env-overridable.
APPROACH_KP_SCALE = float(os.environ.get("APPROACH_KP_SCALE", "2.0"))
MAX_ACTION = float(os.environ.get("MAX_ACTION", "8.0"))  # |action|>this -> damp; gantry legs spike, env-tunable

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
# Falls are unforgiving on the ground; start well below the gantry cap and tune up
# only after a clean tethered segment. Env-overridable.
GROUND_MAX_ACTION = float(os.environ.get("GROUND_MAX_ACTION", "6.0"))
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
    msg = sub.Read(int(timeout_s * 1000))
    if msg is None:
        raise SystemExit(f"no LowState within {timeout_s}s — robot off / wrong iface / LAN down. NO-GO.")
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
    """Return (pos[3], vel_field[3]) from rt/odommodestate, or None if not received.

    Non-fatal by design: the caller decides. A ground run that needs odometry must
    treat None as NO-GO (don't fall back to fabricated terms mid-run)."""
    msg = sub.Read(int(timeout_s * 1000))
    if msg is None:
        return None
    return np.array(list(msg.position), float), np.array(list(msg.velocity), float)


# ---- observation builder (real robot state -> 160-D mjlab obs) -----------------
def build_obs(meta: Meta, ref: Reference, q, dq, imu_quat, gyro, last_action, tick):
    ref_jp, ref_jv, ref_apos, ref_aquat = ref.at(tick)
    ref_apos0 = ref.apos[0]
    R_rob = quat_wxyz_to_mat(imu_quat)   # robot torso orientation from IMU (pelvis~torso approx)
    terms = {
        "command": np.concatenate([ref_jp, ref_jv]),                    # 58
        # gantry approx: robot torso pos ~= reference start -> displacement of ref in robot frame
        "motion_anchor_pos_b": R_rob.T @ (ref_apos - ref_apos0),        # 3
        "motion_anchor_ori_b": mat_first_two_cols_b(ref_aquat, imu_quat),  # 6
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
    R_rob = quat_wxyz_to_mat(imu_quat)
    terms = {
        "command": np.concatenate([ref_jp, ref_jv]),                    # 58
        "motion_anchor_pos_b": R_rob.T @ (ref_disp - robot_disp),       # 3 (HONEST)
        "motion_anchor_ori_b": mat_first_two_cols_b(ref_aquat, imu_quat),  # 6
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
    terms = {
        "command": np.concatenate([ref_jp, ref_jv]),                       # 58
        "motion_anchor_ori_b": mat_first_two_cols_b(ref_aquat, imu_quat),  # 6 (IMU-only)
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
    last_action = np.zeros(meta.n)
    obs, terms = build_obs(meta, ref, q, dq, imu_quat, gyro, last_action, tick=0)

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
    status, result = msc.CheckMode()
    tries = 0
    while result.get("name"):
        msc.ReleaseMode()
        status, result = msc.CheckMode()
        time.sleep(1)
        tries += 1
        if tries > 10:
            raise SystemExit("could not release motion service after 10 tries — abort")
    print("   motion service released — rt/lowcmd accepted for full-body.")


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


def _send_cmd(pub, low_cmd, crc, mode_machine, targets, kp, kd, meta, damping=False):
    """Mutate the REUSED low_cmd and publish. damping=True -> hold, kp=0, small kd."""
    low_cmd.mode_pr = PR_MODE
    low_cmd.mode_machine = mode_machine
    for i in range(29):
        mc = low_cmd.motor_cmd[i]
        mc.mode = 1          # enable
        mc.dq = 0.0
        mc.tau = 0.0
        if damping:
            mc.q = 0.0
            mc.kp = 0.0
            mc.kd = 2.0
        else:
            mc.q = float(np.clip(targets[i], meta.q_lo[i], meta.q_hi[i]))
            mc.kp = float(kp[i])
            mc.kd = float(kd[i])
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


def _finalize_and_exit(code=0):
    """Guarantee soft robot, then exit PROMPTLY (DDS teardown can hang -> os._exit)."""
    _damp_burst(30)
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(code)


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


def mode_run(meta, session, ref, iface, watch, max_secs=None):
    """Stage 1: firm move-to-default (no damping gap). Stage 2: policy loop from default.
    Mirrors the h1_2 example's posture->behavior pattern."""
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
    last_action = np.zeros(meta.n)
    last_target = meta.default.copy()
    try:
        # STAGE 1 — reach the ready pose at firm gains, seamlessly (no damping gap).
        _ramp_to_pose(pub, low_cmd, crc, mode_machine, q0, meta.default, 4.0, kp_a, kd_a, meta)
        _hold(pub, low_cmd, crc, mode_machine, meta.default, 0.6, kp_a, kd_a, meta)
        print("at default — starting policy. (Legs may look odd on the gantry: the policy "
              "trained with ground contact. Watch for fault/violence; arms should track.)")
        # STAGE 2 — policy loop at TRAINED gains. Robot is already AT default and the
        # ramped motion (thriller_deploy) starts from default -> no lurch on entry.
        for tick in range(n_ticks):
            t0 = time.time()
            q, dq, imu_quat, gyro, _ = read_state(sub, timeout_s=0.5)
            obs, _ = build_obs(meta, ref, q, dq, imu_quat, gyro, last_action, tick)
            if not np.all(np.isfinite(obs)):
                raise RuntimeError(f"non-finite obs at tick {tick}")
            action = run_policy(session, obs, tick)
            if not np.all(np.isfinite(action)) or np.any(np.abs(action) > MAX_ACTION):
                raise RuntimeError(f"bad action at tick {tick} (|a|max={np.abs(action).max():.2f})")
            last_action = action
            last_target = action_to_target(meta, action)
            _send_cmd(pub, low_cmd, crc, mode_machine, last_target, meta.kp, meta.kd, meta)
            elapsed = time.time() - t0
            if elapsed > 2 * dt:
                raise RuntimeError(f"cycle overrun {elapsed*1000:.0f}ms -> damp")
            time.sleep(max(0.0, dt - elapsed))
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
    print(f"STAND-HOLD: firm move-to-default over {secs:.1f}s, then hold indefinitely "
          f"(approach gains {APPROACH_KP_SCALE:.1f}x). Ctrl-C / remote-damp to stop.")
    try:
        _ramp_to_pose(pub, low_cmd, crc, mode_machine, q0, meta.default, secs, kp, kd, meta)
        print("at default — HOLDING. Watch stance; damp when done.")
        while True:  # SIGINT/SIGTERM handler damps + exits; this loop just streams the hold
            _send_cmd(pub, low_cmd, crc, mode_machine, meta.default, kp, kd, meta)
            time.sleep(1.0 / CONTROL_HZ)
    except BaseException as e:  # noqa: BLE001 - ANY failure -> immediate damp
        print(f"\nSTOP: {e} -> damping")
    finally:
        _damp(pub, low_cmd, crc, mode_machine, meta, secs=1.0)
    _finalize_and_exit(0)


def mode_ground_run(meta, session, ref, iface, watch, max_secs, obs_order):
    """GROUND stage B: firm move-to-default, then run the ESTIMATOR-FREE ground policy
    for a short capped segment. Same safety spine as mode_run but with build_obs_ground
    (no fabricated estimator terms) and the conservative GROUND_MAX_ACTION cap.
    Requires --max-secs (no unbounded ground runs while bringing this up)."""
    if not watch:
        raise SystemExit("REFUSED: pass --i-will-watch-the-robot")
    if not max_secs or max_secs <= 0:
        raise SystemExit("REFUSED: ground-run requires --max-secs > 0 (cautious capped segment)")
    _require_human("ground-run")
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
    n_ticks = min(ref.T, int(max_secs * CONTROL_HZ))
    obs_dim = sum(w for _, w in obs_order)
    print(f"GROUND-RUN: stage-1 firm move-to-default (4s)+hold, then estimator-free policy "
          f"({obs_dim}-dim obs) {n_ticks}/{ref.T} ticks @ {CONTROL_HZ:.0f}Hz "
          f"[--max-secs {max_secs:.1f}], action cap {GROUND_MAX_ACTION:.1f}. "
          f"Tethered. Ctrl-C / remote-damp to stop.")
    last_action = np.zeros(meta.n)
    last_target = meta.default.copy()
    try:
        _ramp_to_pose(pub, low_cmd, crc, mode_machine, q0, meta.default, 4.0, kp_a, kd_a, meta)
        _hold(pub, low_cmd, crc, mode_machine, meta.default, 0.6, kp_a, kd_a, meta)
        print("at default — starting ground policy. Keep tension on the tether; damp at "
              "the first sign of a fault, lurch, or lean.")
        for tick in range(n_ticks):
            t0 = time.time()
            q, dq, imu_quat, gyro, _ = read_state(sub, timeout_s=0.5)
            obs, _ = build_obs_ground(meta, ref, q, dq, imu_quat, gyro, last_action, tick, obs_order)
            if not np.all(np.isfinite(obs)):
                raise RuntimeError(f"non-finite obs at tick {tick}")
            action = run_policy(session, obs, tick)
            if not np.all(np.isfinite(action)) or np.any(np.abs(action) > GROUND_MAX_ACTION):
                raise RuntimeError(f"bad action at tick {tick} (|a|max={np.abs(action).max():.2f})")
            last_action = action
            last_target = action_to_target(meta, action)
            _send_cmd(pub, low_cmd, crc, mode_machine, last_target, meta.kp, meta.kd, meta)
            elapsed = time.time() - t0
            if elapsed > 2 * dt:
                raise RuntimeError(f"cycle overrun {elapsed*1000:.0f}ms -> damp")
            time.sleep(max(0.0, dt - elapsed))
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


def mode_ground_run_odom(meta, session, ref, iface, watch, max_secs):
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
    last_action = np.zeros(meta.n)
    last_target = meta.default.copy()
    odom_pos0 = None
    prev_pos, prev_t = None, None
    try:
        _ramp_to_pose(pub, low_cmd, crc, mode_machine, q0, meta.default, 4.0, kp_a, kd_a, meta)
        _hold(pub, low_cmd, crc, mode_machine, meta.default, 0.6, kp_a, kd_a, meta)
        # Capture the re-anchor origin at policy start (robot is now at the reference pose).
        o = read_odom(odom, timeout_s=0.5)
        if o is None:
            raise RuntimeError("lost odom at policy start -> damp")
        odom_pos0 = o[0].copy()
        prev_pos, prev_t = o[0].copy(), time.time()
        print("at default — starting odometry-fed policy. Keep tension on the tether; "
              "damp at the first sign of a fault, lurch, or lean.")
        for tick in range(n_ticks):
            t0 = time.time()
            q, dq, imu_quat, gyro, _ = read_state(sub, timeout_s=0.5)
            o = read_odom(odom, timeout_s=0.5)
            if o is None:
                raise RuntimeError(f"lost {ODOM_TOPIC} at tick {tick} -> damp")
            pos, vel_field = o
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
            if not np.all(np.isfinite(action)) or np.any(np.abs(action) > GROUND_MAX_ACTION):
                raise RuntimeError(f"bad action at tick {tick} (|a|max={np.abs(action).max():.2f})")
            last_action = action
            last_target = action_to_target(meta, action)
            _send_cmd(pub, low_cmd, crc, mode_machine, last_target, meta.kp, meta.kd, meta)
            elapsed = time.time() - t0
            if elapsed > 2 * dt:
                raise RuntimeError(f"cycle overrun {elapsed*1000:.0f}ms -> damp")
            time.sleep(max(0.0, dt - elapsed))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["read", "move-to-default", "run", "stand-hold", "ground-run",
                             "ground-run-odom"],
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
    ap.add_argument("--ground-meta", default=str(GROUND_META))
    ap.add_argument("--ground-policy", default=str(GROUND_POLICY))
    ap.add_argument("--ground-motion", default=str(GROUND_MOTION))
    ap.add_argument("--i-will-watch-the-robot", action="store_true",
                    help="required for any motion mode; you are watching, remote in hand")
    a = ap.parse_args()

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
        return mode_ground_run(gmeta, gsession, gref, a.iface, a.i_will_watch_the_robot,
                               a.max_secs, obs_order) or 0

    meta = Meta(Path(a.meta))
    ref = Reference(Path(a.motion_npz))
    session = ort.InferenceSession(a.policy, providers=["CPUExecutionProvider"])

    if a.mode == "read":
        return mode_read(meta, ref, session, a.iface, a.timeout_s)
    if a.mode == "move-to-default":
        return mode_move_to_default(meta, session, ref, a.iface, a.secs, a.i_will_watch_the_robot) or 0
    if a.mode == "run":
        return mode_run(meta, session, ref, a.iface, a.i_will_watch_the_robot, a.max_secs) or 0
    if a.mode == "ground-run-odom":
        # PROVEN gantry policy (meta/ref/session above) + honest odometry-fed obs.
        return mode_ground_run_odom(meta, session, ref, a.iface, a.i_will_watch_the_robot,
                                    a.max_secs) or 0


if __name__ == "__main__":
    sys.exit(main() or 0)
