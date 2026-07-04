#!/usr/bin/env python3
"""ARM-DANCE-OVER-ONBOARD-BALANCE runtime: stream the dance's ARM choreography through
Unitree's arm-sdk weight-blend interface while the robot's ONBOARD controller keeps
balance (normal standing/balance mode STAYS ACTIVE the whole time).

WHY: the full-body low-level path (pipeline/deploy_runtime.py) must release the onboard
motion service and re-implement standing — that is where the thermal wall, gain boosts
and estimator gaps live. Thriller is ~90% arm choreography, so streaming ONLY the arms
over the onboard balancer keeps most of the show with a fraction of the risk: the legs
are never ours, and handing the arms back is a firmware-native weight blend.

MECHANISM (recon: ~/robot teleop + unitree_sdk2_python g1 examples — citations in
docs/ARM_DANCE_DESIGN.md):
  * publish unitree_hg LowCmd_ on topic "rt/arm_sdk" (NOT rt/lowcmd);
  * motor_cmd[29].q (kNotUsedJoint0) carries the BLEND WEIGHT: 0 = onboard owns the
    arms, 1 = arm-sdk owns the arms; ramp it for a soft handoff both ways;
  * command ONLY the 14 arm joints (DDS motor idx 15..28); legs+waist stay onboard's;
  * CRC + mode_machine from LowState, mode_pr=0, motor_cmd.mode=1 (teleop convention).

This runtime NEVER calls MotionSwitcherClient.ReleaseMode and NEVER publishes rt/lowcmd
— keeping onboard balance active is the entire point.

Run in the `tv` conda env (unitree_sdk2py + CycloneDDS + numpy):

    conda activate tv
    python -m pipeline.arm_dance_runtime --mode read      # SAFE default: offline plan, no DDS

SAFETY — non-negotiable:
  * --mode read is the default: fully OFFLINE (no DDS init, no publishers, no robot).
  * --mode arm-run refuses without BOTH --i-will-watch-the-robot AND env
    CONFIRMED_BY_HUMAN=alois (same gates as deploy_runtime), AND --max-secs N
    (--max-secs 0 = full dance needs env ARM_FULL_RUN=1 on top).
  * EVERY exit path (normal end, Ctrl-C, SIGTERM, crash) ramps the arm-sdk weight
    smoothly to 0 — the onboard controller always gets the arms back; we never leave
    them commanded. Mirrors deploy_runtime's damp-on-any-exit pattern.
  * Targets clamped to joint bands; trajectory refused if any arm joint exceeds
    MAX_ARM_SPEED_RAD_S (teleop's own arm velocity ceiling).

TIMING / MUSIC: the dance npz is 50 fps and the loop is 50 Hz wall-clock paced exactly
like deploy_runtime, 1 frame per tick — so dance frame 0 is the same musical reference
point in both runtimes and the existing music-sync guidance applies unchanged
(thriller_deploy embeds a 2.5 s activation ramp then the 1.5 s standing lead-in, i.e.
music starts 4.0 s after frame 0; see docs/audio_sync_design.md). In arm-run, frame 0
begins ARM_WEIGHT_RAMP_S + ARM_APPROACH_S (default 4.0 s) after streaming starts, and
the runtime prints an explicit "DANCE FRAME 0" cue line at that moment.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

# House patterns reused from the deploy runtime (module import is SDK-free: the
# unitree_sdk2py imports there are lazy, inside functions — same rule applies here).
from pipeline.deploy_runtime import (
    CONTROL_HZ,
    Meta,
    Telemetry,
    lowstate_subscriber,
    make_dds,
    read_state,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_META = ROOT / "data/policies/thriller/policy_meta.json"
DEFAULT_MOTION = ROOT / "data/policies/thriller/thriller_deploy.npz"

# ---- arm-sdk interface constants (recon-confirmed, see docs/ARM_DANCE_DESIGN.md) ----
ARM_SDK_TOPIC = "rt/arm_sdk"          # robot_arm.py:18 kTopicLowCommand_Motion
ARM_SDK_WEIGHT_IDX = 29               # kNotUsedJoint0 — .q field = blend weight 0..1
ARM_MOTOR_MODE = 1                    # motor_cmd.mode used by teleop (robot_arm.py:121)
PR_MODE = 0                           # LowCmd.mode_pr (robot_arm.py:111)

# ---- env knobs ----------------------------------------------------------------------
IFACE = os.environ.get("IFACE", "enp0s31f6")
ARM_WEIGHT_RAMP_S = float(os.environ.get("ARM_WEIGHT_RAMP_S", "2.0"))  # weight 0 -> 1
ARM_APPROACH_S = float(os.environ.get("ARM_APPROACH_S", "2.0"))    # current -> frame 0
ARM_RETURN_S = float(os.environ.get("ARM_RETURN_S", "1.5"))        # last frame -> start pose
ARM_RELEASE_S = float(os.environ.get("ARM_RELEASE_S", "1.5"))      # weight 1 -> 0
ARM_KP_SCALE = float(os.environ.get("ARM_KP_SCALE", "1.0"))
# Gain source: "meta" = per-joint trained gains from policy_meta.json (scaled by
# ARM_KP_SCALE, kd scaled too so the joint stays overdamped — house pattern);
# "teleop" = the values ~/robot's arm teleop has proven daily through this exact
# interface (kp 80 shoulder/elbow, 40 wrist; kd 3.0 / 1.5). Rationale in the doc:
# meta gains are SOFT (kp 14-17) because in training a closed-loop policy compensated
# tracking error; open-loop streaming will sag/lag at those gains — safe first test,
# then step up (ARM_KP_SCALE) or switch to the proven teleop preset for show quality.
ARM_GAINS = os.environ.get("ARM_GAINS", "meta")
TELEOP_KP_ARM, TELEOP_KD_ARM = 80.0, 3.0      # robot_arm.py:76-77 (kp_low/kd_low)
TELEOP_KP_WRIST, TELEOP_KD_WRIST = 40.0, 1.5  # robot_arm.py:78-79

# Teleop's own arm target velocity ceiling (robot_arm.py:82 arm_velocity_limit=20.0,
# raised to 30 max). A vetted dance is far below this; tripping it means a broken npz.
MAX_ARM_SPEED_RAD_S = 20.0

# ---- joint naming -> DDS motor index (G1 29-dof, hg LowCmd motor_cmd[]) --------------
# Source: G1_29_JointIndex in ~/robot/xr_teleoperate/teleop/robot_control/robot_arm.py
# :300-345 and G1JointIndex in unitree_sdk2_python g1_arm7_sdk_dds_example.py:19-64.
# policy_meta joint_order_29dof happens to match this order 1:1 today, but the mapping
# is built BY NAME so any future reorder is caught, never silently miswired.
G1_DDS_MOTOR_INDEX = {
    "left_hip_pitch_joint": 0, "left_hip_roll_joint": 1, "left_hip_yaw_joint": 2,
    "left_knee_joint": 3, "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6, "right_hip_roll_joint": 7, "right_hip_yaw_joint": 8,
    "right_knee_joint": 9, "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
    "waist_yaw_joint": 12, "waist_roll_joint": 13, "waist_pitch_joint": 14,
    "left_shoulder_pitch_joint": 15, "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17, "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19, "left_wrist_pitch_joint": 20,
    "left_wrist_yaw_joint": 21,
    "right_shoulder_pitch_joint": 22, "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24, "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26, "right_wrist_pitch_joint": 27,
    "right_wrist_yaw_joint": 28,
}
ARM_KEYWORDS = ("shoulder", "elbow", "wrist")
ARM_DDS_RANGE = set(range(15, 29))   # the only motors this runtime may command

# Telemetry stage codes (Telemetry.add(stage=...)) for the arm-run phases.
STAGE_WEIGHT_UP, STAGE_APPROACH, STAGE_DANCE, STAGE_RETURN = 1, 2, 3, 4

# Set once the arm-sdk publisher is up: dict with everything the exit path needs to
# ramp the weight to 0 (handing the arms back to onboard) from ANY state. Mirrors
# deploy_runtime._DAMP_CTX. state["weight"]/state["targets"] are updated every send.
_HANDBACK_CTX = None
_TELEM = None
_FINALIZING = False


# ---- pure planning helpers (offline-testable, no SDK) --------------------------------
def arm_joint_map(meta: Meta):
    """[(npz_column, dds_motor_index, joint_name)] for every ARM joint, matched by NAME.

    Refuses anything that is not exactly the 14 shoulder/elbow/wrist joints mapping
    into DDS motors 15..28 — a wrong map here would command legs. Waist is deliberately
    EXCLUDED (onboard owns it; v1 is arms-only, see design doc)."""
    rows = []
    for col, name in enumerate(meta.joint_order):
        if any(k in name for k in ARM_KEYWORDS):
            if name not in G1_DDS_MOTOR_INDEX:
                raise SystemExit(f"REFUSED: unknown arm joint name {name!r} — cannot map to a DDS motor")
            rows.append((col, G1_DDS_MOTOR_INDEX[name], name))
    if len(rows) != 14:
        raise SystemExit(f"REFUSED: expected 14 arm joints, matched {len(rows)}: {[r[2] for r in rows]}")
    bad = [r for r in rows if r[1] not in ARM_DDS_RANGE]
    if bad:
        raise SystemExit(f"REFUSED: arm joint mapped outside DDS arm range 15..28: {bad}")
    return rows


def extract_arm_trajectory(meta: Meta, npz_path: Path):
    """(traj[T,14], rows) — the dance's arm columns, validated finite/shaped/slow-enough."""
    rows = arm_joint_map(meta)
    d = np.load(npz_path)
    jp = d["joint_pos"]
    if jp.ndim != 2 or jp.shape[1] != meta.n:
        raise SystemExit(f"REFUSED: joint_pos shape {jp.shape} != (T, {meta.n})")
    if jp.shape[0] < 2:
        raise SystemExit(f"REFUSED: motion has {jp.shape[0]} frames")
    traj = jp[:, [c for c, _, _ in rows]].astype(float)
    if not np.all(np.isfinite(traj)):
        raise SystemExit("REFUSED: non-finite values in the arm trajectory")
    return traj, rows


def max_arm_speed(traj, hz=CONTROL_HZ):
    """Fastest commanded arm-joint speed in rad/s (frame-to-frame at the stream rate)."""
    return float(np.max(np.abs(np.diff(traj, axis=0))) * hz)


def cosine_blend(q0, q1, secs, hz=CONTROL_HZ):
    """[steps+1, n] cosine interpolation q0 -> q1. Row 0 == q0 EXACTLY (no lurch),
    last row == q1 exactly. Same easing as deploy_runtime._ramp_to_pose."""
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    steps = max(1, int(secs * hz))
    a = 0.5 - 0.5 * np.cos(np.pi * np.arange(steps + 1) / steps)   # 0 -> 1
    return (1.0 - a)[:, None] * q0[None, :] + a[:, None] * q1[None, :]


def weight_profile(w0, w1, secs, hz=CONTROL_HZ):
    """Cosine-eased scalar ramp w0 -> w1 (endpoints exact, monotonic)."""
    steps = max(1, int(secs * hz))
    a = 0.5 - 0.5 * np.cos(np.pi * np.arange(steps + 1) / steps)
    return (1.0 - a) * float(w0) + a * float(w1)


def dance_ticks(n_frames: int, max_secs: float, hz=CONTROL_HZ) -> int:
    """How many dance frames to stream. max_secs==0 => full dance (gated by
    ARM_FULL_RUN in require_arm_run_gates); else cap at max_secs of 50 Hz ticks."""
    if max_secs is None:
        raise ValueError("max_secs must be validated by require_arm_run_gates first")
    if max_secs == 0:
        return int(n_frames)
    return max(1, min(int(n_frames), int(max_secs * hz)))


def arm_gains(meta: Meta, rows):
    """(kp[14], kd[14]) for the arm motor_cmds, per ARM_GAINS/ARM_KP_SCALE (see doc)."""
    if ARM_GAINS == "teleop":
        kp = np.array([TELEOP_KP_WRIST if "wrist" in name else TELEOP_KP_ARM
                       for _, _, name in rows])
        kd = np.array([TELEOP_KD_WRIST if "wrist" in name else TELEOP_KD_ARM
                       for _, _, name in rows])
        return kp * ARM_KP_SCALE, kd * ARM_KP_SCALE
    if ARM_GAINS != "meta":
        raise SystemExit(f"REFUSED: ARM_GAINS={ARM_GAINS!r} (use 'meta' or 'teleop')")
    cols = [c for c, _, _ in rows]
    # scale kd with kp (deploy_runtime house pattern: both scaled -> stays overdamped)
    return meta.kp[cols] * ARM_KP_SCALE, meta.kd[cols] * ARM_KP_SCALE


def require_arm_run_gates(watch: bool, max_secs, env=None):
    """The refusal gates for --mode arm-run. Raises SystemExit unless ALL pass."""
    env = os.environ if env is None else env
    if not watch:
        raise SystemExit("REFUSED: pass --i-will-watch-the-robot")
    if env.get("CONFIRMED_BY_HUMAN") != "alois":
        raise SystemExit("REFUSED: --mode arm-run needs env CONFIRMED_BY_HUMAN=alois")
    if max_secs is None:
        raise SystemExit("REFUSED: arm-run requires --max-secs N (seconds; start with 5). "
                         "--max-secs 0 = FULL dance, allowed only with env ARM_FULL_RUN=1.")
    if max_secs < 0:
        raise SystemExit("REFUSED: --max-secs must be >= 0")
    if max_secs == 0 and env.get("ARM_FULL_RUN") != "1":
        raise SystemExit("REFUSED: --max-secs 0 (full dance) needs env ARM_FULL_RUN=1 — "
                         "run a short capped test first.")


# ---- arm-sdk publishing (SDK imported lazily; fakes injectable for tests) ------------
def _arm_publisher():
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    pub = ChannelPublisher(ARM_SDK_TOPIC, LowCmd_)
    pub.Init()
    return pub, unitree_hg_msg_dds__LowCmd_(), CRC()


def send_arm_cmd(pub, low_cmd, crc, mode_machine, dds_idx, targets, kp, kd,
                 q_lo, q_hi, weight):
    """Mutate the reused LowCmd and publish ONE rt/arm_sdk tick.

    Writes ONLY the 14 arm motor_cmds and the weight slot (motor_cmd[29].q). Legs and
    waist entries are never touched — the onboard balancer owns them, and arm_sdk
    ignores them anyway; leaving them zeroed makes that impossible to get wrong.
    Targets are clamped to the meta joint band; weight clamped to [0, 1]."""
    low_cmd.mode_pr = PR_MODE
    low_cmd.mode_machine = mode_machine
    low_cmd.motor_cmd[ARM_SDK_WEIGHT_IDX].q = float(np.clip(weight, 0.0, 1.0))
    for k, mi in enumerate(dds_idx):
        mc = low_cmd.motor_cmd[mi]
        mc.mode = ARM_MOTOR_MODE
        mc.q = float(np.clip(targets[k], q_lo[k], q_hi[k]))
        mc.dq = 0.0
        mc.tau = 0.0
        mc.kp = float(kp[k])
        mc.kd = float(kd[k])
    low_cmd.crc = crc.Crc(low_cmd)
    pub.Write(low_cmd)


def _hand_back_and_exit(code=0):
    """GUARANTEE the arms are handed back to onboard, then exit promptly (os._exit —
    DDS teardown can hang). From ANY state: ramps the weight from wherever it is to 0
    over ARM_RELEASE_S while holding the LAST commanded targets (the weight blend does
    the smoothing — no pose jump), then a short weight=0 burst so the release lands.
    Telemetry is flushed AFTER the handback — safety never waits on it."""
    global _FINALIZING
    if _FINALIZING:            # second signal while finalizing: get out NOW
        os._exit(1)
    _FINALIZING = True
    ctx = _HANDBACK_CTX
    if ctx:
        try:
            w0 = float(ctx["state"]["weight"])
            targets = np.asarray(ctx["state"]["targets"], float)
            print(f"\nhanding arms back to onboard: weight {w0:.2f} -> 0 "
                  f"over {ARM_RELEASE_S:.1f}s ...", flush=True)
            for w in weight_profile(w0, 0.0, ARM_RELEASE_S):
                try:
                    send_arm_cmd(ctx["pub"], ctx["low_cmd"], ctx["crc"], ctx["mode_machine"],
                                 ctx["dds_idx"], targets, ctx["kp"], ctx["kd"],
                                 ctx["q_lo"], ctx["q_hi"], w)
                except Exception:
                    break
                time.sleep(1.0 / CONTROL_HZ)
            for _ in range(10):     # make sure weight=0 (release) actually lands
                try:
                    send_arm_cmd(ctx["pub"], ctx["low_cmd"], ctx["crc"], ctx["mode_machine"],
                                 ctx["dds_idx"], targets, ctx["kp"], ctx["kd"],
                                 ctx["q_lo"], ctx["q_hi"], 0.0)
                except Exception:
                    break
                time.sleep(0.01)
            print("arm-sdk weight = 0 — onboard owns the arms again.", flush=True)
        except Exception:
            pass
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


def _install_handback_on_signals():
    """SIGTERM/SIGINT -> weight-ramp handback, then exit. Default SIGTERM would kill the
    stream mid-blend with the weight still up — never allowed (mirrors deploy_runtime)."""
    def handler(signum, _frame):
        try:
            print(f"\n[signal {signum}] -> hand arms back to onboard, then exit", flush=True)
        except Exception:
            pass
        _hand_back_and_exit(0)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


# ---- MODE: read (SAFE default — fully offline, zero DDS) -----------------------------
def mode_read(meta: Meta, npz_path: Path, max_secs=None):
    traj, rows = extract_arm_trajectory(meta, npz_path)
    kp, kd = arm_gains(meta, rows)
    cols = [c for c, _, _ in rows]
    T = traj.shape[0]
    dur = T / CONTROL_HZ
    def_arm = meta.default[cols]
    delta0 = traj[0] - def_arm

    print("=== ARM DANCE PLAN (read-only, offline — no DDS) ===")
    print(f"motion: {npz_path}")
    print(f"frames: {T} @ {CONTROL_HZ:.0f} fps = {dur:.1f}s  (1 frame per 50 Hz tick, "
          f"wall-clock paced like deploy_runtime)")
    print(f"gains:  ARM_GAINS={ARM_GAINS} x ARM_KP_SCALE={ARM_KP_SCALE:g}")
    print(f"\n{'npz':>4} {'dds':>4}  {'joint':<28}{'kp':>7}{'kd':>6}"
          f"{'default':>9}{'frame0':>9}{'delta':>8}")
    for k, (col, mi, name) in enumerate(rows):
        print(f"{col:>4} {mi:>4}  {name:<28}{kp[k]:>7.1f}{kd[k]:>6.2f}"
              f"{np.degrees(def_arm[k]):>8.1f}°{np.degrees(traj[0][k]):>8.1f}°"
              f"{np.degrees(delta0[k]):>7.1f}°")
    print("(legs + waist: NOT streamed — onboard balance owns them; weight slot = "
          f"motor_cmd[{ARM_SDK_WEIGHT_IDX}].q on {ARM_SDK_TOPIC})")

    spd = max_arm_speed(traj)
    n_ticks = dance_ticks(T, max_secs) if max_secs is not None else T
    t_frame0 = ARM_WEIGHT_RAMP_S + ARM_APPROACH_S
    t_dance_end = t_frame0 + n_ticks / CONTROL_HZ
    print("\n=== TIMELINE (arm-run) ===")
    print(f"  0.0s          weight 0 -> 1 over {ARM_WEIGHT_RAMP_S:.1f}s (holding the CURRENT arm pose)")
    print(f"  {ARM_WEIGHT_RAMP_S:.1f}s          cosine approach: current pose -> dance frame 0 over {ARM_APPROACH_S:.1f}s")
    print(f"  {t_frame0:.1f}s          DANCE frame 0 (music reference point — same as deploy tick 0;")
    print("                thriller_deploy embeds 2.5s ramp + 1.5s lead-in, so music starts")
    print("                frame0 + 4.0s; see docs/audio_sync_design.md)")
    print(f"  {t_dance_end:.1f}s{'' :8}dance ends ({n_ticks}/{T} frames"
          f"{' — capped by --max-secs' if n_ticks < T else ''})")
    print(f"                cosine return to start pose over {ARM_RETURN_S:.1f}s, "
          f"then weight 1 -> 0 over {ARM_RELEASE_S:.1f}s")

    wp = weight_profile(0.0, 1.0, ARM_WEIGHT_RAMP_S)
    ok_ramp = wp[0] == 0.0 and wp[-1] == 1.0 and np.all(np.diff(wp) >= 0)
    print("\n=== SANITY ===")
    print(f"  weight ramp: {len(wp)} ticks, monotonic 0->1: {'OK' if ok_ramp else 'FAIL'}")
    print(f"  max arm speed in trajectory: {spd:.2f} rad/s "
          f"(limit {MAX_ARM_SPEED_RAD_S:.0f} — teleop's own ceiling): "
          f"{'OK' if spd <= MAX_ARM_SPEED_RAD_S else 'FAIL'}")
    worst0 = float(np.max(np.abs(delta0)))
    print(f"  frame0 vs default pose: worst delta {np.degrees(worst0):.2f}° "
          f"(thriller_deploy embeds its own activation ramp, so this should be ~0)")
    print("  first streamed frame = the robot's CURRENT arm pose (captured live) -> no lurch")
    if not ok_ramp or spd > MAX_ARM_SPEED_RAD_S:
        print("\nNO-GO: plan sanity failed.")
        return 2
    print("\nPLAN OK. arm-run gates: --i-will-watch-the-robot + CONFIRMED_BY_HUMAN=alois "
          "+ --max-secs N (0=full needs ARM_FULL_RUN=1).")
    return 0


# ---- MODE: arm-run (GATED — human-supervised only) ------------------------------------
def mode_arm_run(meta: Meta, npz_path: Path, iface: str, watch: bool, max_secs,
                 timeout_s: float = 2.0):
    require_arm_run_gates(watch, max_secs)
    print("[arm-run] human-confirmed. Onboard balance STAYS ACTIVE; remote in hand anyway.")
    traj, rows = extract_arm_trajectory(meta, npz_path)
    spd = max_arm_speed(traj)
    if spd > MAX_ARM_SPEED_RAD_S:
        raise SystemExit(f"REFUSED: arm trajectory peaks at {spd:.1f} rad/s "
                         f"> {MAX_ARM_SPEED_RAD_S:.0f} (broken npz?)")
    kp, kd = arm_gains(meta, rows)
    cols = [c for c, _, _ in rows]
    dds_idx = [mi for _, mi, _ in rows]
    q_lo, q_hi = meta.q_lo[cols], meta.q_hi[cols]
    n_ticks = dance_ticks(traj.shape[0], max_secs)
    dt = 1.0 / CONTROL_HZ

    make_dds(iface)
    sub = lowstate_subscriber()
    q0, _, _, _, msg0 = read_state(sub, timeout_s)
    mode_machine = int(msg0.mode_machine)
    q_arm0 = q0[dds_idx].copy()   # read_state index i == DDS motor i
    worst_hold = float(np.max(np.abs(q_arm0 - meta.default[cols])))
    print(f"current arm pose captured (worst |current-default| = "
          f"{np.degrees(worst_hold):.1f} deg). mode_machine={mode_machine}.")

    pub, low_cmd, crc = _arm_publisher()
    state = {"weight": 0.0, "targets": q_arm0.copy()}
    global _HANDBACK_CTX, _TELEM
    _HANDBACK_CTX = {"pub": pub, "low_cmd": low_cmd, "crc": crc,
                     "mode_machine": mode_machine, "dds_idx": dds_idx,
                     "kp": kp, "kd": kd, "q_lo": q_lo, "q_hi": q_hi, "state": state}
    _install_handback_on_signals()
    telem = _TELEM = Telemetry("arm-run", meta, extra={
        "runtime": "arm_dance", "topic": ARM_SDK_TOPIC, "arm_gains": ARM_GAINS,
        "arm_kp_scale": ARM_KP_SCALE, "weight_ramp_s": ARM_WEIGHT_RAMP_S,
        "approach_s": ARM_APPROACH_S, "return_s": ARM_RETURN_S,
        "release_s": ARM_RELEASE_S, "max_secs": max_secs, "n_ticks": n_ticks,
        # arm-run telemetry conventions: 'action' = arm-sdk weight (broadcast over 29),
        # 'target' = measured q with the 14 arm slots replaced by the commanded targets.
    })

    def send(targets14, weight):
        state["targets"] = np.asarray(targets14, float).copy()
        state["weight"] = float(weight)
        send_arm_cmd(pub, low_cmd, crc, mode_machine, dds_idx, targets14, kp, kd,
                     q_lo, q_hi, weight)

    def record(tick, targets14, weight, stage):
        try:
            q, dq, imu_quat, gyro, msg = read_state(sub, timeout_s=0.5)
        except SystemExit as e:            # LowState vanished mid-run -> hand back
            raise RuntimeError(str(e)) from None
        target29 = q.copy()
        target29[dds_idx] = np.clip(targets14, q_lo, q_hi)
        telem.add(tick, q, dq, msg, imu_quat, gyro, np.full(meta.n, weight), target29,
                  stage=stage)

    print(f"ARM-RUN on {ARM_SDK_TOPIC}: weight 0->1 {ARM_WEIGHT_RAMP_S:.1f}s | approach "
          f"{ARM_APPROACH_S:.1f}s | dance {n_ticks}/{traj.shape[0]} frames "
          f"({n_ticks / CONTROL_HZ:.1f}s) @ {CONTROL_HZ:.0f}Hz | return {ARM_RETURN_S:.1f}s "
          f"| release {ARM_RELEASE_S:.1f}s. Ctrl-C hands arms back at any time.")
    tick = 0
    try:
        # (b) engage: weight 0 -> 1 while commanding the pose the arms are ALREADY in
        # (frozen snapshot, not live-tracked: target==measured would let gravity sag
        # the arms as the weight rises; the snapshot holds them where onboard had them).
        for w in weight_profile(0.0, 1.0, ARM_WEIGHT_RAMP_S):
            t0 = time.time()
            send(q_arm0, w)
            record(tick, q_arm0, w, STAGE_WEIGHT_UP)
            tick += 1
            time.sleep(max(0.0, dt - (time.time() - t0)))
        # (c) approach: current pose -> dance frame 0
        for row in cosine_blend(q_arm0, traj[0], ARM_APPROACH_S):
            t0 = time.time()
            send(row, 1.0)
            record(tick, row, 1.0, STAGE_APPROACH)
            tick += 1
            time.sleep(max(0.0, dt - (time.time() - t0)))
        # (d) the dance — 50 Hz wall-clock paced, 1 npz frame per tick (music sync
        # depends on this pacing matching deploy_runtime exactly).
        print(f"DANCE FRAME 0 NOW (t=+{ARM_WEIGHT_RAMP_S + ARM_APPROACH_S:.1f}s — music cue "
              "reference; music starts frame0+4.0s per audio guidance).")
        for f in range(n_ticks):
            t0 = time.time()
            target = traj[f]
            if not np.all(np.isfinite(target)):
                raise RuntimeError(f"non-finite target at frame {f}")
            send(target, 1.0)
            record(tick, target, 1.0, STAGE_DANCE)
            tick += 1
            elapsed = time.time() - t0
            if elapsed > 2 * dt:
                raise RuntimeError(f"cycle overrun {elapsed * 1000:.0f}ms at frame {f}")
            time.sleep(max(0.0, dt - elapsed))
        # (e) normal end: cosine return to the pose onboard was holding at engage —
        # the best proxy for "onboard's default" — then the finalizer ramps weight->0.
        print("dance segment done — returning arms to the start pose.")
        for row in cosine_blend(traj[n_ticks - 1], q_arm0, ARM_RETURN_S):
            t0 = time.time()
            send(row, 1.0)
            record(tick, row, 1.0, STAGE_RETURN)
            tick += 1
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except BaseException as e:  # noqa: BLE001 - ANY failure -> smooth handback
        print(f"\nSTOP: {e} -> handing arms back to onboard")
    finally:
        _hand_back_and_exit(0)   # never returns (ramps weight->0, saves telemetry)


def main():
    ap = argparse.ArgumentParser(
        description="Stream dance ARM choreography over the onboard balancer via rt/arm_sdk")
    ap.add_argument("--mode", choices=["read", "arm-run"], default="read")
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--motion-npz", default=str(DEFAULT_MOTION))
    ap.add_argument("--iface", default=IFACE)
    ap.add_argument("--timeout-s", type=float, default=2.0)
    ap.add_argument("--max-secs", type=float, default=None,
                    help="arm-run: REQUIRED. Cap the dance segment to N seconds "
                         "(start with 5). 0 = full dance, needs env ARM_FULL_RUN=1.")
    ap.add_argument("--i-will-watch-the-robot", action="store_true",
                    help="required for arm-run; you are watching, remote in hand")
    a = ap.parse_args()

    meta = Meta(Path(a.meta))
    if a.mode == "read":
        return mode_read(meta, Path(a.motion_npz), a.max_secs)
    if a.mode == "arm-run":
        return mode_arm_run(meta, Path(a.motion_npz), a.iface,
                            a.i_will_watch_the_robot, a.max_secs, a.timeout_s) or 0


if __name__ == "__main__":
    sys.exit(main() or 0)
