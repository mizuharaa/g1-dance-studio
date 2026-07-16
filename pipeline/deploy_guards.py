#!/usr/bin/env python3
"""Deploy-side SAFETY GUARDS for the Unitree G1 dance runtime.

WHY THIS MODULE EXISTS
----------------------
Two real fall incidents (logs/jobs.md, 2026-07-08) drove these guards:
  (1) ENTRY FALL: the onboard balance service is RELEASED, then a STATIC PD ramp
      moves the robot to the policy's ready pose. In that unheld window the robot
      tips/oscillates — worst when SUSPENDED (gantry) or feet-on-ground far from
      the ready pose, because the base-velocity estimate is INVALID with no valid
      foot contact.
  (2) OVER-COMMANDING: the policy can thrash (a v7 sandbox showed ~220% amplitude).

These are DEPLOY-SIDE backstops. They are ADDITIVE and every default is FAIL-SAFE =
DAMP. They must NOT regress the PROVEN feet-off gantry path: the gantry policy is
intentionally suspended (base_lin_vel~=0 in-distribution) and does NOT set
`requires_ground_contact`, so the suspension/contact gates are inert for it. A
free-stand GROUND policy that needs real contact sets the flag and gets the gates.

CONTACT SIGNAL — HONEST INVENTORY (measured from the SDK IDL, 2026-07-16)
------------------------------------------------------------------------
The G1 speaks the `unitree_hg` LowState. Its fields are ONLY:
    imu_state (quaternion, gyroscope, accelerometer, rpy)
    motor_state[35] (q, dq, ddq, tau_est, temperature, vol)
There is **NO foot-force sensor** on this interface. `foot_force` /
`foot_force_est` exist only on `unitree_go` (the quadruped LowState), not on the
G1. rt/odommodestate carries a base pose/velocity/height estimate but FREEZES the
moment we release the motion service (confirmed on hardware 2026-07-04), so it is
useless mid-run.

Therefore contact is inferred from PROXIES, each with a documented blind spot:
  * SUPPORT TORQUE (tau_est on the sagittal weight-bearing leg joints: hip_pitch,
    knee, ankle_pitch). Standing, the legs carry ~35 kg of body weight and these
    joints show substantial support torque; SUSPENDED on the gantry the legs hang
    and that torque collapses toward the (small) hanging-limb value. This is the
    ONLY signal that distinguishes a robot STANDING STILL from one HANGING STILL.
  * KINEMATIC PLANTED-CONFIDENCE (pipeline.leg_odometry FusedBaseEstimator.contact):
    a planted foot is stationary in the world -> its implied world velocity ~0.
    This catches a foot SLIP/LIFT *during motion*, but it CANNOT tell static
    suspension from static standing (both have zero foot world velocity), so it is
    only a secondary, mid-run signal.

EVERY numeric threshold below is an UNVALIDATED starting point — the robot is DOWN
(burnt DC-DC, RMA pending) so none of this has been measured on hardware. The
gantry validation checklist for each is in docs/DEPLOY_SAFETY_GUARDS.md; the
support-torque threshold in particular MUST be calibrated (suspended-still vs
standing-still) before it is trusted.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np


# ============================================================================
# GUARD 3 — ACTION CLAMPS + RATE LIMITS
# ============================================================================
# Hard per-joint position clamp AND per-tick change (rate) limit on the policy's
# commanded joint TARGETS. This is a deploy-side backstop against the
# over-commanding/thrashing, INDEPENDENT of the training-side action-rate penalty.
# The clamp bounds where a joint can be told to go; the rate limit bounds how fast
# the TARGET can move, killing single-tick teleports/thrash that a pure amplitude
# cap misses. Uses the hardware-truth per-joint limits from pipeline.g1_limits.

# Rate-limit cap = RATE_LIMIT_FRAC * per-joint velocity_limit * dt. Default 1.5:
# a joint's TARGET should never need to slew faster than ~1.5x the actuator's own
# no-load speed, so this is a teleport/thrash catcher, NOT a dance limiter — a
# physically realizable command never trips it. MUST be cross-checked against the
# proven-gantry-run telemetry (max per-tick |dtarget|) before trusting on the ground.
RATE_LIMIT_FRAC = float(os.environ.get("RATE_LIMIT_FRAC", "1.5"))
RATE_LIMIT_ENABLE = os.environ.get("RATE_LIMIT_ENABLE", "1") != "0"
# Fallback velocity limit [rad/s] if g1_limits is unavailable (conservative).
_FALLBACK_VEL_LIMIT = 20.0


def joint_pos_limits(meta):
    """(lo, hi) per-joint position clamp. Prefers the official-MJCF ranges from
    pipeline.g1_limits (hardware truth); falls back to the meta's default band."""
    try:
        from pipeline import g1_limits
        lo, hi = g1_limits.POS_LO, g1_limits.POS_HI
        if lo is not None and hi is not None and len(lo) == meta.n:
            return np.asarray(lo, float).copy(), np.asarray(hi, float).copy()
    except Exception:  # noqa: BLE001 - g1_limits/mujoco optional; fall back safely
        pass
    return np.asarray(meta.q_lo, float).copy(), np.asarray(meta.q_hi, float).copy()


def rate_limit_step(meta, dt, frac=None):
    """Per-joint max |target change| per tick [rad] = frac * velocity_limit * dt."""
    frac = RATE_LIMIT_FRAC if frac is None else float(frac)
    try:
        from pipeline import g1_limits
        vlim = np.asarray(g1_limits.VELOCITY_LIMIT, float)
        if len(vlim) != meta.n:
            vlim = np.full(meta.n, _FALLBACK_VEL_LIMIT)
    except Exception:  # noqa: BLE001
        vlim = np.full(meta.n, _FALLBACK_VEL_LIMIT)
    return frac * vlim * float(dt)


def clamp_and_rate_limit(target, last_target, lo, hi, max_step):
    """Backstop the commanded target: rate-limit the per-tick change, then clamp to
    the joint position range. Returns (safe_target, n_rate_limited, n_clamped).

    FAIL-SAFE: a non-finite target RAISES (the caller's except path damps) — a NaN
    must never be silently clamped into a plausible-looking command.
    """
    t = np.asarray(target, float)
    if not np.all(np.isfinite(t)):
        raise ValueError("non-finite commanded target -> damp")
    prev = np.asarray(last_target, float)
    if RATE_LIMIT_ENABLE:
        step = np.clip(t - prev, -max_step, max_step)
        rl = prev + step
        n_rl = int(np.sum(np.abs(t - prev) > max_step + 1e-9))
    else:
        rl, n_rl = t, 0
    out = np.clip(rl, lo, hi)
    n_cl = int(np.sum((rl < lo - 1e-9) | (rl > hi + 1e-9)))
    return out, n_rl, n_cl


# ============================================================================
# GUARD 4 — ESTIMATOR-VALIDITY CHECK
# ============================================================================
# If the state estimate feeding the policy is IMPLAUSIBLE (NaN, out-of-range base
# velocity, base height off the body, or contact lost) -> mark INVALID -> caller
# damps. This is the "don't fly the policy blind" gate: a bad estimate is exactly
# what tips a suspended robot (incident recap).

# Trained base_lin_vel noise was +-0.5 m/s; leg_odometry hard-clips at 2.5 m/s. A
# body-frame base speed above this is not real locomotion, it is a bad estimate.
BASE_SPEED_MAX = float(os.environ.get("EST_BASE_SPEED_MAX", "2.5"))
# Plausible torso/base height band [m] (G1 torso stands ~0.6-0.8 m; crouch lower).
EST_HEIGHT_MIN = float(os.environ.get("EST_HEIGHT_MIN", "0.20"))
EST_HEIGHT_MAX = float(os.environ.get("EST_HEIGHT_MAX", "1.00"))
# Minimum kinematic contact confidence [0..1] to consider the estimate anchored.
CONTACT_MIN_CONF = float(os.environ.get("EST_CONTACT_MIN_CONF", "0.30"))


def check_estimate_valid(base_lin_vel=None, height=None, contact=None):
    """Return (ok: bool, reason: str). Any implausible/NaN input -> (False, ...).
    All args optional; only the provided ones are checked (a mode passes what it has)."""
    for name, val in (("base_lin_vel", base_lin_vel), ("height", height),
                      ("contact", contact)):
        if val is not None and not np.all(np.isfinite(np.asarray(val, float))):
            return False, f"{name} non-finite"
    if base_lin_vel is not None:
        sp = float(np.linalg.norm(np.asarray(base_lin_vel, float)))
        if sp > BASE_SPEED_MAX:
            return False, f"base speed {sp:.2f} m/s > {BASE_SPEED_MAX:.2f} (bad estimate)"
    if height is not None:
        h = float(height)
        if not (EST_HEIGHT_MIN <= h <= EST_HEIGHT_MAX):
            return False, f"base height {h:.2f} m outside [{EST_HEIGHT_MIN}, {EST_HEIGHT_MAX}]"
    if contact is not None and float(contact) < CONTACT_MIN_CONF:
        return False, f"contact confidence {float(contact):.2f} < {CONTACT_MIN_CONF} (lost)"
    return True, "ok"


# ============================================================================
# GUARDS 1 & 2 — SUSPENSION / FOOT-CONTACT via a PROXY
# ============================================================================
# Guard 2: REQUIRE STABLE bilateral foot contact before entering run mode (before
#          releasing onboard) — debounced over a short window.
# Guard 1: NEVER run a state-estimation-dependent (free-stand) policy while
#          SUSPENDED — refuse at entry, and damp if contact is lost mid-run.
# Both are gated on the policy's `requires_ground_contact` flag so the proven
# feet-off gantry policy (flag unset) is completely unaffected.

# Sagittal weight-bearing leg joints — their tau_est carries body weight when the
# feet are planted and collapses when the robot hangs. Matched BY NAME.
_SUPPORT_TOKENS = ("hip_pitch", "knee", "ankle_pitch")
# Total |tau_est| across the 6 support joints below which we treat the robot as
# NOT bearing weight (suspended). *** UNVALIDATED default *** — MUST be calibrated
# on the gantry: record support_torque() suspended-still vs standing-still and set
# the threshold to the midpoint. 12 Nm is a deliberately conservative placeholder
# (a standing G1's knees alone carry well above this; a hanging robot's support
# joints sit far below). SUSPENSION_TAU_MIN=0 disables the torque gate.
SUSPENSION_TAU_MIN = float(os.environ.get("SUSPENSION_TAU_MIN", "12.0"))
# Debounce windows (ticks @50Hz).
CONTACT_CONFIRM_TICKS = int(os.environ.get("CONTACT_CONFIRM_TICKS", "10"))   # 200 ms stable
CONTACT_LOST_TICKS = int(os.environ.get("CONTACT_LOST_TICKS", "8"))          # 160 ms lost


def support_joint_indices(joint_order):
    return [i for i, n in enumerate(joint_order)
            if any(tok in n for tok in _SUPPORT_TOKENS)]


def support_torque(tau_est, joint_order):
    """Sum of |tau_est| over the sagittal weight-bearing leg joints [Nm]. High when
    the feet bear body weight, low when the robot hangs."""
    idx = support_joint_indices(joint_order)
    tau = np.asarray(tau_est, float)
    return float(np.sum(np.abs(tau[idx])))


def tau_est_from_msg(msg, n=29):
    """Extract the tau_est vector from a unitree_hg LowState message (best effort)."""
    return np.array([float(msg.motor_state[i].tau_est) for i in range(n)], float)


class ContactEstimator:
    """Debounced contact/suspension state from the available PROXIES.

    A sample is judged 'in contact' if EITHER available proxy says so:
      * support torque >= SUSPENSION_TAU_MIN  (the static suspension discriminator), OR
      * kinematic planted confidence >= CONTACT_MIN_CONF  (mid-run slip/lift signal).
    We OR them because each covers the other's blind spot; require BOTH via
    `require_both=True` for a stricter gate. If NEITHER proxy is provided the sample
    is UNKNOWN and, fail-safe, counts as NOT in contact.

    stable_contact(): True once contact has held CONTACT_CONFIRM_TICKS in a row
      (the entry gate — guard 2).
    lost_contact(): True once contact has been absent CONTACT_LOST_TICKS in a row
      (the mid-run loss trip — guard 1).
    """

    def __init__(self, joint_order, *, tau_min=None, confirm_ticks=None,
                 lost_ticks=None, conf_min=None, require_both=False):
        self.joint_order = list(joint_order)
        self.tau_min = SUSPENSION_TAU_MIN if tau_min is None else float(tau_min)
        self.conf_min = CONTACT_MIN_CONF if conf_min is None else float(conf_min)
        self.confirm_ticks = CONTACT_CONFIRM_TICKS if confirm_ticks is None else int(confirm_ticks)
        self.lost_ticks = CONTACT_LOST_TICKS if lost_ticks is None else int(lost_ticks)
        self.require_both = bool(require_both)
        self._ok_run = 0
        self._lost_run = 0
        self.last = {}

    def sample(self, tau_est=None, contact_conf=None):
        """Feed one tick; returns True/False for 'in contact THIS tick' (undebounced)."""
        tau_ok = None
        if tau_est is not None and self.tau_min > 0:
            st = support_torque(tau_est, self.joint_order)
            tau_ok = st >= self.tau_min
            self.last["support_torque"] = st
        kin_ok = None
        if contact_conf is not None:
            kin_ok = float(contact_conf) >= self.conf_min
            self.last["contact_conf"] = float(contact_conf)
        votes = [v for v in (tau_ok, kin_ok) if v is not None]
        if not votes:
            in_contact = False           # no signal -> fail-safe NOT in contact
        elif self.require_both:
            in_contact = all(votes)
        else:
            in_contact = any(votes)
        self._ok_run = self._ok_run + 1 if in_contact else 0
        self._lost_run = 0 if in_contact else self._lost_run + 1
        self.last["in_contact"] = in_contact
        return in_contact

    def stable_contact(self):
        return self._ok_run >= self.confirm_ticks

    def lost_contact(self):
        return self._lost_run >= self.lost_ticks


def policy_requires_ground_contact(meta):
    """Does this policy REQUIRE real foot contact (a free-stand ground policy)?

    True -> the suspension/contact gates apply (refuse to start suspended, damp on
    contact loss). False (the PROVEN gantry policy) -> gates are inert, feet-off is
    intentional and in-distribution. Sources, in order: env override, then the meta
    key `requires_ground_contact` (a ground exporter sets it). Env ALLOW_SUSPENDED=1
    force-permits a flagged policy to run suspended for DELIBERATE gantry validation.
    """
    env = os.environ.get("REQUIRE_GROUND_CONTACT")
    if env is not None:
        return env not in ("0", "", "false", "False")
    return bool(getattr(meta, "requires_ground_contact", False))


def allow_suspended():
    """Explicit operator escape: run a ground-contact-required policy suspended
    anyway (gantry validation of a ground policy). Loudly warned by the caller."""
    return os.environ.get("ALLOW_SUSPENDED", "0") == "1"


# ============================================================================
# GUARD 5 — INDEPENDENT DAMPING WATCHDOG
# ============================================================================
# A SEPARATE thread that damps the robot if the main 50 Hz control loop stalls
# (misses its deadline by a margin) or raises a fault flag (NaN / invalid estimator
# / loss of contact). It is independent of the main loop so that a HUNG main loop
# still triggers damping — the main loop's own except/finally can only fire if the
# loop is still executing.
#
# INDEPENDENCE CAVEAT (be honest — this is safety-critical):
#   Python threads share the GIL. If the main loop hangs inside a BLOCKING call
#   that releases the GIL (DDS Read, time.sleep, onnxruntime inference — i.e. every
#   realistic stall mode of this loop) the watchdog thread WAKES and fires. But a
#   pure-Python CPU-bound infinite loop would hold the GIL and STARVE this thread.
#   The firmware remote B-damp remains the ultimate hard stop (there is no hardware
#   e-stop). Measure watchdog->damp latency on the gantry (see docs).
WATCHDOG_DEADLINE_S = float(os.environ.get("WATCHDOG_DEADLINE_S", "0.20"))  # 10x the 20ms tick
WATCHDOG_POLL_S = float(os.environ.get("WATCHDOG_POLL_S", "0.02"))
WATCHDOG_ENABLE = os.environ.get("WATCHDOG_ENABLE", "1") != "0"


class DampingWatchdog:
    """Independent damping watchdog.

    Usage:
        wd = DampingWatchdog(damp_cb)      # damp_cb: 0-arg callable that damps
        wd.start()
        ... each control tick: wd.beat()
        ... on a detected fault:  wd.raise_fault("reason")
        wd.stop()                          # in finally

    Fires damp_cb() exactly once (latched) when EITHER:
      * no beat() for longer than `deadline_s` (loop stalled), OR
      * raise_fault() was called (NaN / bad estimator / lost contact).
    `clock` is injectable for tests. Never raises out of the thread.
    """

    def __init__(self, damp_cb, *, deadline_s=None, poll_s=None,
                 clock=time.monotonic):
        self.damp_cb = damp_cb
        self.deadline_s = WATCHDOG_DEADLINE_S if deadline_s is None else float(deadline_s)
        self.poll_s = WATCHDOG_POLL_S if poll_s is None else float(poll_s)
        self._clock = clock
        self._last_beat = clock()
        self._fault = None
        self._fired = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self.fire_reason = None

    def beat(self):
        self._last_beat = self._clock()

    def raise_fault(self, reason="fault"):
        if self._fault is None:
            self._fault = str(reason)

    def start(self):
        if not WATCHDOG_ENABLE:
            return self
        self._thread = threading.Thread(target=self._run, name="damp-watchdog",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and threading.current_thread() is not t:
            t.join(timeout=1.0)

    @property
    def fired(self):
        return self._fired.is_set()

    def check_once(self):
        """One evaluation of the trip conditions (also used directly in tests).
        Returns the fire reason if it fired this call, else None."""
        if self._fired.is_set():
            return None
        reason = None
        if self._fault is not None:
            reason = f"fault: {self._fault}"
        else:
            late = self._clock() - self._last_beat
            if late > self.deadline_s:
                reason = f"main loop stalled {late*1e3:.0f} ms > {self.deadline_s*1e3:.0f} ms deadline"
        if reason is not None:
            self._fired.set()
            self.fire_reason = reason
            try:
                self.damp_cb()
            except Exception:  # noqa: BLE001 - watchdog must never raise
                pass
            return reason
        return None

    def _run(self):
        while not self._stop.is_set():
            if self.check_once() is not None:
                return   # latched; damp_cb fired
            time.sleep(self.poll_s)
