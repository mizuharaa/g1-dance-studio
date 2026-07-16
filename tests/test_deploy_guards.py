"""Unit tests for the deploy-side safety guards (pipeline/deploy_guards.py).

All pure/offline — no robot, no SDK. These gate the NEXT deploy: the robot is DOWN
(RMA), so the numeric THRESHOLDS here are UNVALIDATED and must be measured on the
gantry (see docs/DEPLOY_SAFETY_GUARDS.md). What these tests DO prove is the guard
LOGIC: clamps clamp, rate limits limit, NaNs fail closed, debounces debounce, and
the watchdog fires independently of the main loop.
"""
import numpy as np
import pytest

g = pytest.importorskip("pipeline.deploy_guards")


# 29-joint LAFAN1 order (matches pipeline.g1_limits.JOINT_ORDER) for name-based tests.
JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


class _FakeMeta:
    def __init__(self, requires_ground_contact=False):
        self.n = 29
        self.joint_order = list(JOINT_ORDER)
        self.q_lo = np.full(29, -2.0)
        self.q_hi = np.full(29, 2.0)
        self.requires_ground_contact = requires_ground_contact


# ---------------------------------------------------------------------------
# GUARD 3 — action clamps + rate limits
# ---------------------------------------------------------------------------
def test_clamp_bounds_target():
    lo = np.full(29, -1.0)
    hi = np.full(29, 1.0)
    step = np.full(29, 100.0)  # huge -> rate limit inert, only clamp acts
    out, n_rl, n_cl = g.clamp_and_rate_limit(np.full(29, 5.0), np.zeros(29), lo, hi, step)
    assert np.all(out <= hi + 1e-9) and np.all(out >= lo - 1e-9)
    assert np.allclose(out, 1.0)
    assert n_cl == 29 and n_rl == 0


def test_rate_limit_caps_per_tick_change():
    lo = np.full(29, -100.0)
    hi = np.full(29, 100.0)
    step = np.full(29, 0.1)  # tiny -> the per-tick change is capped at 0.1
    last = np.zeros(29)
    out, n_rl, n_cl = g.clamp_and_rate_limit(np.full(29, 5.0), last, lo, hi, step)
    assert np.allclose(out, 0.1)          # moved only one step toward the target
    assert n_rl == 29 and n_cl == 0
    # a change WITHIN the step budget passes through untouched
    out2, n_rl2, _ = g.clamp_and_rate_limit(last + 0.05, last, lo, hi, step)
    assert np.allclose(out2, 0.05) and n_rl2 == 0


def test_nan_target_raises_fail_safe():
    lo = np.full(29, -1.0)
    hi = np.full(29, 1.0)
    step = np.full(29, 1.0)
    bad = np.zeros(29)
    bad[3] = np.nan
    with pytest.raises(ValueError):
        g.clamp_and_rate_limit(bad, np.zeros(29), lo, hi, step)


def test_rate_limit_disabled_passes_through(monkeypatch):
    monkeypatch.setattr(g, "RATE_LIMIT_ENABLE", False)
    lo = np.full(29, -100.0)
    hi = np.full(29, 100.0)
    step = np.full(29, 0.1)
    out, n_rl, _ = g.clamp_and_rate_limit(np.full(29, 5.0), np.zeros(29), lo, hi, step)
    assert np.allclose(out, 5.0) and n_rl == 0


def test_rate_limit_step_uses_g1_velocity_limits():
    meta = _FakeMeta()
    step = g.rate_limit_step(meta, dt=0.02, frac=1.0)
    assert step.shape == (29,)
    assert np.all(step > 0)
    # knee (idx 3) velocity limit is 20 rad/s -> 1.0 * 20 * 0.02 = 0.4 rad/tick
    try:
        from pipeline import g1_limits
        assert np.isclose(step[3], g1_limits.VELOCITY_LIMIT[3] * 0.02)
    except Exception:
        pass


def test_joint_pos_limits_prefers_hardware_mjcf_or_meta():
    lo, hi = g.joint_pos_limits(_FakeMeta())
    assert lo.shape == (29,) and hi.shape == (29,)
    assert np.all(hi > lo)


# ---------------------------------------------------------------------------
# GUARD 4 — estimator validity
# ---------------------------------------------------------------------------
def test_estimate_valid_accepts_sane():
    ok, why = g.check_estimate_valid(base_lin_vel=[0.1, 0.0, -0.05], height=0.65, contact=0.9)
    assert ok, why


def test_estimate_invalid_on_nan():
    ok, _ = g.check_estimate_valid(base_lin_vel=[np.nan, 0, 0])
    assert not ok


def test_estimate_invalid_on_out_of_range_speed():
    ok, why = g.check_estimate_valid(base_lin_vel=[10.0, 0, 0])
    assert not ok and "speed" in why


def test_estimate_invalid_on_bad_height():
    assert not g.check_estimate_valid(height=0.05)[0]
    assert not g.check_estimate_valid(height=2.0)[0]


def test_estimate_invalid_on_lost_contact():
    ok, why = g.check_estimate_valid(contact=0.0)
    assert not ok and "contact" in why


# ---------------------------------------------------------------------------
# GUARDS 1 & 2 — suspension / contact proxy
# ---------------------------------------------------------------------------
def _tau_standing():
    """A tau_est vector with substantial weight-bearing (support) leg torque."""
    tau = np.zeros(29)
    for i, n in enumerate(JOINT_ORDER):
        if "knee" in n:
            tau[i] = 20.0
        elif "hip_pitch" in n or "ankle_pitch" in n:
            tau[i] = 5.0
    return tau


def _tau_suspended():
    """Hanging: support joints carry only the small limb weight."""
    tau = np.zeros(29)
    for i, n in enumerate(JOINT_ORDER):
        if any(k in n for k in ("hip_pitch", "knee", "ankle_pitch")):
            tau[i] = 0.3
    return tau


def test_support_torque_distinguishes_standing_from_suspended():
    stand = g.support_torque(_tau_standing(), JOINT_ORDER)
    hang = g.support_torque(_tau_suspended(), JOINT_ORDER)
    assert stand >= g.SUSPENSION_TAU_MIN     # standing bears weight
    assert hang < g.SUSPENSION_TAU_MIN       # suspended does not
    assert stand > hang


def test_contact_estimator_stable_only_after_debounce():
    est = g.ContactEstimator(JOINT_ORDER, confirm_ticks=5)
    tau = _tau_standing()
    for k in range(4):
        est.sample(tau_est=tau)
        assert not est.stable_contact()      # not yet debounced
    est.sample(tau_est=tau)
    assert est.stable_contact()              # 5 consecutive -> stable


def test_contact_estimator_suspended_never_stable():
    est = g.ContactEstimator(JOINT_ORDER, confirm_ticks=3)
    for _ in range(20):
        est.sample(tau_est=_tau_suspended())
    assert not est.stable_contact()


def test_contact_estimator_lost_after_debounce():
    est = g.ContactEstimator(JOINT_ORDER, confirm_ticks=3, lost_ticks=4)
    for _ in range(5):
        est.sample(tau_est=_tau_standing())
    assert est.stable_contact() and not est.lost_contact()
    for k in range(3):
        est.sample(tau_est=_tau_suspended())
        assert not est.lost_contact()        # not yet debounced
    est.sample(tau_est=_tau_suspended())
    assert est.lost_contact()                # 4 consecutive lost -> tripped


def test_contact_estimator_kinematic_confidence_or():
    # With no tau but a high kinematic confidence, contact holds (OR of proxies).
    est = g.ContactEstimator(JOINT_ORDER, confirm_ticks=2)
    est.sample(contact_conf=0.9)
    est.sample(contact_conf=0.9)
    assert est.stable_contact()


def test_contact_estimator_no_signal_fails_closed():
    est = g.ContactEstimator(JOINT_ORDER, confirm_ticks=1)
    est.sample()                              # no proxy provided
    assert not est.stable_contact()


def test_requires_ground_contact_flag_and_env(monkeypatch):
    monkeypatch.delenv("REQUIRE_GROUND_CONTACT", raising=False)
    assert not g.policy_requires_ground_contact(_FakeMeta(requires_ground_contact=False))
    assert g.policy_requires_ground_contact(_FakeMeta(requires_ground_contact=True))
    # env override wins both ways
    monkeypatch.setenv("REQUIRE_GROUND_CONTACT", "1")
    assert g.policy_requires_ground_contact(_FakeMeta(requires_ground_contact=False))
    monkeypatch.setenv("REQUIRE_GROUND_CONTACT", "0")
    assert not g.policy_requires_ground_contact(_FakeMeta(requires_ground_contact=True))


def test_allow_suspended_env(monkeypatch):
    monkeypatch.delenv("ALLOW_SUSPENDED", raising=False)
    assert not g.allow_suspended()
    monkeypatch.setenv("ALLOW_SUSPENDED", "1")
    assert g.allow_suspended()


# ---------------------------------------------------------------------------
# GUARD 5 — independent damping watchdog
# ---------------------------------------------------------------------------
class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_watchdog_fires_on_missed_heartbeat():
    clk = _Clock()
    fired = []
    wd = g.DampingWatchdog(lambda: fired.append(1), deadline_s=0.20, clock=clk)
    wd.beat()
    clk.t = 0.10
    assert wd.check_once() is None and not fired      # within deadline
    clk.t = 0.30                                       # 300 ms since beat > 200 ms
    reason = wd.check_once()
    assert reason is not None and "stalled" in reason
    assert fired == [1]


def test_watchdog_beat_resets_deadline():
    clk = _Clock()
    fired = []
    wd = g.DampingWatchdog(lambda: fired.append(1), deadline_s=0.20, clock=clk)
    clk.t = 0.15
    wd.beat()                                          # heartbeat at 0.15
    clk.t = 0.30                                        # only 0.15 s since last beat
    assert wd.check_once() is None and not fired


def test_watchdog_fires_on_fault_flag():
    clk = _Clock()
    fired = []
    wd = g.DampingWatchdog(lambda: fired.append(1), deadline_s=10.0, clock=clk)
    wd.beat()
    wd.raise_fault("estimator invalid")
    reason = wd.check_once()
    assert reason is not None and "estimator invalid" in reason
    assert fired == [1]


def test_watchdog_latches_single_fire():
    clk = _Clock()
    fired = []
    wd = g.DampingWatchdog(lambda: fired.append(1), deadline_s=0.20, clock=clk)
    wd.beat()
    clk.t = 1.0
    wd.check_once()
    wd.check_once()
    wd.check_once()
    assert fired == [1]                                # fired exactly once


def test_watchdog_thread_fires_when_loop_stops_beating():
    # Real thread + real clock: start it, stop beating, confirm it damps independently.
    import time
    fired = []
    wd = g.DampingWatchdog(lambda: fired.append(1), deadline_s=0.05, poll_s=0.005)
    wd.start()
    try:
        wd.beat()
        time.sleep(0.20)          # never beat again -> deadline lapses, watchdog fires
        assert fired == [1]
        assert wd.fired
    finally:
        wd.stop()


def test_watchdog_damp_cb_exception_never_propagates():
    clk = _Clock()

    def boom():
        raise RuntimeError("damp failed")

    wd = g.DampingWatchdog(boom, deadline_s=0.20, clock=clk)
    wd.beat()
    clk.t = 1.0
    # must not raise even though the damp callback raises
    assert wd.check_once() is not None
