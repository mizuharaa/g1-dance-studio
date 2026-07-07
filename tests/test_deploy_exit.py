"""Offline tests for the OPT-IN end-of-run 'stand' exit (--exit stand) — no robot, no SDK.

The show path (demo.sh -> tools/show_run.sh -> deploy_runtime ground-run-legodom) ends,
on a CLEAN full dance, by fading kp->0 into damping. That leaves the known end-of-run
catch-step. The --exit stand option instead keeps holding the final STANDING pose and
hands the robot back to the onboard balancer while it is still balanced.

Non-negotiable properties pinned here (all SDK/LowCmd calls are faked):
  (a) --exit damp (and the DEFAULT) is unchanged: clean completion still fades to damping;
  (b) --exit stand on a CLEAN completion holds the final pose, THEN restores onboard
      control, THEN stops publishing — in that order, and NEVER damps;
  (c) an ABORT (NaN/bad action) with --exit stand STILL damps immediately — the stand
      handoff is never reached on a fault;
  (d) the final-pose guard refuses --exit stand for a motion that does not end standing
      (incl. the real staged Thriller motion) and falls back to damp;
  (e) the CLI default is damp, and 'fly' is rejected.

unitree_sdk2py is NOT needed: deploy_runtime imports it lazily, so the module import is
pure numpy (same pattern as the other deploy tests).
"""
import inspect
import os

import numpy as np
import pytest

dr = pytest.importorskip("pipeline.deploy_runtime")

HAVE_ARTIFACTS = dr.DEFAULT_META.exists() and dr.DEFAULT_MOTION.exists()
needs_artifacts = pytest.mark.skipif(not HAVE_ARTIFACTS, reason="staged policy artifacts absent")

LEG_IDX = dr.LEG_JOINT_IDX


class _ModeExit(BaseException):
    """Stands in for os._exit / process teardown so a faked motion mode can unwind out to
    the test instead of terminating the interpreter."""


# ---- lightweight SDK fakes -----------------------------------------------------------
class _FakeMotor:
    def __init__(self):
        self.q = self.dq = self.tau_est = 0.0
        self.temperature = [40.0, 40.0]


class _FakeMsg:
    def __init__(self):
        self.mode_machine = 5
        self.motor_state = [_FakeMotor() for _ in range(29)]


class _FakeLegOdom:
    def __init__(self, joint_order, *a, **k):
        self.joint_order = joint_order

    def reset_filter(self):
        pass

    def estimate(self, q, dq, R_base, gyro):
        return np.zeros(3), 0.8, None

    def gravity_comp(self, q_target, R_base=None):
        return np.zeros(29)


# ======================================================================================
# (d) final-pose GUARD + exit-mode resolution
# ======================================================================================
class _Ref:
    """Minimal Reference stand-in: only .jp is read by the guard."""
    def __init__(self, jp):
        self.jp = np.asarray(jp, float)


@needs_artifacts
def test_guard_accepts_a_motion_that_ends_standing():
    meta = dr.Meta(dr.DEFAULT_META)
    ref = _Ref(np.tile(meta.default, (8, 1)))            # final frame == default pose
    assert dr._final_pose_is_standing(meta, ref) is True
    assert dr._resolve_exit_mode("stand", meta, ref) == "stand"


@needs_artifacts
def test_guard_refuses_a_non_standing_motion_and_falls_back_to_damp():
    meta = dr.Meta(dr.DEFAULT_META)
    jp = np.tile(meta.default, (8, 1)).copy()
    jp[-1, 3] += 0.30                                     # left knee 0.30 rad > 0.15 tol
    ref = _Ref(jp)
    assert dr._final_pose_is_standing(meta, ref) is False
    assert dr._resolve_exit_mode("stand", meta, ref) == "damp"     # refused -> damp


@needs_artifacts
def test_guard_boundary_is_the_tolerance():
    meta = dr.Meta(dr.DEFAULT_META)
    within = _Ref(np.tile(meta.default, (3, 1)))
    within.jp[-1, 5] += dr.STAND_GUARD_TOL_RAD - 1e-6
    assert dr._final_pose_is_standing(meta, within) is True
    over = _Ref(np.tile(meta.default, (3, 1)))
    over.jp[-1, 5] += dr.STAND_GUARD_TOL_RAD + 1e-3
    assert dr._final_pose_is_standing(meta, over) is False


@needs_artifacts
def test_damp_request_never_becomes_stand():
    meta = dr.Meta(dr.DEFAULT_META)
    standing = _Ref(np.tile(meta.default, (4, 1)))
    assert dr._resolve_exit_mode("damp", meta, standing) == "damp"
    assert dr._resolve_exit_mode("anything-else", meta, standing) == "damp"


@needs_artifacts
def test_real_thriller_motion_does_not_end_standing_so_stand_is_refused():
    """MEASUREMENT (2026-07-07): the staged Thriller deploy motion ends ~39 deg off the
    default standing pose at the elbows/knees, so --exit stand is correctly refused for the
    current show and falls back to damp. The motion must be authored to end standing before
    --exit stand can apply to the frozen show path."""
    meta = dr.Meta(dr.DEFAULT_META)
    ref = dr.Reference(dr.DEFAULT_MOTION)
    dev = np.abs(ref.jp[-1] - meta.default)
    assert dev.max() > dr.STAND_GUARD_TOL_RAD                       # it does NOT end standing
    assert dr._final_pose_is_standing(meta, ref) is False
    assert dr._resolve_exit_mode("stand", meta, ref) == "damp"


# ======================================================================================
# (b) unit: the stand handoff holds the final pose, THEN restores, THEN exits — no damp
# ======================================================================================
@needs_artifacts
def test_stand_handoff_holds_then_restores_then_exits_and_never_damps(monkeypatch):
    meta = dr.Meta(dr.DEFAULT_META)
    ref = _Ref(np.tile(meta.default, (10, 1)))            # ends standing (== default)
    order, sends = [], []

    def _fake_send(*a, **k):
        target, kp = np.asarray(a[4], float), np.asarray(a[5], float)
        sends.append((target.copy(), float(kp.flat[0]), bool(k.get("damping", False))))
        order.append("send")

    monkeypatch.setattr(dr, "_TELEM", None)
    monkeypatch.setattr(dr, "HANDOFF_HOLD_S", 0.2)                  # 0.2s * 50Hz = 10 holds
    monkeypatch.setattr(dr, "HANDOFF_OVERLAP_S", 0.1)              # 0.1s * 50Hz = 5 overlap holds
    monkeypatch.setattr(dr, "_send_cmd", _fake_send)
    monkeypatch.setattr(dr.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(dr, "_restore_motion_service", lambda *a, **k: order.append("restore"))
    monkeypatch.setattr(dr, "_damp", lambda *a, **k: order.append("damp"))          # must NOT fire
    monkeypatch.setattr(dr, "_damp_burst", lambda *a, **k: order.append("damp_burst"))

    def _exit(code=0):
        order.append("exit")
        raise _ModeExit
    monkeypatch.setattr(dr.os, "_exit", _exit)

    # Pass BOOSTED holding gains (what legodom hands in) — the handoff must use exactly them.
    kp_boost, kd_boost = meta.kp * 3.0, meta.kd * 3.0
    with pytest.raises(_ModeExit):
        dr._stand_handoff_and_exit(None, None, None, 5, meta, ref, kp_boost, kd_boost)

    # HANDOFF_HOLD_S (10) holds BEFORE restore, then HANDOFF_OVERLAP_S (5) overlap holds AFTER
    assert order.count("send") == 15
    ri = order.index("restore")
    assert order[:ri].count("send") == 10          # 10 hold sends before handing off
    assert order[ri:].count("send") == 5           # 5 overlap sends bridge the takeover
    # ORDER: restore happens after the hold, overlap sends after restore, exit last
    assert ri < order.index("exit")
    assert max(i for i, x in enumerate(order) if x == "send") < order.index("exit")
    # NEVER damped on the stand path
    assert "damp" not in order and "damp_burst" not in order
    # every command held the FINAL standing pose at the PASSED holding gains (not damping)
    for target, kp0, damping in sends:
        assert damping is False
        assert np.allclose(target, meta.default)                   # ref.jp[-1] == default
        assert kp0 == pytest.approx(meta.kp[0] * 3.0)


# ======================================================================================
# mode driver: run ground-run-legodom (the show path) with every SDK call faked
# ======================================================================================
def _drive_legodom(monkeypatch, *, exit_mode, abort=False, spy_stand=True,
                   leg_scale=1.0, max_secs=0.1):
    """Drive mode_ground_run_legodom end-to-end with fakes. Returns (events, sends, captured).
    events: ordered high-level exit markers (stand/damp/finalize).
    sends:  (kp0, damping) for each policy/damp-ramp _send_cmd.
    captured: kp/kd handed to the stand handoff (when spy_stand)."""
    import pipeline.leg_odometry as lo
    meta = dr.Meta(dr.DEFAULT_META)
    ref = dr.Reference(dr.DEFAULT_MOTION)                 # fresh each call (align_yaw mutates)
    events, sends, captured = [], [], {}

    monkeypatch.setenv("CONFIRMED_BY_HUMAN", "alois")
    monkeypatch.setattr(lo, "LegOdometry", _FakeLegOdom)
    monkeypatch.setattr(dr, "GROUND_LEG_KP_SCALE", leg_scale)
    monkeypatch.setattr(dr, "make_dds", lambda *a, **k: None)
    monkeypatch.setattr(dr, "lowstate_subscriber", lambda *a, **k: object())
    monkeypatch.setattr(dr, "read_state",
                        lambda *a, **k: (meta.default.copy(), np.zeros(29),
                                         np.array([1.0, 0, 0, 0]), np.zeros(3), _FakeMsg()))
    monkeypatch.setattr(dr, "_release_motion_service", lambda *a, **k: None)
    monkeypatch.setattr(dr, "_lowcmd_setup", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(dr, "_install_damp_on_signals", lambda *a, **k: None)
    monkeypatch.setattr(dr, "_ramp_to_pose", lambda *a, **k: None)
    monkeypatch.setattr(dr, "_hold", lambda *a, **k: None)
    monkeypatch.setattr(dr.time, "sleep", lambda *a, **k: None)

    def _fake_send(*a, **k):
        sends.append((float(np.asarray(a[5], float).flat[0]), bool(k.get("damping", False))))
    monkeypatch.setattr(dr, "_send_cmd", _fake_send)

    def _policy(session, obs, tick):
        return np.full(29, np.nan) if abort else np.zeros(29)
    monkeypatch.setattr(dr, "run_policy", _policy)

    monkeypatch.setattr(dr, "_damp", lambda *a, **k: events.append("damp"))

    def _final(code=0):
        events.append("finalize")
        raise _ModeExit
    monkeypatch.setattr(dr, "_finalize_and_exit", _final)

    if spy_stand:
        def _stand(pub, low_cmd, crc, mm, meta_, ref_, kp=None, kd=None):
            captured["kp"], captured["kd"] = kp, kd
            events.append("stand")
            raise _ModeExit                              # stand in for os._exit inside the mode
        monkeypatch.setattr(dr, "_stand_handoff_and_exit", _stand)

    with pytest.raises(_ModeExit):
        dr.mode_ground_run_legodom(meta, object(), ref, "iface", True, max_secs, exit_mode)
    return events, sends, captured


# ---- (a) --exit damp / default: clean completion still ramps to damping, no stand -----
@needs_artifacts
def test_clean_completion_damp_ramps_to_damping_and_never_stands(monkeypatch):
    events, sends, _ = _drive_legodom(monkeypatch, exit_mode="damp")
    assert "stand" not in events                          # stand handoff NOT taken
    assert events == ["damp", "finalize"]                 # proven damp exit path
    # the smooth ramp-to-damping executed: kp fades to ~0 on the final send (unchanged path)
    assert sends[-1][0] == pytest.approx(0.0)
    assert sends[-1][1] is False                          # ramp uses kp*f, not the damping flag
    # and it actually faded (an early ramp/policy send had full stiffness)
    assert max(s[0] for s in sends) > 1.0


# ---- (b) routing: clean completion with --exit stand takes the stand handoff ----------
@needs_artifacts
def test_clean_completion_stand_routes_to_handoff(monkeypatch):
    events, _sends, _ = _drive_legodom(monkeypatch, exit_mode="stand")
    # the stand handoff is the FIRST post-loop action on a clean finish.
    # (In production _stand_handoff_and_exit calls os._exit here, so the finally-damp never
    # runs; the direct unit test above is the authoritative no-damp proof. Here the spy
    # raises a sentinel so the harness can unwind, which is why later markers still appear.)
    assert events[0] == "stand"
    assert "stand" in events


@needs_artifacts
def test_stand_handoff_receives_the_boosted_policy_gains(monkeypatch):
    """legodom must hand the SAME (boosted) leg gains it danced with to the handoff, so the
    robot does not sag when it stops damping — no gain discontinuity at the handoff."""
    meta = dr.Meta(dr.DEFAULT_META)
    _events, _sends, captured = _drive_legodom(monkeypatch, exit_mode="stand", leg_scale=1.5)
    assert captured["kp"] is not None and captured["kd"] is not None
    for i in LEG_IDX:                                     # sagittal weight-bearing legs boosted
        assert captured["kp"][i] == pytest.approx(meta.kp[i] * 1.5)
        assert captured["kd"][i] == pytest.approx(meta.kd[i] * 1.5)
    # a non-boosted joint (an arm) is untouched
    assert captured["kp"][15] == pytest.approx(meta.kp[15])


# ---- (c) SAFETY INVARIANT: an abort with --exit stand STILL damps, never stands -------
@needs_artifacts
def test_abort_with_exit_stand_still_damps_and_never_stands(monkeypatch):
    events, _sends, _ = _drive_legodom(monkeypatch, exit_mode="stand", abort=True)
    assert "stand" not in events                          # stand handoff NEVER reached on a fault
    assert "damp" in events                               # went straight to damping
    assert events == ["damp", "finalize"]


@needs_artifacts
def test_abort_with_exit_damp_also_damps(monkeypatch):
    events, _sends, _ = _drive_legodom(monkeypatch, exit_mode="damp", abort=True)
    assert events == ["damp", "finalize"] and "stand" not in events


# ======================================================================================
# (e) CLI default is damp; choices enforced; mode signatures default to damp
# ======================================================================================
def test_cli_default_exit_is_damp():
    a = dr._build_parser().parse_args([])
    assert a.exit_mode == "damp"


def test_cli_accepts_stand_and_rejects_unknown():
    assert dr._build_parser().parse_args(["--exit", "stand"]).exit_mode == "stand"
    with pytest.raises(SystemExit):
        dr._build_parser().parse_args(["--exit", "fly"])


def test_all_policy_modes_default_exit_to_damp():
    for name in ("mode_run", "mode_ground_run", "mode_ground_run_odom",
                 "mode_ground_run_legodom"):
        sig = inspect.signature(getattr(dr, name))
        assert sig.parameters["exit_mode"].default == "damp", name


# ======================================================================================
# constant: ARM_ACTION_CAP_SCALE default raised to 2.2 (v3-family envelope max 17.1)
# ======================================================================================
def test_arm_action_cap_scale_default_is_2_2():
    if "ARM_ACTION_CAP_SCALE" in os.environ:
        pytest.skip("ARM_ACTION_CAP_SCALE overridden in env")
    assert dr.ARM_ACTION_CAP_SCALE == pytest.approx(2.2)


# ======================================================================================
# FALL DETECTOR: physical-state trigger (torso topple) -> damp + onboard handoff
# ======================================================================================
def _pitch_R(deg):
    """Body->world rotation for a `deg` forward pitch; R[2,2]=cos(deg)=torso uprightness."""
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1.0, 0], [-np.sin(t), 0, np.cos(t)]])


def test_fall_signal_topple_and_choreography_relative_height(monkeypatch):
    monkeypatch.setattr(dr, "FALL_UPRIGHT_MIN", 0.35)
    monkeypatch.setattr(dr, "FALL_HEIGHT_DROP_M", 0.15)
    # upright + dance-scale lean + on-height -> no signal (real Thriller max tilt 35.7 deg)
    assert dr._fall_signal(np.eye(3), 0.7, 0.7, 0.0)[0] is False
    assert dr._fall_signal(_pitch_R(26), 0.7, 0.7, 0.0)[0] is False
    assert dr._fall_signal(_pitch_R(60), 0.7, 0.7, 0.0)[0] is False    # 0.50 uprightness, clear
    # topple -> signal
    assert dr._fall_signal(_pitch_R(80), 0.7, 0.7, 0.0)[0] is True
    # height collapse: upright but sunk 0.2 m below the choreographed height -> signal
    assert dr._fall_signal(np.eye(3), 0.5, 0.7, 0.0)[0] is True
    # an INTENTIONAL squat that matches the choreography (ref_dz also -0.2) -> NO signal
    assert dr._fall_signal(np.eye(3), 0.5, 0.7, -0.2)[0] is False


def test_fall_detector_debounces_single_tick(monkeypatch):
    monkeypatch.setattr(dr, "FALL_UPRIGHT_MIN", 0.35)
    monkeypatch.setattr(dr, "FALL_HEIGHT_DROP_M", 0.15)
    monkeypatch.setattr(dr, "FALL_CONFIRM_TICKS", 3)
    # a SINGLE toppled tick must NOT raise (else it would damp a healthy robot on one glitch)
    rt = dr._check_fall(0, _pitch_R(80), 0.7, 0.7, 0.0, 0); assert rt == 1
    rt = dr._check_fall(rt, _pitch_R(80), 0.7, 0.7, 0.0, 1); assert rt == 2
    # one good tick RESETS the counter (transient glitch cleared)
    assert dr._check_fall(rt, np.eye(3), 0.7, 0.7, 0.0, 2) == 0
    # only FALL_CONFIRM_TICKS in a row raise
    rt = dr._check_fall(0, _pitch_R(80), 0.7, 0.7, 0.0, 10)
    rt = dr._check_fall(rt, _pitch_R(80), 0.7, 0.7, 0.0, 11)
    with pytest.raises(RuntimeError, match="FALL DETECTED"):
        dr._check_fall(rt, _pitch_R(80), 0.7, 0.7, 0.0, 12)


def test_fall_detector_default_thresholds():
    if "FALL_UPRIGHT_MIN" in os.environ:
        pytest.skip("FALL_UPRIGHT_MIN overridden in env")
    assert dr.FALL_UPRIGHT_MIN == pytest.approx(0.35)
    assert dr.FALL_HEIGHT_DROP_M == pytest.approx(0.15)
    assert dr.FALL_CONFIRM_TICKS >= 2                       # debounce is on
    assert float(np.degrees(np.arccos(dr.FALL_UPRIGHT_MIN))) > 60.0


# ======================================================================================
# ENTRY handoff (onboard -> policy): pre-arm before release + catch the current pose
# ======================================================================================
@needs_artifacts
def test_entry_handoff_prearms_before_release_and_catches_current_pose(monkeypatch):
    """The onboard->policy takeover must not leave the robot unheld (fall risk untethered):
    our publisher + damp context + signal handler are armed BEFORE the onboard release, and
    the CURRENT pose is held for ENTRY_CATCH_S the instant onboard lets go (before the ramp
    to the ready pose). Mirror of the exit overlap."""
    import pipeline.leg_odometry as lo
    meta = dr.Meta(dr.DEFAULT_META)
    ref = dr.Reference(dr.DEFAULT_MOTION)
    q0 = meta.default.copy() + 0.3                       # DISTINCT current pose (not ready pose)
    events, holds = [], []

    monkeypatch.setenv("CONFIRMED_BY_HUMAN", "alois")
    monkeypatch.setattr(lo, "LegOdometry", _FakeLegOdom)
    monkeypatch.setattr(dr, "make_dds", lambda *a, **k: None)
    monkeypatch.setattr(dr, "lowstate_subscriber", lambda *a, **k: object())
    monkeypatch.setattr(dr, "read_state",
                        lambda *a, **k: (q0.copy(), np.zeros(29), np.array([1.0, 0, 0, 0]),
                                         np.zeros(3), _FakeMsg()))
    monkeypatch.setattr(dr, "ENTRY_CATCH_S", 0.4)
    monkeypatch.setattr(dr, "_lowcmd_setup",
                        lambda *a, **k: (events.append("setup") or (None, None, None)))
    monkeypatch.setattr(dr, "_release_motion_service", lambda *a, **k: events.append("release"))
    monkeypatch.setattr(dr, "_install_damp_on_signals", lambda *a, **k: events.append("signals"))

    def _fake_hold(pub, low_cmd, crc, mm, q, secs, kp, kd, meta_):
        events.append("hold"); holds.append((np.asarray(q, float).copy(), secs))
    monkeypatch.setattr(dr, "_hold", _fake_hold)
    monkeypatch.setattr(dr, "_ramp_to_pose",
                        lambda *a, **k: events.append("ramp"))
    monkeypatch.setattr(dr.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(dr, "_send_cmd", lambda *a, **k: None)
    monkeypatch.setattr(dr, "run_policy", lambda *a, **k: np.zeros(29))
    monkeypatch.setattr(dr, "_damp", lambda *a, **k: events.append("damp"))

    def _final(code=0):
        events.append("finalize"); raise _ModeExit
    monkeypatch.setattr(dr, "_finalize_and_exit", _final)

    with pytest.raises(_ModeExit):
        dr.mode_ground_run_legodom(meta, object(), ref, "iface", True, 0.05, "damp")

    # pre-arm: our controller + safety spine are up BEFORE the onboard release
    assert events.index("setup") < events.index("release")
    assert events.index("signals") < events.index("release")
    # entry catch: a hold AFTER release and BEFORE the ramp to the ready pose
    ri, rampi = events.index("release"), events.index("ramp")
    assert any(e == "hold" and ri < i < rampi for i, e in enumerate(events)), \
        "no entry catch-hold between release and ramp"
    # the FIRST hold catches the CURRENT pose q0 (not the ready/default pose), for ENTRY_CATCH_S
    caught_pose, caught_secs = holds[0]
    assert np.allclose(caught_pose, q0)
    assert caught_secs == pytest.approx(0.4)
    # the ENTRY is damp-free — nothing damps before the ramp to the ready pose (a trailing
    # damp is the exit_mode=damp handoff at the END, which is a separate concern)
    assert "damp" not in events[:rampi]
