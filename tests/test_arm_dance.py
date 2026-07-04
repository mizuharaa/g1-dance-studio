"""Offline tests for the ARM-DANCE-OVER-ONBOARD-BALANCE runtime — no robot, no SDK.

pipeline/arm_dance_runtime.py streams the dance's ARM joints over Unitree's rt/arm_sdk
weight-blend while the onboard controller keeps balance. These tests pin down exactly
the properties that keep that safe without hardware:
  * the name->DDS mapping can only ever address the 14 arm motors (15..28), never legs;
  * the weight profile is a smooth monotonic 0->1 (and 1->0) blend — the soft handoff;
  * the first streamed frame equals the captured current pose — no engagement lurch;
  * the refusal gates (watch flag, CONFIRMED_BY_HUMAN, --max-secs, ARM_FULL_RUN);
  * send_arm_cmd writes the weight slot + arm motors ONLY (verified with fakes).

unitree_sdk2py is NOT needed: the runtime imports it lazily (like deploy_runtime), so
the module import is pure numpy. Skips cleanly if the pipeline module can't import.
"""
import numpy as np
import pytest

ad = pytest.importorskip("pipeline.arm_dance_runtime")
dr = pytest.importorskip("pipeline.deploy_runtime")

HAVE_ARTIFACTS = ad.DEFAULT_META.exists() and ad.DEFAULT_MOTION.exists()
needs_artifacts = pytest.mark.skipif(not HAVE_ARTIFACTS,
                                     reason="staged thriller policy artifacts absent")


def _meta():
    return dr.Meta(ad.DEFAULT_META)


# ---- joint index mapping -------------------------------------------------------------
@needs_artifacts
def test_arm_joint_map_is_the_14_arm_joints():
    rows = ad.arm_joint_map(_meta())
    assert len(rows) == 14
    names = [n for _, _, n in rows]
    assert all(any(k in n for k in ("shoulder", "elbow", "wrist")) for n in names)
    # no leg/waist joint can ever be in the map
    assert not any(("hip" in n) or ("knee" in n) or ("ankle" in n) or ("waist" in n)
                   for n in names)
    # DDS side must be exactly motors 15..28 (G1_29_JointArmIndex)
    assert sorted(mi for _, mi, _ in rows) == list(range(15, 29))


@needs_artifacts
def test_arm_joint_map_matches_meta_order_identity():
    """For the staged thriller meta, joint_order matches DDS order 1:1 — the map must
    reproduce that (npz column == DDS motor index), and be built BY NAME so it would
    catch a reorder instead of silently miswiring."""
    rows = ad.arm_joint_map(_meta())
    for col, mi, _name in rows:
        assert col == mi


def test_arm_joint_map_refuses_wrong_count():
    class FakeMeta:
        joint_order = ["left_shoulder_pitch_joint", "left_elbow_joint"]  # only 2 arms
    with pytest.raises(SystemExit):
        ad.arm_joint_map(FakeMeta())


def test_dds_motor_index_table_is_complete_29():
    assert len(ad.G1_DDS_MOTOR_INDEX) == 29
    assert sorted(ad.G1_DDS_MOTOR_INDEX.values()) == list(range(29))
    assert ad.ARM_SDK_WEIGHT_IDX == 29  # kNotUsedJoint0 — the blend-weight slot


# ---- weight ramp profile -------------------------------------------------------------
def test_weight_ramp_up_monotonic_0_to_1():
    w = ad.weight_profile(0.0, 1.0, 2.0)
    assert w[0] == 0.0 and w[-1] == 1.0
    assert np.all(np.diff(w) >= 0)
    assert len(w) == int(2.0 * dr.CONTROL_HZ) + 1


def test_weight_ramp_down_monotonic_1_to_0():
    w = ad.weight_profile(1.0, 0.0, 1.5)
    assert w[0] == 1.0 and w[-1] == 0.0
    assert np.all(np.diff(w) <= 0)


def test_weight_ramp_from_partial_weight():
    """The exit path must ramp from WHEREVER the weight currently is (e.g. Ctrl-C
    mid-engage) down to 0."""
    w = ad.weight_profile(0.37, 0.0, 1.5)
    assert w[0] == pytest.approx(0.37) and w[-1] == 0.0
    assert np.all(np.diff(w) <= 1e-12)


def test_weight_ramp_never_leaves_0_1():
    for w in (ad.weight_profile(0, 1, 2.0), ad.weight_profile(1, 0, 1.5)):
        assert np.all(w >= 0.0) and np.all(w <= 1.0)


# ---- no-lurch engagement -------------------------------------------------------------
def test_cosine_blend_first_row_equals_start_exactly():
    """First streamed frame == current pose (bitwise) — the no-lurch guarantee."""
    q0 = np.random.uniform(-1, 1, 14)
    q1 = np.random.uniform(-1, 1, 14)
    b = ad.cosine_blend(q0, q1, 2.0)
    assert np.array_equal(b[0], q0)
    assert np.allclose(b[-1], q1, atol=1e-12)
    # monotonic progress along the line q0->q1 (cosine easing never overshoots)
    a = (b - q0[None, :]) @ (q1 - q0) / np.dot(q1 - q0, q1 - q0)
    assert np.all(np.diff(a) >= -1e-12) and a[-1] == pytest.approx(1.0)


# ---- trajectory extraction -----------------------------------------------------------
@needs_artifacts
def test_extract_arm_trajectory_shape_and_finite():
    meta = _meta()
    traj, rows = ad.extract_arm_trajectory(meta, ad.DEFAULT_MOTION)
    assert traj.shape == (2589, 14)
    assert np.all(np.isfinite(traj))
    assert ad.max_arm_speed(traj) <= ad.MAX_ARM_SPEED_RAD_S


@needs_artifacts
def test_dance_frame0_matches_default_arm_pose():
    """thriller_deploy embeds a 2.5s activation ramp FROM default, so its frame 0 must
    sit on the default arm pose — the approach blend then has ~nothing to do."""
    meta = _meta()
    traj, rows = ad.extract_arm_trajectory(meta, ad.DEFAULT_MOTION)
    cols = [c for c, _, _ in rows]
    assert np.max(np.abs(traj[0] - meta.default[cols])) < 1e-3


def test_extract_refuses_nonfinite(tmp_path):
    meta = _meta() if HAVE_ARTIFACTS else None
    if meta is None:
        pytest.skip("needs meta for joint order")
    jp = np.zeros((10, 29))
    jp[3, 20] = np.nan
    p = tmp_path / "bad.npz"
    np.savez(p, joint_pos=jp)
    with pytest.raises(SystemExit):
        ad.extract_arm_trajectory(meta, p)


def test_extract_refuses_wrong_shape(tmp_path):
    if not HAVE_ARTIFACTS:
        pytest.skip("needs meta for joint order")
    p = tmp_path / "bad.npz"
    np.savez(p, joint_pos=np.zeros((10, 12)))
    with pytest.raises(SystemExit):
        ad.extract_arm_trajectory(_meta(), p)


# ---- refusal gates -------------------------------------------------------------------
def test_gate_refuses_without_watch_flag():
    with pytest.raises(SystemExit):
        ad.require_arm_run_gates(False, 5.0, env={"CONFIRMED_BY_HUMAN": "alois"})


def test_gate_refuses_without_human_env():
    with pytest.raises(SystemExit):
        ad.require_arm_run_gates(True, 5.0, env={})
    with pytest.raises(SystemExit):
        ad.require_arm_run_gates(True, 5.0, env={"CONFIRMED_BY_HUMAN": "someone_else"})


def test_gate_refuses_without_max_secs():
    with pytest.raises(SystemExit):
        ad.require_arm_run_gates(True, None, env={"CONFIRMED_BY_HUMAN": "alois"})


def test_gate_refuses_negative_max_secs():
    with pytest.raises(SystemExit):
        ad.require_arm_run_gates(True, -1.0, env={"CONFIRMED_BY_HUMAN": "alois"})


def test_gate_full_run_needs_env():
    env = {"CONFIRMED_BY_HUMAN": "alois"}
    with pytest.raises(SystemExit):
        ad.require_arm_run_gates(True, 0, env=env)          # 0 = full dance: refused
    env["ARM_FULL_RUN"] = "1"
    ad.require_arm_run_gates(True, 0, env=env)              # now allowed
    ad.require_arm_run_gates(True, 5.0, env={"CONFIRMED_BY_HUMAN": "alois"})  # capped: ok


# ---- max-secs math -------------------------------------------------------------------
def test_dance_ticks_math():
    assert ad.dance_ticks(2589, 5.0) == 250          # 5s * 50Hz
    assert ad.dance_ticks(2589, 0) == 2589           # full dance
    assert ad.dance_ticks(2589, 1e9) == 2589         # capped at T
    assert ad.dance_ticks(2589, 0.001) == 1          # never zero ticks
    with pytest.raises(ValueError):
        ad.dance_ticks(2589, None)


# ---- gains ---------------------------------------------------------------------------
@needs_artifacts
def test_arm_gains_meta_default():
    meta = _meta()
    rows = ad.arm_joint_map(meta)
    kp, kd = ad.arm_gains(meta, rows)
    cols = [c for c, _, _ in rows]
    assert np.allclose(kp, meta.kp[cols] * ad.ARM_KP_SCALE)
    assert np.allclose(kd, meta.kd[cols] * ad.ARM_KP_SCALE)
    assert np.all(kp > 0) and np.all(kd > 0)


@needs_artifacts
def test_arm_gains_teleop_preset(monkeypatch):
    monkeypatch.setattr(ad, "ARM_GAINS", "teleop")
    meta = _meta()
    rows = ad.arm_joint_map(meta)
    kp, kd = ad.arm_gains(meta, rows)
    for (_, _, name), kpi, kdi in zip(rows, kp, kd):
        if "wrist" in name:
            assert kpi == pytest.approx(40.0 * ad.ARM_KP_SCALE)
            assert kdi == pytest.approx(1.5 * ad.ARM_KP_SCALE)
        else:
            assert kpi == pytest.approx(80.0 * ad.ARM_KP_SCALE)
            assert kdi == pytest.approx(3.0 * ad.ARM_KP_SCALE)


# ---- send_arm_cmd with fakes (no SDK): weight slot + arm motors ONLY -------------------
class _FakeMotorCmd:
    def __init__(self):
        self.mode = 0
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.kp = 0.0
        self.kd = 0.0


class _FakeLowCmd:
    def __init__(self):
        self.mode_pr = -1
        self.mode_machine = -1
        self.motor_cmd = [_FakeMotorCmd() for _ in range(35)]
        self.crc = 0


class _FakeCRC:
    def Crc(self, msg):  # noqa: N802 - SDK naming
        return 0xC0FFEE


class _FakePub:
    def __init__(self):
        self.writes = 0

    def Write(self, msg):  # noqa: N802 - SDK naming
        self.writes += 1


def _fake_send(targets, weight, kp=None, kd=None):
    pub, cmd, crc = _FakePub(), _FakeLowCmd(), _FakeCRC()
    dds_idx = list(range(15, 29))
    n = len(dds_idx)
    kp = np.full(n, 14.3) if kp is None else kp
    kd = np.full(n, 0.91) if kd is None else kd
    q_lo, q_hi = np.full(n, -2.0), np.full(n, 2.0)
    ad.send_arm_cmd(pub, cmd, crc, 5, dds_idx, targets, kp, kd, q_lo, q_hi, weight)
    return pub, cmd


def test_send_arm_cmd_writes_weight_slot_and_arms_only():
    targets = np.linspace(-0.5, 0.5, 14)
    pub, cmd = _fake_send(targets, 0.7)
    assert pub.writes == 1
    assert cmd.crc == 0xC0FFEE
    assert cmd.mode_pr == ad.PR_MODE and cmd.mode_machine == 5
    # weight slot (kNotUsedJoint0 = 29)
    assert cmd.motor_cmd[29].q == pytest.approx(0.7)
    # arm motors 15..28 commanded
    for k, mi in enumerate(range(15, 29)):
        mc = cmd.motor_cmd[mi]
        assert mc.mode == 1
        assert mc.q == pytest.approx(targets[k])
        assert mc.kp > 0 and mc.kd > 0
        assert mc.tau == 0.0 and mc.dq == 0.0
    # legs (0..11) and waist (12..14) NEVER touched — onboard owns them
    for mi in range(0, 15):
        mc = cmd.motor_cmd[mi]
        assert mc.mode == 0 and mc.q == 0.0 and mc.kp == 0.0 and mc.kd == 0.0


def test_send_arm_cmd_clamps_targets_and_weight():
    targets = np.full(14, 99.0)      # way outside the band
    _, cmd = _fake_send(targets, 3.0)
    for mi in range(15, 29):
        assert cmd.motor_cmd[mi].q == pytest.approx(2.0)   # clamped to q_hi
    assert cmd.motor_cmd[29].q == pytest.approx(1.0)       # weight clamped to [0,1]
    _, cmd = _fake_send(np.full(14, -99.0), -0.5)
    for mi in range(15, 29):
        assert cmd.motor_cmd[mi].q == pytest.approx(-2.0)
    assert cmd.motor_cmd[29].q == 0.0


# ---- safety constants pinned ----------------------------------------------------------
def test_runtime_never_uses_lowcmd_topic():
    """The whole design: arms via rt/arm_sdk, onboard balance untouched. The module
    must not reference the raw low-level command topic anywhere."""
    import inspect
    src = inspect.getsource(ad)
    assert ad.ARM_SDK_TOPIC == "rt/arm_sdk"
    assert '"rt/lowcmd"' not in src
    assert "ReleaseMode(" not in src          # never a call (docstring may mention it)
    assert "MotionSwitcherClient()" not in src


def test_control_rate_matches_deploy_runtime():
    """Music sync depends on identical 50 Hz wall-clock pacing (1 npz frame per tick)."""
    assert dr.CONTROL_HZ == 50.0
    assert ad.CONTROL_HZ == dr.CONTROL_HZ
