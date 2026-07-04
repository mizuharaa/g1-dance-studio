"""Offline tests for the stage-0 READ-ONLY measurement kit (no robot, no SDK).

Covers: the analysis math (staleness stats, torque summary, thermal rate, the
kp*err delivery regression), the recorder + npz round-trip with fully mocked
SDK message objects, the strict read-only guarantee of the capture tool, and
the ANKLE_TRIM_DEG clamp/application logic added to deploy_runtime.
"""
import json
import types
from pathlib import Path

import numpy as np
import pytest

from deploy import capture_stage0 as cs

dr = pytest.importorskip("pipeline.deploy_runtime")

RNG = np.random.default_rng(7)


# ---- staleness stats -----------------------------------------------------------

def test_staleness_clean_50hz_stream():
    n = 500
    t = np.arange(n) * 0.020                      # perfect 20 ms reads
    tick = (np.arange(n) * 20).astype(np.uint32)  # tick in ms, perfectly fresh
    st = cs.staleness_stats(t, tick)
    assert st["n"] == n
    assert st["tick_unit_ms_est"] == pytest.approx(1.0, abs=1e-6)
    assert st["wall_read_interval_ms"]["p50"] == pytest.approx(20.0, abs=1e-6)
    assert st["tick_advance_ms"]["p50"] == pytest.approx(20.0, abs=1e-6)
    assert st["staleness_ms"]["p95"] < 0.01       # zero jitter -> ~0 staleness
    assert st["latency_beyond_2ms"] is False
    assert st["repeated_tick_fraction"] == 0.0


def test_staleness_detects_stale_reads_and_gaps():
    n = 500
    t = np.arange(n) * 0.020
    tick = (np.arange(n) * 20).astype(float)
    # inject staleness: 10 reads receive a tick 5 ms OLDER than the trend
    t2 = t.copy()
    t2[100:110] += 0.005
    st = cs.staleness_stats(t2, tick)
    assert st["staleness_ms"]["max"] == pytest.approx(5.0, abs=0.5)
    assert st["latency_beyond_2ms"] is True or st["staleness_ms"]["max"] > 2.0
    # a 60 ms wall gap shows in the read-interval max
    t3 = t.copy()
    t3[200:] += 0.040   # one interval becomes 60 ms
    st3 = cs.staleness_stats(t3, tick)
    assert st3["wall_read_interval_ms"]["max"] == pytest.approx(60.0, abs=1e-6)


def test_staleness_repeated_ticks_and_uint32_wrap():
    n = 200
    t = np.arange(n) * 0.020
    tick = np.repeat(np.arange(n // 2) * 40, 2).astype(float)  # every msg read twice
    st = cs.staleness_stats(t, tick)
    assert st["repeated_tick_fraction"] == pytest.approx(0.5, abs=0.01)
    # uint32 wraparound must not produce a negative advance
    tick_wrap = (np.arange(n, dtype=np.int64) * 20 + 2**32 - 1000) % 2**32
    st2 = cs.staleness_stats(t, tick_wrap.astype(float))
    assert st2["tick_unit_ms_est"] == pytest.approx(1.0, abs=1e-6)
    assert st2["tick_advance_ms"]["max"] == pytest.approx(20.0, abs=1e-6)


def test_staleness_too_few_samples():
    st = cs.staleness_stats([0.0, 0.02], [0, 20])
    assert "error" in st


# ---- torque summary --------------------------------------------------------------

def test_torque_summary_math():
    n = 1000
    tau = np.zeros((n, 29))
    tau[:, 4] = 15.0                                   # left_ankle_pitch constant
    tau[:, 9] = np.where(np.arange(n) % 2 == 0, 3.0, -3.0)  # right_knee alternating
    ts = cs.torque_summary(tau)
    la = ts["per_joint"]["left_ankle_pitch_joint"]
    assert la["mean_nm"] == pytest.approx(15.0)
    assert la["rms_nm"] == pytest.approx(15.0)
    assert la["p95_abs_nm"] == pytest.approx(15.0)
    rk = ts["per_joint"]["right_knee_joint"]
    assert rk["mean_nm"] == pytest.approx(0.0)
    assert rk["rms_nm"] == pytest.approx(3.0)
    # legs subset is exactly the 12 leg motors; ankles highlighted
    assert len(ts["legs"]) == 12
    assert set(ts["ankle_pitch"]) == {"left_ankle_pitch_joint", "right_ankle_pitch_joint"}


# ---- thermal rate -----------------------------------------------------------------

def test_thermal_rate_and_argmax_named():
    n = 300
    t = np.arange(n) * 1.0                 # 1 Hz for 5 min
    temp = np.full((n, 29), 35.0)
    temp[:, 10] = 35.0 + 2.0 * (t / 60.0)  # right_ankle_pitch heats 2 C/min
    temp[:, 3] = 55.0                      # left_knee hottest but flat
    th = cs.thermal_summary(temp, t)
    ra = th["per_motor"]["right_ankle_pitch_joint"]
    assert ra["rate_C_per_min"] == pytest.approx(2.0, rel=0.05)
    assert th["fastest_heating_motor"]["name"] == "right_ankle_pitch_joint"
    assert th["hottest_motor"]["name"] == "left_knee_joint"
    assert th["duration_min"] == pytest.approx((n - 1) / 60.0)


# ---- kp*err delivery regression ----------------------------------------------------

def _synth_pd_run(ratio=0.35, n=2000, noise=0.05):
    kp = np.full(29, 40.0)
    kd = np.full(29, 2.5)
    q = RNG.normal(0.0, 0.3, (n, 29))
    dq = RNG.normal(0.0, 0.5, (n, 29))
    target = q + RNG.normal(0.0, 0.2, (n, 29))
    x = kp[None, :] * (target - q) - kd[None, :] * dq
    tau = ratio * x + RNG.normal(0.0, noise, (n, 29))
    return q, dq, tau, target, kp, kd


def test_delivery_recovers_035_slope():
    q, dq, tau, target, kp, kd = _synth_pd_run(ratio=0.35)
    dl = cs.kp_err_delivery(q, dq, tau, target, kp, kd)
    for name in ("left_knee_joint", "right_knee_joint", "left_ankle_pitch_joint"):
        assert dl[name]["ratio"] == pytest.approx(0.35, abs=0.01)
        assert dl[name]["r2"] > 0.95


def test_delivery_full_and_undefined_cases():
    q, dq, tau, target, kp, kd = _synth_pd_run(ratio=1.0, noise=0.0)
    dl = cs.kp_err_delivery(q, dq, tau, target, kp, kd)
    assert dl["right_ankle_pitch_joint"]["ratio"] == pytest.approx(1.0, abs=1e-6)
    # a joint with ~zero commanded torque must report ratio=None, not a garbage slope
    q2 = np.zeros((100, 29)); dq2 = np.zeros((100, 29))
    tgt2 = np.zeros((100, 29)); tau2 = RNG.normal(0, 0.1, (100, 29))
    dl2 = cs.kp_err_delivery(q2, dq2, tau2, tgt2, kp, kd)
    assert dl2["left_knee_joint"]["ratio"] is None
    assert "undefined" in dl2["left_knee_joint"]["note"]


def test_effective_gains_corrects_stand_hold_and_legodom():
    d = {"kp": np.full(29, 10.0), "kd": np.full(29, 1.0)}
    kp, kd, notes = cs._effective_gains(d, {"mode": "stand-hold", "approach_kp_scale": 2.0})
    assert np.allclose(kp, 20.0) and np.allclose(kd, 2.0) and notes
    kp, kd, notes = cs._effective_gains(
        d, {"mode": "ground-run-legodom", "ground_leg_kp_scale": 1.5, "gravity_ff": True})
    assert kp[0] == pytest.approx(15.0) and kp[3] == pytest.approx(15.0)
    assert kp[1] == pytest.approx(10.0)      # roll joints untouched
    assert any("GRAVITY_FF" in n for n in notes)
    # plain run: unchanged
    kp, kd, notes = cs._effective_gains(d, {"mode": "run", "approach_kp_scale": 2.0})
    assert np.allclose(kp, 10.0) and notes == []


# ---- recorder + analyze round-trip with MOCKED SDK messages -------------------------

class _FakeMotor:
    def __init__(self, i, tau=0.0, temp=40):
        self.q = 0.01 * i
        self.dq = 0.0
        self.tau_est = tau
        self.temperature = [temp, temp - 5]   # int16[2] channel semantics


class _FakeIMU:
    quaternion = [1.0, 0.0, 0.0, 0.0]
    gyroscope = [0.0, 0.0, 0.0]
    accelerometer = [0.0, 0.0, 9.81]


class _FakeLowState:
    def __init__(self, tick, ankle_tau=5.0, temp=40):
        self.tick = tick
        self.imu_state = _FakeIMU()
        self.motor_state = [
            _FakeMotor(i, tau=(ankle_tau if i in (4, 10) else 0.5), temp=temp)
            for i in range(35)]   # LowState motor array is 35 long; we read 29


def test_recorder_roundtrip_and_full_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "TELEMETRY_DIR", tmp_path)
    rec = cs.Stage0Recorder("unit test/label", "lo", 0.1)
    assert "stage0_unit-test-label" in rec.path.name   # label sanitized
    n = 200
    for i in range(n):
        msg = _FakeLowState(tick=i * 20, ankle_tau=5.0, temp=40 + i * 0.01)
        odom = (np.array([0.0, 0.0, 0.72]), np.zeros(3), 100.0 + i * 0.02) \
            if i % 2 == 0 else None
        rec.add(msg, t_wall=1000.0 + i * 0.02, t_mono=i * 0.02, odom=odom)
    path = rec.save()
    assert path is not None and path.exists()

    summary = cs.analyze_npz(path, print_fn=lambda *_a, **_k: None)
    assert summary["n_rows"] == n
    # a) staleness present and clean
    assert summary["staleness"]["latency_beyond_2ms"] is False
    # b) ankle baseline visible
    ap = summary["torque"]["ankle_pitch"]["left_ankle_pitch_joint"]
    assert ap["mean_nm"] == pytest.approx(5.0)
    # c) thermal ~0.03 C per 4 s window -> ~0.3 C/min... check rate positive
    th = summary["thermal"]["per_motor"]["left_hip_pitch_joint"]
    assert th["rate_C_per_min"] > 0
    # odom recorded as optional (half the reads)
    assert summary["odom"]["fraction_present"] == pytest.approx(0.5, abs=0.01)
    # run_meta provenance embedded
    assert summary["run_meta"]["read_only"] is True
    assert "LowState_.tick" in summary["run_meta"]["lowstate_tick_field"]
    # analysis JSON written next to the npz and is valid JSON
    j = json.loads(path.with_suffix(".analysis.json").read_text())
    assert j["staleness"]["n"] == n


def test_recorder_add_never_raises():
    rec = cs.Stage0Recorder("x", "lo", 1)
    rec.add(object(), 0.0, 0.0)          # garbage msg -> swallowed, no rows
    assert not rec.rows["t_mono"]


def test_analyze_telemetry_run_npz(tmp_path):
    """--analyze on a deploy_runtime.Telemetry-format npz recovers the delivery slope."""
    q, dq, tau, target, kp, kd = _synth_pd_run(ratio=0.35)
    n = q.shape[0]
    path = tmp_path / "20260707-000000_ground-run-legodom.npz"
    np.savez_compressed(
        path, tick=np.arange(n), t=np.arange(n) * 0.02, stage=np.ones(n, int),
        q=q, dq=dq, tau_est=tau, temp=np.full((n, 29), 40.0),
        imu_quat=np.tile([1.0, 0, 0, 0], (n, 1)), gyro=np.zeros((n, 3)),
        action=np.zeros((n, 29)), target=target,
        joint_order=np.array(cs.G1_JOINT_ORDER), kp=kp, kd=kd,
        run_meta_json=np.array(json.dumps({"mode": "run"})))
    summary = cs.analyze_npz(path, print_fn=lambda *_a, **_k: None)
    assert summary["delivery"]["left_knee_joint"]["ratio"] == pytest.approx(0.35, abs=0.01)
    # Telemetry 'tick' is the loop index, NOT the LowState clock -> no staleness section
    assert "staleness" not in summary
    assert path.with_suffix(".analysis.json").exists()


# ---- READ-ONLY guarantee (source-level guard) ---------------------------------------

def test_capture_tool_is_strictly_read_only():
    """The capture tool must never gain a publish/command/mode-switch path.
    (Code tokens only — the docstring legitimately NAMES LowCmd etc. in prose.)"""
    src = Path(cs.__file__).read_text()
    for forbidden in ("ChannelPublisher", "LowCmd_", "MotionSwitcherClient(",
                      "SelectMode(", "ReleaseMode(", ".Write(", "rt/lowcmd"):
        assert forbidden not in src, f"READ-ONLY violation: '{forbidden}' in capture_stage0.py"


def test_capture_module_imports_without_sdk():
    # importing the module must not require unitree_sdk2py (lazy SDK imports only)
    import sys
    assert "deploy.capture_stage0" in sys.modules or cs is not None
    assert not any(m.startswith("unitree_sdk2py") for m in sys.modules)


# ---- ANKLE_TRIM_DEG (deploy_runtime stand-hold knob) --------------------------------

def _meta_stub():
    m = types.SimpleNamespace()
    m.default = np.linspace(-0.5, 0.5, 29)
    return m


def test_ankle_trim_default_is_zero_and_identity():
    meta = _meta_stub()
    assert dr.ANKLE_TRIM_DEG == 0.0          # default env -> no trim
    tgt, trim = dr._stand_hold_targets(meta)
    assert trim == 0.0
    assert np.allclose(tgt, meta.default)


def test_ankle_trim_applies_to_both_ankle_pitch_only():
    meta = _meta_stub()
    tgt, trim = dr._stand_hold_targets(meta, trim_deg=3.0)
    assert trim == 3.0
    for i in dr.ANKLE_PITCH_IDX:
        assert tgt[i] == pytest.approx(meta.default[i] + np.deg2rad(3.0))
    others = [i for i in range(29) if i not in dr.ANKLE_PITCH_IDX]
    assert np.allclose(tgt[others], meta.default[others])
    # negative sweep leg
    tgt2, trim2 = dr._stand_hold_targets(meta, trim_deg=-3.0)
    assert trim2 == -3.0
    assert tgt2[4] == pytest.approx(meta.default[4] - np.deg2rad(3.0))


def test_ankle_trim_clamped_to_pm6():
    meta = _meta_stub()
    tgt, trim = dr._stand_hold_targets(meta, trim_deg=45.0)
    assert trim == dr.ANKLE_TRIM_MAX_DEG == 6.0
    assert tgt[10] == pytest.approx(meta.default[10] + np.deg2rad(6.0))
    _, trim_lo = dr._stand_hold_targets(meta, trim_deg=-45.0)
    assert trim_lo == -6.0


def test_ankle_trim_reads_env_knob(monkeypatch):
    meta = _meta_stub()
    monkeypatch.setattr(dr, "ANKLE_TRIM_DEG", 2.5)   # what the env parse sets
    tgt, trim = dr._stand_hold_targets(meta)
    assert trim == 2.5
    assert tgt[4] == pytest.approx(meta.default[4] + np.deg2rad(2.5))
    # ankle indices really are the ankle_pitch joints in the 29-joint order
    assert [cs.G1_JOINT_ORDER[i] for i in dr.ANKLE_PITCH_IDX] == \
        ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]
