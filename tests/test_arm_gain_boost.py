"""Tests for the ARM_GROUND_KP_SCALE deploy knob (dance-quality program 2026-07-06).

The promoted s2r-b policy dances Thriller on hardware but the arms track ~2x
worse than sim (arm RMS 13.2 deg, wrist lag 100-160 ms). System-ID
(data/reports/system_id_20260706.json) showed the arm plant lags its commanded
target 81-141 ms with 0.1-0.5 Nm Coulomb friction at the soft trained gains.
The knob scales kp AND kd of the 14 arm joints (BY NAME) during the policy
phase of ground-run-legodom ONLY — mirroring the GROUND_LEG_KP_SCALE pattern —
and must be recorded in telemetry run_meta so capture_stage0's delivery
analysis regresses against the gains actually commanded (the audit's
fabricated-deficit failure mode).
"""
import json

import numpy as np
import pytest

from deploy import capture_stage0 as cs

dr = pytest.importorskip("pipeline.deploy_runtime")

ARM_IDX = list(range(15, 29))


def _fixt():
    return dr.Meta(dr.DEFAULT_META)


# ---- index resolution is by NAME, and matches the project-wide order ----------

def test_arm_indices_by_name_are_15_to_28():
    meta = _fixt()
    idx = dr._arm_joint_indices(meta.joint_order)
    assert idx == ARM_IDX
    for i in idx:
        n = cs.G1_JOINT_ORDER[i]
        assert ("shoulder" in n) or ("elbow" in n) or ("wrist" in n)
    # and no leg/waist joint sneaks in
    assert all("hip" not in cs.G1_JOINT_ORDER[i] and "ankle" not in cs.G1_JOINT_ORDER[i]
               and "knee" not in cs.G1_JOINT_ORDER[i] and "waist" not in cs.G1_JOINT_ORDER[i]
               for i in idx)


# ---- default: knob off, gains untouched ---------------------------------------

def test_default_scale_is_identity():
    import os
    if "ARM_GROUND_KP_SCALE" not in os.environ:
        assert dr.ARM_GROUND_KP_SCALE == 1.0
    meta = _fixt()
    kp, kd = meta.kp.astype(float).copy(), meta.kd.astype(float).copy()
    kp2, kd2, idx = dr._arm_boost_gains(meta, kp, kd, scale=1.0)
    assert idx == []
    assert np.array_equal(kp2, kp) and np.array_equal(kd2, kd)


# ---- boost applies to the 14 arm joints only, kp AND kd by the same factor ----

def test_boost_scales_arms_only_kp_and_kd():
    meta = _fixt()
    kp0, kd0 = meta.kp.astype(float).copy(), meta.kd.astype(float).copy()
    kp, kd, idx = dr._arm_boost_gains(meta, kp0, kd0, scale=2.5)
    assert idx == ARM_IDX
    for i in range(29):
        f = 2.5 if i in ARM_IDX else 1.0
        assert kp[i] == pytest.approx(kp0[i] * f)
        assert kd[i] == pytest.approx(kd0[i] * f)
    # inputs not mutated (copies returned)
    assert np.array_equal(kp0, meta.kp.astype(float))
    # resulting arm kp stays inside the teleop-proven envelope (80 / 40)
    assert kp[15] == pytest.approx(14.251 * 2.5, rel=1e-3)   # shoulder ~35.6 < 80
    assert kp[27] == pytest.approx(16.778 * 2.5, rel=1e-3)   # wrist_pitch ~41.9 ~ 40


def test_boost_composes_with_leg_boost_without_overlap():
    meta = _fixt()
    kp, kd = meta.kp.astype(float).copy(), meta.kd.astype(float).copy()
    kp[dr.LEG_JOINT_IDX] *= 1.5   # what mode_ground_run_legodom does first
    kd[dr.LEG_JOINT_IDX] *= 1.5
    kp2, kd2, _ = dr._arm_boost_gains(meta, kp, kd, scale=2.0)
    for i in dr.LEG_JOINT_IDX:   # leg boost preserved, not double-touched
        assert kp2[i] == pytest.approx(meta.kp[i] * 1.5)
    for i in ARM_IDX:
        assert kp2[i] == pytest.approx(meta.kp[i] * 2.0)


# ---- refusal outside the proven envelope --------------------------------------

def test_refuses_out_of_range_scales():
    meta = _fixt()
    kp, kd = meta.kp.astype(float), meta.kd.astype(float)
    with pytest.raises(SystemExit):
        dr._arm_boost_gains(meta, kp, kd, scale=0.8)     # softer than trained: no use case
    with pytest.raises(SystemExit):
        dr._arm_boost_gains(meta, kp, kd, scale=3.5)     # beyond the proven envelope
    # boundary is allowed
    kp3, _, idx = dr._arm_boost_gains(meta, kp, kd, scale=dr.ARM_GROUND_KP_SCALE_MAX)
    assert idx == ARM_IDX and kp3[15] == pytest.approx(meta.kp[15] * 3.0)


def test_env_knob_path(monkeypatch):
    meta = _fixt()
    monkeypatch.setattr(dr, "ARM_GROUND_KP_SCALE", 2.0)   # what the env parse sets
    kp, kd, idx = dr._arm_boost_gains(meta, meta.kp.astype(float), meta.kd.astype(float))
    assert idx == ARM_IDX and kp[18] == pytest.approx(meta.kp[18] * 2.0)


# ---- telemetry provenance ------------------------------------------------------

def test_run_meta_records_arm_scale(tmp_path, monkeypatch):
    meta = _fixt()
    monkeypatch.setattr(dr, "ARM_GROUND_KP_SCALE", 2.5)
    telem = dr.Telemetry("ground-run-legodom", meta)
    telem.path = tmp_path / "t.npz"

    class _M:
        q = dq = tau_est = 0.0
        temperature = [40, 40]

    msg = type("Msg", (), {"motor_state": [_M() for _ in range(29)]})()
    telem.add(0, meta.default, np.zeros(29), msg, np.array([1.0, 0, 0, 0]),
              np.zeros(3), np.zeros(29), meta.default)
    telem.save(quiet=True)
    rm = json.loads(str(np.load(tmp_path / "t.npz")["run_meta_json"]))
    assert rm["arm_ground_kp_scale"] == 2.5
    assert rm["mode"] == "ground-run-legodom"


# ---- capture_stage0 delivery analysis uses the boosted gains -------------------

def test_effective_gains_corrects_arm_boost():
    d = {"kp": np.full(29, 10.0), "kd": np.full(29, 1.0)}
    kp, kd, notes = cs._effective_gains(
        d, {"mode": "ground-run-legodom", "ground_leg_kp_scale": 1.0,
            "arm_ground_kp_scale": 2.5})
    for i in range(29):
        f = 2.5 if i in ARM_IDX else 1.0
        assert kp[i] == pytest.approx(10.0 * f)
        assert kd[i] == pytest.approx(1.0 * f)
    assert any("ARM" in n for n in notes)
    # combined leg + arm boosts
    kp, kd, notes = cs._effective_gains(
        d, {"mode": "ground-run-legodom", "ground_leg_kp_scale": 1.5,
            "arm_ground_kp_scale": 2.0})
    assert kp[0] == pytest.approx(15.0) and kp[15] == pytest.approx(20.0)
    assert kp[1] == pytest.approx(10.0)   # hip_roll: neither boost touches it
    # absent key (old telemetry) -> unchanged
    kp, kd, notes = cs._effective_gains(
        d, {"mode": "ground-run-legodom", "ground_leg_kp_scale": 1.0})
    assert np.allclose(kp, 10.0) and notes == []
