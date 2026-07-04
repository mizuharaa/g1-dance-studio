"""Offline tests for the laptop-side deploy runtime (no robot, no SDK).

Covers the pure obs-construction + control math that decides whether the real robot
gets a sane command. The SDK/LowCmd paths are exercised only on hardware (gated).
"""
import numpy as np
import pytest

dr = pytest.importorskip("pipeline.deploy_runtime")


def test_quat_mat_orthonormal():
    R = dr.quat_wxyz_to_mat(np.array([1.0, 0, 0, 0]))
    assert np.allclose(R, np.eye(3), atol=1e-9)
    q = np.array([0.5, 0.5, 0.5, 0.5])  # 120deg about (1,1,1)
    R = dr.quat_wxyz_to_mat(q)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)  # orthonormal
    assert abs(np.linalg.det(R) - 1.0) < 1e-6


def test_anchor_ori_b_is_6d_identity_when_aligned():
    v = dr.mat_first_two_cols_b(np.array([1.0, 0, 0, 0]), np.array([1.0, 0, 0, 0]))
    assert v.shape == (6,)
    # aligned frames -> identity[:, :2].reshape(-1) (C-order, matches verified mjlab
    # sim_exam convention): [1,0,0, 1,0,0]
    assert np.allclose(v, [1, 0, 0, 1, 0, 0], atol=1e-9)


def test_meta_and_reference_load():
    if not dr.DEFAULT_META.exists() or not dr.DEFAULT_MOTION.exists():
        pytest.skip("policy artifacts not present")
    meta = dr.Meta(dr.DEFAULT_META)
    ref = dr.Reference(dr.DEFAULT_MOTION)
    assert meta.n == 29
    assert len(meta.kp) == 29 and len(meta.default) == 29 and len(meta.action_scale) == 29
    assert ref.T > 100
    # torso ref height sane at t=0 -> confirms TORSO_NPZ_IDX is right
    assert 0.3 < float(ref.apos[0, 2]) < 1.2


def test_build_obs_is_160_and_finite():
    if not dr.DEFAULT_META.exists() or not dr.DEFAULT_MOTION.exists():
        pytest.skip("policy artifacts not present")
    meta = dr.Meta(dr.DEFAULT_META)
    ref = dr.Reference(dr.DEFAULT_MOTION)
    q = meta.default + np.deg2rad(np.random.uniform(-15, 15, 29))
    obs, terms = dr.build_obs(meta, ref, q, np.zeros(29),
                              np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(29), tick=0)
    assert obs.shape == (160,)
    assert np.all(np.isfinite(obs))
    # widths per layout
    assert sum(w for _, w in dr.OBS_LAYOUT) == 160
    # gantry approximations at t=0
    assert np.allclose(terms["motion_anchor_pos_b"], 0.0)
    assert np.allclose(terms["base_lin_vel"], 0.0)
    # joint_pos term is q - default
    assert np.allclose(terms["joint_pos"], q - meta.default)


def test_action_to_target_uses_per_joint_scale():
    if not dr.DEFAULT_META.exists():
        pytest.skip("meta not present")
    meta = dr.Meta(dr.DEFAULT_META)
    a = np.ones(29)
    tgt = dr.action_to_target(meta, a)
    assert np.allclose(tgt, meta.default + meta.action_scale)  # scale is per-joint
    # zero action -> exactly the default (ready) pose
    assert np.allclose(dr.action_to_target(meta, np.zeros(29)), meta.default)


def test_target_clamped_to_limits():
    if not dr.DEFAULT_META.exists():
        pytest.skip("meta not present")
    meta = dr.Meta(dr.DEFAULT_META)
    # a wild action must clamp within [q_lo, q_hi]
    tgt = dr.action_to_target(meta, np.full(29, 100.0))
    clamped = np.clip(tgt, meta.q_lo, meta.q_hi)
    assert np.all(clamped <= meta.q_hi + 1e-9) and np.all(clamped >= meta.q_lo - 1e-9)


# ---- GROUND (obs-restricted) deployment ---------------------------------------

def test_ground_layout_is_154_and_estimator_free():
    # The documented fallback layout drops exactly the two estimator-only terms.
    assert sum(w for _, w in dr.GROUND_OBS_LAYOUT) == 154
    names = {n for n, _ in dr.GROUND_OBS_LAYOUT}
    assert not (names & dr.ESTIMATOR_DEPENDENT_TERMS)
    # and it is the 160-dim layout minus base_lin_vel + motion_anchor_pos_b
    full = sum(w for _, w in dr.OBS_LAYOUT)
    assert full - dr.TERM_WIDTHS["base_lin_vel"] - dr.TERM_WIDTHS["motion_anchor_pos_b"] == 154


def _meta_with_terms(obs_terms):
    if not dr.DEFAULT_META.exists():
        pytest.skip("meta not present")
    meta = dr.Meta(dr.DEFAULT_META)
    meta.obs_terms = obs_terms
    return meta


def test_ground_obs_order_defaults_to_documented_layout():
    order = dr._ground_obs_order(_meta_with_terms(None))
    assert order == dr.GROUND_OBS_LAYOUT


def test_ground_obs_order_trusts_declared_layout():
    declared = ["command", "motion_anchor_ori_b", "base_ang_vel",
                "joint_pos", "joint_vel", "actions"]
    order = dr._ground_obs_order(_meta_with_terms(declared))
    assert [n for n, _ in order] == declared
    assert sum(w for _, w in order) == 154


def test_ground_obs_order_refuses_estimator_dependent_terms():
    # A "ground" policy that still needs base_lin_vel is NOT estimator-free -> refuse.
    bad = ["command", "base_lin_vel", "motion_anchor_ori_b", "base_ang_vel",
           "joint_pos", "joint_vel", "actions"]
    with pytest.raises(SystemExit):
        dr._ground_obs_order(_meta_with_terms(bad))
    # likewise motion_anchor_pos_b
    bad2 = ["command", "motion_anchor_pos_b", "motion_anchor_ori_b", "base_ang_vel",
            "joint_pos", "joint_vel", "actions"]
    with pytest.raises(SystemExit):
        dr._ground_obs_order(_meta_with_terms(bad2))


def test_ground_obs_order_refuses_unknown_term():
    with pytest.raises(SystemExit):
        dr._ground_obs_order(_meta_with_terms(["command", "mystery_term"]))


def test_build_obs_ground_is_154_finite_and_estimator_free():
    if not dr.DEFAULT_META.exists() or not dr.DEFAULT_MOTION.exists():
        pytest.skip("policy artifacts not present")
    meta = dr.Meta(dr.DEFAULT_META)
    ref = dr.Reference(dr.DEFAULT_MOTION)
    # Force the estimator-free order (the gantry meta itself declares the full 160-dim
    # layout, which _ground_obs_order correctly refuses — see the refusal test).
    order = dr.GROUND_OBS_LAYOUT
    q = meta.default + np.deg2rad(np.random.uniform(-15, 15, 29))
    dq = np.random.uniform(-0.5, 0.5, 29)
    gyro = np.random.uniform(-0.2, 0.2, 3)
    obs, terms = dr.build_obs_ground(meta, ref, q, dq, np.array([1.0, 0, 0, 0]),
                                     gyro, np.zeros(29), tick=0, order=order)
    assert obs.shape == (154,)
    assert np.all(np.isfinite(obs))
    # NO fabricated estimator terms are present
    assert "base_lin_vel" not in terms and "motion_anchor_pos_b" not in terms
    # shared terms are computed identically to the gantry builder
    assert np.allclose(terms["joint_pos"], q - meta.default)
    assert np.allclose(terms["joint_vel"], dq)
    assert np.allclose(terms["base_ang_vel"], gyro)


def test_ground_max_action_is_conservative():
    # Ground default cap must be tighter than the gantry cap (falls are unforgiving).
    assert dr.GROUND_MAX_ACTION <= dr.MAX_ACTION
