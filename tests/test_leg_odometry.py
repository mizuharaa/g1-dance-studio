"""Offline sanity tests for kinematic (leg) odometry — needs mujoco + the G1 model, no robot."""
import numpy as np
import pytest

dr = pytest.importorskip("pipeline.deploy_runtime")
lo = pytest.importorskip("pipeline.leg_odometry")
pytest.importorskip("mujoco")


def _odo():
    meta = dr.Meta(dr.DEFAULT_META)
    return lo.LegOdometry(list(meta.joint_order)), meta


def test_zero_joint_velocity_gives_zero_base_velocity():
    """No joint motion + no rotation -> the feet aren't moving relative to the base, so the
    kinematic base velocity is exactly zero."""
    odo, meta = _odo()
    v, h, info = odo.estimate(meta.default, np.zeros(29), np.eye(3), np.zeros(3))
    assert np.allclose(v, 0.0, atol=1e-9)
    # standing default: base height above the feet should be a plausible ~0.5-0.8 m
    assert 0.3 < h < 0.95


def test_contact_weights_normalize_and_favor_lower_foot():
    odo, meta = _odo()
    _, _, info = odo.estimate(meta.default, np.zeros(29), np.eye(3), np.zeros(3))
    w = info["weights"]
    assert np.isclose(w.sum(), 1.0) and np.all(w >= 0)


def test_velocity_is_clipped_to_physical_bound():
    """A huge joint velocity (degenerate) must not produce an unbounded base velocity."""
    odo, meta = _odo()
    v, _, _ = odo.estimate(meta.default, np.full(29, 50.0), np.eye(3), np.zeros(3))
    assert np.all(np.abs(v) <= lo.MAX_BASE_SPEED + 1e-9)


def test_matches_reference_base_velocity_within_tolerance():
    """End-to-end: leg-odom base_lin_vel tracks the reference's true base velocity inside the
    policy's ±0.5 m/s trained band on the large majority of frames."""
    odo, meta = _odo()
    d = np.load(dr.DEFAULT_MOTION)
    jp, jv, bq, ba, bl = (d["joint_pos"], d["joint_vel"], d["body_quat_w"],
                          d["body_ang_vel_w"], d["body_lin_vel_w"])
    T = jp.shape[0]
    within = 0
    for t in range(0, T, 5):  # subsample for speed
        R = dr.quat_wxyz_to_mat(bq[t, 0])
        v_est, _, _ = odo.estimate(jp[t], jv[t], R, R.T @ ba[t, 0])
        v_true = R.T @ bl[t, 0]
        within += int(np.all(np.abs(v_est - v_true) <= 0.5))
    n = len(range(0, T, 5))
    assert within / n > 0.9, f"only {within}/{n} frames within ±0.5 m/s"
