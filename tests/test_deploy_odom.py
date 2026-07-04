"""Offline tests for the ODOMETRY-FED ground obs path (build_obs_odom) — no robot, no SDK.

These verify the honest math that lets the PROVEN full-obs gantry policy deploy on the
ground using the onboard estimate (rt/odommodestate) instead of the gantry fakes:
  * motion_anchor_pos_b = R_robᵀ · (ref_disp − robot_disp)   [re-anchored torso pos error]
  * base_lin_vel        = R_robᵀ · v_world                    [torso velocity in body frame]
The whole point is to NOT feed a fabricated estimator quantity to a standing robot, so the
correctness of these two terms is exactly what keeps it from falling.
"""
import numpy as np
import pytest

dr = pytest.importorskip("pipeline.deploy_runtime")

Q0 = np.array([1.0, 0, 0, 0])          # identity quat (wxyz)
# a non-trivial orientation: 90° yaw
QYAW = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])


def _fixt():
    meta = dr.Meta(dr.DEFAULT_META)
    ref = dr.Reference(dr.DEFAULT_MOTION)
    return meta, ref


def test_build_obs_odom_is_160_and_finite():
    meta, ref = _fixt()
    q = meta.default + np.deg2rad(np.random.uniform(-10, 10, 29))
    obs, terms = dr.build_obs_odom(meta, ref, q, np.zeros(29), Q0, np.zeros(3),
                                   np.zeros(29), tick=5,
                                   robot_disp=np.array([0.1, -0.05, 0.0]),
                                   v_world=np.array([0.2, 0.0, 0.0]))
    assert obs.shape[0] == 160 and np.all(np.isfinite(obs))
    assert sum(w for _, w in dr.OBS_LAYOUT) == 160


def test_perfect_tracking_zeroes_anchor_pos_b():
    """If the robot's displacement equals the reference's displacement (perfect
    tracking), the anchor position error term must be ~0 — the training-time value."""
    meta, ref = _fixt()
    tick = 40
    ref_disp = ref.at(tick)[2] - ref.apos[0]        # reference torso displacement
    _, terms = dr.build_obs_odom(meta, ref, meta.default, np.zeros(29), Q0, np.zeros(3),
                                 np.zeros(29), tick=tick,
                                 robot_disp=ref_disp, v_world=np.zeros(3))
    assert np.allclose(terms["motion_anchor_pos_b"], 0.0, atol=1e-9)


def test_base_lin_vel_is_world_rotated_into_body():
    """base_lin_vel must be the world velocity expressed in the body frame (R_robᵀ·v)."""
    meta, ref = _fixt()
    v_world = np.array([0.3, 0.1, -0.05])
    _, terms = dr.build_obs_odom(meta, ref, meta.default, np.zeros(29), QYAW, np.zeros(3),
                                 np.zeros(29), tick=0,
                                 robot_disp=np.zeros(3), v_world=v_world)
    R = dr.quat_wxyz_to_mat(QYAW)
    assert np.allclose(terms["base_lin_vel"], R.T @ v_world, atol=1e-9)
    # 90° yaw: a +x world velocity should read as -y (or +y) in body, never still +x.
    assert not np.allclose(terms["base_lin_vel"], v_world)


def test_matches_gantry_fake_when_static():
    """With zero robot displacement and zero velocity (the gantry case), the odom builder
    must reduce EXACTLY to the gantry build_obs — same anchor term, base_lin_vel = 0."""
    meta, ref = _fixt()
    q = meta.default + np.deg2rad(np.random.uniform(-10, 10, 29))
    tick = 25
    obs_g, terms_g = dr.build_obs(meta, ref, q, np.zeros(29), Q0, np.zeros(3),
                                  np.zeros(29), tick)
    obs_o, terms_o = dr.build_obs_odom(meta, ref, q, np.zeros(29), Q0, np.zeros(3),
                                       np.zeros(29), tick,
                                       robot_disp=np.zeros(3), v_world=np.zeros(3))
    assert np.allclose(terms_o["motion_anchor_pos_b"], terms_g["motion_anchor_pos_b"])
    assert np.allclose(terms_o["base_lin_vel"], 0.0)
    assert np.allclose(obs_o, obs_g)


def test_anchor_error_points_from_robot_to_reference():
    """If the robot lags the reference (smaller displacement), the body-frame error must
    point toward where the reference wants the torso — sign sanity, identity orientation."""
    meta, ref = _fixt()
    tick = 60
    ref_disp = ref.at(tick)[2] - ref.apos[0]
    robot_disp = 0.5 * ref_disp                      # robot only went half as far
    _, terms = dr.build_obs_odom(meta, ref, meta.default, np.zeros(29), Q0, np.zeros(3),
                                 np.zeros(29), tick=tick,
                                 robot_disp=robot_disp, v_world=np.zeros(3))
    # identity orientation -> body == world; error = ref_disp - robot_disp = +0.5*ref_disp
    assert np.allclose(terms["motion_anchor_pos_b"], ref_disp - robot_disp, atol=1e-9)


def test_odom_vel_source_default_is_diff():
    """Safety default: velocity from position differencing (frame-unambiguous), not the
    unvalidated EKF velocity field."""
    assert dr.ODOM_VEL_SOURCE in ("diff", "field")
    # default (no env override in test env) must be the safe 'diff'
    import os
    if "ODOM_VEL_SOURCE" not in os.environ:
        assert dr.ODOM_VEL_SOURCE == "diff"
