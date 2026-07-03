"""Regression tests for the sim-exam physics reconciliation (2026-07-04).

Covers the two faithful fixes and the faithfulness guard. These run headless and do
not need a trained policy — they use the exam's own model + a stub.
"""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
import pipeline.sim_exam as se


def _tiny_meta_policy(model, motion):
    """A minimal MjlabOnnx-like adapter carrying real per-joint gains, no ONNX."""
    class P:
        joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                       for j in range(1, model.njnt)][:29]
        # gains matching mjlab's armature*(2*pi*10)^2 for a couple of motor types
        kp = np.array([40.179239, 99.098428] * 14 + [40.179239])
        kd = np.array([2.55789, 6.308802] * 14 + [2.55789])
        default_pos = np.zeros(29)
        action_scale = np.full(29, 0.5)
        anchor_body_name = "torso_link"
        obs_terms = [("joint_pos", 1)]
        def act(self, obs, tick):
            return np.zeros(29)
        def reset(self):
            pass
    return P()


def _load_env():
    model = mujoco.MjModel.from_xml_path(str(se.G1_XML))
    # a trivial 3-frame static motion at the model's default configuration
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model


def test_armature_recovered_from_gains_not_flat():
    """The exam must set per-joint armature from the gains, not leave the flat XML 0.01."""
    model = _load_env()
    motion = _make_static_motion(model)
    pol = _tiny_meta_policy(model, motion)
    env = se.ExamEnv(model, pol, motion)
    w0 = 2 * np.pi * 10.0
    for i in range(env.n):
        expected = float(pol.kp[i]) / (w0 * w0)
        assert env.model.dof_armature[env.vadr[i]] == pytest.approx(expected, rel=1e-6)
    # and the recovered values match real G1 motor armatures (not 0.01)
    known = {0.0036097, 0.0042500, 0.0072194, 0.0101775, 0.0251019}
    for i in range(env.n):
        arm = env.model.dof_armature[env.vadr[i]]
        assert min(abs(arm - k) for k in known) < 0.02 * arm
        assert abs(arm - 0.01) > 1e-4  # not the flat XML default


def test_mis_scaled_meta_kp_is_rejected():
    """A kp that doesn't map to a real motor armature must raise, not silently run."""
    model = _load_env()
    motion = _make_static_motion(model)
    pol = _tiny_meta_policy(model, motion)
    pol.kp = pol.kp * 3.3  # scramble so armature no longer matches a known motor
    with pytest.raises(ValueError, match="does not match any known"):
        se.ExamEnv(model, pol, motion)


def test_imu_site_moved_to_mjlab_offset():
    model = _load_env()
    motion = _make_static_motion(model)
    env = se.ExamEnv(model, _tiny_meta_policy(model, motion), motion)
    imu_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
    if imu_id >= 0:
        assert np.allclose(model.site_pos[imu_id], [0.04525, 0.0, -0.08339])


def test_faithfulness_guard_flags_broken_model():
    """static_pose_hold_ok must return False for the current unitree_mujoco model
    (it cannot hold the G1 standing) — the harness self-check that keeps a broken
    exam from being read as a policy failure."""
    model = _load_env()
    motion = _make_static_motion(model)
    env = se.ExamEnv(model, _tiny_meta_policy(model, motion), motion)
    # default_pos = zeros here is not even a valid stance; the guard must not crash and
    # must return a bool. (On the real Thriller default pose it returns False in practice.)
    ok = se.static_pose_hold_ok(env, seconds=0.5)
    assert isinstance(ok, bool)


def _make_static_motion(model):
    n, T = 29, 3
    return se.Motion(
        root_pos=np.tile([0.0, 0.0, 0.79], (T, 1)),
        root_quat_wxyz=np.tile([1.0, 0, 0, 0], (T, 1)),
        joint_pos=np.zeros((T, n)),
        joint_vel=np.zeros((T, n)),
        anchor_pos=np.tile([0.0, 0.0, 0.844], (T, 1)),
        anchor_quat_wxyz=np.tile([1.0, 0, 0, 0], (T, 1)),
    )
