"""Ground-truth validation of the mjlab tracking observation layout.

The fixture mjlab_obs_sample.npz was captured from the REAL mjlab env
(Mjlab-Tracking-Flat-Unitree-G1) with the Thriller motion, corruption off.
It pins the exact 160-dim actor-obs term order so any MjlabOnnxPolicy obs
reconstruction can be checked against reality (not just width-sum arithmetic).
Verified layout: command(58=ref jp29+jv29) | anchor_pos_b(3) | anchor_ori_b(6) |
base_lin_vel(3) | base_ang_vel(3) | joint_pos(29=jp-default) | joint_vel(29) | actions(29).
"""
import numpy as np
from pathlib import Path

FIX = Path(__file__).parent / "fixtures" / "mjlab_obs_sample.npz"


def _g(d, k, i):
    return d[f"{k}_{i}"]


def _qinv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / (q @ q)


def _qapply(q, v):
    w, x, y, z = q
    u = np.array([x, y, z])
    return 2 * (u @ v) * u + (w * w - u @ u) * v + 2 * w * np.cross(u, v)


def _mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_obs_layout_matches_real_mjlab():
    d = np.load(FIX, allow_pickle=True)
    for i in range(1, 8):
        obs = _g(d, "actor_obs", i)
        assert obs.shape == (160,)
        # command [0:58] = ref joint_pos + joint_vel
        assert np.allclose(obs[0:29], _g(d, "ref_joint_pos", i), atol=1e-4)
        assert np.allclose(obs[29:58], _g(d, "ref_joint_vel", i), atol=1e-4)
        # anchor pos [58:61] and ori 6D [61:67] via subtract_frame_transforms
        t01, q01 = _g(d, "robot_anchor_pos_w", i), _g(d, "robot_anchor_quat_w", i)
        t02, q02 = _g(d, "anchor_pos_w", i), _g(d, "anchor_quat_w", i)
        q10 = _qinv(q01)
        pos_b = _qapply(q10, t02 - t01)
        assert np.allclose(pos_b, obs[58:61], atol=1e-4)
        rel = _mat(q01).T @ _mat(q02)
        assert np.allclose(rel[:, :2].reshape(-1), obs[61:67], atol=1e-4)
        # joint_pos [73:102] = q - default (encoder-bias DR is 0 in a clean exam)
        jpr = _g(d, "joint_pos", i) - _g(d, "default_joint_pos", i)
        assert np.max(np.abs(obs[73:102] - jpr)) < 0.02  # <=bias tolerance
        # joint_vel [102:131] exact
        assert np.allclose(obs[102:131], _g(d, "joint_vel", i), atol=1e-4)
