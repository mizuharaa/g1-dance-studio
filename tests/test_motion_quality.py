"""tools/motion_quality: injected spike is detected + removed; sharp clean
motion passes through within tolerance. No MuJoCo needed."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from tools.motion_quality import analyze, clean_motion, reject_outliers, smooth_quat

FPS = 30.0
N = 300


def _sharp_motion() -> np.ndarray:
    """Plausible fast dance: 2 Hz sines on the joints (vel peak ~10 rad/s),
    yaw-swinging root, identity-ish quat track."""
    t = np.arange(N) / FPS
    m = np.zeros((N, 36))
    m[:, 2] = 0.79
    m[:, 3:7] = Rotation.from_euler(
        "z", 0.5 * np.sin(2 * np.pi * 0.5 * t)).as_quat()
    for j in range(29):
        m[:, 7 + j] = 0.8 * np.sin(2 * np.pi * 2.0 * t + j)
    return m


def test_spike_detected_and_removed():
    m = _sharp_motion()
    m[150, 7 + 4] += 1.5  # single-frame GVHMR-style limb flip
    rep = analyze(m, FPS)
    assert rep["spike_frame_count"] >= 1
    assert any(abs(f - 150) <= 2 for f in rep["spike_frames"])
    assert any(w["dof_index"] == 4 for w in rep["worst_joints"])

    cleaned, info = clean_motion(m, FPS)
    assert info["outlier_frames_replaced"] >= 1
    # the flip is gone: cleaned value back near the un-spiked sine
    truth = _sharp_motion()[150, 7 + 4]
    assert abs(cleaned[150, 7 + 4] - truth) < 0.1
    assert info["jerk_peak_after"] < info["jerk_peak_before"] / 3
    assert analyze(cleaned, FPS)["spike_frame_count"] == 0


def test_clean_sharp_motion_untouched():
    m = _sharp_motion()
    cleaned, info = clean_motion(m, FPS)
    assert info["outlier_frames_replaced"] == 0
    # sharp choreography not blurred: SG(7,3) on a 2 Hz sine barely moves it
    assert np.abs(cleaned[:, 7:] - m[:, 7:]).max() < 0.03
    # quat track stays a valid, close rotation
    norms = np.linalg.norm(cleaned[:, 3:7], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    ang = (Rotation.from_quat(cleaned[:, 3:7])
           * Rotation.from_quat(m[:, 3:7]).inv()).magnitude()
    assert ang.max() < 0.02


def test_outlier_rejection_ignores_smooth_and_fast_content():
    # monotone ramp and a fast-but-smooth sine must NOT be flagged (regression
    # for the everything-is-an-outlier / MAD-goes-to-zero failure modes)
    t = np.arange(200) / FPS
    x = np.stack([np.linspace(0, 2.0, 200),
                  0.8 * np.sin(2 * np.pi * 2.0 * t)], axis=1)
    _, n = reject_outliers(x)
    assert n == 0


def test_smooth_quat_kills_single_frame_flip():
    t = np.arange(N) / FPS
    q = Rotation.from_euler("z", 0.4 * np.sin(2 * np.pi * t)).as_quat()
    bad = q.copy()
    bad[100] = Rotation.from_euler("z", 2.5).as_quat()  # outlier orientation
    sm = smooth_quat(bad)
    err_before = (Rotation.from_quat(bad[100])
                  * Rotation.from_quat(q[100]).inv()).magnitude()
    err_after = (Rotation.from_quat(sm[100])
                 * Rotation.from_quat(q[100]).inv()).magnitude()
    assert err_after < err_before / 2
