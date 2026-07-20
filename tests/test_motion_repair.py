"""beat_preserving_warp (v10): time conservation + smoothness. Uses a small
synthetic motion so the contact-aware dynamic pass stays fast. Needs MuJoCo."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from pipeline import g1_limits as L  # noqa: E402
from tools.motion_repair import beat_preserving_warp  # noqa: E402

FPS = 30.0


def _motion_with_hard_burst(n=360):
    """12 s standing motion with one violent ~0.4 s squat-drop burst at t=6 s
    (binding ratio ~1.7, verified) surrounded by long quiet holds (donors)."""
    t = np.arange(n) / FPS
    m = np.zeros((n, 36))
    m[:, 2] = 0.76
    m[:, 6] = 1.0
    m[:, 7:] = L.DEFAULT_JOINT_POS
    burst = np.exp(-0.5 * ((t - 6.0) / 0.10) ** 2)
    for j, amp in ((0, 1.4), (3, 1.7), (6, -1.4), (9, 1.7)):   # hips + knees
        m[:, 7 + j] += amp * burst
    m[:, 2] -= 0.30 * burst                                     # fast deep drop
    return m


def test_time_conserved_and_demand_reduced():
    m = _motion_with_hard_burst()
    src_dur = (len(m) - 1) / FPS
    out, rep = beat_preserving_warp(m, fps=FPS, margin=0.9, verbose=False)
    dur = (len(out) - 1) / FPS
    # total duration conserved to within a frame or two
    assert abs(dur - src_dur) < 0.1, f"duration {src_dur} -> {dur}"
    assert 0.97 < rep["duration_ratio"] < 1.03
    # the warp actually stretched something and repaid it
    assert rep["max_local_gain"] > 1.05
    assert rep["beat_drift_max_s"] < 1.0
    # monotone time map (no frame duplication / time reversal)
    src_t = np.asarray(rep["time_map"]["source_t"])
    assert (np.diff(src_t) > -1e-9).all()


def test_quiet_motion_untouched():
    n = 240
    m = np.zeros((n, 36))
    m[:, 2] = 0.76
    m[:, 6] = 1.0
    m[:, 7:] = L.DEFAULT_JOINT_POS
    out, rep = beat_preserving_warp(m, fps=FPS, verbose=False)
    assert rep["max_local_gain"] == 1.0
    assert len(out) == n
