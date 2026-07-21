"""Regression guards for the 2026-07-21 ML-audit fixes (experiments/ml_audit_20260721).
Covers the two subtle orchestrator-owned fixes: A (velocity-aware SG exemption) and
E (shared deploy/preview history stacking). The reward-stack fixes (F/G/H/I) need mjlab
and are gated by the box selfcheck; the eval/preview fixes have their own suites."""
from __future__ import annotations

import numpy as np


# --- FIX A: velocity-aware SG exemption keeps sharp choreography -----------------
def test_fast_move_survives_cleaning_even_when_unflagged():
    """A genuinely fast choreographic sweep (not a 1-2 frame glitch) must keep its
    peak velocity through clean_motion — the SG pass used to blunt it ~29% because a
    fast joint's own MAD hides the move from the spike detector (audit finding A)."""
    from tools.motion_quality import clean_motion
    fps = 30.0
    n = 240
    t = np.arange(n) / fps
    m = np.zeros((n, 36))
    m[:, 2] = 0.79
    m[:, 6] = 1.0
    # a fast smooth sweep on one joint: 0.8 rad over ~0.13 s -> ~6 rad/s peak, feasible
    j = 20
    c, w = 4.0, 0.13
    s = np.clip((t - (c - w / 2)) / w, 0, 1)
    m[:, 7 + j] = 0.8 * 0.5 * (1 - np.cos(np.pi * s)) * (t >= c - w / 2)
    raw_pv = np.abs(np.diff(m[:, 7 + j])).max() * fps
    cleaned, info = clean_motion(m, fps)
    clean_pv = np.abs(np.diff(cleaned[:, 7 + j])).max() * fps
    assert clean_pv > 0.9 * raw_pv, (
        f"SG blunted an unflagged fast move: {raw_pv:.2f} -> {clean_pv:.2f} rad/s")


def test_slow_jitter_is_still_smoothed():
    """The velocity gate must NOT disable smoothing of low-speed jitter."""
    from tools.motion_quality import clean_motion, analyze
    rng = np.random.default_rng(3)
    fps = 30.0
    n = 300
    t = np.arange(n) / fps
    m = np.zeros((n, 36))
    m[:, 2] = 0.79
    m[:, 6] = 1.0
    base = 0.3 * np.sin(2 * np.pi * 0.5 * t)          # slow, < 2 rad/s
    for j in range(29):
        m[:, 7 + j] = base + rng.normal(0, 0.01, n)    # small high-freq jitter
    before = analyze(m, fps)["jerk_peak_rad_s3"]
    cleaned, _ = clean_motion(m, fps)
    after = analyze(cleaned, fps)["jerk_peak_rad_s3"]
    assert after < before, f"jitter not smoothed: jerk {before} -> {after}"


# --- FIX E: shared history stacker, correct layout ------------------------------
def test_history_stacker_layout():
    from pipeline.deploy_runtime import HistoryStacker
    order = (("a", 2), ("b", 3))
    hs = HistoryStacker(order, 3)
    hs.push({"a": np.array([1, 1]), "b": np.array([10, 10, 10])})   # warmup backfill
    o = hs.push({"a": np.array([2, 2]), "b": np.array([20, 20, 20])})
    # term-major, oldest->newest: a-block then b-block, frames [t1,t1,t2]
    assert o[:6].tolist() == [1, 1, 1, 1, 2, 2]
    assert o[6:].tolist() == [10, 10, 10, 10, 10, 10, 20, 20, 20]
    assert o.shape[0] == (2 + 3) * 3


def test_history_stacker_n_hist_inference():
    from pipeline.deploy_runtime import HistoryStacker
    assert HistoryStacker.n_hist_for(770, 154) == 5   # v8/v10/v11 contract
    assert HistoryStacker.n_hist_for(154, 154) == 1   # single-frame policy
    assert HistoryStacker.n_hist_for(160, 154) == 1   # non-multiple -> no fabrication
    assert HistoryStacker.n_hist_for("dynamic", 154) == 1


def test_history_stacker_single_frame_passthrough():
    from pipeline.deploy_runtime import HistoryStacker
    hs = HistoryStacker((("a", 2), ("b", 3)), 1)
    o = hs.push({"a": np.array([1, 2]), "b": np.array([3, 4, 5])})
    assert o.tolist() == [1, 2, 3, 4, 5]
