"""Activation-ramp deployable-motion generator (pipeline/deploy_ramp.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.deploy_ramp import (FPS, RAMP_FRAMES, RAMP_S, add_activation_ramp,
                                  default_joint_pos, make_deploy_csv)

from .conftest import make_motion

PROJECT = Path(__file__).resolve().parent.parent


def test_default_joint_pos_from_canonical_interface():
    dj = default_joint_pos()
    assert dj.shape == (29,)
    assert dj[3] == pytest.approx(0.669)     # left knee (bent standby)


def test_ramp_shape_and_endpoints():
    show = make_motion(frames=20, joint_overrides={3: 0.1, 18: -0.5})
    dj = default_joint_pos()
    dep = add_activation_ramp(show, dj)
    assert dep.shape == (20 + RAMP_FRAMES, 36)
    assert RAMP_FRAMES == round(RAMP_S * FPS) == 75
    # frame 0 = exact standby joints (zero activation lurch)
    np.testing.assert_allclose(dep[0, 7:], dj, atol=1e-12)
    # last ramp row reaches the show's first pose exactly
    np.testing.assert_allclose(dep[RAMP_FRAMES - 1, 7:], show[0, 7:], atol=1e-12)
    # root (xyz + quat) held at show frame-0 throughout the ramp
    np.testing.assert_allclose(dep[:RAMP_FRAMES, :7],
                               np.tile(show[0, :7], (RAMP_FRAMES, 1)), atol=1e-12)
    # the show itself is appended untouched
    np.testing.assert_allclose(dep[RAMP_FRAMES:], show, atol=1e-12)


def test_ramp_is_monotonic_cosine_ease():
    show = make_motion(frames=5, joint_overrides={3: 0.0})  # knee 0 vs standby 0.669
    dep = add_activation_ramp(show, default_joint_pos())
    knee = dep[:RAMP_FRAMES, 7 + 3]
    assert np.all(np.diff(knee) <= 1e-12)          # eases down, no overshoot
    assert abs(np.diff(knee)[0]) < abs(np.diff(knee)[RAMP_FRAMES // 2])  # cosine edges


def test_make_deploy_csv_roundtrip(tmp_path):
    show_csv = tmp_path / "d_show.csv"
    np.savetxt(show_csv, make_motion(frames=12), delimiter=",")
    out = tmp_path / "d_deploy.csv"
    info = make_deploy_csv(show_csv, out)
    assert out.exists()
    assert info["out_frames"] == 12 + RAMP_FRAMES
    assert info["frame0_max_delta_rad"] == 0.0


def test_reproduces_canonical_thriller_deploy_exactly():
    """Golden test against the hardware-validated deployable (2026-07-06 promotion)."""
    show = PROJECT / "data/motions/thriller/thriller_show.csv"
    dep = PROJECT / "data/policies/thriller/thriller_deploy.csv"
    if not (show.exists() and dep.exists()):
        pytest.skip("canonical Thriller motions not present in this checkout")
    got = add_activation_ramp(np.loadtxt(show, delimiter=","), default_joint_pos())
    ref = np.loadtxt(dep, delimiter=",")
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_rejects_wrong_column_count():
    with pytest.raises(ValueError, match="36-col"):
        add_activation_ramp(np.zeros((10, 30)), default_joint_pos())
