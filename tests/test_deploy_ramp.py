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
    """Golden test: the activation-ramp LOGIC against the promoted hardware deployable.

    History: this used to byte-assert add_activation_ramp(thriller_show) == the WHOLE
    thriller_deploy.csv. After the 2026-07-06 SHARP promotion the deployable's BODY is the
    per-joint sharp reference (thriller_deploy_v2_sharp), no longer the thriller_show-derived
    blanket-clamped body, so whole-file equality is stale. The SHARP promotion copied the
    2.5 s activation ramp VERBATIM, so the ramp PREFIX is still EXACTLY reproducible from
    thriller_show — that is the reproducible artifact this golden now guards, together with
    the documented ramp invariants (frame0 == standby default, last ramp row == show frame 0,
    root held across the whole ramp). A broken ramp still fails this.
    """
    show = PROJECT / "data/motions/thriller/thriller_show.csv"
    dep = PROJECT / "data/policies/thriller/thriller_deploy.csv"
    if not (show.exists() and dep.exists()):
        pytest.skip("canonical Thriller motions not present in this checkout")
    show_arr = np.loadtxt(show, delimiter=",")
    dj = default_joint_pos()
    got = add_activation_ramp(show_arr, dj)
    ref = np.loadtxt(dep, delimiter=",")
    assert got.shape == ref.shape                      # ramp + show length preserved
    # Byte-equality guard, pointed at the artifact that IS reproducible now: the ramp prefix
    # (root cols included) that the sharp promotion copied verbatim from thriller_show.
    np.testing.assert_allclose(got[:RAMP_FRAMES], ref[:RAMP_FRAMES], atol=1e-12)
    # ...and the documented ramp invariants, asserted directly on the canonical prefix:
    np.testing.assert_allclose(ref[0, 7:], dj, atol=1e-12)                # frame0 == standby
    np.testing.assert_allclose(ref[RAMP_FRAMES - 1, 7:], show_arr[0, 7:],
                               atol=1e-12)                                # last ramp row == show[0]
    np.testing.assert_allclose(ref[:RAMP_FRAMES, :7],
                               np.tile(show_arr[0, :7], (RAMP_FRAMES, 1)),
                               atol=1e-12)                                # root held across ramp
    # The BODY is the SHARP reference, deliberately NOT a raw thriller_show append — this is
    # exactly why whole-file equality was retired; guards against a silent revert to the
    # blanket-clamped, show-derived deployable.
    assert not np.allclose(ref[RAMP_FRAMES:], show_arr, atol=1e-6)


def test_rejects_wrong_column_count():
    with pytest.raises(ValueError, match="36-col"):
        add_activation_ramp(np.zeros((10, 30)), default_joint_pos())
