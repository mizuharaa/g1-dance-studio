"""Vet gate decision logic via the CLI (--json), on synthetic motions."""
import numpy as np
import pytest

from .conftest import HAVE_MODEL, make_motion, run_vet

pytestmark = [pytest.mark.model,
              pytest.mark.skipif(not HAVE_MODEL, reason="G1 model not present")]


def test_standing_motion_passes(motion_csv):
    rc, rep = run_vet(motion_csv(frames=15))
    assert rc == 0 and rep["pass"] is True
    assert rep["hard"]["root_excursion"]["pass"]
    assert rep["hard"]["joint_limits"]["pass"]
    assert rep["hard"]["pelvis_height"]["pass"]
    assert rep["frames"] == 15
    assert rep["seconds"] == pytest.approx(0.5)


def test_excursion_beyond_limit_fails(motion_csv):
    rc, rep = run_vet(motion_csv(drift_xy=(2.0, 0.0)))
    assert rc == 1 and rep["pass"] is False
    exc = rep["hard"]["root_excursion"]
    assert not exc["pass"]
    assert exc["max_m"] == pytest.approx(2.0, abs=0.01)


def test_excursion_is_relative_to_first_frame(motion_csv):
    # 1.4 m of drift stays under the 1.5 m gate no matter where it starts
    m = make_motion(drift_xy=(1.4, 0.0))
    m[:, 0] += 100.0  # far from the world origin
    rc, rep = run_vet(motion_csv(m))
    assert rep["hard"]["root_excursion"]["pass"]


def test_joint_limit_violation_fails(motion_csv):
    rc, rep = run_vet(motion_csv(joint_overrides={0: 10.0}))
    assert rc == 1
    jl = rep["hard"]["joint_limits"]
    assert not jl["pass"]
    assert jl["worst_violation_rad"] > 1.0


def test_floorwork_fails(motion_csv):
    rc, rep = run_vet(motion_csv(z=0.20))
    assert rc == 1
    ph = rep["hard"]["pelvis_height"]
    assert not ph["pass"]
    assert ph["min_m"] == pytest.approx(0.20, abs=0.01)


def test_velocity_spike_is_advisory_not_fatal(motion_csv):
    # alternate one joint +/-1 rad per frame -> 60 rad/s >> 3*pi limit
    m = make_motion(frames=20)
    m[1::2, 7] = 1.0
    rc, rep = run_vet(motion_csv(m))
    assert rc == 0 and rep["pass"] is True          # advisory only
    jv = rep["advisory"]["joint_velocity"]
    assert jv["peak_rad_s"] > jv["limit"]
    assert jv["frames_over_limit_pct"] > 40
    assert jv["ok"] is False


def test_static_feet_no_skate_warning(motion_csv):
    rc, rep = run_vet(motion_csv(frames=15))
    fs = rep["advisory"]["foot_skate"]
    assert fs["ok"] is True
    assert fs["p95_stance_speed_m_s"] <= 0.01
