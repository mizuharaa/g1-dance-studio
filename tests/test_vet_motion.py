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
    # Footprint = enclosing-circle radius. A 3.5 m straight walk has radius 1.75 m,
    # which overflows the 1.5 m default venue. (A 2 m walk would PASS now — radius
    # 1.0 m — because a 2 m line fits a 3 m-diameter area; that is correct geometry
    # and the deliberate change from the old distance-from-start metric.)
    rc, rep = run_vet(motion_csv(drift_xy=(3.5, 0.0)))
    assert rc == 1 and rep["pass"] is False
    exc = rep["hard"]["root_excursion"]
    assert not exc["pass"]
    assert exc["footprint_radius_m"] == pytest.approx(1.75, abs=0.02)


def test_footprint_is_translation_invariant(motion_csv):
    # The footprint radius depends only on the dance's shape, not where in the
    # world it starts: 1.4 m of travel => 0.7 m radius, passes wherever it sits.
    m = make_motion(drift_xy=(1.4, 0.0))
    m[:, 0] += 100.0  # far from the world origin
    rc, rep = run_vet(motion_csv(m))
    assert rep["hard"]["root_excursion"]["pass"]
    assert rep["hard"]["root_excursion"]["footprint_radius_m"] == pytest.approx(
        0.7, abs=0.02)


def test_joint_limit_violation_fails(motion_csv):
    rc, rep = run_vet(motion_csv(joint_overrides={0: 10.0}))
    assert rc == 1
    jl = rep["hard"]["joint_limits"]
    assert not jl["pass"]
    assert jl["worst_violation_rad"] > 1.0


def test_floorwork_fails(motion_csv):
    # GENUINE floorwork: robot lying down (base pitched 90°) → pelvis close to the
    # floor after grounding. A merely-low root z on a STANDING pose is NOT floorwork
    # (audit HIGH: judge pelvis height relative to floor contact, not un-grounded z).
    m = make_motion(frames=10)
    m[:, 3:7] = [0.0, 0.7071, 0.0, 0.7071]   # 90° pitch about y → lying down
    rc, rep = run_vet(motion_csv(m))
    assert rc == 1
    ph = rep["hard"]["pelvis_height"]
    assert not ph["pass"]
    assert ph["min_m"] < 0.35


def test_low_standing_pose_grounded_passes(motion_csv):
    # Regression for the grounding fix: a standing pose with a LOW absolute root z
    # (un-grounded) must PASS — grounding references it to the floor first.
    rc, rep = run_vet(motion_csv(z=0.20))
    assert rc == 0
    assert rep["hard"]["pelvis_height"]["pass"]


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
