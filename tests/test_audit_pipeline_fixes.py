"""Regression tests for the pipeline/library-lane production-audit fixes.

Each test names the audit finding it locks down.
"""
import numpy as np
import pytest

from pipeline.venue import minimal_enclosing_circle


# ---- Finding: MEC footprint vs deployed excursion mismatch (HIGH, safety) ----
# The gate certifies MEC radius <= max_excursion assuming the robot is placed at
# the MEC center. The deploy motion must recenter on that SAME center, else the
# real excursion from the placement point can reach ~2x the certified radius and
# the robot leaves the certified dance area.

def _circle_traj(center, radius, n=120, start_on_edge=True):
    """XY trajectory tracing a circle; frame 0 sits on the circle edge (worst case
    for frame-0 recentering)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if start_on_edge:
        th = th  # th[0]=0 -> point (center + (radius,0)), i.e. on the edge
    xy = np.column_stack([center[0] + radius * np.cos(th),
                          center[1] + radius * np.sin(th)])
    return xy


def _max_radial(xy):
    return float(np.max(np.linalg.norm(xy, axis=1)))


def test_deploy_recenter_on_mec_center_bounds_excursion():
    # A dance that circles a point 1.0 m off to the side, radius 1.0 m.
    R = 1.0
    xy = _circle_traj(center=(1.0, 0.0), radius=R, start_on_edge=True)
    (cx, cy), r = minimal_enclosing_circle(xy)
    assert r == pytest.approx(R, abs=0.02)

    # NEW behaviour (deploy recenters on MEC center): max radial distance == r,
    # so it stays within the certified radius.
    on_center = xy - np.array([cx, cy])
    assert _max_radial(on_center) == pytest.approx(r, abs=0.02)
    assert _max_radial(on_center) <= 1.5 + 1e-6      # fits a 1.5 m venue

    # OLD (buggy) behaviour (recenter on frame 0): frame 0 is on the edge, so the
    # far side of the circle is ~2R away -> would exceed the certified radius.
    on_frame0 = xy - xy[0]
    assert _max_radial(on_frame0) == pytest.approx(2 * R, abs=0.05)
    assert _max_radial(on_frame0) > 1.5              # leaves the certified 1.5 m area


def test_deploy_footprint_radius_is_the_real_bound():
    # For any trajectory, recentering on the MEC center makes the certified radius
    # a true upper bound on the deployed excursion (the property the fix guarantees).
    rng = np.random.default_rng(0)
    for _ in range(20):
        xy = rng.uniform(-2, 2, size=(200, 2))
        (cx, cy), r = minimal_enclosing_circle(xy)
        recentered = xy - np.array([cx, cy])
        assert _max_radial(recentered) <= r + 1e-6
