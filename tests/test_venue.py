"""Flexible-venue feature: minimal enclosing circle, venue math, gate
parameterization, and the fit-to-venue report. All pure numpy — no robot model."""
import numpy as np
import pytest

from .conftest import make_motion

from pipeline.venue import (
    Venue, _circle_three, create_venue, default_venue, fit_motion_to_venue,
    footprint, get_venue, list_venues, minimal_enclosing_circle)
from pipeline.find_window import longest_window


# ---- brute-force reference MEC (defined by a pair-diameter or a triple) ---- #
def brute_mec(points):
    pts = np.unique(np.asarray(points, float).reshape(-1, 2), axis=0)
    n = len(pts)
    if n == 0:
        return np.zeros(2), 0.0
    if n == 1:
        return pts[0], 0.0
    best = None
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            c = (pts[i] + pts[j]) / 2.0
            cand.append((c, float(np.linalg.norm(pts[i] - c))))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cc = _circle_three(pts[i], pts[j], pts[k])
                if cc is not None:
                    cand.append((cc[0], cc[1]))
    for c, r in cand:
        if np.all(np.linalg.norm(pts - c, axis=1) <= r + 1e-7):
            if best is None or r < best[1] - 1e-12:
                best = (c, r)
    return best


@pytest.mark.parametrize("seed", range(15))
def test_mec_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-5, 5, size=(rng.integers(1, 25), 2))
    _, r = minimal_enclosing_circle(pts)
    _, rb = brute_mec(pts)
    assert r == pytest.approx(rb, abs=1e-4)


def test_mec_degenerate_cases():
    assert minimal_enclosing_circle(np.zeros((0, 2)))[1] == 0.0        # empty
    assert minimal_enclosing_circle([[2.0, 1.0]])[1] == 0.0            # single
    c, r = minimal_enclosing_circle([[0.0, 0.0], [2.0, 0.0]])          # two
    assert r == pytest.approx(1.0) and c == pytest.approx([1.0, 0.0])
    c, r = minimal_enclosing_circle([[0, 0], [1, 0], [2, 0], [3, 0]])  # collinear
    assert r == pytest.approx(1.5) and c == pytest.approx([1.5, 0.0])
    c, r = minimal_enclosing_circle([[0, 0], [0, 0], [0, 0]])          # duplicates
    assert r == pytest.approx(0.0)


def test_footprint_of_straight_walk_is_half_length():
    t = np.linspace(0, 1, 40)
    xy = np.c_[3.0 * t, np.zeros_like(t)]
    c, r = footprint(xy)
    assert r == pytest.approx(1.5, abs=1e-3)          # 3 m walk => 1.5 m radius
    assert c == pytest.approx([1.5, 0.0], abs=1e-3)


# ---- venue model / max-excursion derivation ---- #
def test_default_venue_reproduces_legacy_gate():
    v = default_venue()
    assert v.max_excursion_m == pytest.approx(1.5)   # 2 m radius - 0.5 m margin


def test_rectangle_uses_shorter_side():
    v = Venue(id="x", name="Stage", shape="rectangle",
              width_m=3.0, depth_m=5.0, margin_m=0.5)
    assert v.max_excursion_m == pytest.approx(1.0)    # min(3,5)/2 - 0.5


def test_margin_cannot_go_negative():
    v = Venue(id="x", name="Tiny", shape="circle", radius_m=0.3, margin_m=0.5)
    assert v.max_excursion_m == 0.0


def test_venue_persistence_roundtrip(tmp_path, monkeypatch):
    import pipeline.venue as venue_mod
    monkeypatch.setattr(venue_mod, "VENUES_DIR", tmp_path)
    v = create_venue("Ballroom", shape="circle", radius_m=5.0, margin_m=0.5)
    assert get_venue(v.id).max_excursion_m == pytest.approx(4.5)
    names = [x.name for x in list_venues()]
    assert "Ballroom" in names


def test_list_venues_seeds_default_when_empty(tmp_path, monkeypatch):
    import pipeline.venue as venue_mod
    monkeypatch.setattr(venue_mod, "VENUES_DIR", tmp_path)
    vs = list_venues()
    assert len(vs) == 1 and vs[0].max_excursion_m == pytest.approx(1.5)


# ---- the gate is parameterized by the venue ---- #
def test_window_search_respects_venue_size():
    # 6 m straight walk => footprint radius 3 m.
    m = make_motion(frames=120, drift_xy=(6.0, 0.0))
    s_small, e_small = longest_window(m, max_excursion_m=1.5)
    s_big, e_big = longest_window(m, max_excursion_m=4.0)
    # small venue keeps only a sub-window; a 4 m venue (radius 3 fits) holds it all
    assert (e_small - s_small) < len(m) - 1
    assert (s_big, e_big) == (0, len(m) - 1)


def test_optimal_recenter_fits_offcentre_circle():
    # A dance that circles a point 3 m from the origin, loop radius 0.8 m.
    # Old "distance from first frame" would read ~3.8 m and fail; the footprint
    # radius is 0.8 m, so it fits a 1.5 m venue and the WHOLE motion is one window.
    th = np.linspace(0, 2 * np.pi, 80)
    m = make_motion(frames=80)
    m[:, 0] = 3.0 + 0.8 * np.cos(th)
    m[:, 1] = 0.8 * np.sin(th)
    _, r = footprint(m[:, 0:2])
    assert r == pytest.approx(0.8, abs=1e-2)
    s, e = longest_window(m, max_excursion_m=1.5)
    assert (s, e) == (0, 79)          # everything fits — not truncated


# ---- fit_motion_to_venue report ---- #
def test_fit_report_whole_dance_fits():
    m = make_motion(frames=60, drift_xy=(1.0, 0.0))   # radius 0.5
    rep = fit_motion_to_venue(m, default_venue())
    assert rep["fits_whole"] is True
    assert rep["footprint_radius_m"] == pytest.approx(0.5, abs=1e-2)
    assert rep["min_venue_radius_m"] == pytest.approx(1.0, abs=1e-2)  # +margin
    assert "suggested_window" not in rep
    assert all(t["in_bounds"] for t in rep["timeline"])


def test_fit_report_overflow_offers_window_and_bigger_venue():
    m = make_motion(frames=120, drift_xy=(6.0, 0.0))  # radius 3 m, overflows 1.5
    rep = fit_motion_to_venue(m, default_venue())
    assert rep["fits_whole"] is False
    assert rep["footprint_radius_m"] == pytest.approx(3.0, abs=1e-2)
    assert rep["min_venue_radius_m"] == pytest.approx(3.5, abs=1e-2)
    assert 0.0 < rep["suggested_window"]["duration_s"] < rep["duration_s"]
    ids = {o["id"] for o in rep["options"]}
    assert ids == {"window", "resize_venue", "cancel"}   # never "in_place"
    assert any(not t["in_bounds"] for t in rep["timeline"])
