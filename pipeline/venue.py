"""Configurable performance venue + honest overflow handling for dance motions.

Background — there is NO boundary "wall" in training. The robot is taught to
track the reference motion's global path and simply follows it; it has no concept
of an edge. The spatial limit is therefore a property we check on the REFERENCE
motion, and it exists for physical reasons: the venue is a real fixed size, the
robot's position estimate drifts over distance, and safety (fall margin, e-stop
reach) shrinks as it roams.

So we model the venue explicitly and, when a reference overflows it, we are
transparent about it rather than silently truncating — the consumer chooses to
accept a windowed cut, size the venue up, or cancel. We do NOT offer "dance in
place" (stripping the global travel) — excluded by product decision.

The correctness core is `minimal_enclosing_circle`: the smallest circle covering
the whole root-XY trajectory. Its radius is the dance's intrinsic FOOTPRINT — the
smallest circular venue that holds the entire dance, if the robot is placed at the
circle's centre. This is strictly better than "max distance from the first frame",
which penalises a dance merely for starting off-centre (e.g. circling a point 1 m
away reads as 2 m of excursion but has a 1 m footprint).

CSV convention (project-wide): 36 cols, 0:2 root XY. See pipeline.motion_io.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import DATA_DIR

VENUES_DIR = DATA_DIR / "venues"
CSV_FPS = 30.0

# Safety margin subtracted from the raw venue size: footprint of the standing
# robot + a buffer for positional drift and keeping the e-stop within reach.
DEFAULT_MARGIN_M = 0.5
# The default venue reproduces the historical hardcoded gate exactly:
# a 2 m-radius area minus a 0.5 m margin => 1.5 m max root excursion.
DEFAULT_VENUE = {"name": "Home (2 m)", "shape": "circle", "radius_m": 2.0,
                 "margin_m": DEFAULT_MARGIN_M}


# --------------------------------------------------------------------------- #
# Minimal enclosing circle (Welzl, randomised incremental) — the footprint math
# --------------------------------------------------------------------------- #
def _circle_two(a, b):
    c = (a + b) / 2.0
    return c, float(np.linalg.norm(a - c))


def _circle_three(a, b, c):
    """Circumscribed circle of 3 points, or None if (near) collinear."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    centre = np.array([ux, uy])
    return centre, float(np.linalg.norm(a - centre))


def _in_circle(centre, r, p, eps=1e-9):
    return np.linalg.norm(p - centre) <= r + eps


def _mec_trivial(boundary):
    if not boundary:
        return np.zeros(2), 0.0
    if len(boundary) == 1:
        return boundary[0].astype(float), 0.0
    if len(boundary) == 2:
        return _circle_two(boundary[0], boundary[1])
    c = _circle_three(*boundary[:3])
    if c is not None:
        return c
    # collinear triple: enclosing circle is the two farthest points
    best = _circle_two(boundary[0], boundary[1])
    for i in range(3):
        for j in range(i + 1, 3):
            cc, rr = _circle_two(boundary[i], boundary[j])
            if rr > best[1] and all(_in_circle(cc, rr, boundary[k]) for k in range(3)):
                best = (cc, rr)
    return best


def minimal_enclosing_circle(points: np.ndarray, seed: int = 12345):
    """Smallest circle enclosing all (x, y) points → (centre_xy, radius).

    Randomised-incremental Welzl, expected O(n). Deterministic given `seed` so
    tests are stable. Duplicate/collinear/degenerate inputs are handled."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros(2), 0.0
    # de-duplicate to keep the boundary logic robust, then shuffle deterministically
    pts = np.unique(pts, axis=0)
    rng = np.random.default_rng(seed)
    pts = pts[rng.permutation(len(pts))]

    centre, r = pts[0].astype(float), 0.0
    for i in range(1, len(pts)):
        if _in_circle(centre, r, pts[i]):
            continue
        # pts[i] on boundary; rebuild with it fixed
        centre, r = pts[i].astype(float), 0.0
        for j in range(i):
            if _in_circle(centre, r, pts[j]):
                continue
            centre, r = _circle_two(pts[i], pts[j])
            for k in range(j):
                if _in_circle(centre, r, pts[k]):
                    continue
                c3 = _circle_three(pts[i], pts[j], pts[k])
                if c3 is not None:
                    centre, r = c3
    return centre, float(r)


def footprint(xy: np.ndarray):
    """(centre_xy, radius_m) intrinsic footprint of a root-XY trajectory."""
    return minimal_enclosing_circle(xy)


# --------------------------------------------------------------------------- #
# Venue model + persistence
# --------------------------------------------------------------------------- #
@dataclass
class Venue:
    id: str
    name: str
    shape: str = "circle"          # "circle" | "rectangle"
    radius_m: float = 2.0          # circle
    width_m: float = 0.0           # rectangle
    depth_m: float = 0.0           # rectangle
    margin_m: float = DEFAULT_MARGIN_M
    created_at: float = 0.0

    @property
    def max_excursion_m(self) -> float:
        """Largest footprint radius that safely fits, after the margin.

        Circle: radius - margin. Rectangle: half the SHORTER side - margin
        (a circular footprint is bounded by the tightest dimension)."""
        if self.shape == "rectangle":
            base = min(self.width_m, self.depth_m) / 2.0
        else:
            base = self.radius_m
        return round(max(0.0, base - self.margin_m), 4)

    def to_public(self) -> dict:
        d = asdict(self)
        d["max_excursion_m"] = self.max_excursion_m
        return d


def _durable_write(path: Path, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def _slug(name: str) -> str:
    keep = "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")
    return keep or "venue"


def save_venue(v: Venue) -> Venue:
    VENUES_DIR.mkdir(parents=True, exist_ok=True)
    tmp = VENUES_DIR / f"{v.id}.json.tmp"
    _durable_write(tmp, json.dumps(asdict(v), indent=2))
    os.replace(tmp, VENUES_DIR / f"{v.id}.json")
    return v


def create_venue(name: str, shape: str = "circle", *, radius_m: float = 2.0,
                 width_m: float = 0.0, depth_m: float = 0.0,
                 margin_m: float = DEFAULT_MARGIN_M) -> Venue:
    v = Venue(id=_slug(name) + "-" + time.strftime("%H%M%S"), name=name,
              shape=shape, radius_m=radius_m, width_m=width_m, depth_m=depth_m,
              margin_m=margin_m, created_at=time.time())
    return save_venue(v)


def list_venues() -> list[Venue]:
    """All venues; ensures the default 'Home (2 m)' exists so behaviour never
    regresses to an empty set."""
    VENUES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(VENUES_DIR.glob("*.json")):
        try:
            out.append(Venue(**json.loads(p.read_text())))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue  # skip a corrupt venue file rather than break the list
    if not out:
        out.append(create_venue(**DEFAULT_VENUE))
    return out


def get_venue(venue_id: str) -> Venue | None:
    p = VENUES_DIR / f"{venue_id}.json"
    if not p.is_file():
        return None
    try:
        return Venue(**json.loads(p.read_text()))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def default_venue() -> Venue:
    """The default venue object (1.5 m max excursion), not necessarily saved."""
    return Venue(id="home-2m", name=DEFAULT_VENUE["name"], shape="circle",
                 radius_m=DEFAULT_VENUE["radius_m"], margin_m=DEFAULT_MARGIN_M)


# --------------------------------------------------------------------------- #
# Fit a motion to a venue → transparent report (never mutates the dance)
# --------------------------------------------------------------------------- #
def fit_motion_to_venue(motion: np.ndarray, venue: Venue) -> dict:
    """Analyse a motion against a venue and return a decision REPORT.

    No in-place transform, no silent truncation: the caller decides what to do
    with `fits_whole` / the suggested window / the min venue that would fit.

    `motion` is the (N, 36) array (grounded or not — XY is unaffected by z).
    """
    from .find_window import longest_window, window_center

    xy = np.asarray(motion)[:, 0:2]
    centre, radius = footprint(xy)
    maxexc = venue.max_excursion_m
    fits = radius <= maxexc + 1e-6

    # per-second in/out timeline, relative to the footprint centre (the optimal
    # placement): a frame is out if it lies beyond maxexc from that centre.
    dist = np.linalg.norm(xy - centre, axis=1)
    step = max(1, int(round(CSV_FPS)))
    timeline = [{"t_s": round(i / CSV_FPS, 1),
                 "max_dist_m": round(float(dist[i:i + step].max()), 3),
                 "in_bounds": bool(dist[i:i + step].max() <= maxexc + 1e-6)}
                for i in range(0, len(dist), step)]

    report = {
        "venue": venue.to_public(),
        "footprint_radius_m": round(float(radius), 3),
        "footprint_center_xy": [round(float(centre[0]), 3),
                                round(float(centre[1]), 3)],
        "fits_whole": bool(fits),
        "min_venue_radius_m": round(float(radius) + venue.margin_m, 3),
        "duration_s": round(len(motion) / CSV_FPS, 2),
        "timeline": timeline,
    }

    if not fits:
        s, e = longest_window(motion, max_excursion_m=maxexc)
        wcx, wcy = window_center(motion, s, e)
        report["suggested_window"] = {
            "start_s": round(s / CSV_FPS, 2),
            "end_s": round((e + 1) / CSV_FPS, 2),
            "duration_s": round((e - s + 1) / CSV_FPS, 2),
            "recenter_xy": [round(float(wcx), 3), round(float(wcy), 3)],
            "covers_fraction": round((e - s + 1) / max(1, len(motion)), 3),
        }
        # honest options the consumer chooses between (NO in-place)
        report["options"] = [
            {"id": "window", "label": "Trim to the longest in-venue section",
             "detail": f"Keeps {report['suggested_window']['duration_s']}s of "
                       f"{report['duration_s']}s; drops the rest."},
            {"id": "resize_venue", "label": "Use a larger venue",
             "detail": f"Needs a venue of at least "
                       f"{report['min_venue_radius_m']} m radius to hold the "
                       f"whole dance."},
            {"id": "cancel", "label": "Cancel",
             "detail": "Pick a different dance or re-choreograph a smaller footprint."},
        ]
    return report
