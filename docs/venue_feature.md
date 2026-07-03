# Flexible-venue feature — contract & behaviour

Built 2026-07-04 (user-requested). Replaces the hardcoded 1.5 m root-excursion
gate with a configurable **venue** and transparent overflow handling. **"Dance in
place" (root-motion stripping) is deliberately NOT implemented** — excluded by the
user.

## The mental model (why this exists)

There is no boundary "wall" in training. The robot tracks the reference motion's
global path and simply follows it — it has no concept of an edge. The spatial
limit is a property we check on the **reference motion**, and it exists for
physical reasons: the venue is a real fixed size, the robot's position estimate
drifts over distance, and safety margin (fall clearance, e-stop reach) shrinks as
it roams. So we model the venue explicitly and, on overflow, we show the numbers
and let a human choose — we never silently truncate or mangle the dance.

## Correctness upgrade: footprint = minimal enclosing circle

The old gate measured "max distance from the first frame". That over-counts a
dance that merely starts off-centre (circling a point 1 m away read as 2 m). The
new metric is the **minimal enclosing circle** of the root-XY trajectory
(`pipeline.venue.minimal_enclosing_circle`, Welzl, deterministic). Its radius is
the dance's intrinsic **footprint** — the smallest circular venue that holds the
whole dance *when the robot is placed at the circle's centre*. It is
translation-invariant.

**Behaviour change (documented):** a straight 2 m walk now PASSES a 1.5 m venue
(footprint radius 1.0 m — a 2 m line fits a 3 m-diameter circle), whereas the old
distance-from-start metric failed it. This is correct geometry. Deploy caveat:
the fit guarantee holds only if the robot is positioned at `footprint_center_xy`
at showtime — surface that to the operator.

Tests updated to the new semantics: `test_vet_motion.py`
(`test_excursion_beyond_limit_fails` now uses 3.5 m → radius 1.75 > 1.5;
`test_footprint_is_translation_invariant`) and `test_find_window.py`
(`test_excursion_break_starts_new_window`, `test_cli_out_recenters_xy` — the CLI
`--out` now re-centres on the footprint centre, not the first frame).

## Venue model (`pipeline/venue.py`)

`Venue`: `id, name, shape ("circle"|"rectangle"), radius_m, width_m, depth_m,
margin_m`. Derived `max_excursion_m` = (circle: radius; rectangle: half the
shorter side) − margin, floored at 0. Default margin 0.5 m. JSON store under
`data/venues/`. Default venue **"Home (2 m)"** (radius 2.0, margin 0.5 →
max_excursion 1.5) reproduces the historical gate, so nothing regresses.

Functions: `list_venues()` (seeds the default if empty), `get_venue(id)`,
`create_venue(...)`, `save_venue(v)`, `default_venue()`,
`footprint(xy)`, `minimal_enclosing_circle(points)`,
`fit_motion_to_venue(motion, venue)`.

## Gate parameterization

`vet_motion.py` and `find_window.py` take the venue's `max_excursion_m`:
- `find_window.longest_window(m, max_excursion_m=1.5)` — window validity is now the
  window's enclosing-circle radius ≤ `max_excursion_m`. `window_center(m, s, e)`
  gives the placement point.
- `vet_motion.py` reads env **`G1_MAX_EXCURSION_M`** (default 1.5) — the app sets
  it when running vet as a subprocess for the selected venue. The report's
  `hard.root_excursion` now carries `footprint_radius_m`, `footprint_center_xy`,
  `limit`, `pass` (and `max_m` = footprint radius, kept for existing consumers).

## `fit_motion_to_venue(motion, venue)` → report

```json
{
  "venue": { ...venue fields..., "max_excursion_m": 1.5 },
  "footprint_radius_m": 3.0,
  "footprint_center_xy": [1.5, 0.0],
  "fits_whole": false,
  "min_venue_radius_m": 3.5,          // footprint + margin: smallest venue that fits ALL
  "duration_s": 4.0,
  "timeline": [ {"t_s": 0.0, "max_dist_m": 0.4, "in_bounds": true}, ... ],  // per second
  "suggested_window": {               // present only when !fits_whole
    "start_s": 0.0, "end_s": 2.1, "duration_s": 2.1,
    "recenter_xy": [0.7, 0.0], "covers_fraction": 0.52
  },
  "options": [                        // present only when !fits_whole (NO in-place)
    {"id": "window",       "label": "Trim to the longest in-venue section", "detail": "..."},
    {"id": "resize_venue", "label": "Use a larger venue",                   "detail": "..."},
    {"id": "cancel",       "label": "Cancel",                               "detail": "..."}
  ]
}
```

## API the UI should add (endpoints NOT built here — for the UI/server owner)

These are the shapes the Settings/Create screens need; implement in `ui/server.py`
against `pipeline/venue.py` (no in-place option anywhere):

- `GET  /api/venues` → `[venue.to_public(), ...]`
- `POST /api/venues` `{name, shape, radius_m|width_m+depth_m, margin_m}` → created venue
- `GET  /api/venues/{id}` → venue
- `POST /api/motion/fit` `{motion_csv | job_id, venue_id}` → the fit report above
- Persist a **selected venue** per job/dance (add `venue_id` to the job/dance
  record), and pass its `max_excursion_m` via `G1_MAX_EXCURSION_M` when the
  retarget stage runs `vet_motion.py` / `find_window.py`.

### Overflow dialog UX (Create screen)

When `fits_whole` is false, present the numbers plainly — "This dance needs a
**{footprint_radius_m} m** radius; your venue allows **{max_excursion_m} m**" — and
the three `options`: **Trim** (show it keeps `suggested_window.duration_s` of
`duration_s`), **Use a larger venue** (needs ≥ `min_venue_radius_m`), **Cancel**.
Never auto-apply; the dance is only ever windowed on explicit choice.

## Follow-up (not done here; other owners)

- `pipeline/sim_exam.py`'s excursion threshold should read the selected venue's
  `max_excursion_m` instead of a constant, so the signed exam matches the venue
  the dance was fitted to. (Left to the sim_exam owner — not edited here.)
- `pipeline/stages/local_motion.py` currently re-centres the deployed segment on
  the window's first frame; it should adopt `find_window.window_center()` so the
  deployed dance is centred in the venue (correctness-neutral for training, since
  the footprint metric is translation-invariant, but it aligns the physical
  placement with `footprint_center_xy`).
