# A2 — Fix F8: grounding deletes sustained airborne phases (MEDIUM, blocks A3)

**Finding:** REPORT §F8 + `crosscheck.json` synthetic-flight rows. The floor
estimate is a 9-frame SG low-pass of the SAME contact-height signal it
classifies, so a sustained jump (0.20 m, 2 s plateau) reads as a new floor:
0 flight frames detected, plateau subtracted to ~0. Also `per_contact_height`
takes min over ALL robot geom CENTERS, not foot sole surfaces
(`pipeline/grounding.py:53-68,121-187`).

**Owned files:** `pipeline/grounding.py`, `tests/test_grounding_flight.py` (new).
Public API (`ground_motion(m, model) -> (m, info)`) must keep its signature —
callers in the motion pipeline and repair ladder depend on it.

## Spec
1. **Sole-surface height:** resolve the FOOT collision geoms by name pattern
   (`^(left|right)_foot[1-7]_collision$` if present in the model; fall back to
   the two `*_ankle_roll_link` body geoms) and measure height as geom LOWEST
   SURFACE (center z − geom size along world z for spheres/capsules/boxes —
   use mujoco geom size + orientation; a conservative `center - size.max()` is
   acceptable and must be documented), per foot.
2. **Support-gated floor estimate:** classify per-frame support BEFORE floor
   fitting: a foot is a support candidate when its sole height is within 3 cm of
   the current floor estimate AND its vertical speed < 0.2 m/s. Update the
   slowly-varying floor (low-pass) ONLY from frames with ≥1 support foot; during
   no-support intervals HOLD the last floor value (no update), so a jump plateau
   cannot become the floor. Seed the estimate from the first 0.5 s (assume
   grounded start — assert and warn if not).
3. **Flight preserved:** root-z correction subtracts the HELD floor, so genuine
   flight (both feet above floor+8 cm for ≥4 frames) survives with its true
   height; keep reporting flight windows in `info`.
4. **Info fields:** add `flight_windows_s`, `floor_drift_m` (first→last floor),
   `support_pct`; keep existing keys.
5. **Tests (synthetic, no thriller dependency):** (a) standing + linear 0.15 m
   camera drift → drift removed, 0 flight; (b) 2 s 0.20 m jump plateau → plateau
   HEIGHT preserved within 2 cm and flagged as flight; (c) short 200 ms hop →
   preserved+flagged; (d) alternating single-support walk with drift → drift
   removed, steps not flattened; (e) crouch (feet grounded, root low) → NOT
   flagged as flight. Build tiny synthetic (N,36) motions with the real G1 model
   via `pipeline.g1_limits.build_model()`.

## Acceptance
Existing grounding tests still green; the 5 new cases pass; running the tool on
`data/motions/thriller/thriller_deploy.csv` yields floor-height range ≤ 25 mm
and 0% flight (matches the audit's grounded comparator); signature unchanged.

## Validation
```bash
python -m pytest tests/test_grounding_flight.py tests/ -q -k "ground"
```
Then hand off to A3 (v12 rebuild uses this).
