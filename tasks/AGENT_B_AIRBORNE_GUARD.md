# AGENT B — Airborne / ground-contact guard (stop the "thrashing when suspended")

**Round:** 2026-07-13. **Owner:** coding agent (💻 laptop) **+ human for gantry validation** (🤖).
**Branch:** `feat/airborne-contact-guard`. **Needs:** this repo (CPU), then the robot on a gantry.

## The problem (root-caused this session — read it)
When the robot is hoisted / barely touching the ground and switched into a policy, it thrashes
(limbs flying, ~360° spin). Mechanism: `LegOdometry` **assumes a planted foot** and back-computes
base velocity from leg motion (`pipeline/leg_odometry.py:8-12`). Suspended → it fabricates a phantom
`base_lin_vel`, the policy "sees" motion and over-corrects → positive-feedback thrash, plus no ground
reaction means torques produce runaway joint motion. **Neither guard catches it:**
`_check_start_upright` and `_fall_signal` are **orientation-only** (`R_base[2,2]` + height) — a robot
hanging vertically passes both. This is the closed-loop failure to fix.

## Hard constraint (found this session)
This G1's `unitree_hg` LowState has **NO foot-force sensor** (`foot_force` exists only on
`unitree_go`). So the guard is a **kinematic / IMU heuristic** — it CANNOT read contact directly and
**MUST be hardware-validated before it is trusted / enabled by default.**

## Design (implement in `pipeline/deploy_runtime.py`; unit-test in `tests/test_airborne_guard.py`)
1. **Start guard `_check_feet_planted(...)` at t0**, called BEFORE `_release_motion_service()` /
   before the policy loop (alongside `_check_start_upright`). At the ready pose the grounded robot is
   ~still; check that the two feet's *implied* base velocities (`LegOdometry.estimate(...)["per_foot_v"]`)
   **agree and are near-zero** over a short settle window. Suspended → they disagree / are non-zero at
   rest. On failure → `SystemExit("REFUSED: robot does not appear to be standing on the ground …")`
   (safe: onboard still holds; nothing released).
2. **In-loop trip (debounced, env-gated `AIRBORNE_TRIP`, default OFF until validated)**: if the two
   per-foot base-velocity estimates diverge beyond `AIRBORNE_DIVERGENCE` for `AIRBORNE_CONFIRM_TICKS`
   consecutive ticks (contact-confidence collapse — cf. `FusedBaseEstimator`'s `contact` term,
   `leg_odometry.py:227`), raise → the existing except/finally damps + hands back. **Conservative
   threshold + debounce** so a normal single-foot step never trips it.

## Validation (🤖 gantry — the human step; the guard is NOT trusted until this passes)
- **True positive:** robot hoisted feet-off → the start guard REFUSES; with the in-loop trip on, a
  mid-run hoist damps within the debounce.
- **No false positives:** a full normal grounded Thriller neither refuses nor trips (0 false trips
  across the whole run + a few stepping/settling moments).
- Only after both: flip the in-loop trip default to ON and raise start-guard from advisory to enforcing.

## Deliverable
`_check_feet_planted` + optional in-loop trip, tests (spoofed suspended vs grounded per-foot vels),
a telemetry note, and a `PROJECT_STATE.md` entry with the gantry evidence. Coordinate the merge with
Agent A/C — **this lane owns `pipeline/deploy_runtime.py`**, so land it before other deploy edits.

## Safety
Fail-CLOSED only: a guard may REFUSE or DAMP, never energize. Keep it OFF-by-default until the gantry
test proves no false positives (a false trip mid-show is safe but disruptive; a false refuse is fine).
