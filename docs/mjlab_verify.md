# Held-out mjlab verification gate

Built because the different-simulator sim2sim exam (`pipeline/sim_exam.py`) is blocked:
its plain-MuJoCo G1 model isn't dynamically faithful (a static pose-hold collapses at
1.38 s — see `docs/exam_physics_fix.md`), so it honestly returns `verdict="invalid"`
rather than a false pass/fail. This gate is the pragmatic replacement.

## What it is (and isn't)

- **Method:** `mjlab_heldout_v1`. Evaluate the trained policy **in mjlab** (the faithful
  training engine) under conditions the policy never trained on: **disjoint held-out
  seeds**, **observation corruption** (sensor noise), and **external shoves** (mjlab's
  `push_robot` base-velocity impulses). N=128 parallel episodes, each from motion start.
- **Catches:** a policy that overfit its training seeds, tracks poorly, or can't take a
  shove — cheaply, before we risk the robot.
- **Does NOT catch:** mjlab-specific physics exploitation (same engine as training), and
  it is **not** a substitute for gantry-first robot-day validation. The real independent
  gate is the physical robot.

## Pieces

- `cloud/heldout_eval.py` — box-side eval (adapted from mjlab `tasks/tracking/scripts/
  evaluate.py`): local checkpoint + local motion, held-out seed, nominal + push
  conditions, writes success-rate + tracking-error JSON. `success = survived the full
  motion without termination` (mjlab's own definition).
- `pipeline/mjlab_verify.py` — laptop-side: turns that JSON into a **signed `sim_exam/v1`
  verdict** (reusing `pipeline/exam_verdict.py` signing). Flows through the SAME
  `authorize()`/`derive_pass()` gate as any verdict — show-ready still requires all phases
  pass, the push force floor (`m*dv/dt` impulse-equivalent, ~875 N ≥ 150 N floor), and
  **repeatability clean == runs (every held-out episode survived)**.
- `tests/test_mjlab_verify.py` — 100%→authorizes, 98.4%→refused, tamper→signature breaks.

## Real Thriller result (model_3000, thriller_show motion, N=128, seed 90001)

| Condition | Survival | mpkpe (keypoint err) |
|---|---|---|
| Nominal held-out (noise) | **126/128 = 98.4%** | 0.168 m |
| Push held-out (noise + shoves) | **126/128 = 98.4%** | 0.173 m |

**Verdict: `fail` — does NOT authorize show-ready, correctly.** Thriller is strong and
shove-robust, but 98.4% is not "clean every time": ~1 in 60 held-out starts falls, and
the strict repeatability bar (clean==runs) is not met. Signed verdict at
`data/policies/thriller/heldout_verdict.json`; Thriller stays **DRAFT**.

## Recommendation

The gate is working and honest. To clear it, the policy needs to be tighter (the mpkpe
~0.17 m and the 1.6% held-out failure suggest room): retrain Thriller longer / with the
recipe's conditional deltas (attempt 2 of ≤3), or make the show-ready acceptance
threshold an explicit product decision (e.g. accept ≥99% held-out + gantry as the gate)
rather than a hard 100%. Either way, robot-day gantry validation remains the real gate.

Findings routed earlier stand: `base_lin_vel` isn't measurable on the real G1 (deploy
needs a state estimator); `action_scale` is per-joint (0.074 wrists), already in
`policy_meta.json`.
