# A1 — Fix F1: effort randomization is silently GLOBAL (CRITICAL)

**Finding:** `experiments/external_audit_20260721/REPORT.md` §F1 +
`PINNED_MJLAB_EVIDENCE.md` §effort. mjlab's `dr.effort_limits` selects
`asset_cfg.actuator_ids` (default `slice(None)`) and NEVER reads `joint_names`.
Our three effort events (`cloud/sim2real_task_v8.py:450-477`, base event
rescope included) all pass `joint_names=...`, so the LAST-inserted event
(scale 0.52–0.76) hits **every actuator group**: wrists 2.6–3.8 Nm, arms 13–19,
ankles+waistRP 26–38, hips 45.8–66.9, knee/hipR 72.3–105.6. Every v8+ policy
trained on a globally weakened robot.

**Owned files:** `cloud/sim2real_task_v8.py` (primary), `cloud/sim2real_task.py`
(base `dr_effort_limits` rescope only if needed), `tests/test_effort_scope*.py`
(new). Do NOT touch v10/v11 (they inherit `_apply_v8`).

## Spec
1. Write a custom startup event `scoped_effort_limits(env, env_ids,
   ranges_by_ctrl, ...)` in `sim2real_task_v8.py` that:
   - at first call resolves ANKLE control ids by iterating
     `asset.actuators` and matching each actuator's `target_names` /
     joint targets against `base.ANKLE_JOINT_NAMES` (do NOT assume one
     group == one control; assert every ankle joint resolved exactly once,
     and assert the 6 expected high-level group names are present — fail loud
     on any mismatch with the pinned wheel's `G1_ARTICULATION`);
   - applies scale U(0.80,1.00) to all NON-ankle controls and
     U(0.52,0.76) to the 4 ankle controls, writing from
     `env.sim.get_default_field("actuator_forcerange")` exactly like the wheel
     does (one event, no overwrite hazard);
   - REPLACES all three existing effort events (`dr_effort_limits`,
     `dr_ankle_effort_clamp`, `dr_effort_limits_ankle`) with this single event.
     Keep the two legacy event KEYS as documented no-op aliases ONLY if the
     selfcheck still asserts their presence — otherwise update the selfcheck
     accordingly (preferred: selfcheck asserts the NEW event + realized ranges).
2. Selfcheck upgrade (`_selfcheck` in v8, plus v10/v11 inherit): after building
   the train cfg, print the INTENDED per-joint [min,max] table (29 rows) and
   assert the event's resolved ankle-ctrl count == 4. (Realized-model assertion
   is GPU-only: emit the check as part of the event itself — after applying, read
   back `env.sim.model.actuator_forcerange` for env 0 and print min/max per
   ctrl; assert ankle controls ≤ 38*1.001 and non-ankle ≥ nominal*0.79. This
   runs on the box smoke, free on CPU-imported selfcheck path it must be
   skipped gracefully.)
3. CPU tests (`tests/test_effort_scope.py`, no mjlab): unit-test the pure
   resolution/assertion logic by faking a minimal `asset.actuators` structure
   (mirror the wheel's attribute names from PINNED_MJLAB_EVIDENCE) — verify
   (a) ankle ctrl ids resolve to exactly the 4 ankle joints, (b) a fabricated
   grouping where one actuator owns multiple controls still splits correctly,
   (c) non-ankle range 0.80–1.00 and ankle 0.52–0.76 are assigned to the right
   ids, (d) a missing/renamed group raises. Structure the event so this logic is
   an importable pure function (no env needed).

## Acceptance
- Exactly ONE effort event active in the train cfg; ankle-only scoping proven by
  the pure-function tests; realized-forcerange readback code present and
  box-gated; selfcheck prints the 29-row intended table; `py_compile` on v8/v10/
  v11; new tests green; no change to deploy/play cfg (events are train-only).
- Update the header comment block that previously described "AUDIT FIX F" —
  it must now describe the ACTUAL semantics (scale-from-default, actuator-ids,
  single event), citing F1.

## Validation
```bash
python -m py_compile cloud/sim2real_task_v8.py cloud/sim2real_task_v10.py cloud/sim2real_task_v11.py
python -m pytest tests/test_effort_scope.py -q
```
GPU (later, NOT this wave): 64-env smoke prints realized table; a run is
forbidden unless every control is in its intended band.
