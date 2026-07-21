# A4 — Fix F4: honest eval conditions + exact horizon (HIGH)

**Finding:** REPORT §F4. Every gate condition builds from `play=False` → keeps
startup DR, RSI, the v5 default 0–20 ms cmd delay and 1-step obs delay even in
"nominal" (only cmd delay >0 is overridden; obs delay never cleared); seeds
differ per condition; episode = duration + 0.2 s → 10 extra steps → pinned mjlab
wraps survivors to frame zero (teleport) and rescoring begins.

**Owned files:** `cloud/sim_gap_check.py`, `cloud/heldout_eval.py`,
`cloud/pick_checkpoint.py` (only if row names change), `tests/test_eval_harness.py`
(new; CPU-testable pieces only).

## Spec
1. **Explicit condition construction.** Replace in-place mutation with a builder
   `make_condition_cfg(base_play_cfg, *, seed, cmd_delay_steps, obs_delay_steps,
   noise, push, startup_dr)` starting from the PLAY cfg (no DR/RSI) and adding
   ONLY what the row names: a `clean` row (nothing), one-factor rows
   (`cmd_delay*`, `obs_delay*`, `noise`, `push`, `dr`), and the existing
   composite rows relabeled honestly (`dr+delay40ms+push` etc.). Zeroing must be
   explicit: set cmd/obs delay lags to 0 unless the row sets them.
2. **Paired seeds:** same seed for every row (common random numbers); seed list
   for repetition, recorded per row.
3. **Realized manifest:** per condition write the CONVENTIONS §3.4 `realized`
   block into gap.json from the FINAL cfg values (not intentions).
4. **Exact horizon:** `episode_length_s = motion frames / fps` EXACTLY (no
   +0.2); assert steps_run ≤ T in the rollout loop; success = reached T. Entry/
   exit handoff is explicitly out of scope (separate future scenario) — note it.
5. **Naming/back-compat:** keep old row names emitted as aliases where the
   semantics ACTUALLY match; otherwise new names (`clean`, `dr_nominal`, ...) —
   `pick_checkpoint` SCREEN_ONLY updated to `clean,dr_delay40ms_push`.
   gap.json gains `harness_version: 2`; comparisons across versions must check it.
6. **Tests (no mjlab):** factor out the pure pieces — condition-table builder,
   realized-block extraction, horizon arithmetic — into importable functions and
   unit-test: clean row has zero delays/DR; one-factor rows touch exactly one
   knob; horizon equals T; seeds paired; harness_version stamped.

## Acceptance
`clean` provably contains no DR/delay/noise/push by construction (function
inspection + tests); every row records its realized block; no +0.2 anywhere;
py_compile all three; heldout uses the same builder (no drift between the two
evaluators).

## Validation
```bash
python -m py_compile cloud/sim_gap_check.py cloud/heldout_eval.py cloud/pick_checkpoint.py
python -m pytest tests/test_eval_harness.py -q
```
GPU later: rerun v11 winner under harness v2 to re-baseline (numbers will move —
that is the point; log it).
