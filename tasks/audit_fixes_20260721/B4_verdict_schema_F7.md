# B4 — Fix F7: honest disturbance schema + unique, deduped verdicts (HIGH)

**Finding:** REPORT §F7 (verified). `mjlab_verify.py` labels a 0.5 m/s velocity
overwrite as "875 N" via an arbitrary 20 ms divisor (5 ms would say 3500 N);
`exam_verdict` gates on force_n ≥ 150 — an arbitrary unit conversion, not push
strength. And `shows.record_sim_run_from_verdict` uses `verdict.get("at")`
(never set) as the exam id while incrementing `consecutive_clean` per
SUBMISSION → one signed verdict replayed 3× = "three clean runs" (crosscheck
reproduced credits [1,2,3]).

**Owned files:** `pipeline/mjlab_verify.py`, `pipeline/exam_verdict.py`,
`pipeline/shows.py`, `tests/test_verdict_v2.py` (new), existing verdict/show
tests. Signing mechanism (HMAC) unchanged.

## Spec
1. **Verdict v2 (CONVENTIONS §3.3):** `build_verdict` emits `eval_id` (uuid4),
   `created_at` (iso), and `disturbance: {kind: "delta_velocity",
   delta_v_mps: <sampled range>, axes, seed}`. DELETE `force_n` from new
   verdicts and delete the 875 N constant block; `exam_verdict`'s acceptance
   drops the ≥150 N force floor and instead requires
   `disturbance.kind == "delta_velocity"` with `delta_v_mps >= 0.5` (the actual
   mjlab event magnitude) — the schema tells the truth about what was tested.
   Keep signature coverage over the new fields (they're inside the signed body).
2. **Dedupe at record time:** `shows` stores `exam_ids: []` per dance;
   `record_sim_run_from_verdict` REFUSES a verdict without `eval_id` (clear
   error) and refuses a repeated `eval_id` (409-style error through the API).
   `consecutive_clean` is DERIVED from distinct clean eval_ids (keep the stored
   counter for old records but recompute on new submissions).
3. **Migration/honesty:** old stored verdicts stay readable; on load, a dance
   whose show-ready status rests on null exam ids gets `repeat_evidence:
   "legacy-unverified"` in its payload (UI may badge it). No silent re-crediting.
4. **Tests:** replay the audit's exploit — same signed verdict submitted 3× →
   ONE credit + explicit refusals; verdict without eval_id → refused; distinct
   ids → 3 credits; tampered disturbance block → signature check fails
   (existing mechanism); new schema round-trips sign/verify.

## Acceptance
The crosscheck exploit (`crosscheck.json` §signed-verdict replay) is dead:
re-running its scenario yields credits [1] + errors. No "Newton" claims remain
in new verdicts. Full show/verdict suites green (coordinate with B3/B6 —
same files).

## API delta
Verdict JSON v2 fields (§3.3); dance payload `exam_ids`, `repeat_evidence`.
