# Safety-review residuals closed (2026-07-03)

The three residuals left open by the first remediation pass (docs/safety_remediation.md)
are now closed. Full suite: **90 passed, 8 skipped** (model-gated), + the model-gated DR
effect test passes where the G1 model is present.

## #23/#24 — sim-runs endpoint now authenticates the verdict (the important one)
Before: `POST /api/dances/{id}/sim-runs` trusted a bare `{"passed": true}` — anyone could
walk a dance to show-ready. Now the caller submits a **signed `sim_exam/v1` verdict**
(inline `verdict` or `verdict_path`); the server (`shows.record_sim_run_from_verdict`):
1. verifies the HMAC signature (`pipeline/exam_verdict.signature_valid`),
2. requires `policy_sha256`/`motion_sha256` to match the dance's registered files,
3. **derives** pass from phase contents (never the self-declared string).
Only then is the clean streak credited, and the exam-passed policy sha is **pinned** onto
`dance.policy_sha256`. `promote()` to show-ready re-hashes the policy on disk and refuses
on mismatch — a post-exam policy swap can no longer reach the robot (#24/#25/#27).
Contract §1 updated. Tests: `test_signed_pass_verdict_credits_streak_and_pins_sha`,
`test_unsigned_verdict_is_rejected`, `test_verdict_for_a_different_policy_is_rejected`,
`test_promote_refuses_after_policy_swapped`.

## #28 — per-record locking
All dance/show mutators (`record_sim_run*`, `promote`, `complete_step`, `record_outcome`)
now reload the record fresh and mutate/save inside a per-record `flock` (`_record_lock`),
serializing across the web worker threads and the sim-exam CLI process. A concurrent pass
can no longer mask a failing run via a stale read-modify-write.
Contract change (documented): mutators now RETURN the fresh record — callers must use the
return value, not the object they passed in (server endpoints already did).
Test: `test_concurrent_runs_do_not_mask_a_failure` (6 interleaved threads + a barrier).

## #6 — exam domain randomization
The repeatability phase used identical deterministic seeds with only tiny joint jitter, so
"3× clean" was bit-identical and meaningless. Each repeat run now applies distinct DR
(`sim_exam.DR_RANGES`): ground friction, per-link mass, base pose/velocity offsets,
additive IMU/encoder observation noise, and one control tick of actuation latency, with
de-correlated seeds. The applied ranges are recorded in `verdict.repeatability.dr` for
audit. Nominal stays clean/deterministic as the canonical tracking check. Model + state are
snapshot-restored each run so perturbations never accumulate.
Tests: `test_repeatability_records_dr_and_distinct_seeds`, `test_dr_perturbs_state_and_restores_model` (model-gated).

## Residual-of-residuals / notes for the merge
- The HMAC key (`.secrets/exam_signing.key`) is a soft boundary on a single-user laptop
  (any process that can read `.secrets/` can re-sign). It defeats the actual finding
  (accidental/hand edits, fabrication) but is not a hard trust boundary vs. a hostile
  local process — acceptable for this deployment, flagged if multi-user ever matters.
- The app UI (static JS) still needs a small update to POST the verdict object/path
  instead of a bool; the API contract is the source of truth and is now enforced
  server-side regardless of the UI.
