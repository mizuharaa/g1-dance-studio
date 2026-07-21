# B2 — Fix F3 (part 2): repair the live obs-layout verifier

**Finding:** REPORT §F3 + PINNED_MJLAB_EVIDENCE §configuration/history.
`cloud/verify_obs_layout.py` (the box gate that must prove the deploy
HistoryStacker layout equals the live mjlab actor obs) is nonfunctional:
attribute access on dict-backed `cfg.commands`/`cfg.observations`, and it treats
`CircularBuffer` lag-lookup as a `[history,dim]` array. It is also never called.

**Owned files:** `cloud/verify_obs_layout.py`, `tests/test_verify_obs_layout.py`
(new, CPU parts). The launcher invocation is Lane A's (`run_attempt9.sh`) — the
CLI contract in CONVENTIONS §3.5 is FROZEN: flags and exit codes must not change.

## Spec
1. Fix the config API: `cfg.commands["motion"].motion_file = ...`,
   `cfg.observations["actor"]` (dict indexing per the pinned wheel
   `manager_based_rl_env.py:121`).
2. Fix the buffer read: mjlab `CircularBuffer.buffer` returns
   `[batch, history, dim]` in chronological order (pinned
   `circular_buffer.py:162-175`) — use `buf = circular_buffer.buffer[0]`
   (env 0 → `[history, dim]`), oldest→newest. Rebuild the expected flat vector
   term-major (terms in `om.active_terms["actor"]` order, each term's history
   oldest→newest) and compare to the env's flat actor obs for env 0.
3. Keep the FRAME-MAJOR diagnostic on mismatch. Exit codes: 0 PASS / 1 FAIL /
   3 API-unavailable (unchanged).
4. Guard every mjlab-internal access with a try that exits 3 with the exact
   attribute path that failed (so an mjlab upgrade degrades loudly, not wrongly).
5. CPU tests: factor the "rebuild expected vector from {term: [hist,dim] arrays}
   in term order" into a pure function; test it against
   `pipeline.deploy_runtime.HistoryStacker` output for the same synthetic data —
   the two implementations must agree byte-for-byte (this pins deploy == verifier
   without a GPU). Full env run remains box-only.

## Acceptance
Pure-function equivalence test green (verifier expectation == HistoryStacker
bytes); py_compile; CLI shape unchanged; failure modes exit 3 (not crash);
README coordination note added if ANY CLI change becomes necessary (default: none).
