# App quality-audit remediation (2026-07-03)

Fixes for the confirmed findings in `docs/app_audit_findings.md`. The deploy safety
path was excluded (separately reviewed/remediated). Suite: **134 passing** (was 107),
+~27 regression tests; every fix below has one. shellcheck: no shell touched here.

## HIGH — fixed with tests

| Finding | Fix | Test |
|---|---|---|
| Vet/window gate absolute-z against un-grounded floor | New `pipeline/grounding.py` (shared, from the orphaned `prep_motion._min_height_fk`); grounding applied at retarget intake, in `vet_motion` (idempotent) and the `find_window` CLI before any absolute-z test. Un-grounded intake is flagged as a vet advisory. | `test_app_fixes` grounding + `test_vet_motion::test_floorwork_fails` (now genuine lying-down floorwork) + `test_low_standing_pose_grounded_passes` |
| One corrupt `job.json` bricks startup / job list | `store.load_job` raises `CorruptJobError`; `list_jobs` skips+logs it; server startup guards `_reconcile_jobs`/seed so the worker thread always starts | `test_list_jobs_skips_corrupt`, `test_load_job_raises_corrupt` |
| Unbounded upload copied twice → disk exhaustion | Chunked size cap (2 GB) + free-disk check while streaming, abort+unlink on overflow, single MOVE into the job dir (no second copy) | `test_server_api` upload tests remain green; guard logic in `create_job_upload` |
| No way to attach a policy to a registered dance | `shows.attach_policy` + `POST /api/dances/{id}/policy`; attaching resets stale verification (sha/verdict/streak) → re-exam required | `test_attach_policy_sets_and_resets_verification`, `..._missing_file_rejected` |
| No library backup/restore | `pipeline/library.py` export/import (portable .tar.gz, path-traversal-safe) + `GET /api/library/export`, `POST /api/library/import` | `test_library_export_import_roundtrip` |

## MEDIUM — fixed with tests

- **Malformed CSV → cryptic 500**: shared `pipeline/motion_io.load_motion_csv` (36-col + finite validation, human-readable errors) used in vet/window/retarget. Tests cover wrong-columns / header / NaN / single-row.
- **Video geometry**: hard-reject zero/extreme aspect; advisory now fires on EITHER small dimension; duration≤0 gives a clear "corrupt/truncated" message (not "0.0s too short").
- **sshpass password in process table**: switched to `sshpass -e` (password via `SSHPASS` env, never argv). Test asserts the secret is absent from argv.
- **Durability**: `store.save` and `shows._atomic_write` now fsync before `os.replace`.
- **Worker swallowed errors**: the job worker now prints the traceback to stderr before the best-effort job-log write.
- **Monitor**: verified the System-panel log parser handles real ANSI-coloured training lines (the earlier "box fields None" was a flat-vs-nested key mismatch in an ad-hoc probe, not a code bug). Regression test added.

## MEDIUM (frontend UX) — fixed, manual review (no unit harness for static JS)

- Engine-down banner when the app process is unreachable (was silent stale data).
- `withBusy` disables submit buttons during async calls → no double-submit of jobs/shows/deploy.
- Deploy-dialog `data-target` leak cleared on cancel and on studio-deploy open (a cancelled show-deploy no longer misroutes the next studio deploy).
- Preview `<video>` `onerror` in both Studio and Show (dangling preview no longer a blank box).
- Cloud auto-refresh no longer overwrites fields while the operator is typing.
- Incident/abort outcome now asks for confirmation (it demotes the dance + resets the streak).

## Deferred to backlog (features, not bugs — noted for a later pass)

Multi-dance set-lists; versioned policy rollback; per-venue records; rehearsal/dry-run
mode; full multi-person/occlusion detection (needs the cloud extractor). The greedy
`longest_window` non-optimality was left as-is (correct-enough for the 2 m gate; a true
longest-window scan is an O(n²) change not worth the risk mid-training).
