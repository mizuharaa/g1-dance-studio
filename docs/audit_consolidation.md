# Production-audit consolidation — app lane (2026-07-04)

Fixed the audit findings that live in the app layer (ui/, shows.py, library.py,
local_motion.py). Deploy-path findings (gen_config, mjlab_verify, 02_push, verdict
producer) are the concurrent deploy-kit agent's lane — NOT touched here.

## Fixed (with regression tests in tests/test_audit_consolidation.py)
- **HIGH / safety — MEC recenter** (`pipeline/stages/local_motion.py`): the exported
  window was recentered on frame 0, but the vet gate certifies the enclosing-circle
  (MEC) footprint radius — so the deployed robot could drift ~2x the certified radius
  and leave the dance area. Now recenters on `window_center(m,s,e)`; deployed
  excursion == certified radius. Tests: `test_mec_recenter_bounds_excursion_to_radius`,
  `test_local_motion_uses_window_center`.
- **HIGH / workflow — promote UI** (`ui/static/app.js`): the promote route existed but
  no frontend caller, so an operator could never make a dance show-ready. Added a
  "Promote to Show-Ready" button in the dance modal (shown when sim-verified), surfacing
  the server's gating error inline. Test: `test_promote_gated_on_clean_runs`.
- **HIGH / security — library import trust** (`pipeline/library.py`): import wrote the
  archive's dance.json verbatim, so a crafted backup could inject a fake show-ready
  dance and bypass the signed-verdict gate. Import now resets status→draft and clears
  sim_exam / policy_sha256 / consecutive_clean, forcing a fresh local signed sim-exam.
  Test: `test_library_import_resets_verification`.
- **MEDIUM / data-integrity — dedupe dangling ref** (`pipeline/shows.py`): dedupe
  back-filled a file path from a loser then rmtree'd the loser's dir. Now rescues any
  back-filled policy/motion file into the keeper's dir before deletion. Test:
  `test_dedupe_preserves_backfilled_motion_file`.

## Already covered (verified, locked with tests)
- **CRITICAL — post-show outcome capture**: the just-merged show-production runner
  already renders Clean/Aborted/Incident after deploy (runChecklist -> drawOutcome ->
  POST /api/shows/{id}/outcome), covering the single-show path too. Confirmed the
  incident->demote->reset-streak flow. Tests: `test_incident_outcome_demotes_show_ready`,
  `test_rehearsal_incident_does_not_demote`.
- **HIGH — verdict motion-sha seam**: the deploy-kit agent fixed the producer side
  (mjlab_verify now records motion_sha256 = deployable CSV digest + motion_npz_sha256
  for provenance). The consumer side (shows.record_sim_run_from_verdict) already hashes
  the deployable CSV — correct. No app-lane change needed.

## Deferred (documented, not half-fixed)
- **MEDIUM — venue max_excursion not wired end-to-end**: selecting a non-default venue
  never reaches windowing / vet / mjlab_verify / heldout (always 1.5 m). The current
  1.5 m default is CORRECT for the present 2 m area, so this is a latent false-capability,
  not an active bug. Full wiring spans local_motion + server (this lane) AND mjlab_verify
  + heldout_eval (deploy-kit lane), so it needs one coordinated cross-lane pass — a
  half-wire would be worse. Left as a single follow-up, not attempted overnight.

Full suite: 187 passed, 11 skipped (model-gated).
