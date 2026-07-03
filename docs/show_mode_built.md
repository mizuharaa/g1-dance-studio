# Show mode — build summary (worktree, 2026-07-03)

For main to integrate into PROJECT_STATE at merge. Built in an isolated
worktree; commits: a1281c0 (backend), b918f0d (UI + docs), + battery-validation
fix commit (see git log).

## What exists now

- **pipeline/shows.py** — persistence for the operator side, mirroring
  store.py's atomic-JSON pattern:
  - Dance library at `data/dances/<id>/dance.json`: name, duration, motion CSV,
    policy path, preview, embedded vet report, sim-exam verdict, notes,
    repeatability record. Status ladder `draft → sim-verified → show-ready`
    with guard rails (promotion to show-ready requires latest exam pass AND
    ≥3 consecutive clean sim runs; a failed run demotes and resets the streak).
  - Shows at `data/shows/<id>/show.json` + append-only `show-log.txt`: operator
    name, 5-step pre-show checklist (server-enforced order: robot_health,
    space_clear, battery %, estop, venue_ack), record-only deploy, outcome
    (clean/aborted/incident) closing the show.
  - Idempotent startup seeding: registers `test-segment` (from the committed
    CSV) and `thriller` (from the pipeline job, when present — will pick it up
    in the main repo at merge).
- **API (ui/server.py)**: GET/POST `/api/dances`, GET `/api/dances/{id}`,
  POST `/api/dances/{id}/promote`, POST `/api/dances/{id}/sim-runs`
  (repeatability contract), GET/POST `/api/shows`, GET `/api/shows/{id}`,
  POST `/api/shows/{id}/steps/{step}`, POST `/api/shows/{id}/deploy`
  (checklist-gated + typed-DEPLOY + record-only), POST `/api/shows/{id}/outcome`.
- **UI**: header Studio/Show mode switch (persisted in localStorage); show mode
  is big-type and sparse — dance cards with status + "N/3 clean" badges,
  preview player, operator-name + start, one-step-at-a-time checklist wizard
  with progress chips, deploy gate reusing the typed-DEPLOY dialog, outcome
  buttons, show history list. Studio polling pauses while in show mode.
- **docs/show_mode_contracts.md** — JSON contracts: sim-exam → `/sim-runs`
  payload, dance registration, show-record schema (Phase-6 tooling must treat
  `deploy.requested_at` as a record, never a trigger).
- **docs/OPERATOR_MANUAL.md** — plain-language manual for a non-technical
  operator (golden rules, step-by-step show flow, incident table).

## Verification done (headless, port 8799 + offscreen 8798)

- Full happy path: create show → 5 steps in order → deploy (typed) → outcome
  → closed; show-log.txt records every step with timestamps.
- Negative cases all refused correctly: missing operator; deploy before/mid
  checklist; out-of-order step; battery as bool (found + fixed a real bug:
  `float(True)==1.0`), battery out of 0–100; wrong deploy phrase; promote
  without exam; promote at 2/3 clean; non-bool `passed`; duplicate dance name.
- Repeatability lifecycle: 3 passes → promote succeeds; a fail demotes
  show-ready → sim-verified and resets streak to 0.
- Seeding idempotent; static assets 200; offscreen desktop smoke OK.
- Test data removed; seeding recreates library entries at next startup.

## Open items for main

- Thriller seeds fully (vet + preview + source job) only in the main repo where
  job `20260703-215617-3d5060` exists — verify after merge.
- No PATCH for attaching a policy to an existing dance yet — training side
  currently updates dance.json via pipeline code or registers per-attempt
  names (see contracts doc §2). Decide at integration.
- The studio deploy dialog is reused by show mode via a data-target attribute —
  if the studio panel gets reworked, keep the shared dialog contract.
