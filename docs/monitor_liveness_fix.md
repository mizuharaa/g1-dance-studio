# Monitor liveness + dance dedupe fix (2026-07-04)

## BUG 1 — dead jobs shown as "Active Training"
Root cause: box tmux sessions are named `job-<name>` but monitor.parse_gather
checked `name in sessions` (no prefix) — so tmux liveness NEVER matched, and a job
was called "running" only via `status.json state=="running"`, which a SIGKILL'd job
never updates. A retired job's stale log therefore showed as the active training card
forever.
Fix (pipeline/monitor.py): liveness = `job-<name>` session present. Each job now
carries an explicit `live` bool and an honest `state` (running / done / failed /
stopped[killed] / finished). UI (ui/static/app.js): "Active Training" picks a LIVE
job (or "Idle"); Training Progress dims finished jobs and labels their state; the
"Live" badge only shows when something is actually training.

## BUG 2 — duplicate dances (Thriller vs thriller, test-segment x2)
Root cause: seeding/find matched by exact name, so the policy-attached "Thriller"
show-cut and the seeded "thriller" draft coexisted; merges across worktrees also
duplicated test-segment.
Fix (pipeline/shows.py): `find_dance` matches by normalized name (case-insensitive,
whitespace-collapsed); `dedupe_dances()` (run at startup) keeps the richest record
(policy > sim_exam > verified > vet/preview/motion > longer), back-fills its missing
fields from duplicates (never loses an attached policy), and removes the rest.

Tests: tests/test_monitor.py (+3 liveness), tests/test_shows_dedupe.py (5). Suite green.
