# Audit-fix wave — coordination board (2026-07-21)

**Location (tell every agent):** `/home/alois/g1-dance/tasks/audit_fixes_20260721/`
Read `CONVENTIONS.md` FIRST — it is binding (file ownership, schemas, safety
invariants, commit rules). The findings being fixed are in
`experiments/external_audit_20260721/REPORT.md` (F1–F8). Baseline: `c93b7fe`.

## The split (CPU-only wave; each lane may spawn its own subagents)

| Lane | Who | Tasks (one MD each) |
|------|-----|---------------------|
| **A** | Claude session #1 (this one) | `A1_effort_scope_F1.md` · `A2_grounding_flight_F8.md` · `A3_v12_rebuild_manifest_F2.md` · `A4_eval_harness_F4.md` · `A5_cloud_tests_green.md` |
| **B** | Second agent | `B1_policy_manifest_F3.md` · `B2_obs_layout_verifier_F3.md` · `B3_show_failclosed_F6.md` · `B4_verdict_schema_F7.md` · `B5_scene_labels_F5.md` · `B6_show_tests_green.md` |

Dependencies: A3 needs A2 (grounding) first. B3 consumes the `g1.bundle/1`
manifest (already frozen in `pipeline/artifacts.py` — usable immediately).
Everything else is independent.

**Lane B setup:** work in a worktree —
`git worktree add /home/alois/g1-dance-laneB -b laneB` — then follow your task
MDs in order B1→B6 (B5 anytime). Commit per task with the `[laneB]` prefix,
explicit paths only.

## Status board (update the checkbox + commit sha when a task lands)

- [x] A1 effort scope (F1) — landed (see git log [laneA] F1)
- [x] A2 grounding flight (F8) — landed ([laneA] F8)
- [ ] A3 v12 rebuild + manifest (F2) — 
- [x] A4 eval harness (F4) — landed ([laneA] F4)
- [x] A5 cloud tests green — landed ([laneA] A5)
- [ ] B1 policy manifest/meta v2 (F3) — 
- [ ] B2 obs-layout verifier fix (F3) — 
- [ ] B3 show fail-closed (F6) — 
- [ ] B4 verdict v2 + dedupe (F7) — 
- [ ] B5 scene labels (F5) — 
- [ ] B6 show tests green — 

## Coordination log (append; needed-but-unowned files, schema questions)

- (laneA/A4) gap.json is now HARNESS v2: rows renamed (clean/dr_nominal/cmd_delay*/obs_delay*/
  dr_delay*_push), per-condition `realized` block, top-level harness_version/seeds. Lane B:
  mjlab_verify reads heldout keys `nominal`/`push` which are PRESERVED. Follow-up NEXT wave
  (nobody's file this wave): cloud/autopilot_s2r.py reads v1 `conditions['nominal']` and will
  fail loudly on a v2 gap.json.

- (laneA note) NEW FILE `cloud/effort_scope.py` — pure per-control scope resolver for F1,
  Lane A owned, mjlab-free so tests import it. Lane B: read-only.

- (open) Lane A wires the frozen `verify_obs_layout.py` CLI (CONVENTIONS §3.5)
  into `run_attempt9.sh` as a guarded box-only step — Lane B must not change the
  CLI shape without logging here first.

## Out of scope this wave
GPU training/box changes, robot anything, `~/robot/`, `.secrets/`,
`experiments/external_audit_20260721/**` (read-only evidence).
