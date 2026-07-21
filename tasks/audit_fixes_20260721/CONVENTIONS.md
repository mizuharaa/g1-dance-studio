# CODEBASE REGULATIONS — audit-fix wave (2026-07-21)

Binding for BOTH lanes and every subagent. A change that violates these is
rejected at merge regardless of whether it "works".

## 0. Ground rules
- Repo: `/home/alois/g1-dance`. Baseline commit: `c93b7fe` (external audit landed).
- **CPU-only wave.** Nothing here starts GPU training, touches the boxes, the
  robot, `~/robot/`, or `.secrets/`. Fail-closed is the default posture.
- Audit evidence being implemented: `experiments/external_audit_20260721/REPORT.md`
  (findings F1–F8). Each task MD cites its finding; the report is the spec of
  record when a task MD is ambiguous.
- Conda env: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate g1dance`.

## 1. File ownership (HARD boundaries — do not cross)
**Lane A (this session / Claude #1):**
`cloud/sim2real_task*.py`, `cloud/sim_gap_check.py`, `cloud/heldout_eval.py`,
`cloud/pick_checkpoint.py`, `cloud/run_attempt9.sh`, `pipeline/grounding.py`,
`tools/build_motion_bundle.py` (new), `data/motions/**`,
`tests/test_cloud_audio.py`, `tests/test_cloud_stages.py`, `tests/test_grounding*`,
`tests/test_eval_harness*`, `tests/test_effort_scope*`.

**Lane B (second agent):**
`pipeline/deploy_runtime.py`, `pipeline/publish_policy.py`, `pipeline/show_runner.py`,
`pipeline/shows.py`, `pipeline/preshow.py`, `pipeline/exam_verdict.py`,
`pipeline/mjlab_verify.py`, `cloud/export_ckpt_onnx.py`, `cloud/export_policy.py`,
`cloud/verify_obs_layout.py`, `ui/server.py`, `ui/frontend/**`,
`tools/sim_sandbox.py`, `tools/sim_studio.py`, `tests/test_free_show.py`,
`tests/test_show_run.py`, `tests/test_video_probe.py`, `tests/test_bundle_*`,
`tests/test_verdict_*`.

**FROZEN shared contract (neither lane edits without a coordination note in
README.md):** `pipeline/artifacts.py` + `tests/test_artifacts.py`.

**Everything not listed:** untouched this wave. If a task seems to need a file
the other lane owns, STOP and write the need in README.md § Coordination log —
do not edit it.

## 2. Deploy-safety invariants (Lane B especially)
- The damp spine in `deploy_runtime.py` is UNTOUCHABLE: `_damp*`,
  `_install_damp_on_signals`, `DampingWatchdog` usage, the `except BaseException
  → damp` pattern, `_send_cmd` semantics, guard call sites. Bundle validation is
  ADDED BEFORE `make_dds`/`_release_motion_service`, never inside the 50 Hz loop.
- New validation must REFUSE (raise SystemExit with a reason) — never warn-and-
  continue — and must run before any human-confirmation prompt so consent is to a
  validated artifact.
- No new network calls, threads, or sleeps inside the control loop.

## 3. Shared schemas (the API between the lanes)
### 3.1 Bundle manifest — `g1.bundle/1` (via `pipeline/artifacts.py` ONLY)
File name: `bundle.json`, colocated with the artifact set it describes.
Sections (a producer fills only what it owns; verify tolerates absent sections):
```json
{
  "schema": "g1.bundle/1", "bundle_id": "<sha256>", "created_at": "...",
  "motion": {
    "source_csv":  {"path": "...", "sha256": "..."},
    "final_csv":   {"path": "...", "sha256": "..."},
    "scorecard":   {"path": "...", "sha256": "..."},
    "tempo_npz":   {"060": {"path": "...", "sha256": "..."}, "075": {}, "090": {}, "100": {}},
    "grounding":   {"flight_aware": true, "params": {}}
  },
  "model":  {"mjlab_wheel_sha256": "...", "task_id": "...", "task_module": "...",
             "config_env": {"G1_...": "..."}},
  "policy": {"onnx": {"path": "...", "sha256": "..."}, "meta": {"path": "...", "sha256": "..."},
             "checkpoint": "...", "obs_per_frame": 154, "history_length": 5,
             "flatten_layout": "term-major-oldest-first",
             "requires_ground_contact": true},
  "eval":   {"campaign_ids": ["<uuid4>"]}
}
```
All hashes: full lowercase hex SHA-256 via `artifacts.sha256_file`. Path entries
are always the `{"path","sha256"}` shape so `artifacts.verify_manifest` walks them.
Lane A produces `motion` (+`model.config_env` for training inputs); Lane B
produces `policy` and consumes the whole file at export/publish/show time.

### 3.2 policy_meta v2 (Lane B) — additive fields on the existing meta JSON
`obs_per_frame` (int), `history_length` (int),
`flatten_layout: "term-major-oldest-first"`, real `onnx_inputs`,
`requires_ground_contact` (bool), `actor_obs_terms_in_order` matching the LIVE
cfg (not copied from another policy). Existing consumers read old fields — keep
every existing field present and correct.

### 3.3 verdict v2 (Lane B) — additive
`eval_id` (uuid4, REQUIRED), `created_at`,
`disturbance: {"kind": "delta_velocity", "delta_v_mps": ..., "axes": ..., "seed": ...}`
(replaces the fictitious `force_n`; keep `force_n` absent, do not rename in old
records), plus existing sha fields. `shows.record_*` must DEDUPE on `eval_id`
and refuse verdicts without one. Old stored verdicts: left readable, but no
longer creditable toward repeatability (document in the task).

### 3.4 gap.json condition manifest (Lane A) — additive per condition
`realized: {"seed": int, "play_mode": bool, "startup_dr": bool,
"cmd_delay_steps": [lo,hi], "obs_delay_steps": [lo,hi], "noise": bool,
"push": bool}` — written from what was ACTUALLY configured, not intended.
Lane B's verdict/show code may read it but not write it.

### 3.5 CLI contracts (frozen so cross-lane wiring can't drift)
- `python cloud/verify_obs_layout.py --task <ID> --task-module <mod>
  --motion-file <npz> [--num-envs N]` → exit 0 PASS / 1 FAIL / 3 API-unavailable.
  Lane B fixes its internals; Lane A wires this exact invocation into
  `run_attempt9.sh` (box-only step; guarded so a laptop run skips with a note).
- `python tools/build_motion_bundle.py --source <csv> --out-dir <dir>` (Lane A,
  new): one command → cleaned+grounded+scored CSV + scorecard + bundle.json.

## 4. API endpoint rules (`ui/server.py`, Lane B)
- Paths `/api/<noun>` or `/api/<noun>/<id>/<verb>`; JSON bodies; errors via
  `HTTPException` (404 unknown id, 409 conflict/lock, 422 invalid); UI panel
  endpoints keep the never-raise/last-good pattern.
- Existing response shapes are append-only: add fields, never rename/remove.
- Any NEW endpoint or field must be listed in the task MD's "API delta" section
  and covered in `tests/test_server_api.py`-style tests.

## 5. Dance-store schema (`pipeline/shows.py`, Lane B)
Records are JSON on disk: additive fields only, with `.get()` defaults so old
records load. New fields this wave: `motion_sha256`, `meta_sha256`,
`npz_sha256`, `bundle_id` on Dance; `exam_ids` (list, unique) replacing blind
`consecutive_clean` increments (keep the counter, derive it from distinct ids).

## 6. Code style, tests, commits
- Match the FILE you edit: `cloud/sim2real_task*` uses 2-space indent (mjlab
  style); `pipeline/`, `tools/`, `ui/` use 4-space. Type hints where the file
  already has them. Comments: constraints/why only.
- Every task lands WITH tests in the lane's own `tests/` files. Full-suite goal:
  611+ collected, 0 failed at wave end (the 7 pre-existing reds are split:
  Lane A owns the 3 cloud-fixture ones, Lane B the 3 show + 1 video-probe).
- `git add <explicit paths>` ONLY (never `-A`/`.`): the tree may contain the
  other lane's WIP. Commit prefix `[laneA]`/`[laneB]`; one task = one commit;
  message cites the finding (e.g. `[laneB] F6: fail-closed show bundle`).
- Lane B works in a git WORKTREE to avoid index races:
  `git worktree add /home/alois/g1-dance-laneB -b laneB` (base `c93b7fe` or later
  main). Merge order at the end: laneA commits are already on main; laneB merges
  with `git merge laneB` — ownership disjointness makes conflicts structural
  errors, not merge work.
- Never modify or delete `experiments/external_audit_20260721/**` (evidence).

## 7. Definition of done (per task)
Spec implemented · owned tests green · full CPU suite not worse than baseline ·
README status board updated (checkbox + commit sha) · no cross-lane file edits ·
API/schema deltas documented in the task MD.
