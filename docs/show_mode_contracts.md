# Show-mode & deploy JSON contracts

Interfaces between the pipeline tools, the app's show mode, and the deploy kit.
Merged 2026-07-03 from the show-mode and deploy-kit tracks. Producers/consumers of
the schema'd payloads must version-check the `schema` field.

## 1. Repeatability: sim-exam tool → app

After EACH closed-loop exam run of a dance's policy, the sim-exam tool POSTs:

```
POST /api/dances/{dance_id}/sim-runs
Content-Type: application/json
{
  "passed":  true,                  # REQUIRED bool — this run was fully clean
  "exam_id": "exam-20260704-01",    # optional string — groups runs of one exam batch
  "metrics": {                      # optional object — whatever the exam measured
    "tracking_err_mean_rad": 0.041,
    "survived_s": 44.3,
    "max_root_excursion_m": 0.91,
    "pushes_survived": 5,
    "pushes_total": 5
  },
  "video": "data/exams/thriller/run3.mp4"   # optional project-relative path
}
```

Server behavior (pipeline/shows.py:record_sim_run):
- `passed=true`  → `repeatability.consecutive_clean += 1`, dance `sim_exam.verdict = "pass"`,
  status `draft → sim-verified` (if draft).
- `passed=false` → `consecutive_clean = 0`, `sim_exam.verdict = "fail"`,
  status `show-ready → sim-verified` (demotion — "works every time" broken).
- History kept (newest first, capped at 20). Response: the updated dance JSON.

Show-ready promotion is HUMAN-ONLY: `POST /api/dances/{id}/promote {"status": "show-ready"}`
— refused unless latest exam passed AND `consecutive_clean >= REPEATABILITY_TARGET`.
REPEATABILITY_TARGET is 3 as implemented; the deploy-kit track recommends raising to 5
before real client shows — OPEN DECISION, revisit at Phase 8 hardening.

## 2. Dance registration: training pipeline → app

When a policy lands, the training side registers/updates the dance:

```
POST /api/dances            # or update fields of an existing one via re-POST (name-unique)
{
  "name": "thriller",
  "duration_s": 44.3,
  "motion_csv": "data/jobs/<id>/retarget/motion.csv",
  "policy_path": "data/policies/thriller_a1.onnx",
  "preview": "/previews/job-<id>.mp4",
  "source_job": "<job_id>",
  "notes": "attempt 1, converged @ ..."
}
```

(`POST /api/dances` refuses duplicate names; to attach a policy to an existing
dance, extend with a PATCH later — for now update dance.json via pipeline code
or re-register under a new name per attempt.)

## 3. Show record (produced by the app, consumed by humans/Phase-6 tooling)

`data/shows/<show_id>/show.json`:

```
{
  "id": "20260704-193200-ab12cd",
  "dance_id": "...", "dance_name": "thriller",
  "operator": "Alois",
  "created_at": 1751628720.0,
  "steps": {
    "robot_health": {"at": ..., "confirmed": true},
    "space_clear":  {"at": ..., "confirmed": true},
    "battery":      {"at": ..., "value": 87.0},
    "estop":        {"at": ..., "confirmed": true},
    "venue_ack":    {"at": ..., "confirmed": true}
  },
  "deploy":  {"requested_at": ..., "note": "placeholder — nothing was sent to the robot"},
  "outcome": {"result": "clean" | "aborted" | "incident", "notes": "...", "at": ...},
  "closed": true
}
```

Checklist steps are enforced IN ORDER server-side; the deploy endpoint refuses
until all steps exist, then still requires `{"confirm_phrase": "DEPLOY"}` and
remains record-only. Phase-6 hardware tooling must treat `deploy.requested_at`
as an authorization *record*, never as a trigger.

## 4. sim_exam/v1 — sim2sim exam verdict

Producer: `pipeline/sim_exam.py` (Stage-4 gate). Consumers: show-mode dance library
(readiness badge; per-run results also POSTed per §1) and `deploy/gen_config.py`
(hard gate: no deploy bundle without a passing verdict).

```json
{
  "schema": "sim_exam/v1",
  "dance": "thriller",
  "policy": "data/policies/thriller/policy.onnx",
  "policy_sha256": "16-hex-chars or null (stub)",
  "motion_csv": "data/motions/thriller/thriller_g1_30fps.csv",
  "motion_sha256": "16-hex-chars",
  "at": "2026-07-03T18:00:00+00:00",
  "control_hz": 50.0,
  "nominal": {
    "pass": true,
    "survived_s": 44.3, "duration_s": 44.3,
    "excursion_m": 0.91,
    "mean_anchor_pos_err_m": 0.06, "max_anchor_pos_err_m": 0.14,
    "mean_joint_err_rad": 0.05
  },
  "push": {
    "num_pushes": 4, "recovered": 4, "recovery_rate": 1.0,
    "force_n": 250.0, "fell": false, "pass": true
  },
  "repeatability": {
    "runs": 5, "clean": 5, "consecutive_clean": 5, "pass": true,
    "per_run": [{"seed": 100, "pass": true, "survived_s": 44.3}]
  },
  "verdict": "pass",
  "video": "data/exports/exam_thriller.mp4 or null",
  "wall_s": 210.0
}
```

Semantics:
- `nominal.pass` = survived the whole motion AND excursion ≤ 1.5 m.
- `push.pass` = no fall AND recovery_rate ≥ 0.8 (recovery = anchor error < 0.25 m
  within 2 s after each 0.1 s, 250 N default horizontal shove).
- `repeatability.pass` = all runs clean (each with ±0.02 rad initial joint jitter).
- `verdict` = "pass" only if every phase run passed. `push`/`repeatability` may be
  null when a phase was skipped — treat null as NOT passing for show-readiness.
- A dance's show-ready badge requires: vet PASS + `verdict == "pass"` with BOTH
  phases present + the §1 repeatability threshold.

## 5. deploy_bundle/v1 — generated robot-day bundle manifest

Producer: `deploy/gen_config.py`. Consumers: show-mode deploy screen (display-only)
and `deploy/02_push_bundle.sh`.

```json
{
  "schema": "deploy_bundle/v1",
  "dance": "thriller",
  "created_at": "iso8601",
  "policy": {"file": "policy.onnx", "sha256": "..."},
  "motion": {"file": "motion.csv", "sha256": "...", "duration_s": 44.3},
  "exam": {"file": "exam_verdict.json", "verdict": "pass"},
  "controller": {"image": "qiayuanl/unitree:jazzy", "notes": "see deploy/README.md"},
  "target": {"pc2": "192.168.123.164", "user": "unitree"}
}
```
