# Show-mode JSON contracts

Interfaces between the app's show mode and the other pipeline tools.
Owner: ui/server.py + pipeline/shows.py. Written 2026-07-03.

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
— refused unless latest exam passed AND `consecutive_clean >= 3` (REPEATABILITY_TARGET).

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
