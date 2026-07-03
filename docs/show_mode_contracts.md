# Show-mode JSON contracts

Contracts between the pipeline tools and the show-mode UI. Producers/consumers must
version-check the `schema` field. Defined by the deploy-kit track (2026-07-03);
show-mode track: adapt or propose changes in this file.

## sim_exam/v1 — sim2sim exam verdict

Producer: `pipeline/sim_exam.py` (Stage-4 gate). Consumer: show-mode dance library
(readiness badge + repeatability counter) and `deploy/gen_config.py` (hard gate:
no deploy bundle without a passing verdict).

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
- A dance's show-ready badge should require: vet PASS + `verdict == "pass"` with
  BOTH phases present + `repeatability.consecutive_clean >= N_SHOW` (UI-configurable,
  suggest 5).

## deploy_bundle/v1 — generated robot-day bundle manifest

Producer: `deploy/gen_config.py`. Consumer: show-mode deploy screen (display-only)
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
