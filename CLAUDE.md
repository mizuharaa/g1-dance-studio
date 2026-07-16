# G1 Dance Pipeline — agent guide

**Resume protocol: read `docs/PROJECT_STATE.md` first.** It is the single source of
truth (mission, decision log, current phase, next actions). Sessions are stateless;
that file is not. Update it after every meaningful step. Cloud job state lives in
`logs/jobs.md`. New to the project? `docs/FIELD_GUIDE.txt` explains everything.

Rules:
- Commit early and often; messages describe pipeline progress.
- Never modify `~/robot/` (working teleop setup) — read-only reference.
- Long GPU jobs run in the cloud and outlive reboots: record them in `logs/jobs.md`.
  GreenNode bills creation→deletion — DELETE idle boxes, never leave them running.
- Robot safety: never send low-level commands unless the motion passed sim
  verification AND the user confirmed the robot is secured (gantry/clear space,
  damping remote in hand). Deploy always requires explicit human confirmation.
  This robot has NO hardware e-stop — only the remote's B-damp and the power switch.
- Measurement discipline: never label a finding decisive without an independent
  cross-check; commit every measurement script AND its raw output (`experiments/`
  or `data/telemetry/`) so load-bearing numbers have durable provenance.
- Pinned training env: `cloud/env_lock/requirements.lock.txt`
  (mjlab==1.5.0, mujoco-warp==3.10.0.1, warp-lang==1.14.0, torch cu128).
  `MUJOCO_GL=egl` only for render/verify, never during training.

Local-only working notes (gitignored, not part of the repo): `local/`.
