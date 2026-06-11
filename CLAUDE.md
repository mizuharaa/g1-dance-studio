# G1 Dance Pipeline

**Before doing anything in this project, read `PROJECT_STATE.md`** — it is the single
source of truth (mission, decisions, current phase, next actions, resume protocol).
This is a multi-day project across laptop reboots; sessions are stateless, that file is not.

Rules:
- Update `PROJECT_STATE.md` (status, decision log, next actions) after every meaningful step.
- Commit early and often to the local git repo; messages describe pipeline progress.
- Never modify `~/robot/` (working teleop setup) — read-only reference.
- Long GPU jobs run in the cloud and outlive reboots: record job IDs in `logs/jobs.md`.
- Robot safety: never send low-level commands to the real robot unless the motion passed
  MuJoCo verification AND the user confirmed the robot is secured (gantry/clear space,
  e-stop in hand). The deploy stage must always require an explicit human confirmation.
