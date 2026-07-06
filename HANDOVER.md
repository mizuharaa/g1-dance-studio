# G1 Dance — Handover / Resume (2026-07-06, post-promotion)

**Read this first, then `PROJECT_STATE.md` for the full day-by-day log.** This file is the
fast path to resume; PROJECT_STATE is the source of truth.

---

## ONE-LINE STATUS
**Thriller is SHOW-READY on the s2r-b policy — validated end to end.** The sim2real retrain
closed the gap: 3x full 51.8 s ground dances on hardware (tethered, trained gains, no boost,
ankle 4.5-6.5 Nm mean, temps flat), 3x signed held-out exams at 100%/100% (1536/1536 episodes),
promoted through the app's guarded machinery (sha-pinned), deploy bundle rebuilt+authorized.

## WHERE THINGS ARE
- Show policy (canonical): data/policies/thriller/ (s2r-b; STAGED.txt = provenance chain).
  Fallbacks: thriller_a2_fallback/ (prior HW-proven), thriller_s2r_fallback/ (attempt-1).
  Checkpoints pulled to data/checkpoints/ (box is safe to DELETE — console-only, user click).
- Dance record 20260704-18f65bbd: show-ready, policy sha pinned, motion_csv = the DEPLOYABLE
  data/policies/thriller/thriller_deploy.csv.
- Deploy runtime fixes all hardware-validated: yaw re-anchor (offsets -87..+88 deg observed,
  all handled), torso anchor, per-joint action caps (legs 10 / arms x1.6), telemetry every run.
- Full evidence + history: PROJECT_STATE.md 2026-07-05..06 entries;
  docs/first_principles_audit.md (the audit that redirected the project).

## REMAINING TO PAID-SHOW GRADE (needs the user for robot steps)
1. Slack-tether -> free ground runs (staged, same protocol as 2026-07-06 session).
2. Music-synced rehearsal (audio machinery from show-production work; 1.5 s lead-in rule).
3. End-of-run hold-then-handoff — kill the ~1-1.5 m onboard catch-step at run end (backlog).
4. 2-3 min IN-PLACE choreography for real show pieces (filming guidance already recorded).
5. Arm-over-onboard runtime (pipeline/arm_dance_runtime.py) built+tested, unproven on HW —
   still the low-risk fallback/second act; first 5 s probe pending.

## KEY FACTS / INFRA
- **GPU box** (alive): `root@103.245.250.152:46936`, key `~/g1-dance/.secrets/greennode_ssh_key`,
  work dir on box `/workspace/notebook-data` (envs/mjlab, repos/mjlab, cloud/ scripts,
  motions/thriller_deploy.npz, run_job.sh for detached tmux jobs).
- **Training gains == deploy gains** (verified: ankle kp 29, knee 99, hip 40) — NOT a gain bug.
- Robot model + gains config on box: `repos/mjlab/.../unitree_g1/g1_constants.py`.
- **Proven gantry policy**: `data/policies/thriller/` (policy.onnx, policy_meta.json,
  thriller_deploy.npz) — 100 % in sim, full 160-dim obs.
- **Deploy runtime**: `pipeline/deploy_runtime.py`. Modes: `read` (safe, default),
  `move-to-default`, `run`, `stand-hold`, `ground-run`, `ground-run-odom`, `ground-run-legodom`.
- **Leg odometry + fused estimator + gravity_comp**: `pipeline/leg_odometry.py` (all offline-
  validated; leg-odom is the deploy estimator that works, fusion/FF shelved as not-the-fix).
- Env `tv` = robot runtime (unitree_sdk2py, onnxruntime, mujoco). Env `g1dance` = pipeline/tests.

## ROBOT SAFETY (non-negotiable — a 35 kg robot, no torque-cut e-stop)
- NEVER command motion without: human present, tether rigged to catch, **damping remote in hand**.
- All motion modes need `--i-will-watch-the-robot` AND env `CONFIRMED_BY_HUMAN=alois`.
- Robot iface `enp0s31f6`; robot IP `192.168.123.164`.
- **Motion-service gotcha**: releasing it for low-level control freezes `rt/odommodestate` AND
  can strand the remote — the runtime now auto-restores `SelectMode("ai")` on exit. If the remote
  won't pair, run `SelectMode("ai")` from the laptop or reboot the robot.
- **Signal the PYTHON pid, not the bash wrapper**, to stop a run (else the child orphans and holds
  the robot energized — happened twice). Use `pgrep -f "python.*deploy_runtime"`.
- **Thermal**: read `motor_state[i].temperature`; warn ~80 °C, fault ~90 °C. Monitor drains DDS to
  the LATEST msg (a stale-backlog bug once let a motor hit 80 °C blind — fixed).

## HOW TO RESUME IN A FRESH SESSION
Start the new session in `~/g1-dance` and paste:

> Resuming the G1 dance project. Read `HANDOVER.md` then `PROJECT_STATE.md` (2026-07-05..06
> entries). Thriller is SHOW-READY on the s2r-b policy — hardware-validated 3x full ground
> dances, promoted through the guarded exam machinery. Work the "REMAINING TO PAID-SHOW
> GRADE" list in HANDOVER; robot steps only with me present, damping remote in hand.
> Measurement discipline is in CLAUDE.md: no DECISIVE claims without an independent
> cross-check; commit every measurement script + raw output.

That's enough for a fresh Claude to pick up exactly here.
