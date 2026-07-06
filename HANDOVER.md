# G1 Dance — Handover / Resume (2026-07-06, post-promotion)

**Read this first, then `PROJECT_STATE.md` for the full day-by-day log.** This file is the
fast path to resume; PROJECT_STATE is the source of truth.

---

## ONE-LINE STATUS
Thriller is SHOW-READY on s2r-b (hardware-proven 3x + music rehearsal done). All 5 dance
verdicts are in (v3a/b/d/v4 + v3c@9000) and ALL beat the arm baseline ~30%; a gap-gate CLI
bug voided the v3a/v3d/v4 gate columns — backfill + fluidity sweep ran 2026-07-06 evening;
backflip train-acro-1 is ACTUALLY training now (launcher TERM bug had silently killed it).

## IN FLIGHT ON THE BOX (survives everything; check first)
Jobs via `bash cloud/run_job.sh list` + `tmux ls` (export PATH=/workspace/notebook-data/bin:$PATH
TERM=xterm; status.json can be stale — trust tmux/pgrep: unique trainings =
`pgrep -af "agent.run-name train-" | grep -o "run-name train-[a-z][a-z0-9-]*" | sort -u`).
- VERDICTS IN (exports/thriller_v3{a,b,d}, thriller_v34): v3a arm RMS 9.72 deg (s2r-b baseline
  13.81); v3b 9.42 + ankle RMS 7.05 (coolest) + 100% survival BUT drift 1.20m (gate fail;
  deploy contract: needs ARM_GROUND_KP_SCALE=2.5); v3d 10.20 vs sharp baseline 15.18;
  v4 calm-legs 11.72 (mid ckpt) vs sharp 15.18.
- **GATE DATA CAVEAT (found+fixed 2026-07-06 ~16:10 UTC):** sim_gap_check.py's argv shim
  stripped the literal stock task after --task → gap_check crashed rc=2 for every stock-task
  variant. v3a/v3d/v4 "GATE_FAIL" = "no gap data", NOT a real fail (v3b's drift 1.20 is the
  only REAL gate number). Fixed (commit 7051956) + `gap-backfill` job re-ran the 4 missing
  checks; `fluidity-sweep`/`v3c-fluidity` jobs dumped traces + leg 2-10Hz band numbers and
  APPEND to each RESULT.txt. Check exports/*/​{last,mid}/gap_check.json + fluidity.json.
- v3c: training killed at ~iter 9170/10000 by an ops error (tmux server kill, 2026-07-06
  16:04 UTC — KILLED_NOTE.txt in run dir); model_9000.pt evaluated as final by v3c-autopilot3.
- train-acro-1 backflip: RUNNING since 16:05 UTC (10k iters) + acro-autopilot armed. The
  launcher's TERM=dumb had killed the original launch silently (fixed in
  cloud/launch_acro_when_free.sh).
- dance1-e2e (app job 20260706-172405-2eb6e0): train DONE (policy staged at
  data/policies/dance1_e2e), verify stage running sim-gap on the box via the app server.

## DECISION PROCEDURE (when v4's verdict is in)
Bar (cloud/V3_PROGRAM.md): gate v3 pass/near AND arm RMS < baseline AND 2-10Hz leg action band
<= 0.20 (s2r-b level; forensics docs/fluidity_forensics.md) AND leg amplitude ratio > 0.5
preferred. Compute leg-band via fluidity tool on sim traces for finalists + s2r-b baseline.
Winner -> stage as CANDIDATE dir under data/policies/ (NEVER overwrite data/policies/thriller/
— it is the sha-pinned show policy), 3x held-out exams + mjlab_verify signing, render sign-off,
then ONE tethered HW test w/ telemetry (target: hardware arm RMS well under the 13.2 deg
s2r-b baseline; leg wobble < 0.10 rad/s in 2-10Hz band).

## LOCAL STATE (check on resume)
- App server (the e2e job's orchestrator) runs nohup'd on port 8321
  (logs/app_server_e2e.log). Check `curl -s http://127.0.0.1:8321/api/jobs`; if dead,
  restart: `nohup ~/miniconda3/envs/g1dance/bin/python -m ui.server --port 8321 >
  logs/app_server_e2e.log 2>&1 &` — the runner resumes jobs cleanly.
- All background pollers from the old session are DEAD — re-arm (poll box RESULT.txt files).
- Laptop AUDIO WORKS (sof-arl.ri installed 2026-07-06). data/audio/thriller/music.wav IS THE
  PLACEHOLDER CLICK TRACK — when the user provides the real song:
  `tools/attach_music.py <file>` (converts/replaces/re-attaches; refuses click tracks).

## WAITING ON THE USER
1. Real Thriller audio file -> attach_music.
2. ~1h robot session (remote in hand): winning-policy tethered run + ARM_GROUND_KP_SCALE A/B
   (1.5/2.5) + robot-speaker validation + LED cue (docs/SHOW_AUDIO.md checklist).
3. Backflip HARDWARE decision — only after sim video + risk memo (docs/DYNAMIC_SKILLS.md —
   verify it exists; the acro agent died mid-docs; regenerate from cloud/dynamic_skills_task.py
   + exports/acro1 artifacts if missing).
4. New dance videos (docs/NEW_DANCE_PLAYBOOK.md — the app now does video -> sim-verified alone).

## STANDING ORDERS (user)
- Keep the GPU box busy ALWAYS (it bills regardless; box was never deleted — verify by SSH).
- Robot motion ONLY with the user present + damping remote in hand. No exceptions — held twice.
- Measurement discipline per CLAUDE.md (no DECISIVE without cross-check; commit raw outputs).

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

> Resuming the G1 dance project. Read `HANDOVER.md` top to bottom, then the 2026-07-05..06
> PROJECT_STATE.md entries. A GPU program is in flight on the box (survives sessions):
> collect the remaining verdicts per "IN FLIGHT ON THE BOX", then run the "DECISION
> PROCEDURE". Check "LOCAL STATE" first (app server on :8321, pollers need re-arming).
> Robot motion only with me present + damping remote in hand. Keep the GPU busy always.
> Measurement discipline per CLAUDE.md: no DECISIVE claims without an independent
> cross-check; commit every measurement script + raw output.

That's enough for a fresh Claude to pick up exactly here.
