# Robot Day Runbook — first hardware session (Phase 6 entry)

The complete procedure for the first time a trained dance touches the real G1.
Operator: Alois. Everything here assumes the sim exam passed (verdict JSON in the
deploy bundle) — if it didn't, there is nothing to do on the robot.

**Golden rules**
- Hardware e-stop in hand whenever motors are armed. It beats every script.
- One change at a time; after every step, state matches the runbook or you abort.
- Abort is free: `deploy/kill_now.sh`, or the e-stop, or robot power. Never
  hesitate because "we're almost done".
- The session env var `CONFIRMED_BY_HUMAN=alois` is set by YOU, once, at step 0 —
  scripts refuse to act without it, and Claude never sets it.

## Step 0 — session setup (robot OFF)
```
export CONFIRMED_BY_HUMAN=alois        # you, not Claude
cd ~/g1-dance/deploy
nmcli c up robot-lan                   # wired robot LAN (laptop = 192.168.123.2)
```
Gantry rigged and load-tested with the robot powered off. Straps rated ≥ 60 kg,
feet clearance ~5 cm, nothing within a 2.5 m radius. E-stop battery checked.

## Step 1 — health check (robot ON, motors in damping, ~30 min, READ-ONLY)
- `ping -c2 192.168.123.164` then `ssh unitree@192.168.123.164` (ROS prompt: 1).
- Run the LowState audit (all 29 motors reporting, temperatures nominal,
  firmware versions recorded into `docs/hardware_audit.md`).
- Check Inspire hand services respond (.210/.211). Hands stay DISABLED for the
  first dance sessions — choreography is body-only until Phase 6 exit.
- ABORT IF: any motor silent/hot, firmware unexpectedly changed, IMU drifting.

## Step 2 — controller install (software only, no motion)
- `./01_pc2_install.sh` (dry-run) — read every line it prints.
- `./01_pc2_install.sh --yes-install`
- 2b if the image pull fails (robot LAN has no internet): on the laptop
  `docker save qiayuanl/unitree:jazzy | ssh unitree@192.168.123.164 docker load`.
- ABORT IF: docker missing/broken on PC2 — do not improvise system changes on
  the robot; end the session, solve it offline.

## Step 3 — pin the controller launch line (the one on-robot unknown)
- Read `~/g1dance/motion_tracking_controller/README` ON PC2; identify the exact
  launch command and its damping/start-mode semantics.
- Edit `bundles/<dance>/start_controller_damping_hold.sh` accordingly, then
  `touch bundles/<dance>/LAUNCH_LINE_VERIFIED` and re-run
  `./02_push_bundle.sh --dance <dance> --yes-push`.
- The contract may not be weakened: load policy → HOLD DAMPING → operator arms.
  If the controller cannot hold damping on start, STOP — session over, redesign.

## Step 4 — bundle push
- `./02_push_bundle.sh --dance <dance>` (dry-run, read it) then `--yes-push`.
- Integrity check runs automatically (sha256 vs manifest, exam verdict re-check).

## Step 5 — gantry test, damping only
- Robot hanging, feet off ground. Motors armed via remote into damping.
- `./10_gantry_test.sh --dance <dance> --gantry-confirmed --estop-confirmed`
  (dry-run first, always), then add `--arm`, type the confirmation phrase.
- Container starts, policy loads, robot HOLDS in damping. Watch logs 2 minutes.
- ABORT IF: any joint twitches in damping hold, logs show NaN/joint errors,
  control frequency ≠ 50 Hz.

## Step 6 — gantry playback (first actual motion)
- Operator arms playback per the controller's documented start sequence.
- Feet stay off the ground; the dance plays in the air. Watch: joint smoothness,
  no oscillation, no limit slamming; motor temps after one run.
- Record: `docs/gantry_test_log.md` — date, dance, bundle shas, observations.
- ABORT IF: oscillation, thermal warnings, tracking visibly wrong. Kill, then
  diagnose OFFLINE (exam replay with logged states) — never live-debug hanging.
- Repeat 3 clean runs before proceeding.

## Step 7 — ground, harnessed
- Lower to ground, keep slack safety line + e-stop. Clear 2.5 m radius.
- Damping hold → stand → playback. One spotter (Alois) at e-stop, nobody else.
- 3 clean runs = Phase 6 gantry/ground milestone complete.
- ONLY THEN: gentle push tests (hand pressure on torso, mid-dance), matching
  what the sim exam certified — sim said it recovers; verify reality agrees.

## Step 8 — wrap
- `deploy/kill_now.sh`, robot to damping, power down per ~/robot/RUNBOOK.md.
- Battery/temperature notes + endurance measurement (full-dance battery %) into
  `docs/gantry_test_log.md` — feeds the 2–3 min show-endurance envelope.
- Unset the session var: `unset CONFIRMED_BY_HUMAN`.

## Abort ladder (memorize before step 5)
1. Hardware e-stop (hand) — always first choice while motors are armed.
2. `deploy/kill_now.sh` — kills the controller container, robot falls to damping.
3. Remote damping command (robot remote, per ~/robot/RUNBOOK.md re-arm notes).
4. Robot power switch.
