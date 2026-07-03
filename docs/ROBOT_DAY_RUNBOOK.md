# ROBOT DAY RUNBOOK — first hardware deployment of a trained G1 dance

Deliberately paranoid. This is the first time a learned policy touches the physical robot;
simulation success is necessary but NOT sufficient. Work top to bottom; never skip ahead.
One-page card: `docs/ROBOT_DAY_CHECKLIST.md`. Pre-flight: `./scripts/preflight_robot_day`.

## Hard safety facts (read first)
- **No torque-cut e-stop.** This tether-free G1 EDU has no hardware kill that cuts motor
  power. Your only stops are the **remote's B (damping) button** and the **power switch**.
  Keep the remote in your hand whenever motors are powered.
- **`deploy/kill_now.sh` stops the controller container** (SIGTERM→SIGKILL). It does NOT
  guarantee a safe posture — whether command-loss produces damping is **unverified until
  step 3a measures it**. Treat kill_now as secondary to the remote.
- Nothing in `deploy/` moves a motor without: `CONFIRMED_BY_HUMAN=alois` **and** explicit
  flags **and** (for the gantry step) a typed phrase at a real terminal.
- **sim2real gap known:** the policy consumes `base_lin_vel`, which the real robot can't
  measure directly — the onboard controller must provide it via a state estimator. Confirm
  this is handled before trusting balance (see `docs/sim2real_derisk.md` if present).

## Preconditions (all must hold)
- Dance is **show-ready**: held-out ≥99% survival + sim-verified (`preflight_robot_day` = GO).
- Gantry/harness rigged; robot can hang with feet ~5 cm off the ground; straps rated & locked.
- 2 m clear radius, hard flat floor; battery ≥ 50%; remote tested today; area clear of people.

---

## Step 0 — Pre-flight (laptop, robot OFF)
```
./scripts/preflight_robot_day --dance thriller
```
Must print **GO**. Resolve any NO-GO. Then: `export CONFIRMED_BY_HUMAN=alois` (this shell only).

## Step 1 — Power on + health check (robot ON, NOT deployed, ~30 min)
Power the robot on the gantry. Do NOT start any controller. Verify (robot-side ground truth):
- All **29 motors** present in LowState, zero fault flags; joint encoders sane at rest.
- Firmware versions **recorded** (freeze them; note in `logs/jobs.md`).
- Inspire hands service up (if used). Battery voltage/SOC reads correctly.
- Press remote **B** → confirm the robot damps (goes compliant). This is your primary stop; prove it now.

## Step 2 — Network + install + push (laptop on robot-lan)
```
ping 192.168.123.164                       # PC2 reachable
./deploy/01_pc2_install.sh --yes-install   # installs controller stack; starts nothing
./deploy/02_push_bundle.sh --dance thriller --yes-push
```
If the image pull fails (robot LAN has no internet): `docker save` on the laptop, `scp`, `docker load` on PC2 (see `deploy/README.md`).

## Step 3 — Verify launch line + damping (on PC2, still no motion)
- Open the controller README on PC2; confirm the exact `ros2 launch` entrypoint and that
  `start_mode:=damping` is real. Edit the bundle's `start_controller_damping_hold.sh` launch
  line to match, then `touch <bundle>/LAUNCH_LINE_VERIFIED` and re-push. Until this exists,
  `10_gantry_test.sh` refuses to start.

## Step 3a — KILL / COMMAND-LOSS TEST (feet OFF ground) ★ most important measurement ★
Start the controller in damping hold (step 4), then, with feet off the ground:
1. Run `deploy/kill_now.sh` and **watch**: does the robot go limp (damp) or hold/lurch?
2. Separately, simulate comms loss (unplug the control link) and watch the same.
- **Record both.** If either leaves stiff/lurching torques instead of damping, the robot is
  **NOT cleared for ground** — the controller needs a command-loss→damping watchdog first.

## Step 4 — Gantry, feet OFF ground (DAMPING HOLD)
```
./deploy/10_gantry_test.sh --dance thriller --stage gantry \
    --gantry-confirmed --estop-confirmed --arm
```
Type `FEET OFF GROUND` when prompted (at a real terminal). Controller starts in **damping
hold** — it loads the policy but does NOT play the motion. Watch logs:
`ssh unitree@192.168.123.164 docker logs -f g1dance-controller`.

## Step 5 — Motion playback in the air
Arm playback via the operator remote sequence. Watch joints track the dance with feet off
ground. Compare against the sim preview. Abort on any FMEA symptom below.

## Step 6 — Ground, harnessed, safety line TAUT
```
./deploy/10_gantry_test.sh --dance thriller --stage ground --estop-confirmed --arm
```
Type `GROUND SAFETY LINE SET`. Line taut (partial-weight). Sequence: stand-and-hold →
slow first motion → full dance. Keep the line taut until a clean full run.

## Step 7 — Push test
Only after a clean ground run. Gentle, increasing shoves; confirm recovery. Line still on.

---

## FMEA — failure modes, symptoms, abort, recovery
| Step | Failure | Symptom | ABORT | Recovery |
|---|---|---|---|---|
| 1 | Motor fault / miscount | <29 motors, fault flag | power off | reseat/diagnose motor; don't proceed |
| 3a | Command-loss ≠ damping | robot holds/lurches on kill | remote B, power | add on-Jetson damping watchdog before ground |
| 4 | Policy NaN / bad init pose | log NaN; robot snaps to a pose | remote B → kill_now | check policy_meta/init-pose match; re-verify in sim |
| 5 | Tracking divergence | joints lag/overshoot reference | remote B | lower gains / re-check sim2real; back to sim |
| 5 | Actuator instability | buzz/oscillation at a joint | remote B → kill_now | gain/armature mismatch — recalibrate model |
| 6 | Balance loss on ground | lean/sag past reference | remote B, line catches | keep line; re-tune; more gantry time |
| 6 | Base-vel estimator wrong | drift, wrong recovery | remote B | fix state estimator (base_lin_vel gap) |
| any | Comms drop | logs stop, robot uncommanded | remote B, power | on-Jetson deadman required before shows |
| any | Battery sag | voltage dip, weak motors | remote B, power | charge; never run a show < safe SOC |
| any | Overheat | heat/smell/thermal flag | power off | cool down; inspect |

## Abort ladder (memorize)
1. **Remote B (damping)** — in your hand, beats every script.
2. **`deploy/kill_now.sh`** — stops the container (secondary; needs a working link).
3. **Power switch** — the guaranteed cut.

Never approach the robot until you've **visually** confirmed it is still.
