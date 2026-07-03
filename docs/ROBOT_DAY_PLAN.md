# Robot Day — full staged plan (Thriller, first hardware session)

**Read this top to bottom the night before. Print the one-pager: `docs/ROBOT_DAY_CHECKLIST.md`.**

---
## ☀️ MORNING — START HERE (in order, don't skip)
1. **Laptop wired to the robot** (Ethernet, 192.168.123.x). `cat /sys/class/net/enp0s31f6/carrier` → `1`, then `ping -c2 192.168.123.164` replies.
2. `export CONFIRMED_BY_HUMAN=alois` and `cd ~/g1-dance`, then `scripts/preflight_robot_day --stage gantry` → expect **GO**.
3. **Stage 0a — get the controller onto PC2** (never done before; most likely time-sink — do it early). See below.
4. **Stage 0 — health check** (motors, remote, gantry rigged, battery). Robot on the gantry, feet ~5 cm off the ground.
5. **Verify the robot's standby pose matches the sim** (joint-calibration check) — a real fall risk if skipped. See Stage 0.
6. **Stage 1 — gantry**, and do **Step 3a (kill→damping)** FIRST. Then gantry tracking.
7. Gates → tethered → free → push, each earned. Remote in hand all day; abort at the first weirdness.

If anything is red, work the **First 30 minutes** table at the bottom before proceeding.

---

This is a STAGED day. Between every stage there is a MANDATORY gate. Having a whole
day does NOT mean skipping a gate — it means doing every stage thoroughly and stopping
at the first weirdness. A tired operator with all day is exactly who skips a gate; the
scripts are built so you can't advance without consciously attesting the prior stage was
clean.

## The one safety truth to hold all day
This tether-free G1 has **NO torque-cutting hardware e-stop.** Your only stops are:
1. the **remote's B-damping** (in your hand, tested every stage), and
2. the **power switch**.
`deploy/kill_now.sh` stops the controller container (SIGTERM→SIGKILL) but does NOT
guarantee the robot goes limp — that's exactly what Step 3a measures. Until Step 3a
passes, **the remote is your only trusted stop. Abort at the first twitch, buzz, or
sag — do not wait to see if it recovers.**

## What's staged for today
- Policy: **Thriller attempt-1** (98.4% held-out — GOOD for gantry; base_lin_vel≈0 on the
  gantry is in-distribution). Attempt-2 (targeting ≥99%) may finish overnight; if it does,
  rebuild the bundle from it before the ground-free stage (see "Swapping in attempt-2").
- Motion: **thriller_deploy.csv** — has a 2.5 s cosine ramp from the standby default pose,
  so activation does NOT lurch. NEVER deploy the raw show clip (up to ~39° step at frame 0).
- Gains: the bundle carries **policy_meta.json** — the SIM PD gains (low, overdamped ζ=2).
  The controller MUST load THESE, not stock Unitree gains, or the policy is unstable (fall).

## Preconditions (laptop, before you touch the robot)
```
export CONFIRMED_BY_HUMAN=alois
cd ~/g1-dance
scripts/preflight_robot_day --stage gantry        # expect GO for gantry
```
Preflight is honest: **gantry = GO**, **ground-free = NO-GO** until a ≥99% policy exists.

Build + push the gantry bundle (dry-run first, read every line):
```
python deploy/gen_config.py --dance thriller \
  --policy data/policies/thriller/policy.onnx \
  --motion data/motions/thriller/thriller_deploy.csv \
  --verdict data/policies/thriller/heldout_verdict.json --gantry
deploy/02_push_bundle.sh --dance thriller                 # dry-run
deploy/02_push_bundle.sh --dance thriller --yes-push      # real (needs robot on robot-lan)
```

---

## Stage 0a — Network + controller on PC2 (do this FIRST, ~20–40 min, the likely time-sink)
The controller has **never been installed on the robot** — budget time and do it before anything else.

**Network (each session):** laptop↔robot is the Ethernet cable (interface `enp0s31f6`).
```
cat /sys/class/net/enp0s31f6/carrier      # must be 1 (cable seated, robot on)
ping -c2 192.168.123.164                   # PC2 must reply
ssh unitree@192.168.123.164                # answer the ROS prompt: 1
```
If `carrier` is 0: re-seat the cable, confirm the robot is powered. If ping fails: check the laptop is on 192.168.123.2. (See `~/robot/RUNBOOK.md` §0–1 — read-only.)

**Install the controller (software only — never starts a motor):**
```
export CONFIRMED_BY_HUMAN=alois
deploy/01_pc2_install.sh                    # DRY-RUN — read every line
deploy/01_pc2_install.sh --yes-install      # real (needs the LAN up)
```
What it does: verifies Docker on PC2, pulls `qiayuanl/unitree:jazzy`, clones the controller repo.
**Likely snag — the robot LAN has no internet, so `docker pull` on PC2 may fail.** Fallback (pull on the laptop, ship the image over the LAN):
```
docker pull qiayuanl/unitree:jazzy                              # on the laptop (has internet)
docker save qiayuanl/unitree:jazzy | ssh unitree@192.168.123.164 'docker load'
```
**GATE → health check:** `ssh unitree@192.168.123.164 'docker image inspect qiayuanl/unitree:jazzy'` succeeds AND the controller repo is present on PC2.

## Stage 0 — Health check (robot on, ~30 min, read-only)
Power on, secure on the gantry frame first. Verify: all 29 motors report in LowState,
firmware versions noted, both Inspire hand services up (read `~/robot` runbook, don't
modify it), battery full, remote paired and B-damping tested.

### Joint-calibration check — DO NOT SKIP (a real fall risk)
The policy assumes the robot's joint zeros equal the sim `default_joint_pos`. If the real
offsets differ by tens of degrees, every target is wrong from frame 0 and it can fall.
With the robot in **standby (damping hold), feet off the ground**, in the env that has the
Unitree SDK (per `~/robot/RUNBOOK.md`, e.g. `conda activate tv`):
```
python deploy/check_joint_calibration.py \
  --meta data/policies/thriller/policy_meta.json --iface enp0s31f6 --threshold-deg 8
```
Exit 0 = GO (standby pose matches sim). Non-zero = a joint is off → **do NOT run the
policy**; recalibrate/re-zero or investigate the offset first.

### Initial-pose match on activation
`thriller_deploy.csv` starts with a 2.5 s ramp **from the sim default pose** — so the robot
must actually be AT (or damped near) that standby pose when you arm, or the ramp begins from
the wrong place. The joint-calibration check above confirms this; if it passed, you're good.
If the robot is holding some other posture, put it back to standby before arming.

**GATE → gantry:** all 29 motors healthy, remote damping works, **joint-calibration check
GO**, robot hanging feet ~5 cm off the ground, straps rated + locked.

## Stage 1 — Gantry (feet off ground)
On the robot day: confirm the controller loads the SIM gains (policy_meta.json), then
`touch` the bundle's `SIM_GAINS_LOADED` and `LAUNCH_LINE_VERIFIED` on PC2 (Step 3). Then:
```
deploy/10_gantry_test.sh --dance thriller --stage gantry \
  --estop-confirmed --gantry-confirmed          # dry-run first
deploy/10_gantry_test.sh --dance thriller --stage gantry \
  --estop-confirmed --gantry-confirmed --arm    # type: FEET OFF GROUND
```
Controller starts in **damping hold** — it does not move until you arm playback on the
remote.

### Step 3a — the single most important measurement of the day (do it HERE)
With feet off the ground and the controller running in damping hold:
1. Note the posture. Trigger `deploy/kill_now.sh` (and separately, test remote B-damping).
2. **Watch what the robot does when the controller dies:** does it bleed to a soft damped
   hang, or does it hold/lurch to the last commanded torques?
3. Record it (video + telemetry). This decides whether `kill_now.sh` is ever trustable on
   the ground. **If it does NOT damp cleanly, the remote stays your only stop all day and
   you do NOT go slack-line.**

Then arm playback on the remote and watch feet-off tracking: do the joints follow the
reference smoothly? Pull telemetry: `deploy/pull_telemetry.sh --dance thriller --stage gantry --yes-pull`.
Compare commanded vs actual against the sim.

### State-estimator (DLIO) sanity — needed before ground, check it here
The policy consumes `base_lin_vel`, which the real robot must supply from its onboard
estimator (LiDAR-inertial odometry). On the gantry, true base velocity ≈ 0 — which is
**expected and in-distribution** (training noise ±0.5 m/s covers it), so the gantry run is
safe regardless. BUT verify the estimator OUTPUT isn't garbage: check the estimator/odometry
topic reads a near-zero, non-diverging velocity while hanging (LiDAR odometry can drift when
there's little translation or the gantry frame is in view). If it reports large or growing
velocity on the gantry, that's a **ground-free blocker** — note it; it's the reason the
obs-restricted fallback policy exists (`docs/sim2real_derisk.md`). Do NOT trust ground
free-standing until the estimator is confirmed sane.

**GATE → ground-tethered (ALL required):** tracking looks like the sim; no buzzing/oscillation;
Step 3a done and recorded; you know whether kill→damping works. Repeat gantry until boring.

## Stage 2 — Ground, tethered (taut line, partial weight)
Robot on the ground, safety line **TAUT** bearing partial weight — it cannot hit the floor.
Sub-99% policy is OK here because the line catches a fall.
```
deploy/10_gantry_test.sh --dance thriller --stage ground-tethered \
  --estop-confirmed --tether-taut-confirmed --gantry-passed --arm   # type: TETHER TAUT PARTIAL WEIGHT
```
Do stand-and-hold first, then a slow first section. Watch for the feet taking load
differently than sim (contact/friction gap). Pull telemetry.
**GATE → ground-free (the HARD gate — ALL required):**
- gantry + ground-tethered both clean and repeated,
- Step 3a kill→damping **confirmed good** (or you accept remote-only and think hard),
- onboard **DLIO state-estimator verified sane** (base_lin_vel matters now, unlike gantry),
- a **≥99% show-ready bundle** — OR a conscious `--informed-override` for a sub-99% policy.

## Stage 3 — Ground, free (slack line, self-balancing) — real fall risk
```
# with a >=99% show-ready bundle (preferred):
deploy/10_gantry_test.sh --dance thriller --stage ground-free \
  --estop-confirmed --tether-taut-confirmed \
  --gantry-passed --tethered-passed --kill-damping-confirmed --estimator-verified --arm
# type: GROUND FREE FULL FALL RISK
```
If only attempt-1 (sub-99%) is available, the script REFUSES unless you add
`--informed-override` and it prints a loud higher-fall-risk warning. Start with the
first few seconds, hand on the remote, expect to abort. Build up to the full dance only
after repeated clean short runs. Pull telemetry every run.
**GATE → push-test:** ground-free clean AND repeated (`--free-passed`).

## Stage 4 — Push test (gentle shoves)
```
deploy/10_gantry_test.sh --dance thriller --stage push-test \
  --estop-confirmed --free-passed --arm     # type: PUSH TEST BEGIN
```
Gentle, expected shoves only; recovery like the sim's push tests. Abort on any real
instability.

## Debrief
Pull all telemetry, compare sim-vs-real tracking per stage, note where the gap appeared
(contact, gains, estimator, latency — see docs/sim2real_derisk.md), and what to change
before the next dance. Record the outcome in the app (Show mode) once its outcome-capture
is wired.

## Swapping in attempt-2 (if it exports overnight)
```
# re-run the held-out gate on the box for attempt-2, pull policy.onnx + policy_meta.json,
# then rebuild — if it's >=99% it becomes a FULL (show-ready) bundle, unlocking ground-free
# without --informed-override:
python -m pipeline.mjlab_verify --eval-json <attempt2 eval> \
  --policy data/policies/thriller/policy.onnx \
  --motion data/motions/thriller/thriller_deploy.csv \
  --eval-motion <attempt2 npz> --out data/policies/thriller/heldout_verdict.json
python deploy/gen_config.py --dance thriller \
  --policy data/policies/thriller/policy.onnx \
  --motion data/motions/thriller/thriller_deploy.csv \
  --verdict data/policies/thriller/heldout_verdict.json     # no --gantry => full, if >=99%
deploy/02_push_bundle.sh --dance thriller --yes-push
```

## First 30 minutes — quick troubleshooting
| Symptom | Immediate action |
|---|---|
| `carrier` = 0 / can't ping PC2 | Re-seat the Ethernet cable; confirm robot powered; laptop on 192.168.123.2. (`~/robot/RUNBOOK.md` §1) |
| `ssh unitree@…164` hangs | LAN down — check carrier; power-cycle if needed. Answer the ROS prompt **1**. |
| `docker pull` fails on PC2 | No internet on robot LAN — use the laptop `docker save … | ssh … 'docker load'` fallback (Stage 0a). |
| Laptop↔PC2 DDS not talking (nothing on `rt/lowstate`) | Wrong network interface — use `enp0s31f6`; both ends must share the CycloneDDS iface (`~/robot` sets this). |
| Joint-calibration check = NO-GO | Do **not** run the policy. Re-zero/recalibrate the robot or investigate the offset. A tens-of-degrees error = guaranteed bad motion. |
| Controller won't leave damping | The `SIM_GAINS_LOADED` attestation isn't set, or gains didn't load — confirm the controller loaded `policy_meta.json` gains (NOT stock). It refuses to arm otherwise (by design). |
| Wrong/stock gains loaded | Stop. Reconfigure the controller to the bundle's `policy_meta.json` PD gains (low, overdamped ζ=2). Stock gains → instability/fall. |
| Policy output NaN / robot buzzes or oscillates | Remote-damp immediately. Suspect gains, joint mismatch, or obs (base_lin_vel). Do not re-arm until diagnosed. |
| Step 3a: robot does NOT damp on `kill_now` | The remote stays your ONLY stop all day. **Do not go slack-line / ground-free.** Investigate before any future ground work. |
| Estimator reports large/growing velocity on gantry | Ground-free blocker. Gantry is still fine (base_lin_vel≈0 is expected). Note it; consider the obs-restricted fallback later. |

## Abort ladder (memorize)
1. **Remote B-damping** — always, first, your hand never leaves it.
2. `deploy/kill_now.sh` — stops the container (only trustworthy if Step 3a confirmed damping).
3. **Power switch** — the only guaranteed torque cut.
