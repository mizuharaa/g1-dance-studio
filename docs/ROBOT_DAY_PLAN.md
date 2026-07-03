# Robot Day — full staged plan (Thriller, first hardware session)

**Read this top to bottom the night before. Print the one-pager: `docs/ROBOT_DAY_CHECKLIST.md`.**

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

## Stage 0 — Health check (robot on, ~30 min, read-only)
Power on, secure on the gantry frame first. Verify: all 29 motors report in LowState,
firmware versions noted, both Inspire hand services up (read `~/robot` runbook, don't
modify it), battery full, remote paired and B-damping tested.
**GATE → gantry:** all 29 motors healthy, remote damping works, robot hanging feet ~5 cm
off the ground, straps rated + locked.

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

## Abort ladder (memorize)
1. **Remote B-damping** — always, first, your hand never leaves it.
2. `deploy/kill_now.sh` — stops the container (only trustworthy if Step 3a confirmed damping).
3. **Power switch** — the only guaranteed torque cut.
