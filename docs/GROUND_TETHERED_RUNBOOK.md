# Ground tethered runbook — first steps off the gantry

**Status: procedure only. Do NOT run any step here without a human physically present,
the robot on a tether/gantry set to catch a fall, and the remote in hand for B-damping.**

This is the staged, human-supervised path from "dances on the gantry" to "stands and
dances on the ground." It exists because the gantry policy is **not ground-safe**: its
observation needs a torso state estimator (base linear velocity + anchor position) that
the robot does not have, so on the gantry we fed zeros/approximations. Hanging, that is
harmless; standing, those wrong numbers drive a fall. The ground path uses a separately
trained **estimator-free (154-dim) policy** in `data/policies/thriller_ground/`.

There is **no torque-cut e-stop** on this robot. The only fast stop is the remote's
B-damping (and the runtime's own always-soft-on-exit). Every command below ends the
robot soft on any exit — normal end, Ctrl-C, external kill, crash. That is the floor,
not the plan. The plan is: a human catches it.

---

## Field notes — first ground session (2026-07-04)

Empirical, from the first tethered ground session. Read these; they change what to
EXPECT at each stage.

- **Stage A (stand-hold) sag is EXPECTED, not a fault.** A static PD-to-default is a
  spring to a pose, not a balance controller. With the onboard balancer released and the
  feet loaded, the legs settle into a squat (measured: knees ~55–66°, hips/ankles ~20° off
  the commanded stand) wherever leg torque balances gravity. Raising the gains did NOT
  close it (2.0× and 3.0× gave the same pose) — the equilibrium is set by load/contact,
  not stiffness. The torso stayed vertical and the pose was rock-steady. So Stage A proves
  the robot holds a STABLE tethered posture and that the stop path works — it does NOT
  prove a clean weight-bearing stand. Do not chase a perfect stand with gains here.
- **Weight-bearing comes from the BALANCING policy (Stage B), not from PD.** The same
  network that tracks the dance is what actively holds the robot up. Stand-hold is only
  the pre-check before it.
- **Gain-independence is a useful tell.** If the pose does not change when you raise
  `APPROACH_KP_SCALE`, the robot is resting on the tether/harness, not being held by the
  PD — i.e. the feet are not truly loaded. Slacken the tether to load them.
- **The ground policy is the estimator-free retrain (v2).** It was trained with the
  `ee_body_pos` height bound loosened (0.25→0.6 m) to get past an exploration cliff.
  Before Stage B, verify its `RESULT.txt` says `SIM_READY=YES` AND its ANKLE height error
  is tight (≤0.15 m). Loose *wrist* tracking is acceptable (cosmetic); loose *ankle*
  tracking is a fall risk — the verdict reports them separately.

---

## Preconditions (all must hold before ANY motion)

- [ ] A second person is on the tether/gantry, taking the robot's weight, ready to lift.
- [ ] Remote is on and in hand; you have B-damped this robot at least once today.
- [ ] Wired LAN up: `cat /sys/class/net/enp0s31f6/carrier` → `1`; `ping -c2 192.168.123.164`.
- [ ] Env for the laptop deploy shell:
      `conda activate tv` (has `unitree_sdk2py` + `onnxruntime`), then
      `export CONFIRMED_BY_HUMAN=alois`.
- [ ] Joint calibration is GO (standby pose matches the sim default):
      ```
      python deploy/check_joint_calibration.py \
        --meta data/policies/thriller/policy_meta.json --iface enp0s31f6 --threshold-deg 8
      ```
      Exit 0 = GO. Non-zero = STOP and re-zero before anything else.
- [ ] The ground policy exists: `data/policies/thriller_ground/{policy.onnx,policy_meta.json}`.
      If it is absent, `ground-run` **refuses** — stop at Stage A; the retrain has not landed.

Abort at ANY point by B-damping on the remote and/or Ctrl-C the process. The robot goes
soft either way; the tether holds it.

---

## Stage 0 — read-only sanity (no motion)

Confirms LowState streams and the obs is finite, commands nothing:

```
python -m pipeline.deploy_runtime --mode read --iface enp0s31f6
```

Expect: OBS SANITY block, `non-finite: 0`. If not → STOP.

## Stage A — standing hold, pure PD, NO policy (tethered)

Proves the robot can hold the ready stance standing before any learned control runs.
Firm PD only; holds indefinitely until you Ctrl-C or B-damp.

```
python -m pipeline.deploy_runtime --mode stand-hold \
  --iface enp0s31f6 --secs 5 --i-will-watch-the-robot
```

- Keep the tether taut through the 5 s move-to-default; let the feet gradually take
  weight but stay ready to lift.
- Watch for: a one-sided lean, buzzing/oscillation, any joint fault. **Some sag into a
  squat is normal** under load (see Field notes) — that is not an abort trigger by itself.
- **Go criterion (revised 2026-07-04):** settles into a STABLE, steady posture — torso
  vertical, no oscillation, no fault — held ~30 s. A clean upright stand is NOT expected
  with pure PD and is NOT required to proceed; what matters is *stability*, not reaching
  the commanded pose. (A quick read-only `check_joint_calibration.py` in another shell
  should show the SAME pose twice a few seconds apart — settled, not still sinking.)
- **Abort criteria:** a progressive/worsening sink (still sinking on the second read), a
  one-sided lean, oscillation, fault, or unexpected sound → B-damp immediately. Ctrl-C
  ends the hold soft.

If Stage A is unstable (sinking, leaning, or oscillating), do NOT proceed. Re-check
calibration and tether loading first; gains are NOT the lever for the sag (proven
2026-07-04).

## Stage B — shortest ground policy segment (tethered)

Runs the estimator-free ground policy for a **capped** few seconds, tether taking most
of the weight. `--max-secs` is mandatory. Conservative action cap (`GROUND_MAX_ACTION`,
default 6.0) trips to damping on any spike.

Start tiny and grow only on clean segments:

```
# 3 seconds first
python -m pipeline.deploy_runtime --mode ground-run \
  --iface enp0s31f6 --max-secs 3 --i-will-watch-the-robot
```

Sequence, one step per clean run, re-centering the stance between each:
`--max-secs 3` → `5` → `10` → `20` → full motion.

- Stage 1 of the run does a firm 4 s move-to-default + brief hold, then the policy
  starts from that pose (the motion begins at default → no lurch on entry).
- Keep light tension on the tether the whole time. Damp at the FIRST sign of a fault,
  lurch, lean, or an action-cap STOP message.
- The run ends itself with a smooth kp fade to damping at `--max-secs`.

**Only after a clean full-length tethered run**, with the tether progressively slackened
across several runs, consider a spotter-only (untethered but hands-ready) attempt — that
is a separate go/no-go decision with the human present, not part of this document.

---

## Stage B-ODOM — PROVEN gantry policy, fed the onboard estimate (PREFERRED)

Added 2026-07-04 after finding the robot publishes a base-state estimate on
`rt/odommodestate` (~184 Hz: position + velocity + height). This lets the **gantry
policy — already 100% in sim** — run on the ground with an HONEST 160-D obs (real
`base_lin_vel` + re-anchored `motion_anchor_pos_b`) instead of the estimator-free
retrain (which failed to balance). This is the preferred Stage B.

```
# 3 seconds first — tethered, remote in hand
GROUND_MAX_ACTION=10 python -m pipeline.deploy_runtime --mode ground-run-odom \
  --iface enp0s31f6 --max-secs 3 --i-will-watch-the-robot
```
Same `3 → 5 → 10 → 20 → full` progression as Stage B.

Path-specific notes (all verified OFFLINE 2026-07-04, `tools/sim_ground_odom.py`):
- **Action cap.** The gantry policy's real action range reaches ~8.5 during the dance
  (that is why the gantry run used `MAX_ACTION=12`). The default `GROUND_MAX_ACTION=6`
  would false-trip ~4% of ticks — **set `GROUND_MAX_ACTION=10`** for this mode (above the
  8.5 legitimate max, below a runaway). Start there; do not drop to 6 or it will damp on
  normal dance moves.
- **Odometry is a hard dependency.** The mode reads `rt/odommodestate` before releasing
  the motion service and **refuses (NO-GO) if it is not being published**, and damps if
  the stream drops mid-run. It never falls back to fabricated terms.
- **TWO things must be confirmed on the first tethered run** (they could not be validated
  without motion):
  1. **Odom survives motion-service release.** It publishes with the service released
     (observed at rest), but confirm `position`/`velocity` keep updating sanely once the
     policy has control. If it freezes → damp; the obs goes stale.
  2. **Velocity source.** Default `ODOM_VEL_SOURCE=diff` derives world velocity from
     position differencing (frame-unambiguous, mildly noisy). Only switch to
     `ODOM_VEL_SOURCE=field` (the EKF velocity, smoother) after a gentle **sway test**
     confirms the field's frame matches (push the torso +x; the reported body-frame
     `base_lin_vel` must point the right way regardless of heading).
- **Re-anchoring.** The torso position origin is captured at policy start (robot at the
  reference pose), so absolute-frame offset and slow XY drift cancel — the obs matches how
  the reference anchor behaves in training.

---

## What each safety layer does (so you trust the abort path)

- `--i-will-watch-the-robot` + `CONFIRMED_BY_HUMAN=alois`: both required for any motion
  mode; missing either → refuse before the DDS channel is even opened.
- Estimator-free gate: `ground-run` loads `thriller_ground/` and **refuses** if the
  policy's obs still lists `base_lin_vel` or `motion_anchor_pos_b` — it will not feed a
  fabricated estimator quantity to a standing robot.
- Per-tick guards: non-finite obs, `|action| > GROUND_MAX_ACTION`, or a cycle overrun
  → immediate damp.
- Always-soft-on-exit: normal end, Ctrl-C (SIGINT), external kill (SIGTERM), or any
  exception → a damping burst (kp=0, kd=2) then a prompt `os._exit` (DDS teardown can
  hang; we damp first, then hard-exit so the robot is never left energized).

## Tunables (env)

- `CONFIRMED_BY_HUMAN=alois` — required.
- `APPROACH_KP_SCALE` (default 2.0) — firm move-to-default / stand-hold gains.
- `GROUND_MAX_ACTION` (default 6.0) — ground action cap; raise only after clean runs.
- `--iface` (default enp0s31f6), `--secs` (stand-hold ramp), `--max-secs` (ground-run cap).

## If it falls

B-damp, let the tether take it, power stance down calmly. The process is already
damping on its way out. Note what you saw (which joint/side led, at what second),
re-check calibration, and drop back a stage. Do not raise `GROUND_MAX_ACTION` or skip
the tether to "get past" a fall.
