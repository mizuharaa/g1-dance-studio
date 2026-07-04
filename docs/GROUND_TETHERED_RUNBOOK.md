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
- Watch for: sag, one-sided lean, buzzing/oscillation, any joint fault.
- **Go criterion:** holds the pose calmly, weight on feet, tether slack, for ~30 s.
- **Abort criteria:** any lean past a few degrees, oscillation, fault, or unexpected
  sound → B-damp immediately. Ctrl-C ends the hold soft.

If Stage A is not rock-solid, do NOT proceed. Re-check calibration and gains
(`APPROACH_KP_SCALE`, default 2.0) first.

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
