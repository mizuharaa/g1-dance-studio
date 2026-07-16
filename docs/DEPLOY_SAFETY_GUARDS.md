# Deploy-side safety guards (staged for the next robot-day)

Five additive, fail-safe (= DAMP) guards in `pipeline/deploy_guards.py`, wired into
`pipeline/deploy_runtime.py`. They target the two real fall incidents (logs/jobs.md,
2026-07-08): (1) the ENTRY FALL — onboard balance is released, then a static PD ramp
moves to the policy's ready pose, and in that unheld window a SUSPENDED / no-contact
robot tips because its base-velocity estimate is invalid; (2) the policy OVER-COMMANDS
(v7 sandbox showed ~220% amplitude = thrashing).

**Status: BUILT + UNIT-TESTED, UNVALIDATED ON HARDWARE.** The robot is DOWN (burnt
DC-DC, RMA pending). Every numeric threshold below is a starting point and MUST be
measured on the gantry before it is trusted. These guards gate the NEXT deploy.

**Scope discipline:** the guards are inert on the PROVEN feet-off gantry path. The
gantry policy leaves `requires_ground_contact` unset (feet-off is intentional and
in-distribution), so the suspension/contact gates do nothing for it. The proven
`--mode run` loop keeps its existing target math untouched (no new rate limiter on it).

---

## The contact signal we have (and don't)

Measured from the SDK IDL (`~/robot/unitree_sdk2_python/.../unitree_hg/msg/dds_/`,
2026-07-16): the G1 speaks the **`unitree_hg`** LowState, whose only fields are
`imu_state` (quaternion, gyroscope, accelerometer, rpy) and `motor_state[35]`
(q, dq, ddq, **tau_est**, temperature, vol).

**There is NO foot-force sensor on the G1.** `foot_force` / `foot_force_est` exist only
on the `unitree_go` (quadruped) LowState. `rt/odommodestate` carries a base
pose/velocity/height estimate but FREEZES the instant we release the motion service
(hardware-confirmed 2026-07-04), so it is useless mid-run.

Contact is therefore inferred from **proxies**, each with a documented blind spot:

| Proxy | Source | What it sees | Blind spot |
|---|---|---|---|
| **Support torque** | `tau_est` on hip_pitch + knee + ankle_pitch (both legs) | Standing bears ~35 kg → high support torque; suspended legs hang → torque collapses | Needs calibration; noisy during dynamic motion |
| **Kinematic planted-confidence** | `leg_odometry` foot-world-velocity ≈ 0 | Foot slip/lift *during motion* | CANNOT tell static suspension from static standing (both have zero foot world velocity) |

The support-torque proxy is the **only** signal that distinguishes a robot standing
still from one hanging still, so it is what the entry gate uses.

---

## The five guards

### 1. Never run a state-estimation-dependent policy while SUSPENDED
`_check_stable_contact()` (entry) + mid-run contact-loss trip. Gated on the policy's
`requires_ground_contact` flag: a free-stand ground policy sets it (in its
`policy_meta.json`); the gantry policy does not. `ALLOW_SUSPENDED=1` force-permits a
flagged policy to run suspended for deliberate gantry validation (loud warning).
Mid-run: `ContactEstimator.lost_contact()` (debounced) → raise fault → damp. Wired into
`ground-run`, `ground-run-odom`, `ground-run-legodom`.

### 2. Require stable foot contact before entering run mode
`_check_stable_contact(sub, meta)` runs BEFORE `_release_motion_service()`, so a refusal
leaves the robot self-balanced under onboard control. It confirms support torque
≥ `SUSPENSION_TAU_MIN` for `CONTACT_CONFIRM_TICKS` consecutive samples (debounced ≈200 ms).

### 3. Action clamps + rate limits
`clamp_and_rate_limit()`: hard per-joint POSITION clamp (from `pipeline/g1_limits.py`
MJCF truth, else the meta band) AND a per-tick TARGET change (rate) limit
(`RATE_LIMIT_FRAC * velocity_limit * dt`). Deploy-side backstop against thrashing,
independent of the training-side action-rate penalty. A non-finite target RAISES
(fail-safe). Wired into the three ground modes; the proven `--mode run` is left as-is.

### 4. Estimator-validity check
`check_estimate_valid()`: NaN / base-speed > 2.5 m/s / height off the body / contact lost
→ invalid → raise fault → damp. Wired into `ground-run-odom` (odom base velocity) and
`ground-run-legodom` (leg-odom base velocity + height).

### 5. Independent damping watchdog
`DampingWatchdog`: a SEPARATE daemon thread that damps if the main loop misses its
`beat()` deadline (`WATCHDOG_DEADLINE_S`, default 200 ms ≈ 10 ticks) OR a fault flag is
raised (NaN / bad estimator / lost contact). Independent of the main loop so a HUNG loop
still damps. Wired into all four policy modes (`run`, `ground-run`, `ground-run-odom`,
`ground-run-legodom`).

**Independence caveat (honest):** Python threads share the GIL. If the main loop hangs
in a BLOCKING call that releases the GIL (DDS `Read`, `time.sleep`, onnxruntime
inference — every realistic stall mode of this loop) the watchdog WAKES and fires. A
pure-Python CPU-bound infinite loop would hold the GIL and starve it. **The firmware
remote B-damp remains the ultimate hard stop** (there is no hardware e-stop).

---

## Config knobs (env)

| Env | Default | Meaning |
|---|---|---|
| `REQUIRE_GROUND_CONTACT` | (meta flag) | Force the contact gates on/off, overriding the policy meta |
| `ALLOW_SUSPENDED` | 0 | Permit a ground-contact policy to run suspended (gantry validation) |
| `SUSPENSION_TAU_MIN` | 12.0 Nm | Support-torque threshold for "bearing weight". **UNVALIDATED** |
| `CONTACT_CONFIRM_TICKS` | 10 | Entry-gate debounce (≈200 ms) |
| `CONTACT_LOST_TICKS` | 8 | Mid-run contact-loss debounce (≈160 ms) |
| `RATE_LIMIT_FRAC` | 1.5 | Rate-limit cap = frac × velocity_limit × dt |
| `RATE_LIMIT_ENABLE` | 1 | Disable to restore pre-guard target math on ground modes |
| `EST_BASE_SPEED_MAX` | 2.5 m/s | Estimator-validity base-speed ceiling |
| `EST_HEIGHT_MIN/MAX` | 0.20 / 1.00 m | Plausible base-height band |
| `WATCHDOG_DEADLINE_S` | 0.20 s | Missed-heartbeat deadline |
| `WATCHDOG_ENABLE` | 1 | Disable the watchdog thread |

---

## WHAT MUST BE MEASURED ON THE GANTRY (before trusting each guard)

Follow measurement discipline: commit the script AND its raw output.

1. **Support-torque calibration (guards 1 & 2) — HIGHEST PRIORITY.** With the robot
   under onboard AI-stand, record `support_torque(tau_est, joint_order)` (a) SUSPENDED
   on the gantry (feet clear) and (b) STANDING feet-on-ground, ~30 s each. Confirm a
   clean separation and set `SUSPENSION_TAU_MIN` to the midpoint. Until this is done the
   12 Nm default is a guess; a too-high value would refuse a healthy standing entry, a
   too-low value would fail to catch suspension.
2. **Rate-limit non-interference (guard 3).** On a PROVEN gantry run, record the max
   per-tick |Δtarget| per joint (telemetry already logs `target`); confirm it stays under
   `RATE_LIMIT_FRAC * velocity_limit * dt` so the limiter never clips a healthy dance.
   Tighten `RATE_LIMIT_FRAC` toward the measured envelope afterward.
3. **Estimator-validity band (guard 4).** From leg-odom / odom telemetry on a good run,
   confirm base speed stays well under 2.5 m/s and height inside [0.20, 1.00] m so the
   gate never false-trips; tighten the band to the observed envelope.
4. **Watchdog → damp latency (guard 5).** Inject a deliberate stall (e.g. a one-off
   `time.sleep(0.5)` in a test build, feet-off, remote in hand) and MEASURE the wall time
   from missed beat to motors going soft. Confirm it is < the deadline + a few poll
   periods, and that it beats a real topple. Also confirm the watchdog does NOT fire on
   normal `soft_overruns` (tune `WATCHDOG_DEADLINE_S` above the worst clean-run tick).
5. **Contact-loss debounce (guard 1, mid-run).** Feet-off transition on the gantry:
   confirm `CONTACT_LOST_TICKS` catches a real lift within ~160 ms without tripping on
   transient single-foot unloading during the dance.

---

## Couldn't do without hardware / SDK details

- **No true foot contact.** Everything here is a proxy; a real deploy would benefit from
  a foot-force/pressure sensor the G1 does not expose on `unitree_hg`.
- **All thresholds are unvalidated.** They encode the right SIGN of each effect but not a
  calibrated magnitude — the robot is down.
- **GIL-bound stall** is not covered by the thread watchdog (see caveat); the firmware
  remote B-damp is the backstop. A SIGALRM/`setitimer` watchdog (fires between Python
  bytecodes even under a pure-Python loop) is a complementary future addition.
