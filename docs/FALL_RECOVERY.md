# Fall detection & recovery — design doc

**Scope.** How the deploy runtime detects that the dancing G1 has toppled, and what
happens next. Honest split throughout: what is **built and running today**, the
**interim recovery** that a trip triggers, and the **trained get-up** that is future
work (needs the GPU box, currently deleted). Read alongside the one safety truth in
`docs/ROBOT_DAY_PLAN.md`: **this tether-free G1 has no torque-cutting hardware e-stop.**

Code referenced: `pipeline/deploy_runtime.py` (`_check_fall`, `mode_ground_run_legodom`,
`_finalize_and_exit`, `_restore_motion_service`) and `pipeline/shows.py`
(`record_outcome`).

---

## 1. Detection (BUILT)

### 1.1 The torso-topple trigger
`_check_fall(R_base, tick)` in `pipeline/deploy_runtime.py` is a **physical-state**
tripwire that runs alongside the action-based ones. `R_base = quat_wxyz_to_mat(imu_quat)`
is the pelvis-IMU body→world rotation; `R_base[2, 2]` is the torso z-axis' world-vertical
component — **uprightness**: `+1` fully upright, `0` horizontal, `<0` inverted. When

```
R_base[2, 2] < FALL_UPRIGHT_MIN     # default 0.35  ≈ 69.5° of pelvis tilt
```

`R_base` is the **pelvis** IMU rotation (the IMU sits at the pelvis; `R_base[2,2]` is a
diagonal element, so the uprightness value is robust to the quat-convention ambiguity).
The absolute threshold is valid because the vet gate forbids floorwork and requires pelvis
height ≥ 0.35 m — **only upright in-place dances are supported**, so no allowed choreography
tilts the pelvis past ~70°. Cross-checked against **all 26 legodom hardware runs (38.6 k
ticks): worst real lean 35.7° / uprightness 0.812** — a 0.46 margin to the 0.35 trip.

**Two signals, both debounced.** After the adversarial review (2026-07-07), `_check_fall`
uses `_fall_signal` = topple **OR** a **choreography-relative height collapse**
(`(h_est − h0)` sits `FALL_HEIGHT_DROP_M` = 0.15 m below the reference height change — this
catches a leg-buckle / vertical sag that the topple signal misses while the pelvis is still
upright; cross-checked: clean dances never sink >2.5 cm below the choreographed height, ~6×
margin). Either condition must hold for **`FALL_CONFIRM_TICKS` = 3 consecutive ticks (60 ms)**
before it raises — a single spurious IMU/odom sample can **never** damp a healthy robot (a
one-tick trip that damped-to-limp would itself induce the fall it claims to catch). It is
called each tick in `mode_ground_run_legodom` after the leg-odometry estimate. Replayed over
all 26 clean runs the debounced detector's worst fall-condition streak is **0 ticks** — zero
false trips. `FALL_UPRIGHT_MIN=0` / `FALL_HEIGHT_DROP_M=0` disable each signal.

**Why it exists — it catches what the action triggers miss.** The action tripwires fire on
an *output* that has gone bad. A fall can happen while the policy's actions still look
perfectly bounded — the robot is going down, but every commanded target is finite and
in-cap. The torso-topple trigger reads the *physical* result (orientation) rather than the
command, so it fires on that class of fall.

### 1.2 How it complements the other tripwires
The same control loop (and the shared `except BaseException → finally _damp` spine in every
motion mode) already carries these **action-/timing-/comms-based** triggers:

| Trigger | Condition | What it protects against |
|---|---|---|
| **Action cap** | `\|action\| > _acap` per joint (`action_cap_vector`: legs/waist at `GROUND_MAX_ACTION`, default 10.0; arm joints ×`ARM_ACTION_CAP_SCALE`, default 2.2) | Runaway / out-of-distribution policy output |
| **NaN / inf** | `~isfinite(obs)` or `~isfinite(action)` | Numerical blow-up in the obs build or the policy |
| **Cycle overrun** | tick wall-time `> 2·dt` (i.e. `>40 ms` at 50 Hz) | A stalled loop (missed deadline → stale command) |
| **Comms loss** | `read_state(timeout_s=0.5)` gets no `LowState` → `SystemExit` | Robot off / wrong iface / LAN down |
| **Topple + height collapse** *(this doc, debounced 3 ticks)* | pelvis `R_base[2,2] < FALL_UPRIGHT_MIN` OR torso `> FALL_HEIGHT_DROP_M` below the choreographed height | A fall the actions don't reveal — both a topple AND a straight-down leg-buckle |

All five raise into the **same** except/finally path (§2), so every trigger ends the run
the same safe way. The topple trigger is the only one that observes the robot's body pose
directly.

### 1.3 What it still does NOT detect (be honest about it)
The height signal (added after review) now covers the straight-down leg-buckle/sag case, but
the detector still does **not** catch:

- loss of foot contact, sliding, or the robot walking out of the 2 m dance area;
- a slow topple the **tether** catches before the pelvis passes ~70° and before a 15 cm sag;
- a fall during the non-policy phases (approach ramp / entry-catch / stand handoff) — the
  check runs in the **policy loop only**, and only in `mode_ground_run_legodom` (the show
  path); the superseded `ground-run` / `ground-run-odom` modes have no fall check. During
  those windows the tether + operator remote are the safety net.

It is an **orientation-topple** trigger, not a general balance-loss or floor-contact
detector. The action-cap and cycle triggers cover some of the collapse cases indirectly
(a buckling policy usually saturates actions or overruns), but there is no dedicated
contact/height fall signal yet. The operator with the remote remains the real backstop.

### 1.4 The tuning knob
`FALL_UPRIGHT_MIN` (env-overridable, default **0.35**) is the single knob. Lower = more
permissive (allows a deeper lean before tripping); higher = earlier, more sensitive trip;
`0` disables. The default is deliberately conservative so deep crouches / authored leans in
a dance never false-trigger while a genuine topple blows well past it.

### 1.5 Hardware-telemetry cross-check
Real-hardware dance telemetry gives the margin: **max torso tilt 26.6°** (uprightness min
**0.894**), with **zero frames anywhere near 0.35**. So the default threshold sits far below
any posture the validated dance actually reaches — large headroom (0.894 vs 0.35), no
false-trigger risk on that motion. Re-check this margin for any new choreography before
trusting the default (a dance with legitimately deeper leans would want a lower threshold).

---

## 2. Recovery — INTERIM (BUILT)

There is **no active get-up today.** The built recovery is: **on any trip, put the robot
soft and hand control back to onboard**, then have the operator physically recover it and
demote the dance so it can't be silently redeployed. Full path:

```
policy running (mode_ground_run_legodom loop)
   │  torso topples past FALL_UPRIGHT_MIN  (or: bad action / NaN / cycle overrun / comms loss)
   ▼
_check_fall raises RuntimeError("FALL DETECTED …")          # or a peer trigger raises
   ▼
except BaseException as e:  print("STOP: … -> damping")     # shared mode spine
   ▼
finally: _damp(…, secs=1.0)      → ~1 s of soft cmds (kp=0, kd=2) — robot goes LIMP
   ▼
_finalize_and_exit(0)
   ├─ _damp_burst(30)            → ~0.3 s more damping (belt-and-suspenders)
   ├─ _restore_motion_service()  → SelectMode('ai')  (RESTORE_MOTION_MODE, default "ai")
   │        └─ onboard vendor controller takes over; operator's REMOTE can pair again
   ├─ telemetry saved (after the robot is soft — safety never waits on I/O)
   └─ os._exit(0)
```

Key properties of this path:

- **The robot ends damped (soft), not fighting.** On a fallen robot the damp means it lies
  limp on the ground — the safe interim state — instead of a stiff PD loop struggling
  against the floor.
- **`_restore_motion_service()` re-asserts onboard `'ai'` (SelectMode).** This is a
  *control handoff*, not an auto-recovery: onboard `'ai'` does not itself get a fallen
  robot up. What it buys is that the **vendor controller + the operator's remote regain the
  robot** (a released motion service otherwise strands the robot so the remote can't
  reconnect). From there the operator uses the remote / vendor recovery / power switch.
- **Every abort path funnels here.** Fall trigger, action cap, NaN, overrun, comms loss,
  Ctrl-C, and external SIGTERM/SIGINT (`_install_damp_on_signals`) all route through
  `_finalize_and_exit`, so the robot is guaranteed soft on **any** exit.

### 2.1 Demotion — a fallen dance can't stay show-ready
After the robot is recovered, the operator records the show outcome as **Incident** (the
in-app checklist step; `pipeline/shows.py::record_outcome`). For a **live** show, an
`incident` (or `aborted`) result:

- resets the dance's `repeatability.consecutive_clean` streak to **0**,
- stamps `dance.incident = {show_id, result, at}`, and
- **demotes** the dance `show-ready → sim-verified`.

Because show-ready requires a passing sim exam **and** `REPEATABILITY_TARGET` (3)
consecutive clean live runs, the demoted dance cannot be redeployed as show-ready until it
is re-verified and re-earns its streak. This closes the loop: **a dance that fell on the
floor cannot silently remain show-ready and get pushed out again.** (A `rehearsal`-mode
incident is logged but never demotes — dry runs don't knock a dance out of the library.)

---

## 3. Recovery — TRAINED GET-UP (NOT BUILT; needs the GPU box)

A real recover-and-continue (or recover-and-safe-stand) capability is **future work.** It
cannot be built on the laptop alone — the policy/motion has to be trained on the GPU box,
which was **deleted 2026-07-07**. Recreate it first via **`docs/BOX_RECREATE_RUNBOOK.md`**
(GreenNode 4090 notebook, mjlab trainer). What a trained get-up would require:

**A. A get-up behaviour.** Either a **get-up reference motion** (retargeted the same way as
a dance) or an **RL get-up policy** trained on the box against the same G1 model / actuator
spec used for the dance policies (`policy_meta.json` PD gains, per-joint action scale).

**B. A fall-state classifier.** Prone / supine / on-side determine which get-up applies.
Today `_check_fall` yields only a single uprightness scalar — a classifier needs the full
torso orientation (from `R_base`, already available) plus joint configuration to name the
pose the robot is actually in.

**C. A runtime state machine** layered on the current abort path:

```
detect fall → damp (as today) → ASSESS safe-to-recover (space & people clear, robot settled)
            → classify fall state → select get-up → execute get-up → re-stand
            → hand back to onboard ('ai')
```

The first two boxes exist today; everything from **assess** onward is new.

**D. Safety gates on any auto-get-up** (mandatory, mirror the existing deploy gates):

- **Never** auto-get-up unattended or near people.
- **Operator-armed only** — an explicit human arm per run, like today's
  `CONFIRMED_BY_HUMAN=alois` + `--i-will-watch-the-robot`.
- **Tether-first validation** — a get-up is validated on the gantry/tether before it is
  ever allowed to run free on the ground, exactly like the dance-deploy staircase.

---

## 4. Safety non-negotiables

- **No hardware e-stop.** This tether-free G1 has **no torque-cutting hardware e-stop.** The
  only hard stops are the **remote's B-damp** (in the operator's hand every stage) and the
  **power switch**. Software damping (`_damp` / `_damp_burst`) is a soft landing, not a hard
  stop, and depends on the loop still running and DDS still delivering.
- **Get-up is never a substitute for the operator.** Detection + damp is a best-effort
  interim, not a guarantee (see the tilt-only limitation, §1.3). A future trained get-up
  does not change this: the operator, remote in hand, abort at the first weirdness, remains
  the real safety control. Auto-recovery is an assist, never the backstop.
