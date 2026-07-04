# ARM-DANCE-OVER-ONBOARD-BALANCE — design, recon, and first-test runbook

**Runtime:** `pipeline/arm_dance_runtime.py` · **Tests:** `tests/test_arm_dance.py`
**Status:** built + offline-verified 2026-07-04. Not yet exercised on hardware.

The audit-recommended "bookable show baseline": the robot's ONBOARD controller keeps
balance (normal standing/balance mode stays ACTIVE the entire time) while we stream the
dance's ARM choreography through Unitree's **arm-sdk weight-blend** interface. Thriller
is ~90% arm choreography, so this preserves most of the show while eliminating the
thermal / balance / estimator risks of the full-body low-level path
(`pipeline/deploy_runtime.py`), all of which live in *us owning the legs*.

---

## 1. Recon findings (read-only, with citations)

### 1.1 The arm-sdk interface (confirmed in two independent sources)

**Topic & message.** Publish `unitree_hg.msg.dds_.LowCmd_` on **`rt/arm_sdk`** — not
`rt/lowcmd`:

- `~/robot/xr_teleoperate/teleop/robot_control/robot_arm.py:17-18` —
  `kTopicLowCommand_Debug = "rt/lowcmd"`, `kTopicLowCommand_Motion = "rt/arm_sdk"`;
  `:89-93` — `motion_mode=True` selects the `rt/arm_sdk` publisher.
- `~/robot/unitree_sdk2_python/example/g1/high_level/g1_arm7_sdk_dds_example.py:107` —
  official Unitree G1 example publishes `LowCmd_` on `"rt/arm_sdk"`.

**Weight mechanism.** `motor_cmd[29].q` (the `kNotUsedJoint0` slot) carries the blend
weight: **0 = onboard owns the arms, 1 = arm-sdk owns the arms**:

- `g1_arm7_sdk_dds_example.py:64` — `kNotUsedJoint = 29 # NOTE: Weight`; `:135` —
  `low_cmd.motor_cmd[kNotUsedJoint].q = 1  # 1:Enable arm_sdk, 0:Disable arm_sdk`;
  `:164-168` — release stage ramps the weight `1 -> 0` over ~3 s.
- `robot_arm.py:166-167` — teleop sets weight `1.0` when streaming starts (while
  commanding the arms' *current* positions, so no jump); `:234-237` — on go-home it
  ramps the weight down `np.linspace(1, 0, 101)` with 20 ms steps (~2 s soft handoff).

**Which joints arm_sdk may command.**

- Teleop commands **only the 14 arm joints**, DDS motor indices **15..28**
  (`robot_arm.py:181-184` iterates `G1_29_JointArmIndex`; enum at `:281-298`).
- The official example commands **17 joints: arms + waist yaw/roll/pitch (12-14)**
  (`g1_arm7_sdk_dds_example.py:91-103`) — so arm_sdk *can* take the waist too.
  **v1 here is arms-only** (see §5 open questions): the waist is a balance DoF and the
  onboard controller should keep it.
- Full G1 29-dof DDS index map: `robot_arm.py:300-345` (`G1_29_JointIndex`), identical
  to `g1_arm7_sdk_dds_example.py:19-64`. It matches `policy_meta.json`'s
  `joint_order_29dof` 1:1 (verified in `tests/test_arm_dance.py`), but the runtime maps
  **by name** so any reorder is caught loudly.

**Message init sequence** (`robot_arm.py:108-137`): read one `rt/lowstate`
(`LowState_`) to get `mode_machine`; build `LowCmd_` with `mode_pr = 0`,
`mode_machine = <from lowstate>`, `motor_cmd[i].mode = 1` for commanded joints; each
tick set `q/dq=0/tau=0/kp/kd`, compute `crc = CRC().Crc(msg)`, publish
(`robot_arm.py:181-187`). The runtime mirrors this exactly.

**Rates.** Teleop streams at 250 Hz (`robot_arm.py:83`, `control_dt = 1/250`); the
official example at **50 Hz** (`g1_arm7_sdk_dds_example.py:69`, `control_dt_ = 0.02`).
Our 50 Hz loop (1 npz frame per tick) is squarely within the official usage.

**Velocity ceiling.** Teleop clips arm target motion to 20 rad/s
(`robot_arm.py:82,158-163`). The runtime refuses any trajectory exceeding that
(`MAX_ARM_SPEED_RAD_S`); thriller_deploy peaks at ~8.5 rad/s.

### 1.2 Which mode the proven teleop actually used (honesty note)

`~/robot/start_teleop_armsonly.sh` does **not** pass `--motion`. In
`teleop_hand_and_arm.py:140-148`, *without* `--motion` the teleop enters **debug mode**
(`MotionSwitcher.Enter_Debug_Mode()` → `ReleaseMode()` loop,
`teleop/utils/motion_switcher.py:15-23`) and publishes `rt/lowcmd`, locking the legs
itself; *with* `--motion` it uses `rt/arm_sdk` with the onboard controller active
(comment at `:140`: "motion mode (G1: Regular mode R1+X, not Running mode R2+A)").
So the user's daily arms-only teleop hardware-proved the **arm PD gains and the DDS
plumbing** on this exact robot, but the **arm_sdk weight-blend path itself** comes from
the same codebase's supported `--motion` branch plus Unitree's official example — it has
not yet been exercised on *this* robot. That is open question #1 and exactly what the
first 5 s supervised test verifies.

### 1.3 The dance artifacts

- `data/policies/thriller/policy_meta.json` — `joint_order_29dof`,
  `default_joint_pos_rad`, per-joint `kp_stiffness`/`kd_damping` (arms: kp 14.3
  shoulder/elbow/wrist-roll, 16.8 wrist-pitch/yaw; kd 0.91/1.07), effort limits
  (arms 25 Nm, wrist p/y 5 Nm).
- `data/policies/thriller/thriller_deploy.npz` — `joint_pos [2589, 29]` @ 50 fps
  (51.8 s), ordered per `joint_order_29dof`. Frame 0 == default pose (worst delta
  0.00° — the 2.5 s activation ramp is embedded), so the approach blend starts from a
  known, gentle place.
- Arm columns = the 14 `*shoulder*/*elbow*/*wrist*` entries (npz cols 15-28 == DDS
  motors 15-28 for this meta). `waist_*` and leg columns are **dropped** — onboard owns
  them.

---

## 2. Architecture

```
laptop (tv env)                                    robot (onboard controller ACTIVE)
─────────────────────                              ──────────────────────────────────
thriller_deploy.npz ──> extract_arm_trajectory     rt/lowstate ──> current arm pose,
        (14 arm cols, by-name map)                                 mode_machine, telemetry
                │
                ▼            50 Hz wall-clock paced (same as deploy_runtime)
  [b] weight 0→1 (2 s) holding CURRENT arm pose      ┐
  [c] cosine approach → dance frame 0 (2 s)          │  LowCmd_ on rt/arm_sdk:
  [d] dance frames 1:1 (--max-secs cap)              ├─ motor_cmd[15..28] q/kp/kd
  [e] cosine return → captured start pose (1.5 s)    │  motor_cmd[29].q = weight
  [f] weight 1→0 (1.5 s)  ← runs on EVERY exit path  ┘
```

- **Modes:** `--mode read` (default; fully offline — zero DDS, prints plan/mapping/
  timeline/sanity) and `--mode arm-run` (gated).
- **Gates** (`require_arm_run_gates`): `--i-will-watch-the-robot` AND
  `CONFIRMED_BY_HUMAN=alois` AND `--max-secs N`; `--max-secs 0` (full dance) needs
  `ARM_FULL_RUN=1` on top. Same spirit as `deploy_runtime`.
- **Exit safety:** `_hand_back_and_exit` mirrors `deploy_runtime._finalize_and_exit` /
  `_install_damp_on_signals`: SIGINT/SIGTERM/crash/normal-end all ramp the weight from
  *wherever it is* to 0 over `ARM_RELEASE_S` while holding the last commanded targets
  (the weight blend itself is the smoothing), then a 10-msg weight-0 burst so the
  release lands, telemetry flush, `os._exit`. A second signal during finalize exits
  immediately (escape hatch, weight mid-ramp — onboard still blends).
- **Telemetry:** reuses `deploy_runtime.Telemetry` (mode `arm-run`, per-tick q/dq/
  tau_est/temps/IMU → `data/telemetry/<stamp>_arm-run.npz`). Conventions: `action`
  channel = arm-sdk weight broadcast over 29 slots; `target` = measured q with the 14
  arm slots replaced by commanded targets; `stage` 1=weight-up 2=approach 3=dance
  4=return.
- **Env knobs:** `IFACE` (default `enp0s31f6`), `ARM_WEIGHT_RAMP_S` (2.0),
  `ARM_APPROACH_S` (2.0), `ARM_RETURN_S` (1.5), `ARM_RELEASE_S` (1.5),
  `ARM_KP_SCALE` (1.0), `ARM_GAINS` (`meta`|`teleop`), `ARM_FULL_RUN`,
  plus `TELEMETRY`/`TELEMETRY_DIR` inherited from deploy_runtime.

### Gain choice (documented decision)

| source | shoulder/elbow/wrist-roll | wrist pitch/yaw | provenance |
|---|---|---|---|
| `meta` (**default**) | kp 14.3 / kd 0.91 | kp 16.8 / kd 1.07 | trained policy gains (policy_meta.json) |
| `teleop` preset | kp 80 / kd 3.0 | kp 40 / kd 1.5 | `robot_arm.py:74-79` — hardware-proven daily on THIS robot's arms (via the debug-path teleop; same motors, same PD law) |
| Unitree example | kp 60 / kd 1.5 (uniform, arms+waist) | | `g1_arm7_sdk_dds_example.py:74-75` |

Default is **meta × `ARM_KP_SCALE`** (kd scaled too — house pattern, stays overdamped).
Rationale: softest possible first contact. But note the trained gains assumed a
*closed-loop policy* correcting errors every tick; open-loop streaming at kp ≈ 14 will
visibly sag/lag under gravity (several Nm of shoulder gravity torque ⇒ tens of degrees
of sag). Expect the show-quality setting to be `ARM_GAINS=teleop` (proven values) or
`ARM_KP_SCALE≈4` (≈ the official example's 60) — step up only after the 5 s test looks
clean at soft gains.

### Music / timing (unchanged guidance)

The npz is 50 fps and the dance loop is 50 Hz wall-clock paced with the same
pacing/overrun logic as `deploy_runtime` — 1 frame per tick — so **dance frame 0 is the
identical musical reference point in both runtimes** and docs/audio_sync_design.md
applies unchanged: thriller_deploy embeds a 2.5 s activation ramp then the 1.5 s
standing lead-in (`audio_delay_s`), so **music starts at frame0 + 4.0 s**. In arm-run,
frame 0 occurs `ARM_WEIGHT_RAMP_S + ARM_APPROACH_S` (default 4.0 s) after streaming
starts and the runtime prints an explicit `DANCE FRAME 0 NOW` cue line.

---

## 3. Safety model — why this is the lower-risk path

1. **Onboard balance is never disengaged.** No `MotionSwitcherClient.ReleaseMode`, no
   `rt/lowcmd`, anywhere in the module (pinned by a source-scan test). The legs and
   waist are the onboard controller's problem for the entire run — the whole class of
   sag / gain-boost / ankle-thermal / estimator failures from the low-level path cannot
   occur.
2. **We can only touch arm motors.** The by-name map refuses anything outside DDS
   15..28; `send_arm_cmd` writes only those 14 slots + the weight slot (verified with
   fakes: legs/waist `motor_cmd` stay zeroed).
3. **The handoff is firmware-native.** Weight 0→1→0 ramps mean both takeover and
   release are blends computed onboard, not a controller swap. Engage streams the arms'
   *current* (frozen-snapshot) pose so there is no lurch; frozen rather than
   live-tracked because target=measured would let gravity walk the arms down as the
   weight rises.
4. **Every exit ramps the weight to 0** (normal, Ctrl-C, SIGTERM, crash) — the arms are
   always handed back, never left commanded. Verified in an offline smoke with an
   injected mid-dance sensor-loss crash.
5. **Bounded inputs:** targets clamped to the meta joint band; trajectory refused above
   20 rad/s (teleop's own ceiling); cycle-overrun watchdog → handback; LowState loss →
   handback.
6. **Human gates** identical in spirit to deploy_runtime, plus a mandatory `--max-secs`
   cap with `ARM_FULL_RUN=1` required for a full-length run.

Residual risk (be honest): fast arm swings move the CoM and the onboard balancer must
absorb that; the remote's B-damping remains the only true stop (no torque-cut e-stop on
this G1). Hence: capped first runs, human watching, clear space.

---

## 4. First supervised-test runbook

Prereqs: laptop on robot LAN (`enp0s31f6`, robot net 192.168.123.x), `tv` conda env,
robot in its normal standing **balance mode** (the mode the show will use — per
xr_teleoperate this is Regular mode R1+X, *not* Running mode), space clear of people /
obstacles (gantry fine but not required — feet stay onboard-controlled), **remote in
hand ready to damp**.

1. **Offline plan (no robot):**
   `python -m pipeline.arm_dance_runtime --mode read --max-secs 5`
   Check: 14-joint map, timeline, `PLAN OK`.
2. **5 s capped arm-run (soft gains):**
   `CONFIRMED_BY_HUMAN=alois python -m pipeline.arm_dance_runtime --mode arm-run --max-secs 5 --i-will-watch-the-robot`
   Watch for: (a) weight engage — arms must NOT jump (they should barely move);
   (b) gentle approach to the ready pose; (c) 5 s of (soft, possibly saggy) dance;
   (d) automatic return + release — onboard takes the arms back and holds them.
   ANY balance wobble from the torso → damp via remote, stop here, reassess.
3. **Mid-run abort test:** repeat the 5 s run, Ctrl-C during the dance. Verify the
   smooth weight-down handback (this also stands in for the crash path).
4. **Extend:** 15 s → 60 s at soft gains. Pull `data/telemetry/*_arm-run.npz`; check
   arm `tau_est` ≪ 25 Nm and temperatures flat (legs are onboard's — expect no thermal
   story at all).
5. **Tracking-quality pass:** if arms sag/lag (likely at meta gains), re-run 5 s with
   `ARM_KP_SCALE=2`, then `4`, or `ARM_GAINS=teleop` (hardware-proven arm gains).
   Re-verify the engage/release feel after every gain change.
6. **Full dance + music:** `ARM_FULL_RUN=1 ... --max-secs 0`. Start music 4.0 s after
   the printed `DANCE FRAME 0 NOW` cue (§2 music note).

## 5. Open questions / risks for the first hardware test

1. **Does arm_sdk work in the mode the show uses?** The proven daily script ran the
   debug path (§1.2). xr_teleoperate's comment says arm_sdk expects Regular mode
   (R1+X), not Running mode. The 5 s test answers this; if arms ignore the stream,
   check the active mode name (read-only `CheckMode`) before touching anything else.
2. **Firmware behavior if the stream stops at weight > 0** (e.g. laptop power loss —
   the one exit we cannot ramp): unknown. Assume the arms are stuck being blended
   toward the last command until weight times out or the mode changes. Mitigation:
   remote damp; keep runs capped until this is understood.
3. **Waist: include or not?** arm_sdk *can* command waist 12-14 (official example).
   **Recommendation: arms-only v1** — waist is a balance DoF; streaming it may fight
   the balancer. The dance's waist choreography is dropped in v1; revisit only after
   v1 is verified and only with a dedicated capped test.
4. **Blend semantics between 0 and 1** (linear torque/target blend vs threshold) are
   undocumented; our ramps are safe either way, but watch the engage carefully.
5. **Handback pose:** normal end returns the arms to the pose onboard held at engage
   (best proxy for "onboard's default"); an emergency handback releases wherever the
   arms are — verify onboard accepts that gracefully (teleop's own go-home ramps to
   zero pose before releasing weight, `robot_arm.py:222-238`, so a non-neutral release
   is the less-trodden path).
6. **CoM disturbance from big arm swings** (Thriller lunges/claws): onboard balance
   should handle it standing still, but this is exactly what the staged 5→15→60 s
   progression is for.
7. **50 Hz smoothness:** official example uses 50 Hz so it is in-spec; if the arms
   buzz/step, interpolate to 100-250 Hz laptop-side (teleop runs 250 Hz) — trivial
   change, not needed until observed.
