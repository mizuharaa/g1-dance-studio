# AGENT A — SDK Communication Audit & Latency Hardening (Python + C++)

**Owner: USER'S MANUAL AGENT + a human on robot days.** Phases 2–3 need the Ubuntu laptop,
the robot, and `.secrets/` — none of which exist on the Windows handoff machine.

## Goal

A live paid show tolerates zero avoidable delay. Harden the command path end-to-end:
verify the Python SDK loop never misses its 20 ms tick, then (evidence-gated) build a C++
onboard runtime so inference + lowcmd publishing runs on the robot's Jetson PC2 instead of
over the wire from a laptop.

## Context you must internalize first

- Read `HANDOVER.md` §2/§4/§8 and `data/telemetry/latency_diag_20260709/DIAGNOSIS.md`.
- **The 44 s fall was NOT a comms/language problem.** Measured: wire RTT 0.16 ms, DDS staleness
  ~2 ms; sensorimotor latency (actuation + leg-odometry estimation) 40–80 ms. The retrain with
  0–80 ms latency DR is the fix for that. This lane removes the *remaining, controllable* latency
  and jitter so the policy operates deep inside its trained envelope during shows.

## Code audit — ALREADY DONE (2026-07-10, on `pipeline/deploy_runtime.py`)

What's good (keep):
- unitree_sdk2py + CycloneDDS, single pre-allocated `LowCmd_` (no per-tick alloc), CRC reuse.
- 50 Hz loop with `time.sleep(max(0, dt - elapsed))` + overrun warning at 2·dt.
- Comms-loss deadman on `read_state`, guaranteed damping on any exit, DDS drained to latest msg.

Gaps found (your Phase 1/2 work):
1. **Relative sleep pacing** — each tick re-anchors on "now", so overruns accumulate as phase
   drift and `time.sleep` adds ~1 ms OS jitter. Fix: absolute-clock pacing
   (`next_t += dt; sleep(next_t - monotonic())`).
2. **No tick-time telemetry** — overruns only print. Add per-tick histogram (inference ms,
   publish ms, total ms) written to `data/telemetry/` on every run, so shows produce evidence.
3. **No RT scheduling** — the laptop process runs at default priority. `chrt -f 50` /
   `os.sched_setscheduler` + CPU pinning; document in the runbook.
4. **onnxruntime session opts unspecified** — set `intra_op_num_threads=1`,
   `inter_op_num_threads=1`, warm the session before releasing the motion service.

## Phases

### Phase 1 — instrument + measure (Ubuntu laptop, robot in `read` mode — SAFE, no motion)
- **CODE DONE (2026-07-10, Windows side, committed to main):** `TickClock` in
  `pipeline/deploy_runtime.py` — absolute-deadline pacing (no more schedule drift from
  relative sleeps) + per-tick work/late stats saved into every run's telemetry npz under
  `run_meta_json.tick_timing`; `_ort_session()` pins onnxruntime to 1 thread + pre-warms
  3 inferences before the motion service is released. Overrun→damp semantics unchanged
  (regression-tested: `tests/test_tick_clock.py`). Gap 3 (RT priority) is a run-command
  change, not code: launch with `chrt -f 50 python pipeline/deploy_runtime.py ...`.
- **MEASUREMENT still TODO (needs robot):** run a policy mode (gantry `run` is fine) or
  any real motion segment; the summary prints at damp time and lands in the npz. Commit it.
- **Gate:** if p99 work < 5 ms of the 20 ms budget and zero soft overruns, the Python loop
  is show-grade; C++ becomes an onboard-migration play, not a rescue.

### Phase 2 — C++ onboard runtime (Jetson PC2, `unitree_sdk2` C++)
- New `deploy/cpp/`: loads `policy.onnx` (onnxruntime C++ API), replicates the obs builder
  **exactly** (160-dim layout in `deploy_runtime.py` `OBS_*` constants + `policy_meta.json`
  per-joint kp/kd/action_scale — a mismatch = fall), publishes `rt/lowcmd` at 50 Hz.
- Must replicate the safety envelope: damping on any exit/signal, comms deadman, fall
  detector, `SelectMode("ai")` restore. **Port the safety code first, the policy second.**
- Cross-validate against Python: same obs log in, byte-compare actions out (tolerance 1e-5)
  before it ever touches the real robot.
- Keep the Python runtime as the fallback — the app's show flow selects the runtime.

### Phase 3 — hardware validation (human + damping remote, gantry/tether first)
- Same staged gates as `docs/ROBOT_DAY_PLAN.md`. Measure command→response latency again
  (`tools/measure_latency_from_telemetry.py`) and commit the before/after comparison.

## Acceptance
- Committed tick-timing evidence from a real run, Python and (if built) C++.
- C++ runtime passes action cross-validation + gantry stage before any ground use.
- `PROJECT_STATE.md` decision-log entries for every phase.

## Files you may touch
`pipeline/deploy_runtime.py` (instrumentation/pacing only — no behavior changes to safety
paths without flagging), `tools/measure_*`, new `deploy/cpp/**`, `docs/`.
