# AGENT C — Hardware validation (latency fix, drift/buckle, stand-exit handback)

**Round:** 2026-07-13. **Owner:** **human operator + damping remote** (🤖 — cannot be automated).
**Branch:** `hw/validate-v5-and-exit-fix`. **Needs:** the robot on the control net, gantry available,
the signed v5 policy from **Agent A**.

## Prereqs (blocking)
- Agent A has delivered `data/policies/thriller_v5fid/` with a **signed ≥99% held-out verdict**
  (a sub-99% or unsigned policy is gantry-only, never ground/show).
- (Recommended) Agent B's start guard merged, so a suspended start is refused before it can thrash.
- Read the Safety tab reminders: **feet flat on solid ground, upright start, remote B-damp in hand.**

## The three things this lane closes (all purely hardware-gated)
1. **Validate the latency fix.** Run the v5 policy gantry → slack-tether → free (per the
   `PROJECT_STATE.md` progression). Bring **tether slack** and **watch for the ~45 s buckle**
   recurring — that is the exact drift/latency failure the retrain targets. Clean 45 s+ with no
   buckle = latency fix confirmed on hardware.
2. **Stand-exit handback.** Never got a clean end-of-dance test. Run with `EXIT_MODE=stand`
   (`exit_stand` opt-in in the run gate) and confirm the robot ends **standing** and hands back to
   onboard without sagging. The runtime's `--exit stand` guard falls back to damp unless the motion
   ends near the standing pose, so it cannot topple a non-stand-ending dance — but the handback
   window itself is what needs eyes.
3. **Drift / station-keeping.** Confirm root-pos-weight-1.0 + the curriculum actually holds station
   on the ground (nominal drift stayed <1 m in sim; verify it holds on hardware).

## Method
Gantry-first (feet off is fine for the first contact — it's where an unproven policy is validated).
Then slack-tether, then free — only advance on a clean prior stage. If the ~45 s buckle recurs or
drift grows, STOP, record **Aborted/Incident** in the app (auto-demotes the policy), and send the
telemetry back to Agent A for a recipe iteration.

## Deliverable
Recorded show outcomes in the app (Clean/Aborted/Incident) + a `PROJECT_STATE.md` entry with
telemetry paths. On 3× clean free runs at ≥the show bar, the v5 policy becomes the signed show config.

## Safety (non-negotiable — CLAUDE.md)
No torque-cut e-stop on this robot. Human present, tether, **damping remote in hand**, typed
confirmation. The app's software **E-STOP** (Safety tab) is a SECOND stop, not a replacement — it only
damps app-launched runs. Never send low-level commands except through the vetted run path.
