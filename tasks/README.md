# Multi-Agent Task Board — G1 Dance (updated 2026-07-10)

Parallel work lanes, each with its own instruction file. Lanes touch **disjoint files**
so agents can run simultaneously without merge conflicts.

## THE central problem (tester report 2026-07-10): the robot does ~60–70 % of the dance
The 3D preview plays the **reference** motion (design intent); the robot runs an RL **policy**
that only approximately tracks it — subtle/fast moves wash out or get skipped. **The preview is
dishonest.** Lane D (the policy-in-the-loop sandbox) makes the gap *visible + testable*; Lanes
B+E *close* it; Lane A removes the controllable latency that erodes it. All five lanes serve this.

| Lane | File | Owner | Needs | Status |
|---|---|---|---|---|
| A — SDK latency & C++ hot path | `AGENT_A_SDK_LATENCY.md` | **manual agent + human** | Ubuntu laptop, robot, remote | not started |
| B — Motion quality (de-glitch **+ feasibility**) | `AGENT_B_MOTION_QUALITY.md` | Claude agent | this repo (CPU) | **de-glitch DONE (merged `16f6aa7`)**; Phase-2 feasibility TODO |
| C — Frontend dashboard revamp | `AGENT_C_FRONTEND_UI.md` | manual agent | Node, `ui/server.py` | **reviewed SAFE-TO-MERGE** (branch `frontend/show-mode-preview-revamp`) |
| **D — Policy-in-the-loop sim sandbox** (the "honest preview") | `AGENT_D_SIM_SANDBOX.md` | Claude agent | laptop (CPU + onnxruntime) | **NEW — flagship** |
| **E — Fidelity retrain** (track subtle moves, right latency DR) | `AGENT_E_FIDELITY_RETRAIN.md` | manual agent + human | Ubuntu + GPU box | **NEW** |

**Review outcomes (2026-07-10):** Lane B was implemented twice in parallel; main's version
(`16f6aa7`) is the canonical one (hybrid robust-z+floor spike detector, cubic-spline outlier
interp, tangent-space quat SG) and is verified on the real Thriller (spikes 25→0, jerk_peak
11,939→2,454). The redundant `motion-quality-filter` branch is retired. The last **latency
retrain FAILED** (`data/telemetry/latency_retrain_20260710/`) — Lane E has the corrected recipe.

## Rules for ALL agents (from CLAUDE.md — non-negotiable)

1. Read `PROJECT_STATE.md` before starting; update it + commit after every meaningful step.
2. **Never** send commands to the real robot. Robot motion = human present + tether +
   damping remote + typed confirmation. Agents prepare; humans deploy.
3. Never modify `~/robot/` (original laptop teleop setup).
4. Measurement discipline: no "decisive" finding without an independent cross-check;
   commit every measurement script AND its raw output (`logs/` or `data/telemetry/`).
5. Stay inside your lane's file list. If you must touch another lane's file, stop and flag it.
6. This machine is **Windows, no GPU, no robot, no `.secrets/`**. Cloud/GPU/robot steps are
   out of scope here — write them up as instructions for the Ubuntu laptop instead.

## Lane file boundaries

- **A**: `pipeline/deploy_runtime.py` (instrumentation only), `tools/measure_*`, new `deploy/cpp/` — plus docs.
- **B**: `pipeline/prep_motion.py`, `pipeline/retarget_gvhmr.py`, `pipeline/vet_motion.py`, new `tools/motion_quality.py` — plus tests.
- **C**: `ui/` only (server.py changes limited to static-file serving), new `ui/frontend/`.
