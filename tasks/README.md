# Multi-Agent Task Board — G1 Dance (2026-07-10)

Three parallel work lanes, each with its own instruction file. Lanes touch **disjoint files**
so agents can run simultaneously without merge conflicts.

| Lane | File | Owner | Needs |
|---|---|---|---|
| A — SDK latency & C++ hot path | `AGENT_A_SDK_LATENCY.md` | **USER'S MANUAL AGENT + human** (hardware required for measurement/validation) | Ubuntu laptop, robot, damping remote |
| B — Motion quality (twitch/glitch fix) | `AGENT_B_MOTION_QUALITY.md` | **Claude-orchestrated agent** (launched 2026-07-10) | This repo only (CPU, no GPU) |
| C — Frontend dashboard revamp | `AGENT_C_FRONTEND_UI.md` | **USER'S MANUAL AGENT** (needs shadcn + Playwright MCP servers) | Node, running `ui/server.py` |

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
