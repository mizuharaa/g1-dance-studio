# Multi-Agent Task Board — G1 Dance

Parallel work lanes, each with its own instruction file. Lanes touch **disjoint files**
so agents can run simultaneously without merge conflicts.

## ▶ CURRENT ROUND — 2026-07-13 (twitch + latency + safety)

| Agent | File | Where | Status (2026-07-13) |
|---|---|---|---|
| **A** — Latency-curriculum retrain (v5) | `AGENT_A_LATENCY_RETRAIN.md` | ☁️ GPU box | 🟡 **IN PROGRESS — Claude is executing it.** Box up (103.245.250.152:55792), v5 curriculum **stage 2/3, ~55% (~1h47m to finish)**. **Do NOT delegate** — collides with the live run. |
| **B** — Airborne/ground-contact guard | `AGENT_B_AIRBORNE_GUARD.md` | 💻 code (+🤖 later) | 🟢 **OPEN — delegate this NOW.** Code + tests are fully doable on the laptop; only the *gantry validation* waits on the robot. The one parallel task worth spinning up. |
| **C** — Hardware validation | `AGENT_C_HARDWARE_VALIDATION.md` | 🤖 robot | 🔴 **BLOCKED — robot is down** (pelvis power-electronics fault, battery out; see `data/telemetry/pelvis_diag_20260713/`). Resume only after the robot is repaired. |
| **D (new)** — Robot pelvis repair | `data/telemetry/pelvis_diag_20260713/REPORT.md` | 🔧 human + Unitree | 🔴 **NEW BLOCKER.** Inspect/repair the pelvis board (ranked culprits + checklist in the report); warranty/safety event. Unblocks C. |
| **Me (Claude)** | `CLAUDE_THIS_SESSION.md` | 💻 laptop | ✅ Lane 1 (show-display) + Lane 2 (twitch reprep) DONE; safety UI/E-kill/dark-mode/video/**app training monitor** DONE; pelvis diagnosis DONE; **executing Agent A**. |

**So: the only thing to hand a parallel manual agent right now is Agent B** (A is mine/live, C+D are blocked on the robot).
Step-by-step retrain: `docs/RETRAIN_RUNBOOK.md`. Shipped branches: `fix/show-display-mpv`, `fix/twitch-source-reprep`.

<details><summary>◾ PRIOR ROUND — 2026-07-10 (DONE — kept for history)</summary>

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

</details>

## Push (from a machine with `.secrets/`) + resync to `main`
This checkout can't push (no key/PAT loaded). From your laptop with creds:
```bash
git push handoff fix/show-display-mpv fix/twitch-source-reprep      # or: git push origin <branch>
# Agents A/B/C create their branches from main and push similarly.
```
**Merge order** (deploy_runtime is the one shared file — Agent B owns it, land it before other deploy edits):
1. `fix/show-display-mpv`, `fix/twitch-source-reprep`  — safe now.
2. `feat/airborne-contact-guard` (Agent B) — after gantry validation.
3. `train/latency-curriculum-v5` (Agent A) — after the ≥99% held-out gate.
4. `hw/validate-v5-and-exit-fix` (Agent C) — record-only; artifact = signed outcome + PROJECT_STATE entry.
```bash
git checkout main && git merge --no-ff <branch> -m "merge <branch>: <what/gate passed>"
```
