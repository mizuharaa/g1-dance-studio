# AGENT D — Policy-in-the-loop Simulation Sandbox (the "honest preview")

**Owner: Claude-orchestrated agent.** Laptop, CPU + `onnxruntime`. No robot, no GPU.
FLAGSHIP lane — everything else is judged against what this reveals.

## Why (the 60–70 % problem — tester report 2026-07-10)
The 3D preview plays the **reference** motion (the design intent). The robot runs an RL
**policy** that only *approximately tracks* it: subtle/fast moves get washed out, some are
skipped — the robot does ~50–70 % of what the preview shows. **The preview is dishonest.**
Build a sim that runs the ACTUAL policy so the preview reflects what the robot will really do,
and so motions/commands can be test-loaded before they ever touch hardware.

## What to build
`tools/sim_sandbox.py` + `pipeline/obs_builder.py` (refactor) — a MuJoCo policy-in-the-loop
rollout that mirrors `deploy_runtime.py` EXACTLY.

1. **Share the obs math.** Refactor the 160-dim obs construction out of `deploy_runtime.py`
   into a pure `pipeline/obs_builder.py` (`build_obs(state, reference, meta) -> np.ndarray`),
   imported by BOTH deploy_runtime (no behavior change) and the sandbox. A re-implementation
   risks a layout mismatch = garbage rollout; SHARE the code. See `OBS_LAYOUT`, `_anchor_quat`,
   `_align_reference`, the `Reference` class, and `policy_meta.json` (kp/kd/action_scale/default).
2. **Sandbox loop** (50 Hz control, decimation 4, sim_dt 0.005): mujoco state + reference →
   `build_obs` → `policy.onnx` (onnxruntime) → action → PD at deploy kp/kd/action_scale →
   `mj_step` → record. Use `third_party/mujoco_menagerie/unitree_g1/scene.xml`.
3. **Inject the MEASURED latency** (`data/telemetry/latency_diag_20260709/DIAGNOSIS.md`:
   40–80 ms as obs + action delay) so the twin matches HARDWARE, not ideal sim. Make it a flag
   (`--latency-ms`) so you can render ideal-sim vs hardware-like side by side.
4. **Outputs:** (a) rendered rollout video (EGL) = the honest preview; (b) a per-joint
   **reference-vs-achieved tracking report** — which DoFs / timestamps the policy drops
   (the "skipped / subtle" moves), reusing `tools/motion_quality` + a tracking-error metric
   (RMS + peak per joint, and % of reference range achieved).
5. **Test-load mode:** feed an arbitrary reference segment OR a scripted command sequence
   (SDK `LocoClient` style — stand / squat / set_velocity / arm-action IDs, see the G1 SDK
   High-Motion guide) and render the policy/robot response before hardware.

## Trust gate (measurement discipline — do NOT skip)
Cross-validate the sandbox obs against a REAL read-mode run: `deploy_runtime --mode read`
logs the robot's actual `LowState`; feed the same logged state into `obs_builder`; byte-compare
the obs (tol 1e-5). Only then is the twin trustworthy. Commit the comparison to `data/telemetry/`.

## Acceptance
- `obs_builder` shared; `deploy_runtime` behavior unchanged (existing `deploy_exit`/deploy tests green).
- Sandbox renders the Thriller policy rollout; the tracking report quantifies the
  reference→achieved gap (should reproduce the ~60–70 % the tester saw).
- One test-load command sequence runs end-to-end.
- Committed evidence + `PROJECT_STATE.md` decision-log entry.

## Files you may touch
`tools/sim_sandbox.py` (new), `pipeline/obs_builder.py` (new, refactor), `deploy_runtime.py`
(extract-only, no safety-path behavior change — flag if you must), `data/telemetry/`, `docs/`.
Out of scope: `cloud/`, `ui/`, the safety envelope behavior.

---
## Phase-2 DELIVERED (2026-07-10): side-by-side "dance studio" preview
`tools/sim_studio.py` renders two frame-synced panels with a live state overlay (achieved
fraction, fell@). Default = REFERENCE (intended, kinematic) | POLICY (actual, dynamic sandbox)
— the honest fidelity gap ON SCREEN (Thriller: reference 100% vs policy 74%, subtle moves
visibly washed out). `--dance-b <after>` switches to POLICY(before) | POLICY(after) — the
before/after retrain comparison, ready for Lane E's output. Tests in tests/test_sim_sandbox.py.
Future: "send SDK-style command signals" (LocoClient stand/squat/velocity + arm-action IDs) is
a separate onboard-controller sim, not the RL-policy path — scope it as its own step.
