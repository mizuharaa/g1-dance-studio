# G1 Dance Pipeline: Synthesized Defect Report

## Executive summary — what has been silently costing training sessions

Across the confirmed findings, three mechanisms explain most of the wasted sessions, and they fall into three independent failure classes. Fixing any one does not fix the others.

1. **You have been training against a pre-blunted and sometimes wrong-tempo reference.** The unconditional Savitzky–Golay pass (motion_quality.py) low-passes exactly the sharp choreography hits that the glitch guards were built to protect (the left-knee 480→341 deg/s loss), so the policy is *rewarded for tracking a softened motion it can then never sharpen* (Finding A). Separately, a swallowed `csv_to_npz` failure silently substitutes the previous tempo variant's npz, so a speed-curriculum stage can train against the wrong tempo with no error (Finding B). Every session inherits A; B fires intermittently on fresh-box runs.

2. **The preview you judge policies by is not policy truth.** Every manually-pulled policy is previewed against a frozen Jul‑7 npz that drives *both* the reference pane and the policy's own command input (Finding C); the reference pane is additionally rendered ~90° rotated from the policy pane (Finding D); and the 770‑dim history-stacked obs contract that the preview relies on *does not exist in the deploy path at all* — deploy is broken for v8/v10/v11 and the preview's obs layout has never been validated against real deploy (Finding E). Net effect: "the dance looks wrong" judgments — and the re-runs they trigger — have been drawn from a broken witness.

3. **The v8→v11 ankle-safety recipe is largely inverted or dead.** The 40 Nm "velocity-honest" clamp is overwritten by the effort DR and never applied (Finding F); the ankle_torque_barrier is identically zero across the entire real operating band (15–19 Nm) and it *replaced* the working broad ankle-L2 penalty, so v10/v11 have **less** ankle-unloading pressure than v6/v7, not more (Finding G). Two more reward terms are mis-scaled: stance_foot_flat is an absolute penalty that fights the v11 leg-orientation tracking it was supposed to complement (Finding H), and torque_saturation_duration is structurally inert on legs and instead taxes the arm/wrist motion the recipe pays to keep (Finding I).

A fourth class — eval comparability (Findings J/K/L) — did not corrupt training but has made cross-version "did v11 beat v8" judgments meaningless, which plausibly drove decisions to keep iterating.

---

## CONFIRMED findings — fix specs

### A. Choreography snaps are blunted by the unconditional Savitzky–Golay pass
`tools/motion_quality.py:272-274` (`clean_motion`)

**Distortion:** The glitch guards (`MAX_GLITCH_CORE`, `GLITCH_DEV_RATIO`) only set `col[s:e+1]=False` (line 132), which exempts a protected run from *outlier rejection*. The subsequent `savgol_filter(cols, SG_WINDOW=7, SG_POLY=3, ...)` runs on every joint/frame unconditionally, so a genuine sharp move survives the guards and is then low-passed anyway (~29% peak-velocity loss on a 2–3 frame hit; verified 480→341 deg/s). The prep velocity clamp is *not* the cause (its threshold 486 deg/s sits above the raw 480 snap). `_fidelity_retention` only warns on amplitude retention, not peak-velocity, so this passes silently.

**Fix:** In `clean_motion`, exempt protected-run frame spans from the SG pass by **blending raw values back over those spans** (do not hard-splice — that creates a boundary jerk). The span data already exists in `runs["protected_runs"]` as `(frame_start, frame_end, col)` tuples in the same concatenated root-xyz+joints space SG operates on. **Do NOT** lower `SG_WINDOW` globally as an alternative — it is also `smooth_quat`'s default window (line 194) and is imported by `experiments/clean_motion_audit`, so it would silently weaken root-quaternion smoothing everywhere. No GPU. Must land before the v12 npz is (re)built, or the rebuilt reference is still soft.

### B. Failed csv_to_npz silently copies the previous tempo variant's npz
`cloud/run_attempt9.sh:49-52`

**Distortion:** `csv_to_npz` hardcodes output to `/tmp/motion.npz` (`csv_to_npz.py`, vendored line 309 on the box) and never clears it. Line 49 ends with `|| true` (swallows the exit code); line 50 `[ -f /tmp/motion.npz ] && cp /tmp/motion.npz "$NPZ"` copies whatever is there; line 52's guard checks existence, not tempo. The loop reuses `/tmp/motion.npz` across `060/075/090/100`, so a flaky EGL/replay failure on variant 075/090/100 copies variant k‑1's stale npz under the wrong tempo name. (Variant 060 is protected — a fresh `/tmp` makes the copy a no-op and `die` fires.)

**Fix:** In `run_attempt9.sh`, add `rm -f /tmp/motion.npz` immediately before each `csv_to_npz` call, and add a frame-count/duration check on the copied npz against the expected tempo before line 52 accepts it. **Do NOT** change `csv_to_npz.py`'s output path — it is vendored (pip mjlab==1.5.0 on the box; editing the repo copy is inert) and its `/tmp/motion.npz` sink is consumed by nine sibling scripts (`run_attempt3-8.sh`, `verify_v8_rerun.sh`, `retrain_v5_box.sh`). No GPU.

### C. Preview replays a stale, wrong-lineage motion as BOTH reference pane and command input
`pipeline/publish_policy.py:91-96` (+ `cloud/export_policy.py`, `scripts/retrain_pull.sh`)

**Distortion:** `ensure_preview_assets` copies the shared `data/policies/thriller/thriller_deploy.npz` (md5 b06a8508, Jul‑7 retarget) into any pulled dir lacking its own npz. `export_policy.py` stages only `policy.onnx` (its `npz` arg is accepted but never used); `retrain_pull.sh` pulls `exports/…/*` + `thriller_g1_clean.csv` but no npz — so the manual-pull path *always* hits the fallback. Both consumers read that same file: `tools/sim_studio.py:87` (left reference pane) and `tools/sim_sandbox.py:111` → `build_obs_ground` `command` term (58 of 154 dims, the policy's tracking target). Confirmed physically: `thriller_v6sk` and `thriller_v7ank` carry byte-identical b06a8508 copies. (Note: the app's own `cloud_motion.py:571` pull path already stages the correct npz — the defect is specific to `retrain_pull.sh`.)

**Fix:** (a) Have `export_policy.py` stage its `--motion-file` npz into `exports/`; (b) pull it in `retrain_pull.sh`; (c) as the `ensure_preview_assets` fallback, convert the already-pulled `thriller_g1_clean.csv` via mjlab `csv_to_npz` FK rather than copying the shared npz. **Do NOT** make `ensure_preview_assets` fail loudly — it has an explicit never-raise contract and `publish()` returns None (unregistered dance) on False, reintroducing the v6/v7 no-preview regression. No GPU.

### D. Reference pane rendered in npz world-yaw (~90°) while policy pane starts at identity yaw
`tools/sim_studio.py:93` (`_kinematic_reference`)

**Distortion:** `_kinematic_reference` uses the raw npz pelvis quat (`quat = d["body_quat_w"][:n,0,:].copy()`) as base_quat with no yaw alignment, while `run_sandbox` sets the sim base to identity (`sim_sandbox.py:139`) and yaw-aligns the policy's *reference* to that heading (`:150`); the npz t=0 yaw is ~90.3° (`deploy_runtime.py:397`). Both panes are drawn with one fixed `CAM_AZIMUTH=-45` (`sim_studio.py:40/122`; `lookat` translates but does not rotate). Result: a faithfully-tracking policy looks ~90° rotated / "reading from the back" relative to the reference — a render-frame illusion, not a behavior difference.

**Fix:** Render-side only. In `_kinematic_reference`, rotate the npz pelvis `base_pos` path and `base_quat` by the same `dyaw` that `run_sandbox`'s `align_yaw` applies (`dyaw = sim_yaw0 − npz_ref_yaw0 ≈ −90.3°`), so both panes share yaw‑0 before the fixed camera. This also corrects `render_overlay` and keeps `tests/test_sim_sandbox.py:43` green (it asserts only `rec["q"]`). **Do NOT** change the sim init quat in `run_sandbox` — it is shared with deploy diagnostics and the deploy-contract heading. No GPU.

### E. deploy_runtime has NO history stacking — the 770-dim obs contract lives only in the preview
`pipeline/deploy_runtime.py:1421` (`mode_ground_run`)

**Distortion:** `build_obs_ground` returns a single 154-dim frame (`expected = sum(w for _,w in order)`); `mode_ground_run` feeds it straight to `run_policy` (`obs[None]`, no reshape). No deque/history buffer exists in any deploy run loop. The 154→770 stacking exists **only** in `tools/sim_sandbox.py:81-105 _V8HistoryObs`, which re-implements (not imports) the layout — contradicting the sandbox docstring's "same code as the real robot" claim. A real v8/v10/v11 ground deploy would hand a 154-dim obs to a 770-input ONNX → shape-mismatch crash. So deploy is currently broken for the entire trained lineage, and the preview's flatten layout has no deploy implementation to validate against. (The warmup at `:746-749` uses 770-dim zeros, masking the mismatch until the live loop feeds 154.)

**Fix:** Implement a shared `_HistoryObs` used by BOTH `mode_ground_run` and the sandbox, with `n_hist = onnx_obs_dim // 154` (do not hardcode 5 — degrade to 1 for a genuine single-frame policy), warmup = first-frame backfill. Add a box smoke test that dumps one real 770-dim actor obs and asserts the term-major, oldest→newest layout — a transposed flatten is silent (no crash) and would drive a fall, so this test is a hard gate before signing any 770-dim build. **Needs the box** to validate.

### F. Ankle 40 Nm clamp is overwritten by the ankle effort DR — never applied
`cloud/sim2real_task_v8.py:438-456` (inherited by v10/v11)

**Distortion:** Two startup `dr.effort_limits` scale events target the same 4 ankle joints: `dr_ankle_effort_clamp` (scale 0.80 → 40 Nm, inserted first) then `dr_effort_limits_ankle` (scale U(0.65,0.95), inserted last). In pinned mjlab `dr.effort_limits` the scale branch always reads `get_default_field("actuator_forcerange")` (pristine 50 Nm) — it does not compose. The event manager applies startup terms in insertion order, so the DR event wins: trained ankle limit = 50 × U(0.65,0.95) = **32.5–47.5 Nm**, not the documented 26–38 nor the intended 40 cap. The cap is exceeded up to 47.5 Nm (~19% over) in ~half of envs; the intended 26 Nm weak-ankle floor is never trained. The v8/v10 selfchecks only assert the event *names* exist and print the intended envelope — blind to the overwrite. (Cushioned: the ankle_torque_barrier is intact and real ankle p95 is far below even 32.5, so the hard clamp rarely binds on a converged policy — moderate, not order-of-magnitude.)

**Fix:** Keep BOTH event keys (the string-presence selfchecks at `v8:603`, `v10:362-363` will fail if either is deleted). Make only the **last-inserted** `dr_effort_limits_ankle` carry the full intended range — `scale U(0.52,0.76)` (=0.65×0.80, 0.95×0.80) → 26–38 Nm — and turn `dr_ankle_effort_clamp` into a documented `scale (1.0,1.0)` no-op. **Do NOT** use abs-clamp-40 + scale-DR: the scale still reads default 50 and overwrites the 40. Add a selfcheck that reads the realized `actuator_forcerange` for the ankle ctrl_ids after startup, not just event presence. **Needs the box/GPU** to dump the realized forcerange and to retrain.

### G. ankle_torque_barrier is identically zero at real torques, and it replaced the only broad ankle penalty
`cloud/sim2real_task_v8.py:279-283, 400-405` (inherited by v10/v11)

**Distortion:** The barrier is `relu(|tau|−35)²` with `tau_soft=35`, so it produces zero gradient below 35 Nm — i.e. across the entire real operating band (ankle-pitch p95 15–19 Nm). `_apply_v8` pops the proven `ankle_torque_l2` (v7's −1e‑3, ~−0.6/step of continuous in-band unloading, credited with driving p95 to 10.7 Nm) and registers only the dead barrier. The only surviving torque penalty is all-joint `joint_torques_l2` −2e‑5 (~−0.01/step on the ankles, negligible). Net: v8/v10/v11 carry **no effective broad ankle-torque shaping** — less than v6/v7. `tau_soft=35` was tuned for the falsified 1.8×/114 Nm regime and never lowered when v10 moved to native tempo.

**Fix:** Lower `tau_soft` into the actual operating band (e.g. 12–18 Nm) via the env-overridable `G1_ANKLE_BARRIER_TAU` (only the weight −5e‑3 is asserted by selfcheck, not `tau_soft`), and **re-tune the weight** since the square term grows fast once the barrier bites in-band. **Do NOT** simply restore `ankle_torque_l2` or delete the barrier — both trip the hard selfcheck gates at `v8:526/530` and `v10:356`. Confined to the s2r v8+ lineage (stock task and `smoke_v3.py` unaffected). **Needs GPU** to retrain and re-tune. **Land together with F** — both re-shape ankle authority; tuning either alone will mis-balance.

### H. stance_foot_flat is an absolute foot-flatness penalty that fights the reference-tilt tracking terms
`cloud/sim2real_task_v10.py:236-242` (interacts with v11 motion_leg_ori)

**Distortion:** The tilt branch computes penalty from the robot quaternion alone (`q = command.robot_body_quat_w[:, self._foot_cols]`; `val = a² + b²` = sin² of absolute sole tilt) with no reference subtraction — unlike its lin_vel/yaw_rate siblings, which are reference residuals specifically so they "never fight the tracking objective." At w=−0.5 on the two ankle_roll bodies, on any stance frame where the reference sole is genuinely tilted (weight-shifts/edge-rolls) it opposes v11's `motion_leg_ori` (w=1.0, std 0.40, same ankle_roll bodies) with comparable magnitude (~0.25 each at ~15° tilt), flattening the ankle where the choreography wants tilt — degrading exactly the passages v11 was built to sharpen. Local to tilted-stance frames, not whole-trajectory.

**Fix:** Make the `kind=="tilt"` branch a residual against the reference sole tilt: penalize `(a_robot − a_ref)² + (b_robot − b_ref)²` using `command.body_quat_w` (the reference foot quaternion exists, body-indexed). Edit is confined to the tilt branch, leaving the lin_vel/yaw_rate stance terms untouched. **Do NOT** edit `FOOT_BODY_NAMES` (shared by all 3 stance penalties → would destroy the anti-drift guards) or `LEG_BODY_NAMES` (shared by motion_leg_pos + motion_leg_ori). **Needs GPU** to retrain. Confidence medium — whether it bites depends on how tilted Thriller's stance soles actually are.

### I. torque_saturation_duration is inert on legs/ankles and taxes the arms instead
`cloud/sim2real_task_v10.py:155-172, 262-271` (inherited by v11)

**Distortion:** Thresholds are 0.90 × per-joint effort limit. Leg/ankle bars (knee 125, hip 79–125, ankle 36 Nm) are 3–8× above measured dance torques (knee p95 31–34, ankle 15–19) → the term never counts a leg joint, delivering none of its stated leg/ankle saturation shaping. The only reachable bars are the 5 Nm wrists (4.5) and 25 Nm arms (22.5), so the −0.02/joint-step penalty lands on the upper body, opposing the w=1.0 `motion_arm_pos/ori` fidelity rewards. Effect on weights is real but modest (a few % of summed reward) — a persistent wrong-signed tax on arm crispness plus zero leg shaping.

**Fix:** Set realistic per-joint thresholds (or a much lower fraction for the high-limit leg joints) inside the v10 `SIM_EFFORT_LIMIT_NM` table; **decouple the ankle rows from `v8.ANKLE_EFFORT_LIMIT_NM`** (that symbol also drives the real actuator clamp and effort-DR — changing it would alter trained dynamics). Merely scoping the term to legs while keeping 0.90 leaves it inert, so pair scoping with a threshold reduction. Confined to v10/v11 training (no deploy/gate consumer). **Needs GPU** to retrain. **Land together with H** — both are stance/torque reward-stack terms; validate in one run.

### J. Ankle-p95 gate bar moved 15/20 → 22/25; PASS/FAIL not comparable across versions
`cloud/sim_gap_check.py:97-98`, `cloud/pick_checkpoint.py:39/41`, `cloud/train_v10_curriculum.sh:110-111`, `cloud/train_v11_curriculum.sh:110-111`

**Distortion:** v5–v9 gate ankle-p95 against the 15/20 Nm defaults; v10/v11 export `G1_GATE_ANKLE_P95_NOMINAL_NM=22 / WORST=25`. The PASS/FAIL string embeds the bar (`f"ankle_p95<={GATE[...]}Nm"`), and `pick_checkpoint` also honors the override — so under v10/v11 both the gate *and* checkpoint selection use the loosened bar. Since native-tempo ankle p95 (15–19 Nm) lands exactly in the band that fails 15 but passes 22, "v11 cleared the ankle gate v8 failed" is an apples-to-oranges verdict. Raw p95 Nm values remain comparable; PASS flags do not.

**Fix:** Record the actual bar as an explicit numeric field inside each `gap.json` gate block, and when comparing runs compare raw p95 Nm against one fixed bar, not the per-run PASS flag. Additive metadata — zero blast radius (`pick_checkpoint._score` reads only raw p95). No GPU, no retrain.

### K. Drift gate silently changed metric+bar; verify banner still says "beat drift 4.31"
`cloud/train_v11_curriculum.sh:159, 161`; gate at `cloud/sim_gap_check.py:112, 355-362, 536-539`

**Distortion:** v8 gated `drift_max ≤ 1.0 m` (its 4.31 is a global max). v11 gates `episode_max_p95_m ≤ 1.5 m` (a different reduction: per-episode max, then 0.95 quantile). The banner still tells the operator to "Beat v8 … drift 4.31" (a max_m value) and line 159 additionally misstates the live bar as "≤1.0 m". So `4.31 → 2.21` reads as a huge win but compares max_m vs episode_p95; like-for-like max_m is 4.31 → 3.19.

**Fix:** Pure echo-text edits (no consumer parses these lines; the gate PASS/FAIL is JSON-keyed in `sim_gap_check.py`). Quote v8 and v11 on the **same** labeled drift statistic (both max_m: 4.31 vs 3.19, or both episode_max_p95), and update line 159's "≤1.0 m" to the live "episode_max_p95 ≤ 1.5 m". **Do NOT** export `G1_GATE_DRIFT_P95_M=1.0` to match the old banner — `pick_checkpoint.py:38` also reads it and would silently change checkpoint selection and flip the gate. No GPU.

### L. heldout_eval certifies only the first 20 s of a ~50 s dance
`cloud/heldout_eval.py:60` (`_run_condition`)

**Distortion:** `_run_condition` overrides motion/num_envs/seed/corruption/push but never `episode_length_s`, so it inherits the train cfg's 20 s cap (`sim2real_task.py:211`). With `sampling_mode="start"` on a ~49.5 s motion, an env that reaches step 1000 (20 s) times out and is scored `success = newly & truncated & ~terminated` — a PASS at 20 s. So the 3-seed heldout "survival" only certifies ~40% of the dance and systematically over-states robustness vs the full-length `gap.json` gate; the two survival numbers measure different horizons. `sim_gap_check.py:236/453` already does the correct override; heldout was never patched.

**Fix:** In `_run_condition`, set `env_cfg.episode_length_s = _motion_duration_s(cfg.motion_file) + 0.2` (mirror `sim_gap_check`; add/import the helper — `max_steps=4000` already accommodates the full motion). Standalone script, no library importers — only makes heldout stricter. No GPU, but re-runs the eval (fast).

---

## Prioritized fix order for the orchestrator

**GROUP 0 — reference & preview truth (land before the next run; no GPU except the box smoke test).** These do not require retraining to fix, but until they land you cannot trust the reference you train on or the preview you judge by.
- **A** (savgol blunt, `motion_quality.py`) — must land *before* the v12 npz is rebuilt.
- **B** (stale tempo copy, `run_attempt9.sh`) — must land before any fresh-box variant conversion.
- **C + D + E** as one "preview/deploy honesty" batch: C (`publish_policy.py`/`export_policy.py`/`retrain_pull.sh`), D (`sim_studio.py` yaw), E (shared `_HistoryObs` in `deploy_runtime.py` + sandbox). Validate together in one preview render; **E's layout assertion needs a box smoke test** before signing any 770-dim deploy build. C without E leaves deploy broken; C without D still shows a rotated pane.

**GROUP 1 — ankle load recipe (must land + retrain TOGETHER; needs GPU/box).**
- **F** (effort-DR overwrite, `v8` + `v10` selfcheck) and **G** (dead barrier + restore ankle shaping, `v8`). These jointly determine ankle authority and torque shaping; tuning one alone will mis-balance. F's realized-forcerange assertion needs the box (mjlab installed); both need a training run to confirm stability.

**GROUP 2 — stance/torque reward stack (land + retrain together; needs GPU).**
- **H** (stance_foot_flat → residual, `v10`) and **I** (torque_saturation thresholds, `v10`). Both are stance/torque terms interacting with the v11 leg/arm fidelity rewards; validate in one run. Can be folded into the Group 1 retrain if you want a single validation run, but keep the reward changes reviewable separately.

**GROUP 3 — eval comparability (land before reading the next gate; no retrain).**
- **J + K** together (gate-bar recording + drift banner) — both are "make historical verdicts comparable" edits.
- **L** (heldout episode length) — independent, re-runs the held-out eval only.

**GPU/box required:** E (box smoke test), F (box forcerange dump + retrain), G/H/I (retrain to validate). Everything in Groups 0 (except E's test) and 3 is CPU/no-retrain and should land immediately.

---

## Watchlist (PLAUSIBLE — code-true but adjudicated immaterial; each had a refute on magnitude)

Fix only if the named symptom appears; none is worth a session on its own.

- **anchor_drift_xy resolves to mjlab's 3D `bad_anchor_pos`, not XY-only** (`v6:81`). Real, but a 3D norm still upper-bounds horizontal drift and z is separately capped at 0.25 m, so it only tightens the reset ~13% at a rarely-reached z boundary. *Settle:* only matters if any telemetry keyed on "xy" drives a decision — cosmetic mislabel otherwise. Safe one-line fix (call `_bad_anchor_pos_xy` directly) if you want the label honest.
- **Torso/pelvis lack a dedicated tracking term** (`v11:96`). Refuted: torso *is* the anchor and carries `motion_global_root_pos/ori` (gradient ~11, the most-tracked body). Only the **pelvis** is genuinely under-weighted, and it is kinematically pinned to the torso through the waist joints. *Settle:* watch rollouts for pelvis carriage drift; if seen, add a pelvis pos/ori term (additive, safe) — do NOT stop slacking the torso (intentional waist-window feature).
- **Per-frame grounding injects root vz and runs twice** (`grounding.py:187`). Real mechanism (min-over-both-feet corner → dg/dt → base_lin_vel_z), but negligible for drift-only Thriller and the second pass is near-idempotent. *Settle:* measure `base_lin_vel_z` p95 before/after grounding on v12; if >~0.05 m/s, replace the min-based SG ground line with a support-switch-continuous per-foot low-pass (inside `ground_motion_per_frame` only — do not touch `per_contact_height` or drop either pass).
- **stance_foot_lin_vel double-counts motion_body_lin_vel** (`v10:224`). Refuted: the whole-body term (std=1.0, mean over 14 bodies) gives near-zero foot gradient, so the stance term is the only real foot-velocity signal and is masked to reference-stance frames — an anti-slip term, not a freeze cause. *Settle:* only revisit if stepping is observed over-suppressed.
- **publish_policy substitutes a 160-dim gantry `policy_meta.json`** (`publish_policy.py:82`). Real substitution, but the PD-governing meta fields (kp/kd/effort/action_scale) are byte-identical across generations and the sandbox reads obs_dim from the ONNX. The genuine effect is deploy-side and fail-safe (`_ground_obs_order` hard-REFUSES). *Settle:* fix alongside C by staging a per-policy meta at export; add a guard refusing a meta whose onnx obs dim disagrees with `policy.onnx`.
- **Sandbox hardcodes obs term order** (`sim_sandbox.py:77`). Matches deploy's fallback today; only bites on a future term reorder. *Settle:* have the sandbox call `D._ground_obs_order(meta)` for the per-frame order and assert `per_frame_dim × n_hist == onnx_obs_dim` — cheap latent-bug insurance, folds into E.
- **sim_gap_check section boundaries hardcoded to the 49.5 s clock** (`sim_gap_check.py:118`). Diagnostic labels only (never gated); but on v12 (51.8 s, +2.5 s intro) the section table mislabels choreography and the last ~2.3 s gets no fall attribution. *Settle:* offset boundaries by the intro delta (+2.5 s) and extend the last section to true motion end — additive, no consumer breaks. (Not proportional scaling — the change is an additive front-intro restore.)
- **Verify banner drift "≤1.0 m" vs enforced 1.5 m** (`train_v11:159`) — same fix as K; banner text only.
- **success_estimate / anchor-on-M100 / pick_checkpoint screen** (`success_estimate.py:74`, `train_v11:139`, `pick_checkpoint.py:28`). All eval-advisory, non-gating or non-training: the success band's optimistic edge is mis-anchored by ~5 pts (floor intact); the anchor reference row either no-ops on a wrong default path or scores the anchor on a motion/task it never tracked; the checkpoint screen ranks on `delay40ms_push` only while the gate takes min-survival over four conditions (but delay40ms_push is the harshest by construction). *Settle:* record true per-row scored-motion provenance in `success_estimate`; point `G1_ANCHOR_ONNX` at the real anchor and score it under its own base task (or drop the on-M100 anchor line); make `pick_checkpoint` screen all four delay conditions (or compute "worst" as the gate does) — ~2× screen cost. None distorts weights or preview.