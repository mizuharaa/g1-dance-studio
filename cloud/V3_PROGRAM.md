# V3 DANCE-QUALITY PROGRAM — 2026-07-06 (launched ~12:50 ICT / 05:50 UTC)

Goal: recover the crispness gap between the promoted show policy (s2r-b,
`data/policies/thriller/policy.onnx`) and its sim rollout. Hardware evidence
(2026-07-06 telemetry vs `thriller_deploy.npz`): arm RMS 13.2 deg / wrist
15.2 deg (p95 27-30), two wrists lag 100-160 ms, several arm joints at
0.78-0.92 amplitude. Timing overall fine (median lag 20 ms).

HARD CONSTRAINTS HONORED: robot untouched; `data/policies/thriller/`,
`data/dances/`, `deploy/bundles/` untouched; cloud v2 files (`sim2real_task.py`,
`sim_gap_check.py`, `autopilot_s2r.py`, …) unedited — all v3 work in NEW files;
nothing committed to git.

---

## 1. SYSTEM-ID (done first; informs V3B)

Tools: `deploy/capture_stage0.py --analyze` (delivery ratios) +
`tools/system_id_arms.py` (new; lag/amplitude/friction from q-vs-target
dynamics). Inputs: the three full-dance ground runs
`data/telemetry/20260706-11{4445,5004,5456}_ground-run-legodom.npz`.
Findings: `data/reports/system_id_20260706.json`.

| group           | plant lag vs command | amp ratio | Coulomb fit (Nm) | stiction med (Nm) | delivery |
|-----------------|---------------------|-----------|------------------|-------------------|----------|
| shoulder (x6)   | 114 mean / 141 max ms | 0.83-0.98 | 0.08-0.53        | 0.61-2.76         | 0.76-0.98 |
| elbow (x2)      | 101 ms              | 0.90-0.91 | 0.26-0.38        | ~0.88             | 0.78-0.86 |
| wrist_roll (x2) | 81 ms               | 0.96-0.98 | 0.21-0.23        | ~0.18             | **0.24-0.37** |
| wrist_p/y (x4)  | 81-101 ms           | 0.98-0.99 | 0.16-0.21        | 0.21-0.35         | 0.59-0.90 |

Reading: at trained arm gains (kp 14.25 shoulder/elbow/wrist_roll, 16.78
wrist_p/y) the PD command on the wrists (~0.5 Nm RMS) sits AT the measured
friction floor — wrist_roll delivers only a quarter of its command. Shoulders
carry gravity load and under-swing 8-17 %. Legs/waist deliver 0.83-0.97
(healthy — no drivetrain deficit; the knee 0.70 on run 3 is R²-limited).
Sensing is NOT the lag (obs staleness p95 1.78 ms, measured 2026-07-05):
this is plant + soft-gain lag. Teleop on THIS robot drives the same motors
at kp 80 (shoulder/elbow) / 40 (wrist) — the proven envelope.

**Cross-check that reframed the metric:** s2r-b's SIM arm tracking on the same
joint-space metric is **13.81 deg RMS** (`reports/arm_tracking_s2rb_baseline.json`,
box) — i.e. by |q − reference| the sim policy is about as sloppy as the
hardware (13.2). The visible crispness deficit is therefore substantially IN
the policy (soft gains + smoothness-leaning rewards), not only in the plant.
This strengthens the precision variants; hardware still adds lag/friction on
top (the 100-160 ms wrist lag is real), which V3B + the deploy boost address.

## 2. THE VARIANTS (`cloud/sim2real_task_v3.py`, new; wrapper `cloud/train_sim2real_v3.py`)

All start from the v2 s2r recipe (torque penalties, all DR, legodom obs
model, 20 s episodes) and keep obs 160-dim. Registered task ids:

- **V3A "precision"** `Mjlab-Tracking-Flat-Unitree-G1-S2R-V3A`
  action_rate_l2 −0.2 → −0.1 (stock; the −0.2 was attempt-2's anti-jerk delta),
  motion_body_pos & motion_body_ori 1.0 → 1.5. Rationale: buy tracking
  sharpness back from smoothness; keep everything that fixed torque/drift.
- **V3B "arm-plant realism"** `…-S2R-V3B`
  = V3A + (a) arm frictionloss DR **0.1-0.6 Nm abs** (measured Coulomb
  0.08-0.53 + stiction headroom; body joints keep 0-0.4; the v2 single
  friction event is split into two DISJOINT events so ordering can't matter),
  (b) **arm actuator stiffness AND damping ×2.5** (G1_ACTUATOR_5020 +
  G1_ACTUATOR_4010 cfgs = all 14 arm joints = the four deploy gain groups).
  Why 2.5: lag ∝ 1/√kp → shoulder 114-141 ms → ~75-90 ms; PD authority 2.5×
  over the friction floor; resulting kp 35.6/41.9 stays inside the
  teleop-proven 80/40; damping by the same factor → ζ ×√2.5, strictly more
  overdamped. **DEPLOY CONTRACT: a V3B policy must be deployed with
  ARM_GROUND_KP_SCALE=2.5** (see §4) or a policy_meta with arm kp/kd ×2.5.
- **V3C "converge longer"** `…-S2R-V3C` = V3A env, 10000 iterations.
- **V3D "sharp reference"** (added mid-program by main session) = V3A recipe
  trained on `motions/thriller_deploy_v2_sharp.npz` — the retarget-fidelity
  fix: prep_motion's blanket 8.48 rad/s velocity clamp had blunted 58 dance
  accents 60-85 % (true motor limits 20-37 rad/s; docs/retarget_fidelity.md).
  Sharp CSV pushed and converted on-box with mjlab csv_to_npz (30→50 fps):
  same 2589 frames / 51.8 s; max |joint_vel| 31.4 vs 8.48 rad/s (p99
  unchanged 5.3 — the restoration is localized accents). v3d tests the
  ceiling with the un-blunted reference; front-end owned the "soft hits".
- **Eval-only** `…-S2R-V3B-GAPEVAL`: STOCK task + ×2.5 arm gains — V3B
  candidates are gap-checked/rendered on the plant the deploy contract gives
  them, while staying comparable to the a2/s2r baselines.

Smoke-tested on the box before launch (`cloud/smoke_v3.py`): arm gains ×2.5
land on exactly the 2 arm actuator cfgs (35.63/2.268, 41.95/2.670), stock task
uncontaminated, reward weights as designed, friction DR split disjoint, obs
(8,160), 30-step env runs clean for V3A / V3B / GAPEVAL.

## 3. LAUNCHES (box `/workspace/notebook-data`, via `cloud/run_job.sh`)

| job | task / motion | iters | started (UTC) | ETA (shared GPU) |
|-----|---------------|-------|----------------|------------------|
| `train-thriller-v3a` | V3A / thriller_deploy.npz | 5000 | 05:50 | ~4.5 h |
| `train-thriller-v3b` | V3B / thriller_deploy.npz | 5000 | 05:50 | ~4.5 h |
| `train-thriller-v3c` | V3C / thriller_deploy.npz | 10000 | 05:50 | ~9 h |
| `train-thriller-v3d` | V3A / thriller_deploy_v2_sharp.npz | 5000 | 06:01 | ~5-6 h |
| `arm-baseline-s2rb` | s2r-b model_4999 vs old ref | — | done 05:53 | ARM_RMS 13.81 deg |
| `arm-baseline-s2rb-sharp` | s2r-b model_4999 vs sharp ref | — | 06:01 | ~15 min |
| `v3{a,b,c,d}-autopilot` | wait → export → gate → arm metric → render | — | 05:52/06:01 | tails the trainings |

Exact train command (per variant):
```
bash cloud/run_job.sh start train-thriller-v3<x> -- \
 "cd /workspace/notebook-data && MUJOCO_GL=egl WANDB_API_KEY=$(cat .wandb_key) \
  ./envs/mjlab/bin/python cloud/train_sim2real_v3.py <TASK_ID> \
  --env.commands.motion.motion-file /workspace/notebook-data/motions/<MOTION>.npz \
  --env.scene.num-envs 4096 --agent.max-iterations <5000|10000> \
  --agent.run-name train-thriller-v3<x> --video False"
```
Observed iteration times with 3 jobs: 2.8-3.1 s/it; 4-job number checked after
v3d warm-up (bar: stagger v3d if >4 s/it — see §6 status notes).
Budget: 4 trainings ≈ 25-45k VND each + box 18k/h, per standing user order.

## 4. AUTOPILOT + METRICS (`cloud/autopilot_v3.py`, per variant)

When `train-thriller-v3<x>` is done: for the last (and, if the gate fails,
mid) checkpoint —
1. `cloud/export_policy.py` → `exports/thriller_v3<x>/<tag>/policy.onnx`
2. `cloud/sim_gap_check_v3.py` (wrapper registering v3 tasks; gate v3 itself
   lives UNEDITED in `cloud/sim_gap_check.py`) — 128 envs, full motion;
   V3B on `…-V3B-GAPEVAL`; **v3d gated against the SHARP npz**
3. `cloud/arm_tracking_eval.py` (new) — the JOINT-SPACE ARM METRIC:
   per-joint RMS/p95 of |q − reference| (deg), arm-group rollup, nominal
   conditions, 64 envs. Must BEAT the s2r-b baseline:
   `reports/arm_tracking_s2rb_baseline.json` = **13.81 deg** (old ref);
   v3d compares against `reports/arm_tracking_s2rb_baseline_sharp.json`.
4. `cloud/headless_render_v3.py` — full-dance mp4 on the deploy-matched task.
5. Writes `exports/thriller_v3<x>/RESULT.txt`:
   VERDICT = WIN (gate pass + arm RMS beats baseline) |
   GATE_PASS_ARM_MISS | GATE_FAIL — plus the V3B deploy-contract note.

## 5. DEPLOY-SIDE PROTOTYPE (repo, offline — the only deploy_runtime change)

`ARM_GROUND_KP_SCALE` (default 1.0) in `pipeline/deploy_runtime.py`,
**mode ground-run-legodom only**: scales kp AND kd of the 14 arm joints,
identified BY NAME from meta.joint_order. kd uses the MATCHING factor
(not sqrt): ζ = kd/(2√(kp·J)) then rises ×√S — never less damped than
trained, and it reproduces V3B's train-time actuator scaling exactly, so one
knob serves the boost experiment on s2r-b AND a future V3B deploy (2.5
required there). Refuses S outside [1.0, 3.0] (above ~3× the wrist kp exits
the teleop-proven envelope). Loud `***` print when active; recorded in
telemetry `run_meta` (`arm_ground_kp_scale`); `capture_stage0._effective_gains`
extended so delivery analysis regresses against the gains actually commanded.
Tests: `tests/test_arm_gain_boost.py` (8 tests — name-resolution, arms-only
scaling, leg-boost composition, refusal bounds, env-knob path, run_meta
provenance, effective-gains correction). Full suite: **304 passed, 3 skipped**.

Suggested first hardware use (needs user + remote): one tethered
ground-run-legodom of the CURRENT s2r-b with `ARM_GROUND_KP_SCALE=1.5`, then
2.0/2.5 if clean — telemetry auto-records; re-run `tools/system_id_arms.py`
on the new npz and compare lag/amp/RMS to the 20260706 rows.

## 6. DECISION MATRIX (fill from RESULT.txt files)

| criterion | v3a | v3b | v3c | v3d | s2r-b baseline |
|---|---|---|---|---|---|
| sim arm RMS (deg, beat 13.81 / sharp-baseline for v3d) | | | | | 13.81 |
| gate v3 pass (survival/ankle/rr_mpkpe/drift) | | | | | 5/9→promoted |
| render visual (arms crisp, accents land) | | | | | reference video |
| deploy complexity | none | ARM_GROUND_KP_SCALE=2.5 required | none | new reference CSV to stage | — |

Winner rule: highest arm-RMS improvement that PASSES the gate and looks
right in the render. If v3b wins, its hardware test REQUIRES the ×2.5 knob;
if v3d wins, the sharp CSV/npz must be staged as the dance's motion artifacts
(same promotion machinery — motion sha changes!). Combinations (e.g. v3b
recipe on the sharp reference) are a follow-up training, not assumed.

## 7. STATUS NOTES (append-only)

- 05:50 UTC: v3a/b/c launched, 2.7-2.9 s/it each, GPU 6.1 GB / 99 %.
- 05:52: autopilots a/b/c + old-ref arm baseline launched; baseline done
  05:53: **ARM_RMS 13.81 deg** (wrist_yaw worst 21.5 — action_scale 0.0745
  gives the policy little wrist-yaw authority; note for a possible v4).
- 06:00: sharp CSV converted (2589 frames, accents 31.4 rad/s max vel).
- 06:01: v3d + sharp baseline + v3d autopilot launched.
- 06:02: sharp baseline done: **s2r-b vs SHARP reference ARM_RMS 15.18 deg**
  (vs 13.81 on the old ref — s2r-b never saw the accents; this is v3d's bar).
- 06:05: 4-way sharing measured: 3.5-3.7 s/it (under the 4 s/it stagger bar —
  v3d stays concurrent). GPU 8.2/24.5 GB. ETAs: v3a/b ~10:00, v3d ~11:45,
  v3c ~14:15 UTC. All four autopilots confirmed polling with the right
  eval task (v3b on …-V3B-GAPEVAL) and motion (v3d on the sharp npz).
- Results will appear at `exports/thriller_v3{a,b,c,d}/RESULT.txt` (+
  `<tag>/policy.onnx`, `<tag>/gap_check.json`, `<tag>/arm_tracking.json`,
  `rollout_v3<x>.mp4`) on the box. Nothing auto-stages into deploy dirs.
