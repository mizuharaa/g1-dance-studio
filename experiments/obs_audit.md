# OBSERVATION-CONTRACT AUDIT — v8 (154-dim no-state-estimation actor)

**Date:** 2026-07-16 · **Auditor:** obs-contract auditor (read-only on code; NO GPU; no running sim/robot)
**Scope:** every actor observation term the v8 policy will consume, cross-checked TRAIN (mjlab) vs DEPLOY
(`pipeline/deploy_runtime.py`). This report GATES the v8 training run.
**Verdict up front:** the v8 **actor observation contract is internally correct and self-consistent — training
is SAFE to launch.** Every must-fix item is DEPLOY-side (robot is down; deploy is an explicit later wave) and
every one of them **fails LOUD or SAFE**, not silently. No silent poison found on the training path.

## Provenance / how each fact was established
- TRAIN obs: `cloud/sim2real_task_v8.py` (read in full) → `sim2real_task_v7.py` → `sim2real_task.py` (base, read
  in full) → `third_party/mjlab_mdp_ref/{tracking_env_cfg.py, g1_config/env_cfgs.py, g1_config/rl_cfg.py,
  mdp/observations.py, mdp/commands.py}` (all read in full). v5/v6 deltas were NOT re-read line-by-line; the v8
  drop runs LAST and is guarded, and `v8 --selfcheck` independently asserts the split + 154 dim.
- DEPLOY obs: `pipeline/deploy_runtime.py` — `OBS_LAYOUT`, `GROUND_OBS_LAYOUT`, `TERM_WIDTHS`,
  `ESTIMATOR_DEPENDENT_TERMS`, `build_obs`, `build_obs_ground`, `_ground_obs_order`, `Meta`, `Reference`,
  `read_state`, `action_to_target`, the `mode_run` loop.
- Normalization: **inspected the actual exported ONNX graph** (`data/policies/thriller/policy.onnx`) with the
  `onnx` module — decisive, not inferred.
- Cross-checks: `data/policies/thriller/policy_meta.json` (joint order / gains / scales), `experiments/
  upstream_alignment_report.md` (Agent 0), `pipeline/g1_limits.py`.

---

## v8 ACTOR contract (the 154-dim vector, in concatenation order)

mjlab concatenates actor terms in dict-insertion order. Base order is
`[command, motion_anchor_pos_b, motion_anchor_ori_b, base_lin_vel, base_ang_vel, joint_pos, joint_vel, actions]`;
v8 `_drop_privileged_actor_terms` does `del actor.terms[...]` for the two privileged terms (runs LAST, after all
v5–v7 wiring), leaving the order below. This is **byte-for-byte** `GROUND_OBS_LAYOUT` in deploy_runtime.

| # | term | dims | units | frame | TRAIN source | DEPLOY source | MATCH |
|---|---|---|---|---|---|---|---|
| 1 | command | 58 | rad + rad/s | ref joint space (abs) | `mdp.generated_commands("motion")` = `MotionCommand.command` = cat(joint_pos[29], joint_vel[29]) of the motion npz at `time_steps` | `np.concatenate([ref_jp, ref_jv])` from `Reference.at(tick)` (npz `joint_pos`,`joint_vel`) | ✅ |
| 2 | motion_anchor_ori_b | 6 | dimensionless (2 cols of R) | **torso_link** (anchor body) rel. to robot | `mdp.motion_anchor_ori_b` = first 2 cols of `R_robotTorsoᵀ·R_refTorso` | `mat_first_two_cols_b(ref_aquat, anchor_q)`; `anchor_q` = pelvis IMU quat ∘ waist-joint FK (→ torso); `ref_aquat` = npz torso quat, yaw-aligned | ✅ |
| 3 | base_ang_vel | 3 | rad/s | **pelvis** (imu_in_pelvis gyro) | `mdp.builtin_sensor("robot/imu_ang_vel")` — gyro at the pelvis IMU site | raw `msg.imu_state.gyroscope` (physical pelvis IMU) | ✅ |
| 4 | joint_pos | 29 | rad | joint space, **rel. to default** | `mdp.joint_pos_rel` (q − default) | `q − meta.default` | ✅ |
| 5 | joint_vel | 29 | rad/s | joint space | `mdp.joint_vel_rel` (default vel = 0 → = dq) | `dq` | ✅ |
| 6 | actions | 29 | unitless policy action (t−1) | — | `mdp.last_action` | `last_action` (prev tick; init 0) | ✅ |

**Joint order (all 29 terms that are joint-indexed):** `meta.joint_order_29dof` = the standard G1 order
(L-leg 0–5, R-leg 6–11, waist 12–14, L-arm 15–21, R-arm 22–28). Deploy reads `motor_state[0..28]` **positionally**
and treats index i as `joint_order[i]`; the ankle-reward / action-cap code resolves joints BY NAME. Upstream
`deploy.yaml` `joint_ids_map` is sequential `[0..28]` (RL index == motor id, no remap) — Agent 0 confirmed this
matches our meta exactly. **MATCH (inferred from meta + upstream agreement; see Risk R4 — not bit-verified against
a live mjlab model here, no GPU).**

### The two DROPPED terms (were the prime sim2real suspects — now eliminated from the actor)

| term | dims | why it was a landmine | v8 status | evidence |
|---|---|---|---|---|
| base_lin_vel | 3 | MuJoCo **velocimeter at site imu_in_pelvis (PELVIS frame)** — includes the ω×r lever-arm, NOT the torso base frame; and it is **unmeasurable on the real robot** (needs a state estimator). Deploy faked it (zeros on gantry / leg-odometry on ground). | **REMOVED from actor** (kept critic-only, fed clean sim truth). Deployed actor never sees it → nothing to fake, lever-arm frame question is **moot**. | v8 `PRIVILEGED_ACTOR_TERMS`; `env_cfgs.py has_state_estimation=False` drops the same two; selfcheck asserts actor-DROPPED + critic-KEPT |
| motion_anchor_pos_b | 3 | torso XY/Z position error vs reference — also needs a base-position estimate, equally unmeasurable. | **REMOVED from actor** (critic-only). | same |

Removing these is the single biggest structural correctness win in v8: it deletes a built-in train/deploy
divergence on the actor's own input vector.

---

## NORMALIZATION — resolved decisively (was the top unlisted risk)

`rl_cfg.py` sets `obs_normalization=True` for both actor and critic → rsl_rl EmpiricalNormalization. `policy_meta.json`
carries **no** mean/var and deploy applies **no** external normalization — so correctness hinges entirely on whether
the normalizer is baked into the ONNX. **It is.** Inspecting `data/policies/thriller/policy.onnx` shows, immediately
after the `obs` input and before the first Gemm:

```
Sub  in=[obs, policy.obs_normalizer._mean]         -> Sub_output
Div  in=[Sub_output, onnx::Div_47 (std)]           -> Div_output
Gemm in=[Div_output, policy.mlp.0.weight, ...]     (first MLP layer)
```

So the running mean/std are frozen initializers **inside the graph**. The gate and deploy both run the SAME ONNX on
**raw** obs and get identical normalization. **MATCH — normalization is applied identically train vs deploy, and the
running mean/var ARE exported with the policy.** (This is why deploy correctly applies none itself.)

---

## TIMING — 50 Hz, phase, one-step delay

| aspect | TRAIN | DEPLOY | MATCH |
|---|---|---|---|
| control rate | sim_dt 0.005 × decimation 4 = **50 Hz** | `CONTROL_HZ`=50, absolute-deadline clock, 2·dt watchdog | ✅ |
| `actions` term | `last_action` = action applied at t−1 | `last_action`, updated AFTER `run_policy` each tick (init 0) | ✅ |
| reference phase | `MotionCommand.time_steps` starts 0, +1 per control step | `ref.at(tick)`, tick 0..N, +1 per loop; `time_step` also fed to ONNX for the baked motion tensors | ✅ by construction (see R5) |
| training latency DR | cmd-bus delay 0–80 ms + obs delay 0–80 ms on measured terms (base `_apply_sim2real`); play/export = no delay | real actuation+leg-odom latency 40–80 ms (measured) is inside the trained band | ✅ (DR covers deploy latency) |

---

## PRIORITIZED FINDINGS

### Does anything have to change in the v8 recipe BEFORE we train?  → **NO.**
The v8 actor obs contract (order, units, frames, joint order, dropped privileged terms, 154 dim) is correct and
self-consistent, the normalizer bakes into the export, and `v8 --selfcheck` guards the asymmetric split on the
actual installed mjlab. **Training can be launched.** All items below are DEPLOY-wave (robot is down) and fail
loud/safe — none silently poisons training.

**R1 — HIGHEST RISK (deploy-wave, must-fix before any v8 robot deploy; NOT training-gating).**
The export meta template `docs/mjlab_policy_interface.json` STILL declares the OLD 160-dim / 8-term contract
(`actor_obs_terms_in_order` lists all 8 incl. base_lin_vel + motion_anchor_pos_b; `onnx_inputs.obs = [1,160]`).
`pipeline/stages/cloud_motion.py:573-583` writes each policy's `policy_meta.json` by copying this template verbatim
(only task/run fields are updated — the obs contract is NOT derived from the trained task). So the v8 policy_meta.json
will **misdescribe** its own 154-dim ONNX. Fails SAFE, not silent: `build_obs` (160) → ONNX shape mismatch (loud);
`_ground_obs_order` reads `meta.obs_terms`, sees `base_lin_vel` ∈ `ESTIMATOR_DEPENDENT_TERMS` → hard `SystemExit`
REFUSE. **Fix before v8 deploy:** update the template (or make the exporter emit the real actor term list) to the
6-term / `[1,154]` contract so the v8 meta's `actor_obs_terms_in_order` == `GROUND_OBS_LAYOUT`.

**R2 — MEDIUM, UNVERIFIED (verify at v8 export).** The 154-dim ONNX must re-bake ITS OWN normalizer.
`obs_normalization=True` is inherited (rl_cfg), so the machinery is on, but I could only inspect the existing 160-dim
ONNX. **Owed cross-check:** after v8 export, re-run the graph inspection and assert `policy.obs_normalizer._mean`
has length **154** and the `obs` input is `[1,154]`. Cheap, decisive; do it before trusting any v8 gate number.

**R3 — NOTE, intentional (not a bug).** Ankle `effort_limit_nm` = 50 Nm in meta and at deploy; v8 clamps the SIM
ankle to 40 Nm (train-only) and widens ankle effort DR down to 26–38 Nm (`ANKLE_EFFORT_DR`). The deployed policy
therefore sees MORE ankle authority than it trained with — the SAFE direction (velocity-honest actuator finding,
Agent 0/D). Flagging so it is not later "corrected" into a train/deploy symmetry that re-optimism the ankle.

**R4 — LOW, INFERRED (no GPU).** Joint order match rests on `policy_meta` + upstream `deploy.yaml` agreement, not on a
live read of the mjlab model's joint order on this box. **Recommend** the box GPU-smoke step assert
`asset.find_joints(...)` order == `meta.joint_order_29dof` so the ONE positional assumption is machine-verified once.

**R5 — LOW, INFERRED.** Reference-phase alignment (obs at t vs. reference frame at t) is consistent by construction
(both start at 0, +1/step) but was not replay-verified against a running sim. The owed "trust gate" (replay real/sim
obs step-by-step, HANDOFF §3.2) would close this and R4 together.

---

## Summary
**Terms audited:** 8 (6 live in the v8 actor + the 2 correctly dropped). **MATCHES:** all 6 live terms match on
units, frame, joint order, and timing; normalization matches (baked in ONNX, verified from the graph); the 2
unmeasurable/lever-arm terms are removed from the actor by construction. **MISMATCHES that poison training:** none.
**Highest-risk item:** R1 — the stale 160-dim export-meta template will mislabel the v8 policy (deploy-wave; fails
safe). **Must change before we TRAIN:** nothing. **Must change before we DEPLOY v8:** R1 (meta template → 154/6-term)
and R2 (verify the v8 ONNX bakes a 154-length normalizer). Verified from code + the actual ONNX graph; R4/R5 are the
only items that could not be machine-confirmed without a running sim/robot (no GPU).
