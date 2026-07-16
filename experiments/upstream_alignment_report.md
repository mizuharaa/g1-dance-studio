# UPSTREAM ALIGNMENT AUDIT (Agent 0) — report

**Date:** 2026-07-16 · **Author:** Agent 0 (upstream-alignment auditor) · **Status:** gating report for Agents A/B/D/F
**Scope:** read-only on our code; no GPU. This report compares Unitree's first-party RL repos against
our custom mjlab stack and reshapes the scope of the downstream agents.

## Method / provenance (measurement discipline)

| Repo | How read | Notes |
|---|---|---|
| `unitree_rl_mjlab` | `git clone` **hung** on this sandbox's network (only `.git` populated, 0 files after minutes). Fell back to **WebFetch on raw.githubusercontent.com** + GitHub tree API. | Load-bearing files read: `README.md`, `scripts/csv_to_npz.py`, `deploy/robots/g1/config/policy/mimic/dance1_subject2/params/deploy.yaml`, `setup.py`, `LICENCE`. |
| `unitree_rl_lab` | `git clone --depth 1` **succeeded** (160 files). Read source files **directly on disk**. | This is the **independent second engine** — used to cross-check the mjlab obs/reward contract. Two-source agreement noted throughout. |
| `unitree_sdk2_python` | `git clone` succeeded. | Confirms G1 uses the `unitree_hg` IDL (`unitree_sdk2py/idl/unitree_hg/`), distinct from `unitree_go` (Go2/H1). Deploy-time only; robot is down. |

Every numeric claim below is backed by a file+line in one of the two engines, or by a value already in our
`policy_meta.json`. Where a single WebFetch summarizer editorialized (e.g. it mislabeled the G1 as a
"quadruped"), I ignored the prose and kept only numbers that a second source or our own meta confirms.

---

## HEADLINE: the stale assumption is CONFIRMED, and it points at a real architectural defect

1. **Unitree ships first-party mimic RL on our exact engine.** `unitree_rl_mjlab` trains a G1-29dof
   BeyondMimic dance-mimic task on the **same mjlab/MuJoCo-Warp framework we use**. Our
   `cloud/sim2real_task*.py` already `import`s the upstream task straight out of the `mjlab` library
   (`from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg`). We were never
   "building an RL stack" — we were **layering deltas on the upstream mjlab tracking task** and did not
   track that the upstream task evolved.

2. **The upstream task is now named `Unitree-G1-Tracking-No-State-Estimation`** (plus a 23-dof variant),
   and its actor observation **deliberately excludes `base_lin_vel` and `motion_anchor_pos_b`** — the two
   terms we flagged as unmeasurable on the real robot (§3.4). Our task
   (`Mjlab-Tracking-Flat-Unitree-G1`, 160-dim actor) reflects an **older, full-observation actor** that
   was refactored upstream into the no-state-estimation design. This is the single most important finding.

3. **The two "unmeasurable" terms are, by upstream design, PRIVILEGED CRITIC-ONLY observations.** Confirmed
   independently in the Isaac engine (`unitree_rl_lab .../dance_102/tracking_env_cfg.py`,
   `ObservationsCfg`): `base_lin_vel` and `motion_anchor_pos_b` live in the `PrivilegedCfg` **critic** group,
   never in the `PolicyCfg` **actor** group. This is textbook asymmetric actor-critic: the critic sees full
   sim truth during training; the deployed actor sees only proprioception + command. **We put privileged
   critic terms into the actor**, then built an entire leg-odometry estimator + sim2real DR machinery to
   feed the actor a quantity upstream's design proves it never needs.

**Consequence:** our "state-estimation hole" is not a hole to patch — it is a symptom of diverging from the
upstream asymmetric design. This also plausibly feeds the trust problem (§3.2): our actor consumes a
noisy, hard-to-reproduce estimate (`base_lin_vel`) that the gate supplies from clean sim truth but that
deploy must fake — a built-in sim/real divergence on the actor's own input vector.

---

## 2. OBSERVATION CONTRACT reconciliation (prime suspect for the trust problem)

**Ours (actor, 160-dim)** — from `policy_meta.json` + `pipeline/deploy_runtime.py:210`:

| term | dims |
|---|---|
| command | 58 |
| motion_anchor_pos_b | 3 |
| motion_anchor_ori_b | 6 |
| base_lin_vel | 3 |
| base_ang_vel | 3 |
| joint_pos | 29 |
| joint_vel | 29 |
| actions | 29 |
| **total** | **160** |

**Upstream (actor / "No-State-Estimation", 155-dim)** — mjlab `deploy.yaml` obs list, cross-confirmed by
Isaac `PolicyCfg` (`tracking_env_cfg.py:142-149`):

| term | dims |
|---|---|
| motion_command | 59 |
| motion_anchor_ori_b | 6 |
| base_ang_vel | 3 |
| joint_pos_rel | 29 |
| joint_vel_rel | 29 |
| last_action | 29 |
| **total** | **155** |

**Differences (flag every one):**
- **`base_lin_vel` (3): DROPPED from the actor upstream.** It is critic-only. This is the observability leak.
- **`motion_anchor_pos_b` (3): DROPPED from the actor upstream.** Also critic-only. (It is the reference-vs-robot
  XY/Z position error — needs a base-position estimate, equally unmeasurable.)
- **`command` 58 (ours) vs `motion_command` 59 (upstream): off by one.** Upstream carries one extra command
  element (likely a phase/time-to-target scalar). Net: 160 − 6 (dropped terms) + 1 (command) = 155. Reconciled.
- Term **ordering** differs (upstream: command, anchor_ori, ang_vel, joint_pos, joint_vel, action). Order is
  load-bearing for a concatenated obs — must be matched exactly if we adopt.
- Upstream uses `joint_pos_rel`/`joint_vel_rel` (relative to default) with explicit obs noise
  (`Unoise` ±0.01 pos, ±0.5 vel, ±0.2 ang_vel, ±0.05 anchor_ori). Our meta names them `joint_pos`/`joint_vel`;
  confirm ours are also default-relative (they should be — same mjlab base).

**Note — our deploy runtime already anticipated this.** `pipeline/deploy_runtime.py` defines
`ESTIMATOR_DEPENDENT_TERMS = {"base_lin_vel", "motion_anchor_pos_b"}` and a `build_obs_ground()` that drops
exactly those two terms (a ~154-dim reduced vector, line 261). **But no policy was ever trained on that
reduced obs** — so feeding it to our 160-dim policy is a shape/space mismatch. Upstream ships the *trained*
counterpart. The fix is to make our **training** obs match upstream's actor group, not to hack deploy.

---

## 3. NO-STATE-ESTIMATION VARIANT — **YES, it exists. ADOPT it.** (HIGH PRIORITY)

- **Exists:** yes. `Unitree-G1-Tracking-No-State-Estimation` is the **default/headline** mimic task in
  `unitree_rl_mjlab` (README train command:
  `python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=...dance1_subject2.npz --env.scene.num-envs=4096`).
  A `Unitree-G1-23Dof-Tracking-No-State-Estimation` variant also exists.
- **Does adopting it close our leak?** **Yes, by construction.** The actor never observes `base_lin_vel` or
  `motion_anchor_pos_b`, so there is nothing to estimate on the robot. The critic still gets them from sim
  truth during training (asymmetric AC), so tracking quality is preserved.
- **Recommended adoption path (Agent D/F):** do **not** blindly `pip install unitree_rl_mjlab` (version
  conflict — see §6). Instead **port the actor observation group** in our existing mjlab-1.5.0 task cfg:
  remove `base_lin_vel` + `motion_anchor_pos_b` from the actor, keep them in the critic/privileged group,
  match term order and the +1 command dim. This is a small config edit on the engine we already trust, and
  it **retrains** a genuinely deployable policy. Then re-export with the 155-dim contract.

---

## 4. RETARGETING / MOTION AUTHORING — **NOT inheritable from these repos.**

- `unitree_rl_mjlab/scripts/csv_to_npz.py` and `unitree_rl_lab/scripts/mimic/csv_to_npz.py` both expect a CSV
  **already in G1 joint space**: columns `[base_pos_xyz(0:3), base_quat_xyzw(3:7), dof_positions(7:)]`
  (verified: `csv_to_npz.py:121-124` slices exactly that; xyzw→wxyz reorder at line 123). Sample
  `G1_Take_102.bvh_60hz.csv` first row confirms: base pos ~(0.009, 0.001, 0.78), a unit quaternion, then 29 dof.
- The tool does **pure kinematic replay + resample** (lerp positions, slerp quats, `--input_fps`→`--output_fps`;
  it calls `sim.render()` **not** `sim.step()` — no physics), then logs forward-kinematics body poses/vels to
  npz fields `fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w`. **This is
  exactly the npz schema our `sim_gap_check._motion_duration_s` and mjlab expect.** So npz *packaging* is a
  match, but there is **no human→G1 retargeting and no feasibility/torque vetting** anywhere in these repos.
- The `.bvh_60hz.csv` naming implies the motion was retargeted from BVH mocap by an **external** BeyondMimic
  tool; the README defers to "BeyondMimic documentation" for preprocessing. **dance1_subject2.csv arrives clean.**

**Implication for Agent B:** the entire human-video→G1 front end (GVHMR pose-est, GMR retarget, `grounding.py`,
`prep_motion.py`, `vet_motion.py`, `motion_feasibility.py`) and the **"degrade gracefully to G1 torque limits"**
philosophy have **zero upstream implementation to inherit**. That work stays 100% ours (or must be pulled from
BeyondMimic proper, not from these Unitree repos). What IS inheritable is the **final npz contract + the
30→50 fps `csv_to_npz` convention**, which we already match.

---

## 5. KP/KD, EFFORT, JOINT MAP — **already inherited; matches ours. Keep.**

- mjlab `dance1_subject2/params/deploy.yaml`: stiffness 14.3–99.1, damping 0.9–6.3, per-joint action_scale
  0.07–0.55, `step_dt 0.02` (50 Hz), `joint_ids_map` = **sequential [0..28]** (RL output index == physical
  motor id; no remap). These are **identical** to our `policy_meta.json` (kp 14.251–99.098, kd 0.9072–6.3088,
  action_scale 0.0745–0.5475, effort 5–139 Nm).
- Why they match exactly: our gains are **derived** from the upstream armature via the mjlab impedance model
  `kp = armature·(2π·10)²`, `kd = 2·ζ·armature·(2π·10)`, ζ=2. Cross-checked against Isaac
  `unitree_actuators.py` armatures: N7520_14p3 armature 0.01018 → kp 40.18 ✓; N7520_22p5 0.02510 → 99.1 ✓;
  N5020_16 0.003610 → 14.25 ✓. Our `policy_meta` already states "These SIM gains ARE the deploy gains
  (BeyondMimic)". **Confirmed correct — no change needed. Agent D can trust the existing kp/kd/effort/joint map.**
- **Fidelity gap worth flagging (Agent D/F):** Isaac's `UnitreeActuator` models a **torque–speed (T–N) curve**
  (`Y1/Y2/X1/X2` knee-point derating) **plus joint friction (`Fs/Fd`)** — available torque **falls as joint
  speed rises**. Our mjlab setup uses a **flat effort clamp** (ankle 50 Nm). At the two failure beats
  (13–18 s, 25–36 s) the ankles are simultaneously **fast and high-torque**, so the *real* available torque is
  **below** the flat 50 Nm the sim believes. This is a strong candidate mechanism for both the ankle
  saturation wall AND a sim-optimistic gate. Recommend Agent D/F evaluate adopting the T–N-curve actuator
  model (or at least a velocity-derated effort limit) — it may reframe the "ankle wall" as a sim-fidelity
  artifact rather than a pure choreography limit.

---

## 6. LICENSE / VERSION COMPAT

- **License: OK.** Both repos are **Apache-2.0** (`unitree_rl_mjlab/LICENCE`; `unitree_rl_lab` README badge +
  `LICENCE`). Permits use/modification/derivative works with attribution + modified-file notices. No blocker.
- **Version conflict (Agent F must resolve):** `unitree_rl_mjlab/setup.py` pins **`mjlab==1.2.0`,
  `mujoco-warp==3.5.0`**. Our known-good lock (`cloud/env_lock/requirements.lock.txt`) is **`mjlab==1.5.0`,
  `mujoco-warp==3.10.0.1`, `warp-lang==1.14.0`, `torch==2.11.0+cu128`**. Installing `unitree_rl_mjlab`
  wholesale would **downgrade** mjlab 1.5.0→1.2.0 and mujoco-warp 3.10.0.1→3.5.0 — and our lock was pinned
  precisely because unpinned installs CUDA-crash at env reset. **Do NOT adopt upstream's package/pins.**
  Adopt the **design** (no-state-estimation actor obs group) into our existing 1.5.0 task instead — it's a
  config-level change that is engine-version-neutral.

---

## INHERIT-vs-KEEP table

| FEATURE | ours | upstream | verdict | why |
|---|---|---|---|---|
| RL engine | mjlab 1.5.0 + custom task | mjlab 1.2.0 (`unitree_rl_mjlab`) | **keep ours (engine)** | we already run the upstream mjlab tracking task as the base; our lock is newer & stable |
| Actor obs contract | 160-dim, **includes** base_lin_vel + motion_anchor_pos_b | 155-dim **No-State-Estimation** (drops both; critic-only) | **INHERIT (port design)** | closes the observability leak by construction; prime trust suspect |
| base_lin_vel / motion_anchor_pos_b | in actor + leg-odom estimator + DR to fake them | privileged **critic-only** | **INHERIT design; DELETE our estimator path** | upstream proves the actor never needs them |
| Motion npz schema + csv_to_npz | ours matches | `[base_pos,quat_xyzw,dof]`→npz, 30→50 fps | **already aligned; keep** | same fields/fps convention |
| Human→G1 retargeting | GVHMR→GMR + grounding/prep/vet | **none** (expects clean G1 CSV) | **KEEP OURS (not inheritable)** | upstream has no retargeter or feasibility vetting |
| "Degrade to G1 limits" motion surgery | ours (Agent B/D philosophy) | **none** | **KEEP OURS** | no upstream analog |
| kp/kd/effort/action_scale/joint_map | policy_meta.json | mjlab deploy.yaml (identical) | **already inherited; keep** | derived from same armatures/impedance model |
| Actuator torque model | flat effort clamp | **T–N curve + friction** (Isaac) | **EVALUATE inheriting** | may explain ankle wall + sim optimism |
| Reward: ankle_torque_l2 (custom -1e-3) | ours | not present (only generic joint_torques_l2 -1e-5) | **KEEP OURS** | our answer to the ankle wall |
| Reward: base set (action_rate, joint_torque, motion_* exp terms) | ours (forked from mjlab) | Isaac: action_rate -0.1, joint_torque -1e-5, motion pos/ori/vel exp | **reconcile** | our action_rate -0.25 vs upstream -0.1; note the divergence |
| Termination: XY drift | our `anchor_drift_xy` | **none** (upstream terminates on anchor **Z-height** + orientation only) | **KEEP OURS, but note** | upstream deliberately leaves XY free; our drift battle may be self-imposed |
| Acceptance gate | `cloud/sim_gap_check.py` | **none shipped** | **KEEP OURS** | upstream ships no gate; Agent A calibration still needed |
| Deploy IDL | unitree_hg (correct) | unitree_hg | **keep; correct** | G1 uses hg not go |

---

## PRIORITIZED list of custom files/code we can DELETE or replace

**P0 — replace the actor obs contract (highest leverage; reshapes A/D/F):**
1. Edit the actor `ObservationsCfg` in our mjlab task (`cloud/sim2real_task.py` base + the `mjlab.tasks.tracking`
   config it builds on) to **match upstream No-State-Estimation**: drop `base_lin_vel` + `motion_anchor_pos_b`
   from the actor, keep them in the critic/privileged group. Retrain v8 on the 155-dim actor.

**P0 — delete the machinery that only existed to fake base_lin_vel for the actor:**
2. `legodom_like_base_lin_vel` class in `cloud/sim2real_task.py` (lines ~122-176) + its wiring
   (`cfg.observations["actor"]["base_lin_vel"].func = legodom_like_base_lin_vel`) — **DELETE** once base_lin_vel
   leaves the actor.
3. The obs-delay DR on `base_lin_vel`/`motion_anchor_pos_b` in `DELAYED_OBS_TERMS` — prune those two terms.
4. `pipeline/leg_odometry.py` and deploy's `build_obs_odom()` (deploy_runtime ~630) — the whole honest-odometry
   deploy path becomes **dead** (kept only for reference/reading real obs). `build_obs` (gantry, base_lin_vel=0)
   and the ad-hoc `build_obs_ground()` (154-dim) are **superseded** by the exact 155-dim upstream actor obs.

**P1 — reconcile, don't delete:**
5. `action_rate_l2` weight: we run -0.25 (v7) vs upstream -0.1. Keep ours if evidence-backed but log the divergence.
6. `anchor_drift_xy` termination: keep for the 2 m dance area, but note upstream terminates on Z-height only —
   revisit whether our XY-drift termination is over-constraining the policy (v7 diagnosed drift-vs-survival trade).

**KEEP (genuinely ours, no upstream to inherit):**
- Entire motion front end: `retarget_gvhmr.py`, `grounding.py`, `prep_motion.py`, `vet_motion.py`,
  `motion_feasibility.py`, `motion_quality.py`.
- `cloud/sim_gap_check.py` gate (upstream ships none) + Agent A's calibration.
- `ankle_torque_l2` custom reward + the ankle-wall strategy.

---

## What this changes for the downstream agents

- **Agent A (trust):** the gate feeds the actor `base_lin_vel` from clean sim truth, but deploy cannot — this
  is a concrete, previously-unquantified sim/real divergence **on the actor's input**. Adopting the
  no-state-estimation actor removes this variable entirely. Calibrate the *new* 155-dim policy, not the 160-dim one.
- **Agent B (retarget):** upstream gives you the **npz contract + 30→50 fps convention only**. Retargeting and
  feasibility/graceful-degradation are 100% yours. Target the same `[base_pos, quat, dof]` CSV → csv_to_npz npz.
- **Agent D (deploy/actuator):** kp/kd/effort/joint_map are correct and inherited — trust them. Evaluate the
  Isaac **T–N-curve actuator model** as a possible explanation of the ankle wall. Build deploy obs to the new
  155-dim actor contract (you already have `build_obs_ground` scaffolding).
- **Agent F (training):** do NOT pull upstream's package/pins (mjlab 1.2.0 / mujoco-warp 3.5.0 conflict). Port
  the obs design into our 1.5.0 lock. v8 = first run on the no-state-estimation actor + calibrated gate.
