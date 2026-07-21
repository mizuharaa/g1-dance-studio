# External full-audit prompt (paste into another AI with repo access)

## How to give the auditor (e.g. Codex) access to this repo

The repo is a local git checkout at `/home/alois/g1-dance` (private GitHub remote:
`mizuharaa/unitree_dance_mimic`). Two ways to run the audit:

- **RECOMMENDED — Codex CLI (or any local coding agent), run in the checkout:**
  `cd /home/alois/g1-dance && codex` then paste this file (or say "read and follow
  `docs/EXTERNAL_AUDIT_PROMPT.md`"). It sees the FULL working tree on disk —
  including the vendored `third_party/whole_body_tracking/` and mujoco models that
  are gitignored and therefore NOT on GitHub. To verify claims empirically it can use
  the local conda env: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate
  g1dance` (has mujoco, numpy, scipy, onnxruntime, torch-CPU).
- **Codex cloud / GitHub clone:** FIRST `git push` (local is ahead of origin), then
  point Codex at the GitHub repo. Caveats: `.secrets/` is gitignored (good — keep it
  that way), but the vendored BeyondMimic source and mujoco models under
  `third_party/` are ALSO gitignored, so a clone loses them (`third_party/mjlab_mdp_ref/`
  IS committed and is the most important reference). The repo is ~1.9 GB.

**What the auditor CANNOT and should NOT be given:** the `.secrets/` credentials, SSH
access to the live GPU boxes, and the robot (down). The audit is code + committed
data only. **GPU limitation:** mjlab is not installed locally and there is no GPU, so
NO agent can validate training/GPU-dependent claims by running them — those must be
reasoned from code + `third_party/mjlab_mdp_ref/` and flagged "needs GPU to confirm."

---

You are a senior robotics + reinforcement-learning + software engineer brought in
to perform an **independent, adversarial full audit** of the `~/g1-dance` codebase.
Your job is to **diagnose problems, optimize what exists, and propose improvements**
— not to praise it. You have full read access to the repository. Treat every prior
conclusion in the repo (including the previous AI's own audits and "fixes") as a
CLAIM to be verified against the code and data, not as fact. If something is wrong,
say so and prove it with a file:line citation and a quantified argument.

## 0. What this project is (one screen)

A pipeline + desktop app that turns a **reference dance video** into a controller
that makes a **Unitree G1 humanoid (29 DoF + Inspire hands)** perform that dance —
balanced, push-robust, at paid-show quality. Not open-loop playback: an **RL
whole-body motion-tracking controller** (BeyondMimic / mjlab, trained on a cloud
RTX 4090). End goal is a plug-and-play product for 2–3 minute dances.

**Pipeline:** video → GVHMR (3D pose) → GMR (retarget to G1) → clean / ground /
feasibility-repair → `csv_to_npz` → mjlab tracking training (speed curriculum) →
`policy.onnx` → sim gate (sim2sim) → deploy on the onboard Jetson (currently blocked
— see constraints). The single validated real-world datapoint is the "anchor"
policy `thriller_csv_ankle_penalty`, which performed the Thriller dance live at
~70–80% quality before the robot's power board failed.

**Current status:** ~11 training attempts (v3–v11). v11 is the best sim policy
(native tempo, 98.4% sim survival). A prior 84-agent audit (2026-07-21) found and
fixed 12 defects; a 2-box hyperparameter sweep is running now. The **robot hardware
is down** (burnt DC-DC converter), so everything recent is sim-only.

## 1. Orient yourself first (read in THIS order)

1. `docs/PROJECT_STATE.md` — **the single source of truth.** Mission, full decision
   log (read the last ~15 entries carefully), current phase. Long; skim older
   entries, read 2026-07 in full.
2. `docs/FIELD_GUIDE.txt` — plain-English explainer of the whole system.
3. `CLAUDE.md` — the working rules (safety, measurement discipline, pinned env).
4. `experiments/ml_audit_20260721/REPORT.md` — the previous audit's 12 confirmed
   defects + fixes. **Verify these fixes actually landed and actually work — do not
   trust the report.** Its `PLAUSIBLE.md` is a watchlist of 12 unfixed items.
5. `docs/architecture.md`, `docs/DEPLOY_SAFETY_GUARDS.md`, `docs/BOX_CONSTRAINTS.md`.
6. `logs/jobs.md` — every cloud training run and its outcome.

## 2. Repo map (where things live)

- **Training recipes** (each inherits the previous): `cloud/sim2real_task.py` (base)
  → `_v5.py` (arm fidelity) → `_v6` (drift termination) → `_v7` (ankle penalty) →
  `_v8` (asymmetric obs, 154×5=770 history, ankle barrier/clamp/DR) → `_v10`
  (stance penalties, saturation, speed curriculum) → `_v11` (leg tracking).
  Launchers: `cloud/run_attempt*.sh`, `cloud/train_v*_curriculum.sh`.
- **Eval / gate:** `cloud/sim_gap_check.py` (sim2sim gate), `cloud/pick_checkpoint.py`,
  `cloud/heldout_eval.py`, `cloud/verify_obs_layout.py`.
- **Motion pipeline:** `pipeline/retarget_gvhmr.py`, `pipeline/prep_motion.py`,
  `pipeline/grounding.py`, `pipeline/motion_dynamics.py` (contact-aware feasibility),
  `pipeline/g1_limits.py` (actuator envelope), `pipeline/vet_motion.py`,
  `tools/motion_quality.py` (cleaning), `tools/motion_repair.py` (beat-preserving retime).
- **Deploy runtime:** `pipeline/deploy_runtime.py` (50 Hz onboard loop),
  `pipeline/deploy_guards.py` (safety), `pipeline/leg_odometry.py`.
- **Preview/sim:** `tools/sim_sandbox.py` (policy-in-the-loop), `tools/sim_studio.py`
  (reference vs policy render), `pipeline/sim_preview.py`, `pipeline/publish_policy.py`.
- **App:** `ui/server.py` (FastAPI), `ui/frontend/` (React), `pipeline/monitor.py`,
  `pipeline/success_estimate.py`.
- **Evidence:** `experiments/*/` (measurement scripts + raw outputs — the project's
  rule is every load-bearing number has a committed script + raw data),
  `exports/train-*/gap.json` (per-attempt gate results), `data/motions/thriller/`,
  `data/telemetry/*.npz` (real robot runs — ground truth), `tests/` (53 files).
- **Vendored:** `third_party/mjlab_mdp_ref/` (the pinned mjlab==1.5.0 semantics —
  the box runs pip mjlab; flag anywhere our code assumes semantics this contradicts),
  `third_party/whole_body_tracking/` (BeyondMimic), `third_party/unitree_*`.

## 3. Ground truths you must NOT dispute (verified from real hardware/telemetry)

- Real robot measured ankle-pitch p95 **15–19 Nm**, knee p95 **31–34 Nm** at native
  tempo (`experiments/torque_crosscheck_20260720`, from `data/telemetry/*.npz`).
- The anchor policy performed the native-speed dance **live at 70–80% quality** with
  no catastrophic drift — the one real datapoint that the sim gate should be
  calibrated against.
- Robot: G1 EDU, 29 DoF, no hardware e-stop (only the remote's B-damp + power
  switch). Onboard compute is a Jetson (PC2). Actuators are per-motor-class (NOT one
  torque limit for all joints — see `pipeline/g1_limits.py`).
- The dev laptop has **no GPU**; all training is on a cloud 4090 (GreenNode notebooks).

## 4. What we already believe is wrong / hard (build on these, and CHALLENGE them)

Do not merely re-report these; either confirm-and-deepen or refute them:
- **sim2real gap is the core risk:** the gate trains and evaluates in the same mjlab
  sim, with only ONE real calibration point. Is the gate trustworthy? Is anything
  still optimistic-by-construction?
- **drift:** the policy is deliberately position-blind (dropped `base_lin_vel`/anchor
  from the actor obs to avoid a leg-odometry spike that caused a tethered "fling").
  It drifts ~0.5–0.9 m in clean sim. Is dropping state the right call, or should the
  onboard `rt/odommodestate` estimate be fed back? (see the hardware memory / decision log)
- **recipe complexity:** 7 generations of inherited reward terms. The last audit
  found several were inert or inverted. Are there MORE dead/conflicting/mis-scaled
  terms? Is the whole reward stack over-engineered — could a simpler recipe do better?
- **efficiency:** one dance took 10+ training sessions. Where is the wasted compute
  and wasted human iteration? (motion prep, curriculum length, verify chain, gate design)
- **the recent 12 fixes** (`experiments/ml_audit_20260721`): are any wrong,
  incomplete, or did they introduce regressions? The ankle-recipe fixes (F/G) and the
  deploy history-stacking (E) are unvalidated on GPU/hardware — scrutinize them.

## 5. Your audit objectives

Cover all five; prioritize by "does fixing this reduce future training sessions or
close the sim2real gap?":

1. **Correctness / ML-weight distortion:** wrong, conflicting, double-counted, or
   mis-scaled reward/termination terms across the v3→v11 chain and the base task;
   buggy custom term implementations (tensor shape/axis/sign/units); obs-contract
   consistency between training, the sandbox, and `deploy_runtime`.
2. **Motion & retarget fidelity:** GVHMR→GMR→clean→ground→repair→npz — quaternion
   conventions (CSV xyzw vs mujoco wxyz), resampling/velocity artifacts, ROM
   under-scaling, grounding, and whether cleaning still blunts choreography.
3. **Sim2real & eval trust:** is the gate calibrated to the one real datapoint? Are
   cross-version comparisons valid? Does the sim model (friction, contact, actuator
   model, `tools/assets/g1_faithful/`) match the training model and reality?
4. **Efficiency / optimization:** compute (envs, iters, curriculum, checkpointing,
   verify chain) and human-iteration (what makes runs inconclusive). Propose a
   leaner path to a good policy.
5. **Deploy safety & readiness:** `deploy_runtime` + `deploy_guards` — is the safety
   spine sound for an e-stop-less robot? Is the 770-dim history-stacked deploy path
   correct (it was crash-broken until a recent fix)?

Also welcome: architecture-level suggestions (is BeyondMimic tracking the right
approach? phase-conditioning? residual control? better retargeting?), and any
over-engineering you'd delete.

## 6. Constraints & method

- **Do NOT modify `~/robot/`** (the working teleoperation stack — read-only).
- **Robot safety is paramount:** never propose sending low-level robot commands
  without sim verification + explicit human confirmation; there is no hardware e-stop.
- **Measurement discipline (their rule, adopt it):** never call a finding decisive
  without an independent cross-check; if you run analysis, commit the script AND its
  raw output under `experiments/`. Quantify claims — no vibes.
- **Be adversarial and specific:** every finding needs a `file:line` citation, the
  distortion mechanism, its concrete effect (on learned weights / the preview / the
  sim2real gap), a proposed fix with blast-radius, and a confidence level. Prefer a
  few load-bearing findings over many trivia.
- **What you CANNOT access:** `.secrets/` (credentials), the live GPU boxes, and the
  robot (down). You can read all code, data, telemetry, experiments, and git history
  (`git log`, 405 commits). mjlab is not installed on the laptop, so GPU-dependent
  claims must be reasoned from code + the vendored `third_party/mjlab_mdp_ref/` and
  flagged as "needs GPU to confirm."
- Pinned training env: `cloud/env_lock/requirements.lock.txt`
  (mjlab==1.5.0, mujoco-warp==3.10.0.1, warp-lang==1.14.0, torch cu128).

## 7. Deliverable

A single markdown report with:
1. **Executive summary** — the 3–5 mechanisms most responsible for slow progress /
   the sim2real gap, and what to change first.
2. **Confirmed findings** — each with file:line, mechanism, quantified effect, fix
   spec (concrete enough to implement), blast-radius, confidence, and whether it
   needs a GPU/robot to validate.
3. **Optimization opportunities** — compute + human-iteration, with rough savings.
4. **Architecture-level recommendations** — bigger bets, with the risk/reward.
5. **Prioritized action plan** — grouped by "land together," flagged for what needs
   GPU vs CPU-only, ordered by impact on training-session count and sim2real trust.
6. **Watchlist** — plausible-but-unverified items and what evidence would settle each.

Be ruthless about cutting cosmetic issues. The bar for every finding: does fixing it
plausibly make the next training session more decisive, close the sim2real gap, or
make the robot safer? Start by reading Section 1's files, then form your own view —
do not anchor on the previous audit.
