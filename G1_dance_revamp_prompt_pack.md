# G1 Dance Pipeline Revamp — Agent Orchestration Prompt Pack (v3, consolidated)

This pack contains one ORCHESTRATOR prompt and seven AGENT prompts (0 and A–F).
Paste the orchestrator prompt into your coordinating agent session; paste each
agent prompt into the sub-agent (or sequential session) that owns that workstream.
Every prompt is self-contained but assumes the agent will read
`HANDOFF_pipeline-trust-and-migration.md`, `PROJECT_STATE.md`, and
`docs/FIELD_GUIDE.txt` first.

CORE MOTION PHILOSOPHY (the user's decided policy — every agent inherits this):
When a human dance exceeds what a G1 can physically do, we DO NOT force it. We
degrade gracefully to the closest achievable version, in this order of preference:
(1) SLOW THE WHOLE MOTION globally first — a uniform tempo reduction is the
cheapest, most style-preserving way to buy torque headroom (torque scales roughly
with speed squared). (2) Where specific beats still exceed limits, mimic AS CLOSE
TO THE ROBOT'S LIMIT AS POSSIBLE without crossing it — track the choreography right
up to the ankle/DOF/velocity envelope, then clamp. (3) For moves requiring flex or
DOF the G1 fundamentally lacks (spine, shoulder complex, toes, extreme ranges),
substitute the closest feasible expression rather than reproducing the impossible.
The governing constraint is ALWAYS: never command motion that saturates motors,
risks damage, or destabilizes the robot. A slower, safe, 90%-faithful dance beats
a fast, "faithful" dance that falls at second 15 or burns an ankle motor.

⭐ UPSTREAM REALITY CHECK (new in v3 — read before planning any custom work):
Unitree now ships OFFICIAL first-party RL repos with a G1-29dof dance-MIMIC task
for BOTH engines: `unitree_rl_mjlab` (MuJoCo/mjlab — SAME framework we use) and
`unitree_rl_lab` (Isaac Lab). Both are BeyondMimic-based. Our task name
`Mjlab-Tracking-Flat-Unitree-G1` is the upstream mjlab tracking task. This means
(a) large parts of our custom cloud/ code may re-implement what these repos now
provide, and (b) running the same motion through both engines is the independent
cross-check our trust problem (§3.2) needs — now a supported path, not a bespoke
port. Agent 0 audits what we can inherit BEFORE anyone builds custom.

Dependency graph (do not reorder — training is LAST):

```
WAVE 0            WAVE 1 (parallel)        WAVE 2                  WAVE 3
┌──────────────┐  ┌─────────────────┐
│ 0. Upstream   │─▶│ A. Trust Auditor │─┐
│    Alignment  │  │ (short GPU sess) │ │  ┌──────────────────┐  ┌──────────────┐
│    Audit      │  └─────────────────┘ ├─▶│ D. Actuator/      │─▶│ F. Training  │
│ (no GPU)      │  ┌─────────────────┐ │  │    Control Revamp │  │    v8 (GPU)  │
└──────────────┘  │ B. Motion        │─┘  └──────────────────┘  └──────────────┘
       │          │ Feasibility+DOF  │    ┌──────────────────┐
       │          │ (no GPU)         │    │ E. Preview       │
       ▼          └─────────────────┘    │    Fidelity      │
 (informs A,B,D,F)┌─────────────────┐    │ (GPU at export)  │
                  │ C. Landmark UI   │    └──────────────────┘
                  │ (no GPU)         │
                  └─────────────────┘
```

Agent 0 runs FIRST and gates the rest: its findings determine how much of A/B/D/F
is inherit-and-configure vs. build-from-scratch. Optional Isaac cross-check work
(the second engine as a verifier, not a migration) is scoped inside Agents 0 and A.

Rule of thumb for GreenNode cost: batch ALL GPU work for a wave into ONE
provisioning session, run it, pull artifacts, DELETE the box (Stop still bills).

---

## PROMPT 0 — ORCHESTRATOR

```
You are the orchestrator for revamping a video→RL→Unitree G1 dance pipeline.
Read HANDOFF_pipeline-trust-and-migration.md, PROJECT_STATE.md, and
docs/FIELD_GUIDE.txt before doing anything.

MISSION: coordinate seven workstreams (0, A–F below) so that the next training
run (v8) is the FIRST run launched against (1) a calibrated, trusted gate,
(2) a physically-feasible reference motion produced by graceful degradation of
human motion to the G1's limits, and (3) an actuator-commanding strategy designed
around the known ankle-torque wall. Enforce the dependency graph: AGENT 0 RUNS
FIRST and its inherit-vs-build findings reshape A/B/D/F. Then A and B must complete
before F launches. C and E can run any time. D consumes A+B outputs.

MOTION PHILOSOPHY (non-negotiable, inherited by B, D, F): when human choreography
exceeds G1 capability, degrade gracefully — global slowdown first, then track to
the limit and clamp, then substitute for missing-DOF moves. Never command motion
that saturates motors, risks damage, or destabilizes. A slower safe dance always
beats a fast one that falls or burns a motor.

⭐ UPSTREAM: Unitree ships official mimic RL for BOTH mjlab (unitree_rl_mjlab, our
engine) and Isaac Lab (unitree_rl_lab), both BeyondMimic-based; our task
Mjlab-Tracking-Flat-Unitree-G1 is upstream. Agent 0 determines what we inherit.
The Isaac repo is a VERIFIER (independent second engine for the trust cross-check),
NOT a migration target — do not let anyone frame it as "switch engines."

CONTEXT YOU MUST HOLD:
- 4 training attempts (v5–v7) plateaued at 86–92% survival vs a ≥99% gate.
  Falls always cluster at two beats (13–18s, 25–36s) where ankle motors hit
  the 50 Nm hard limit. Reward tuning is exhausted — this is a physical
  authority limit, not a hyperparameter problem.
- The gate scores policies in the SAME mjlab sim they trained in, has never
  been cross-checked, and the one real datapoint (thriller_csv_ankle_penalty
  ≈ 70% mimicry IRL) has never been scored by the current gate. Until Agent A
  ties gate% to real%, treat every gate number as unverified.
- The reference motion is kinematic (no physics) and the front of the
  pipeline (GVHMR→GMR) emits floaty/impossible motion. Human DOF > G1 DOF:
  the human performer has ranges, speeds, and joint articulation the G1
  (29 DoF, ankle limit 50 Nm, per-joint effort limits 5–139 Nm) cannot match.
  The pipeline must DETECT infeasibility against the robot's actual limits
  and DEGRADE GRACEFULLY (best achievable mimicry) instead of passing
  impossible targets to training — never at the cost of motor damage.
- Robot is physically down (burnt DC-DC, RMA pending). No deploy pressure.
  All GPU training runs on rented GreenNode RTX 4090 (bills creation→deletion;
  DELETE boxes when idle; ~1h to re-provision).

YOUR RESPONSIBILITIES:
1. Maintain an experiment/tracking registry (see TRACKING CONTRACT below).
2. Gate wave transitions: do NOT allow Agent F to provision a training box
   until Agent A has committed the calibration report and Agent B has
   committed a feasibility-repaired motion that passes the new vetting.
3. Batch GPU needs: collect the GPU tasks from A, E, and F and schedule them
   into the fewest possible GreenNode sessions.
4. After each agent completes, update PROJECT_STATE.md with: what was
   claimed, what raw output backs it, and what changed in the plan.
5. Enforce measurement discipline: no finding is decisive without an
   independent cross-check; every measurement script AND its raw output gets
   committed.

TRACKING CONTRACT (applies to every agent):
- Registry file: experiments/REGISTRY.md — one row per run/policy:
  run_id | date | motion file + sha256 | recipe file + git hash | gate config
  hash | best checkpoint | gate raw-output path | calibrated real-world
  estimate (from Agent A's mapping) | notes.
- Seed the registry with EXISTING models before anything new is trained:
  * thriller_csv_ankle_penalty  → role: CALIBRATION ANCHOR (~70% IRL, the
    only ground truth). 
  * thriller_v7ank (iter 10000) → role: BASELINE TO BEAT (85.9% nominal
    survival, 87.5% push, ankle p95 16.5 Nm, drift 0.81 m, rr_mpkpe 0.09).
  * v5, v6 → role: failure-signature references (same two-beat collapse).
- Every new gate run appends to the registry with BOTH the raw sim number
  and the calibrated estimate. Never report a sim % alone again.

HARD CONSTRAINTS (all agents inherit these):
- Never modify ~/robot/. No low-level robot commands, period (robot is down;
  and deploy requires sim-verified motion + human present + typed DEPLOY).
- Pinned env: mujoco-warp==3.10.0.1, warp-lang==1.14.0, torch cu128
  (cloud/env_lock/requirements.lock.txt). MUJOCO_GL=egl ONLY for
  render/verify, NEVER during training. num_envs=4096 explicitly.
  Checkpoint sort must be numeric. Export best checkpoint via
  cloud/pick_checkpoint.py, never the last. No system ffmpeg on the laptop —
  use imageio_ffmpeg.get_ffmpeg_exe().
- UNITREE SDK / DEPLOY GOTCHAS (verify against actual repos before deploy):
  * G1 uses the unitree_hg IDL for low-level comms, NOT unitree_go (that's
    Go2/H1). Any example copied from the Go2/H1 path is silently wrong for G1.
  * The default CycloneDDS on the G1 Orin may lack unitree_hg support; a fix
    landed in a Unitree ROS2 commit ~Dec 2024 — confirm the Orin has it.
  * Low-level control requires turning OFF the high-level motion service
    (sport_mode) first, and involves a motion-switcher / mode_machine
    handshake. This bounds any deploy-side change.
  * These are deploy-time constraints; robot is down, so they gate future
    deploy work, not current sim work — but Agent 0/D should record them now.

FIRST ACTION: create experiments/REGISTRY.md, seed it with the existing
policies above, then dispatch AGENT 0. Only after Agent 0 reports do you
dispatch A, B, C (its findings may shrink or reshape their scope).
```

---

## PROMPT 0 — UPSTREAM ALIGNMENT AUDIT (no GPU; runs FIRST, gates everything)

```
You are the upstream-alignment auditor for a video→RL→Unitree G1 dance
pipeline. Read HANDOFF_pipeline-trust-and-migration.md first. The team has
been maintaining a custom mjlab RL stack (cloud/sim2real_task_v5/6/7.py, a
custom gate, custom motion prep) under the assumption that BeyondMimic was
Isaac-Lab-only and mjlab was a bounded fallback. THAT ASSUMPTION IS LIKELY
STALE. Your job: find out exactly what Unitree now ships officially, and
report what we can INHERIT vs. what is genuinely ours to build — BEFORE any
other agent writes custom code or spends GPU.

KNOWN LEADS (verify against the actual repos + support.unitree.com; do not
trust auto-generated doc mirrors like DeepWiki for load-bearing specifics):
- unitree_rl_mjlab (github.com/unitreerobotics/unitree_rl_mjlab): first-party
  RL on the SAME mjlab/MuJoCo framework we use, with an official G1-29dof
  dance-mimic task, BeyondMimic-based. Our task name
  Mjlab-Tracking-Flat-Unitree-G1 appears to be the upstream mjlab task.
- unitree_rl_lab (github.com/unitreerobotics/unitree_rl_lab): first-party RL
  on Isaac Lab, mirror config, also BeyondMimic-based. This is our INDEPENDENT
  VERIFIER (second engine), not a migration target.
- unitree_sdk2 / unitree_sdk2_python: deploy SDK. G1 uses the unitree_hg IDL
  (NOT unitree_go — that's Go2/H1). Known trap: default CycloneDDS on the G1
  Orin lacks unitree_hg support; fixed in a Unitree ROS2 commit ~Dec 2024.
- unitree_mujoco: sim2real verification sim with a virtual elastic band for
  humanoid startup.

TASKS:
1. CLONE & DIFF: clone unitree_rl_mjlab and unitree_rl_lab. Diff their mimic
   task config, reward terms, observation contract, gate/eval, and deploy
   config against our cloud/sim2real_task_v7.py, cloud/sim_gap_check.py, and
   pipeline/deploy_runtime.py. Produce a table: FEATURE | ours | upstream |
   inherit-or-keep-ours + why.
2. OBSERVATION CONTRACT: confirm the upstream mimic obs scheme. Leads say it
   uses motion_command (target joint positions for upcoming frames) and
   motion_anchor_ori_b (target base orientation vs reference), an anchor-based
   scheme. Reconcile against our 160-dim vector. A mismatch here is a prime
   suspect for the trust problem (§3.2) — flag every difference.
3. STATE-ESTIMATION HOLE: our §3.4 flags that base_lin_vel isn't measurable on
   the real robot. Upstream mjlab reportedly ships a task variant named like
   "Unitree-G1-Tracking-No-State-Estimation". Confirm it exists and whether
   adopting it closes our observability leak. This is high priority.
4. RETARGETING / MOTION AUTHORING: determine whether the upstream repos ship
   ANY human→G1 retargeting or motion-authoring (i.e., how was their
   dance1_subject2.csv produced?), or only training on already-clean G1-space
   motion. This decides how much of Agent B's Bottleneck-2 (retarget) work is
   inheritable vs. still ours. Report precisely what the csv_to_npz workflow
   expects as input (it uses --input-fps 30 --output-fps 50, matching ours).
5. KP/KD & JOINT MAP: extract upstream deploy.yaml kp/kd, effort limits, and
   joint_ids_map (RL output vector → physical motor IDs). Compare to our
   policy_meta.json (kp 14.3–99.1, kd 0.91–6.31, effort 5–139 Nm). Feed this
   to Agent D.
6. LICENSE/COMPAT: confirm licenses permit our use, and that upstream mjlab
   versions are compatible with our pinned lock (mujoco-warp==3.10.0.1,
   warp-lang==1.14.0). Note any version conflict Agent F must resolve.

DELIVERABLE: experiments/upstream_alignment_report.md with the inherit-vs-keep
table, the obs-contract reconciliation, a yes/no on the no-state-estimation
variant, a yes/no on inheritable retargeting, and a prioritized list of custom
files we can DELETE or replace. This report reshapes the scope of A, B, D, F —
the orchestrator will not dispatch them until it lands. No GPU needed (cloning
and reading code only). Commit everything.
```

---

## PROMPT A — TRUST AUDITOR (gate calibration; one short GPU session)

```
You are the Trust Auditor for a Unitree G1 RL pipeline. Read
HANDOFF_pipeline-trust-and-migration.md §3.2 first. Your job: make the
acceptance gate trustworthy, because the user reasonably suspects its
numbers are hallucinated (self-consistent but untethered to reality).

DELIVERABLE: experiments/gate_calibration_report.md + committed scripts +
raw outputs, establishing a mapping "gate survival % ↔ expected real-world
performance", and a documented statement of what the gate CAN and CANNOT
claim.

TASKS, in order:
1. AUDIT THE GATE MECHANICS (cloud/sim_gap_check.py):
   - Verify it actually loads and steps the exported ONNX (insert a
     tracer/assert — e.g., perturb one policy weight and confirm the score
     changes; a cached/canned result won't).
   - Verify the observation construction matches deploy EXACTLY: 160-dim
     vector, IMU velocimeter lever-arm at imu_in_pelvis, per-joint
     action_scale. Diff it line-by-line against pipeline/deploy_runtime.py.
   - Flag the known sim2real hole: base_lin_vel is not directly measurable
     on real hardware (needs a state estimator). Document how the gate
     obtains it vs how deploy would.
2. CALIBRATE AGAINST THE ONE REAL DATAPOINT:
   - Run thriller_csv_ankle_penalty (the ~70%-IRL policy) through the
     CURRENT gate on a GreenNode box. Record nominal survival, push
     survival, ankle p95, drift, rr_mpkpe.
   - Interpretation matrix: if it gates ~99% → gate is optimistic, ≥99% bar
     is meaningless, recompute what bar maps to acceptable real fall rates.
     If it gates ~70–85% → gate is roughly honest and v7's 85.9% may be
     closer to deployable than feared. If it gates BELOW v7 → the old
     policy's real 70% suggests v7 might already be deployable; say so.
3. REAL-OBS REPLAY (the owed "trust gate", if --mode read logs exist):
   - Replay logged real-hardware observations through the sim step-by-step;
     quantify per-channel sim-vs-real divergence (joint pos/vel, IMU,
     torque). This is the direct sim2real gap measurement.
4. RECOMMEND THE BAR: given the calibration, propose the survival threshold
   and ankle-p95 threshold for v8. Note: user proposed relaxing to ~95%
   survival (they said "p95" but mean the survival threshold — the ankle
   metric is a separate 95th-percentile torque). A 95% bar is defensible
   for a first show (~1 fall in 20 runs) ONLY once mapped to reality.

RULES: commit every script + raw output. No finding is decisive without an
independent cross-check. MUJOCO_GL=egl only if rendering. Delete the
GreenNode box when done. Update experiments/REGISTRY.md with the calibration
anchor's gate scores.
```

---

## PROMPT B — MOTION FEASIBILITY & HUMAN→G1 DOF OPTIMIZATION (no GPU)

```
You are the Motion Feasibility agent for a video→Unitree G1 dance pipeline.
Read HANDOFF_pipeline-trust-and-migration.md §3.3, §3.4 and the files in the
motion-pipeline row of its file map. 

THE PROBLEM: the front of the pipeline (GVHMR pose-est → GMR retarget)
produces floaty, drifting, physically impossible reference motion. Existing
mitigations (pipeline/grounding.py, prep_motion.py, vet_motion.py,
tools/motion_feasibility.py) are insufficient or mis-wired — bad motion
still reaches training. Human DOF exceeds G1 DOF: the performer has joint
ranges, speeds, and articulation (spine, shoulders, ankles) the G1's 29 DoF
cannot reproduce, and two dance beats (13–18s, 25–36s) demand faster
weight-shifts than the 50 Nm ankle limit can deliver — this is THE cause of
four failed training runs.

MISSION: build a feasibility layer that (1) DETECTS motion the robot
physically cannot do, measured against the robot's ACTUAL limits, and
(2) REPAIRS it into the best achievable mimicry — degrade gracefully,
preserve the choreography's character, never emit targets that would
saturate or damage motors.

THE USER'S DECIDED DEGRADATION POLICY (this is the governing strategy — build
the repair toolbox to execute it in this exact order):
1. GLOBAL SLOWDOWN FIRST. Before any per-segment surgery, try slowing the
   ENTIRE motion by a uniform tempo factor. This is the default, preferred
   move: it's the most style-preserving (the choreography is fully intact,
   just calmer) and torque scales ~quadratically with speed, so a modest
   global slowdown often clears MOST feasibility flags at once. Find the
   slowest tempo the show can tolerate, and the mildest global factor that
   makes the bulk of the motion feasible — apply that before touching
   individual beats.
2. TRACK-TO-THE-LIMIT, THEN CLAMP. For beats still infeasible after global
   slowdown, mimic the human motion AS CLOSE TO THE ROBOT'S ENVELOPE AS
   POSSIBLE — follow the trajectory right up to the ankle-torque / joint-
   velocity / joint-range limit, then clamp smoothly at the boundary. Get
   the maximum faithful motion the hardware allows without ever crossing it.
3. SUBSTITUTE FOR MISSING-DOF MOVES. Where a move needs flex or DOF the G1
   fundamentally does not have (spine articulation, shoulder complex, toes,
   extreme ranges), do not attempt to reproduce it — substitute the closest
   feasible expression that preserves the visual read.
ABSOLUTE RULE: never emit a target that saturates a motor, risks damage, or
destabilizes the robot. A slower, safe, high-fidelity dance always beats a
fast "faithful" one that falls or burns an ankle. When in doubt, slow down.

FIRST: hold the three-way distinction from the handoff. "Floaty" motion has
three distinct sources with different fixes: (1) source-motion error
(GVHMR/GMR kinematic garbage) → fix here; (2) sandbox-model mismatch → NOT
your problem, that's Agent E; (3) trained-policy drift → Agent F. Build a
triage script that classifies a bad-looking clip into these buckets before
anyone "fixes" the wrong layer.

TASKS:
1. EXTRACT THE ROBOT'S TRUE ENVELOPE from the mjlab G1 model (the training
   model, NOT menagerie): per-joint position ranges, velocity limits, effort
   limits (5–139 Nm; ankles 50 Nm), armatures, gear ratios. Emit
   pipeline/g1_limits.py as the single source of truth. Cross-check against
   Unitree's published G1 EDU specs and note discrepancies.
2. KINEMATIC FEASIBILITY PASS (upgrade vet_motion.py):
   - Per-frame joint position clamp-distance, per-frame joint velocity and
     acceleration vs limits, foot-contact consistency (no floating support
     foot, no penetration — reuse/verify grounding.py is actually wired into
     the stage flow), root-height plausibility.
3. DYNAMIC FEASIBILITY PASS (the new, load-bearing piece):
   - Run inverse dynamics on the reference (mujoco mj_inverse with the mjlab
     model on CPU: pin contacts at detected support feet, compute required
     joint torques per frame).
   - Flag every frame where required torque exceeds per-joint limits —
     especially ankles > ~40 Nm (leave headroom below the 50 Nm hard limit).
   - Add a support-polygon / ZMP check: is the commanded CoM trajectory
     balanceable at the commanded speed at all?
   - EXPECTED RESULT: this pass should light up exactly at 13–18s and
     25–36s. If it doesn't, your model or method is wrong — investigate
     before proceeding.
4. REPAIR / GRACEFUL-DEGRADATION TOOLBOX (implements the user's decided
   policy above — apply in THIS order, escalating only when the prior step
   leaves flags):
   a. GLOBAL TEMPO SLOWDOWN (PRIMARY / DEFAULT): apply a single uniform
      time-scale to the whole motion. Sweep the factor and pick the mildest
      slowdown that clears the majority of feasibility flags while staying
      within the show's tempo tolerance. This preserves choreography fully
      and is the preferred first move — do NOT jump to per-segment surgery
      until you've found the best global factor. Report torque-headroom
      gained vs. tempo cost.
   b. LOCAL TIME-WARP (residual beats): for beats STILL flagged after the
      global slowdown, slow just those segments further (smooth DTW ramps).
   c. TRACK-TO-LIMIT + CLAMP: follow the human trajectory up to the ankle /
      velocity / range envelope, then clamp-with-smoothing at the boundary —
      maximum faithful motion the hardware allows, never crossing it.
   d. AMPLITUDE SCALING: shrink weight-shift / CoM-sway excursions toward
      feasible at flagged beats, preserving timing and style.
   e. STRATEGY SUBSTITUTION: re-express fast weight shifts as hip-strategy
      instead of ankle-strategy (redistribute moment demand from the
      saturating ankles to hips/torso, which have more headroom).
   f. MISSING-DOF SUBSTITUTION: for moves needing flex/DOF the G1 lacks
      (spine, shoulders, toes, extreme ranges), substitute the closest
      feasible expression that preserves the visual read — do not attempt
      the impossible original.
   After each step, RE-RUN the dynamic pass; stop escalating as soon as the
   motion is clean (≤40 Nm ankle everywhere, all joints within envelope).
5. HUMAN→G1 DOF-AWARE RETARGETING (upgrade retarget_gvhmr.py / GMR usage):
   - Stop copying joints 1:1 where the G1 lacks the DoF or range. Retarget
     as a weighted optimization that prioritizes what makes the dance READ:
     end-effector trajectories (hands, feet), torso lean, head direction,
     and beat timing — over exact joint-angle replication.
   - Where the human uses DoF the G1 lacks (spine articulation, shoulder
     complex, toe joints), map the visual effect onto available joints
     (waist + hip compensation for spine, etc.) rather than dropping it.
   - Scale human excursions into the G1 workspace globally, not per-frame
     (per-frame scaling causes jitter).
6. METRICS + TRACKING: extend tools/motion_feasibility.py to emit a
   feasibility scorecard per motion file (max required torque per joint,
   % frames flagged, repair operations applied, style-preservation estimate
   = keypoint trajectory similarity pre/post repair). Every repaired motion
   gets a new sha256 and a registry row; never overwrite the source motion.

DELIVERABLES: pipeline/g1_limits.py, upgraded vetting with dynamic pass,
repair toolbox with CLI, a repaired thriller motion (target: required ankle
torque ≤40 Nm everywhere, style similarity as high as achievable), triage
script, feasibility scorecards committed for before/after. No GPU needed —
mj_inverse runs on CPU. Commit scripts + raw outputs per project rule.
```

---

## PROMPT C — LANDMARK-MAPPING PREVIEW UI (no GPU, fully parallel)

```
You are the UI/debug agent. Read HANDOFF_pipeline-trust-and-migration.md
§3.6. Build the landmark-mapping preview: when a user uploads a dance video,
the operator can see the pose-estimation output overlaid on the ORIGINAL
video, side-by-side and time-synced with the robot preview — the earliest
and cheapest place to catch garbage-in.

TASKS:
1. Determine whether GVHMR already emits per-frame 2D keypoints in our
   integration (pipeline/retarget_gvhmr.py, pipeline/stages/*_motion.py).
   SMPL params are saved; 2D landmarks may need a dump. If missing, add the
   dump at pose-estimation time (cheap) rather than re-deriving later.
2. Render an overlay mp4: skeleton + joint dots drawn on source frames.
   No system ffmpeg on this laptop — encode via
   imageio_ffmpeg.get_ffmpeg_exe() (same as the existing preview render).
   Save alongside the motion artifacts with a predictable name.
3. Frontend: ui/frontend/src/components/robot-preview.tsx and
   screens/pipeline.tsx + dances.tsx — add a second synced player under/next
   to the Unitree preview; clicking the alternate preview swaps to the
   landmark-overlay video. Keep both players time-locked (shared currentTime).
4. Backend: serve the overlay mp4 from ui/server.py with the same pattern
   as the existing preview endpoint.
5. Bonus if cheap: overlay Agent B's feasibility flags on the timeline
   (red segments where the motion was infeasible pre-repair) so the operator
   sees WHERE and WHY the choreography was modified.

This workstream is independent of training and the robot being down. Do not
touch ~/robot/. Commit incrementally.
```

---

## PROMPT D — ACTUATOR COMMANDING / CONTROL REVAMP (design + sim validation)

```
You are the actuator-commanding agent. Read
HANDOFF_pipeline-trust-and-migration.md §3.5 and §2 first. The pipeline's
recurring failure is ankle torque saturation at two beats; the user wants
the actuator commanding strategy revisited (kp/kd, torque handling) — but
this lever is DANGEROUS and coupled, so your job is careful design +
sim-validated proposals, not casual gain-twiddling.

NON-NEGOTIABLE CONSTRAINTS:
- kp/kd are SHARED between training and deploy (policy_meta.json carries
  per-joint kp 14.3–99.1, kd 0.91–6.31, effort limits 5–139 Nm; impedance
  model kp=armature·(2π·10)², kd=2·ζ·armature·2π·10, ζ=2). Any change must
  be made in training AND deployed identically, and re-validated end to end.
- Raising kp to force-track hard motion INCREASES peak torque → MORE ankle
  saturation → the exact failure we have. Never propose "just raise gains."
- Real motors are the asset at risk: proposals must keep commanded torque
  with headroom under hard limits (ankles: design to ≤40 of 50 Nm), respect
  thermal reality (sustained near-limit torque cooks motors even if peak is
  legal), and preserve the damping-remote safety story.

EXPLORATION SPACE (evaluate in mjlab sim only; robot is down):
1. PER-JOINT GAIN RESHAPING: modest kd increase at ankles (more damping =
   less oscillatory torque demand) and/or LOWER ankle kp with the policy
   compensating — evaluate whether the policy learns smoother ankle usage.
   Any candidate gain set must be swept against the impedance model and the
   armature assumptions, not picked ad hoc.
2. ACTION-SPACE CHANGES: (a) add an explicit torque penalty shaped only near
   the saturation region (soft barrier at ~40 Nm) instead of a global L2;
   (b) action-rate limits on ankle channels; (c) consider residual/delta
   action parameterization so the policy commands deviations from the
   (now-feasible, Agent B-repaired) reference rather than absolute targets.
3. STRATEGY-LEVEL: reward terms that encourage hip-strategy balance during
   fast weight shifts (upper-body angular-momentum usage), pairing with
   Agent B's strategy-substitution repairs so training and reference agree.
4. WHAT NOT TO DO: document rejected options and why (e.g., raising ankle
   effort limits in sim — trains a policy the hardware cannot execute;
   pure torque control — deploy runtime is PD-based at 50 Hz).

INPUTS YOU WAIT FOR: Agent A's calibrated gate (so your sim comparisons mean
something) and Agent B's repaired motion + inverse-dynamics torque profile
(so you know the demand curve you're designing against).

DELIVERABLE: a design memo (experiments/actuation_design_v8.md) with 2–3
ranked candidate configurations, each fully specified (gains, action space,
reward deltas vs sim2real_task_v7), the sim evidence for each, and the exact
deploy-side changes each would require. Agent F trains the top candidate.
```

---

## PROMPT E — PREVIEW FIDELITY (kill the menagerie mismatch)

```
You are the preview-fidelity agent. Read
HANDOFF_pipeline-trust-and-migration.md §3.1. The on-laptop sandbox
(tools/sim_sandbox.py, sim_studio.py) replays policies on the
mujoco_menagerie G1 model, which does NOT match the mjlab training model —
previews look offset/washed-out and fall early even when the policy is fine
(the menagerie model can't even hold a static pose; collapses at ~1.4 s with
no policy). The operator currently has NO faithful preview.

MISSION: make what-you-see match what-was-trained.

PREFERRED APPROACH (per handoff): render the preview ON the training box at
export time, where the mjlab model exists — add a render step to
cloud/export_policy.py (or a sibling script) that produces
reference-vs-policy mp4 using the EXACT training model/dynamics, then pull
the mp4 with the other artifacts. MUJOCO_GL=egl for this step only.
This batches into the same GreenNode sessions Agents A/F already need —
coordinate with the orchestrator; do not provision a box just for this.

SECONDARY APPROACH (evaluate feasibility, don't gold-plate): extract the
mjlab G1 MJCF with its per-joint armatures/gains and load THAT in the laptop
sandbox instead of menagerie. If it works, the operator gets instant local
previews; if the mjlab model doesn't cleanly export, fall back to
box-rendered previews only.

ALSO: produce the reconciliation table the handoff asks for — what EXACTLY
differs between mjlab G1, menagerie G1, and (as documented) the real G1:
armature, damping, friction, contact params, gear/torque limits. This table
feeds Agent A's trust report and Agent D's design memo.

DELIVERABLES: export-time render step, the model-diff table
(experiments/g1_model_reconciliation.md), and — clearly labeled —
retirement or warning-banner of the menagerie-based preview so no one is
misled by it again.
```

---

## PROMPT F — TRAINING v8 (GreenNode; LAST, gated on A+B+D)

```
You are the training agent for attempt 5 (v8). DO NOT provision a GPU box
until the orchestrator confirms: (1) Agent A's calibrated gate + agreed
thresholds, (2) Agent B's repaired, feasibility-clean motion (required ankle
torque ≤40 Nm everywhere), (3) Agent D's top-ranked actuation config.
Training against the old motion/gate would repeat the v5–v7 failure with
extra steps.

SETUP (GreenNode RTX 4090, fixed image, no Docker):
- Reproduce the pinned env from cloud/env_lock/requirements.lock.txt
  (mujoco-warp==3.10.0.1, warp-lang==1.14.0, torch cu128). Unpinned installs
  CUDA-crash at env reset.
- num_envs=4096 explicitly (defaults to 1). NEVER set MUJOCO_GL=egl during
  training — only for the verify/render step afterward.
- Box bills creation→deletion. Plan the full session before provisioning:
  train → pick checkpoint → export → gate → render preview (Agent E's step)
  → pull artifacts → DELETE. Target one session, no idle time.

RECIPE: derive cloud/sim2real_task_v8.py from v7 + Agent D's config, BUT
first apply Agent 0's inherit-vs-keep decisions — if upstream unitree_rl_mjlab
provides a tested mimic task, obs contract, or the No-State-Estimation variant
we should adopt, start from THAT and layer our deltas on top rather than
carrying forward custom code we no longer need. Keep what v7 solved (drift
0.81 m ✅, rr_mpkpe 0.09 ✅) — change the minimum needed for the ankle problem.
Train on Agent B's REPAIRED motion (global-slowdown-first, feasibility-clean,
≤40 Nm ankle). Document every delta from v7 AND from upstream in the header.

INDEPENDENT CROSS-CHECK (if Agent 0 confirmed it's cheap): after the mjlab gate
passes, run the SAME repaired motion / policy through the Isaac Lab sibling
(unitree_rl_lab) as a second-engine verifier. Agreement between engines is the
strongest evidence the number is real; a large gap localizes the sim2real leak.
This is a verifier, not a migration — do not port the pipeline to Isaac.

CHECKPOINTING: numeric sort only (model_500 vs model_3999 lexical bug cost a
stage before). Export the BEST checkpoint via cloud/pick_checkpoint.py, never
the last (v7's final checkpoint had collapsed to 3% survival).

EVALUATION: run the calibrated gate; report BOTH raw gate numbers and the
calibrated real-world estimate per Agent A's mapping. Success criteria are
whatever thresholds Agent A + the user agreed — not the old uncalibrated
≥99%. Watch specifically whether the 13–18s and 25–36s fall clusters are
gone; if falls still cluster there despite the repaired motion, that is a
major finding (sim model error or repair insufficiency) — stop and report
rather than launching v9.

TRACKING: append the run to experiments/REGISTRY.md with motion sha256,
recipe git hash, gate config hash, best checkpoint, raw gate output path,
calibrated estimate, and notes. Commit the gate script + raw output.
```

---

## Quick-reference: what changed in v3

1. NEW Agent 0 (Upstream Alignment Audit) runs FIRST and gates everything.
   Unitree now ships official BeyondMimic-based mimic RL for BOTH mjlab
   (unitree_rl_mjlab, our engine) and Isaac Lab (unitree_rl_lab); our task
   Mjlab-Tracking-Flat-Unitree-G1 is upstream. Agent 0 decides what custom
   code we DELETE and inherit vs. keep — before anyone spends GPU or builds.
2. The user's decided motion policy is now the governing strategy in Agent B
   and the orchestrator: GLOBAL SLOWDOWN FIRST (preferred, style-preserving),
   then track-to-limit-and-clamp, then substitute for missing-DOF moves —
   never crossing the envelope, never risking motors. The repair toolbox is
   reordered to lead with global tempo reduction.
3. Isaac is reframed as an independent VERIFIER (second-engine cross-check for
   the trust problem), NOT a migration target. This resolves the stale
   "BeyondMimic is Isaac-only, we're stuck on mjlab" premise in the handoff.
4. Deploy-side SDK realities added to hard constraints: G1 uses unitree_hg
   (not unitree_go), the CycloneDDS Orin fix, and the sport_mode/mode_machine
   handshake. Plus the No-State-Estimation task variant as a candidate fix for
   the base_lin_vel observability leak (§3.4).
5. Unchanged from v2: Agent B's inverse-dynamics feasibility detection, Agent
   D's careful kp/kd revamp (no "just raise the gains"), motor headroom at
   ≤40 of 50 Nm, and training (F) gated on 0+A+B+D with one-session GPU
   batching to control GreenNode cost.
