# AGENT HANDOFF — G1 Dance pipeline: trust, fidelity, migration & UI debug

**Written:** 2026-07-16 · **For:** a fresh agent (Fable 5) to brainstorm solutions.
**Read these first for background, then come back here:**
`docs/FIELD_GUIDE.txt` (plain-language explainer of the whole project) and
`PROJECT_STATE.md` (live decision log — the single source of truth).

This document is NOT a plan to execute blindly. It is a problem statement + full
context so you can brainstorm approaches with the user. Several of the user's asks
contain assumptions worth pressure-testing — those are flagged **[CHECK]**.

---

## 0. One-paragraph situation

We turn a dance video into an RL controller that makes a Unitree G1 humanoid
perform that dance while balancing. The **pipeline runs end-to-end**, but after
**4 training attempts (v5→v7)** the controller **won't clear the acceptance gate**
(needs ≥99% simulated survival; stuck at ~86–92%). Worse, the user has lost trust
in the whole measurement chain: the on-laptop simulation **looks visibly worse and
offset** vs. what the real robot actually did (the last real deploy,
`thriller_csv_ankle_penalty`, mimicked ~70% of the motion IRL and looked fine),
the motion pipeline sometimes produces **physically impossible / floaty motion**,
and the user **suspects the passing/failing numbers may be effectively
hallucinated** — i.e. not corresponding to reality. The robot is **physically
down** (burnt DC-DC converter, awaiting RMA), so there is **no deploy urgency** —
this is the right moment to fix fidelity and trust before spending more compute.

---

## 1. What the system is (fast version)

Pipeline (assembly line, each stage feeds the next):

```
video ─▶ [1 pose est: GVHMR→SMPL] ─▶ [2 retarget: GMR→29-DoF] ─▶ [3 clean/vet]
      ─▶ motion.npz ─▶ [4 train: mjlab RL on GPU] ─▶ policy.onnx
      ─▶ [5 verify: the "gate"] ─▶ pass? ─▶ [6 deploy on robot @50Hz]
```

- **Robot:** Unitree G1 EDU Ultimate, **29 DoF** + Inspire hands. Onboard compute =
  Jetson Orin ("PC2"). ~48 V battery. **No hardware e-stop** — only remote B-damp +
  power switch (safety-critical).
- **Training framework:** **mjlab 1.5.0** (MuJoCo + Warp GPU physics), algo PPO via
  rsl_rl. Task `Mjlab-Tracking-Flat-Unitree-G1`. 4096 parallel sim robots.
- **Why mjlab and not Isaac Lab / BeyondMimic:** the intended research stack
  (BeyondMimic) runs on Isaac Lab → Isaac Sim (Omniverse). Our rented **GreenNode**
  cloud GPU has a **fixed image, no Docker**, and cannot install Isaac Sim →
  "permanently dead on this image." mjlab was the bounded fallback and became the
  main road. **This constraint is now in question — see §4 (migration).**
- **Compute:** no local GPU on the current dev laptop → training on a rented
  GreenNode RTX 4090 (billed creation→deletion; box for attempt 4 is now DELETED).
- **App:** FastAPI backend (`ui/server.py`, `:8735`) + React/Vite frontend
  (`ui/frontend/`) wrapped as a desktop window via pywebview (`ui/desktop.py`).
  It's an operator console (show mode, pipeline view, safety UX, monitors).

---

## 2. Where we are RIGHT NOW

- **Attempt 4 (v7) FAILED the gate.** Best checkpoint (iter 10000, auto-selected):
  nominal survival **85.9%** (need ≥99%), 40ms+push survival 87.5% (need ≥95%),
  ankle p95 16.5 Nm (need ≤15). Drift was **SOLVED** (1.67 m → 0.81 m ✅) and pose
  fidelity is good (rr_mpkpe 0.09 ✅). Artifacts: `exports/train-thriller_v7ank-0715/`
  and `data/policies/thriller_v7ank/`.
- **Recurring failure signature across v5/v6/v7:** survival plateaus ~86–92%; the
  falls ALWAYS cluster at the same two dance beats (**13–18 s** and **25–36 s**),
  where the **ankle motors saturate** (hit the 50 Nm hard limit). Interpretation on
  file: at those beats the choreography demands a faster weight-shift than the ankles
  can deliver → runs out of authority → tips. **You cannot add torque with a reward
  weight**, so pure reward-tuning has plateaued.
- **GPU box:** DELETED (not billing). To train again we must re-provision (~1 h on
  GreenNode) OR migrate (see §4).
- **Robot:** DOWN — burnt DC-DC converter in the torso power bay (steps ~48 V →
  compute rails). Powered off, battery disconnected, support RMA pending. Independent
  of software; training can proceed without it.
- **Last rendered preview:** `data/previews/v7_thriller_reference_vs_policy.mp4`
  (left = intended reference, right = v7 policy in the on-laptop sandbox). The policy
  under-reaches then topples ~15 s. **This sandbox is known-pessimistic (see §3.1).**

---

## 3. THE PROBLEMS (the meat — each with what's known vs. suspected)

### 3.1 The simulation "looks worse and offset" vs. the real robot  ⚠️ partly a KNOWN artifact
**User observation:** the G1 in simulation looks worse and offset compared to the
original `thriller_csv_ankle_penalty` video, which the real robot can mimic ~70%.

**What's actually going on (verified in code):** there are **TWO different sim
models in this project, and they don't match:**
- **Training + gate** use the **mjlab** G1 model (per-joint armatures, tuned gains).
- **The on-laptop sandbox/preview** (`tools/sim_sandbox.py`, `tools/sim_studio.py`)
  uses a **DIFFERENT model**: `third_party/mujoco_menagerie/unitree_g1/scene.xml`.
  The code itself prints: *"menagerie model != mjlab training model → sim
  UNDER-represents hardware."* A policy that trained against mjlab dynamics is being
  replayed on menagerie dynamics → it looks offset, washed-out, and falls early.

**So "looks worse/offset" is substantially a model-mismatch rendering artifact, NOT
proof the deployed policy is that bad.** Evidence: the SAME class of policy did ~70%
on real hardware. BUT — this cuts both ways: it also means **we have no on-laptop
preview that faithfully represents the trained policy.** (History: a true sim-to-sim
gate was attempted and abandoned because the menagerie model can't even hold a
static pose — it collapses at ~1.4 s with no policy at all. See PROJECT_STATE
2026-07-04.)

**Brainstorm targets:**
- Build a faithful on-laptop preview by loading the **mjlab** model/dynamics (or its
  exact MJCF + per-joint armature/gains) into the sandbox instead of menagerie.
- Or render the preview **on the training box** right after export (where the mjlab
  model exists) and pull the mp4 — no laptop model needed.
- Reconcile: what EXACTLY differs between mjlab's G1 and the real G1 and menagerie's
  G1 (armature, damping, friction, contact params, gear/torque limits)?

### 3.2 SUSPECTED HALLUCINATED RESULTS — the trust problem  ⭐ TREAT AS FIRST-CLASS
**User concern:** "I suspect the result is hallucinated so we can never get an actual
accurate measurement if it ever actually works."

**This is legitimate and is the most important problem to resolve.** Not necessarily
"the code fabricates numbers," but: **is the gate measuring something that
corresponds to reality?** Reasons the concern is well-founded:
- The gate (`cloud/sim_gap_check.py`) scores the policy **in the same mjlab sim it
  trained in.** If that sim doesn't match the real robot, 86% survival is an
  internally-consistent number that may not predict hardware at all. Self-consistent
  ≠ true.
- There is currently **NO independent cross-check.** The intended independent
  verifier (sim-to-sim in a different engine) was abandoned (§3.1). So the gate is
  unverified against anything external.
- The project has a documented history of **measurement bugs** driving wrong
  conclusions (a mis-indexed sim readout burned a day; an `np.diff(x, 0)` — diff
  zero times instead of `axis=0` — produced a fake "16× too fast" motion alarm this
  week). So "trust but verify the measurement tooling" is earned, not paranoia.
- The one piece of GROUND TRUTH we have — real robot doing ~70% on the ankle-penalty
  policy — has **never been tied back to a sim number.** We don't know what gate
  survival that real-70% policy scored, so we can't calibrate "gate % ↔ real %."

**Brainstorm targets (build a trustworthy measurement chain):**
- **Calibrate against the one real datapoint:** run the OLD `thriller_csv_ankle_penalty`
  policy (the one that did ~70% IRL) through the CURRENT gate. If it scores ~99% in
  sim but 70% IRL, the gate is optimistic and the ≥99% bar is meaningless — recalibrate
  the bar to reality.
- **Replay real hardware obs through the sim** ("`--mode read`" logs exist in the
  deploy runtime; PROJECT_STATE calls this the owed "trust gate"). Compare sim obs vs.
  real obs step-by-step to quantify the sim2real gap directly.
- **Audit the gate end-to-end:** independently re-run it, confirm it actually loads and
  steps the exported ONNX (not a cached/canned result), confirm the obs construction
  matches deploy exactly (the 160-dim vector; IMU velocimeter lever-arm at
  `imu_in_pelvis`; per-joint action_scale). Commit script + raw output (project rule).
- Decide honestly what the gate CAN and CANNOT claim (it's a robustness/generalization
  check in ONE model — label it as such; the real gate is robot day).

### 3.3 Motion pipeline produces impossible / floaty / loose motion  ⚠️ real, upstream of everything
**User observation:** the motion detection + sim pipeline produce lots of impossible
and loose motion — "floating in the air," drifting, not grounded, not respecting
friction/balance/torque.

**What's known:** the front of the pipeline (GVHMR pose-est → GMR retarget) is noisy
and physically unconstrained. There ARE mitigations already built, but the user is
still seeing bad motion, so they're insufficient or mis-wired:
- `pipeline/grounding.py` — grounds motion at retarget intake + in vetting (fixes
  feet-in-floor / floating-standing).
- `pipeline/prep_motion.py` — de-jitter (spike rejection + Savitzky–Golay smoothing),
  velocity clamp, activation ramp.
- `pipeline/vet_motion.py` — feasibility vetting (joint speed limits, drift, smoothness).
- `tools/motion_feasibility.py`, `tools/motion_quality.py` — feasibility/quality metrics.
- `pipeline/retarget_gvhmr.py` — GMR retarget with velocity-limit option.

**Key distinction the next agent MUST hold:** "floating/impossible" motion can come
from THREE different places, and the fix differs for each:
1. **Source-motion error** (GVHMR/GMR gives feet-off-ground, root sliding, impossible
   joint speeds) → fix in retarget/grounding/vet BEFORE training. This is kinematic;
   there's no physics here — the reference is just a sequence of poses, and nothing
   stops it floating.
2. **Sandbox-model mismatch** (the reference LOOKS floaty in the menagerie preview) →
   that's §3.1, a rendering artifact, not a motion defect.
3. **Policy behavior** (the trained policy drifts/under-reaches) → that's the RL
   recipe / sim-fidelity problem.
The user's phrasing mixes all three. **Separating them is step one of the brainstorm.**

**[CHECK] Important nuance for the user's framing:** the *reference motion* is
KINEMATIC (a puppet with no physics) — it can and will "float" because nothing
enforces gravity/contact on it; that's expected and the vetting is what's supposed to
catch impossible references. It's the *policy* (trained in physics) that must respect
friction/torque/balance. So "make the reference respect friction" isn't quite the
right target — "reject or repair references the robot can't physically achieve, and
train the policy in a sim whose friction/contact/torque match the real robot" is.

### 3.4 Match the real robot's environment (friction / balance / torque)
**User ask:** mimic the actual robot's environment — real friction, balancing, torque
to stay still (not float/drift), with ankle penalty.

**Status:** mjlab training DOES model friction, contact, torque limits, and balance
(that's the whole point of physics-based RL) and DOES include the ankle_torque_l2
penalty + domain randomization (mass, friction, latency, pushes). The open question
is **fidelity**: do mjlab's friction/contact/armature/torque-limit params match the
REAL G1? This is the core sim2real question and directly feeds §3.2's trust problem.
Relevant known facts on file: mjlab uses per-joint armatures (0.0036–0.025+) with
gains matched to them; `base_lin_vel` is NOT directly measurable on the real robot
(needs a state estimator) — a real sim2real hole to flag for deploy.

### 3.5 kp/kd (PD gains) as a lever
**User ask:** change kp/kd if needed to shift, and for physically-impossible motion,
mimic as close as possible without breaking other systems / causing offset / damage.

**Critical caution for the next agent:** kp (stiffness) / kd (damping) are the PD
gains that turn the policy's target angles into motor torque. **They are shared
between training and deploy** — per BeyondMimic, the SIM gains ARE the deploy gains
(`policy_meta.json` carries per-joint kp 14.3–99.1, kd 0.91–6.31, effort limits
5–139 Nm, impedance model kp=armature·(2π·10)², kd=2·ζ·armature·2π·10, ζ=2). So
**changing kp/kd is not a free knob** — change them in training and you must deploy
the same values, and they interact with the armature model and the torque limits.
Raising kp to force-track an impossible motion can INCREASE peak torque → more ankle
saturation → the exact failure we have. This lever must be explored carefully, tied
to the motion-feasibility work, and re-validated end to end. Do NOT let a brainstorm
casually "just raise the gains."

### 3.6 UI: landmark-mapping preview for debugging  ✅ concrete, achievable feature
**User ask:** when a user uploads a video, be able to SEE the actual landmark mapping
(pose-estimation overlay) for debugging. Specifically: make the preview **side-by-side**
under the Unitree preview, so clicking another preview shows the **landmark video
mapping** overlaid on the example dance/practice video.

**What this means concretely:** render the GVHMR/pose-estimation output
(2D/3D keypoints + skeleton) as an overlay on the ORIGINAL uploaded video, and add a
second synced player next to the robot preview in the frontend so the operator can
visually debug "did pose-estimation even track the dancer correctly?" — the earliest
and cheapest place to catch garbage-in.

**Where to build it:**
- Backend/pipeline: GVHMR runs in `pipeline/retarget_gvhmr.py` / `pipeline/stages/
  *_motion.py`. Need to (a) confirm whether GVHMR's per-frame keypoints are saved
  anywhere today (SMPL params are; 2D landmarks may need to be dumped), (b) render an
  overlaid mp4 (draw skeleton on source frames — reuse the imageio-ffmpeg encoder;
  NOTE: **no system ffmpeg on this laptop** — must use
  `imageio_ffmpeg.get_ffmpeg_exe()`, as we just did for the preview render).
- Frontend: `ui/frontend/src/components/robot-preview.tsx` and
  `ui/frontend/src/screens/pipeline.tsx` / `dances.tsx` — add a side-by-side player.
- This is independent of the training problems and can be built while the robot is
  down / while brainstorming the harder items.

---

## 4. THE BIG STRATEGIC ASK — migrate off GreenNode / go Isaac Lab or mjlab "native"

**User ask:** migrate the training env to Isaac Lab/mjlab native, find a workaround
for the GreenNode cloud, so we can train with better precision.

**⭐ THE KEY UNLOCK the next agent must know:** earlier this session the user
confirmed **the NEW dev PC has an NVIDIA/CUDA GPU.** The ENTIRE reason we're on
GreenNode's constrained image (and therefore on mjlab instead of Isaac Lab) is that
**this laptop has no GPU.** A local GPU **removes the GreenNode constraint entirely**
and makes this whole request feasible. This reframes the ask:

**Option A — mjlab locally on the new PC's GPU (lowest risk, recommended first step).**
Same framework we already use and trust, but: no fixed cloud image, no Docker
restriction, no billing, no idle-box waste, instant iteration, full control of the
model/params. Almost everything in `cloud/` already works; it just needs to point at
a local GPU instead of the GreenNode box. **This likely gets "better precision"
faster than an Isaac Lab port**, because it removes the friction without a rewrite.

**Option B — Isaac Lab / BeyondMimic native on the new PC (higher fidelity, higher
effort).** The originally-intended stack. Pros: it's what the reference research uses;
potentially higher-fidelity contact; a second independent physics engine would ALSO
give us the independent cross-check we lack (§3.2). Cons: needs a capable GPU
(**[CHECK] VRAM — Isaac Sim wants ≥8 GB, comfortably more; get `nvidia-smi` from the
new PC**), a big Omniverse/Isaac Sim install, Docker or a heavy native setup, and a
port of our custom recipes/gate from mjlab to Isaac Lab conventions. This is a
multi-day migration, not a config change.

**[CHECK] "native" — the user says "Isaac Lab/mjlab native."** These are two
different frameworks, not one thing. Clarify with the user which they want: (A) mjlab,
just run locally/natively (no cloud), or (B) actually switch engines to Isaac Lab.
Recommendation to pitch: **do A now (fast win, removes the cloud pain, better
iteration), and evaluate B as a parallel fidelity experiment** — running the same
motion in BOTH engines is also the independent cross-check that would resolve the
"hallucinated results" fear.

**What we need from the user to decide:** `nvidia-smi` output from the new PC (GPU
model, VRAM, driver). This gates everything above.

---

## 5. FILE MAP (precise starting points)

| Area | Files |
|---|---|
| RL recipes (custom on top of mjlab) | `cloud/sim2real_task_v5.py`, `_v6.py`, `_v7.py`, base `sim2real_task.py` |
| Training launchers / curriculum | `cloud/train_v7_curriculum.sh`, `cloud/run_attempt4.sh`, `cloud/pick_checkpoint.py` |
| The GATE / verification | `cloud/sim_gap_check.py`, `cloud/heldout_eval.py`, `cloud/export_policy.py`, `pipeline/mjlab_verify.py` |
| Motion pipeline (front) | `pipeline/retarget_gvhmr.py`, `pipeline/grounding.py`, `pipeline/prep_motion.py`, `pipeline/vet_motion.py`, `pipeline/motion_io.py`, `pipeline/stages/{cloud,local}_motion.py` |
| Motion metrics | `tools/motion_feasibility.py`, `tools/motion_quality.py` |
| On-laptop sim/preview (⚠ menagerie model) | `tools/sim_sandbox.py`, `tools/sim_studio.py`, `tools/render_deploy_sim.py` |
| Deploy runtime (50 Hz, obs contract, PD, safety) | `pipeline/deploy_runtime.py` |
| App backend / frontend | `ui/server.py`, `ui/desktop.py`, `ui/frontend/src/{screens,components}/` |
| Live status / decisions | `PROJECT_STATE.md`, `logs/jobs.md`, `docs/FIELD_GUIDE.txt` |
| Trained policies | `data/policies/thriller*/` (incl. `thriller_csv_ankle_penalty` = the ~70%-IRL one) |

---

## 6. HARD CONSTRAINTS & GOTCHAS (do not relearn these the hard way)

- **Safety (non-negotiable):** no low-level robot commands unless motion passed sim
  verification AND a human is physically present with the damping remote AND typed
  DEPLOY confirmation. No hardware e-stop exists — only remote B-damp + power switch.
  Robot is down anyway, so this is moot short-term, but it bounds any deploy idea.
- **Never modify `~/robot/`** — that's the working teleop setup (robot IPs/creds).
- **Measurement discipline (project rule):** never call a finding decisive without an
  independent cross-check; commit the measurement script AND its raw output. Directly
  relevant to the "hallucination" audit.
- **Cost model:** GreenNode bills creation→deletion (Stop still bills). We've twice
  burned ~$13 on idle boxes. Migrating to a local GPU eliminates this class of waste.
- **Version pinning:** mjlab needs pinned `mujoco-warp==3.10.0.1` + `warp-lang==1.14.0`
  + torch cu128 (unpinned installs pull versions that CUDA-crash at env reset). Any
  new training env must reproduce this lock (`cloud/env_lock/requirements.lock.txt`).
- **`MUJOCO_GL=egl`** is needed for RENDERING but CRASHES training (GL/Warp CUDA
  clash) — set it only for the verify/preview step, never during training.
- **`num_envs` defaults to 1** in mjlab — must pass 4096 explicitly.
- **No system ffmpeg on the laptop** — use `imageio_ffmpeg.get_ffmpeg_exe()` to encode.
- **Checkpoint sort must be NUMERIC** (model_500 vs model_3999 lexical bug cost a stage).
- **Export the BEST checkpoint, not the last** — late training can collapse (v7's last
  ckpt was 3% survival); `cloud/pick_checkpoint.py` screens the last several.

---

## 7. OPEN QUESTIONS TO BRAINSTORM WITH THE USER

1. **Migration:** mjlab-local (fast) vs. Isaac Lab (fidelity + independent check) vs.
   both? — gated on the new PC's `nvidia-smi`.
2. **Trust:** what's the minimum viable "trustworthy measurement" before we believe
   ANY sim number again? (calibrate old-70%-policy through the gate; replay real obs?)
3. **The survival wall:** motion-surgery on the two hard beats (13–18 s, 25–36 s) to
   drop peak ankle torque under 50 Nm — vs. keeping the choreography and accepting a
   lower bar (see §8). These interact.
4. **Faithful preview:** load the mjlab model into the laptop sandbox, or render on
   the box at export time?
5. **Landmark-overlay UI:** does GVHMR already emit 2D keypoints, or must we add that
   dump?

---

## 8. NOTE on the user's separate request to relax the standard

The user separately asked to **lower the training bar to ~95%** (they said "~95% on
p95 is universally acceptable"). Two things for the brainstorm:
- **Terminology:** "p95" is a *percentile of a distribution* (we use it for ankle
  torque — the 95th-percentile torque value). The pass/fail thing the user likely
  means is the **survival threshold** (currently ≥99%). Worth clarifying which bar to
  move. Both are just thresholds in the gate config (`cloud/sim_gap_check.py` gate
  block) and are trivial to change.
- **Substance:** lowering survival 99%→95% is defensible for a first live show (95%
  ≈ falls ~1 in 20 full runs) and would make v6/v7 much closer to passing — BUT it is
  meaningless if the gate itself isn't trustworthy (§3.2). **Fix trust first, then set
  the bar against reality.** A 95% sim bar on an optimistic sim could still be a 70%
  real robot. Recommend: calibrate the gate to the one real datapoint, THEN choose a
  bar that maps to an acceptable real-world fall rate.

---

*End of handoff. Ground every load-bearing claim in the referenced file + raw output
before acting on it — this project has been bitten by confident-but-wrong measurements
more than once.*
