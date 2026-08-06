# AGENT HANDOFF — generate an end-to-end study guide for Alois

**You are an agent receiving this handoff.** Your task: produce a comprehensive,
end-to-end STUDY GUIDE that takes Alois from fundamentals to full command of
every domain this project touches. He built and operates this system but has
been working intuition-first ("vibing") — the guide must connect theory to the
EXACT bugs, fixes, and artifacts listed below, because every concept here has a
real incident he personally debugged attached to it. That is the pedagogical
gold: teach each topic THROUGH its incident.

**Learner profile:** hands-on builder; owns a Unitree G1 EDU (29 DoF); operates
the full pipeline; comfortable with terminals/git/Python; wants mech-e + CS +
AI/ML depth, from first principles up to the research frontier this project sits
on. Prefers concrete-first explanations, then formalism.

**Repo:** `github.com/mizuharaa/g1-dance-studio` — every file cited below exists.
Read `docs/PROJECT_STATE.md` (decision log, reverse-chronological) and
`docs/FIELD_GUIDE.txt` for narrative; `docs/HANDOFF_20260805.md` for recent state.

---

## 1. The system (so you can structure the guide end-to-end)

Dance video → **GVHMR** (monocular human pose → SMPL, cloud GPU) → **GMR
retargeting** (IK, human skeleton → 29-DoF robot joint trajectories) → motion
**cleaning / grounding / foot-flattening / feasibility vetting** (laptop) →
**RL motion-tracking training** (BeyondMimic-lineage task on mjlab/MuJoCo-warp,
PPO, GPU box) → **sim gate** (survival/drift/torque/amplitude/latency bars) →
ONNX export with a hash-pinned contract → **deploy runtime** (laptop Python,
CycloneDDS, 50 Hz PD control) or PC2 docker controller → **show app**
(FastAPI + React) with safety gates, kill switches, camera recording.

## 2. Topic map — teach each through its incident

### A. Computer vision & human motion estimation
- Monocular 3D human pose estimation; **SMPL/SMPL-X** body models (shape β,
  pose θ); world-grounded motion recovery (GVHMR).
- Video timing: VFR vs CFR, frame-rate resampling. *Incident:* uneven VFR
  dropped 15.5% of frames → judder at fast hits (`_reencode_30fps` fix,
  `pipeline/stages/cloud_motion.py`).
- Pose-estimation noise through double differentiation. *Incident:* GVHMR
  jitter made the torque model's ZMP "flail" (falsified-model saga, §C).

### B. Kinematics, retargeting, rotations
- Joint space vs task space; forward kinematics (used daily —
  `pipeline/grounding.py` FK for foot heights/angles).
- **Inverse kinematics as optimization**; objective design. *Incident:* GMR's
  IK has NO foot-orientation objective → planted feet tilted median 20°, p90
  55°, while the G1 ankle roll only reaches ±15° — mechanically impossible
  references; prime suspect for the 58–64% leg-amplitude collapse. Fix:
  `flatten_stance_feet` (stance-gated 2×2 finite-difference **Jacobian**,
  Gauss-Newton, joint-limit clamped) — 2026-08-06.
- Quaternion conventions (wxyz vs xyzw — a real transform bug), rotation
  matrices, tilt from R[2,2], yaw alignment of reference to robot heading.
- Retargeting morphology gaps; joint limits; degrade-to-limits philosophy.

### C. Dynamics, contact, actuation
- Floating-base dynamics; inverse dynamics τ = M q̈ + C + g − Σ Jᵀf (the
  **Jacobian transpose** mapping contact forces to joint torques) —
  `pipeline/motion_dynamics.py`.
- **ZMP / CoP / support polygon**; GRF distribution between feet. *Incident:*
  billing the whole ZMP moment to one ankle inflated torque demand; an
  inverted support-margin sign made every centred ZMP read "outside support".
- **Model validation against reality.** *The flagship lesson:* the torque-demand
  model predicted ankle p95 114 Nm; hardware telemetry measured 15–19 Nm — a
  6–10× overestimate that drove two over-conservative slowdown generations
  (`experiments/torque_crosscheck_20260720`). Teach: absolute vs relative model
  validity, cross-checking against independent measurement.
- Motors: PD control, kp/kd (sim gains ARE deploy gains here), effort limits,
  torque-speed derating (Isaac's curve vs mjlab's flat clamp), thermal RMS.
- Physics engines: MuJoCo vs Isaac contact/friction differences (mesh vs
  primitive collisions, tangential-only friction randomization).

### D. Reinforcement learning
- PPO; actor-critic; **asymmetric actor-critic**. *Incident:* privileged
  critic-only terms (base_lin_vel, anchor pos) were fed to the ACTOR, forcing a
  fake estimator on hardware — the July root-cause audit
  (`experiments/upstream_alignment_report.md`).
- Motion-tracking reward design: exp-kernel tracking terms and their σ;
  velocity-tracking terms; reward shaping vs **termination design** (HoloSoma's
  keypoint termination vs our reward stack — `cloud/sim2real_task_v14.py`).
- Reward-hacking/inertness: the audit that found the ankle recipe inert or
  INVERTED (DR overwrote a clamp; a barrier with zero gradient in-band) —
  `experiments/ml_audit_20260721/REPORT.md`, 12 confirmed defects.
- **Curriculum learning** (speed/latency/drift stages); late-stage collapse
  pattern (v7/v9 diverged, checkpoint-picker exists because of it).
- **Domain randomization**; units matter. *Flagship incident:* delay DR counted
  5 ms PHYSICS steps — "0–80 ms" was actually 60 ms max, zero margin at the
  robot's measured 40–60 ms latency (fixed 2026-08-06).
- Observation design: history stacking (770 = 154×5), estimator-free contracts,
  obs noise, delay injection ordering.
- Off-policy alternatives (FastSAC) and why we stayed PPO (reproduction reports
  of jitter).
- Sim2real: the latency cliff; **gate calibration against a hardware anchor**
  (the ~70%-IRL policy run through the gate to tie sim% to real%); style
  metrics (leg amplitude 58/64/93/96% — `experiments/style_gap_20260722`).

### E. Simulation stacks
- MuJoCo, MJCF, mujoco-warp (GPU), mjlab, Isaac Lab/Gym/Sim, menagerie models.
- Model fidelity: menagerie-vs-mjlab preview mismatch (policy "danced 7%" in
  the wrong model, 96–118% in the faithful one).
- Headless rendering (EGL) vs training CUDA contexts (the GL/Warp crash).
- Dependency pinning: mujoco-warp 3.10.0.2 device-assert regression → the
  pinned lock file (`cloud/env_lock/requirements.lock.txt`).

### F. Robotics middleware & deployment
- **DDS** pub/sub (CycloneDDS), `rt/lowstate` / `rt/lowcmd`, unitree_sdk2py;
  LowState anatomy (motor_state, imu_state, wireless_remote bytes).
- 50 Hz real-time loops: tick deadlines, work budgets, absolute-deadline pacing.
- IMU: quaternion attitude, gyro bias (we calibrate 1 s at stand and subtract).
- ONNX runtime inference; export contracts; metadata binding (meta v2:
  obs_per_frame/history/onnx_inputs — loader refuses mismatches).
- Estimation without sensors: no foot F/T sensor → support-torque contact
  proxy (tau_est), its blind spots, debouncing, upright gating. *Incident:* a
  weight shift false-tripped the 12 Nm contact bar while the robot stood
  perfectly upright (telemetry 20260805-172326).
- Latency measurement: command→response cross-correlation from telemetry.

### G. Safety engineering (this project's specialty)
- **Fail-closed design**: hash-pinned bundles, signed verdicts, launch refusals.
- Kill chains and their coverage: factory L2+B is INERT under custom control
  (hardware finding) → in-loop chord detection + independent watchdog process +
  LocoClient.Damp() RPC + power switch. Watchdogs (heartbeat, GIL caveat).
- Human gates: typed phrases, CONFIRMED_BY_HUMAN, staged validation ladders
  (gantry → tethered → free), kill tests before motion.
- Guards: action clamps, rate limits, estimator validity, entry contact gates.

### H. Software/system engineering (as lived here)
- Content-addressed artifacts (sha256 bundle manifests, byte-reproducible
  builds); provenance discipline (commit scripts AND raw outputs).
- Contract versioning (meta v1→v2 migration); editable installs and their
  fragility (*incident:* the SDK egg-link pointed at a moved folder — every
  robot run crashed for weeks, masked by upstream gates).
- Environment/Linux: NIC binding and DDS interfaces, NetworkManager profiles,
  udev/v4l2 cameras, pipewire device contention, `pgrep` self-match (bit us 4×),
  signal handling and process groups, conda env isolation.
- Testing: 700+ test suite, tests-pin-contracts philosophy, the pytest ffmpeg
  that squatted a real camera (test isolation from hardware).

## 3. The bug catalog (chronological case studies — use as chapter anchors)

1. Obs-contract root cause (critic terms in actor) → estimator hole.
2. Motion physically impossible (173 Nm demand vs 50 clamp) → feasibility gates.
3. Foot-float grounding bug (78% frames floating) → per-frame contact grounding.
4. Outlier rejector deleting real dance hits → choreography guard.
5. Torque model falsified 6–10× by hardware telemetry → relative-only use.
6. Drift is under-sway (policy 3.5× too STIFF), not jitter → honest p95 gate.
7. Preview lied: stale motion + 90° rotated reference pane.
8. Latency DR unit bug: physics steps ⇒ 60 ms ceiling, not 80.
9. Foot-tilt retarget defect: 20° median stance tilt vs ±15° ankle roll range.
10. Factory remote inert in dev mode → 3-layer software kill chain.
11. Dead editable install after folder move → weeks of masked breakage.
12. Contact-guard false trip during weight shift → upright-gated guard.
13. Gate blind spots: never measured leg amplitude or 60 ms survival → new bars.
14. v12 hardware debut: 25.3 s, legs 58%, lag 40 ms — anchor comparison method.

## 4. Key-term seed list (expand each in a glossary)

SMPL, GVHMR, GMR, retargeting, IK/FK, Jacobian, Gauss-Newton, quaternion,
ZMP, CoP, support polygon, GRF, inverse dynamics, floating base, PD control,
kp/kd, effort limit, torque derating, PPO, actor-critic, asymmetric critic,
privileged observation, reward shaping, exp-kernel tracking, termination,
curriculum, domain randomization, sim2real gap, latency DR, history stacking,
observation contract, ONNX, checkpoint selection, MuJoCo, MJCF, mujoco-warp,
mjlab, Isaac Lab, menagerie, EGL, DDS, CycloneDDS, LowState/LowCmd, IMU bias,
tau_est, contact estimation, debounce, watchdog, fail-closed, content
addressing, sha256 manifest, signed verdict, deploy gate, e-stop chain,
gantry validation ladder, telemetry, cross-correlation latency estimate.

## 5. Instructions for generating the guide

1. **Structure**: fundamentals → applied → project-specific, per domain (A–H),
   but OPEN each chapter with its incident from §3 as the motivating story.
2. **Depth**: undergrad-level math where needed (linear algebra for Jacobians/
   rotations, basic probability for RL, ODEs lightly); derive only what pays off.
3. **Exercises**: end each chapter with 2–3 tasks doable IN THIS REPO (e.g.,
   "recompute the foot-tilt stats with `flatten_stance_feet` disabled",
   "extract the latency estimate from a telemetry npz", "read the v14 selfcheck
   and predict what breaks if a term is renamed").
4. **Verify against the repo** — every claim about the project must match the
   files; cite paths. Do not invent history; `docs/PROJECT_STATE.md` is truth.
5. **Resources**: pair each domain with 1–2 canonical references (e.g., Lynch &
   Park *Modern Robotics* for B/C; Sutton & Barto + PPO paper for D; MuJoCo
   docs for E; DDS spec/Unitree SDK for F) — but the repo is the primary text.
6. **Length**: comprehensive but studyable — suggest a chapter order with
   rough hour estimates and a "fast path" for pre-robot-day review.
7. Output as `docs/STUDY_GUIDE.md` (or a small set of per-domain files).
