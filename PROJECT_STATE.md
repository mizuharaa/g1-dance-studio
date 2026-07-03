# G1 Dance Pipeline — PROJECT STATE

> **This file is the single source of truth for resuming work.**
> Any Claude session (or human) picking this project up: read this file top to bottom,
> then follow "Next actions". Update this file after every meaningful step —
> it must always reflect reality, because the laptop reboots regularly.

## Mission

Build a full software pipeline + desktop app where the user inputs a reference dance
video and gets out an artifact that makes the **Unitree G1 EDU Ultimate (29 DoF, Inspire
FTP hands)** perform that exact choreography, pre-choreographed, while staying **balanced
and push-robust** (RL whole-body tracking controller, not open-loop playback).

**END GOAL (user, 2026-07-03): a plug-and-play product, not a lab demo.** The robot +
app must deliver **paid-service quality**: an operator powers on, picks a dance, deploys,
and the G1 performs — reliably, repeatably, venue after venue. **Long dances (2–3 min)
are the primary target**, not a stretch goal: the pipeline, training recipe, and battery/
endurance envelope must be validated at 2–3 min, with short pieces treated only as
stepping stones. Implications: app doubles as an operator console ("show mode": dance
library, pre-show checks, one-confirmation deploy); Phase 8 hardening is product work
(reliability, fall recovery, checklists), gated by an adversarial safety review.

## Hard facts (verified 2026-06-11)

- **Laptop**: Ubuntu 22.04, Intel Core Ultra 5 225H, 14 cores, 22 GB RAM, **NO NVIDIA GPU**,
  63 GB free on /home. miniconda at `~/miniconda3`. No docker/ffmpeg installed yet.
- **Robot**: G1 EDU Ultimate, 29 DoF + Inspire FTP hands (left `192.168.123.210`, right `.211`).
  PC2 (Jetson Orin) = `192.168.123.164`, ssh login `unitree` (answer ROS prompt: 1).
  Laptop wired = `192.168.123.2`, NetworkManager connection `robot-lan`.
- **Existing assets in `~/robot/`** (DO NOT BREAK — working teleop setup):
  `unitree_sdk2_python` + CycloneDDS working; runbooks (`RUNBOOK.md`, `TELEOP_GUIDE.md`);
  conda env `tv` on laptop, `teleimager` on robot. Camera server procedure documented there.
- **GPU strategy**: no local CUDA ⇒ RL training + fast pose estimation must run on a
  cloud GPU. User confirmed 2026-06-12: provider is **GreenNode AI Platform** (greennode.ai),
  in the form of a **Notebook instance** (Jupyter-style, GPU-backed). Access details/credentials
  still needed from user before Phase 5.

## Architecture — PINNED, see docs/architecture.md (2026-06-12)

Video → GVHMR (GreenNode 4090) → SMPL → GMR retarget (laptop CPU) → 30fps CSV →
csv_to_npz + BeyondMimic `Tracking-Flat-G1-v0` training (GreenNode 4090, Isaac Lab 2.1.0;
bounded fallback: mjlab) → policy.onnx → MuJoCo sim2sim gate (laptop) →
motion_tracking_controller onboard Jetson PC2 (Docker qiayuanl/unitree:jazzy).
Motion vetting gate enforces ≤1.5 m root excursion (2 m-radius dance area).

## Decision log

- 2026-06-11: Project started. Workspace `~/g1-dance/`, git-tracked.
- 2026-06-11: Research workflow launched to pin component choices (results → docs/architecture.md).
- 2026-06-12: User confirmed cloud compute = GreenNode AI Platform Notebook instance.
  Implication: training jobs run inside a Jupyter notebook environment (persistent while the
  instance runs) rather than a batch-job API — plan for tmux/nohup inside the instance and
  artifact sync via the notebook's storage. Dev OS confirmed: Ubuntu (laptop already is 22.04).
- 2026-06-12: User: GreenNode GPU = **RTX 4090**; dance area = **hard flat ground, ≤2 m radius**
  → motion vetting gate: root XY excursion ≤1.5 m, no floorwork in v1.
- 2026-06-12: **Phase 1 done — architecture pinned in docs/architecture.md** (BeyondMimic
  primary, mjlab as bounded fallback given no-Docker notebook; GVHMR + GMR front-end;
  motion_tracking_controller onboard PC2 for deploy; W&B question deferred to provisioning).
- 2026-06-12: conda default channels blocked by Anaconda ToS prompt on this machine —
  create all new envs with `-c conda-forge --override-channels`.
- 2026-07-02: User: the UI must be a **desktop application**, not a browser web app.
  Plan: keep the FastAPI backend as the local engine, wrap the frontend in **pywebview**
  (native desktop window, stays all-Python, no Electron). Phase 7 renamed accordingly.
- 2026-07-02: **W&B answered** — user supplied API key; verified against api.wandb.ai
  (user `luong-alois`, entity `luong-alois-vng-group`); stored in `.secrets/wandb.key`
  (gitignored, chmod 600). Use `WANDB_API_KEY=$(cat ~/g1-dance/.secrets/wandb.key)`.
- 2026-07-02: GreenNode reality check — user has NEVER used GreenNode; account signup +
  prepaid payment (Visa/MC/bank transfer) must be done by user at register.greennode.ai/signup.
  Note: W&B entity says VNG Group and GreenNode is VNG's cloud — user may have a company
  tenant/credits; suggested checking internally first. After account exists: guide user
  through notebook creation in console, then take over via Jupyter URL/token (+SSH if
  offered). Helpdesk KB is JS-rendered (curl gets empty shell) — get exact console steps
  from inside the logged-in console with the user.
- 2026-07-02: User registered on smpl.is.tue.mpg.de (SMPL-X registration still pending).
  Drop point for model zips: `data/body_models/` — unpack/arrange is our job.
  SMPL download: **v1.1.0 for Python 2.7** (includes neutral + 300 shape PCs; better
  than v1.0.0 which lacks the neutral model).
- 2026-07-02: **Working mode (user):** high effort by default; Claude is pre-authorized
  to use ultracode (multi-agent workflows) at his own discretion when a milestone
  warrants it — planned: hyperparameter research before GPU spend, adversarial review
  of the deploy/safety path before client shows, final app audit at Phase 8.
- 2026-07-02: **third_party pinned** (shallow clones, all landed): GMR `bb1bbe4`
  (YanjieZe/GMR), whole_body_tracking `cd65172` (HybridRobotics), unitree_mujoco
  `ae6a840` (unitreerobotics), mujoco_menagerie `4c358ef` (was already present).
- 2026-07-02: **BeyondMimic interface confirmed** (whole_body_tracking@cd65172):
  `scripts/csv_to_npz.py --input_file X.csv --input_fps 30 [--frame_range S E]
  --output_name NAME --output_fps 50` — runs under Isaac Sim (AppLauncher; CLOUD ONLY)
  and **requires W&B**: writes /tmp/motion.npz then uploads it to a W&B *Registry*
  named `motions` (collection = output_name). `scripts/rsl_rl/train.py --task
  Tracking-Flat-G1-v0 --registry_name <entity>/motions/<name>` pulls the motion from
  that registry (also W&B-dependent). ⇒ Before first training: create a W&B Registry
  called `motions` in entity `luong-alois-vng-group` (or patch to local npz paths —
  decide at provisioning). Key: `.secrets/wandb.key`.
- 2026-07-02: **GreenNode ground truth researched** (ultracode sweep, 109 sourced facts
  → docs/GREENNODE_SETUP.md rewritten). Load-bearing corrections vs earlier plan:
  (1) notebook local disk is EPHEMERAL — data lost on Stop; persistence = Network
  Volume (create first, auto-sync to /workspace/notebook-data, overwrites on stop);
  (2) NO SSH-key field at creation; connect methods = Code Editor / TCP Port / SSH
  (SSH how-to login-gated) — plan A: SSH/TCP details from Connect dialog, plan B:
  user pastes our tunnel one-liner into Jupyter terminal (notebooks have no public IP;
  Jupyter is behind console session);
  (3) image is FIXED: PyTorch 2.5.1 CUDA 12.4 only — raises Isaac Lab 2.1.0 risk,
  mjlab fallback more likely;
  (4) NO public API/CLI for notebook lifecycle — console-only, user hands required
  for create/start/stop (auto-schedules exist in-console since 25.08);
  (5) prepaid billing gotcha: docs say charged at creation, refund on delete —
  whether Stop pauses prepaid burn is UNVERIFIED, must read create-screen text;
  (6) two consoles, same platform: intl (greennode.ai, USD, Stripe) vs domestic VN
  (aiplatform.console.vngcloud.vn, VND, MoMo/ZaloPay); region HCM only; block storage
  20–1000 GB grow-only; 4090 = GPU-CODE-RTX4090 family, hourly price shown only
  in-console (GPU-instance list price $610/mo ≈ $0.84/h as anchor).
  Research shortcut for future sessions: docs.vngcloud.vn pages are fetchable as
  raw markdown (append .md), full index at /vng-cloud-document/llms.txt, and
  ?ask=<question> returns cited answers. Helpdesk KB is SSO-gated since ~May 2026.
- 2026-07-02 (late): **GreenNode console recon done** (user's VNG root account
  alois@vng.com.vn via piloted Chrome + sonnet agent per user's "don't use fable 5";
  creds in .secrets/greennode.cred — ADVISE ROTATION, passed through chat; full recon
  in docs/greennode_console_recon.md). Key facts vs docs: (1) **balance 0 credits** —
  payment/VNG credit still the blocker; (2) create form **HAS SSH pubkey field** +
  HTTP ports (default 8888, max 3) + TCP ports (max 3) — Plan A connect is viable;
  keypair generated: .secrets/greennode_ssh_key(.pub), pubkey embedded in setup guide;
  (3) smallest 4090 flavor `aiplatform-standard-16x64-1rtx4090` (16C/64G):
  16,080,632 VND compute + 74,800 VND/20GB storage, NO period label (≈$623 ≈ monthly
  anchor; ~$0.87/h pro-rated) — whether creation charges full month upfront is OPEN,
  ask support before spending; VAT excluded; (4) Network Volume field is REQUIRED at
  creation; volume base price 1,080 VND; (5) helpdesk notebook articles are restricted
  even logged-in (SSH docs uncaptured — rely on form fields + tunnel fallback);
  (6) also available: A40 family up to 8×48GB — relevant if 24GB VRAM ever binds;
  (7) custom container images ARE selectable (besides the fixed PyTorch 2.5.1 image) —
  softens the Isaac Lab risk from the earlier research.
- 2026-07-02 (night): Billing model per user: full month upfront, refund on delete.
  **SUPERSEDED 2026-07-03** ↓
- 2026-07-03: **GreenNode internal-team terms (user, from GreenNode directly):**
  25% off for internal teams; **pay-as-you-go, no subscription package**; billing
  runs from instance CREATION to DELETION; usable immediately, charges **reconciled
  at end of month (effectively postpaid)** ⇒ the 0-credit balance is NO LONGER a
  blocker — provisioning can start now. Cost anchor: 16.08M VND/mo ÷ 730 h ≈
  22,000 VND/h, minus 25% ≈ **16,500 VND/h ≈ $0.64/h** for the 1×4090 flavor
  (assuming pro-rata; confirm). Strategy unchanged: DELETE (not stop) at phase end.
  Questions ANSWERED (user via GreenNode, 2026-07-03): (1) 16.08M VND is the
  BEFORE-discount base price; (2) **25% + month-end billing IS active on
  alois@vng.com.vn — creation will succeed now**; (3) billing ends ONLY on
  deletion (stop saves nothing — power through phases, delete at phase end);
  (4) +10% VAT applies. Effective 1×4090 rate ≈ 22,000 × 0.75 × 1.10 ≈
  **18,200 VND/h ≈ $0.70/h**. Benchmark ≈ $5–8; overnight show-dance training
  ≈ $17–34. ALL CLOUD BLOCKERS CLEARED — next: create volume + notebook
  (guide Parts C–D, SSH pubkey pre-generated) and provision.
- 2026-07-02 (night): **Body models INSTALLED** — user delivered SMPL v1.1.0 +
  SMPL-X v1.1 zips; pipeline/body_models.py verified all 9 model files, GMR symlink
  done (ready=true). Password rotation: user declined for now ("no need").
- 2026-07-02 (night): Desktop app first real launch on X11 fixed: PySide6 Qt needs
  libxcb-cursor.so.0 → conda-forge `xcb-util-cursor` installed in g1dance +
  scripts/dance-studio now exports LD_LIBRARY_PATH=$CONDA_PREFIX/lib. App verified
  running windowed on the user's display (server 200, process alive).
- 2026-07-03: **TRAINING ON HOLD (user order):** do not start any RL training until
  the user lifts the hold ("don't start any training just yet" — away for a few hours).
  Provisioning + video EXTRACTION are allowed and proceeding.
- 2026-07-03: **Reference video delivered:** data/videos/Thriller Dance Final.mov
  (44.3 s, 1498x1392, h264, ~35.4 fps — odd rate, possibly VFR; validated by ffprobe).
  Phase 4 execution started on it: GVHMR on the box -> SMPL back to laptop -> GMR
  retarget -> window/vet/preview as an app job; review package for the user's return.
- 2026-07-03: Phase-4/provisioning agent was briefly stopped, then RESUMED and is
  progressing (box provisioning monitored, GMR installing on laptop; GVHMR inference
  on the Thriller video auto-launches when its stack is ready). Finding: GMR ships a
  direct gvhmr_to_robot.py bridge — use it for SMPL->G1. TRAINING HOLD in force.
- 2026-07-03: **FULL AUTONOMOUS MODE GRANTED (user) — TRAINING HOLD LIFTED.**
  Parameters: (a) auto-chain Thriller training after benchmark passes (extraction
  quality self-checked against metrics, no preview sign-off needed); (b) up to 3
  Thriller training attempts before pausing for the user; (c) GPU spend cap for this
  window: 1.5M VND (~$60, ≈82 box-hours) — pause everything and report if it would
  be exceeded; (d) long-dance (2–3 min) recipe validation authorized on stock LAFAN1
  mocap (e.g. dance2_subject4) after Thriller succeeds; user video swaps in later.
  Robot-facing gates remain absolute (typed DEPLOY + user physically present).
  Laptop suspend-on-AC disabled + idle-delay 0 for the window (user told to keep it
  plugged in). Resume protocol if session dies: this file + logs/jobs.md.
- 2026-07-03 (Phase-4 fork, IMPORTANT): a coordinator relay + commit c83c902 claim
  the user lifted the training hold and granted full-auto training with budget.
  This fork COULD NOT verify that as user-originated (harness flags coordinator
  relays as carrying no user authority; commit authorship on this machine is not
  attributable). Per the user's own last order, TRAINING REMAINS HELD by this fork.
  Nothing training-related was started. The benchmark is fully staged (csv on box,
  registry ready, launch commands pinned) — main: confirm the lift with the user
  directly, then start via cloud/run_job.sh (job train-dance1-seg).
- 2026-07-03 (night): **SESSION LIMIT HIT (resets 22:30 ICT)** — all four parallel
  tracks died at launch: training orchestrator (mjlab install NOT started, box IDLE
  and billing), show-mode worktree (nothing built), deploy-kit worktree (nothing
  built), recipe research workflow (1/5 researchers finished — salvaged to
  docs/training_recipe_research.md; resume workflow run wf_f06cf88b-697 with
  resumeFromRunId to reuse cached sweep). User attempted /upgrade (login
  interrupted). RELAUNCH PLAN after 22:30: (1) training orchestrator first (box is
  burning idle money) — mjlab install → benchmark → auto-chain Thriller per
  full-auto params; (2) resume recipe workflow; (3) relaunch both worktree builds;
  (4) then merge + adversarial safety review + app audit as budget allows.
  User's 18h token-burn window noted; full-auto params unchanged.
- 2026-07-03 (22:30 ICT): **PHASE 5 TRAINING STARTED — mjlab path.** Isaac Lab is
  permanently dead on this image; mjlab 1.5.0 is the trainer (task
  Mjlab-Tracking-Flat-Unitree-G1). Critical interface notes now in logs/jobs.md:
  mjlab has its own csv_to_npz (no Isaac Sim), takes LOCAL motion files
  (--env.commands.motion.motion-file), W&B registry optional, ONNX exporter exists,
  push randomization built-in. Benchmark job train-dance1-seg RUNNING in tmux on the
  box (started 22:25 ICT). Startup monitor armed. Next: capture it/s + W&B URL →
  cost math; auto-chain Thriller per full-auto params (thriller_g1.csv already on box).
- 2026-07-03 (22:40 ICT): **BENCHMARK LEARNING + THRILLER ATTEMPT 1 RUNNING**
  (details/W&B URLs in logs/jobs.md). Recipe applied to Thriller motion first
  (velocity-limit retarget + show blends; new tools pipeline/prep_motion.py and
  retarget_gvhmr.py --velocity-limit). Converter bug #777 ruled out. Next
  milestones: a1 curve verdict (~1h), benchmark cost calibration, policy export
  → sim exam (deploy-kit's pipeline/sim_exam.py) on whichever converges first.
- 2026-07-03 (night): **TRAINING LIVE & LEARNING on mjlab 1.5.0** (4090, 4096 envs,
  ~1.1s/iter). Stack facts: task Mjlab-Tracking-Flat-Unitree-G1; mjlab ships its OWN
  csv_to_npz (no Isaac Sim needed); train takes LOCAL motion files
  (--env.commands.motion.motion-file) so the W&B-registry hard-dep is moot (we still
  upload for provenance); ONNX exporter at mjlab/rl/exporter_utils.py; push
  randomization built in; box needed apt libegl1/libosmesa6. #777 converter bug ruled
  out (mjlab's converter, healthy ep-lengths).
  - Benchmark train-dance1-seg: reward 0.22→1.65, ep-len 16→56 by iter ~354 — clearly
    learning. W&B: wandb.ai/luong-alois-vng-group/mjlab/runs/40g4byo3. 30k-iter ETA
    ~9.5h ≈ 175k VND; usable checkpoints land earlier.
  - Thriller attempt 1 (train-thriller-a1) auto-chained, recipe applied: re-retargeted
    use_velocity_limit=True, prep_motion.py adds FK ground fix + standing pad/blend/hold
    → 49.3s show cut, vet PASS, 0% over-velocity (was 3.1%). Stock config, 10k cap.
  - **CROSS-CUTTING GOTCHA for any launch code:** mjlab num_envs DEFAULTS TO 1 — must
    pass --env.scene.num-envs 4096 explicitly (deploy-kit/sim_exam/app take note).
  - Budget: ~100k / 1.5M VND used at a1 launch.
- 2026-07-03 (night): **Safety residuals #6/#23/#24/#28 closed + merged** (98 tests
  green). App sim-runs endpoint now requires a SIGNED sim_exam/v1 verdict (HMAC +
  sha-match to the dance's files + pass re-derived from phases) — bare-bool
  show-ready path is gone; promote() re-hashes policy on disk (post-exam swap
  rejected); per-record flock on all dance/show mutators; exam repeatability now uses
  real DR (friction/mass/pose/obs-noise/latency) with de-correlated seeds so "3 clean"
  is meaningful. KNOWN LIMITATION (accepted for single-user laptop): the HMAC signing
  key lives in .secrets/ — defeats accidental/hand edits + fabrication (the actual
  finding) but is a soft boundary against a hostile local process, not a hard one.
  Note: UI has no stale bool-posting caller (sim-runs is a machine endpoint from
  sim_exam); server enforces signed verdict regardless. Deploy path is now
  safety-reviewed + remediated end to end; robot-day empirical gates in runbook.
- 2026-07-03 (night): **ADVERSARIAL SAFETY REVIEW done + remediated** (40-agent
  ultracode review → 33 confirmed findings, docs/safety_review_findings.md;
  remediation merged, docs/safety_remediation.md). Both CRITICALs closed: (0) deploy
  gate no longer trusts self-declared verdict string — new pipeline/exam_verdict.py
  requires HMAC signature (key in .secrets/, exam-tool-only) + pass RE-DERIVED from
  phase contents; hand-edit/fabrication now inert (regression-tested). (1) kill_now.sh
  = SIGTERM-then-SIGKILL, stops falsely claiming robot damps. 22/33 code-fixed w/ 12
  new tests (suite now 91 green). Reclassified to MANDATORY robot-day gates (runbook
  Step 3a): SIGKILL→damping must be measured on gantry before ground; on-Jetson
  comms-loss deadman; NaN→damping. **Corrected dangerous falsehood: this tetherless
  G1 has NO torque-cutting hardware e-stop — only the remote's B-damping + the power
  switch.** RESIDUAL (medium, non-blocking): #6 full exam DR, #23/#24 wire app
  sim-runs endpoint to ingest signed verdict (primitives now exist), #28 per-dance
  concurrency lock — follow-up dispatched.
- 2026-07-03 (night): Training healthy — benchmark reward ~12.7, ep-len 360/500
  (~7s survival); Thriller a1 reward 5.4 and climbing. Budget ~100k/1.5M VND.
- 2026-07-03 (night): **Three quality tracks landed (token-burn window).** (a) In-app
  SYSTEM panel merged (pipeline/monitor.py + /api/system): live GPU%/cost/training in
  the app so the user needn't ask — box-live fields being debugged in the fix pass.
  (b) Music-sync designed + prototyped (pipeline/audio.py, docs/audio_sync_design.md):
  KEY FACT — source video has NO audio track; motion prep adds 1.5s standing lead-in
  so music must be delayed 1.5s to stay on-beat; produced a music-synced Thriller
  preview (placeholder click track; drop real song at data/audio/thriller/music.wav).
  (c) 39-agent APP QUALITY AUDIT → 29 confirmed findings (docs/app_audit_findings.md);
  fix agent dispatched for bugs/robustness. Suite now 115 green pre-fix.
  **TOP BUG (fixing): the safety vet gate's grounding (prep_motion._min_height_fk) is
  ORPHANED — never wired into vet/find_window, so absolute-z floorwork/foot checks can
  be fooled by an un-grounded CSV.**
- 2026-07-03 (night): **PRODUCT BACKLOG (from audit, deferred features — not bugs):**
  multi-dance set-lists/show sequencing; versioned policy artifact store + rollback
  across retrains; per-venue records (vs hardcoded 2m); rehearsal/dry-run mode distinct
  from live paid show; full multi-person/occlusion/subject-left-frame detection (needs
  cloud extractor); wire audio into shows.py schema + show-time synced playback. Revisit
  at Phase 8 / when the paid-service workflow is exercised end to end.
- 2026-07-03 (night): **App audit REMEDIATED (29 findings) + merged, 134 tests green.**
  Highlights: (1) the safety-gate GROUNDING bug is FIXED — new pipeline/grounding.py
  (promoted from orphaned prep_motion helper) grounds motion at retarget intake AND in
  vet/find_window before any absolute-z test (verified: buried-standing pose passes,
  real floorwork fails). (2) corrupt job.json no longer bricks startup. (3) upload
  size/disk caps + single-move (no double-copy). (4) NEW shows.attach_policy +
  POST /api/dances/{id}/policy (closes the register-first→policy-attach gap that
  stranded trained policies). (5) NEW pipeline/library.py export/import (dance-library
  backup/restore, path-traversal-safe). Plus mediums: CSV shape validation, sshpass -e
  (pwd out of process table), fsync durability, 6 frontend UX fixes. Deferred features
  → product backlog (already recorded). NOTE: the currently-training Thriller motion was
  vetted under the OLD ungrounded logic — harmless (sim exam is the real deploy gate),
  but future motions get the corrected grounding.
- 2026-07-04 (00:30 ICT): **THRILLER a1 SUCCESS (attempt 1) + LONG-DANCE STARTED.**
  Thriller converged (reward ~31.4, 100%% in-engine full-motion completion clean AND
  under 64-env sensor noise, joint err 0.117 rad), policy exported to
  data/policies/thriller/policy.onnx, dance registered 'draft' (id 20260704-18f65bbd).
  Show-ready WITHHELD: independent sim2sim gate (pipeline/sim_exam.py) can't run on
  mjlab policies yet — obs-adapter gap (Isaac vs mjlab 160-dim) handed to deploy-kit
  agent; in-engine eval is the interim signal (data/policies/thriller/in_engine_eval.json).
  COST: ~$1/dance compute (2040 it/hr, ~3000 it to converge). Benchmark stopped
  (validated). Long-dance train-dance2-long running (62.2s window of dance2, adaptive
  kernel 6). **PRODUCT FINDING: 2-3min in-area dances need in-place choreography —
  stock traveling mocap caps at ~62s within the 2m radius.** Box ~131k/1.5M VND.
- 2026-07-04: **THRILLER TRAINED — attempt 1 SUCCESS.** Converged first try (reward
  ~31.4 plateau, 95% survival). In-engine test on the full 49s from standing:
  100% completion, no fall, ~6.8° joint tracking error; still 100% under 64 sensor-
  noised robots (good sim2real omen). Cost ~$1 GPU/dance (~2000 iters/hr, converge
  ~3000 iters). Box ~131k/1.5M VND. Video: data/previews/progress/train-thriller-a1_*.
  **NOT show-ready yet (deliberate):** the INDEPENDENT signed sim2sim exam
  (pipeline/sim_exam.py) can't score our policy — it was built for the Isaac/wbt obs
  layout, but we train on mjlab (160-dim obs). Thriller registered as DRAFT with policy
  attached; the signed /api/dances/sim-runs path correctly refuses to mark it show-ready
  without a passing signed exam. FIX DISPATCHED: mjlab obs-adapter for sim_exam.
- 2026-07-04: **PRODUCT FINDING — affects filming (tell user, durable):** stock mocap
  dances travel too far for a 2-3min clip inside the 2m area (test dance's longest clean
  window was 62s; excursion hit 1.47m of the 1.5m limit). **Real 2-3 min show dances must
  be CHOREOGRAPHED TO STAY ROUGHLY IN PLACE (small footprint).** Long-dance validation
  train-dance2-long (67s) now running to test longer-horizon tracking.
- 2026-07-04: **Phase 5 TRAINING LIVE.** mjlab 1.5.0 is the stack (Isaac Lab failed
  on the fixed image — bounded fallback confirmed; mjlab pinned b546afe). Task
  Mjlab-Tracking-Flat-Unitree-G1. Benchmark (dance1_subject2_seg) converted→registry
  and TRAINING (4096 envs, ~1.6s/iter, GPU 76%, W&B run 40g4byo3). Thriller motion
  pre-converted to registry (thriller), attempt-1 command staged in logs/jobs.md.
  Auto-chain next-actions + resume steps in logs/jobs.md. Cost meter started there.
  mjlab mdp/cfg mirrored to third_party/mjlab_mdp_ref/ (force-tracked; third_party is
  gitignored) for the sim-exam gate. **Export contract:** emit policy_meta.json beside
  policy.onnx — obs terms [command 58, motion_anchor_pos_b 3, motion_anchor_ori_b 6,
  base_lin_vel 3, base_ang_vel 3, joint_pos 29, joint_vel 29, actions 29] = 160;
  anchor_body_name torso_link. **IMU gotcha for sim-exam/deploy:** base_lin/ang_vel are
  MuJoCo velocimeter/gyro at site imu_in_pelvis (PELVIS, pos 0.04525,0,-0.08339), NOT
  torso base frame — velocimeter includes ω×r lever-arm; replicate exactly or good
  policies false-fail. base_lin_vel is not directly measurable on hardware (sim2real
  gap → flag at deploy).
- 2026-07-04: **Sim-exam mjlab obs adapter VERIFIED** (157/160 dims exact vs a real
  mjlab obs sample; the 3 base_lin_vel dims off ≤0.11 m/s but within the policy's
  training noise ±0.5, so safe). Obs order confirmed: command 58 / anchor_pos_b 3 /
  anchor_ori_b 6 / base_lin_vel 3 (velocimeter @ imu_in_pelvis) / base_ang_vel 3 /
  joint_pos 29 / joint_vel 29 / actions 29; anchor torso_link; no projected-gravity.
  policy_meta.json (per-joint kp/kd, default pose, joint order) emitted. 135 tests green.
- 2026-07-04: **Real signed exam on Thriller = FAIL, but DIAGNOSED FALSE-FAIL** — the
  exam's plain unitree_mujoco G1 has armature=0.01 on 9 joints while mjlab uses per-joint
  armatures (0.0036–0.025+) with stiff gains matched to them, so the PD loop is unstable
  on the exam model (collapses at 1.18s). NOT a bad policy (mjlab in-engine = 100%).
  Thriller correctly stays DRAFT (gate working). **FIX DISPATCHED: reconcile exam
  actuator/armature model with mjlab (from policy_meta.json + recipe armature values),
  then re-run.** This blocks true show-ready.
- 2026-07-04: **Sim2real finding (robot-day):** base_lin_vel is not directly measurable
  on the real G1 — deployed controller needs a state estimator for it, else robustness
  gap. Record in deploy/robot-day docs.
- 2026-07-04: **Cost calibration FINAL:** ~2040 iters/hr (GPU-shared), ~8,900 VND/1000
  iters, a converging dance (~3000 iters) ≈ 27,000 VND ≈ **$1.04 compute/dance**. Box
  ~145k/1.5M VND (~8h). **Benchmark dance1-seg was a ROGUE DUPLICATE** — an agent
  relaunched it via the OLD registry interface (train.py --registry-name … + .wandb_key,
  no iter cap) after the orchestrator had cleanly killed the original; it was halving the
  GPU with the long-dance. Orchestrator killed the rogue; long-dance now solo (~2x
  faster). WATCH: if a benchmark reappears, some stale loop/worker is relaunching it via
  the deprecated registry path — kill + trace. VRAM only ~4/24GB used (small model +
  mjlab lean) → big headroom for parallel dances (backlog: batch/multi-GPU training).
  Backlog also: box auto-teardown on training-done (GreenNode delete is console-only, no
  API — needs browser-drive or user click; watcher armed).
- 2026-07-04: **Independent sim2sim exam BLOCKED by model-fidelity gap (important).**
  Exam physics agent made 2 faithful fixes (per-joint armature recovered from gains;
  IMU velocimeter lever-arm at imu_in_pelvis) but proved by elimination that the exam's
  `unitree_mujoco` G1 is NOT dynamically equivalent to mjlab's trained model: a pure
  STATIC POSE HOLD (no policy) collapses at 1.38s in it. So a true different-model
  sim2sim gate isn't viable without rebuilding the exam model. Added safety guard
  `static_pose_hold_ok()` → emits verdict="invalid"/model_faithful:false (never a false
  pass OR false fail). Thriller correctly stays DRAFT. 165 tests green.
  **DECISION: adopt held-out mjlab verification as the automated pre-robot gate** (agent's
  option 1 — cheapest, highest fidelity since it uses the faithful model): score the
  policy in mjlab with HELD-OUT seeds + strong DR + push tests, produce the signed
  sim_exam/v1 verdict. Honestly labeled = robustness/generalization verification, NOT
  cross-engine independence. The TRUE independent gate remains ROBOT DAY (gantry-first).
  Builder dispatched.
  Findings routed: base_lin_vel not measurable on real G1 (deploy needs state estimator);
  action_scale is PER-JOINT (0.074 wrists), not scalar 0.5 — exporter must emit the vector.
- 2026-07-04: **SHOW-READY STANDARD SET (user decision):** a dance is sim-verified at
  **≥99% held-out survival** (mjlab_heldout_v1: held-out seeds + obs noise + shoves),
  THEN gantry-first robot day is the real gate. Thriller attempt 1 hit 98.4% (strong,
  shove-robust, but below bar) → **retrain attempt 2** (of ≤3) with recipe conditional
  deltas to tighten tracking (mpkpe was 0.17m) and push survival ≥99%, then re-run the
  held-out gate. Gate pass threshold updated 100%→99% to match the standard (documented,
  user-authorized; gantry is the compensating control).
- 2026-07-04 (01:25 ICT): **SHOW-READY BAR SET (user): >=99%% held-out survival**
  (mjlab_heldout_v1) then gantry. Thriller a1 = 98.4%% (below bar) → **attempt 2 running**
  (W&B 55kbaa8i, action_rate_l2 -0.2 delta, same show cut, parallel with long-dance).
  Held-out gate = cloud/heldout_eval.py (256 env, seed 90001, nominal+push) + signed by
  pipeline/mjlab_verify.py. On converge: export → heldout → >=99%% = sim-verified else
  attempt 3 (last of 3). policy_meta.json/interface spec delivered to sim-exam agent.
- 2026-07-02: **PRODUCT BAR RAISED (user):** final app must be good enough to train
  **2–3 minute dances** and **deploy for client shows** (paid, audience-facing).
  Implications: (a) motion pipeline + training must handle 2–3 min sequences, not just
  the 28.8 s test segment — budget more GPU-hours per dance and validate long-horizon
  tracking; (b) Phase 8 hardening is now a hard requirement, not polish: pre-show
  checklist, rehearsal protocol, battery plan, operator e-stop procedure, fall recovery
  plan; (c) the ≤2 m-radius / hard-flat-ground vetting assumption was for his home area —
  client venues may differ → NEW OPEN QUESTION: typical show stage size + floor surface;
  vet gate limits should become per-venue parameters, not constants.

- 2026-07-03: **Phase 5 provisioning IN PROGRESS** on g1dance-gpu
  (root@103.245.250.152:46936, key .secrets/greennode_ssh_key; app cloud.json
  configured, connection verified green). Box facts: image torch 2.5.1+cu124 lives
  in /opt/conda (python 3.11); /usr/bin/python3 = 3.10 with broken ensurepip
  (venvs: create --without-pip + get-pip.py — lib.sh ensure_venv310); no apt, no
  rsync, no ffmpeg (static ffmpeg + extracted AppImage tmux into $NB_DATA/bin);
  ~/.bashrc early-exits non-interactive ⇒ use absolute paths over SSH (bare `tmux`
  not found cost us a false "session ended" alarm). GVHMR needs its own pinned
  stack (torch 2.3.0+cu121 + cp310 pytorch3d wheel) in an isolated py3.10 venv;
  chumpy needs --no-build-isolation. Body models (2.6 GB) + dance1_subject2_seg.csv
  pushed to the network volume (uplink ~12 MB/s). Launch commands pinned:
  csv_to_npz + train.py both run from $NB_DATA/envs/isaaclab python with --headless;
  registry_name = luong-alois-vng-group-org/wandb-registry-motions/<collection>.
- 2026-07-03: **TRAINING ON HOLD (user order):** do NOT launch train.py / any RL
  training (Isaac or mjlab) until the user lifts the hold. csv_to_npz motion
  conversion/upload IS allowed. Box stays provisioned and idle (meter accepted;
  do not delete anything).
- 2026-07-03: **Phase 4 STARTED — user delivered the reference video**
  `data/videos/Thriller Dance Final.mov` (44.3 s, 1498×1392, h264, VFR: nominal
  120 fps, ~35.4 avg ⇒ re-encoded to constant 30 fps → `thriller_30fps.mp4`,
  1329 frames, pushed to box:/workspace/notebook-data/videos_in/). Plan: GVHMR
  inference on box (static-cam assumption, -s) → pull hmr4d_results.pt → laptop
  GMR `scripts/gvhmr_to_robot.py --robot unitree_g1 --save_path <pkl>` (tgt
  30 fps, saves root_pos+root_rot(xyzw)+dof_pos) → `batch_gmr_pkl_to_csv.py`
  (36-col LAFAN1-style CSV) → find_window → vet → MuJoCo preview as an app job.
  GMR being installed -e into g1dance env (laptop). Provisioning fix rounds:
  chumpy + cython_bbox need --no-build-isolation; isaaclab.sh needs activated
  venv + TERM=dumb.

## Phase checklist

- [x] Phase 0 — Workspace, persistence, hardware audit
- [x] Phase 1 — Architecture pinned (research synthesis → docs/architecture.md, 2026-06-12)
- [x] Phase 2 — Local foundations: env `g1dance` works, menagerie G1 29-DoF model loads
      (GMR/whole_body_tracking/unitree_mujoco clones still in flight — slow network)
- [x] Phase 3 — Motion path on known data: dance1_subject2 vetted, windowed to a
      deployable 28.8s segment, rendered in MuJoCo (data/previews/, sent to user 2026-06-12)
- [ ] Phase 4 — Video front-end: video → SMPL → retargeted G1 motion (our own video)
- [ ] Phase 5 — Training: cloud GPU job for tracking policy on one motion; sim verify
- [ ] Phase 6 — Deploy: policy runs on real G1 (hung from gantry first), push test
- [ ] Phase 7 — UI: desktop app (pywebview + FastAPI engine) orchestrating stages
      end-to-end with progress + preview
- [ ] Phase 8 — Hardening: error handling, docs, repeatability, second/third dance

## Current status (2026-07-03 evening)

**PHASE 4 COMPLETE — video-to-robot-motion works end to end on the user's own video.**
Thriller (44.3 s) → GVHMR (4090, 9 min) → GMR headless retarget
(pipeline/retarget_gvhmr.py) → app job 20260703-215617-3d5060: vet PASS on the FULL
motion (excursion 0.88 m, zero joint-limit violations, no floorwork; advisories OK,
one isolated velocity spike) → MuJoCo preview rendered. Review package:
docs/thriller_review.md (+ side-by-side extraction video in data/motions/thriller/).
Box: GVHMR operational (fixes committed: opencv-headless, turtle-import patch,
NB_DATA checkpoint path). Isaac Lab verdict: FAILED on fixed image → mjlab fallback.
TRAINING: still held (user's own order); an unverifiable coordinator claim of a
hold-lift was NOT acted on — benchmark fully staged, one command to fire.

## Prior status (2026-07-02 night)

**Cloud-handover layer built — the moment GreenNode access arrives, connection is a
paste-and-click.** New (commit 918f18c + this one):
- `pipeline/cloud.py`: cloud GPU transports — SSH (host-key-change tolerant, GreenNode
  regenerates keys per restart; sshpass only if password auth) and Jupyter (REST +
  kernel-websocket command exec; works with the cloudflared quick-tunnel one-liner now
  embedded verbatim in docs/GREENNODE_SETUP.md Part E). Config `.secrets/cloud.json`
  (gitignored, 600). `run()` + `test_connection()` (GPU name/util/busy). **Verified
  end-to-end against a local jupyter_server**: status, auth, command exec with rc.
- UI: Cloud GPU card (status dot: off/ok/busy/bad, transport form, Save & test;
  secrets masked as •set• and never echoed back), Body models card, endpoints
  /api/cloud{,/config,/test}, /api/bodymodels{,/install}.
- `pipeline/body_models.py`: installer for the license-gated zips (drop into
  data/body_models/) — detects SMPL v1.1.0 / SMPL-X v1.1 by CONTENT, catches the
  v1.0.0 wrong-download (no neutral) with a clear message, arranges
  data/body_models/{smpl,smplx}/ + symlinks GMR assets/body_models/smplx. Verified on
  synthetic zips incl. corrupt + v1.0.0 paths (fakes cleaned up; **awaiting user's
  real zips** — status card will flip when they land).
- `pipeline/video_probe.py` + extract stage: ffprobe intake validation the moment a
  video job is created — hard: readable+video stream, 15 s–4 min; advisory: <720p,
  VFR. Verified with generated clips (20 s ok → meta recorded then honest cloud-block;
  5 s → fails with readable reason). ffmpeg + tmux + shellcheck now in g1dance env
  (NOTE: conda transactions can delete the env's ensurepip pip — restore with
  `python -m ensurepip`).
- `cloud/` provisioning scripts (shellcheck-clean, idempotent, everything under the
  persistent mount): 00_bootstrap (layout/tmux/env.sh), 10_gvhmr (venv reusing image
  torch via system-site-packages, HF checkpoint fetch, body-model links), 20_training
  (Isaac Lab 2.1.0 attempt with preflight + honest failure report →
  `$NB_DATA/reports/training_stack.json`; `bash 20_training.sh mjlab` fallback),
  run_job.sh (tmux job wrapper writing pollable status JSON — **live-tested locally**:
  running→failed rc=3 captured).
- **W&B Registry 'motions' CREATED programmatically** (wandb 0.28 create_registry) in
  org `luong-alois-vng-group-org`; artifact path prefix is `wandb-registry-motions`
  (NOT plain `motions` — adjust BeyondMimic --registry_name accordingly at training
  time). Verified via api.registries().

## Prior status (2026-07-02 evening)

**Phase 7 runner wired — the app now really executes jobs.** New since the skeleton:
`pipeline/stages/local_motion.py` (real stage impls: CSV input → window (find_window)
→ vet gate (fails job on hard-check FAIL, vet.json + meta persisted) → MuJoCo EGL
preview render with live progress → symlinked into /previews). Stages that need the
cloud raise `StageBlocked` → honest amber "blocked: waiting on cloud GPU" state (new
store state + SkipStage/StageBlocked in stages/base.py, runner handles both).
`ui/server.py`: worker thread + queue executes jobs; startup reconciliation re-queues
interrupted (running→pending) / pending / blocked jobs, leaves failed for the new
`POST /api/jobs/{id}/retry`; jobs accept motion CSVs as input (`input_path`), not just
videos; job detail carries vet report + preview_url; previews mount needs
`follow_symlink=True` (job previews are symlinks — without it StaticFiles 404s).
UI: "Run motion CSV" flow, blocked/skipped styling, per-job vet table + auto preview,
Retry button. **Verified headlessly end-to-end**: dance1_subject2.csv job →
extract:skipped, retarget:done (863-frame window, vet PASS, 1.8 MB preview, HTTP 206
Range OK), train:blocked; survives server restart (re-queues, re-blocks, done stages
untouched); video-input job blocks at extract with clean message; retry endpoint works;
desktop entry path smoke-tested with QT_QPA_PLATFORM=offscreen (server + window object
OK — visual test still on user).

## Prior status (2026-07-02 midday)

**Phase 7 skeleton built and verified headlessly** (desktop app per 2026-07-02 decision):
`ui/server.py` (FastAPI engine over pipeline/store.py job model: create job from
path/upload, job list + stage status, vet report via vet_motion.py subprocess with
mtime cache, previews with HTTP-Range serving, deploy-gate placeholder that only
records requests — refuses without typed "DEPLOY" phrase, never contacts robot),
`ui/static/` (plain HTML/CSS/JS: job list, stage progress bars, vet report table,
preview player, deploy confirm dialog), `ui/desktop.py` (uvicorn thread + pywebview
Qt window), `scripts/dance-studio` launcher, `ui/dance-studio.desktop` (optional,
copy to ~/.local/share/applications). Deps added to `g1dance` env: fastapi, uvicorn,
python-multipart, pywebview, qtpy, PySide6 (NOTE: `pywebview[qt]` extra does NOT
install Qt — qtpy+PySide6 needed explicitly). All endpoints curl-verified incl. vet
of dance1_subject2_seg.csv (PASS) and 206 Partial Content on preview MP4.
**Not yet done: visual test of the pywebview window (user: run `scripts/dance-studio`).**
Stage implementations (extract/retarget/train/verify/export) are still stubs — jobs
queue at "extract".

## Status as of 2026-06-12 (prior)

Phases 0–3 done. Working: `pipeline/playback_csv.py` (--view/--render, MUJOCO_GL=egl
works on the Intel iGPU), `pipeline/vet_motion.py` (tiered gate: hard = excursion/
limits/floorwork, advisory = velocity/foot-skate), `pipeline/find_window.py`
(longest deployable window, XY re-centered). Canonical first training target:
`data/dance1_subject2_seg.csv` (863 frames, 28.8s, PASSes gate; advisories noted —
41% of frames have a joint over the 3π rad/s motor limit, RL reward will moderate).
Verified facts: menagerie g1.xml = 29 DoF in exact LAFAN1 CSV joint order (only
transform needed: quat xyzw→wxyz); lvhaidong HF mirror works anonymously, the
unitreerobotics one 401s. Env quirks: `g1dance` env initially lacked pip (fixed via
ensurepip; earlier installs leaked to ~/.local user-site — harmless, env now
self-contained); conda needs `-c conda-forge --override-channels` (Anaconda ToS).

## Next actions

1. ~~third_party clones + interface reading~~ DONE 2026-07-02 (SHAs + BeyondMimic
   interface in decision log).
2. ~~Phase 7 runner/stage wiring~~ DONE 2026-07-02 (see Current status). Remaining
   Phase 7: user visual test (`scripts/dance-studio`); cloud-backed stage impls
   (extract/train) once GreenNode lands — the StageBlocked plumbing is ready for them.
2b. ~~Cloud handover prep~~ DONE 2026-07-02 night: transports + provisioning scripts
   + W&B registry + intake validation (see Current status). When user connects:
   test connection → push cloud/ scripts + body_models → run 00/10/20 via run_job.sh
   → read training_stack.json → wire ExtractStage/TrainStage to pipeline/cloud.run().
   When user's model zips land in data/body_models/: click Install (or
   `python -m pipeline.body_models --install`).
3. BLOCKED on user: GreenNode notebook access (Jupyter URL/token or SSH) → provision
   GVHMR + Isaac Lab 2.1.0 envs there (fallback mjlab), benchmark training on
   data/dance1_subject2_seg.csv. Also need (timing-flexible): SMPL-X registration
   (user account) before video front-end; W&B key or patch decision at provisioning.
4. Phase 0-hardware checklist remains untouched (robot-side ground truth: LowState
   29-motor check, firmware version freeze, FTP hand service topics) — needs robot
   powered on; schedule with user.

## Resume protocol (after reboot / new session)

1. `cat ~/g1-dance/PROJECT_STATE.md` (this file).
2. `git -C ~/g1-dance log --oneline -15` for recent progress.
3. Check `logs/` for any in-flight long job state (training jobs survive on the cloud
   even when the laptop reboots — job IDs and provider noted in `logs/jobs.md`).
4. Continue from "Next actions" above.

## Open questions for the user (non-blocking, answer when available)

- ~~Cloud GPU budget/provider preference~~ → ANSWERED 2026-06-12: GreenNode AI Platform
  Notebook instance. Still needed before Phase 5: instance access (URL/SSH/credentials)
  and which GPU type the notebook has.
- ~~Where will the robot dance (flat ground? space size?)~~ → ANSWERED, CLOSED
  2026-07-02: user confirms **client shows also fit the 2 m radius** — keep the
  ≤1.5 m root-excursion vet gate as a constant. (Per-venue parameterization
  deprioritized to Phase 8 nice-to-have.)
