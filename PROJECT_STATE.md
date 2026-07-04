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
- 2026-07-04: **Production audit (whole-system, real artifacts) → 33 confirmed**
  (docs/production_audit_findings.md): 1 CRITICAL, 7 HIGH, rest med/low. Theme =
  integration seams from parallel builds + one safety regression. Most FAIL CLOSED
  (block deploy, not unsafe-deploy). Must-fix before robot/paid use:
  - **CRITICAL: outcome capture dropped in UI rebuild** — no in-app way to record a
    show's outcome, so a fallen/incident dance is never demoted and stays show-ready →
    redeployed. Endpoint (record_outcome, demote-on-incident) EXISTS but is orphaned
    from the rebuilt UI. Fix: add Clean/Aborted/Incident step after deploy in the
    checklist wizard + force open-show resolution. (ui/ + shows.py)
  - **HIGH: verdict motion-sha seam** — mjlab_verify signs the .npz sha; every consumer
    (shows.record_sim_run_from_verdict, gen_config, authorize) hashes the deployable
    .csv → never matches → no dance can EVER reach show-ready via real artifacts. Fix:
    record BOTH npz+csv digests, bind consumers to the csv. (mjlab_verify + shows.py)
  - **HIGH: gen_config can't consume mjlab verdict** — globs exam_*.json (verdict is
    heldout_verdict.json) AND reads nominal.duration_s (mjlab verdict has no such key)
    → KeyError. Deploy path is a dead-path with real artifacts. (deploy/gen_config.py)
  **REMEDIATION PLAN (disciplined, avoid new integration debt):** these fixes touch
  ui/ + shows.py which the show-production build is editing NOW. DO NOT fix in parallel.
  Sequence: (1) let show-production land + merge; (2) ONE consolidation pass fixes all
  audit critical+high on top of it, with regression tests through the REAL artifacts
  (not synthetic matching shas — the audit noted existing tests miss these seams);
  (3) re-run production audit to confirm. Full list: docs/production_audit_findings.md.
- 2026-07-04: **Hands spike verdict (R&D):** hand pose NOT recoverable from our GVHMR
  front-end (body-only SMPL; GMR zeros the hands) — expressive hands must be AUTHORED
  until a whole-body+hands estimator (SMPLer-X/OSX/HaMeR) replaces the front-end. No
  Inspire RH56DFTP MJCF exists (only Dex3), so the collision gate (built, works) can't
  yet certify the REAL hands and the 'oversized' look is unconfirmed. v1 rec: use hands
  SPARINGLY (authored held gestures like the Thriller claw, open-loop synced), ONLY after
  collision-gated on a real Inspire model. Blockers: source Inspire MJCF + whole-body
  hands estimator. Worktree agent-a3cfcfe00dfde754c merge-ready (additive).
- 2026-07-04 (night): **ROBOT DAY = TOMORROW MORNING (user, at work). Gantry/hoist
  CONFIRMED available (feet fully off ground) → full gantry test possible.** Reframe:
  first contact = MEASUREMENT + SAFETY, not a performance. Goal order: powered health
  check → Step 3a kill/command-loss→damping measurement (THE key test; remote B-damping
  is the ONLY stop, no torque-cut e-stop) → feet-off policy run + sim-vs-real tracking.
  Do NOT go to ground tomorrow regardless of results. Gantry policy does NOT need ≥99%
  (gantry is where we validate a not-yet-proven policy). TONIGHT staging (agent
  a3c1cb3223b4b2d13): fix deploy dead-end bugs (gen_config glob+duration_s, verdict
  npz/csv sha), add GANTRY-SCOPE authorization (feet-off-only, sub-99% ok, ground/show
  still needs ≥99%), build+stage the Thriller gantry bundle, accurate gantry GO/NO-GO
  preflight, docs/GANTRY_MORNING_BRIEF.md. Orchestrator staging best Thriller policy +
  policy_meta.json. User needs tomorrow: gantry rigged, remote in hand, laptop on robot
  net (PC2 192.168.123.164). base_lin_vel≈0 on gantry so current policy is gantry-safe.
- 2026-07-04 (night): **Robot day is now a FULL DAY** (user full-time on it) → staged
  progression, NOT just measure: health → gantry (Step 3a damping + tracking) → GATE →
  ground-tethered (taut line, partial weight) → GATE → ground-free (slack line, 2m area,
  full dance — REAL fall risk) → GATE → push tests → debrief. Each stage GATED on the
  prior passing; more time must NOT skip gates. ground-free requires: (a) gantry 3a
  damping CONFIRMED, (b) ≥99% policy (attempt-2, else loud informed-override), (c) DLIO
  estimator verified sane (base_lin_vel matters on ground, ~0 on gantry). Staging agent
  a3c1cb3223b4b2d13 extended: full-day runbook docs/ROBOT_DAY_PLAN.md + ground-scope auth
  + telemetry capture. STRETCH MILESTONE: Thriller on the ground, repeatably — earned
  stage by stage, not assumed.
- 2026-07-04 (night): **Thriller policy STAGED for robot day + 2 safety catches.**
  data/policies/thriller/: policy.onnx, policy_meta.json (COMPLETE PD spec: per-joint
  kp 14.3–99.1, kd 0.91–6.31, effort 5–139Nm, default_joint_pos 29dof, action_scale
  incl 0.074 wrists/0.35 knees, ζ=2 overdamped), thriller_deploy.csv/.npz.
  **(a) ROBOT MUST USE THESE SIM GAINS, NOT stock Unitree gains** — stock would put the
  policy out-of-distribution → fall. Bundle carries them; start scripts must assert.
  **(b) ACTIVATION LURCH HAZARD fixed** — clip frame-0 differed from standby default
  pose by up to 39°(elbow)/38°(knee); controller holds default in standby with no
  auto-ramp → would lurch on R1+A. thriller_deploy has a 2.5s cosine ramp from default
  into the dance (frame-0 delta 0.000), re-verified 100% in-engine. **Deploy bundle uses
  thriller_deploy, NOT thriller_show.** Gantry candidate = attempt-1 (98.4% held-out,
  gantry-safe: base_lin_vel≈0 in-distribution). attempt-2 (→≥99% for ground-free) still
  training (~1150/4000). Details forwarded to deploy-kit agent a3c1cb3223b4b2d13.
- 2026-07-04 (night): **Show-production workflow MERGED** (191 tests): music sync
  (Dance.audio, 1.5s-aligned muxed preview), set-lists (pipeline/setlist.py, builder +
  sequential runner, show-ready only if all items are), rehearsal mode (Show.mode,
  rehearsal incident never demotes), show timeline. Presentation-only; never touches
  show-ready gating. Seeded dances still draft so set-lists show blockers until Thriller
  is show-ready (expected). **AUDIT CONSOLIDATION still QUEUED** (docs/production_audit_
  findings.md): deferred behind the deploy-kit robot-day agent (which is fixing the
  deploy-path findings gen_config/mjlab_verify NOW). After robot day + deploy-kit merge:
  ONE consolidation pass for the CRITICAL outcome-capture regression (verify whether the
  show-production runner already covers it) + remaining highs, tested through REAL
  artifacts. NOT needed for tomorrow's gantry test.
- 2026-07-04 (night): **FULL-AUDIT remediation — all 33 findings assigned across 3
  disjoint-lane agents** (user: continue the full audit + ensure tomorrow works):
  - deploy-kit (a3c1cb3223b4b2d13): deploy/ + gen_config + mjlab_verify — gen_config
    glob+duration_s, 02_push dead-code, verdict npz/csv sha (producer side), + robot-day
    package.
  - app-consolidation (a4e7a19c5c7df698c): ui/ + shows/store/cloud — CRITICAL
    outcome-capture regression, promote-UI gap, cloud atomic+ssh-hostkey, dedupe dangling,
    deploy_requests lock, FastAPI hardening.
  - pipeline-orphan (adb0500288d8fd6ca): vet/find_window/local_motion/library/venue/
    config/monitor/desktop — HIGH MEC-vs-excursion safety (robot leaving area), HIGH
    library-import security (fake show-ready/traversal/tar-bomb), venue pipeline wiring,
    monitor cost-after-delete, desktop stale-port.
  Frozen/deferred: exam_verdict 99%+3-run floor (user-authorized), sim_exam docstrings,
  a few cross-lane UI/venue wires (noted per-agent). **Plan: merge each as it lands
  (disjoint lanes = clean), keep suite green, THEN a final integration test + preflight
  to guarantee robot day works. Robot-day readiness personally verified before done.**
- 2026-07-04 (night): **ROBOT-DAY PACKAGE MERGED + PERSONALLY VERIFIED (main-thread).**
  196 tests green. preflight_robot_day --stage gantry = **GO** (8 GO / 1 WARN=PC2-offline-
  expected / 0 NO-GO): policy + policy_meta(sim gains) + thriller_deploy.csv(ramp) present,
  verdict signed+bound (fail=sub-99% but OK for gantry), scripts present, shellcheck clean,
  gantry bundle assembles, kill_now fires from any shell. --stage ground-free = correctly
  **NO-GO** (sub-99% → needs ≥99% or conscious --informed-override) — safety gate proven.
  Deploy-path bugs fixed (gen_config finds *verdict*.json + derives duration; mjlab_verify
  binds motion_sha to the deployable CSV + records npz provenance). Full-day staged
  10_gantry_test (gantry→ground-tethered→ground-free→push, per-stage typed phrases + hard
  gates), gains-assertion (refuses to leave damping without SIM_GAINS_LOADED), telemetry
  pull. Materials: docs/ROBOT_DAY_PLAN.md, ROBOT_DAY_CHECKLIST.md, preflight_robot_day.
  **TOMORROW IS GO for the gantry stage.** Ground-free unlocks if attempt-2 hits ≥99%
  overnight, else user makes a conscious informed-override call on the day.
- 2026-07-04 (night, user to bed): **FULL AUTONOMY for robot-day success tomorrow.**
  User confirmed the **gantry can lower to a taut line** → full staircase unlocked
  (gantry→ground-tethered→ground-free) IF a ≥99% policy exists. Overnight guarantees
  in motion: (1) app-consolidation MERGED (203 tests; CRITICAL outcome-capture confirmed
  closed, MEC-excursion safety fixed, promote-UI, library-import security); (2)
  pipeline-orphan DESCOPED to venue.py/monitor.py/desktop.py only (MEC+library were
  dup-assigned, now app-consolidation authoritative); (3) runbook-hardening agent
  acfc735422cdfa4d8 adding first-contact gotchas: PC2 controller install (never done!)
  + joint zero-offset calibration verify + estimator sanity + 'first 30 min'
  troubleshooting; (4) orchestrator pushing attempt-2 → ≥99% (ground unlock) + rebuild
  full bundle on convergence. **Robot day GANTRY = verified GO. Ground = unlocks if
  attempt-2 ≥99% overnight, else conscious informed-override on the day.** MORNING SWEEP
  owed by main: after agents land, re-merge/verify + final preflight + write MORNING
  STATUS at top of docs/ROBOT_DAY_PLAN.md.
- 2026-07-04 (02:30 ICT): **OVERNIGHT AUTOPILOT running (tools/overnight_a2.sh, bg).**
  Goal: unlock GROUND stage for tomorrow (gantry already GO on attempt-1's 98.4%;
  ground-free needs >=99%% held-out or informed override). Autopilot waits for
  train-thriller-a2 to converge, then auto: export ONNX -> heldout_eval.py (256 env,
  seed 90001, nominal+push) on the DEPLOYABLE motion (thriller_deploy) -> sign via
  pipeline/mjlab_verify.py -> writes data/policies/thriller_a2/RESULT.txt with survival
  %% + GROUND_READY yes/no. **RESUME ACTION (if session rotates):** read
  data/policies/thriller_a2/RESULT.txt; if GROUND_READY=YES → stage a2 as ground policy
  at data/policies/thriller/ (keep a1 as gantry fallback), gen thriller_deploy ramp for
  it, tell main to rebuild --full bundle; if NO → report honestly (attempt 2 of <=3).
  Attempt-1 stays the staged gantry policy regardless (complete policy_meta.json w/ PD
  gains + thriller_deploy 2.5s activation ramp; gantry-safe confirmed).
  Long-dance train-dance2-long ~3722/6000 (reward 33) converging — verdict via watchdog.
  Do NOT start unrelated training tonight (GPU focused on a2 + dance2-long).
- 2026-07-04 (night, autopilot): **THRILLER ATTEMPT-2 = 100% HELD-OUT SURVIVAL →
  SHOW-READY. GROUND STAGE UNLOCKS.** Overnight autopilot (tools/overnight_a2.sh)
  exported a2 (iter 1500 ckpt), ran the 256-env held-out gate on the deployable motion:
  nominal 256/256 + push 256/256 = 100%, signed verdict PASS, authorizes show-ready.
  Artifacts: data/policies/thriller_a2/{policy.onnx, heldout_verdict.json (signed,
  bound to a2 sha + thriller_deploy.csv)}. **QUALITY NOTE: a2 mpkpe 0.221m vs a1 0.168m**
  — a2 never falls (100%) but tracks a bit looser than a1 (the stronger action-rate
  penalty traded precision for robustness). For a FIRST ground dance, never-falls is the
  right priority; a1 stays available as a crisper-but-98.4% fallback.
  IN PROGRESS: orchestrator staging a2 as the ground policy (confirm policy_meta.json
  config-identical to a1: kp/kd/default_pose/action_scale — these drive the real PD loop,
  errors=fall); then main rebuilds the --full ground bundle + re-runs preflight (ground-
  free should flip to GO) + writes MORNING STATUS atop ROBOT_DAY_PLAN.md. 207 tests green
  (pipeline-orphan merged: venue z-grounding, monitor cost-freeze-on-delete, desktop
  stale-port).
- 2026-07-04 (02:30 ICT): **GROUND STAGE UNLOCKED — Thriller attempt-2 = 100%% held-out**
  (256/256 nominal AND push, signed PASS, on the deployable motion). action_rate_l2 -0.2
  delta took a1 98.4%% → a2 100%%. STAGED as primary at data/policies/thriller/ (a1 kept as
  fallback at thriller_a1_fallback/). Trade-off: a2 mpkpe 0.221m (looser but 100%% stable)
  vs a1 0.168m (crisper, 98.4%%). Autopilot relaunched (bug fixed: fired early at iter 1500
  on SSH blip) to verify a2's FINAL checkpoint and hot-swap if tighter. GROUND policy
  ready for robot day. Long-dance ~4446/6000 converging. **MAIN: tell deploy-kit to
  rebuild the FULL-scope bundle (gen_config --full) — ground-free preflight now unlockable.**
- 2026-07-05 (early AM): **OVERNIGHT COMPLETE — ROBOT DAY VERIFIED GO (full staircase).**
  All 5 build/audit agents merged (207 tests green): show-production, deploy-kit robot-day
  package, app-consolidation (CRITICAL outcome-capture closed, MEC safety, promote-UI,
  library security), pipeline-orphan (venue z-grounding, monitor cost-freeze, desktop
  stale-port), runbook-hardening (PC2 install + joint-calibration check + troubleshooting).
  **Thriller attempt-2 = 100% held-out → SHOW-READY, staged as ground policy; full-scope
  ground bundle BUILT + authorized (deploy/bundles/thriller/ complete).** Preflight:
  gantry=GO, ground-free=GO (only WARN = PC2 offline, expected). a1 preserved at
  thriller_a1_fallback. MORNING STATUS written atop docs/ROBOT_DAY_PLAN.md. Non-blocking
  background: a2_final autopilot (optional crisper upgrade, non-destructive marker at
  thriller_a2_final/RESULT.txt — NOT auto-applied), long-dance verdict pending. Nothing
  blocks robot day. RESUME (any session): read docs/ROBOT_DAY_PLAN.md top; the user runs
  preflight then the staged day.
- 2026-07-05 (early AM): **LONG-DANCE (67s) VALIDATED — 2-3min target de-risked on the
  TRAINING side.** train-dance2-long converged (reward 34.6), full-motion eval = 100%
  clean AND 100% under 64-env noise, joint error 0.099 rad (TIGHTER than 49s Thriller's
  0.117) → longer clips don't degrade with the single-clip + larger-adaptive-kernel
  recipe. Remaining long-dance constraint is CHOREOGRAPHY not capability: stock traveling
  mocap caps a clean in-2m-area window ~62s, so 2-3min show pieces must be choreographed
  to stay roughly in place (already flagged for filming). Registered as dance
  "Dance2-Long" (draft).
- 2026-07-05 (early AM): **BOX DECISION (Claude, pre-decided): DO NOT delete the GreenNode
  box until AFTER robot day.** Rationale: robot day is TODAY; if the policy needs a retrain
  from a sim2real finding on the hardware, a live box lets us iterate same-day; recreating
  it needs the user's console clicks + time. Cost is ~18k VND/h (trivial; budget 182k/1.5M).
  Delete AFTER robot day once we know no immediate retrain is needed. (a2_final autopilot
  still using the box; when it finishes the box idles but STAYS UP through robot day.)
- 2026-07-05 (AM): **a2_final RESOLVED — staged iter-1500 is the best, FINAL.** The
  attempt-2 final checkpoint also hit 100% but tracked LOOSER (0.249m vs iter-1500's
  0.221m — more training kept trading precision for smoothness), so no swap. Staged
  Thriller ground policy is locked. Long-dance policy preserved to data/policies/
  dance2_long/. All overnight objectives achieved; robot day GO. Box still up per keep-
  through-robot-day decision (all data on laptop; user can kill the meter anytime).
- 2026-07-05 (AM, user heading to office): **DEPLOY-RUNTIME BLOCKER surfaced by user —
  robot network has NO INTERNET.** Per ~/robot/RUNBOOK.md: two nets = Ethernet cable to
  robot (192.168.123.x, control) + a SEPARATE local router ('no internet cable needed')
  for laptop/Quest wifi (company wifi blocks device-to-device). Existing teleop uses
  unitree_sdk2_python + CycloneDDS directly — NO Docker. The dance deploy assumed
  motion_tracking_controller in Docker (qiayuanl/unitree:jazzy) on PC2 — that image is
  NOT on PC2/laptop and CANNOT be pulled at the office (no internet). Docker-save fallback
  also dead (no Docker/image on laptop). ⇒ **Today is realistically DIAGNOSTICS + gantry,
  NOT dance-on-hardware** unless PC2 already has Docker+image (user to check first:
  ssh unitree@192.168.123.164; docker images). Safe no-controller work today: health
  check, deploy/check_joint_calibration.py (SDK-based, no Docker), SDK comms/state read
  on gantry. **REAL FIX to build: a DOCKER-FREE deploy runtime** — run the ONNX policy via
  unitree_sdk2_python + onnxruntime directly (we have the obs layout, gains, SDK). This is
  the right architecture for the isolated robot net. Build next; gantry-test it (safe,
  feet-off) before any ground use. Robot-day plan/preflight assumed Docker path — needs a
  non-Docker deploy path added.
- 2026-07-05 (AM): **DEPLOY FIX — run the policy LAPTOP-SIDE over Ethernet, not
  PC2-Docker.** The existing teleop already drives the robot from the laptop via
  unitree_sdk2_python over 192.168.123.x (no Docker). A dance runtime works the same:
  ONNX policy on the laptop → LowCmd to robot over Ethernet. NO robot-internet, NO Docker,
  NO image pull needed — sidesteps the whole blocker. **Network insight:** Ethernet-to-
  robot + company-wifi-for-internet run SIMULTANEOUSLY (separate interfaces); the local
  router/Quest is teleop-only, NOT needed for the dance — so at the office the user can be
  ONLINE with Claude AND control the robot at once. If online at office: build the
  laptop-side runtime live (obs from sim_exam MjlabOnnxPolicy, gains from policy_meta.json,
  LowCmd via unitree_sdk2_python, 50Hz, damping-mode safety) + gantry-test it (safe,
  feet-off). TODO: this laptop-side runtime is unbuilt; robot-day preflight/scripts assumed
  the Docker path. Baseline today regardless = health check + check_joint_calibration.py
  (SDK, no Docker).
- 2026-07-05 (AM): **Two deploy paths for robot day (user suggested hotspot for robot
  internet).** Cleaner than hotspot-to-robot: SHARE the laptop's internet to PC2 over the
  EXISTING Ethernet cable (192.168.123.x) via IP-forwarding+NAT on the laptop — robot needs
  no wifi. Internet source = phone hotspot OR company wifi (either).
  - **PREFERRED (Path B): original reference controller** — internet→PC2, PC2 pulls
    qiayuanl/unitree:jazzy, run motion_tracking_controller (battle-tested BeyondMimic
    deploy; deploy-kit already built for this). Needs: Docker ON PC2 (verify: ssh
    unitree@192.168.123.164; docker --version) + multi-GB image pull (slow over hotspot).
  - **FALLBACK (Path A): laptop-side ONNX runtime** over Ethernet (like teleop), no
    internet/Docker needed, but unbuilt/new code.
  When user is online at office: (1) set up laptop→PC2 internet sharing (sysctl
  net.ipv4.ip_forward=1 + iptables MASQUERADE on the wifi iface + default route on PC2 via
  192.168.123.2) LIVE with Claude; (2) verify Docker on PC2 → Path B, else Path A. Baseline
  regardless: health check + check_joint_calibration.py (SDK, no Docker/internet). Robot's
  first real policy run stays GANTRY-FIRST either path.
- 2026-07-05 (AT ROBOT): **First real-robot contact — SDK comms CONFIRMED.** Laptop
  reads live LowState from the G1 over Ethernet (enp0s31f6, DDS, tv conda env has
  unitree_sdk2py). Fixed check_joint_calibration.py key names (meta uses
  default_joint_pos_rad + joint_order_29dof). Ran it live: 16/29 joints 'off' up to 63°,
  BUT pattern = limp/hanging gravity pose (legs straight, arms dangling), NOT an encoder
  miscalibration (G1 encoders are absolute/factory-cal). Awaiting user's physical
  observation (slack vs stiff) to confirm. **REAL DEPLOY REQUIREMENT surfaced: the robot
  must be brought smoothly to default_joint_pos (ready pose: bent knees/set arms) BEFORE
  the policy runs — a controller startup step; the activation ramp assumes the robot is
  ALREADY at default. Any deploy path (reference controller or laptop-side runtime) must
  do damping → move-to-default → run.** Robot on gantry, feet off, powered. Colleague's
  g1-siu/g1plus_pc4 setup on PC2 = unrelated, do not touch.
- 2026-07-05 (AT ROBOT — MILESTONE): **First successful COMMANDED motion — robot moved
  into the ready pose under our deploy runtime.** pipeline/deploy_runtime.py (laptop-side,
  tv env, unitree_hg over Ethernet) verified end-to-end on hardware: read LowState → build
  160-D obs → ONNX → LowCmd. Fixes en route: factory ctor unitree_hg_msg_dds__LowCmd_() +
  reuse; MotionSwitcherClient ReleaseMode (frees balance for full-body low-level, gantry-
  only); Mode.PR=0; matched h1_2 low-level example. **Gain finding: policy kp too soft to
  statically hold a pose vs gravity (limbs sag); move-to-default needs firmer gains — 2×
  (scale both kp+kd to stay overdamped) reaches the ready pose cleanly.** APPROACH_KP_SCALE
  env tunable. move-to-default damps at end (settles back) — for the dance, run must
  position-then-dance seamlessly. **NEXT: run the Thriller policy on the GANTRY (feet off
  ground) — EXPECT legs to flail (policy trained with ground contact; none on gantry),
  focus on arms/torso tracking + no fault/violence.** User is the physical e-stop (remote
  damping); considering a webcam for Claude to observe (Claude drives+sees, user stays
  hand-on-damping — Claude will NOT run autonomous motion without the human e-stop).
- 2026-07-05 (AT ROBOT — 🎉 FIRST DANCE ON HARDWARE): **The G1 performed the opening of
  Thriller on the gantry.** `run --max-secs 3`: released balance service, Stage-1 to ready
  pose, then the POLICY executed — webcam confirms the arms moved through distinct Thriller
  poses (frame 13 arms down/back → later arms raised out). Legs bent/passive (no ground —
  expected). Policy runs on real hardware end-to-end. Webcam working (video0; pipewire
  grabs it periodically — reclaim via `systemctl --user restart pipewire.service
  pipewire.socket`). Cost/budget fine.
  **BUG (fixing, agent adeb4398f5662f914): the run process HUNG on exit** (CycloneDDS
  shutdown) → outer timeout SIGTERM-killed it → robot left HOLDING an energized pose
  instead of soft-settling. Fix: clean prompt exit (os._exit after damping) + SIGTERM/INT
  handler that DAMPS before exit (robot must ALWAYS end soft on any exit) + confirm
  max-secs strictly caps. User damped via remote for now. **Paused autonomy on this
  anomaly — resume clean gantry runs after the exit fix; GROUND still hard-blocked (needs
  torso state estimator; not built).**
- 2026-07-05 (AT ROBOT — GANTRY STAGE COMPLETE): **Full Thriller performed on the gantry
  TWICE, cleanly (all 2589 ticks), ending soft each time.** Upper-body choreography tracks
  beautifully (webcam: arm reaches overhead, gestures, torso turns, distinct controlled
  poses). Legs are the known gantry limit (policy trained w/ ground contact; free-swinging
  legs occasionally spike large 'balance' actions — one grazed the 8.0 action cap and the
  safety correctly damped; raised MAX_ACTION→12 env-tunable for gantry, full dance then
  completed). Exit-fix + damp-on-any-exit + clean-exit all verified on hardware. Deploy
  runtime (pipeline/deploy_runtime.py) is the working laptop-side path (tv env, unitree_hg
  over Ethernet, no Docker). Robot always ends soft.
  **GANTRY has shown all it usefully can — the dance works on real hardware.** NEXT FRONTIER
  = GROUND, HARD-BLOCKED until: (1) a torso-position state estimator (DLIO or obs-restricted
  policy) feeds the 12 gantry-approximated obs dims (base_lin_vel + anchor terms) — without
  it the ground feeds the policy wrong data → fall; (2) staged tethered→free ground per the
  runbook. That estimator is a separate build (not done). Webcam works (reclaim pipewire
  as needed). Budget fine.
- 2026-07-05 (user away ~1h): **Prepping for a STAGED TETHERED-GROUND try on return.**
  RULE HELD: no robot motion (or even read-only) while user away — no human = no motion.
  Running SIM/CODE ONLY: (1) obs-restricted Thriller retrain (agent af55f59b0755ac816,
  drops estimator-dependent position obs → data/policies/thriller_ground/, GPU box alive);
  (2) staged ground mode in deploy_runtime + docs/GROUND_TETHERED_RUNBOOK.md (agent
  aaa5ed6183f9b2c35): stand-hold → ground-run --max-secs (short, gated) → extend, all
  always-soft-on-exit. RETURN SEQUENCE (human-supervised): read-only brain check →
  stand-hold (watch stability, tether bearing weight) → ground-run 3s → extend. Retrain is
  GPU-hours; may not finish in 1h — if not, everything else staged, tethered try waits on
  the verified policy (NEVER rush an unverified policy onto the robot). Ground still =
  human present + tether rigged + damping-in-hand + Claude watching webcam; NOT autonomous.
  Videos of today's gantry Thriller: data/previews/thriller_gantry_{realtime,2x}.mp4.
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

## Ground runtime prepared (2026-07-04, sim/code only — no robot touched)

Staged tethered-ground path added to `pipeline/deploy_runtime.py`, ready for a
human-supervised session (NOT autonomous — no ground motion has run):
- `stand-hold` mode: firm PD move-to-default then hold the ready stance standing,
  indefinitely until Ctrl-C / B-damp. No policy — proves the robot can stand tethered.
- `ground-run --max-secs N` mode: runs the estimator-free ground policy for a capped
  segment. `--max-secs` mandatory; conservative `GROUND_MAX_ACTION` (default 6.0).
- Both gated by `--i-will-watch-the-robot` + `CONFIRMED_BY_HUMAN=alois`, both ALWAYS
  end soft (SIGINT/SIGTERM/crash → damp + os._exit).
- `build_obs_ground` builds the 154-dim estimator-free obs (drops base_lin_vel +
  motion_anchor_pos_b). `_ground_obs_order` reads the ground meta's declared obs order
  and **hard-refuses** any policy that still needs an estimator term, or if the
  `data/policies/thriller_ground/` artifacts are absent (retrain not yet landed).
- 13 new offline tests (154-dim, estimator-free, refusal paths); full suite 220 passed.
- Procedure: `docs/GROUND_TETHERED_RUNBOOK.md` (Stage 0 read → A stand-hold → B capped
  ground segments 3→5→10→20s→full, abort criteria, safety-layer explanation).
- BLOCKED on: obs-restricted retrain (agent af55f59b0755ac816) producing
  `data/policies/thriller_ground/` + `docs/ground_policy.md`. Until then ground-run
  refuses by design; stand-hold works now (pure PD).

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

## 2026-07-04 ~13:55 ICT — GROUND RETRAIN FAILED (estimator-free). Ground DANCE blocked.
- train-thriller-ground ran full 3000 iters but NEVER learned: mean episode length ~5
  steps, ee_body_pos termination = 1.0000 for the entire run. Held-out eval 0/256 nominal.
- Cause: dropping motion_anchor_pos_b + base_lin_vel from the actor obs made the task
  unlearnable AS CONFIGURED (no re-anchoring / no startup termination grace). Same motion
  + termination trains to 100% WITH full obs (gantry a2). See data/policies/thriller_ground/RESULT.txt.
- ACTION: no estimator-free ground policy exists; ground-run correctly refuses (no policy_meta).
  Fix path (sim-only): enable reference re-anchoring and/or startup ee-termination grace, retrain,
  re-gate at nominal>=0.95 full-length.
- TODAY: ground DANCE not possible. stand-hold (no policy) IS valid -> Stage 0 read + Stage A
  tethered stand-hold remain the safe, ready steps. Robot untouched today.

## 2026-07-04 ~14:20 ICT — First GROUND session (tethered). Stand-hold works; tether bears weight.
- Ran stand-hold (no policy) on the GROUND, fully tethered, human on damping. 3 clean cycles.
- SAFETY SPINE PROVEN ON GROUND: motion-service release, firm move-to-default ramp, indefinite
  hold, and damp-on-signal all worked; robot ended SOFT on every stop (SIGTERM->damp->os._exit).
  Lesson: signal the PYTHON pid directly (exec/-u), not the bash wrapper, or the child orphans.
- FINDING (important): holding pose is GAIN-INDEPENDENT. 2.0x and 3.0x approach gains gave a
  near-identical stance (hip_pitch sag ~22deg, ankle_pitch ~-40 vs cmd -21, knees on target).
  A PD deflection would scale ~1/kp; it didn't move -> the legs rest against an EXTERNAL
  constraint, i.e. the TETHER/HARNESS is bearing weight and holding a suspended crouch. The
  feet are not truly load-bearing => this was NOT a real weight-bearing stand test. Torso stayed
  vertical (waist_pitch ~0.4), pose rock-steady, no oscillation/faults.
- IMPLICATION: to validate a real stand, slacken the tether so the feet load — deliberately,
  higher risk (first real weight-bearing, no e-stop, and NO working ground dance policy for
  dynamic recovery). Gains are not the lever.
- NEXT (sim, no robot): fix the ground retrain (reference re-anchoring + startup ee-termination
  grace), retrain, re-gate nominal>=0.95 full-length. THEN a real weight-bearing stand test.

## 2026-07-04 ~14:35 ICT — Weight-bearing stand-hold: legs sag under load (EXPECTED, not a fault).
- Slackened tether so feet bear weight; ran stand-hold 8s ramp @2.0x. Feet loading engaged
  (pose became load-dependent, unlike the tether-borne case). Under load the legs SETTLED
  (2 identical reads) into a stable deep squat: knee 55/66, hip_pitch +12/+6, ankle -50
  (cmd knee 38, hip -18, ankle -21). Torso vertical (waist_pitch 0.4). Asymmetric (R knee 66 vs L 55).
- INTERPRETATION (important, avoid misfiling): a static PD-to-default is a spring to a pose,
  NOT a balance controller. Sag under body weight is EXPECTED once the onboard balancer is
  released. The robot stands fine on its OWN onboard controller (normal standby). Weight-bearing
  in the dance comes from the ACTIVELY-BALANCING policy, not from stand-hold. Stand-hold was the
  pre-check and did its job: safety spine proven, feet-loading confirmed, static-PD gap measured.
- CONCLUSION: today's real blocker is unchanged — the ground POLICY retrain failed. No robot dance
  today. Next = fix retrain in sim (re-anchoring + startup ee-termination grace), retrain, re-gate.
- Robot damped/soft. 4 clean stop cycles total this session; always ended soft.

## 2026-07-04 ~14:45 ICT — Ground retrain FIX WORKS (v2 bootstrapping). Sim only.
- Root cause was an exploration CLIFF, not re-anchoring (which mjlab already does every step)
  nor bad tracking (torso anchor err was 0.076m in the failed run). With the 2 obs terms dropped,
  the untrained policy died at step ~5 on ee_body_pos (wrist/ankle HEIGHT err > 0.25m) and never
  got a learning signal.
- FIX: loosen ONLY that threshold 0.25->0.6 so it survives the transient and bootstraps. Launched
  job train-thriller-ground-v2 (No-State-Estimation, motion thriller_deploy.npz, action_rate_l2
  -0.2, 3000 iters, ee-thresh 0.6). CLI override only, no mjlab source edits.
- RESULT so far: mean episode length 5 -> 24-26 and climbing by iter 180; ee_body_pos term rate
  1.0 -> 0.0-0.5. It is LEARNING. Training out ~30 min.
- NEXT: on completion, export 154-dim ONNX (export_policy_ground: swap task to No-State-Estimation)
  + held-out gate on the SAME task. Eval at STRICT 0.25 first (honest bar) AND at 0.6 (training-
  matched); report ACHIEVED mpkpe + ankle-height err separately. If ankle tracking is loose,
  do v3 with SPLIT thresholds (tight ankle for safety, loose wrist). SIM_READY only if nominal>=0.95.

## 2026-07-04 ~15:05 ICT — v2 mid-training: wall MOVED to balance/anchor. Verdict pending eval.
- v2 past the ee cliff, but by iter ~2200 the binding wall is the ANCHOR terminations
  (anchor_ori torso tilt >0.8rad, error_anchor_rot ~1.0; anchor_pos torso height >0.25m).
  time_out pinned at 0; ep length regressed 24->8. Read: it can't hold the torso upright/at
  height without base_lin_vel obs — the fundamental estimator-free balance cost. BUT mid-train
  metrics are adaptive-sampler-pessimistic; the frame-0 held-out eval is the real test.
- Built this session (sim/code only): heldout_eval_ground.py (ankle/wrist height breakdown,
  configurable ee-threshold), export_policy_ground.py, autopilot_v2.sh (on box, auto export+gate
  @0.25 & @0.6 -> RESULT), laptop watcher (pulls to data/policies/thriller_ground_v2/ + notifies).
- Updated docs/GROUND_TETHERED_RUNBOOK.md (Stage-A sag is EXPECTED; revised go-criterion) and
  wrote docs/ground_retrain_next.md (v3 decision tree branched on v2 eval: ship / split-ee /
  balance-curriculum / reconsider-estimator).
- NOT declaring v2 a win; earlier optimism was premature. Waiting on RESULT.txt.

## 2026-07-04 ~15:30 ICT — PIVOTAL: robot publishes a base-state ESTIMATE (rt/odommodestate).
- v2 verdict: SIM_READY=NO. 0/256 survival at both 0.25 and 0.6 ee bounds; mpkpe 0.115m,
  ankle 0.128m (tracks tight but FALLS — the balance wall, as diagnosed). Estimator-free
  Thriller does not balance without velocity obs. Artifacts in data/policies/thriller_ground_v2/.
- READ-ONLY robot probe (no motion): LowState has NO base vel/pos (only imu quat/gyro/accel +
  joint q/dq). BUT topic **rt/odommodestate** (SportModeState_) IS LIVE at ~184Hz with:
  position[3], velocity[3], body height. At rest: vel clean ~0, pos steady (~1mm/4s drift),
  h~0.566m. This is a real onboard EKF — exactly the two obs terms the estimator-free path
  dropped (base_lin_vel + torso position for motion_anchor_pos_b).
- IMPLICATION / PIVOT: instead of fighting estimator-free training, feed the PROVEN full-obs
  GANTRY policy (100% in sim) REAL base_lin_vel + anchor position on the GROUND from
  rt/odommodestate. Sidesteps the balance wall entirely.
- OPEN VALIDATIONS (not all doable read-only): (1) does odom stay published + accurate when we
  RELEASE the motion service for low-level control? (it publishes now, and we released it during
  stand-hold, so likely persistent — confirm during a supervised run). (2) velocity is reliable;
  XY position DRIFTS over 49s -> handle by RE-ANCHORING at deploy (reset position frame at start,
  matching training); height/Z directly usable. (3) frame mapping odom(world)->body via imu quat.
  (4) whose estimator is it (onboard vs colleague's stack) + persistence.
- NEXT (recommended): wire deploy_runtime to consume rt/odommodestate -> build HONEST 160-dim obs
  -> sim-check -> tethered-first ground bring-up of the PROVEN gantry policy.

## 2026-07-04 ~16:10 ICT — Odometry-fed ground obs path BUILT + offline-validated (sim/code only).
- deploy_runtime.py: added odom_subscriber/read_odom (rt/odommodestate), build_obs_odom (HONEST
  base_lin_vel = R.T@v_world + re-anchored motion_anchor_pos_b = R.T@(ref_disp-robot_disp)), and
  mode `ground-run-odom` running the PROVEN gantry policy with the estimate (GROUND cap, --max-secs,
  full safety spine, NO-GO if odom absent, always-soft).
- Tests: tests/test_deploy_odom.py (6) — perfect-tracking->anchor=0, base_lin_vel body-frame, reduces
  to gantry fake when static, dim/finite. Full deploy suite 20/20 green.
- Offline pipeline smoke (tools/sim_ground_odom.py): real ONNX policy over full 51.8s motion with
  odom-from-reference (+noise): 0 non-finite, base_lin_vel mean 0.25/max 1.26 m/s (vs fake 0), robust
  to noise. FINDING: gantry policy's real action range ~8.5 -> GROUND_MAX_ACTION=6 false-trips ~4%;
  ground-run-odom needs GROUND_MAX_ACTION=10 (documented in runbook Stage B-ODOM).
- Runbook: added Stage B-ODOM (preferred path) w/ the 2 supervised validations (odom survives
  motion-service release; velocity-frame sway test) + re-anchoring notes.
- OPEN (needs tethered bring-up, no robot while away): confirm odom persists under our control +
  velocity-field frame. Estimator-free v3 deprioritized (odom path is primary, no GPU needed).

## 2026-07-04 ~16:35 ICT — Odometry obs path VALIDATED against the simulator (frame math exact).
- Sim cross-check (gantry policy rollout, 25 ticks): build_obs_odom's two terms vs mjlab's
  authoritative obs. motion_anchor_pos_b MAX ERR = 0.000000. base_lin_vel MAX ERR = 0.000000
  when rotated by the ROOT(pelvis) quat (initial 0.068 was torso-vs-root body choice, not a bug).
  => R.T@(ref-rob) and R.T@v_world are EXACTLY mjlab's frame conventions.
- Robustness margins (training noise on the two terms): base_lin_vel ±0.5 m/s, motion_anchor_pos_b
  ±0.25 m. HUGE. => (a) diff-velocity noise (~0.05) is 10x under tolerance — ODOM_VEL_SOURCE=diff is
  adequate, field optional; (b) the pelvis-vs-torso approximation (odom=pelvis, mjlab anchor=torso;
  one IMU quat for both terms) is well within ±0.25 — same approximation the WORKING gantry deploy used.
- base_lin_vel obs = imu_lin_vel sensor = root_link_lin_vel_b (pelvis, body frame). On robot: odom
  pelvis velocity + IMU pelvis quat -> exact. Confirms odom (pelvis) is the right source.
- NET: the odometry-fed path is frame-correct and noise-robust by construction. Remaining unknowns are
  purely hardware (does odom persist under our low-level control; does odom vel field frame = body) and
  resolve on the first tethered bring-up. Offline validation is as complete as it can be.

## 2026-07-04 ~17:15 ICT — Onboard odom FREEZES on release; built KINEMATIC (leg) odometry instead.
- First ground-run-odom on the tether: frozen-estimate GUARD FIRED — rt/odommodestate stamp froze
  the moment the motion service was released for low-level control; robot damped safe. So the onboard
  estimate and full-body control are mutually exclusive. Safety design validated on hardware.
- FIX: pipeline/leg_odometry.py (LegOdometry) — base_lin_vel + torso height from LEG kinematics
  (MuJoCo FK/Jacobian on the menagerie g1.xml, planted-foot assumption, contact-weighted blend,
  ±2.5 m/s clip). Service-INDEPENDENT (only LowState q/dq + IMU).
- VALIDATED offline vs reference ground truth (tools/validate_leg_odom.py): base_lin_vel within the
  policy's ±0.5 m/s trained band on 97.8% of frames (mean err 0.13). End-to-end policy smoke
  (tools/sim_ground_legodom.py): finite, bounded actions over the full 51.8s. Tests: tests/
  test_leg_odometry.py (4) + full suite green (23).
- deploy_runtime: new mode `ground-run-legodom` (PROVEN gantry policy + leg-odom obs: real base_lin_vel,
  anchor XY=tracking-assumption/drift-free, anchor Z=real leg-odom height feedback). Full safety spine,
  GROUND_MAX_ACTION=10, --max-secs. Runbook: Stage B-LEGODOM is now PREFERRED; B-ODOM superseded.
- READY: tethered try of ground-run-legodom (human-supervised). Stage 0 read still valid to pre-check.

## 2026-07-04 ~18:00 ICT — Motion-service left RELEASED stranded the robot; added auto-restore.
- ROOT CAUSE of "nothing happens": our runs call MotionSwitcherClient.ReleaseMode and never
  restored it. CheckMode confirmed name='' (released). SelectMode('ai') restored the DDS service
  (CheckMode->'ai') BUT the remote/app still would not pair -> the Bluetooth/pairing failure is
  ONBOARD-PC level, separate from the DDS service. Recovery = REBOOT the robot (coordinate w/ the
  colleague's Docker). Robot safe/limp throughout; no fall.
- FIX (committed): deploy_runtime _restore_motion_service() re-SelectMode(RESTORE_MOTION_MODE, default
  'ai') inside _finalize_and_exit, AFTER the damp — so EVERY exit path (normal/Ctrl-C/SIGTERM/crash)
  hands control back to onboard and the remote can pair. Best-effort, never blocks the (already-soft)
  exit. env RESTORE_MOTION_MODE='' to disable.
- STATUS: leg-odom ground path proved a BALANCED segment before the strand (3s ok per user). Once the
  robot is rebooted + remote pairs, resume the staged runs (5s->10s->20s->full); the runtime now cleans
  up the motion service each run so this can't recur.

## 2026-07-04 ~18:40 ICT — Recovered after reboot; control CONFIRMED; auto-restore validated on HW.
- Reboot fixed the onboard/Bluetooth strand (remote+app+damping all working). Robot back under 'ai'.
- BUG found+fixed: _release_motion_service crashed on a transient None from CheckMode right after Init
  (crashed BEFORE taking control -> "move-to-default did nothing" #2). Now retries on None (15x), robust.
- Instrumented move-to-default (post-fix): "service released -> lowcmd accepted -> at default pose".
  Mid-ramp read proved commands LAND (legs drove 70->50deg toward default, arms tracked). Remaining leg
  gap = known load-bearing sag. Control fully restored.
- AUTO-RESTORE VALIDATED ON HARDWARE: after the run, CheckMode='ai' (the on-exit SelectMode fired) —
  robot handed back to onboard control, remote stays reachable. The strand can't recur.
- READY: resume ground-run-legodom staged runs (5s->10s->20s->full). Each run now self-restores the
  service on exit.

## 2026-07-04 ~19:15 ICT — BREAKTHROUGH: robot STANDS + BALANCES + dances on the ground (tethered).
- ground-run-legodom with BOOSTED LEG GAINS worked. User confirmed: "it did stand and balance."
- WINNING CONFIG: APPROACH_KP_SCALE=3.0 (stand-up in move-to-default) + GROUND_LEG_KP_SCALE=2.0
  (legs hold standing during policy; arms at trained gains) + GROUND_MAX_ACTION=10.
- Telemetry (policy window): avg knee 36deg (target 38 = STANDING) vs the prior 50-70deg CROUCH;
  hips ~-11/-13 (target -18), ankles ~-29 (target -21). Legs held a standing config while the arms
  danced (shoulder/elbow motion 40deg std). No oscillation/fault; clean completion + auto-restore.
- ROOT CAUSE (resolved): the trained gains stand the robot in SIM but are too soft on real HW under
  load -> it sagged into a crouch and danced from there. Boosting ONLY the leg kp/kd (kept overdamped,
  torque clamped) let the legs bear weight and stand. Diagnosis via joint telemetry + user's eyes.
- NEXT: confirm it's GENUINELY weight-bearing (slacken tether at 3s) then extend 5s->10s->20s->full.

## 2026-07-04 ~19:45 ICT — LATERAL FALLING was a HARDWARE FAULT (stuck right hand). Fixed -> balances.
- Root cause of the sideways falls: the RIGHT HAND was mis-installed, locked at a wrong angle ->
  CoM pulled off-center -> persistent right lean -> lateral fall. NOT a policy robustness gap
  (my retrain call was premature). User re-installed the hand correctly.
- Post-fix telemetry (3s, config APPROACH_KP_SCALE=3.0 + GROUND_LEG_KP_SCALE=2.0 sagittal-only):
  L_knee 35 / R_knee 36 (asymmetry -1deg, was +8-9), torso mean roll +1.1deg (level), swing 6.6deg.
  User: "it looks balanced now." Symmetry+level are tether-independent -> genuine CoM fix.
- SHOW WORKFLOW (user, important): during the show the robot is WALKED up to the stage via the
  remote and is in RUN/WALK mode (onboard, standing+balanced) BEFORE the dance is initiated. So our
  deploy hands off from an ALREADY-STANDING pose (not from limp/crouch) -> easier + more robust.
- NEXT: extend duration (5s->full) now balance works; then test the real walk-mode->dance handoff.

## 2026-07-04 ~20:15 ICT — 30s: stable on average but ONE sudden lateral "acrobatic" move (tether caught); motors overheating -> robot cooldown.
- Progression today (post hand-fix, config APPROACH_KP_SCALE=3.0 + GROUND_LEG_KP_SCALE=2.0 sagittal-only,
  GROUND_MAX_ACTION=10): 3s/5s/10s all CLEAN + balanced (roll swing <5deg, no drift). Walk-mode handoff
  worked end-to-end. 30s: by-segment roll flat (-1.7..+1.8, NO drift) BUT one transient to roll -23.7deg
  = a SUDDEN anomalous sideways "acrobatic" move; TETHER caught it (not self-recovered).
- DIAGNOSIS (not drift/margin): a single bad policy ACTION or bad OBS at one instant -> likely findable
  in data, not fundamental. Suspects: (a) action spike under the cap-10 but large lateral; (b) leg-odom
  velocity spike (swing-phase, clipped to 2.5 but still) feeding a bad obs term; (c) a specific hard
  Thriller move. NEXT (offline, no robot): replay policy over full motion, flag the tick(s) with large
  lateral/roll-driving actions or obs outliers; cross-ref the timestamp of the -23.7 event.
- MOTORS OVERHEATING: the boosted leg gains hold high continuous torque to bear weight -> hip/knee/ankle
  heat over 30s. Show implication: trim gains (walk-mode standing start needs less stand-up authority) or
  plan cooldowns. User shut robot down to cool.
- STATUS: ground dance WORKS + balances to 30s tethered; blockers before full = (1) the one anomalous move,
  (2) motor thermal at these gains. Both addressable.

## 2026-07-04 ~20:45 ICT — Root-caused the "acrobatic" move (leg-odom velocity SPIKE) + fixed it (offline).
- Offline replay of the policy over the full motion: the biggest action JUMP (3.32 @ t=17.8s, inside the
  30s window) COINCIDED with a leg-odom base_lin_vel SPIKE (2.98 m/s vs true max 1.21). Mechanism: swing-
  phase foot not cleanly planted -> kinematic velocity spike -> bad base_lin_vel obs -> policy overreacts
  = sudden lateral move. Confirmed: 58 frames (2.2%) had >0.5 m/s vel error; est max 2.98 vs true 1.21.
- FIX (pipeline/leg_odometry.py): temporal smoothing — per-tick rate limit (VEL_MAX_STEP 0.30 m/s) + EMA
  (alpha 0.35), with reset_filter() called at run start (wired into ground-run-legodom). Results: vel err
  >1.0 m/s 12->0, >2.0 4->0; est max 2.98->1.14 (true 1.21); within +-0.5 band 97.8%->99.0%; policy action
  jump max 3.32->2.48 and the 17.8s spike eliminated. (Remaining ~3 jumps>2.0 are genuine fast Thriller
  moves, not obs artifacts.) Tests: +2 smoother tests, suite green.
- STATUS: acrobatic-move cause fixed in code (unverified on robot — robot cooling). When motors cool:
  re-run 30s to confirm the anomaly is gone, then push to full. STILL OPEN: motor thermal at boosted gains.

## 2026-07-04 ~21:30 ICT — Root cause CONFIRMED: leg-odom degrades during STEPPING -> whole-body brace.
- User saw at ~14-16s: "strange acrobatic position (legs AND arms) arched and stiff, then attempt to
  dance, unstable, better toward end." => NOT a topple (balance); a whole-body BRACE = policy reacting
  to a BAD OBS, then recovering. Points to estimator, not balance.
- OFFLINE PROOF: leg-odom error by section — 0-10s (danced CLEAN): vel_err 0.04/height 0.08m.
  13-17s (the arched brace): vel_err 0.15/height 0.18m = WORST in the dance, in BOTH channels. Aligns
  with feet lifting (stepping, foot Z 0.12-0.21m). Planted-foot assumption breaks during a step ->
  velocity AND height estimate degrade -> policy braces -> recovers when feet re-plant.
- Smoothing fix (prev) only killed single-frame SPIKES; this is SUSTAINED (~0.4s) degradation -> smoothing
  can't fix it. Re-run 30s still spiked to roll -28.9deg @ ~14-16s (consistent timing = choreography step).
- Note: offline uses REFERENCE (perfect-tracking) joints; real hardware joint tracking is noisier, so the
  real stepping error is likely WORSE than the 0.15/0.18 measured.
- FIX PLAN (offline, no robot): FUSED estimator — integrate IMU accel to carry base vel/height THROUGH the
  step (when kinematics is blind), correct with kinematics when a foot is solidly planted (complementary
  filter / the piece the onboard EKF did before it froze on service release). Keeps the proven policy.
  Alt: retrain with heavier estimator-noise/disturbance randomization (slower, GPU). Motors OK.

## 2026-07-04 ~22:15 ICT — AUDIT + DISAMBIGUATION: estimator was a RED HERRING; PIVOT to arm-over-balance.
- Independent first-principles audit + a decisive offline disambiguation test resolved the 14-16s brace.
- DISAMBIGUATION (perfect vs leg-odom vs fused obs -> same policy, measure |action-perfect|):
  * estimator corrupts the action EQUALLY in clean (0.35) and stepping (0.36) windows -> estimator is
    NOT concentrated at the failure. In clean 0-10s |action-perfect| max was 2.76 yet robot danced fine;
    at the failing 14-16s it's only 0.79. So estimator error does NOT cause the brace.
  * with a PERFECT estimate the policy STILL commands violent moves at 14-16s (action jump 0.81 clean ->
    2.05 stepping). => the stepping section is CHOREOGRAPHY-hard/dynamic, not estimator-hard.
  CONCLUSION: NO estimator (leg-odom/fused/perfect) fixes 14-16s. The fused-estimator polish was
  optimizing the wrong variable (audit called this). Fusion stays in tree (offline-validated) but is NOT
  the fix.
- STRUCTURAL BLOCKERS of the full-body-low-level path (audit): (1) THERMAL — boosted leg gains needed to
  stand overheat at 30s; show is 2-3min -> likely show-killer. (2) hard stepping moves it can't execute on
  ground. (3) faked XY anchor (robot_disp XY==0) = uncorrected lateral drift over a long dance.
- HIGHEST-LEVERAGE PIVOT (audit #1, now empirically supported): let ONBOARD control BALANCE+LEGS (it
  walks the robot up already) and drive the ARM dance via low-level. Thriller is ~arms. This is ALREADY
  PROVEN on THIS robot: ~/robot start_teleop_armsonly.sh does arm control (--arm=G1_29, g1_arm7 arm_sdk
  weight-blend) WHILE onboard balances. Eliminates estimator+thermal+stepping+XY at once.
- PROCESS FIX: prove the root cause with a cheap decisive test (disambiguation) BEFORE building the fix.
  I built fusion before proving the diagnosis; the audit+disambiguation corrected course.
- DECISION PENDING (user): pivot to arm-dance-over-onboard-balance (reliable, loses leg choreo) vs keep
  full-body (fragile, thermal-capped). Recommend PIVOT for a show-grade product.
