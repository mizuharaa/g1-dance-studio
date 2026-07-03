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

## Current status (2026-07-02 night)

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
