# G1 Dance Pipeline — PROJECT STATE

> **This file is the single source of truth for resuming work.**
> Any Claude session (or human) picking this project up: read this file top to bottom,
> then follow "Next actions". Update this file after every meaningful step —
> it must always reflect reality, because the laptop reboots regularly.

## Mission

Build a full software pipeline + web UI where the user inputs a reference dance video
and gets out an artifact that makes the **Unitree G1 EDU Ultimate (29 DoF, Inspire FTP
hands)** perform that exact choreography, pre-choreographed, while staying **balanced and
push-robust** (RL whole-body tracking controller, not open-loop playback).

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
- 2026-07-02: **PRODUCT BAR RAISED (user):** final app must be good enough to train
  **2–3 minute dances** and **deploy for client shows** (paid, audience-facing).
  Implications: (a) motion pipeline + training must handle 2–3 min sequences, not just
  the 28.8 s test segment — budget more GPU-hours per dance and validate long-horizon
  tracking; (b) Phase 8 hardening is now a hard requirement, not polish: pre-show
  checklist, rehearsal protocol, battery plan, operator e-stop procedure, fall recovery
  plan; (c) the ≤2 m-radius / hard-flat-ground vetting assumption was for his home area —
  client venues may differ → NEW OPEN QUESTION: typical show stage size + floor surface;
  vet gate limits should become per-venue parameters, not constants.

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

## Current status (2026-07-02 evening)

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
