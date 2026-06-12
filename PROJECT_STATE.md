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

## Phase checklist

- [x] Phase 0 — Workspace, persistence, hardware audit
- [x] Phase 1 — Architecture pinned (research synthesis → docs/architecture.md, 2026-06-12)
- [ ] Phase 2 — Local foundations: conda envs, repos cloned, MuJoCo G1 model loads
- [ ] Phase 3 — Motion path on known data: pre-retargeted dance motion (e.g. LAFAN1)
      visualized in MuJoCo viewer
- [ ] Phase 4 — Video front-end: video → SMPL → retargeted G1 motion (our own video)
- [ ] Phase 5 — Training: cloud GPU job for tracking policy on one motion; sim verify
- [ ] Phase 6 — Deploy: policy runs on real G1 (hung from gantry first), push test
- [ ] Phase 7 — UI: web app orchestrating stages end-to-end with progress + preview
- [ ] Phase 8 — Hardening: error handling, docs, repeatability, second/third dance

## Current status

Phase 1 complete (architecture pinned). Phase 2 in progress: conda env `g1dance`
(py3.10, conda-forge) being created; shallow clones of GMR / whole_body_tracking /
unitree_mujoco / mujoco_menagerie into `third_party/`.

## Next actions

1. Finish Phase 2: verify mujoco imports, G1 29-DoF MJCF loads (check unitree_mujoco
   and menagerie assets — confirm which has the 29-DoF model with right joint order).
2. Phase 3: download `dance1_subject2.csv` from LAFAN1_Retargeting_Dataset on HF
   (try `lvhaidong/...` mirror if `unitreerobotics/...` 401s anonymously) →
   `pipeline/` playback script (CSV → qpos, render MP4 to `data/previews/`) →
   visually verify the dance.
3. Write the motion vetting module (root excursion ≤1.5 m, joint limits, foot-skate)
   against the same CSV — reusable for every future motion.
4. When user provides GreenNode notebook access: provision envs there (GVHMR + Isaac
   Lab 2.1.0 pip install; fallback mjlab) and benchmark training on dance1_subject2.

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
- Where will the robot dance (flat ground? space size?) — affects motion vetting.
