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

## Architecture (filled in after research — see docs/architecture.md)

Pipeline stages:
1. **Ingest**: dance video upload via web UI
2. **Motion extraction**: video → world-grounded SMPL motion (GPU)
3. **Retarget**: SMPL → G1 29-DoF reference trajectory (CPU-ok)
4. **Train**: RL whole-body tracking policy in Isaac Lab, push-randomized (cloud GPU)
5. **Verify**: sim2sim playback in MuJoCo, automated metrics + viewer (CPU-ok)
6. **Export/Deploy**: package policy → run on robot via unitree_sdk2 low-level control

## Decision log

- 2026-06-11: Project started. Workspace `~/g1-dance/`, git-tracked.
- 2026-06-11: Research workflow launched to pin component choices (results → docs/architecture.md).
- 2026-06-12: User confirmed cloud compute = GreenNode AI Platform Notebook instance.
  Implication: training jobs run inside a Jupyter notebook environment (persistent while the
  instance runs) rather than a batch-job API — plan for tmux/nohup inside the instance and
  artifact sync via the notebook's storage. Dev OS confirmed: Ubuntu (laptop already is 22.04).

## Phase checklist

- [x] Phase 0 — Workspace, persistence, hardware audit
- [ ] Phase 1 — Architecture pinned (research synthesis → docs/architecture.md)
- [ ] Phase 2 — Local foundations: conda envs, repos cloned, MuJoCo G1 model loads
- [ ] Phase 3 — Motion path on known data: pre-retargeted dance motion (e.g. LAFAN1)
      visualized in MuJoCo viewer
- [ ] Phase 4 — Video front-end: video → SMPL → retargeted G1 motion (our own video)
- [ ] Phase 5 — Training: cloud GPU job for tracking policy on one motion; sim verify
- [ ] Phase 6 — Deploy: policy runs on real G1 (hung from gantry first), push test
- [ ] Phase 7 — UI: web app orchestrating stages end-to-end with progress + preview
- [ ] Phase 8 — Hardening: error handling, docs, repeatability, second/third dance

## Current status

Phase 0 complete; Phase 1 in progress — research workflow running
(6 parallel deep-dives: motion extraction, retargeting, RL controller, deployment,
GPU strategy, turnkey alternatives).

## Next actions

1. Read research synthesis → write `docs/architecture.md` + decision log entries.
2. Create conda envs + clone chosen repos into `third_party/`.
3. Get a pre-made G1 dance motion playing in the MuJoCo viewer (Phase 3) — proves the
   motion format end of the pipeline before touching video or training.

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
