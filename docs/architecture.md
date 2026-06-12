# Architecture: Video → Unitree G1 Dance Pipeline

**Status:** PINNED (Phase 1 complete). **Date:** 2026-06-12.
**Derived from:** `docs/research-findings.md` (6 parallel deep-dives, 2026-06-11) + user decisions 2026-06-12.
**Robot:** Unitree G1 EDU Ultimate, 29 DoF, Inspire FTP hands, Jetson Orin PC2 @ 192.168.123.164.
**Hard requirements:** push-robust RL whole-body tracking (no open-loop playback); minimal human input per dance; one-button start; dance area = hard flat ground, **≤2 m radius**.
**Compute:** laptop Ubuntu 22.04 CPU-only; cloud GPU = **GreenNode AI Platform notebook instance with one RTX 4090** (managed via webpage, not an API).

---

## 1. Component choices (per stage)

| Stage | Component | Where it runs |
|---|---|---|
| Video → world-grounded SMPL | **GVHMR** (zju3dv, ckpt `gvhmr_siga24_release.ckpt`; static-camera flag for tripod video) | GreenNode 4090 |
| SMPL → G1 29-DoF trajectory | **GMR** (`unitree_g1`, branch `master`, pinned commit) | Laptop CPU (real-time) |
| CSV → training NPZ | `whole_body_tracking/scripts/csv_to_npz.py` (launches Isaac Lab headless) | GreenNode 4090 |
| RL tracking policy | **BeyondMimic** (`whole_body_tracking`, task `Tracking-Flat-G1-v0`, rsl_rl PPO, 4096 envs, Isaac Lab **2.1.0** / Isaac Sim **4.5.0**) | GreenNode 4090 |
| Sim verification gate | `motion_tracking_controller mujoco.launch.py` (identical controller binary) + `unitree_mujoco` (DDS domain 1, iface `lo`) | Laptop CPU |
| Deploy | **motion_tracking_controller + unitree_bringup**, ONNX Runtime CPU, 500 Hz controller / 50 Hz policy, Docker `qiayuanl/unitree:jazzy` (multi-arch) | Jetson Orin PC2 (onboard) |
| Hands (later, optional) | separate publisher on `rt/inspire_hand/ctrl/*` (FTP topics), beat-synced keyframes | PC2 or laptop |
| Smoke-test motions | Unitree **LAFAN1_Retargeting_Dataset** (`dance1_subject2` etc., 30 fps 29-DoF CSV) | data |
| UI | local web app (laptop) orchestrating all stages with progress, MuJoCo preview, and an explicit human-confirmed deploy step | Laptop |

**Why these** (full reasoning + rejected alternatives in `research-findings.md` §SYNTHESIS):
- BeyondMimic is the only stack with peer-reviewed real-G1 push-recovery evidence (arXiv 2508.08241) — push robustness is a hard requirement. GMR ships an exporter literally written "for beyondmimic" (zero glue-format risk), and the deploy half is verified onboard-capable with full 29-joint config.
- GVHMR is the only video estimator with first-class integration in both GMR and PBHC, skips SLAM on tripod footage, and emits per-joint contact probabilities for cleanup.
- Rejected as primary: unitree_rl_mjlab (too young — but see §2 fallback), SONIC/GEAR universal tracker (JetPack 6 reflash + TensorRT sharp edges; kept as Phase-4 evaluation track), PBHC (non-commercial license, 23-DoF annealed, IsaacGym legacy), VideoMimic (terrain-centric, stalled).

## 2. GPU strategy — GreenNode notebook (user decision 2026-06-12)

Replaces the research recommendation (RunPod API dispatch). Consequences:

- **One persistent JupyterLab instance, RTX 4090, managed manually via the GreenNode webpage.** No programmatic pod create/terminate; the UI orchestrator drives the instance over SSH/HTTP (terminal-in-notebook), or jobs are started by Claude/user in a notebook terminal. Check later whether GreenNode exposes any API; assume webpage-only.
- **No Docker inside a notebook container** (assume unavailable). Therefore the NGC `isaac-lab:2.1.0` image plan is dead. Instead: **pip-install Isaac Sim 4.5.0 + Isaac Lab v2.1.0 into a conda env inside the notebook** (`pip install isaacsim[all,extscache]==4.5.0 --extra-index-url https://pypi.nvidia.com`, then Isaac Lab repo at tag v2.1.0). RTX 4090 satisfies the RT-core requirement (A100/H100 would not).
- **Bounded fallback:** if Isaac Sim refuses to run in the notebook image (driver/EGL/glibc issues) after ~half a day of effort, switch the training backend to **mjlab / unitree_rl_mjlab** (same BeyondMimic recipe, same authors, pure pip+CUDA, no Isaac Sim) and use its `csv_to_npz`/ONNX-export path. This was already the designated migration target; the notebook constraint just lowers the bar for switching.
- **Storage:** notebook instances have persistent disk while they exist — W&B is no longer needed as the artifact bus. But `whole_body_tracking` hard-depends on a W&B registry (collection `Motions`) for motion fetch. Options, in order: (a) user creates a free W&B account, we use it as designed; (b) patch the motion-loading code to read local NPZ. Decide when provisioning. **Artifacts flow laptop ↔ GreenNode via the Jupyter file API / scp**; everything important is mirrored back to `~/g1-dance/data/` so the instance is disposable.
- **GVHMR runs on the same instance** (own conda env: Python 3.10, PyTorch 2.3 + CUDA 12.1; checkpoints ~few GB). Keep per-repo conda envs — Isaac Lab and GVHMR must not share one env.
- **Cost sanity:** 1.5k–10k PPO iterations ≈ 2–10 h per dance on the 4090. Benchmark `dance1_subject2` first. Pause/stop the instance when idle — an always-on notebook is the main cost risk.

## 3. End-to-end data flow (formats at each boundary)

```
[1] dance.mp4 — single continuous shot, one person, full body, 30/60 fps (trim cuts!)
      ↓  GreenNode 4090 (GVHMR env)
[2] GVHMR → output.pt — smpl_params_global {global_orient, transl, body_pose(63), betas}
      + per-joint stationary probabilities, at video fps
      ↓  laptop CPU (GMR env)
[3] GMR gvhmr_to_robot.py --robot unitree_g1 → motion.pkl
      (root_pos xyz, root_rot quat **xyzw**, dof_pos[29], fps)
      → visual gate: vis_robot_motion.py + automated vet (see §5)
      ↓  laptop CPU
[4] GMR batch_gmr_pkl_to_csv.py → motion.csv — 30 fps LAFAN1 convention:
      cols 0-2 root xyz, 3-6 quat xyzw, 7-35 joints
      (legs 0-11, waist y/r/p 12-14, L arm 15-21, R arm 22-28)
      ↓  GreenNode 4090 (Isaac Lab env)
[5] csv_to_npz.py → motion.npz @50 fps → registry
[6] train.py --task=Tracking-Flat-G1-v0 --headless → policy.onnx
      (joint order + PD gains embedded in ONNX metadata)
      ↓  laptop CPU
[7] sim2sim gate: mujoco.launch.py policy_path:=policy.onnx (+ unitree_mujoco contract check)
      ↓  Jetson PC2 (Docker qiayuanl/unitree:jazzy, --network host --privileged)
[8] real.launch.py network_interface:=eth0 policy_path:=...
      Joystick: L1+A standby → R1+A start → B damping e-stop
```

Quaternion contract: CSV is **xyzw** (converted to wxyz internally downstream). Never hand-roll the pkl→csv step. Keep one canonical motion contract: 30 fps 36-col CSV.

## 4. Build phases

(Tracked in `PROJECT_STATE.md`; renumbered there as Phases 2–8.)

1. **Local foundations** — conda envs, repos in `third_party/`, G1 model loads in MuJoCo on the laptop.
2. **Known-good motion** — LAFAN1 `dance1_subject2` CSV plays back in MuJoCo viewer (kinematic). Proves format handling before any GPU spend.
3. **Video front-end** — GVHMR (cloud) → GMR (laptop) → screened G1 motion from our own video.
4. **Training** — provision GreenNode notebook, benchmark + train `dance1_subject2` policy, then our video's motion; sim2sim gate.
5. **Deploy** — onboard PC2 container, gantry-first protocol, push test. Exit: G1 dances + survives a moderate push.
6. **UI** — web app: upload video → progress through stages → MuJoCo preview → explicit deploy confirmation → one-button start.
7. **Hardening** — second/third dance, error handling, docs; optional: hands channel, SONIC eval, mjlab migration.

## 5. Motion vetting gate (2 m-radius constraint, user 2026-06-12)

Every retargeted motion passes an automated check before training (laptop, step [3]→[4]):
- **Root XY excursion** from start ≤ **1.5 m** (0.5 m margin for estimator drift — contact+IMU odometry drifts in XY; choreography is robot-relative). Reject or offer root-translation scaling.
- Joint position/velocity within G1 limits after GMR clamping; flag frames near limits.
- Foot-skate metric + ground penetration; flag fast-spin segments (GVHMR L/R-flip risk) for the human preview.
- No floorwork in v1 (kneeling/lying retargets poorly and the area is small).
- Always: human watches the MuJoCo preview in the UI before training is launched (the one "minimal human input" step, plus the deploy confirmation).

## 6. Risks (top items; full list in research-findings.md §5)

1. Factory motion service vs `rt/lowcmd` conflict → violent jitter. Protocol: hoist → zero torque (L2+Y) → debug mode (L2+R2) / `MotionSwitcherClient.ReleaseMode()` before any custom controller.
2. Debug mode has **no factory e-stop** — only the controller's B-damping. Gantry is the real e-stop for every first run; never run factory recovery modes with Inspire hands installed.
3. Inspire FTP hand mass (~0.5 kg/wrist) unmodeled → degraded tracking / shoulder overheat (unitree_sdk2_python #129). Add hand mass or payload DR to the training asset; sim2sim ablation before hardware.
4. GVHMR dance failure modes: foot-skate, L/R flips on back-facing spins, prior collapse under blur. Mitigate via §5 gate + static-camera flag + single-shot clips.
5. Silent format corruption (quat order, fps, 23-vs-29 DoF configs). Use GMR's official exporter only; never mix PBHC/ASAP 23-DoF motion files.
6. Isaac Sim pip-install may fight the GreenNode notebook image → bounded fallback to mjlab (§2).
7. License: GVHMR (ZJU non-commercial), LAFAN1 motions (NC-ND), SMPL-X (research). Fine for personal/research use; chokepoint only if productized.

## 7. What I need from the user (timing)

- **Before Phase "Training":** GreenNode notebook instance up + a way for me to reach it (Jupyter URL/token, or SSH); confirm GPU = RTX 4090. Optionally a free W&B API key (else I patch motion loading).
- **Before video front-end:** SMPL-X body-model download requires a free registration at smpl-x.is.tue.mpg.de (license click-through) — user must create the account; I handle the rest.
- **Before Deploy:** robot on gantry, e-stop in hand, explicit go.
