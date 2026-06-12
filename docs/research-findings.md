# Research findings (raw, 2026-06-11)


## video-to-motion

### Summary
As of mid-2026, world-grounded monocular video-to-SMPL has consolidated around a handful of methods, and the humanoid-imitation community has effectively standardized on GVHMR (zju3dv, SIGGRAPH Asia 2024 / TPAMI 2026) as the video front-end. Concretely: KungfuBot/PBHC's README documents its video pipeline as Video -> GVHMR -> SMPL -> contact-aware filtering -> retargeting (Mink or PHC-style) -> IsaacGym RL -> MuJoCo/real Unitree G1; GMR (ICRA 2026, the retargeter behind TWIST and the most popular standalone SMPL-X -> G1-29DoF retargeter, 2.3k stars, active Jan 2026) officially supports GVHMR as its recommended monocular-video estimator; the community video path into BeyondMimic (which itself only consumes pre-retargeted generalized-coordinate NPZ motions) is GVHMR + GMR. ASAP (LeCAR/NVIDIA, RSS 2025) used TRAM to reconstruct its Ronaldo/LeBron/Kobe athlete motions from video into SMPL. VideoMimic (CoRL 2025 Best Student Paper) uses VIMO (TRAM's per-person video transformer) + ViTPose + SAM2 + BSTRO contact + MegaSaM/MonST3R for joint human-and-scene world-coordinate reconstruction -- powerful but scene-centric and heavy, overkill for flat-ground dance. WHAM (CVPR 2024) and 4D-Humans/PHALP are now legacy for this use case: WHAM is superseded by GVHMR/TRAM in every 2025-2026 robot pipeline, and 4D-Humans/PHALP output camera-frame (not world-grounded) poses, surviving mainly as the HMR2 feature extractor inside GVHMR and GENMO and as a multi-person tracker.

The 2025-2026 newcomers worth tracking: PromptHMR (CVPR 2025, same author as TRAM, effectively its successor -- SMPL-X output, multi-person, promptable, video mode = DROID-SLAM + Metric3D for world coordinates, SOTA on world-frame metrics); GENMO/GEM (NVlabs, ICCV 2025, code Oct 2025, renamed GEM Dec 2025, GEM-SMPL released Mar 2026 -- a unified estimation+generation diffusion model that does global motion estimation AND music-to-dance, but NVIDIA non-commercial license); and Meta's SAM 3D Body (Nov 2025, MHR representation), on top of which a Dec 2025 paper (arXiv 2512.21573) built world-coordinate recovery with contact-aware global optimization and direct Unitree G1 retargeting -- promising but its pipeline code release is unverified. CRISP (Dec 2025) and HumanMM (multi-shot videos) beat TRAM on world-grounded benchmarks but have no robot-pipeline adoption yet.

On compute: every viable method requires CUDA for practical inference. GVHMR officially needs Python 3.10 + PyTorch 2.3/CUDA 12.1, with checkpoints for GVHMR itself, HMR2, ViTPose-h, and YOLOv8x; since March 2025 its default visual odometry is SimpleVO (DPVO optional, discouraged), and for static-camera (tripod) dance videos no SLAM runs at all, making it the lightest of the bunch (a 1-minute clip processes in minutes on a single RTX 4090/L4-class GPU; ~8-12 GB VRAM suffices). TRAM/PromptHMR need DROID-SLAM CUDA compilation and more VRAM (16GB+ recommended). CPU-only inference is not supported or documented for any of them and would be hours-per-minute-of-video at best -- not viable on the user's GPU-less laptop. Hosted options: an official-author GVHMR HuggingFace Space (LittleFrog/GVHMR) for one-off clips, and Meshcapade MoCapade 3.5 (commercial SaaS by the SMPL creators, free testing tier, outputs SMPL/SMPL-X as GLB/FBX/.smpl from a single video) as a zero-GPU API route. Since the RL tracking policy (BeyondMimic/PBHC) needs a cloud GPU anyway, running GVHMR on the same rented GPU is the natural fit; the Jetson Orin should only run the exported policy, not pose extraction.

Known failure modes on dance content (all methods, GVHMR included): foot sliding/skating (GVHMR issue #21; the HTD-Refine paper measurably reduces it as a post-process), left/right 2D-keypoint flips during fast spins when the dancer faces away (causing root-yaw glitches and accumulated heading drift), near-stationary or prior-collapsed predictions under heavy occlusion and motion blur, floor penetration/floating requiring contact-aware correction (which PBHC's motion-processing stage explicitly handles, exploiting GVHMR's per-joint stationary probabilities), scale/trajectory drift with moving cameras, and broken global trajectories across video cuts (multi-shot dance edits must be trimmed to single continuous shots). GVHMR is single-person (YOLO-tracked subject); TRAM and PromptHMR handle multi-person. Output formats: GVHMR -> .pt dict with smpl_params_global (global_orient, transl, 63-d body_pose, betas) at video FPS, directly consumed by GMR's smplx loader and PBHC; TRAM -> per-person .pkl with SMPL pose + world trajectory + camera; PromptHMR -> SMPL-X in MCS/GLB; GENMO -> SMPL .pt in global + camera coordinates.

### Recommendation
Use GVHMR (github.com/zju3dv/GVHMR, checkpoint gvhmr_siga24_release.ckpt, main branch) running on a rented cloud GPU as the video -> world-grounded SMPL-X front-end, feeding either PBHC's motion-processing stage (if you adopt KungfuBot's full stack) or GMR (github.com/YanjieZe/GMR, MIT, G1-29DoF) for retargeting into BeyondMimic's NPZ motion format. Justification: it is the only estimator with first-class, documented integration in BOTH of the leading G1 motion-imitation stacks (PBHC uses it for video extraction; GMR lists it as the recommended video estimator), it outputs per-joint stationary/foot-contact probabilities that PBHC's contact-aware filtering needs for dance-quality ground contact, it skips SLAM entirely for static-camera dance videos (fastest and most robust option for tripod-shot choreography), and it is actively maintained (SimpleVO default Mar 2025, TPAMI 2026 acceptance). Run it on the same cloud GPU you will need anyway for Isaac Lab policy training; for quick experiments without renting, use the LittleFrog/GVHMR HuggingFace Space or Meshcapade MoCapade as a hosted fallback. Keep PromptHMR (github.com/yufu-wang/PromptHMR) as the upgrade path if you later need multi-person or moving-camera footage.

### Repos
- **GVHMR (zju3dv)** — https://github.com/zju3dv/GVHMR
  - why: De-facto standard video->world-grounded-SMPL front-end for humanoid imitation: explicitly used by KungfuBot/PBHC and officially supported by GMR for G1 retargeting. Gravity-View coordinates avoid yaw drift; outputs smpl_params_global (global_orient, transl, body_pose, betas) plus per-joint stationary (foot contact) probabilities. Static-camera mode needs no SLAM -- ideal for tripod dance videos.
  - reqs: Python 3.10, PyTorch 2.3.0 + CUDA 12.1 (GPU required, ~8-12GB VRAM; no CPU path); checkpoints: gvhmr_siga24_release.ckpt, HMR2, ViTPose-h, YOLOv8x; SMPL/SMPL-X model registration; hosted demo at huggingface.co/spaces/LittleFrog/GVHMR
  - license: Custom ZJU license: educational/research/non-profit only; commercial use requires permission (xwzhou@zju.edu.cn)
  - maturity: 1.6k stars; SIGGRAPH Asia 2024, TPAMI 2026; last notable update Mar 2025 (SimpleVO replaced DPVO as default, f_mm focal option); known issue: foot sliding (issue #21)
- **GMR (General Motion Retargeting)** — https://github.com/YanjieZe/GMR
  - why: The retargeting bridge: converts GVHMR SMPL-X output (also AMASS, BVH, FBX) to Unitree G1 29-DoF motions in real time ON CPU (35-70 FPS) -- runs on the GPU-less laptop. ICRA 2026 paper; retargeter for TWIST; its NPZ output feeds BeyondMimic training.
  - reqs: Pure Python, CPU-only real-time; no CUDA needed
  - license: MIT
  - maturity: 2.3k stars, 392 forks, active Jan 2026; supports 18 humanoids incl. G1 29-DoF
- **KungfuBot / PBHC** — https://github.com/TeleHuman/PBHC
  - why: End-to-end reference pipeline closest to the goal: Video -> GVHMR -> SMPL -> contact-aware filtering/motion correction -> retargeting (Mink/PHC) -> IsaacGym RL tracking policy -> MuJoCo sim2sim -> real G1. Proves GVHMR quality is sufficient for highly dynamic motions on G1 hardware.
  - reqs: IsaacGym Preview 4 (deprecated, Python 3.8, needs older NVIDIA driver stacks -- cloud GPU required), MuJoCo for sim2sim
  - license: CC BY-NC 4.0 (non-commercial)
  - maturity: 897 stars; Oct 2025 update added general motion tracking; NeurIPS 2025 paper
- **BeyondMimic (whole_body_tracking)** — https://github.com/HybridRobotics/whole_body_tracking
  - why: SOTA robust motion-tracking RL controller for G1 (handles spins, dances, cartwheels; survives pushes). Consumes retargeted generalized-coordinate NPZ motions (e.g., from GMR), NOT video directly. Deployment via separate https://github.com/HybridRobotics/motion_tracking_controller (C++, 500Hz state estimation).
  - reqs: Isaac Lab v2.1.0, Python 3.10+, Linux, NVIDIA GPU for training (cloud)
  - license: MIT
  - maturity: 2.1k stars, 296 forks, 181 commits, active; task Tracking-Flat-G1-v0
- **TRAM** — https://github.com/yufu-wang/tram
  - why: What ASAP used for its athlete video->SMPL motions; masked DROID-SLAM gives metric-scale global trajectory; multi-person. Its VIMO video model is also VideoMimic's human-pose component. Now effectively superseded by PromptHMR from the same author (integrated June 2025).
  - reqs: Python 3.10, CUDA (DROID-SLAM compilation), ~16GB+ VRAM recommended; heavier/slower than GVHMR
  - license: MIT
  - maturity: 619 stars; Feb 2025 training code + gravity/floor prediction; author has moved focus to PromptHMR
- **PromptHMR** — https://github.com/yufu-wang/PromptHMR
  - why: CVPR 2025 successor to TRAM; SOTA world-coordinate video motion among published methods; SMPL-X output, multi-person, promptable (boxes/text); video mode = DROID-SLAM + Metric3D. Upgrade path if multi-person or moving-camera footage matters.
  - reqs: PyTorch 2.4/2.6 + CUDA, custom CUDA wheels for world-coordinate pipeline; GPU mandatory
  - license: License file present, type not standard-labeled -- verify before commercial use
  - maturity: 434 stars, 12 commits, young repo (2025), no releases yet
- **GENMO / GEM (NVlabs)** — https://github.com/NVlabs/GENMO
  - why: ICCV 2025 generalist estimation+generation model: global motion from video AND music-to-dance generation in one framework; could both extract and synthesize choreography. GEM-SMPL released Mar 2026.
  - reqs: Python 3.10, PyTorch + CUDA 12.4, HMR2 feature extractor, HF checkpoints; optional ONNX runtime webcam mode
  - license: NVIDIA OneWay Noncommercial (non-commercial only)
  - maturity: 442 stars, code released Oct 2025, renamed GEM Dec 2025, active Mar 2026
- **WHAM** — https://github.com/yohanshin/WHAM
  - why: CVPR 2024 world-grounded baseline (SLAM camera + motion decoder). Legacy: no 2025-2026 robot pipeline adopted it over GVHMR/TRAM; include only as fallback/comparison.
  - reqs: CUDA GPU, DPVO/DROID-SLAM, ViTPose, SMPL registration; Colab demo available
  - license: MIT
  - maturity: 1.1k stars; effectively in maintenance mode, no 2025-2026 updates visible
- **4D-Humans (HMR2.0) + PHALP** — https://github.com/shubham-goel/4D-Humans
  - why: Camera-frame only -- NOT world-grounded, so unsuitable alone for dance global motion. Relevant because HMR2 is the feature backbone inside GVHMR and GENMO, and PHALP (https://github.com/brjathu/PHALP) provides multi-person tracking if needed.
  - reqs: CUDA GPU for practical inference
  - license: MIT
  - maturity: 1.6k stars (4D-Humans), 348 stars (PHALP); stable, low activity
- **VideoMimic** — https://github.com/hongsukchoi/VideoMimic
  - why: Full real-to-sim-to-real video->G1 pipeline (CoRL 2025 Best Student Paper). Front-end = SAM2 + ViTPose + VIMO + BSTRO contact + MegaSaM/MonST3R scene reconstruction; joint human-scene world-coordinate optimization. Best when terrain/scene interaction matters; heavyweight overkill for flat-floor dance.
  - reqs: Multiple CUDA-heavy models (MegaSaM, VIMO); cloud GPU strongly required
  - license: MIT
  - maturity: 798 stars; sim + preliminary sim2real code released Sep 15 2025
- **SAM 3D Body (Meta)** — https://github.com/facebookresearch/sam-3d-body
  - why: Nov 2025 robust whole-body mesh recovery (MHR representation, checkpoints facebook/sam-3d-body-dinov3 and -vith on HF). A Dec 2025 paper (arXiv 2512.21573) builds world-coordinate motion recovery with contact-aware global optimization plus direct Unitree G1 retargeting on top of it -- watch this as a potential 2026 replacement front-end; per-frame robustness under occlusion is its strength.
  - reqs: CUDA GPU; per-image model so video use needs the external temporal/trajectory optimization layer
  - license: SAM license (check repo; research-permissive, restrictions apply)
  - maturity: New (Nov 2025), Meta-backed, active; the world-coordinate retargeting pipeline (2512.21573) code release unverified
- **Meshcapade MoCapade 3.5 (hosted)** — https://meshcapade.com/
  - why: Only credible hosted/API service: single-video markerless mocap to SMPL/SMPL-X (also GLB/FBX) by the SMPL creators; free testing tier, developer API. Zero-GPU option from the laptop; also the licensing channel for commercial SMPL use.
  - reqs: None local (browser/API); per-clip credits
  - license: Commercial SaaS, custom/enterprise pricing; free trial credits
  - maturity: MoCapade 3.5 live (2025-2026); company founded by SMPL authors; PromptHMR authors affiliated

### Gotchas
- No CPU-only path exists for any world-grounded estimator: GVHMR/TRAM/PromptHMR/GENMO all require CUDA. The GPU-less laptop cannot run extraction locally; plan for one cloud GPU (single 4090/L4-class, ~8-16GB VRAM) used for both GVHMR extraction (minutes per clip) and the RL training you need anyway. Do NOT plan to run pose extraction on the Jetson Orin -- aarch64 builds of these stacks are undocumented and the Orin should be reserved for policy inference.
- License minefield for a product: GVHMR is research/non-commercial (commercial needs written permission from ZJU), PBHC is CC BY-NC 4.0, GENMO is NVIDIA non-commercial, and the SMPL/SMPL-X body models themselves are research-only (commercial SMPL licensing goes through Meshcapade). MIT-clean components: GMR, BeyondMimic, TRAM, WHAM, 4D-Humans, VideoMimic, ASAP.
- Dance-specific failure modes to engineer around: (1) foot sliding/skating in GVHMR output (issue #21) -- rely on PBHC-style contact-aware filtering using GVHMR's stationary-joint probabilities, or HTD-Refine post-processing; (2) fast spins cause left/right keypoint flips when the dancer faces away, producing root-yaw glitches and heading drift -- prefer GVHMR's gravity-view representation and manually inspect spin segments; (3) heavy occlusion/motion blur collapses predictions toward priors (near-stationary output); (4) multi-shot edited dance videos break global trajectories -- trim to single continuous shots before processing; (5) extracted motions can exceed G1 joint velocity/torque limits -- GMR clamps limits but the RL tracking reward, not the retargeter, is what makes it physically executable.
- GVHMR is single-person (YOLO-tracked subject). For duet/group dance videos or background people, either crop the video or switch to PromptHMR/TRAM (multi-person). For static tripod footage always use GVHMR's static-camera flag to bypass visual odometry entirely (faster and more stable global translation).
- BeyondMimic does NOT take video or SMPL directly -- it consumes retargeted NPZ motions in robot generalized coordinates (their HuggingFace LAFAN1 set, or your own via GMR). The full chain is: video -> GVHMR (cloud GPU) -> SMPL-X -> GMR (laptop CPU) -> NPZ -> BeyondMimic Isaac Lab v2.1.0 training (cloud GPU) -> motion_tracking_controller C++ deployment on G1. PBHC instead uses deprecated IsaacGym Preview 4 (Python 3.8, old driver constraints) -- harder to provision on 2026 cloud images than Isaac Lab.
- None of these estimators produce articulated finger motion usable for the Inspire FTP hands: GVHMR outputs SMPL-X with neutral hands. If finger choreography matters, add a separate hand estimator (e.g., WiLoR or HaMeR via the mocap-wrapper repo https://github.com/AClon314/mocap-wrapper, which also one-command-installs GVHMR/TRAM) or script hand poses to the beat.
- Verify DoF mapping: G1 EDU Ultimate is the 29-DoF configuration (incl. 3-DoF waist and wrists); GMR and BeyondMimic target g1_29dof models but several older pipelines (some ASAP configs) assume 23-DoF with waist locked -- mismatched URDF/MJCF DoF configs are a common silent retargeting bug.

### Open questions
- Has the SAM 3D Body world-coordinate retargeting pipeline (arXiv 2512.21573, targets Unitree G1 directly) released code? If yes, it could replace GVHMR+contact-filtering with better occlusion robustness for spins -- worth re-checking facebookresearch repos before freezing the architecture.
- Meshcapade MoCapade exact API pricing/quota and whether its SMPL-X export includes the per-frame world translation quality (foot contact) needed for retargeting -- free tier testing on one dance clip would answer both.
- GVHMR commercial-permission terms from ZJU (xwzhou@zju.edu.cn) if this project ever becomes a product rather than research.
- Whether PromptHMR's world-coordinate video mode (DROID-SLAM + Metric3D) measurably beats GVHMR on a real dance test clip with fast spins -- published metrics say yes on EMDB, but no robot pipeline has adopted it yet; a 1-clip A/B on the rented GPU would settle it.
- GEM-SMPL (Mar 2026 release) quality for music-to-dance generation as a complementary feature (generate choreography from audio when no reference video exists) -- unverified beyond paper claims, and non-commercial license applies.

## retargeting

### Summary
As of mid-2026, retargeting SMPL/SMPL-X motion to the Unitree G1 29-DoF skeleton is a solved, commoditized step with one clear open-source leader: GMR (YanjieZe/GMR, "General Motion Retargeting", ICRA 2026, arXiv:2510.02252 "Retargeting Matters"). GMR natively supports `unitree_g1` as legs(2x6)+waist(3)+arms(2x7)=29 DoF including all three waist joints, takes SMPL-X (AMASS/OMOMO), BVH (LAFAN1/Xsens), FBX, and — critically for this project — raw GVHMR video-pose output via `scripts/gvhmr_to_robot.py`. It runs real-time on CPU only (35-70 FPS on desktop CPUs; no CUDA anywhere), outputs .pkl per-frame tuples (root_pos, root_rot, dof_pos, fps), and ships `scripts/batch_gmr_pkl_to_csv.py` that emits the exact LAFAN1-convention CSV (cols 0-2 root xyz, 3-6 quat xyzw, 7-35 the 29 joints) downsampled to 30 fps that BeyondMimic's whole_body_tracking `scripts/csv_to_npz.py` consumes (default --input_fps 30, resamples to 50 fps npz with fps/joint_pos/joint_vel/body_pos_w/body_quat_w/body_lin_vel_w/body_ang_vel_w). I verified the G1 joint order in whole_body_tracking matches GMR/Unitree convention including waist_yaw/waist_roll/waist_pitch between legs and arms. The "Retargeting Matters" paper shows GMR-retargeted motions achieve downstream RL tracking success comparable to Unitree's closed-source retargeting and better than other open-source baselines.

The alternatives each have a disqualifier for this pipeline. Unitree's official retargeting (interaction-mesh + IK numerical optimization with end-effector, joint position/velocity, and anti-foot-slip constraints) is closed-source — only the pre-retargeted LAFAN1_Retargeting_Dataset on HuggingFace exists (G1 29 DoF, 30 fps, 37-col CSV), so it cannot retarget new dance videos; it serves as gold-standard test data and as the format spec everyone copies. KungfuBot/PBHC (TeleHuman/PBHC, NeurIPS 2025) has a solid SMPL retargeting stage (Mink differential-IK and PHC gradient-optimization variants, plus contact-mask correction and physics-based motion filtering) but its robot config is `unitree_g1_29dof_anneal_23dof` — wrists and waist roll/pitch are annealed/locked, so you lose 6 DoF of expressiveness that matter for dance; it also trains in legacy IsaacGym and is CC BY-NC 4.0. ProtoMotions v3 (NVlabs) supports G1 with PyRoki-based retargeting and a full deployment tutorial, but it is its own heavyweight ecosystem (Isaac Lab 2.3, multi-A100 training examples) with a different motion format — better viewed as an alternative to BeyondMimic entirely, not as a retargeting front-end for it. The newest serious entrant is OmniRetarget (arXiv:2509.26633), whose code shipped inside Amazon FAR's holosoma framework (amazon-far/holosoma, Apache-2.0, ~1.4k stars): interaction-preserving Laplacian mesh deformation with hard kinematic/collision constraints, excellent for loco-manipulation and contact-rich motion, but again packaged with its own tracking stack and npz qpos format rather than BeyondMimic's. A March 2026 paper "Make Tracking Easy: Neural Motion Retargeting" (arXiv:2603.22201) signals a coming wave of learned retargeters but has no proven public tooling yet.

Practical interface notes verified from source: GMR install is conda python=3.10, `pip install -e .`, plus SMPL-X body model files (registration download from smpl-x.is.tue.mpg.de; may need to flip `ext` from npz to pkl in smplx/body_models.py) and `conda install -c conda-forge libstdcxx-ng` for the MuJoCo viewer on Ubuntu 22.04. Single-motion command: `python scripts/smplx_to_robot.py --smplx_file <motion.npz> --robot unitree_g1 --save_path out.pkl`; batch: `scripts/smplx_to_robot_dataset.py`; video-pose: `scripts/gvhmr_to_robot.py --gvhmr_pred_file <pred> --robot unitree_g1`; visual check: `scripts/vis_robot_motion.py`. Quality knobs: per-robot IK task weights in the GMR config, default motor velocity limit clamp (use_velocity_limit=True, 3*pi rad/s), joint position limits enforced by the IK; no explicit self-collision optimization (MuJoCo-IK based, documented imperfect cases in TEST_MOTIONS.md: ground-lying/contact-heavy motions and some fast DanceDB dances jitter). Note the entire GMR stage runs on the no-GPU laptop, but BeyondMimic's csv_to_npz.py itself launches Isaac Lab/Isaac Sim headless to compute body kinematics and pushes to a WandB registry, so that conversion belongs on the cloud GPU box alongside training (Isaac Lab v2.1.0 / IsaacSim 4.5.0, WandB registry collection named "Motions", WANDB_ENTITY set).

### Recommendation
Use GMR (github.com/YanjieZe/GMR, robot id `unitree_g1`) as the retargeting stage, running entirely on the operator laptop (CPU-only, python 3.10 conda env): video-pose output -> scripts/gvhmr_to_robot.py (or scripts/smplx_to_robot.py for AMASS-style SMPL-X) -> .pkl -> scripts/batch_gmr_pkl_to_csv.py -> 30fps LAFAN1-convention CSV -> (on cloud GPU) whole_body_tracking scripts/csv_to_npz.py -> 50fps npz -> BeyondMimic training. Justification: it is the only actively maintained (commits through Jan 2026), MIT-licensed retargeter that simultaneously (a) supports the full G1 29-DoF including the 3-DoF waist and 3-DoF wrists, (b) ships an explicit BeyondMimic-compatible CSV exporter so there is zero glue-format risk, (c) is CPU-real-time so it fits the no-NVIDIA laptop, (d) already has a GVHMR video front-end script matching the video->dance goal, and (e) was validated head-to-head (arXiv:2510.02252) to match Unitree's closed-source retargeting in downstream RL tracking success. Keep Unitree's LAFAN1_Retargeting_Dataset dance CSVs as a known-good smoke-test input for the training pipeline before trusting your own retargets, and treat holosoma/OmniRetarget as the fallback only if GMR quality on contact-heavy choreography (floorwork) proves insufficient.

### Repos
- **GMR — General Motion Retargeting (YanjieZe)** — https://github.com/YanjieZe/GMR
  - why: Primary recommendation for the retargeting stage. Supports `unitree_g1` 29 DoF (legs 2x6 + waist 3 + arms 2x7), inputs SMPL-X (AMASS/OMOMO), BVH (LAFAN1/Xsens), FBX, and GVHMR video-pose output (scripts/gvhmr_to_robot.py). Output .pkl with (root_pos, root_rot xyzw, dof_pos, fps); scripts/batch_gmr_pkl_to_csv.py converts to BeyondMimic/LAFAN1-convention 30fps CSV (cols: 3 root pos, 4 quat, 29 joints) explicitly 'for beyondmimic'. Quality: IK with joint position limits, default motor velocity clamp (use_velocity_limit=True, 3*pi rad/s), per-robot tunable IK weights; no self-collision term. Install: conda create -n gmr python=3.10; pip install -e .; download SMPL-X body models (registration); flip ext npz->pkl in smplx/body_models.py if using pkl models; conda install -c conda-forge libstdcxx-ng for viewer. Default branch: master.
  - reqs: CPU-only, real-time (35-70 FPS on desktop CPUs); Ubuntu 20.04/22.04; Python 3.10; MuJoCo; SMPL-X body model files (free registration)
  - license: MIT
  - maturity: ~2.3k stars; ICRA 2026 paper arXiv:2510.02252 ('Retargeting Matters'); active through at least 2026-01-21 (Xsens BVH support); documented imperfect cases in TEST_MOTIONS.md (ground-lying motions, jitter on some DanceDB dances)
- **BeyondMimic whole_body_tracking (HybridRobotics) — consumer of retargeted motion** — https://github.com/HybridRobotics/whole_body_tracking
  - why: Defines the target format the retargeter must produce. scripts/csv_to_npz.py expects 30fps CSV (root xyz, quat xyzw cols 3-6, then 29 joints in order: 12 leg joints, waist_yaw/roll/pitch, 7 left arm, 7 right arm — verified to include the full waist) and resamples to 50fps npz {fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w} via forward kinematics run inside Isaac Lab, then uploads to a WandB registry. Train: python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 --registry_name {org}/wandb-registry-motions/{motion}. Officially feeds from the Unitree LAFAN1 CSV dataset plus KungfuBot/ASAP/HuB motions; GMR is the documented community path for new motions.
  - reqs: Isaac Lab v2.1.0 on IsaacSim 4.5.0, Linux x86_64, Python 3.10+, NVIDIA GPU mandatory (cloud), WandB account with registry collection 'Motions' + WANDB_ENTITY
  - license: MIT
  - maturity: ~2.1k stars, 296 forks, 181 commits, no tagged releases; the de-facto open G1 motion-tracking baseline in 2025-2026 (many forks/ports)
- **Unitree LAFAN1_Retargeting_Dataset (official Unitree retargeting output)** — https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset
  - why: Unitree's official retargeting: interaction-mesh + IK numerical optimization with end-effector pose, joint position/velocity constraints and anti-foot-slip. Code is NOT public — dataset only, so it cannot retarget new dance videos. G1 set is 29 DoF (incl. waist) at 30fps, 37-column CSV (7 floating-base + 30 incl. root_joint XYZQXQYQZQW notation) and is the format spec BeyondMimic adopted. Use its dance clips as known-good smoke-test motions for the training pipeline. Note: HF page returns 401 to anonymous scripted fetches; works in browser / with HF token.
  - reqs: None to consume (CSV); retargeting method itself closed-source
  - license: Non-commercial (derived from Ubisoft LAFAN1 license)
  - maturity: Published Feb 2025 by Unitree; widely used as ground truth in 2025-2026 retargeting papers (incl. arXiv:2510.02252)
- **KungfuBot / PBHC (TeleHuman) — PHC-style + Mink retargeting stage** — https://github.com/TeleHuman/PBHC
  - why: Investigated as the PHC-style option. smpl_retarget/ has two pipelines: mink_retarget (differential IK, adapted from MaskedMimic; python mink_retarget/convert_fit_motion.py <PATH>, --correct for contact-mask correction) and phc_retarget (PHC gradient-based shape+motion fitting; python phc_retarget/fit_smpl_motion.py robot=unitree_g1_29dof_anneal_23dof). Includes physics-based motion filtering and video->SMPL via GVHMR. DISQUALIFIER for dance: robot config anneals 29->23 DoF (wrists + waist roll/pitch locked), losing expressiveness; trains in legacy IsaacGym; non-commercial license. Its processed motions are still useful as extra training data (BeyondMimic docs reference them). KungfuBot2 (arXiv:2509.16638) extends to general motion tracking.
  - reqs: IsaacGym (legacy, needs older CUDA/python 3.8 stack) for training; retargeting itself CPU-capable; MuJoCo sim2sim
  - license: CC BY-NC 4.0
  - maturity: ~900 stars, NeurIPS 2025; Oct 2025 update added general motion tracking; real-G1 kungfu/dance deployments demonstrated
- **holosoma (Amazon FAR) — OmniRetarget retargeting + tracking stack** — https://github.com/amazon-far/holosoma
  - why: The newest credible alternative (code home of OmniRetarget, arXiv:2509.26633, ICRA 2026). Interaction-preserving Laplacian mesh-deformation retargeting with hard kinematic and collision constraints — measurably less foot-skating and penetration than IK baselines, best-in-class for contact-rich/floorwork motion. Supports Unitree G1 (29 DoF; dataset npz qpos = 7D base + 29 joints at 30fps, MIT-licensed on HF: huggingface.co/datasets/omniretarget/OmniRetarget_Dataset) and Booster T1. Comes bundled with its own whole-body-tracking RL pipeline (IsaacGym/IsaacSim/MuJoCo-Warp, PPO + FastSAC) — a full-stack alternative to BeyondMimic rather than a drop-in front-end; output is not BeyondMimic npz without a converter.
  - reqs: Ubuntu 22.04+; NVIDIA GPU for training (IsaacGym/IsaacSim/MJWarp); retargeting module optimization-based (CPU-feasible); Python
  - license: Apache-2.0
  - maturity: ~1.4k stars, 207 forks, 81 commits; released late 2025 by Amazon Frontier AI for Robotics; actively developed
- **ProtoMotions v3 (NVlabs)** — https://github.com/NVlabs/ProtoMotions
  - why: Checked for completeness: v3 supports Unitree G1 with a full data-prep->train->deploy tutorial and switched retargeting from Mink to a PyRoki-based optimizer. It is a self-contained ecosystem (own motion format, MaskedMimic lineage) targeting Isaac Lab 2.3/Newton/Genesis — adopting it means replacing BeyondMimic, not feeding it. Training examples cite 4xA100 for AMASS-scale; overkill for single-dance tracking. Keep as reference, not pipeline component.
  - reqs: NVIDIA GPU (Isaac Lab 2.3.0 / IsaacGym / Newton; MuJoCo inference); Python 3.x per simulator stack
  - license: Apache-2.0
  - maturity: ~1.7k stars, 203 forks; active in 2026 with v3 release and deployment docs

### Gotchas
- Quaternion order: BeyondMimic csv_to_npz.py reads CSV root quaternion as xyzw and converts to wxyz internally; Unitree LAFAN1 CSV is also XYZQXQYQZQW. If you write any custom pkl->csv glue instead of GMR's batch_gmr_pkl_to_csv.py, a wxyz/xyzw swap silently produces garbage training motions.
- Frame-rate chain: AMASS SMPL-X is often 60-120fps; GMR pkl stores source fps; batch_gmr_pkl_to_csv.py downsamples anything >30fps to 30 via downsample_factor = fps/30 — verify behavior for non-integer factors (e.g., 100fps sources) by inspecting output length; csv_to_npz.py then resamples 30->50fps. Keep your video front-end at a clean 30 or 60fps.
- GMR wants SMPL-X, not SMPL: video HMR models (GVHMR) emit SMPL — use scripts/gvhmr_to_robot.py (handles it) or scripts/smpl_to_smplx.py first; plain smplx_to_robot.py on SMPL params will fail or distort. Also requires registered download of SMPL-X body models and possibly editing ext npz->pkl in smplx/body_models.py.
- PBHC/KungfuBot motions are 23-DoF (unitree_g1_29dof_anneal_23dof: wrists + waist roll/pitch locked). If you mix their motion files into training, the locked joints stay at defaults — fine as filler data, wrong as a retargeting front-end for expressive dance.
- GMR's default motor velocity clamp (3*pi rad/s, use_velocity_limit=True) can flatten very fast dance moves; conversely disabling it produces references the real G1 cannot track. Tune per-motion and sanity-check with scripts/vis_robot_motion.py before training.
- GMR has no self-collision avoidance term; fast arm-cross choreography can produce arm-torso intersections that the RL tracking policy will then 'learn around' or fail on — visually screen every retargeted dance in MuJoCo, and check TEST_MOTIONS.md-style failure modes (ground-contact/floorwork motions are the weakest).
- whole_body_tracking's csv_to_npz.py is not a pure-CPU script — it launches Isaac Lab headless to compute body kinematics and uploads to a WandB registry. Plan for CSV->npz to run on the cloud GPU box, not the laptop; you also must pre-create the WandB registry collection 'Motions' and export WANDB_ENTITY or training fails at motion fetch.
- Unitree's own retargeting is closed-source; do not plan around 'Unitree official retargeting tools' for new videos — only the pre-retargeted LAFAN1 dataset exists (non-commercial license inherited from Ubisoft LAFAN1).
- None of these retarget fingers: the 29 DoF stop at wrist_yaw. The Inspire FTP hands need a separate channel (e.g., scripted hand poses or a hand-retargeting add-on) layered on top of the body motion artifact.
- GitHub default branch for YanjieZe/GMR is master (not main) — raw-file URLs and pip/git pins should reference master or a commit SHA.

### Open questions
- Does GMR's batch_gmr_pkl_to_csv.py handle non-integer downsample factors (e.g., 100fps->30fps) by interpolation or naive striding? Inspect locally before relying on odd-fps video sources.
- holosoma/OmniRetarget as fallback: how much work is a holosoma-retarget -> BeyondMimic CSV converter (npz qpos 36D at 30fps maps cleanly to the 36-col CSV layout, but root quat convention and joint order need verification)?
- 'Make Tracking Easy: Neural Motion Retargeting' (arXiv:2603.22201, Mar 2026) — code not yet confirmed public; re-check before committing, as a learned retargeter could improve fast-dance quality over IK.
- Exact GMR IK weight overrides for dance (upper-body faithfulness vs foot anchoring trade-off) — the per-robot config exposes task weights but optimal dance settings are undocumented; needs empirical tuning on 2-3 reference clips.
- Whether whole_body_tracking's current master still pins Isaac Lab 2.1.0/IsaacSim 4.5.0 or has moved to a newer Isaac Lab by mid-2026 — verify against the chosen cloud GPU image before provisioning.

## rl-controller

### Summary
As of mid-2026 the RL whole-body motion-tracking landscape for the Unitree G1 has consolidated around two camps. Camp 1: per-motion DeepMimic-style trackers, dominated by BeyondMimic (HybridRobotics/whole_body_tracking, MIT, 2.1k stars, UC Berkeley Hybrid Robotics + Stanford). It trains one PPO policy per reference motion (Isaac Lab 2.1.0 / Isaac Sim 4.5.0, rsl_rl, task `Tracking-Flat-G1-v0`), consumes G1-retargeted CSV motions at 30 fps (drop-in compatible with the HuggingFace LAFAN1_Retargeting_Dataset which includes dance1/dance2 clips), exports ONNX, and deploys via HybridRobotics/motion_tracking_controller (C++ ONNX CPU inference <1 ms, 50 Hz, onboard-only state estimation, ROS 2 Jazzy + legged_control2). Its paper (arXiv 2508.08241v4) documents push/disturbance recovery during tracking, heavy domain randomization, and real-G1 deployment of cartwheels, spin-kicks, sprints and dances. Critically, the BeyondMimic recipe has been re-platformed twice in 2025-26: (a) mjlab (mujocolab/mjlab, Apache-2.0, 2.5k stars, v1.4.0 May 2026, same authors — Zakka/Liao) reimplements the tracking task on MuJoCo-Warp so training needs only a CUDA GPU, no Isaac Sim; and (b) Unitree itself shipped unitree_rl_mjlab (Apache-2.0), which is the BeyondMimic tracking task ("Unitree-G1-Tracking-No-State-Estimation", with a dance1_subject2 example) plus Unitree's own official C++ ONNX-Runtime/unitree_sdk2/CycloneDDS deploy stack and unitree_mujoco sim2sim for the 29-DoF G1. KungfuBot (TeleHuman/PBHC, NeurIPS 2025) is the same per-motion idea on IsaacGym with the best documented video→motion front-end (GVHMR→SMPL→retarget with contact-aware filtering) but is CC BY-NC 4.0 and its real-robot deploy module is not fully released. ASAP (MIT) and ExBody2 (official repo edpsw/exbody2, includes 8 dance motions but no deployment code) are 2025-era and largely superseded; unitree_rl_gym/unitree_rl_lab proper only do velocity locomotion.

Camp 2: pretrained universal trackers that need NO per-motion training. The standout is NVIDIA SONIC inside NVlabs/GR00T-WholeBodyControl (2.3k stars; code Apache-2.0, weights NVIDIA Open Model License — commercial use permitted): a 42M-param motion-tracking foundation model trained on 100M frames (BONES-SEED, 142k motions retargeted to G1), released Feb 2026 with ONNX checkpoints on HuggingFace (nvidia/GEAR-SONIC: model_encoder.onnx, model_decoder.onnx, planner_sonic.onnx). It reports 99.2% real-world success over 123 motion sequences on a 29-DoF G1, push robustness via root-velocity perturbation DR, and runs fully onboard the G1's Jetson Orin via TensorRT (1-2 ms/policy step at 50 Hz) — and a third party (R. Tajima, note.com, Apr 2026) already demonstrated exactly the target pipeline: music-video → NVIDIA GEM (video/text/music→motion) → SONIC → G1 dance. HoloMotion (HorizonRobotics, Apache-2.0 code) also ships pretrained G1 tracking checkpoints (HF: HorizonRobotics/HoloMotion_models, v1.3 = 0.4B-param MoE, ~300 FPS inference, explicit "offline motion replay for dance demos" mode) but its weights are CC-BY-NC-SA 4.0. ProtoMotions3 (NVlabs, Apache-2.0) added a full G1 sim2real path with baked-obs ONNX export and is the best general research framework (MaskedMimic), but per-robot pretrained G1 checkpoints are not its focus. GMT, UniTracker, InternHumanoid, Heracles, and FRoM-W1 are real but either inference-only, checkpoint-pending, or without released deployment code.

For a video→dance product on a 29-DoF G1 EDU with no local NVIDIA GPU, the practical answer is a two-track architecture on one deploy stack: per-video BeyondMimic-style training on a rented single cloud GPU using the mjlab backend (pip-installable, no Isaac Sim container, ONNX out), deployed through Unitree's official unitree_rl_mjlab C++ controller which matches the existing CycloneDDS/unitree_sdk2 setup; plus SONIC's pretrained checkpoints as the zero-training fast path and robustness baseline running on the robot's own Jetson Orin. The video front-end is shared either way: GVHMR or NVIDIA GEM for video→SMPL motion, then GMR (YanjieZe/GMR, ICRA 2026, real-time CPU retargeting, the de-facto standard SMPL→G1-29DoF retargeter) to produce the 30 fps CSV that both BeyondMimic-style training and replay tooling consume.

### Recommendation
Use the BeyondMimic recipe on the mjlab/MuJoCo-Warp backend as the base, concretely via Unitree's official unitree_rl_mjlab (https://github.com/unitreerobotics/unitree_rl_mjlab, Apache-2.0): per-video pipeline = video → GVHMR (or NVIDIA GEM) → SMPL → GMR retarget to G1-29DoF CSV @30fps → csv_to_npz.py → train `Unitree-G1-Tracking-No-State-Estimation` with 4096 envs on a single rented cloud GPU (RTX 4090/L40S class; no Isaac Sim install needed, plain pip + CUDA, hours per motion) → policy.onnx artifact → unitree_mujoco sim2sim gate → Unitree's own C++ ONNX-Runtime/unitree_sdk2/CycloneDDS `g1_ctrl` deploy on the real G1. Justification: it is literally the BeyondMimic method (the only per-motion tracker with peer-reviewed real-G1 push-recovery evidence and dozens of third-party hardware reproductions), but productized by Unitree for this exact robot — permissive license end-to-end, 29-DoF G1 support, dance example included, ONNX output, and a deployment path that matches the already-working laptop↔G1 DDS setup; keep HybridRobotics/whole_body_tracking + motion_tracking_controller as the upstream reference when behavior diverges. In parallel, deploy the pretrained SONIC checkpoints (nvidia/GEAR-SONIC ONNX + gear_sonic_deploy TensorRT on the G1's Jetson Orin at 192.168.123.164) as a day-1, zero-training preview/fallback tracker — a third party already demonstrated video→dance on G1 with GEM+SONIC in April 2026 — and only spend cloud-GPU training on choreographies where the universal model's fidelity is insufficient.

### Repos
- **BeyondMimic training (whole_body_tracking)** — https://github.com/HybridRobotics/whole_body_tracking
  - why: SOTA per-motion whole-body tracking for G1; the method underlying the recommended stack. CSV(30fps)→NPZ motion input, WandB motion registry, PPO/rsl_rl, task Tracking-Flat-G1-v0; paper (arXiv 2508.08241v4, Nov 2025) shows real-G1 dances/cartwheels/spin-kicks, push-disturbance recovery, onboard-only state estimation.
  - reqs: Isaac Lab v2.1.0 + Isaac Sim 4.5.0, Python 3.10, Linux, NVIDIA GPU (paper used multi-L40S; single 4090 works with fewer envs)
  - license: MIT
  - maturity: 2.1k stars, 296 forks, 181 commits, active 2025-26; widely reproduced (forks, mjlab port, Unitree's own port)
- **BeyondMimic deployment (motion_tracking_controller)** — https://github.com/HybridRobotics/motion_tracking_controller
  - why: Official sim2sim+sim2real inference stack for BeyondMimic policies on real G1: C++ ONNX CPU inference (<1 ms/step, 50 Hz), policies loaded from WandB or local ONNX, MuJoCo sim2sim, explicit real-G1 support.
  - reqs: ROS 2 Jazzy (Ubuntu 24.04), legged_control2 framework, unitree_description/unitree_systems; CPU-only inference
  - license: MIT
  - maturity: 508 stars, 53 forks; maintained alongside the training repo
- **mjlab** — https://github.com/mujocolab/mjlab
  - why: Isaac-Lab-style API on MuJoCo-Warp by the BeyondMimic authors; ships the tracking task Mjlab-Tracking-Flat-Unitree-G1. Removes the Isaac Sim dependency entirely — train BeyondMimic-style policies on any CUDA cloud GPU with pip install. Best training backend given no local GPU.
  - reqs: NVIDIA GPU + CUDA only (no Isaac Sim); Python
  - license: Apache-2.0
  - maturity: 2.5k stars, v1.4.0 (May 26, 2026), 1041 commits, CI/CD, production-ready
- **unitree_rl_mjlab (RECOMMENDED BASE)** — https://github.com/unitreerobotics/unitree_rl_mjlab
  - why: Unitree's official BeyondMimic-style training-to-deployment for G1 29-DoF and 23-DoF: task Unitree-G1-Tracking-No-State-Estimation with dance1_subject2 example, csv_to_npz.py (30→50 fps), policy.onnx export, unitree_mujoco sim2sim, and official C++ ONNX-Runtime deploy (g1_ctrl) on unitree_sdk2 + CycloneDDS — matches the user's existing network/DDS setup.
  - reqs: mjlab + mujoco_warp (NVIDIA CUDA GPU for training, multi-GPU via --gpu-ids); deploy: cmake, unitree_sdk2, cyclonedds; robot net 192.168.123.x
  - license: Apache-2.0
  - maturity: 439 stars, 119 forks, active 2026, 30 open issues, no tagged releases yet (young — pin commits)
- **GR00T-WholeBodyControl / GEAR-SONIC** — https://github.com/NVlabs/GR00T-WholeBodyControl
  - why: NVIDIA's open-sourced (Feb 2026) universal motion-tracking foundation model for G1 (29 actuated joints): tracks arbitrary kinematic reference motions zero-shot, 99.2% real-world success on 123 sequences, push robustness via root-velocity perturbation DR. PRETRAINED ONNX checkpoints on HuggingFace (https://huggingface.co/nvidia/GEAR-SONIC: model_encoder.onnx, model_decoder.onnx, planner_sonic.onnx) — deployable WITHOUT any training. gear_sonic_deploy C++/TensorRT runs onboard Jetson Orin (1-2 ms/step, 50 Hz policy, 500 Hz command). Third-party video→G1-dance pipeline demonstrated with GEM+SONIC (note.com/ryosuke_tajima/n/n1341bc889c4e, Apr 2026). Includes GEM (video/text/music→motion) integration and MotionBricks.
  - reqs: Inference: Jetson Orin GPU w/ TensorRT (or desktop NVIDIA GPU). Training from scratch: 128 GPUs x 7 days; finetune '64+ GPUs recommended' (Isaac Lab 2.3.2) — use checkpoints, don't train
  - license: Apache-2.0 (code) + NVIDIA Open Model License (weights; commercial use permitted)
  - maturity: 2.3k stars, 318 forks; training code released Apr 10, 2026; active (May 2026 VLA workflow); SONIC paper arXiv 2511.07820
- **KungfuBot (PBHC)** — https://github.com/TeleHuman/PBHC
  - why: NeurIPS 2025 per-motion tracker for highly dynamic skills (kungfu, dance) on real G1; best-documented open video→motion front-end (GVHMR→SMPL, contact-aware filtering, IPMAN), adaptive tracking-reward tolerance; Oct 2025 'KungfuBot2' general motion tracking. Use its motion-processing recipe even if not its trainer.
  - reqs: IsaacGym (legacy), rsl_rl PPO, MuJoCo sim2sim; real deploy = external PC + DDS (module pluggable, not fully released)
  - license: CC BY-NC 4.0 (NON-commercial — blocks productization)
  - maturity: 897 stars, 110 forks, active through Oct 2025
- **ASAP** — https://github.com/LeCAR-Lab/ASAP
  - why: LeCAR Lab's sim2real delta-action framework with motion tracking on G1 23/29-DoF; multi-simulator (IsaacGym/IsaacSim 4.2/Genesis via HumanoidVerse); ONNX export; some pretrained checkpoints in-repo. Influential but 2025-era; per-motion quality and deployment ergonomics now behind BeyondMimic.
  - reqs: Python 3.8-3.10, CUDA GPU, IsaacGym/IsaacSim/Genesis, ROS2 + unitree_sdk2 for real robot
  - license: MIT
  - maturity: 2.0k stars; low recent commit volume (27 commits main)
- **ExBody2 (official)** — https://github.com/edpsw/exbody2
  - why: Official ExBody2 implementation (teacher-student tracking, G1+H1); ships motions_dance_release.pkl with 8 dance sequences for direct training. However: deployment code still unreleased, tiny community.
  - reqs: IsaacGym + legged_gym + rsl_rl, CUDA GPU
  - license: in-repo (check before use)
  - maturity: 65 stars, 9 commits — low activity, deploy code pending
- **HoloMotion** — https://github.com/HorizonRobotics/HoloMotion
  - why: Horizon Robotics foundation tracker: v1.3 (May 2026) 0.4B-param MoE transformer, 2000+ hours motion data, ~300 FPS inference; v1.2+ ships PRETRAINED motion/velocity tracking models (https://huggingface.co/HorizonRobotics/HoloMotion_models) with explicit 'offline motion replay for dance demos' and online VR-streaming modes, plus real-robot deploy code. Weights are non-commercial.
  - reqs: SMPL→HDF5 data pipeline, Isaac-based training (GPU); pretrained checkpoints avoid training
  - license: Apache-2.0 (code); CC-BY-NC-SA 4.0 (weights)
  - maturity: 550 stars, v1.3.2 released June 9, 2026 — very active
- **ProtoMotions3 (MaskedMimic)** — https://github.com/NVlabs/ProtoMotions
  - why: NVIDIA's general humanoid-RL framework, v3: documented G1 zero-shot sim2real pipeline, ONNX export with observation computation baked in, PyRoki one-command retargeting, MaskedMimic generative control; AMASS-scale training (~12 h on 4xA100). Great research substrate but heavier than needed for one-motion-per-video.
  - reqs: IsaacGym Prev4 / Isaac Lab 2.3.0 / Newton / MuJoCo 3+ / Genesis; multi-GPU recommended
  - license: Apache-2.0
  - maturity: 1.7k stars, actively maintained 2026
- **unitree_rl_lab** — https://github.com/unitreerobotics/unitree_rl_lab
  - why: Unitree's official Isaac Lab repo (Go2/H1/G1-29dof) with train→sim2sim→sim2real C++ deploy; but tasks are velocity locomotion, NOT motion tracking — the tracking/dance example lives in unitree_rl_mjlab instead.
  - reqs: Isaac Lab 2.3.0 + Isaac Sim 5.1.0; unitree_sdk2 C++ deploy
  - license: Apache-2.0
  - maturity: 1.1k stars, 263 forks, active
- **unitree_rl_gym** — https://github.com/unitreerobotics/unitree_rl_gym
  - why: Legacy official legged_gym-based locomotion repo (G1/H1/Go2) with LibTorch C++ deploy example; no motion tracking — relevant only as deployment reference.
  - reqs: IsaacGym/legged_gym, CUDA GPU
  - license: BSD-3-Clause
  - maturity: 3.3k stars; superseded in practice by unitree_rl_lab / unitree_rl_mjlab
- **GMT (General Motion Tracking)** — https://github.com/zixuan417/humanoid-general-motion-tracking
  - why: Universal single-policy tracker for G1 with a pretrained checkpoint — but inference-only release (MuJoCo), training and real-robot deployment code never released; 4 commits total.
  - reqs: Python 3.8, PyTorch, MuJoCo; .pkl motion files
  - license: MIT
  - maturity: 416 stars, stale (research dump)
- **InternHumanoid** — https://github.com/InternRobotics/InternHumanoid
  - why: InternRobotics all-in-one zero-shot whole-body tracking toolbox for G1/H1/GR-1 (July 2025); checkpoints and deployment code still listed as TODO — watch, don't build on.
  - reqs: legged_gym + rsl_rl + MuJoCo, CUDA GPU
  - license: MIT (code), CC-BY-NC-SA 4.0 (data)
  - maturity: 184 stars, checkpoints/deploy pending
- **GMR (General Motion Retargeting)** — https://github.com/YanjieZe/GMR
  - why: ICRA 2026 real-time CPU SMPL→humanoid retargeter (G1 supported) — the missing link between video pose estimation (GVHMR/GEM output) and the G1 CSV/NPZ motion format consumed by BeyondMimic-style training and SONIC references.
  - reqs: CPU-only, real-time
  - license: see repo (research)
  - maturity: active 2025-26, de-facto standard retargeter (used by TWIST)
- **VideoMimic** — https://github.com/hongsukchoi/VideoMimic
  - why: CoRL 2025 Best Student Paper: monocular video → real-to-sim-to-real G1 (23-DoF, 50 Hz onboard) for terrain/context skills. Validates the video→G1 concept but targets environment interaction, not precise choreography tracking.
  - reqs: GPU for 4D reconstruction + RL training
  - license: see repo
  - maturity: sim + preliminary sim2real code released Sept 15, 2025
- **LAFAN1 Retargeting Dataset (pre-retargeted G1 dance motions)** — https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
  - why: Ready-made G1-retargeted CSV motions @30 fps including dance1/dance2 subjects — the standard test input for whole_body_tracking and unitree_rl_mjlab (dance1_subject2 example); lets you validate the entire train→deploy loop before the video front-end exists.
  - reqs: none (CSV)
  - license: CC BY-NC-ND 4.0 (motions; non-commercial, no derivatives) / MIT (code)
  - maturity: ~2.7k monthly downloads, 314 MB

### Gotchas
- No NVIDIA GPU on the laptop: ALL training (mjlab, Isaac Lab, IsaacGym) and video pose estimation (GVHMR/GEM need CUDA) must run on a rented cloud GPU; the laptop can only do CPU MuJoCo visualization/sim2sim and orchestration. mjlab/unitree_rl_mjlab makes this cheap (plain pip+CUDA VM, no Isaac Sim image/licensing).
- Per-motion training wall-clock for BeyondMimic-style policies is NOT officially documented; the paper used multi-L40S training and 4096-env defaults. Budget several hours per dance on a single 4090/L40S and benchmark early; adaptive sampling halves iterations for short motions.
- License minefield: KungfuBot/PBHC is CC BY-NC 4.0 (non-commercial, explicitly bans commercial demos); HoloMotion and InternHumanoid weights/data are CC-BY-NC-SA; LAFAN1 retargeted motions are CC BY-NC-ND. Only BeyondMimic (MIT), mjlab/unitree_rl_mjlab (Apache-2.0), ProtoMotions (Apache-2.0), and SONIC (Apache-2.0 + NVIDIA Open Model License weights) are productization-safe.
- DoF mismatch trap: ASAP, PBHC, and VideoMimic results are mostly on the 23-DoF G1 (wrists locked); the user's robot is the 29-DoF EDU. BeyondMimic, unitree_rl_mjlab, SONIC, and unitree_rl_lab support 29-DoF — verify joint ordering/configs when mixing motion files between ecosystems.
- None of these trackers control the Inspire FTP dexterous hands — they are body-only (29 joints). Hand choreography must be a separate synchronized channel via the Inspire SDK, and the added wrist payload of the FTP hands is unmodeled in stock configs: add payload/mass domain randomization or expect degraded tracking; sim2sim re-validate.
- Two competing deployment stacks: HybridRobotics/motion_tracking_controller needs ROS 2 Jazzy (Ubuntu 24.04) + legged_control2, while unitree_rl_mjlab's deploy is plain C++ unitree_sdk2 + CycloneDDS (no ROS). Pick ONE; mixing both on the Jetson invites DDS/domain conflicts with the factory services.
- SONIC caveats: finetuning realistically requires 64+ GPUs (use checkpoints as-is); deployment inference requires the Jetson Orin GPU via TensorRT (engine build on-device takes time; FP16); GEM's video front-end assumes a single person with full body visible — multi-cut dance music videos need manual segment editing (documented third-party pain point); SMPL-X assets require registration under a non-commercial license.
- Isaac version fragmentation: whole_body_tracking is pinned to Isaac Lab 2.1.0/Isaac Sim 4.5.0, unitree_rl_lab to 2.3.0/5.1.0, GR00T-WBC to 2.3.2. Keep per-repo conda envs/containers; do not share one Isaac install.
- unitree_rl_mjlab is young (no tagged releases, 30 open issues): pin a known-good commit, and keep HybridRobotics/whole_body_tracking + mjlab upstream as reference implementations when its tracking task misbehaves.
- Safety: whole-body tracking policies bypass Unitree's factory balance/damping stack. Always gate every new policy through unitree_mujoco/MuJoCo sim2sim, start suspended on the gantry in zero-torque → debug mode (L2+R2), and keep the wireless e-stop in hand; repos explicitly disclaim hardware damage.
- Floor-contact choreography (kneeling, floorwork, get-ups) is where per-motion trackers shine and universal trackers degrade; BeyondMimic optionally uses LiDAR-inertial odometry for contact-rich motions — plan for the G1's LiDAR if dances include ground work.

### Open questions
- Exact wall-clock/VRAM to converge one dance policy in mjlab/unitree_rl_mjlab on a single rented GPU (4090 vs L40S vs A100) — undocumented; needs a benchmark run on dance1_subject2 before committing to a cloud budget.
- Does the pretrained SONIC checkpoint track full choreography (fast footwork, spins, floor contact) with acceptable fidelity on the 29-DoF G1 with Inspire hands attached, or will most dances still require per-motion training? (99.2% success over 123 motions is reported, but dance-specific MPJPE is not broken out.)
- Can HybridRobotics/motion_tracking_controller (ROS 2 Jazzy) run directly on the G1's Jetson Orin PC2, or is the simpler unitree_rl_mjlab g1_ctrl the only practical onboard option? (BeyondMimic inference is <1 ms on CPU so compute is fine; the question is ROS 2 Jazzy on Jetson L4T.)
- How much does the Inspire FTP hand mass/inertia shift degrade policies trained on stock G1 models, and is adding payload randomization in the mjlab robot cfg sufficient?
- Quality gap between GVHMR and NVIDIA GEM/GEM-X for dance videos (multi-person, camera cuts, partial body) — which video→SMPL front-end yields fewer retargeting artifacts per unit of manual cleanup?
- unitree_rl_mjlab has no tagged releases — is Unitree treating it as the successor to unitree_rl_lab for tracking tasks, and how stable is its API across commits?

## deployment

### Summary
As of mid-2026 the de-facto open stack for deploying robust RL whole-body motion tracking on a Unitree G1 is BeyondMimic (UC Berkeley Hybrid Robotics + Stanford TML, arXiv 2508.08241). It is split into two repos: training in https://github.com/HybridRobotics/whole_body_tracking (Isaac Lab v2.1.0, one policy per motion clip, motions managed through a WandB registry via csv_to_npz.py, MIT, ~2.1k stars) and deployment in https://github.com/HybridRobotics/motion_tracking_controller (MIT, ~500 stars, 47 commits, active issues through 2026). The deployment stack is C++ on ROS 2 Jazzy using qiayuanl's legged_control2/ros2_control framework plus https://github.com/qiayuanl/unitree_bringup (hardware interface speaking plain Unitree DDS to the MCU — ROS 2 is only the process/controller framework, the robot needs no ROS). Policies are ONNX with joint order and PD impedance embedded in ONNX metadata; inference is ONNX Runtime CPU (<1.0 ms/step per the paper). config/g1/controllers.yaml runs the controller_manager at 500 Hz with the MotionTrackingController policy at 50 Hz over the FULL 29-joint body (wrists + 3-DoF waist included), with a 500 Hz contact-based state estimator (generalized momentum observer + Kalman filter, no mocap; optional Mid-360 LIO). It can run either on an external PC over ethernet (set static IP 192.168.123.11, `ros2 launch motion_tracking_controller real.launch.py network_interface:=<iface> policy_path:=<x>.onnx`) or onboard: the companion Docker image qiayuanl/unitree:jazzy is multi-arch amd64+arm64 (last pushed 2026-04-29), so it runs on the G1's Jetson Orin PC2 with `--network host --privileged` — the BeyondMimic paper ran fully onboard. Joystick protocol: L1+A standby (PD to default pose), R1+A start tracking policy, B = damping e-stop. Built-in MuJoCo sim2sim via mujoco.launch.py uses the identical controller binary.

Unitree's official path for custom policies confirms the same low-level contract. unitree_rl_gym (BSD-3, ~3.3k stars) ships deploy/deploy_real/deploy_real.py + configs/g1.yaml (unitree_sdk2py, `hg` message family, topics rt/lowcmd + rt/lowstate, 50 Hz policy dt=0.02 — but it only learns 12 leg joints, arms/waist are PD-held, so it is a reference, not the dance controller). The newer unitree_rl_lab (Apache-2.0, ~1.1k stars, Isaac Lab 2.3.0) explicitly supports G1-29dof with a C++ deploy at deploy/robots/g1_29dof, and the brand-new unitree_rl_mjlab (Apache-2.0, MuJoCo-Warp "mjlab") even cites whole_body_tracking for its motion-imitation task — Unitree itself is converging on the BeyondMimic recipe. The hardware procedure for any custom controller: hoist the robot by the shoulder suspension buckles, power on, wait for zero-torque (or L2+Y), then L2+R2 enters debug/develop mode which kills the built-in ai_sport motion service (otherwise it keeps publishing zero-velocity lowcmds that conflict with yours and cause jitter). Programmatically the same is done with MotionSwitcherClient (msc) CheckMode()/ReleaseMode() as in unitree_sdk2_python example/g1/low_level/g1_low_level_example.py, which also shows the required CRC on every LowCmd, the mode_machine field that must echo LowState, mode_pr (PR=0 serial vs AB=1 parallel ankle/waist control), and G1_NUM_MOTOR=29 with the canonical index map (0-11 legs, 12-14 waist yaw/roll/pitch, 15-21 left arm, 22-28 right arm).

The 29-DoF + Inspire FTP hands question resolves cleanly: hands are NOT on rt/lowcmd. They live on a separate DDS<->RS485 bridge service on PC2 — DFX hands use rt/inspire/cmd / rt/inspire/state (unitree_go MotorCmds_, 12 motors; repo unitreerobotics/dfx_inspire_service), while the FTP (tactile) hands use rt/inspire_hand/ctrl/*, rt/inspire_hand/state/* and rt/inspire_hand/touch/* with inspire::inspire_hand_ctrl/state/touch IDLs. So the body joint index map is untouched and the RL policy stays 29-DoF; finger choreography is a separate low-rate publisher. The real coupling is dynamics: each hand adds ~0.5 kg at the wrist, and there is an open Unitree issue (unitree_sdk2_python #129) of shoulder-pitch overheat on a G1 with FTP hands — the training model must include hand mass (or payload randomization) or sim2real will degrade. For sim2sim, unitree_mujoco (BSD-3, ~1k stars, C++ recommended) exposes the exact rt/lowcmd / rt/lowstate topics on DDS domain 1 / interface "lo" (real robot is domain 0), so the unitree-sdk2-based deploy binary runs unmodified against MuJoCo; the BeyondMimic stack has its own equivalent mujoco.launch.py gate. Safety-wise there are no credible SDK-induced bricking reports — debug mode is official and reverts on reboot; the notable 2025 risk was the BLE WiFi command-injection worm (CVE-2025-35027), patched ~Sep 2025, so update firmware once and then freeze it (button mappings changed between fw 1.0.2 and 1.0.4). Critically, once in debug mode the factory remote behaviors are gone, so the only software e-stop is the damping fallback your controller implements (B in motion_tracking_controller, select/Ctrl+C in unitree_rl_gym) — keep the gantry as the true e-stop for all first runs, and never run recovery/lying modes with dexterous hands installed (manual warning).

### Recommendation
Adopt the BeyondMimic deployment stack unchanged: train per-dance tracking policies in HybridRobotics/whole_body_tracking on a cloud GPU (Isaac Lab v2.1.0), export ONNX (joint order + impedance in metadata), verify with motion_tracking_controller's mujoco.launch.py (and optionally unitree_mujoco for the SDK-level contract), then deploy with HybridRobotics/motion_tracking_controller + qiayuanl/unitree_bringup running ONBOARD the G1's Jetson Orin PC2 inside the multi-arch qiayuanl/unitree:jazzy Docker image (--network host --privileged, network_interface:=eth0), giving a tether-free 500 Hz controller / 50 Hz ONNX-CPU policy loop with built-in standby (L1+A), policy start (R1+A) and damping e-stop (B). Justification: it is the only verified, actively maintained (2026) open stack that already tracks all 29 joints of the G1 EDU including wrists and waist with a push-robust RL controller and proven hardware results on dynamic motions (dance is its core demo class, LAFAN1 dance clips train "without tuning any parameters"); the operator laptop (Ubuntu 22.04, no NVIDIA GPU) is then only a launch/SSH console, sidestepping both the ROS 2 Jazzy/Ubuntu 24.04 requirement and the no-CUDA constraint, while Unitree's own unitree_rl_lab/unitree_rl_mjlab confirm the same rt/lowcmd 29-motor DDS contract as fallback. Only modification needed: add Inspire FTP hand mass to the training asset and drive fingers via the separate rt/inspire_hand/ctrl service.

### Repos
- **whole_body_tracking (BeyondMimic training)** — https://github.com/HybridRobotics/whole_body_tracking
  - why: Trains the per-motion RL whole-body tracking policy (DeepMimic-style, push-robust) for G1; the dance video pipeline's retargeted motion (CSV->NPZ via scripts/csv_to_npz.py) feeds this; exports ONNX consumed directly by the deployment repo. Claims any sim2real-ready LAFAN1 motion (incl. dance1_subject1 etc.) trains without parameter tuning.
  - reqs: Isaac Lab v2.1.0 + Isaac Sim (NVIDIA GPU required -> cloud GPU box), Python 3.10, WandB account mandatory (motion registry named 'Motions')
  - license: MIT
  - maturity: ~2.1k stars, 296 forks, 181 commits, active 2025-2026; reference implementation of arXiv 2508.08241 (CoRL 2025)
- **motion_tracking_controller (BeyondMimic sim2real)** — https://github.com/HybridRobotics/motion_tracking_controller
  - why: THE deployment code path: C++ ROS 2 controller (legged_control2/ros2_control) loading the ONNX policy (joint order + kp/kd read from ONNX metadata), controller_manager 500 Hz / policy 50 Hz, 500 Hz momentum-observer+KF state estimation, full 29-joint G1 config in config/g1/controllers.yaml (wrists+waist included). real.launch.py (network_interface:=<iface>, policy_path:=<x>.onnx or wandb_path) and mujoco.launch.py for sim2sim. Joystick: L1+A standby, R1+A policy, B damping e-stop.
  - reqs: ROS 2 Jazzy (Ubuntu 24.04) — on the 22.04 laptop or Jetson PC2 use Docker; ONNX Runtime CPU (no GPU needed, <1 ms/inference); colcon build --packages-up-to motion_tracking_controller
  - license: MIT
  - maturity: ~508 stars, 53 forks, 47 commits, 10 open issues (watch #28 sim discrepancy, #23 state_estimator joint-name error, #27 passive joints); active 2026
- **unitree_bringup (qiayuanl)** — https://github.com/qiayuanl/unitree_bringup
  - why: Companion hardware-interface/bringup for motion_tracking_controller: wraps Unitree DDS (rt/lowcmd-rt/lowstate) as a ros2_control hardware interface, ships Docker image qiayuanl/unitree:jazzy that is multi-arch amd64+arm64 (pushed 2026-04-29) — this is what lets the whole stack run onboard the G1's Jetson Orin PC2 (--network host --privileged, network_interface:='eth0'), incl. auto-restart service containers.
  - reqs: Docker, host networking, ethernet to robot MCU; external-PC alternative: static IP 192.168.123.11 on the robot LAN
  - license: not stated in README (verify before redistribution)
  - maturity: 19 stars, 23 commits — small but maintained by the BeyondMimic first author (Qiayuan Liao); known issue: G1 LIO auto-start has unresolved timestamp sync (LIO non-functional), contact+IMU estimator unaffected
- **unitree_rl_gym** — https://github.com/unitreerobotics/unitree_rl_gym
  - why: Unitree's official minimal custom-policy deployment reference: deploy/deploy_real/deploy_real.py + deploy/deploy_real/configs/g1.yaml (unitree_sdk2py, msg type 'hg', rt/lowcmd-rt/lowstate, control_dt 0.02=50 Hz, CRC, full 29-motor index map with legs 0-11 / waist 12-14 / arms 15-28) and deploy/deploy_mujoco/deploy_mujoco.py for sim2sim. Documents the canonical hardware procedure: hoist -> zero torque -> L2+R2 debug mode -> run script -> start (default pose) -> A (policy) -> select (damping). Its policy is legs-only (12 actions), so use it as protocol reference, not as the dance controller.
  - reqs: Python + unitree_sdk2_python + CycloneDDS (already working on the laptop); C++ variant in deploy/deploy_real/cpp_g1
  - license: BSD-3-Clause
  - maturity: ~3.3k stars, 557 forks; stable/low-churn (42 commits); README itself warns it 'is not a stable control program... demonstration purposes'
- **unitree_sdk2_python (g1 low-level example)** — https://github.com/unitreerobotics/unitree_sdk2_python
  - why: example/g1/low_level/g1_low_level_example.py is the ground truth for the raw joint command interface: unitree_hg LowCmd_/LowState_ on rt/lowcmd-rt/lowstate, G1_NUM_MOTOR=29, 500 Hz (control_dt 0.002), CRC required, mode_machine echoed from LowState, mode_pr PR=0/AB=1, and MotionSwitcherClient (msc service) CheckMode()/ReleaseMode() to programmatically kill the built-in motion service before low-level control. Issue #129 documents shoulder-motor overheat with Inspire FTP hands (mass mismatch gotcha).
  - reqs: Python 3.8+, CycloneDDS — already installed and working on the laptop per context
  - license: BSD-3-Clause
  - maturity: official Unitree SDK, actively maintained 2024-2026
- **unitree_mujoco** — https://github.com/unitreerobotics/unitree_mujoco
  - why: Sim2sim gate: MuJoCo simulator exposing the identical unitree_sdk2 DDS topics (rt/lowcmd, rt/lowstate) so the exact deploy binary/script runs against simulation before hardware. Sim uses DDS domain id 1 on interface 'lo'; real robot is domain 0 — a one-line config switch. G1 uses the unitree_hg message family. C++ version (unitree_sdk2) recommended; Python version (unitree_sdk2_python) also provided.
  - reqs: MuJoCo >= 3.x, CPU only — runs fine on the no-GPU laptop
  - license: BSD-3-Clause
  - maturity: ~1.0k stars, 351 forks, 55 commits, maintained
- **unitree_rl_lab** — https://github.com/unitreerobotics/unitree_rl_lab
  - why: Unitree's official Isaac Lab 2.3.0 RL repo with explicit G1-29dof support and a C++ onboard deploy at deploy/robots/g1_29dof (+ unitree_mujoco sim2sim); credits whole_body_tracking. Fallback deployment path if you want to avoid ROS 2 entirely (plain unitree_sdk2 C++).
  - reqs: Isaac Lab 2.3.0 (NVIDIA GPU for training, cloud); deploy binary is CPU C++ built with CMake against unitree_sdk2
  - license: Apache-2.0
  - maturity: ~1.1k stars, 263 forks, 77 commits, active (33 open issues; see #44 sim2real velocity-mode unresponsiveness)
- **unitree_rl_mjlab** — https://github.com/unitreerobotics/unitree_rl_mjlab
  - why: Unitree's newest (2026) RL repo on 'mjlab' (Isaac-Lab-style API over MuJoCo-Warp) with a motion-imitation task that explicitly defers to BeyondMimic/whole_body_tracking docs for motion preprocessing; ONNX export into deploy/robots/g1/config/policy/.../exported + CMake C++ deploy over ethernet. Worth tracking as it may become the simplest official video->policy->robot path.
  - reqs: NVIDIA GPU for training (multi-GPU --gpu-ids supported); CPU C++ deploy
  - license: Apache-2.0
  - maturity: 439 stars, 119 forks, 20 commits — very new, less battle-tested than the others
- **dfx_inspire_service (Inspire hand bridge)** — https://github.com/unitreerobotics/dfx_inspire_service
  - why: Shows the hand control architecture: a PC2-resident service bridging DDS to the hands' RS485/USB. DFX topics: rt/inspire/cmd + rt/inspire/state (unitree_go MotorCmds_, 12 motors both hands, position-only). The user's FTP (tactile) hands use the newer topics rt/inspire_hand/ctrl/*, rt/inspire_hand/state/*, rt/inspire_hand/touch/* (inspire::inspire_hand_ctrl/state/touch IDLs) — see Unitree doc page G1_developer/inspire_dfx_dexterous_hand and the FTP page. Key fact: hands are completely outside rt/lowcmd, so the 29-DoF body joint map is unaffected.
  - reqs: runs on the Jetson PC2 (usually preinstalled on EDU units with hands); USB-RS485
  - license: BSD-3-Clause
  - maturity: official Unitree repo, low churn (driver-level, stable)

### Gotchas
- motion_tracking_controller requires ROS 2 Jazzy = Ubuntu 24.04, but the laptop is 22.04 and the Jetson PC2 runs an older JetPack Ubuntu — you MUST use the qiayuanl/unitree:jazzy Docker image (verified multi-arch amd64+arm64, pushed 2026-04-29) with --network host --privileged; do not try to apt-install Jazzy on either machine.
- Never publish to rt/lowcmd while the factory motion service is running: hoist the robot, reach zero-torque (L2+Y), then L2+R2 to enter debug/develop mode (kills ai_sport), or call MotionSwitcherClient.ReleaseMode() programmatically — otherwise the built-in controller's zero-velocity commands conflict with yours and the robot jitters violently (unitree_sdk2_python issue #43).
- Once in debug mode the factory remote e-stop/damping behaviors are GONE; the only software fallback is what your controller implements (B = damping in motion_tracking_controller, select/Ctrl+C in unitree_rl_gym). Treat the gantry as the real e-stop; first runs fully suspended by the shoulder suspension buckles, then feet barely touching, then slack rope.
- Every LowCmd must carry a valid CRC and a mode_machine value echoed from LowState, and mode_pr must match your joint convention (PR=0 serial pitch/roll vs AB=1 parallel) — wrong values are silently ignored or move the wrong ankle/waist DOFs.
- Inspire FTP hands are not on rt/lowcmd (body stays 29 motors, indices 0-28: legs 0-11, waist 12-14, left arm 15-21, right arm 22-28), but they add ~0.5 kg per wrist: train with hand mass in the G1 asset or add payload randomization, and monitor shoulder/wrist motor temperatures — there is an open report of shoulder-pitch overheat with FTP hands (unitree_sdk2_python issue #129). FTP topics differ from DFX (rt/inspire_hand/ctrl|state|touch vs rt/inspire/cmd|state) — most published examples target DFX.
- BeyondMimic trains ONE policy PER motion clip — every new dance = a fresh cloud-GPU Isaac Lab training run (hours) + ONNX export; budget this in the pipeline UX. whole_body_tracking also hard-depends on WandB (registry collection named 'Motions'), though the deployed controller can load a plain local .onnx via policy_path.
- unitree_mujoco sim runs on DDS domain id 1 / interface 'lo' while the real robot is domain 0 — forgetting to switch this is the classic 'works in sim2sim, robot silent' failure; motion_tracking_controller issue #23 (unknown joint name: state_estimator) and #28 (sim-vs-sim policy discrepancy) are the top deployment-time issues to read first.
- Firmware: no credible reports of bricking from low-level SDK use (debug mode reverts on reboot), but update once to pick up the Sep-2025 BLE WiFi command-injection patches (CVE-2025-35027, worm-capable) and then FREEZE the firmware version mid-project — remote button mappings and behavior changed between fw 1.0.2 and 1.0.4 and OTA can break a validated deployment.
- Do not run the factory get-up/lying/squat recovery modes with dexterous hands installed (explicit manual warning — wrist/hand collision with ground); after a damping e-stop mid-dance, hoist before recovering.
- unitree_bringup's G1 LIO (Mid-360) auto-start is documented non-functional (timestamp sync); the default contact+IMU estimator is fine for dance balance but global XY position will drift over long choreographies — anchor the motion in robot-relative frame, not world frame.

### Open questions
- Onboard (Jetson Orin PC2, arm64 Docker, tether-free — what BeyondMimic's paper did) vs external (laptop amd64 Docker at 192.168.123.11 over wired DDS) execution of motion_tracking_controller: both are supported; need a latency/jitter measurement on the Jetson under load (it also runs the FTP hand service and any video/teleop services) before committing.
- Exact firmware version and variant flags on this specific G1 EDU Ultimate (mode_machine value, all 29 joints unlocked incl. waist roll/pitch — some units ship with waist fasteners installed): verify with g1_low_level_example.py reading LowState before any training-asset decisions.
- Confirm the Inspire FTP service is preinstalled and running on PC2 and its exact topic names on current firmware (the support.unitree.com FTP page is JS-rendered and could not be fetched; verified topic scheme rt/inspire_hand/ctrl|state|touch comes from secondary sources incl. NaCl-1374/inspire_hand_ws hand_ftp.md and Unitree DDS interface docs).
- Per-dance policy training (BeyondMimic, hours of cloud GPU per choreography) vs a single pretrained universal tracker (GMT, TWIST2, unitree_rl_mjlab imitation task): if pipeline turnaround per video must be minutes rather than hours, a universal tracker fine-tuned once may be the better architecture — needs a robustness comparison on dance-style motions with push disturbances.
- Whether the BeyondMimic 29-joint policy's wrist tracking remains stable with the heavier FTP hands without retraining (their hardware G1 used Unitree's standard end-effectors in most demos) — plan one sim2sim ablation with corrected wrist mass before first hardware run.

## gpu-strategy

### Summary
As of June 2026 the cloud GPU choice for this pipeline is dominated by one hard constraint: Isaac Sim officially requires GPUs with RT cores. A100 and H100 are explicitly NOT supported (no RT cores, no NVENC; NVIDIA forums confirm no workaround), so the 'big iron' clouds are the wrong tool. The right targets are RTX 4090, L40S, RTX 6000 Ada, or RTX PRO 6000 — exactly the inventory of the marketplace clouds. Verified mid-2026 on-demand prices: RunPod RTX 4090 ~$0.34/hr (Community Cloud) to $0.69/hr (Secure Cloud), RTX 6000 Ada $0.77/hr, L40S $0.86/hr, A100 PCIe $1.39/hr (irrelevant given the RT-core rule); Vast.ai RTX 4090 $0.29-0.50/hr (interruptible vs on-demand, marketplace-priced); Lambda has no consumer GPUs (lineup is B200/GH200/H100/A100/A10/A6000/RTX 6000, from $0.69/hr) and is A100/H100-centric, making it a poor Isaac Sim fit; Paperspace is being sunset into DigitalOcean 'Gradient GPU Droplets' and gates high-end GPUs behind a $39/mo Growth subscription — skip it.

Isaac Lab ships as an official prebuilt headless NGC container: nvcr.io/nvidia/isaac-lab:<major.minor> (latest 2.3.2; BeyondMimic/whole_body_tracking pins Isaac Lab v2.1.0, so use nvcr.io/nvidia/isaac-lab:2.1.0). Access requires only a free NVIDIA Developer Program account + NGC API key (docker login nvcr.io, user '$oauthtoken'); Isaac Sim 5.x is now open source (Apache-2.0) and Isaac Lab is BSD-3-Clause, so there is no per-seat licensing problem for cloud training. The minimal NGC image is headless-only (no X11/GUI) — which is exactly what `train.py --headless` needs. Both RunPod templates and Vast.ai templates support private-registry docker images with login credentials, so the clean pattern is: bake a private image FROM nvcr.io/nvidia/isaac-lab:2.1.0 with whole_body_tracking installed, push to a private Docker Hub/GHCR repo, and launch it programmatically.

Automation: RunPod has the strongest programmatic story — official Python SDK (`pip install runpod`, runpod.create_pod()/terminate_pod()), a new REST API (rest.runpod.io), GraphQL API, runpodctl CLI, per-second billing, and serverless GPU endpoints (RTX 4090 flex workers ~$1.10/hr-equivalent billed per second) which are ideal for the short GVHMR jobs dispatched from a local web UI. Vast.ai is a close second (`pip install vastai`; `vastai search offers 'gpu_name=RTX_4090 ...'`, `vastai create instance OFFER_ID --image ... --disk 60 --onstart-cmd ... --ssh`, official repo vast-ai/vast-python, MIT) and is the cheapest per 4090-hour, but host reliability varies and storage is host-local. Persistent storage: RunPod network volumes are $0.07/GB/mo (<1TB) but only attach in Secure Cloud datacenters — Community Cloud pods get no network volumes. In practice you barely need cloud-persistent storage: BeyondMimic's workflow is already W&B-centric (motions uploaded to a W&B registry, checkpoints/logs to W&B), so pods can be fully ephemeral with W&B as the artifact bus between laptop, cloud, and Jetson.

Cost per artifact: the BeyondMimic paper reports convergence in 1.5k-10k PPO iterations depending on motion difficulty (4096 envs, RSL-RL); on a single RTX 4090 that is roughly 2-10 wall-clock hours, i.e. ~$1-5 per dance on Vast/Community-Cloud 4090s, ~$2-7 on RunPod Secure (4090/RTX 6000 Ada), worst case <$10 on L40S. GVHMR inference is trivial by comparison (core network: 0.28 s for a 1430-frame video on a 4090; whole pipeline with YOLO+ViTPose preprocessing: a few minutes) — well under $0.10/video, so run it in the same pod before training or as a RunPod serverless endpoint. The robot's Jetson Orin (PC2, Orin NX 16GB on the G1 EDU) cannot run Isaac Sim/Lab (x86_64 + RTX dGPU required; ARM/Tegra containers exist only for Isaac ROS-style inference) and should be reserved for ONNX policy inference; GVHMR on it is technically possible but painful (ARM wheels for pytorch3d etc.) and competes with the control stack. Google Colab can run Isaac Sim only via unofficial hacks (j3soon/isaac-sim-colab), has no docker, no job-dispatch API, and disconnects mid-run — fine as a manual GVHMR fallback notebook, unusable as a pipeline backend.

### Recommendation
Use RunPod as the primary backend, dispatched from the local web UI via the official `runpod` Python SDK: (1) bake one private docker image FROM nvcr.io/nvidia/isaac-lab:2.1.0 (NGC, free dev account) containing HybridRobotics/whole_body_tracking + GVHMR, pushed to a private GHCR/Docker Hub repo with registry creds stored in a RunPod template; (2) per dance: orchestrator uploads video-derived motion .npz to the W&B registry, calls runpod.create_pod() on a Secure Cloud RTX 6000 Ada ($0.77/hr) or RTX 4090, the pod's start command runs GVHMR (if not done separately) then `train.py --task=Tracking-Flat-G1-v0 --headless --logger wandb`, pushes the checkpoint/ONNX to W&B, and self-terminates; orchestrator polls W&B and pulls the .onnx for sim2sim then Jetson deployment. Justification: RunPod is the only provider combining RT-core GPUs (mandatory for Isaac Sim — A100/H100 are unsupported), a first-class Python SDK/REST API for job dispatch, per-second billing, and optional serverless endpoints for the pose-estimation step; expected cost ~$2-7 per trained dance policy. Keep Vast.ai (vastai CLI, 4090s from ~$0.29/hr) as a drop-in cost fallback for long runs, and keep everything stateless via W&B so no provider lock-in or network volume is required.

### Repos
- **Isaac Lab (+ official NGC headless container)** — https://github.com/isaac-sim/IsaacLab
  - why: The RL training framework. NVIDIA publishes a prebuilt headless-only container per major release: nvcr.io/nvidia/isaac-lab:2.3.2 (latest) / nvcr.io/nvidia/isaac-lab:2.1.0 (version BeyondMimic pins). Pull from NGC (catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-lab) with `docker login nvcr.io` user `$oauthtoken` + NGC API key. This is the base image for the cloud training pod.
  - reqs: x86_64 Linux, NVIDIA driver + Container Toolkit, Docker >=26, RT-core GPU mandatory (RTX 3070 8GB min, 16GB+ recommended; A100/H100 NOT supported)
  - license: BSD-3-Clause (Isaac Lab); Isaac Sim 5.x is Apache-2.0 open source; NGC container under NVIDIA Software License (free with NVIDIA Developer Program)
  - maturity: Official NVIDIA project, very active through 2026; docs explicitly cover docker/cluster/cloud deployment; minimal NGC image is headless-only (no X11/GUI)
- **runpod-python (RunPod SDK)** — https://github.com/runpod/runpod-python
  - why: Primary automation path for the local web UI: runpod.create_pod(name, image, gpu_type), stop/terminate, serverless endpoint client. Backed by REST API (rest.runpod.io), GraphQL (graphql-spec.runpod.io) and runpodctl CLI. Per-second billing; network volumes $0.07/GB/mo (Secure Cloud only); SOC2 Type II since Oct 2025.
  - reqs: Python 3, RUNPOD_API_KEY; verified mid-2026 prices: RTX 4090 $0.34/hr community / $0.69/hr secure, RTX 6000 Ada $0.77/hr, L40S $0.86/hr, A100 PCIe $1.39/hr; serverless 4090 flex ~$1.10/hr-equivalent per-second
  - license: MIT
  - maturity: Official RunPod SDK on PyPI, actively maintained 2026; REST API launched 2025 superseding GraphQL
- **vast-python (Vast.ai CLI/SDK)** — https://github.com/vast-ai/vast-python
  - why: Cost fallback: cheapest RTX 4090s ($0.29-0.50/hr marketplace). `pip install vastai; vastai set api-key ...; vastai search offers 'gpu_name=RTX_4090 verified=true' -o 'dlperf_usd-'; vastai create instance OFFER_ID --image <private image> --disk 60 --onstart-cmd ... --ssh --direct`. Supports private registries (nvcr.io login) and onstart provisioning scripts.
  - reqs: Python 3, Vast API key; storage is host-local (no cross-host volumes), so pair with W&B artifacts
  - license: MIT
  - maturity: Official Vast.ai package, 196 stars, 879 commits, v0.4.0 Sep 2025, active; marketplace hosts vary in reliability
- **whole_body_tracking (BeyondMimic)** — https://github.com/HybridRobotics/whole_body_tracking
  - why: The training workload being dispatched. Pins Isaac Lab v2.1.0; W&B registry for motion .npz in, W&B for checkpoints out — this W&B-centric design is what makes ephemeral stateless cloud pods viable. Train cmd: `python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 --registry_name {org}/wandb-registry-motions/{motion} --headless --logger wandb`. Paper: convergence in 1.5k-10k iterations => ~2-10 h on one 4090 => ~$2-7/motion.
  - reqs: Isaac Lab 2.1.0 (Isaac Sim 4.5), single RT-core GPU (4090-class, 24GB ample), W&B account
  - license: MIT
  - maturity: 2.1k stars, 296 forks, 25 open issues, active 2025-2026; real-robot-validated on Unitree G1
- **GVHMR** — https://github.com/zju3dv/GVHMR
  - why: Workload A: world-grounded SMPL motion from monocular video. Light GPU footprint (core net 0.28 s per 1430-frame video on a 4090; full pipeline with YOLO+ViTPose a few minutes) — runs in the same cloud pod before training or as a RunPod serverless endpoint; <$0.10/video.
  - reqs: CUDA GPU, PyTorch; ~6GB VRAM class; x86 wheels (pytorch3d) make Jetson/ARM installs painful
  - license: Custom ZJU academic license: research/non-profit only, commercial use requires written permission (xwzhou@zju.edu.cn)
  - maturity: 1.6k stars, TPAMI 2026 version, active (Mar 2025: DPVO replaced by SimpleVO, simplifying installation)
- **isaac-sim-colab (unofficial)** — https://github.com/j3soon/isaac-sim-colab
  - why: Evidence for the Colab verdict: headless Isaac Sim on Colab is possible only via hacks; no docker, sessions disconnect, no dispatch API => not usable as an automated training backend. Colab remains at most a manual GVHMR fallback.
  - reqs: Colab GPU runtime; free-tier T4 has RT cores but 12-24h session caps kill multi-hour RL runs
  - license: MIT
  - maturity: Community project, demo-grade (Cartpole/Ant); NVIDIA has no official Colab support (IsaacLab issue #2387 still a proposal)

### Gotchas
- Isaac Sim officially requires RT-core GPUs: A100 and H100 are NOT supported (no RT cores/NVENC, no workaround per NVIDIA forums). Do not buy A100/H100 hours for training; use RTX 4090 / RTX 6000 Ada / L40S / RTX PRO 6000. This also rules out Lambda's main inventory and most hyperscaler GPU SKUs.
- BeyondMimic pins Isaac Lab v2.1.0 — use nvcr.io/nvidia/isaac-lab:2.1.0, not the latest 2.3.2; NGC only publishes images for major.minor releases.
- NGC containers need a free NVIDIA Developer Program account + NGC API key (`docker login nvcr.io`, username literally `$oauthtoken`). Bake your derived image into a PRIVATE registry (GHCR/Docker Hub) — public redistribution of NGC-derived images is a license gray area — and store the registry creds in the RunPod/Vast template.
- RunPod network volumes ($0.07/GB/mo) attach only in Secure Cloud datacenters; Community Cloud pods (the cheap $0.34/hr 4090s) have no network volumes. Design pods stateless with W&B as the artifact store instead.
- Vast.ai cheap rates are 'interruptible' (preemptible) — a multi-hour RL run can be killed; pay the on-demand premium or rely on W&B checkpoint resume (whole_body_tracking supports rsl_rl resume).
- RunPod marketplace pricing drifts (search results from May 2026 showed community 4090 at $0.34/hr vs $0.69/hr on the live pricing page); have the orchestrator query available GPU types/prices via the API at dispatch time rather than hardcoding a GPU type.
- GVHMR's license is research/non-commercial only (custom ZJU license; commercial use needs emailed permission). Fine for a personal/research robot project, a blocker if this ever becomes a product.
- The G1's Jetson Orin (PC2) cannot run Isaac Sim/Lab at all (x86_64-only for simulation; ARM/Tegra images exist only for Isaac ROS inference) and 4096-env PPO is orders of magnitude beyond it; keep it for ONNX policy inference at deploy time. Running GVHMR on it is technically possible but fights ARM wheel availability and steals cycles from the realtime control loop.
- Google Colab cannot back an automated pipeline: no docker (so no NGC container), no job-dispatch API, sessions disconnect after hours; Isaac Sim runs there only via unofficial hacks. At most a manual GVHMR notebook fallback.
- Paperspace is being absorbed into DigitalOcean Gradient GPU Droplets and gates A100/H100-class GPUs behind a $39/mo Growth subscription — poor fit and shaky product continuity; avoid.
- Training pods should self-terminate in their start command (train && push artifacts && runpodctl/API terminate) — orphaned pods at $0.69-0.86/hr are the main real cost risk, not the training itself (~$2-7/motion).

### Open questions
- Exact wall-clock hours per dance motion with whole_body_tracking's default config (4096 envs) on a 4090 vs L40S — the paper gives 1.5k-10k iterations to convergence but no its/sec; benchmark once on a $0.69/hr pod before budgeting a multi-dance library.
- Current RunPod Secure Cloud availability of RTX 4090 (some reports say it is no longer publicly listed there) — if gone, default to RTX 6000 Ada $0.77/hr or L40S $0.86/hr in Secure Cloud, or 4090s in Community Cloud/Vast.
- Whether headless PhysX-only Isaac Lab training actually fails on A100/H100 in practice (some users report it runs despite being unsupported) — only worth testing if a free A100 grant appears; otherwise moot given 4090s are cheaper anyway.
- W&B free tier limits (registry artifacts + run storage) for a growing library of motions/checkpoints — may need a paid W&B plan or a swap to S3-compatible storage (e.g., Cloudflare R2) as the artifact bus.
- Inspire FTP hand/finger choreography is outside BeyondMimic's 29-DoF body tracking — hand motion retargeting (e.g., from GVHMR/MANO hand poses) needs a separate, likely open-loop, channel and was not part of this cloud-GPU scoping.

## turnkey-alternatives

### Summary
As of June 2026 there IS a near-turnkey "video -> G1 dance" path, and it is new (Feb 2026): NVIDIA's GR00T-WholeBodyControl stack. GEAR-SONIC (arXiv 2511.07820, released 2026-02-19 in github.com/NVlabs/GR00T-WholeBodyControl) is a pretrained universal motion-tracking "behavior foundation model" for the Unitree G1, trained on Bones-SEED (142K+ motions, ~288h, already retargeted to G1). It ships ONNX checkpoints on HuggingFace (nvidia/GEAR-SONIC: model_encoder.onnx, model_decoder.onnx, planner_sonic.onnx), a C++ TensorRT inference stack (gear_sonic_deploy) that explicitly supports Jetson hardware (docs include a "G1 JetPack 6 Flashing Guide"), and a ZMQ motion-reference interface. Crucially, NO per-motion RL training is needed: any kinematic reference the policy is fed gets tracked by one robust RL policy. The video front-end is NVIDIA GEM (NVlabs/GENMO; GEM-SMPL and GEM-X on HF), a monocular-video -> world-grounded SMPL/SOMA motion model that includes hands. A community writeup (note.com/ryosuke_tajima/n/n1341bc889c4e) documents exactly the target use case end-to-end: music video -> GEM (smpl_param.pt) -> ZMQ -> MuJoCo sim2sim -> real G1 dancing, on a single RTX 5060 Ti 16GB.

The proven per-motion alternative is BeyondMimic (UC Berkeley Hybrid Robotics): HybridRobotics/whole_body_tracking (MIT, ~2.1k stars, Isaac Lab 2.1.0 + Isaac Sim 4.5.0) trains one tracking policy per motion "without tuning parameters" directly from Unitree's LAFAN1 retargeted CSVs, and HybridRobotics/motion_tracking_controller (MIT, ~508 stars) deploys it on the real G1 via ROS 2 Jazzy + ONNX CPU inference (no GPU needed at runtime). This has state-of-the-art real-G1 motion quality (jump-spins, cartwheels) and is the de-facto community standard — Unitree's own unitree_rl_mjlab cites its docs. KungfuBot/PBHC (TeleHuman/PBHC, CC BY-NC 4.0) is the most complete *published* video->real pipeline (video -> GVHMR -> physics-based filtering + contact masks -> SMPL->G1 retarget -> IsaacGym RL -> MuJoCo sim2sim) but does not include real-robot deployment code and is built on deprecated IsaacGym/Python 3.8; KungfuBot2 (VMS, arXiv 2509.16638) moves to a single multi-skill policy. HoloMotion (HorizonRobotics, Apache-2.0, v1.3.2 released 2026-06-09, very active) is a second "pretrained generalist tracker" option (0.4B-param MoE transformer, 2000+h data) whose README lists offline dance playback as an explicit use case. TWIST (YanjieZe/TWIST, MIT) is a teleop-oriented general motion tracker that can also consume offline motion files but is IsaacGym-legacy. VideoMimic (hongsukchoi/VideoMimic, MIT, CoRL 2025 Best Student Paper) covers video->sim->real but is aimed at terrain-contextual skills (stairs, sitting) with environment-conditioned policies — wrong tool for choreography fidelity, and activity stalled since Sep 2025.

On Unitree's own assets: their viral dance demos were made by retargeting LAFAN1 mocap to G1 via interaction-mesh + IK optimization and training internal RL tracking policies; there is no public choreography tool/format. What they DID release is the key bootstrap asset: huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset — G1 CSVs at 30fps (root xyz+quat + 29 joint angles), including dance1_subject1/2 etc. These are consumed directly by BeyondMimic's csv_to_npz and by Unitree's official unitree_rl_mjlab tracking example (task Unitree-G1-Tracking-No-State-Estimation, Apache-2.0, includes sim2real C++ deploy). unitree_rl_lab (Isaac Lab) only ships a velocity task — the tracking/dance example lives in unitree_rl_mjlab. The "mimic" startup (ETH spin-off, €13.8M Nov 2025) is dexterous-hand manipulation for factories — irrelevant to dance. The universal glue component across all paths is GMR (YanjieZe/GMR, MIT, ICRA 2026, ~2.3k stars): SMPL-X/AMASS/BVH-LAFAN1/FBX/GVHMR -> G1 29-DoF retargeting in real time on CPU — it runs on the no-GPU laptop.

### Recommendation
Build the pipeline around BeyondMimic as the primary controller and run GEAR-SONIC as a parallel turnkey track. Concretely: (1) Day one, validate the back half with zero video work: pull dance CSVs from unitreerobotics/LAFAN1_Retargeting_Dataset, train a BeyondMimic policy on a rented cloud GPU (HybridRobotics/whole_body_tracking, Isaac Lab 2.1.0/Isaac Sim 4.5.0, WandB registry), sim2sim in MuJoCo, deploy with HybridRobotics/motion_tracking_controller (ONNX CPU — runs on the Jetson Orin PC2 or even the laptop). This is the most battle-tested, MIT-licensed, highest-fidelity per-motion path and matches your robustness requirement (perturbation-trained RL whole-body tracking). (2) Front-end: video -> GVHMR or NVIDIA GEM (cloud GPU) -> GMR retarget to G1-29DoF (CPU, laptop) -> same CSV/NPZ format. (3) In parallel, evaluate GEAR-SONIC (NVlabs/GR00T-WholeBodyControl + nvidia/GEAR-SONIC checkpoints): if its pretrained universal policy tracks your choreography well enough in MuJoCo sim2sim, it eliminates per-dance cloud training entirely (new dance = minutes, not hours) and runs TensorRT on the Jetson — that becomes the product path, with BeyondMimic reserved for motions SONIC can't hit. Justification: BeyondMimic is proven/safe/per-motion-optimal but costs a GPU-training run per dance; SONIC is the only true near-turnkey system but is 4 months old with known SMPL-drift and TensorRT-version sharp edges — so de-risk by building the shared motion format (G1 29-DoF reference trajectory NPZ/CSV) that both controllers consume.

### Repos
- **GR00T-WholeBodyControl / GEAR-SONIC (NVIDIA)** — https://github.com/NVlabs/GR00T-WholeBodyControl
  - why: The near-turnkey answer: pretrained universal motion-tracking foundation policy for Unitree G1 (no per-motion RL training), C++ TensorRT deploy stack (gear_sonic_deploy) for desktop AND Jetson, ZMQ reference-motion interface, MuJoCo sim2sim. Community-documented doing exactly video->G1 dance (note.com/ryosuke_tajima/n/n1341bc889c4e). Checkpoints: huggingface.co/nvidia/GEAR-SONIC (model_encoder.onnx, model_decoder.onnx, planner_sonic.onnx). Trained on Bones-SEED: 142K+ motions/~288h retargeted to G1.
  - reqs: Inference: TensorRT on Jetson (JetPack 6 — docs include G1 Jetson flashing guide) or desktop NVIDIA GPU. Finetuning: Isaac Lab, '64+ GPUs recommended' (not needed for inference-only use). Main branch.
  - license: Code Apache-2.0; weights NVIDIA Open Model License (commercial use permitted with attribution)
  - maturity: 2.3k stars, 318 forks; GEAR-SONIC released 2026-02-19; active (motor-error monitoring + TTS alerts added recently); known issues: SMPL tracking drift, TensorRT version mismatch causes erratic robot behavior
- **GENMO / GEM (NVIDIA video-to-motion front-end)** — https://github.com/NVlabs/GENMO
  - why: Monocular video -> world-grounded full-body human motion (GEM-SMPL outputs SMPL params incl. global trajectory; GEM-X adds hands+face via SOMA body model) — the front-end NVIDIA pairs with SONIC. Models: huggingface.co/nvidia/GEM-X, GEM-SMPL released Mar 2026. Hand output could eventually drive the Inspire hands.
  - reqs: NVIDIA GPU (cloud for your laptop); single full-body person in frame; dynamic cameras supported but static-camera flag exists
  - license: NVIDIA OneWay Noncommercial License — research/non-commercial ONLY
  - maturity: 442 stars; GEM-SMPL released March 2026; active
- **BeyondMimic — whole_body_tracking (training)** — https://github.com/HybridRobotics/whole_body_tracking
  - why: De-facto community standard for robust per-motion G1 tracking: trains any sim-to-real-ready LAFAN1 motion 'without tuning parameters'; SOTA real-G1 dynamic motion quality (jump spins, cartwheels). Direct consumer of Unitree LAFAN1 CSVs via csv_to_npz.py + WandB motion registry. Task: Tracking-Flat-G1-v0.
  - reqs: Isaac Lab v2.1.0 + Isaac Sim 4.5.0, Python 3.10, Ubuntu 20.04+, NVIDIA GPU (cloud rental for training); WandB account for motion registry
  - license: MIT
  - maturity: 2.1k stars, 296 forks, 181 commits, active development; CoRL/arXiv 2508.08241
- **BeyondMimic — motion_tracking_controller (deploy)** — https://github.com/HybridRobotics/motion_tracking_controller
  - why: Real-G1 deployment half of BeyondMimic: C++ ONNX CPU inference, loads policies from WandB URL or local ONNX. No GPU needed at runtime — fits the no-NVIDIA laptop or the Jetson PC2.
  - reqs: ROS 2 Jazzy (targets Ubuntu 24.04 — use Docker on your 22.04 laptop or run on Jetson), legged_control2, unitree ROS packages, ethernet to robot (expects 192.168.123.x network)
  - license: MIT
  - maturity: 508 stars, 53 forks, 10 open issues, low recent commit activity (stable)
- **PBHC / KungfuBot (TeleHuman)** — https://github.com/TeleHuman/PBHC
  - why: Most complete published video->robot pipeline recipe: video -> GVHMR -> physics-based motion filtering + contact-mask correction -> SMPL->G1 retargeting (smpl_retarget tools) -> per-motion IsaacGym RL -> MuJoCo sim2sim. Its motion-processing/filtering stages are worth borrowing even if you don't use its trainer. KungfuBot2 (arXiv 2509.16638, VMS single multi-skill policy) is the follow-up.
  - reqs: IsaacGym Preview (deprecated, Python 3.8) + NVIDIA GPU for training; MuJoCo for sim2sim; NO real-robot deployment code included
  - license: CC BY-NC 4.0 — non-commercial only
  - maturity: 897 stars, 110 forks, 39 commits, 4 open issues, no releases; paper arXiv 2506.12851 (NeurIPS-track via OpenReview)
- **GMR — General Motion Retargeting** — https://github.com/YanjieZe/GMR
  - why: The retargeting glue for any path you choose: SMPL-X/AMASS, BVH (LAFAN1), FBX, GVHMR output -> Unitree G1 29-DoF (explicitly supported), real-time on CPU (35-70 FPS) — runs on your no-GPU laptop. ICRA 2026; retargeter used by TWIST.
  - reqs: Python 3.10, Ubuntu 20.04/22.04, CPU only
  - license: MIT
  - maturity: 2.3k stars, 392 forks, 123 commits, updates through 2026; v0.2.0
- **GVHMR (video human motion recovery)** — https://github.com/zju3dv/GVHMR
  - why: Permissively-available alternative video front-end (world-grounded SMPL via gravity-view coordinates; SIGGRAPH Asia 2024, TPAMI 2026); it is what PBHC uses, and GMR ingests its output directly. Use it if GEM's noncommercial license is a blocker.
  - reqs: NVIDIA GPU (cloud for your setup)
  - license: Check LICENSE file in repo before relying on it (ZJU academic-style license; not verified in this pass)
  - maturity: Established, widely used by PBHC/GMR pipelines; TPAMI 2026 acceptance indicates ongoing maintenance
- **Unitree LAFAN1 Retargeting Dataset** — https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset
  - why: Ready-made test choreography before the video front-end exists: LAFAN1 mocap (incl. dance1_subject1/2 etc.) retargeted to G1 by Unitree via interaction-mesh + IK optimization. G1 format: CSV @30fps, root XYZ+quat (QXQYQZQW) + 29 joint angles. Directly consumed by BeyondMimic and unitree_rl_mjlab. This is also how Unitree's own viral dance demos were sourced — there is no public Unitree choreography tool.
  - reqs: None — CSV download
  - license: Dataset card on HF (LAFAN1 upstream is CC BY-NC-ND-style from Ubisoft — check before commercial use)
  - maturity: Official Unitree release (Jan 2025), heavily used by the ecosystem
- **unitree_rl_mjlab (Unitree official tracking/dance example)** — https://github.com/unitreerobotics/unitree_rl_mjlab
  - why: Unitree's official open dance/motion-imitation example: tasks Unitree-G1-Tracking-No-State-Estimation (+23dof variant), example commands train on dance1_subject2.csv, includes csv_to_npz conversion, ONNX export, sim2sim via unitree_mujoco, and C++ sim2real code in deploy/. MuJoCo-based (no Isaac Sim license/GPU stack needed for sim, though training still wants a GPU).
  - reqs: GPU for training (multi-GPU supported); ethernet deploy to real G1
  - license: Apache-2.0
  - maturity: 439 stars, 119 forks; newer official repo, references BeyondMimic docs for motion preprocessing
- **unitree_rl_lab (Unitree official Isaac Lab repo)** — https://github.com/unitreerobotics/unitree_rl_lab
  - why: Unitree's Isaac Lab RL repo — but README only ships velocity tasks (Unitree-G1-29dof-Velocity) plus sim2sim/sim2real workflow (g1_ctrl --network eth0). No dance/tracking example here; it acknowledges BeyondMimic's whole_body_tracking instead. Useful as official deploy reference, not as the dance pipeline.
  - reqs: Isaac Lab 2.3.0 + Isaac Sim 5.1.0 (NVIDIA GPU), CMake C++ controller build
  - license: Apache-2.0
  - maturity: 1.1k stars, 263 forks, 77 commits
- **HoloMotion (Horizon Robotics)** — https://github.com/HorizonRobotics/HoloMotion
  - why: Second pretrained-generalist-tracker option: reference-conditioned MoE transformer, v1.3 scaled to 0.4B params / 2000+h motion data; ships pretrained motion-tracking + velocity models on HF; README explicitly lists 'offline motion tracking to replay local motion clips for demos such as dance'. Covers retargeting -> training -> sim eval -> real deploy.
  - reqs: GPU for training/custom data; verify G1-29DoF real-deploy path maturity before committing
  - license: Apache-2.0
  - maturity: 550 stars; very active: v1.3.2 released 2026-06-09 (two days ago)
- **TWIST (teleop whole-body, CoRL 2025)** — https://github.com/YanjieZe/TWIST
  - why: General motion-tracking controller for G1 driven by retargeted motion streams (GMR); high-level motion server accepts offline --motion_file pkl, so it can replay choreography, not just live mocap. Fully open since 2025-09-29 incl. checkpoints (twist_general_motion_tracker.pt), teacher-student RL+BC training, sim2sim+sim2real scripts. Prior art for your architecture, but IsaacGym-legacy.
  - reqs: IsaacGym (deprecated), Python 3.8, RTX 4090-class GPU 1-2 days for retraining, Redis for IPC; pretrained checkpoint avoids training
  - license: MIT
  - maturity: 773 stars, 73 forks; full release Sep 2025; TWIST2 successor exists (arXiv 2511.02832)
- **VideoMimic (UC Berkeley, CoRL 2025 Best Student Paper)** — https://github.com/hongsukchoi/VideoMimic
  - why: Full real-to-sim-to-real from monocular video on G1, BUT designed for terrain-contextual skills (stairs, sitting, climbing) with environment-conditioned policies — not choreography-fidelity dance tracking. Heavy reconstruction pipeline. Useful reference, wrong tool for this project.
  - reqs: Large NVIDIA GPU for the real-to-sim reconstruction stack + Isaac-based training; C++ torchscript deploy
  - license: MIT
  - maturity: 798 stars, 65 forks, only 21 commits; last substantive code drops Jul/Sep 2025 — activity stalled
- **OpenHomie (Shanghai AI Lab / InternRobotics)** — https://github.com/InternRobotics/OpenHomie
  - why: Exoskeleton-cockpit teleop for G1: RL lower body (walk/squat under arbitrary upper-body poses) + mapped upper body. Not a motion tracker — cannot reproduce leg choreography from video. Ruled out for this use case; deployment code (HomieDeploy) is still a useful G1+hands deploy reference.
  - reqs: IsaacGym for training; G1 + Dex3 hands deploy code (you have Inspire, adaptation needed)
  - license: Check repo (Apache-2.0 typical for InternRobotics)
  - maturity: Released Feb 2025, maintained under InternRobotics org

### Gotchas
- License minefield on the video front-end: NVIDIA GEM/GENMO is OneWay NONcommercial (research only), PBHC/KungfuBot is CC BY-NC 4.0, GVHMR carries a ZJU academic license (verify), and upstream LAFAN1 mocap is Ubisoft non-commercial. The RL/controller side (BeyondMimic MIT, GEAR-SONIC weights NVIDIA Open Model License w/ commercial+attribution, GMR MIT, Unitree Apache-2.0) is fine. If this ever becomes a product, the human-motion-recovery stage is the licensing chokepoint.
- Your laptop cannot run any of the GPU stages: GEM/GVHMR video inference and all Isaac Lab / IsaacGym RL training must go to a cloud GPU (Isaac Lab wants RTX-class; BeyondMimic pins Isaac Lab 2.1.0 + Isaac Sim 4.5.0 exactly — newer Isaac Sim 5.x breaks it, while unitree_rl_lab wants Isaac Lab 2.3.0/Isaac Sim 5.1.0 — keep separate envs). Runtime inference is fine: BeyondMimic controller is ONNX CPU; SONIC is TensorRT on the Jetson Orin PC2.
- GEAR-SONIC deploy expects JetPack 6 on the G1's Jetson (docs ship a 'G1 JetPack 6 Flashing Guide') — reflashing PC2 is invasive and may break Unitree's stock services; also a documented failure mode where wrong TensorRT version makes the robot 'behave weirdly' (community used TensorRT 10.13.3). Verify versions in sim2sim before touching hardware.
- GEM requires a single, full-body-visible subject in frame — group dance videos or framing cuts need manual pre-editing/cropping; GR00T-WBC docs also list 'SMPL tracking is unstable or drifts' as a known issue. Budget for a motion-cleanup stage (PBHC's physics-based filtering + contact masks is the best open recipe to borrow).
- BeyondMimic = one trained policy PER dance (hours of cloud GPU each) and its workflow is hard-wired to a WandB registry ('Motions' artifact collection). Its deploy repo needs ROS 2 Jazzy, which targets Ubuntu 24.04 — on your 22.04 laptop use Docker, or run it on the Jetson. Repo README explicitly warns running these models on real robots is dangerous.
- No open stack controls the Inspire FTP hands as part of motion tracking — every tracker here drives the 29-DoF body only (some only 23 DoF: check that you use 29-DoF tasks/configs, e.g. unitree_rl_mjlab has a separate 23dof task). Finger choreography must be a separate stream (GEM-X does output hand poses you could map to Inspire via its SDK), synced by timestamp.
- Frame-rate/format conversions are a recurring silent-failure source: Unitree LAFAN1 CSVs are 30 fps, BeyondMimic/mjlab convert to 50 fps NPZ via forward kinematics (csv_to_npz.py --input-fps 30 --output-fps 50). Keep one canonical motion format (G1 29-DoF root pose + joint angles NPZ) as the contract between front-end and controllers.
- IsaacGym-based repos (PBHC, TWIST) are legacy: Python 3.8, NumPy 1.23, no longer downloadable through normal NVIDIA channels, painful on modern cloud images. Prefer Isaac Lab (BeyondMimic) or MuJoCo (unitree_rl_mjlab) training paths.
- Unitree's official dance demos are NOT reproducible from a public tool — there is no choreography format or editor; the only public artifacts are the retargeted LAFAN1 CSVs and the unitree_rl_mjlab training example. Don't go looking for a hidden Unitree 'dance API'.
- VideoMimic, despite the perfect-sounding name and Best Paper award, is the wrong base: terrain/context-conditioned policies for stairs/sitting, ~21 commits, stalled since Sep 2025.

### Open questions
- Does GEAR-SONIC's deploy config drive the full 29-DoF G1 (3-DoF waist + 7-DoF arms incl. wrists), and how does its kinematic-planner/encoder interface accept a fully custom long-horizon reference motion (vs teleop/gamepad modes)? Needs a read of gear_sonic_deploy configs and the 'Creating Your Own Reference Motions' doc, then a MuJoCo sim2sim test.
- Tracking fidelity on fast choreography: universal policy (SONIC, HoloMotion) vs per-motion BeyondMimic policy — no published head-to-head on identical dance clips; decide empirically with 2-3 LAFAN1 dance clips in sim2sim (joint MAE + balance-under-push tests).
- Push-robustness specifics: what perturbation/domain-randomization magnitudes were used in SONIC, BeyondMimic, and HoloMotion training, and do any expose a config to increase them when retraining? (BeyondMimic retraining is the lever you control.)
- HoloMotion v1.3.2: is the real-robot deploy path for Unitree G1-29DoF actually complete (it's Horizon Robotics — their own Tien Kung robot may be first-class), and are the pretrained weights Apache-2.0 like the code?
- Is the project commercial? If yes, the GEM (noncommercial) and possibly GVHMR front-ends are blocked — identify a permissively licensed world-grounded HMR alternative or budget for a licensed mocap front-end.
- KungfuBot2 (VMS multi-skill policy) code release status — is it merged into TeleHuman/PBHC or a separate upcoming repo? Worth tracking as a middle ground between per-motion and foundation policies.
- Can the Jetson Orin PC2 simultaneously run SONIC TensorRT inference and the Inspire hand control stream at required rates, or should the BeyondMimic CPU-ONNX controller run on the laptop with PC2 only doing DDS bridging?

---

# SYNTHESIS

# Architecture Decision: Video → Unitree G1 Dance Pipeline

**Status:** Decided. **Date:** 2026-06-11. **Robot:** Unitree G1 EDU Ultimate, 29 DoF, Inspire FTP hands, Jetson Orin PC2 @ 192.168.123.164. **Constraint:** RL whole-body motion tracking (push-robust), no open-loop playback.

---

## 1. Chosen Components Per Stage

| Stage | Component | Repo / Version | Runs On |
|---|---|---|---|
| Video → SMPL | **GVHMR** | https://github.com/zju3dv/GVHMR (main, ckpt `gvhmr_siga24_release.ckpt`, SimpleVO default; use static-camera flag for tripod footage) | Cloud GPU |
| Retargeting | **GMR** | https://github.com/YanjieZe/GMR (**branch: `master`**, pin a post-2026-01-21 commit), robot id `unitree_g1` (29 DoF) | Laptop CPU |
| Motion format conversion | **whole_body_tracking `scripts/csv_to_npz.py`** | https://github.com/HybridRobotics/whole_body_tracking (requires Isaac Lab headless) | Cloud GPU |
| RL training | **BeyondMimic** — `whole_body_tracking`, task `Tracking-Flat-G1-v0`, rsl_rl PPO, 4096 envs | https://github.com/HybridRobotics/whole_body_tracking, pinned **Isaac Lab v2.1.0 / Isaac Sim 4.5.0**, base image `nvcr.io/nvidia/isaac-lab:2.1.0` | Cloud GPU |
| Sim verify | **motion_tracking_controller `mujoco.launch.py`** (identical controller binary) + **unitree_mujoco** for SDK-contract check | https://github.com/HybridRobotics/motion_tracking_controller ; https://github.com/unitreerobotics/unitree_mujoco (DDS domain 1, iface `lo`) | Laptop CPU |
| Deploy | **motion_tracking_controller + unitree_bringup**, ONNX Runtime CPU (<1 ms/step), 500 Hz controller / 50 Hz policy, Docker image `qiayuanl/unitree:jazzy` (multi-arch arm64, pushed 2026-04-29), `--network host --privileged` | https://github.com/HybridRobotics/motion_tracking_controller ; https://github.com/qiayuanl/unitree_bringup | **Onboard Jetson Orin PC2** |
| Hands (optional channel) | Separate publisher on `rt/inspire_hand/ctrl/*` (FTP topics, NOT `rt/inspire/cmd` DFX topics) via unitree_sdk2_python; scripted/beat-synced poses v1 | https://github.com/unitreerobotics/unitree_sdk2_python ; ref https://github.com/unitreerobotics/dfx_inspire_service | Jetson PC2 or laptop |
| Smoke-test motions | **Unitree LAFAN1 Retargeting Dataset** (dance1_subject1/2, 30fps 29-DoF CSV) | https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset | n/a (data) |

### Conflict resolutions (explicit)

- **RL stack: BeyondMimic (Isaac Lab) over unitree_rl_mjlab (mjlab/MuJoCo-Warp).** The rl-controller researcher recommended unitree_rl_mjlab; deployment and turnkey researchers recommended upstream BeyondMimic. **Decision: upstream BeyondMimic.** Reasons: (a) it is the only stack with peer-reviewed real-G1 push-recovery evidence (arXiv 2508.08241) — push robustness is the project's hard requirement; (b) GMR ships an exporter literally labeled "for beyondmimic" (`scripts/batch_gmr_pkl_to_csv.py`) — zero glue-format risk; (c) its deploy path is verified onboard-capable (multi-arch Docker, full 29-joint config in `config/g1/controllers.yaml`, 500 Hz momentum-observer state estimation); (d) unitree_rl_mjlab has 20 commits and no tagged releases. **mjlab/unitree_rl_mjlab is the designated migration target** (Apache-2.0, no Isaac Sim, drops the RT-core GPU constraint) once one motion has been validated end-to-end on the proven stack — it is the same recipe by the same authors, so migration cost is low.
- **Deployment stack: one stack only — motion_tracking_controller onboard PC2.** Do NOT also install unitree_rl_mjlab's `g1_ctrl` on the Jetson (DDS conflicts with factory services).
- **Video front-end: GVHMR over NVIDIA GEM/TRAM/PromptHMR.** GVHMR is the only estimator with first-class integration in both GMR (`scripts/gvhmr_to_robot.py`) and PBHC; static-camera mode skips SLAM; outputs per-joint stationary probabilities for contact correction. GEM is NVIDIA non-commercial and assumes single full-body subject; PromptHMR is the upgrade path for multi-person/moving camera.
- **Per-motion training over universal tracker (SONIC) as primary.** SONIC (https://github.com/NVlabs/GR00T-WholeBodyControl, checkpoints https://huggingface.co/nvidia/GEAR-SONIC) is kept as a **parallel zero-training preview track**, not the backbone: it requires reflashing PC2 to JetPack 6 (invasive), has documented SMPL-drift and TensorRT-version sharp edges, and dance-specific fidelity is unproven. Evaluate it in sim2sim only until it earns hardware time. Per-motion BeyondMimic gives maximum choreography fidelity, which is the product.
- **PBHC/KungfuBot: borrow its contact-aware motion filtering recipe only.** Disqualified as a stack: CC BY-NC 4.0, deprecated IsaacGym (Python 3.8), 23-DoF annealed config (wrists + waist roll/pitch locked — kills dance expressiveness), no released deploy code.

---

## 2. GPU Strategy

- **Provider: RunPod (primary), Vast.ai (cost fallback).** Dispatch via the official `runpod` Python SDK (https://github.com/runpod/runpod-python, `runpod.create_pod()`), per-second billing. Vast.ai via https://github.com/vast-ai/vast-python.
- **GPU: RTX 4090 (Secure ~$0.69/hr or Community ~$0.34/hr) or RTX 6000 Ada ($0.77/hr).** Hard rule: **Isaac Sim requires RT-core GPUs — A100/H100 are unsupported.** Never rent them. Query GPU availability/price via API at dispatch time, don't hardcode.
- **Image:** one private image (GHCR/Docker Hub) `FROM nvcr.io/nvidia/isaac-lab:2.1.0` (NGC, free dev account, `docker login nvcr.io` user `$oauthtoken`) with whole_body_tracking + GVHMR + GMR baked in.
- **Statelessness:** no network volumes. **W&B is the artifact bus** (registry collection `Motions` must be pre-created; `WANDB_ENTITY` exported). Pods self-terminate in their start command (`train && push && terminate`) — orphaned pods are the real cost risk.
- **Cost per dance:** GVHMR extraction <$0.10; BeyondMimic training 1.5k–10k PPO iterations ≈ 2–10 h on one 4090 ≈ **$2–7 per trained dance policy** (worst case <$10 on L40S). Benchmark wall-clock once on `dance1_subject2` before budgeting a library.

---

## 3. End-to-End Data Flow (formats at each boundary)

```
[1] dance.mp4  — single continuous shot, single person, full body, 30 or 60 fps (trim cuts!)
      │  (cloud GPU pod)
[2] GVHMR → output.pt  — dict smpl_params_global {global_orient, transl, body_pose(63), betas}
      │     + per-joint stationary/contact probabilities, at video FPS
      │  (laptop CPU, GMR conda env py3.10)
[3] GMR scripts/gvhmr_to_robot.py --robot unitree_g1 → motion.pkl
      │     per-frame (root_pos xyz, root_rot quat **xyzw**, dof_pos[29], fps)
      │  visual gate: scripts/vis_robot_motion.py (MuJoCo viewer, laptop)
[4] GMR scripts/batch_gmr_pkl_to_csv.py → motion.csv
      │     30 fps, LAFAN1 convention: cols 0-2 root xyz, 3-6 quat xyzw,
      │     7-35 = 29 joints (legs 0-11, waist yaw/roll/pitch 12-14, L arm 15-21, R arm 22-28)
      │  (cloud GPU — launches Isaac Lab headless)
[5] whole_body_tracking scripts/csv_to_npz.py → motion.npz @50 fps
      │     {fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w}
      │     → uploaded to W&B registry {org}/wandb-registry-motions/{motion}
      │  (same pod)
[6] train.py --task=Tracking-Flat-G1-v0 --registry_name ... --headless --logger wandb
      │     → policy.onnx (joint order + PD kp/kd embedded in ONNX metadata) → W&B
      │  (laptop CPU)
[7] sim2sim gate: motion_tracking_controller mujoco.launch.py policy_path:=policy.onnx
      │     (+ optional unitree_mujoco on DDS domain 1 / iface lo)
      │  (Jetson Orin PC2, Docker qiayuanl/unitree:jazzy, --network host --privileged)
[8] ros2 launch motion_tracking_controller real.launch.py network_interface:=eth0 policy_path:=...
          500 Hz controller_manager / 50 Hz ONNX-CPU policy / 500 Hz contact+IMU estimator
          Joystick: L1+A standby → R1+A start policy → B damping e-stop
[9] (parallel, optional) hands.json keyframes → publisher on rt/inspire_hand/ctrl/* , beat-synced
```

Quaternion contract: CSV is **xyzw** (csv_to_npz converts to wxyz internally). Never hand-roll the pkl→csv step.

---

## 4. Phased Build Plan

**Phase 0 — Hardware ground truth (days 1–2, no cloud spend).**
Run `unitree_sdk2_python example/g1/low_level/g1_low_level_example.py` from the laptop: read LowState, confirm all 29 motors present and unlocked (incl. waist roll/pitch — some units ship with waist fasteners), record `mode_machine`, firmware version. Apply the CVE-2025-35027 (Sep 2025 BLE) firmware patch if missing, then **freeze firmware**. Verify the Inspire FTP service is running on PC2 and confirm its topic names (`rt/inspire_hand/ctrl|state|touch`). Practice gantry + zero-torque (L2+Y) + debug mode (L2+R2) procedure.

**Phase 1 — Robot dances ASAP, zero video work (week 1–2).**
Download `dance1_subject2` from the LAFAN1 Retargeting Dataset. Build the private training image; rent one RunPod 4090; run `csv_to_npz.py` → W&B → `train.py`. Pull policy.onnx; sim2sim on laptop (`mujoco.launch.py`); deploy onboard PC2 in the `qiayuanl/unitree:jazzy` container, robot suspended → feet touching → slack rope. **Exit criterion: G1 performs a LAFAN1 dance on hardware and recovers from a moderate push.** This validates the entire back half with a known-good motion before any front-end uncertainty.

**Phase 2 — Video front-end (week 2–4).**
Install GMR on the laptop (SMPL-X registration, `libstdcxx-ng`). Run GVHMR on a tripod-shot test clip (in the training pod, or LittleFrog/GVHMR HF Space / Meshcapade for one-offs). `gvhmr_to_robot.py` → visual screening in MuJoCo (check spins for L/R flips, foot sliding, arm-torso intersection) → CSV → train → deploy. Tune GMR velocity clamp (`use_velocity_limit`, default 3π rad/s) and IK weights on 2–3 clips. **Exit criterion: robot performs choreography from your own video.**

**Phase 3 — Automation (week 4–6).**
Laptop orchestrator (CLI or small web UI): `runpod.create_pod()` → pod start command runs GVHMR + csv_to_npz + train + W&B push + self-terminate; orchestrator polls W&B, pulls ONNX, runs sim2sim, stages deployment. Add Inspire hand mass to the G1 training asset (or payload DR) and retrain one motion; A/B wrist tracking in sim2sim.

**Phase 4 — Enhancements (ongoing).**
(a) SONIC evaluation track in MuJoCo sim2sim — if fidelity is acceptable, new dances drop from hours to minutes (requires JetPack 6 reflash decision). (b) Hand choreography channel on `rt/inspire_hand/ctrl/*`. (c) Migration eval to mjlab/unitree_rl_mjlab to cut training cost and the Isaac Sim dependency. (d) PromptHMR for multi-person/moving-camera clips. (e) Borrow PBHC contact-aware filtering between steps [2]→[3] if foot-skate persists.

---

## 5. Top 10 Risks & Mitigations

1. **Factory motion service conflicts with rt/lowcmd → violent jitter.** Always: hoist → zero torque → L2+R2 debug mode (or `MotionSwitcherClient.ReleaseMode()`) before any custom controller. First runs fully suspended.
2. **No software e-stop in debug mode.** Factory remote behaviors are gone; only fallback is the controller's B-button damping. Gantry is the true e-stop for every new policy; never run factory get-up/recovery with Inspire hands installed (manual warning — ground collision).
3. **Inspire FTP hand mass (~0.5 kg/wrist) unmodeled → degraded tracking, shoulder overheating** (unitree_sdk2_python #129). Add hand mass/payload randomization to the training asset; monitor motor temps; sim2sim ablation before hardware.
4. **GVHMR dance failure modes** (foot sliding — issue #21; L/R flips on fast back-facing spins; prior collapse under blur). Mandatory human visual screening of every retargeted motion in MuJoCo; static-camera flag; single continuous shots only; PBHC-style contact filtering as needed.
5. **Quaternion/joint-order/fps silent corruption.** Use GMR's official exporter only (xyzw CSV); keep video at clean 30/60 fps; verify non-integer downsample behavior in `batch_gmr_pkl_to_csv.py`; never mix 23-DoF (PBHC/ASAP) motion files as front-end input.
6. **Per-dance training cost/latency (hours, $2–7) surprises.** Benchmark `dance1_subject2` wall-clock on a 4090 in Phase 1; rsl_rl checkpoint-resume for preempted Vast instances; SONIC track as the fast-turnaround hedge.
7. **License contamination.** GVHMR (ZJU non-commercial), PBHC (CC BY-NC), GEM (NVIDIA NC), LAFAN1 motions (NC-ND), SMPL-X (research). Controller side is clean (MIT/Apache). If productized: email ZJU (xwzhou@zju.edu.cn) or switch front-end to Meshcapade MoCapade (commercial SaaS); use only self-captured motions.
8. **Version fragmentation breaks provisioning.** whole_body_tracking pins Isaac Lab 2.1.0/Isaac Sim 4.5.0 — Isaac Sim 5.x breaks it. Bake one image from `nvcr.io/nvidia/isaac-lab:2.1.0`, pin all repo commits (GMR `master` SHA, motion_tracking_controller, unitree_bringup), one env per repo.
9. **Onboard Jetson deployment unknowns** (ROS 2 Jazzy container on L4T under load alongside hand service; known issues #23 state_estimator joint-name, #28 sim discrepancy; LIO auto-start broken). Mitigate: sim2sim with the identical binary first; read issues #23/#28 before first run; fallback = run controller on the laptop at static IP 192.168.123.11 over wired DDS (supported); keep choreography robot-relative (XY drift with contact+IMU estimator).
10. **Floorwork/contact-rich choreography exceeds GMR/IK quality** (documented weakest case, jitter on some DanceDB dances). Screen per-motion; fallback retargeter = OmniRetarget in amazon-far/holosoma (Apache-2.0) with a small npz→CSV converter; for ground-contact dances consider BeyondMimic's optional Mid-360 LIO (currently broken in unitree_bringup — verify first).

---

## 6. Laptop (CPU-only) vs Remote

**Laptop (Ubuntu 22.04, no GPU) — can run:**
- GMR retargeting (real-time CPU, 35–70 FPS) + MuJoCo visual screening (`vis_robot_motion.py`)
- MuJoCo sim2sim: `mujoco.launch.py` (amd64 Docker `qiayuanl/unitree:jazzy`) and unitree_mujoco
- Orchestrator: RunPod SDK dispatch, W&B polling, artifact management, video trimming/cropping (ffmpeg)
- unitree_sdk2_python diagnostics, hand-pose publisher, SSH console to PC2
- Optionally the whole real-time controller itself (ONNX CPU <1 ms) at 192.168.123.11 over wired DDS — fallback if Jetson hosting misbehaves

**Cloud GPU (RunPod RTX 4090 / RTX 6000 Ada) — must be remote:**
- GVHMR pose extraction (CUDA mandatory, no CPU path)
- `csv_to_npz.py` (launches Isaac Lab headless)
- BeyondMimic PPO training (Isaac Lab 2.1.0, RT-core GPU mandatory)

**Jetson Orin PC2 — inference only:**
- motion_tracking_controller in arm64 Docker (ONNX CPU, tether-free) + Inspire FTP hand service
- Never: pose estimation, training, Isaac Sim (x86_64-only)
- (Future, SONIC track only: TensorRT inference, requires JetPack 6 reflash — deliberate decision, not default)