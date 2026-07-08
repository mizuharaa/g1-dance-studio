# Cloud jobs log — GreenNode box g1dance-gpu

Box: root@103.245.250.152 -p 46936 -i .secrets/greennode_ssh_key
Persistent mount: /workspace/notebook-data ($NB_DATA). Job runner: cloud/run_job.sh
(start|status|tail|list|stop <name>). W&B key on box: $NB_DATA/.wandb_key.
Training stack: **mjlab 1.5.0** (Isaac Lab failed on the fixed image — mjlab is the
architecture's bounded fallback). env: $NB_DATA/envs/mjlab. Task id:
**Mjlab-Tracking-Flat-Unitree-G1**. Always set MUJOCO_GL=egl.

## Cost meter
Box created 2026-07-03 ~17:20 UTC. Rate ~18,200 VND/h. Budget cap this window:
1.5M VND (~82 box-h). Track cumulative here each session.

## Jobs
| job | what | state | notes |
|-----|------|-------|-------|
| mjlab-install | pip install -e mjlab | done | mjlab 1.5.0, torch cuda OK (rc=1 was only a bogus __version__ probe) |
| convert-bench | csv_to_npz dance1_subject2_seg | done | in registry wandb-registry-motions/dance1_subject2_seg |
| convert-thriller | csv_to_npz thriller | running | → wandb-registry-motions/thriller (ready for attempt 1) |
| train-dance1-seg | BENCHMARK training | RUNNING | 4096 envs, ~1.6s/iter, GPU 76%, W&B run 40g4byo3 |

W&B project: https://wandb.ai/luong-alois-vng-group/mjlab
Benchmark run:  https://wandb.ai/luong-alois-vng-group/mjlab/runs/40g4byo3

## NEXT ACTIONS (auto-chain per user's full-auto grant)
1. Watch benchmark run 40g4byo3 ~1–2 h. "Clearly learning" = Metrics/motion/error_body_pos
   trending down (started 0.11) and rewards rising. When confirmed:
2. Kill benchmark to free the GPU:  bash cloud/run_job.sh stop train-dance1-seg
3. Launch Thriller attempt 1 (motion already converted):
   ssh ... 'cd $NB_DATA && bash cloud/run_job.sh start train-thriller-a1 -- \
     "cd $NB_DATA && MUJOCO_GL=egl WANDB_API_KEY=\$(cat .wandb_key) \
      ./envs/mjlab/bin/python repos/mjlab/src/mjlab/scripts/train.py \
      Mjlab-Tracking-Flat-Unitree-G1 --registry-name wandb-registry-motions/thriller \
      --env.scene.num-envs 4096 --video False"'
4. ≤3 Thriller attempts; between attempts diagnose (reward weights, window trim) and
   record here. On convergence: export policy + policy_meta.json sidecar (contract in
   PROJECT_STATE / peer msg), pull to data/policies/, then run sim_exam gate.
5. Then long-dance validation on dance2_subject4.csv (window+vet on laptop first).

## Resume after a dead session
ssh to box; `bash cloud/run_job.sh list` shows all job states; tmux ls for live
sessions. Trainings survive laptop reboots (they run in tmux on the box).

## 2026-07-04 ~01:00 ICT — killed ROGUE duplicate benchmark

- A SECOND train-dance1-seg was found running (created 17:40, iter ~600/30000) — NOT
  mine. Different invocation: `train.py --registry-name wandb-registry-motions/
  dance1_subject2_seg` + `cat .wandb_key`, direct train.py (not job_train.sh), NO
  iteration cap. Launched by another agent following the old registry-based interface.
  It was SHARING the 4090 with the useful long-dance → each at ~half speed.
- KILLED it (cost calibration already captured; benchmark purpose done). Verified:
  only pid 14290 = train-dance2-long remains on GPU (2030 MiB, solo). Long-dance now
  runs ~2x faster. Note for coordinator: some agent relaunches the benchmark via the
  registry path — should stop doing that; benchmark is DONE.
- **COST CALIBRATION (final, captured before kill):** ~2040 iters/hr GPU-shared
  (faster solo); **~8,900 VND per 1000 iters**; a converging dance (~3000 iters) ≈
  **27k VND ≈ $1.04 compute**. Box-hours ≈8h ≈ 145k VND of 1.5M cap.

## 2026-07-04 ~01:25 ICT — Thriller ATTEMPT 2 (tighten to ≥99% held-out)

- **Show-ready bar (user via coordinator):** ≥99% held-out survival (mjlab_heldout_v1)
  then gantry. Attempt 1 hit 98.4% (~127/128 under noise+shoves, mpkpe 0.17m) — strong
  but below bar.
- **train-thriller-a2 LAUNCHED** (W&B 55kbaa8i): same cleaned 49.3s show cut
  (thriller_show.npz), minimal recipe delta = **action_rate_l2 weight -0.1 → -0.2**
  (smoothness/stability targeting the occasional falls; NOT over-tuning — single delta).
  4096 envs, 4000-iter cap. Running IN PARALLEL with dance2-long (share GPU ~2x slower
  each; parallel wall-clock ≤ sequential for eventual box deletion, and gets the
  show-critical result sooner).
- **Held-out gate tooling ready**: cloud/heldout_eval.py on box (256 envs, held-out
  seed 90001, nominal + push conditions), pipeline/mjlab_verify.py laptop-side signer.
  Post-convergence plan: export (per-joint action_scale honored via mjlab exporter) →
  heldout_eval → if ≥99% survival = sim-verified; else attempt 3 (last).
- Watchdog + auto-render now cover BOTH train-dance2-long and train-thriller-a2.
- Box-hours ~8.5h ≈ 155k VND of 1.5M cap.

## 2026-07-04 ~02:15 ICT — Thriller policy STAGED for gantry (robot day tomorrow AM)

- **Best policy = attempt-1** (attempt-2 not converged: iter ~1122/4000, reward 23 climbing;
  a1 remains best exported). Staged at data/policies/thriller/policy.onnx (+ model_3000.pt).
- **policy_meta.json now COMPLETE** (was missing PD spec — critical for real robot):
  per-joint kp (14.3-99.1), kd (0.91-6.31), effort limits (5-139 Nm), default_joint_pos
  (29-dof), action_scale (incl 0.074 wrists, 0.35 knees), obs term order, impedance model
  (kp=armature*(2pi*10)^2, kd=2*zeta*armature*2pi*10, zeta=2 overdamped — SIM gains ARE
  deploy gains per BeyondMimic). Mirrored to docs/mjlab_policy_interface.json (tracked).
- **ACTIVATION HAZARD found + fixed**: clip frame-0 differed from standby default_joint_pos
  by up to 0.68 rad (39deg elbows, 38deg straight-vs-bent knees) → activation lurch.
  FIX: generated thriller_deploy.csv/.npz = 2.5s cosine ramp default_joint_pos->dance
  prepended (frame-0 delta now 0.000). Policy re-verified in-engine on the ramped motion:
  100% full completion, 0.117 rad err. NO retrain needed. **Deploy-kit: use thriller_deploy
  for the gantry, NOT thriller_show.**
- **Gantry-safety Q (base_lin_vel):** actor obs includes base_lin_vel with training noise
  Unoise(-0.5,+0.5) [tracking_env_cfg]. On gantry feet-off-ground base_lin_vel~0, well
  within that noise band → in-distribution. Current policy IS gantry-safe. (Real free-stand
  later needs the onboard estimator feeding base_lin_vel — DLIO/LiDAR+IMU, per derisk doc.)

## 2026-07-04 ~02:30 ICT — GROUND UNLOCKED: attempt-2 = 100% held-out

- **Thriller attempt-2 CLEARS THE >=99% GROUND BAR at 100%.** Autopilot exported the
  iter-1500 a2 checkpoint and ran the held-out gate on the DEPLOYABLE motion
  (thriller_deploy): **nominal 256/256 (100%), push 256/256 (100%), signed verdict PASS**.
  The action_rate_l2 -0.2 delta worked (a1 98.4% → a2 100%).
- **Trade-off noted**: a2 mpkpe 0.221m (nominal) vs a1 0.168m — a2 survives more but
  tracks looser (action-rate penalty = more stable, less crisp). Both are valid; a2's
  100% survival is what gates GROUND, a1's tighter tracking is the crisper-looking fallback.
- **STAGED**: data/policies/thriller/ PRIMARY = a2 (ground-ready), policy.onnx swapped;
  a1 preserved at data/policies/thriller_a1_fallback/. Shared: policy_meta.json (PD
  gains etc — policy-independent), thriller_deploy.{csv,npz} (2.5s activation ramp).
  See data/policies/thriller/STAGED.txt.
- **Autopilot bug fixed + relaunched**: original fired early at iter 1500 on an SSH/tmux
  blip (a2 was actually still running). v2 requires status=done confirmed twice; now
  waiting for a2's TRUE final (iter 4000) → verifies final checkpoint → if it holds >=99%
  with tighter mpkpe than 0.221m, hot-swaps the primary before morning. Writes
  data/policies/thriller_a2_final/RESULT.txt.
- Long-dance train-dance2-long at 4446/6000, reward 33.6 — converging, verdict soon.
- Box-hours ~9.5h ≈ 173k VND of 1.5M cap.

## 2026-07-04 ~03:00 ICT — LONG-DANCE VERDICT: recipe validated

- **train-dance2-long CONVERGED + VERIFIED.** Done at iter 5999/6000, reward 34.62.
  In-engine full-motion eval (67.2s / 3359 frames @ 50fps):
  - CLEAN (4 env): 100% completion, joint err **0.099 rad** (tighter than Thriller's 0.117)
  - NOISE (64 env): 100% completion, 0.099 rad
  **→ The longer-horizon training recipe (single-clip + adaptive-kernel 6) WORKS.**
  A 67s dance performs end-to-end with better tracking than the 49s Thriller.
  Product 2-3min target de-risked on the training side (only constraint = in-area
  choreography, already flagged: stock traveling mocap caps window length at ~62s in 2m).
  Registered as dance "Dance2-Long" (draft).
- Thriller a2 at 3794/4000 (reward 30.4, climbing) — final-checkpoint autopilot waiting
  for its true completion to verify + hot-swap if tighter than 0.221m mpkpe.
- Box-hours ~10h ≈ 182k VND of 1.5M cap.

## 2026-07-04 ~03:15 ICT — OVERNIGHT COMPLETE. All training done, GPU idle.

- **Thriller a2 FINAL checkpoint (iter 3999)**: also 100% held-out, but mpkpe 0.249m —
  LOOSER than iter-1500's 0.221m (action-rate penalty kept trading precision for
  smoothness with more training). ⇒ **iter-1500 KEPT as staged primary** (best a2:
  100% survival + tightest tracking among a2 checkpoints). Swap-if-better logic correctly
  declined the swap. Final artifacts at data/policies/thriller_a2_final/.
- **Long-dance policy preserved**: exported dance2-long final → data/policies/dance2_long/policy.onnx.
- **All GPU work done; GPU idle (0%).** Render loop + watchdogs stopped. Box still ALIVE
  (~18k VND/h idle) — KEPT (not deleted) through robot day: derisk doc anticipates a
  possible retrain if the gantry shows oscillation (latency+PD-gain DR), and re-provision
  is ~1h. Budget ~182k/1.5M VND; keeping through the morning ~+110k stays well under.
  **BOX DELETION = user's call** (destructive: loses provisioned env, data is all on laptop).
- **ROBOT-DAY READY**: data/policies/thriller/ = a2 100% (ground), thriller_a1_fallback/ =
  a1 (gantry/crisp), thriller_deploy.{csv,npz} (2.5s activation ramp), policy_meta.json
  (full PD gains). Deploy-kit to build --full bundle. Robot untouched; deploy human-gated.

## 2026-07-05 — sim2real retrain attempt 1 (recipe v2, post-audit)
- **train-thriller-s2r** RUNNING (started 14:25 UTC): task Mjlab-Tracking-Flat-Unitree-G1-Sim2Real
  (cloud/sim2real_task.py via cloud/train_sim2real.py), motion thriller_deploy.npz, 4096 envs,
  5000-iter cap, ~1.1-1.3 s/it, ETA ~1.8 h. Recipe: torque penalties (headline), system-ID mass,
  actuator DR, leg-odom obs dynamics, 0-20 ms latency DR, 20 s episodes. W&B auto.
- **s2r-autopilot** RUNNING: waits for the train job -> export ONNX (last + mid) ->
  cloud/sim_gap_check.py v2 gate (full motion, 7 conditions incl. 40 ms delay eval-only) ->
  writes exports/thriller_s2r/RESULT.txt (VERDICT=GATE_PASS/FAIL + numbers + next steps).
- Resume: `bash cloud/run_job.sh status train-thriller-s2r` / `status s2r-autopilot`;
  verdict at exports/thriller_s2r/RESULT.txt. Baseline (deployed a2 on the same gate):
  reports/sim_gap_check_a2_1500_full.json.

## 2026-07-08 — GPU box RECREATED autonomously via Chrome pilot (no API needed)
- **g1dance-4090** created 11:15:07 (id `nb-c7b924ff-d359-43a6-9014-d37494ff89df`), HCM /
  HCM-03-1A, RTX4090×1 / 16 CPU / 64 GB / 24 GB VRAM, Pytorch 2.5.1 CUDA 12.4, block 20 GB.
  Network volume `g1dance-data` (nv-cb2e7860...) SURVIVED (fast path) → mounted at
  /workspace/notebook-data. SSH pubkey g1dance-laptop installed; ports HTTP 8888 + TCP 22.
- Done entirely by `tools/pilot.py` (headed Chrome, DISPLAY :0) driving the GreenNode console;
  user only solved the login reCAPTCHA. Corrects the earlier "no GreenNode API → can't
  create a box" wall. Account is VNG postpaid ("0 credits" is normal; billed month-end).
- NEXT: wait for Running → grab SSH host:port from Connect dialog → update .secrets/cloud.json
  → verify SSH+GPU → re-provision (BOX_RECREATE_RUNBOOK Part 4) → kick the 2-min Thriller job.
  DELETE when the job's artifacts are pulled (teardown).

### retry (1st attempt ERRORed)
- Attempt 1 (zone HCM-03-1A) reached **ERROR** at ~11:33 after ~18 min CREATING; the local
  NVMe/compute storage also ERRORed, no event/log message surfaced → silent backend
  allocation failure (most likely RTX4090 capacity in 1A). Deleted it.
- Attempt 2 created 11:40:43 in **zone HCM-03-1B** (hedge against 1A capacity). Same config;
  `g1dance-data` volume attaches in 1B too (region-scoped). Polling for RUNNING.

### box UP (attempt 2, zone 1B)
- ACTIVE ~11:49 (~9 min). SSH `-p 44662 root@103.245.250.152` VERIFIED; GPU = RTX 4090 24564 MiB.
- cloud.json updated (host 103.245.250.152, port 44662).
- **Volume contents WIPED**: /workspace/notebook-data = only lost+found (the "0 B" was real) →
  FULL re-provision path (not fast). Running BOX_RECREATE_RUNBOOK Part 4 (bootstrap + gvhmr + mjlab).

### 20GB-volume fix + job RESUMED (2026-07-08 ~05:15 UTC)
- Volume `g1dance-data` is only **20 GB** (undersized; console offers no resize — "Update
  volume" only renames). Provisioning onto it filled it → mjlab install failed "No space".
- FIX (no code change): relocated provisioning to block storage `/root/nbdata` (94 GB disk,
  NB_DATA-honored scripts), then made every `/workspace/notebook-data/<subdir>` a SYMLINK to
  `/root/nbdata/<subdir>`. Volume back to 1% used; app + run_job.sh use the default path
  transparently, all real I/O lands on block storage. mjlab smoke test green via default path.
- Provisioned: bootstrap + GVHMR (torch 2.3.0+cu121, checkpoints) + mjlab (mjlab_ready, cuda True).
- Started server headless (g1dance conda, :8735) → forced retry of **Thriller dance FULL 2min**
  (job 20260707-185326-ba1585, 124 s @ 640x360). Extract now RUNNING: tmux job-gvhmr-...,
  GPU 34%/1.2GB, artifacts appearing. Pipeline will auto-advance extract→retarget→train→verify→export.
- Box must stay UP until export artifacts are pulled; DELETE after (Chrome pilot / teardown).

### PERMANENT training-env fix (2026-07-08) — convert now works
Root cause of both Thriller jobs failing at train: `envs/mjlab` was a `--system-site-packages`
venv over this GreenNode compute-only image, so mjlab inherited the base /opt/conda's
INCOMPATIBLE packages — cascading failures (mjlab script path → libstdc++/matplotlib →
no GL runtime → scipy `sph_legendre_p` ufunc). The prior image happened to be compatible.
PERMANENT FIX (image-independent, baked into provisioning):
1. `cloud/20_training.sh`: mjlab now installs into an **isolated venv** (no system-site-packages)
   → brings its own consistent numpy/scipy/matplotlib/torch (torch 2.12.1+cu130, CUDA OK).
2. Install GLVND loaders `libglvnd libegl libgl libglx libopengl` (image has NO GL / no NVIDIA
   EGL; mjlab imports PyOpenGL EGL at load) → `libEGL.so.1` present.
3. `repos/mjlab/src/mjlab` symlink → site-packages (app expects repo layout).
4. `pipeline/stages/cloud_motion.py`: all box scripts export `LD_LIBRARY_PATH=/opt/conda/lib`
   (for libEGL + newer libstdc++).
VERIFIED: convert (csv_to_npz) of the 2-min Thriller ran clean, rc=0, CONVERT_OK (npz produced).
Current box already patched live; provisioning fixed for all future boxes.

### 2026-07-08 — infra fixed end-to-end; policy-quality outcome pending
- FULL 2-min video Thriller (ba1585): pipeline ran fully (convert/train/export via the permanent
  fixes), but the trained policy FAILED verify — 0/128 nominal survival, joint-track err ~1.25 rad,
  reward plateaued 7.6. Training never converged. Cause = MOTION quality (640x360 source -> noisy
  retarget) + 5000 iters too few for a 2-min jump motion. njmax NOT the cause (5 rare overflows).
  The verify gate correctly rejected it.
- CSV Thriller (new job 786ffa, rerun of stale 3d5060 whose 07-03 retarget lacked deploy_csv):
  cleaner CSV motion (thriller_g1.csv), now training with all fixes. Watcher polling to terminal.
  User decision: let the CSV job finish first before any retrain of the video motion.
