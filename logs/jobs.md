
## 2026-07-03 — benchmark training kickoff (mjlab path)

- **Stack verdict**: Isaac Lab 2.1.0 DEAD on GreenNode image (isaacsim wheels rejected,
  isaaclab.sh install fails). **mjlab 1.5.0 IS the trainer** (repo checkout at
  box:/workspace/notebook-data/repos/mjlab, venv envs/mjlab, MuJoCo 3.10 + Warp 1.14).
  Task: `Mjlab-Tracking-Flat-Unitree-G1` (also a No-State-Estimation variant).
- **Key discoveries**: mjlab ships its OWN csv_to_npz (GPU FK, no Isaac Sim) with the
  same CSV convention as ours (xyzw→wxyz, LAFAN1 29-joint order); train.py accepts
  `--env.commands.motion.motion-file <local.npz>` (W&B registry optional, not required);
  push randomization built into the task (push_by_setting_velocity every 1–3 s);
  ONNX export exists (mjlab/rl/exporter_utils.py). Box needed apt libegl1/libosmesa6
  for headless GL (installed).
- **Motion registered**: dance1_subject2_seg → box:/workspace/notebook-data/motions/
  dance1_subject2_seg.npz (50 fps, from 863-frame 30 fps CSV) + W&B registry
  `wandb-registry-motions/dance1_subject2_seg`.
- **JOB RUNNING**: `train-dance1-seg` (tmux session job-train-dance1-seg,
  log box:/workspace/notebook-data/jobs/train-dance1-seg.log, started 15:25 UTC =
  22:25 ICT). Check: `bash /workspace/notebook-data/cloud/run_job.sh status|tail
  train-dance1-seg` (PATH needs /workspace/notebook-data/bin for tmux).
- **Box-hours**: created ~17:20 ICT 2026-07-03 → ~22:30 ICT ≈ 5.2 h ≈ 95k VND of the
  1.5M cap. Thriller CSV already on box (motions/thriller_g1.csv), ready to convert.

## 2026-07-03 22:40 ICT — benchmark healthy, Thriller attempt 1 launched

- **train-dance1-seg** (benchmark, 28.8s test segment): 4096 envs, ~1.1 s/iter,
  W&B https://wandb.ai/luong-alois-vng-group/mjlab/runs/40g4byo3. Curve HEALTHY:
  reward 0.22→1.65, ep-len 16→56 by iter ~354. Default 30k iters (ETA ~9.5h) —
  fine to let run; can be stopped early once cost calibration + a sim-exam
  checkpoint exist. Converter-ordering bug (#777) RULED OUT (mjlab's own converter
  used; ep-len not pinned at 1). Note: first two launch attempts were broken
  (quoting bug → stuck `cat`; then num_envs=1 default) — fixed via
  cloud/job_train.sh launcher + explicit --env.scene.num-envs.
- **train-thriller-a1**: STOCK config per recipe, 4096 envs, --agent.max-iterations
  10000, motion = thriller_show.npz (49.3s show cut: GMR velocity-limit retarget,
  residual clamp touched 104 frames → 0% over-limit (peak 8.48 rad/s), FK ground
  fix +3.8cm, 1s standing pad + 0.5s blend-in, 1s blend-out + 2.5s standing hold;
  vet PASS all hard checks, foot-skate 0.248 ≤ 0.3). Started 15:40:03 UTC.
- Both jobs share the 4090 (VRAM ~2.6GB each, plenty of headroom); persistent
  monitor reports every ~50 min and immediately on crash/completion.
- Box-hours: ≈5.5h ≈ 100k VND of 1.5M cap at Thriller launch.

## 2026-07-03 ~23:15 ICT — W&B URL correction + visual progress renders

- **W&B correction**: run `40g4byo3` (cited earlier as the live benchmark) is actually
  a KILLED early launch (num_envs=1) — that's the "Crashed" run the user saw. LIVE runs:
  - benchmark train-dance1-seg → **ue5nw8u1** (writing, live)
  - thriller-a1 → **yhx35nb1** (writing, live)
  Both syncing in-process. Box uplink is FLAKY (SSH drops mid-command; a wandb heartbeat
  blip is what "crashed" 40g4byo3's logger) → **watchdog reports are source of truth,
  not W&B run status.** Renders/commands that must survive a drop run in tmux, never bare SSH.
- **Visual progress renders**: cloud/render_progress.sh renders a job's LATEST checkpoint
  via `mjlab play --video` (1 env, headless EGL, 500 steps, 640x480) → box:
  previews_progress/<job>_iter<N>.mp4. Launched in tmux (render-bench, render-thriller).
  Pull to laptop data/previews/progress/. Repeat every ~500-1000 iters for a "robot right
  now" clip. Checkpoints save every 500 iters under cloud/logs/rsl_rl/g1_tracking/<ts>_<job>/.

## 2026-07-04 00:30 ICT — Thriller a1 CONVERGED + exported + in-engine verified; long-dance started

- **Thriller a1 CONVERGED** (W&B yhx35nb1): reward plateau 30.6-32.7 over ~1500 iters,
  max 32.75, ep-len 477/500. Stopped at iter ~3143 (no point running to 10k). Best
  checkpoint model_3000.pt → exported ONNX (obs[160]+time_step, opset 18) →
  data/policies/thriller/{policy.onnx, model_3000.pt}.
- **IN-ENGINE eval** (mjlab play cfg, start@0, full 49.3s / 2464 frames):
  - CLEAN (4 env): 100% full-motion completion, joint_pos err 0.118 rad
  - PUSH/NOISE (64 env, IMU+encoder corruption): 100% completion, err 0.117 rad
  NOTE: same-engine, NOT independent sim2sim. Dance registered 'draft' (id
  20260704-18f65bbd), policy attached; show-ready withheld pending signed sim_exam.
  sim_exam.py obs-adapter gap (Isaac layout vs mjlab 160-dim) handed to deploy-kit agent.
  Verdict: data/policies/thriller/in_engine_eval.json.
- **COST CALIBRATION** (from W&B timing): ~2040 iters/hr (GPU-shared; solo faster).
  ≈8,900 VND/1000-iter. A Thriller-class dance (~3000 iters to converge) ≈ **27k VND
  ≈ $1.04 of compute**. Box-hours so far ≈7.2h ≈ 131k VND of 1.5M cap.
- **Benchmark train-dance1-seg STOPPED** at ~3900 iters (stack-validation + cost-calib
  done; W&B ue5nw8u1, reward ~27). No reason to run to 30k.
- **LONG-DANCE train-dance2-long STARTED**: dance2_subject4 longest clean window =
  62.2s (of 225.7s; longer windows exceed the 2m area → excursion 1.47m at the cap).
  Show-prepped to 67.2s, vet PASS. Recipe delta: --adaptive-kernel-size 6 (long-seq).
  4096 envs, 6000-iter cap. **PRODUCT FINDING: stock LAFAN1 can't yield a 2-3min
  in-area window — real long show dances must be choreographed to stay within 2m.**
- Watchdog re-armed for dance2-long; auto-render loop repointed to it.
