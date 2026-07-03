
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
