
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
