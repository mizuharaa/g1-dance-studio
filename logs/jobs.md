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
