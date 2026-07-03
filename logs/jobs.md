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
