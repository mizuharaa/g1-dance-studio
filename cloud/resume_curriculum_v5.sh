#!/usr/bin/env bash
set -euo pipefail
export NB=${NB:-/workspace/notebook-data}
export WANDB_API_KEY=$(cat "$NB/.wandb_key")
PY=$NB/envs/mjlab/bin/python
TASK=Mjlab-Tracking-Flat-Unitree-G1-S2R-V5
NPZ=$NB/motions/thriller_clean.npz
RUN=train-thriller_v5fid-0713
LOGDIR=$NB/logs/rsl_rl/g1_tracking
COMMON=(--env.scene.num-envs 4096 --env.commands.motion.motion-file "$NPZ")
resolve() {  # $1=suffix -> "RUNDIR_BASENAME CKPT_BASENAME" (highest-iter ckpt, NUMERIC sort)
  local rundir ckpt
  rundir=$(ls -dt "$LOGDIR"/*"-$1" 2>/dev/null | head -1) || true
  [ -n "$rundir" ] || { echo "NO_RUNDIR"; return 1; }
  ckpt=$(ls -1 "$rundir"/model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt$//' | sort -n | tail -1) || true
  [ -n "$ckpt" ] || { echo "NO_CKPT"; return 1; }
  echo "$(basename "$rundir") model_${ckpt}.pt"
}
echo "===== RESUME STAGE 2 (0-50ms, +3000) from s1 $(date -Is) ====="
read -r R1 C1 <<< "$(resolve s1)"; echo "  resume from run=$R1 ckpt=$C1"
G1_CMD_DELAY_MAX_LAG=10 G1_OBS_DELAY_MAX_LAG=2 \
  "$PY" "$NB/cloud/train_sim2real_v5.py" "$TASK" "${COMMON[@]}" \
    --agent.max-iterations 3000 --agent.run-name "${RUN}-s2" \
    --agent.resume True --agent.load-run "$R1" --agent.load-checkpoint "$C1"
echo "===== RESUME STAGE 3 (0-60ms, +3000) from s2 $(date -Is) ====="
read -r R2 C2 <<< "$(resolve s2)"; echo "  resume from run=$R2 ckpt=$C2"
G1_CMD_DELAY_MAX_LAG=12 G1_OBS_DELAY_MAX_LAG=3 \
  "$PY" "$NB/cloud/train_sim2real_v5.py" "$TASK" "${COMMON[@]}" \
    --agent.max-iterations 3000 --agent.run-name "${RUN}-s3" \
    --agent.resume True --agent.load-run "$R2" --agent.load-checkpoint "$C2"
echo "===== VERIFY CHAIN $(date -Is) ====="
read -r R3 C3 <<< "$(resolve s3)"; CKPT="$LOGDIR/$R3/$C3"; echo "  final ckpt: $CKPT"
"$PY" "$NB/cloud/export_policy.py" "$CKPT" "$NPZ" "$NB/exports/${RUN}"
"$PY" "$NB/cloud/sim_gap_check.py" --checkpoint "$CKPT" --motion-file "$NPZ" --num-envs 128 --output-file "$NB/exports/${RUN}/gap.json"
for S in 90001 90011 90021; do "$PY" "$NB/cloud/heldout_eval.py" "$TASK" --checkpoint "$CKPT" --seed "$S" --num-envs 256; done
echo "==================== GATES / PULL artifacts $(date -Is) ===================="
