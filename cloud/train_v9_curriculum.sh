#!/usr/bin/env bash
# Attempt 6 (v9) — v8 recipe UNCHANGED on the ADAPTIVE-time-warp motion.
#
# v9 = the proven v8 architecture (history+teacher-student 770-dim actor, ankle
# soft-barrier, per-channel ankle action-rate, waist slack at the beats, 40 Nm
# velocity-honest ankle clamp) trained on a DEMAND-SHAPED retimed reference
# (tools/motion_repair.py --adaptive): only the moments whose ankle demand exceeds
# 0.85x headroom are slowed; 49% of the dance keeps native 1.0x speed (arm swings
# etc.), overall 1.53x vs v8's uniform 1.8x. Feasibility: ankle max 34.0 Nm,
# p95 32.3, 0% frames over headroom (thriller_g1_grounded_adaptive_scorecard.json).
#
# The motion clock is non-uniform, so the waist-slack beat windows are PRE-WARPED
# by the launcher via the scorecard time_map and passed in G1_WAIST_WINDOWS
# (replaces the G1_SLOWDOWN scaling — G1_SLOWDOWN stays 1.0 and is ignored).
#
# Verify chain runs on the CORRECT task from the start (the v8 lesson):
# --task/--task-module everywhere + export_ckpt_onnx for the picked winner.
set -euo pipefail

export NB=${NB:-/workspace/notebook-data}
[ -f "$NB/.wandb_key" ] && export WANDB_API_KEY=$(tr -d '[:space:]' < "$NB/.wandb_key")
PY=$NB/envs/mjlab/bin/python
ENTRY=$NB/cloud/sim2real_task_v8.py
TASK=Mjlab-Tracking-Flat-Unitree-G1-S2R-V8
TM=sim2real_task_v8
MOTION=${MOTION:?set MOTION=/path/to/thriller_grounded_adaptive.npz}
export G1_SLOWDOWN=1.0                 # adaptive warp replaces uniform slowdown
export G1_OBS_HISTORY=${G1_OBS_HISTORY:-5}
export G1_WAIST_WINDOWS=${G1_WAIST_WINDOWS:?set from the adaptive scorecard time_map (e.g. 16.72-24.81,35.12-53.5)}
RUN=train-thriller_v9adpt-$(date +%m%d)
LOGDIR=$NB/logs/rsl_rl/g1_tracking
EXP=$NB/exports/${RUN}
COMMON=(--env.scene.num-envs 4096 --env.commands.motion.motion-file "$MOTION")

resolve() {  # $1=suffix -> "RUNDIR_BASENAME model_<n>.pt" (newest run, NUMERIC-max ckpt)
  local rundir ckpt
  rundir=$(ls -dt "$LOGDIR"/*"-$1" 2>/dev/null | head -1) || true
  [ -n "$rundir" ] || { echo "NO_RUNDIR"; return 1; }
  ckpt=$(ls -1 "$rundir"/model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt$//' | sort -n | tail -1) || true
  [ -n "$ckpt" ] || { echo "NO_CKPT"; return 1; }
  echo "$(basename "$rundir") model_${ckpt}.pt"
}
assert_iter() { local n; n=$(echo "$1" | sed 's/.*model_//; s/\.pt$//'); [ "$n" -ge "$2" ] || { echo "!! ckpt $1 iter $n < $2 — resume mis-resolved, ABORT"; exit 1; }; }

echo "===== v9 adaptive-warp motion: $MOTION  waist windows: $G1_WAIST_WINDOWS  $(date -Is) ====="
echo "===== STAGE 1/3  0-20 ms, drift<0.8 m, 4000 iters  $(date -Is) ====="
G1_CMD_DELAY_MAX_LAG=4  G1_OBS_DELAY_MAX_LAG=1  G1_DRIFT_TERM_M=0.8 \
  "$PY" "$ENTRY" "$TASK" "${COMMON[@]}" --agent.max-iterations 4000 --agent.run-name "${RUN}-s1"

echo "===== STAGE 2/3  0-50 ms, drift<0.6 m, +3000 (resume s1)  $(date -Is) ====="
read -r R1 C1 <<< "$(resolve s1)"; echo "  resume run=$R1 ckpt=$C1"; assert_iter "$C1" 3900
G1_CMD_DELAY_MAX_LAG=10 G1_OBS_DELAY_MAX_LAG=2  G1_DRIFT_TERM_M=0.6 \
  "$PY" "$ENTRY" "$TASK" "${COMMON[@]}" --agent.max-iterations 3000 --agent.run-name "${RUN}-s2" \
    --agent.resume True --agent.load-run "$R1" --agent.load-checkpoint "$C1"

echo "===== STAGE 3/3  0-60 ms, drift<0.4 m, +5000 (resume s2)  $(date -Is) ====="
read -r R2 C2 <<< "$(resolve s2)"; echo "  resume run=$R2 ckpt=$C2"; assert_iter "$C2" 6900
G1_CMD_DELAY_MAX_LAG=12 G1_OBS_DELAY_MAX_LAG=3  G1_DRIFT_TERM_M=0.4 \
  "$PY" "$ENTRY" "$TASK" "${COMMON[@]}" --agent.max-iterations 5000 --agent.run-name "${RUN}-s3" \
    --agent.resume True --agent.load-run "$R2" --agent.load-checkpoint "$C2"

echo "===== VERIFY CHAIN (correct task + honest gate)  $(date -Is) ====="
export MUJOCO_GL=egl                # verify only — never during training (Warp CUDA clash)
export G1_DRIFT_TERM_M=999          # measure drift, don't clip it (comparable to v5-v8 gates)
read -r R3 C3 <<< "$(resolve s3)"; assert_iter "$C3" 11900
S3DIR="$LOGDIR/$R3"; mkdir -p "$EXP"
echo "  final stage run dir: $S3DIR (last ckpt $C3)"

echo "  screening last 6 checkpoints (v8 task, 64 envs)..."
"$PY" "$NB/cloud/pick_checkpoint.py" --python "$PY" \
    --gap-check "$NB/cloud/sim_gap_check.py" --rundir "$S3DIR" \
    --motion-file "$MOTION" --last 6 --num-envs 64 --workdir "$EXP/screen" \
    --task "$TASK" --task-module "$TM" \
    > "$EXP/pick.log" 2>&1 || true
cat "$EXP/pick.log"
if grep -q "every screen failed" "$EXP/pick.log"; then
  echo "!! all screens failed — refusing the blind last-ckpt fallback"; exit 1
fi
CKPT=$(grep '^WINNER ' "$EXP/pick.log" | tail -1 | sed 's/^WINNER //')
[ -f "$CKPT" ] || { echo "!! picker produced no winner"; exit 1; }
echo "  SELECTED ckpt: $CKPT"

"$PY" "$NB/cloud/export_ckpt_onnx.py" --checkpoint "$CKPT" --motion-file "$MOTION" \
    --out-dir "$EXP" --task "$TASK" --task-module "$TM"
"$PY" "$NB/cloud/sim_gap_check.py" --checkpoint "$CKPT" --motion-file "$MOTION" \
    --task "$TASK" --task-module "$TM" --num-envs 128 --output-file "$EXP/gap.json"
for S in 90001 90011 90021; do
  "$PY" "$NB/cloud/heldout_eval.py" --checkpoint "$CKPT" --motion-file "$MOTION" \
      --task "$TASK" --task-module "$TM" --seed "$S" --num-envs 256 \
      --output-file "$EXP/heldout_${S}.json" \
    || echo "  !! heldout seed $S failed (gap.json is the hard gate)"
done
echo "===== DONE $(date -Is) — gap.json + heldout in $EXP (selected $CKPT) ====="
echo "  Judge vs the CALIBRATED bar (experiments/REGISTRY.md anchor row), NOT the raw"
echo "  absolute bars: anchor = 100% nominal / 34.4% @40ms+push == ~70% IRL."
echo "  v9 must BEAT v8 (99.2% nominal / 59.4% @40ms+push / ankle p95 20.4 / drift 4.31)"
echo "  and land ankle p95 <=15 now that the motion is fully feasible."
echo "  Then pull to laptop, sign, DELETE THE BOX."
