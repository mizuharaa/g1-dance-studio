#!/usr/bin/env bash
# Re-run the v8 (attempt 5) VERIFY chain with the CORRECT task.
#
# Why: train_v8_curriculum.sh's verify section called pick_checkpoint / sim_gap_check /
# heldout_eval WITHOUT --task, so they gated the 770-dim v8 checkpoint against the
# 160-dim STOCK task -> every screen errored (state_dict size mismatch), the picker
# fell back to the LAST checkpoint (the exact v7 failure mode it exists to prevent),
# and no gap.json was produced. This script re-screens, re-exports the true winner
# (export_ckpt_onnx.py — export_policy.py copies the train-end ONNX which is always
# the LAST ckpt), re-gates, and finally runs the Agent A gate CALIBRATION: the
# thriller_csv_ankle_penalty anchor (~70% mimicry IRL, the only ground truth) through
# the same gate via its deploy ONNX.
#
# Honesty notes:
#   * G1_DRIFT_TERM_M=999 neutralizes the TRAINING drift termination during gating so
#     drift_max is MEASURED, not clipped — matches how v5/v6/v7 were gated (stock task
#     has no drift termination). Comparable numbers.
#   * The v8 task's 40 Nm velocity-derated ankle effort clamp is KEPT during gating —
#     it is closer to the real actuator than the stock 50 Nm flat clamp (Agent 0).
set -uo pipefail
NB=${NB:-/workspace/notebook-data}
PY=$NB/envs/mjlab/bin/python
TASK=Mjlab-Tracking-Flat-Unitree-G1-S2R-V8
TM=sim2real_task_v8
MOTION=$NB/motions/thriller_grounded_repaired_1p8x.npz
S3=$NB/logs/rsl_rl/g1_tracking/2026-07-16_10-40-45_train-thriller_v8s2r-0716-s3
EXP=$NB/exports/train-thriller_v8s2r-0716
ANCHOR_DIR=$NB/calibration
export MUJOCO_GL=egl
export G1_SLOWDOWN=${G1_SLOWDOWN:-1.8}
export G1_OBS_HISTORY=${G1_OBS_HISTORY:-5}
export G1_DRIFT_TERM_M=999
[ -f "$NB/.wandb_key" ] && export WANDB_API_KEY=$(tr -d '[:space:]' < "$NB/.wandb_key")
cd "$NB/cloud"
fail=0

$PY -c "import onnxruntime" 2>/dev/null || $PY -m pip install -q onnxruntime

echo "===== 1/5 screen last 6 checkpoints on the V8 task  $(date -Is) ====="
$PY pick_checkpoint.py --python "$PY" --gap-check "$NB/cloud/sim_gap_check.py" \
  --rundir "$S3" --motion-file "$MOTION" --last 6 --num-envs 64 \
  --workdir "$EXP/screen2" --task "$TASK" --task-module "$TM" \
  > "$EXP/pick2.log" 2>&1 || fail=1
tail -25 "$EXP/pick2.log"
if grep -q "every screen failed" "$EXP/pick2.log"; then
  echo "!! screens failed AGAIN — refusing the blind fallback; fix before gating"
  exit 1
fi
CKPT=$(grep '^WINNER ' "$EXP/pick2.log" | tail -1 | sed 's/^WINNER //')
[ -f "$CKPT" ] || { echo "!! no winner checkpoint; aborting"; exit 1; }
echo "SELECTED $CKPT"

echo "===== 2/5 re-export the WINNER to ONNX  $(date -Is) ====="
$PY export_ckpt_onnx.py --checkpoint "$CKPT" --motion-file "$MOTION" \
  --out-dir "$EXP" --task "$TASK" --task-module "$TM" || fail=1

echo "===== 3/5 full gate: 11 conditions x 128 envs  $(date -Is) ====="
$PY sim_gap_check.py --checkpoint "$CKPT" --motion-file "$MOTION" \
  --task "$TASK" --task-module "$TM" --num-envs 128 \
  --output-file "$EXP/gap.json" || fail=1

echo "===== 4/5 heldout x3 seeds  $(date -Is) ====="
for S in 90001 90011 90021; do
  $PY heldout_eval.py --checkpoint "$CKPT" --motion-file "$MOTION" \
    --task "$TASK" --task-module "$TM" --seed "$S" --num-envs 256 \
    --output-file "$EXP/heldout_${S}.json" \
    || echo "  !! heldout seed $S failed (gap.json is the hard gate)"
done

echo "===== 5/5 CALIBRATION (Agent A): anchor through the CURRENT gate  $(date -Is) ====="
# Anchor = thriller_csv_ankle_penalty (sha 444864f9…), ~70% mimicry IRL. Its training
# .pt is gone; run its deploy ONNX on the STOCK task (its native obs contract).
if [ ! -f "$ANCHOR_DIR/anchor_motion.npz" ]; then
  ( cd "$ANCHOR_DIR" && $PY "$NB/repos/mjlab/src/mjlab/scripts/csv_to_npz.py" \
      --input-file "$ANCHOR_DIR/thriller_csv_ankle_penalty_deploy.csv" \
      --output-name anchor_motion --input-fps 30 --output-fps 50 ) || true
  [ -f /tmp/motion.npz ] && cp /tmp/motion.npz "$ANCHOR_DIR/anchor_motion.npz"
fi
if [ -f "$ANCHOR_DIR/anchor_motion.npz" ]; then
  $PY sim_gap_check.py --onnx "$ANCHOR_DIR/policy.onnx" \
    --motion-file "$ANCHOR_DIR/anchor_motion.npz" --num-envs 32 \
    --only "nominal,delay20ms,delay40ms,delay40ms_push" \
    --output-file "$EXP/calibration_anchor_gap.json" || { echo "!! calibration run failed"; fail=1; }
else
  echo "!! anchor npz could not be produced — calibration skipped"; fail=1
fi

echo "===== DONE rc=$fail  $(date -Is) — artifacts in $EXP ====="
echo "Next (laptop): pull $EXP, judge v8 vs the CALIBRATED bar, sign, DELETE THE BOX."
