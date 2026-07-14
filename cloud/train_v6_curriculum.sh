#!/usr/bin/env bash
# Attempt 3 (v6 "station-keeping") — 3-stage curriculum on BOTH latency AND the
# new XY drift-termination band, run on the box. Prefer cloud/run_attempt3.sh,
# which preflights (incl. the v6 --selfcheck) before calling this.
#
# Stages (why: v5 proved learn-the-dance-first then harden; v6 adds a drift band
# that starts loose so early learning isn't over-terminated, then tightens to
# 0.5 m = half the 1.0 m gate):
#   1. 0-20 ms delay, drift<0.8 m, 4000 iters        (fresh)
#   2. 0-50 ms delay, drift<0.6 m, +3000 (resume s1)
#   3. 0-60 ms delay, drift<0.5 m, +3000 (resume s2)
# Then export -> gap_check -> heldout(3 seeds) -> verdict. ~5 h total.
set -euo pipefail

export NB=${NB:-/workspace/notebook-data}
[ -f "$NB/.wandb_key" ] && export WANDB_API_KEY=$(tr -d '[:space:]' < "$NB/.wandb_key")
PY=$NB/envs/mjlab/bin/python
ENTRY=$NB/cloud/sim2real_task_v6.py
TASK=Mjlab-Tracking-Flat-Unitree-G1-S2R-V6
MOTION=${MOTION:?set MOTION=/path/to/thriller_clean.npz}
RUN=train-thriller_v6sk-$(date +%m%d)
LOGDIR=$NB/logs/rsl_rl/g1_tracking
EXP=$NB/exports/${RUN}
COMMON=(--env.scene.num-envs 4096 --env.commands.motion.motion-file "$MOTION")

# newest run dir for a suffix + its highest-iteration checkpoint (NUMERIC sort —
# lexical sort put model_500 above model_3999 and cost us a stage on v5).
resolve() {  # $1=suffix -> "RUNDIR_BASENAME model_<n>.pt"
  local rundir ckpt
  rundir=$(ls -dt "$LOGDIR"/*"-$1" 2>/dev/null | head -1) || true
  [ -n "$rundir" ] || { echo "NO_RUNDIR"; return 1; }
  ckpt=$(ls -1 "$rundir"/model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt$//' | sort -n | tail -1) || true
  [ -n "$ckpt" ] || { echo "NO_CKPT"; return 1; }
  echo "$(basename "$rundir") model_${ckpt}.pt"
}
# fail loud if a resume grabbed the wrong checkpoint (guards the silent-restart trap)
assert_iter() {  # $1=ckpt basename  $2=expected floor
  local n; n=$(echo "$1" | sed 's/.*model_//; s/\.pt$//')
  [ "$n" -ge "$2" ] || { echo "!! checkpoint $1 iter $n < expected $2 — resume mis-resolved, ABORT"; exit 1; }
}

echo "===== STAGE 1/3  0-20 ms, drift<0.8 m, 4000 iters  $(date -Is) ====="
G1_CMD_DELAY_MAX_LAG=4  G1_OBS_DELAY_MAX_LAG=1  G1_DRIFT_TERM_M=0.8 \
  "$PY" "$ENTRY" "$TASK" "${COMMON[@]}" \
    --agent.max-iterations 4000 --agent.run-name "${RUN}-s1"

echo "===== STAGE 2/3  0-50 ms, drift<0.6 m, +3000 (resume s1)  $(date -Is) ====="
read -r R1 C1 <<< "$(resolve s1)"; echo "  resume run=$R1 ckpt=$C1"; assert_iter "$C1" 3900
G1_CMD_DELAY_MAX_LAG=10 G1_OBS_DELAY_MAX_LAG=2  G1_DRIFT_TERM_M=0.6 \
  "$PY" "$ENTRY" "$TASK" "${COMMON[@]}" \
    --agent.max-iterations 3000 --agent.run-name "${RUN}-s2" \
    --agent.resume True --agent.load-run "$R1" --agent.load-checkpoint "$C1"

echo "===== STAGE 3/3  0-60 ms, drift<0.5 m, +3000 (resume s2)  $(date -Is) ====="
read -r R2 C2 <<< "$(resolve s2)"; echo "  resume run=$R2 ckpt=$C2"; assert_iter "$C2" 6900
G1_CMD_DELAY_MAX_LAG=12 G1_OBS_DELAY_MAX_LAG=3  G1_DRIFT_TERM_M=0.5 \
  "$PY" "$ENTRY" "$TASK" "${COMMON[@]}" \
    --agent.max-iterations 3000 --agent.run-name "${RUN}-s3" \
    --agent.resume True --agent.load-run "$R2" --agent.load-checkpoint "$C2"

echo "===== VERIFY CHAIN  $(date -Is) ====="
# Verify RENDERS (gap_check/heldout), so it needs a GL backend — but TRAINING must
# NOT have one: MUJOCO_GL=egl during the 4096-env Warp training collides with the
# CUDA context and throws "illegal memory access" at the first reset. So egl is set
# HERE only, after training (matches the v5-proven split: no GL to train, egl to verify).
export MUJOCO_GL=egl
read -r R3 C3 <<< "$(resolve s3)"; assert_iter "$C3" 9900
CKPT="$LOGDIR/$R3/$C3"; mkdir -p "$EXP"; echo "  final ckpt: $CKPT"
"$PY" "$NB/cloud/export_policy.py"  "$CKPT" "$MOTION" "$EXP"
"$PY" "$NB/cloud/sim_gap_check.py" --checkpoint "$CKPT" --motion-file "$MOTION" \
    --num-envs 128 --output-file "$EXP/gap.json"
# FIXED heldout call: v5 passed $TASK as a bogus positional and omitted the
# REQUIRED --motion-file (hard crash) and --output-file (all seeds overwrote one
# file). Correct form — unique output per seed:
for S in 90001 90011 90021; do
  "$PY" "$NB/cloud/heldout_eval.py" --checkpoint "$CKPT" --motion-file "$MOTION" \
      --seed "$S" --num-envs 256 --output-file "$EXP/heldout_${S}.json" \
    || echo "  !! heldout seed $S failed (continuing; gap.json is the hard gate)"
done
echo "===== DONE $(date -Is) — gap.json + heldout_*.json in $EXP ====="
echo "  Gate PASS iff: nominal survival>=99%, drift_max<=1.0 m, 40ms+push survival>=95%,"
echo "  ankle p95<=15/20 Nm. Then pull to laptop, sign, DELETE THE BOX."
