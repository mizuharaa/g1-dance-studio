#!/usr/bin/env bash
# Attempt 5 (v8) — the v-chain revamp curriculum. Layers Agent 0's asymmetric
# no-state-estimation actor (154-dim) + Agent D's CANDIDATE A actuation deltas onto the
# proven v7 station-keeping + drift-termination curriculum. Prefer cloud/run_attempt5.sh
# (it preflights v8 --selfcheck + the GPU smoke test + does the ground->repair->npz
# motion prep). Drift was SOLVED in v6/v7 — the drift-band curriculum is KEPT as-is.
#
# Stages (latency + drift band, identical schedule to v7 — do NOT regress drift):
#   1. 0-20 ms, drift<0.8 m, 4000 iters        (fresh)
#   2. 0-50 ms, drift<0.6 m, +3000 (resume s1)
#   3. 0-60 ms, drift<0.4 m, +5000 (resume s2)   <-- more time + tighter band, hard beats
# Then screen the last 6 checkpoints -> export the WINNER -> gap_check + heldout.
#
# G1_SLOWDOWN (default 1.8; fallbacks 2.0/2.5) is passed through to the recipe so the
# waist-slack windows scale to the slowed clock. The MOTION npz MUST be the matching
# ground->repair(1.8x)->npz motion (run_attempt5.sh builds it).
set -euo pipefail

export NB=${NB:-/workspace/notebook-data}
[ -f "$NB/.wandb_key" ] && export WANDB_API_KEY=$(tr -d '[:space:]' < "$NB/.wandb_key")
PY=$NB/envs/mjlab/bin/python
ENTRY=$NB/cloud/sim2real_task_v8.py
TASK=Mjlab-Tracking-Flat-Unitree-G1-S2R-V8
MOTION=${MOTION:?set MOTION=/path/to/thriller_grounded_repaired_1p8x.npz}
export G1_SLOWDOWN=${G1_SLOWDOWN:-1.8}
RUN=train-thriller_v8s2r-$(date +%m%d)
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

echo "===== v8 @ ${G1_SLOWDOWN}x slowdown, motion: $MOTION  $(date -Is) ====="
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

echo "===== VERIFY CHAIN  $(date -Is) ====="
export MUJOCO_GL=egl   # verify renders/needs GL; training above must NOT (Warp CUDA clash)
read -r R3 C3 <<< "$(resolve s3)"; assert_iter "$C3" 11900
S3DIR="$LOGDIR/$R3"; mkdir -p "$EXP"
echo "  final stage run dir: $S3DIR (last ckpt $C3)"

# BEST-checkpoint selection (v6 blindly shipped the last ckpt -> low episode length).
# Screen the last 6 checkpoints on the 2 gate-critical conditions, export the winner.
echo "  screening last 6 checkpoints for the best gate fit..."
"$PY" "$NB/cloud/pick_checkpoint.py" --python "$PY" \
    --gap-check "$NB/cloud/sim_gap_check.py" --rundir "$S3DIR" \
    --motion-file "$MOTION" --last 6 --num-envs 64 --workdir "$EXP/screen" \
    > "$EXP/pick.log" 2>&1 || true
cat "$EXP/pick.log"
CKPT=$(grep '^WINNER ' "$EXP/pick.log" | tail -1 | sed 's/^WINNER //')
if [ ! -f "$CKPT" ]; then
  echo "  !! picker produced no winner — falling back to last ckpt $C3"
  CKPT="$S3DIR/$C3"
fi
echo "  SELECTED ckpt: $CKPT"

"$PY" "$NB/cloud/export_policy.py"  "$CKPT" "$MOTION" "$EXP"
"$PY" "$NB/cloud/sim_gap_check.py" --checkpoint "$CKPT" --motion-file "$MOTION" \
    --num-envs 128 --output-file "$EXP/gap.json"
for S in 90001 90011 90021; do
  "$PY" "$NB/cloud/heldout_eval.py" --checkpoint "$CKPT" --motion-file "$MOTION" \
      --seed "$S" --num-envs 256 --output-file "$EXP/heldout_${S}.json" \
    || echo "  !! heldout seed $S failed (gap.json is the hard gate)"
done
echo "===== DONE $(date -Is) — gap.json + heldout in $EXP (selected $CKPT) ====="
echo "  Gate PASS iff: nominal survival>=99%, drift_max<=1.0 m, 40ms+push survival>=95%,"
echo "  ankle p95<=15/20 Nm. v8 EXTRA to eyeball (Agent D GPU-validation items): ankle p95"
echo "  at the 13-18s/25-36s beats (scaled x${G1_SLOWDOWN}) should drop <20 Nm AND trunk"
echo "  angular-momentum use should rise there. Exported policy actor obs MUST be 154-dim."
echo "  If survival marginal (90-95%): walk G1_SLOWDOWN 1.8 -> 2.0 -> 2.5 (one env var)."
echo "  Then pull to laptop, sign, DELETE THE BOX."
