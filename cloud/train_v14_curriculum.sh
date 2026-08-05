#!/usr/bin/env bash
# Attempt 7 (v10) — SPEED-CURRICULUM training to NATIVE (1.0x) tempo.
#
# 4 tempo stages on tempo-resampled copies of the SAME native motion
# (cloud/retime_motion.py, cubic + slerp — no frame duplication):
#     stage 1: 0.60x tempo   0-20 ms latency DR   drift < 0.8 m   iters ->  2500
#     stage 2: 0.75x tempo   0-40 ms              drift < 0.6 m   iters ->  5000
#     stage 3: 0.90x tempo   0-50 ms              drift < 0.5 m   iters ->  7250
#     stage 4: 1.00x tempo   0-60 ms              drift < 0.4 m   iters ->  9500
# HARD BUDGET: 9500 total. NEVER extend the final stage (+5000-style) — v7 and v9
# both diverged after ~9.5-10k iters (late-stage collapse is a PATTERN; v8, which
# stopped earlier, did not). The best-checkpoint picker screens the last 6 ckpts.
#
# G1_SLOWDOWN per stage = 1/tempo — it ONLY scales the v8 waist-slack windows to
# that stage's motion clock (the tempo itself lives in the npz).
#
# Verify chain (v8 lesson, commit 085ab43): --task/--task-module EVERYWHERE,
# re-export the PICKED checkpoint via export_ckpt_onnx.py, G1_DRIFT_TERM_M=999
# during gating. Ankle p95 gate bar = 22 Nm nominal / 25 worst via env override
# (real robot measured 15-19 Nm p95 at NATIVE tempo — decision log 2026-07-20,
# experiments/torque_crosscheck_20260720; the old 15 bar was below physical
# reality). All other bars unchanged.
set -euo pipefail

export NB=${NB:-/workspace/notebook-data}
[ -f "$NB/.wandb_key" ] && export WANDB_API_KEY=$(tr -d '[:space:]' < "$NB/.wandb_key")
PY=$NB/envs/mjlab/bin/python
ENTRY=$NB/cloud/sim2real_task_v14.py
TASK=Mjlab-Tracking-Flat-Unitree-G1-S2R-V14
TM=sim2real_task_v14

# The four tempo variants (produced by run_attempt7.sh preflight).
M060=${M060:?set M060=/path/to/thriller_v10_060.npz}
M075=${M075:?set M075=/path/to/thriller_v10_075.npz}
M090=${M090:?set M090=/path/to/thriller_v10_090.npz}
M100=${M100:?set M100=/path/to/thriller_v10_100.npz}
for f in "$M060" "$M075" "$M090" "$M100"; do
  [ -f "$f" ] || { echo "!! motion npz missing: $f"; exit 1; }
done

export G1_OBS_HISTORY=${G1_OBS_HISTORY:-5}
export G1_PHASE_OBS=${G1_PHASE_OBS:-0}   # default OFF — enabling changes the deploy obs contract
RUN=${RUN_NAME:-train-thriller_v14style-$(date +%m%d)}
LOGDIR=$NB/logs/rsl_rl/g1_tracking
EXP=$NB/exports/${RUN}
COMMON=(--env.scene.num-envs 4096)
cd "$NB"   # mjlab resolves logs/ relative to CWD — pin it or stage-resume breaks (attempt-6 lesson)

resolve() {  # $1=suffix -> "RUNDIR_BASENAME model_<n>.pt" (newest run of THIS RUN NAME only —
  # a bare *-s2 glob once matched the OLD v8 s2 and resumed the wrong lineage)
  local rundir ckpt
  rundir=$(ls -dt "$LOGDIR"/*"${RUN}-$1" 2>/dev/null | head -1) || true
  [ -n "$rundir" ] || { echo "NO_RUNDIR"; return 1; }
  ckpt=$(ls -1 "$rundir"/model_*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt$//' | sort -n | tail -1) || true
  [ -n "$ckpt" ] || { echo "NO_CKPT"; return 1; }
  echo "$(basename "$rundir") model_${ckpt}.pt"
}
assert_iter() { local n; n=$(echo "$1" | sed 's/.*model_//; s/\.pt$//'); [ "$n" -ge "$2" ] || { echo "!! ckpt $1 iter $n < $2 — resume mis-resolved, ABORT"; exit 1; }; }

echo "===== v10 speed curriculum 0.60->0.75->0.90->1.00, budget 9500  $(date -Is) ====="

echo "===== STAGE 1/4  tempo 0.60x, 0-20 ms, drift<0.8 m, 2500 iters  $(date -Is) ====="
if S1=$(resolve s1) && [ "$(echo "$S1" | sed 's/.*model_//; s/\.pt$//')" -ge 2400 ] 2>/dev/null; then
  echo "  stage 1 already complete ($S1) — skipping"
else
G1_SLOWDOWN=1.6667 G1_CMD_DELAY_MAX_LAG=4  G1_OBS_DELAY_MAX_LAG=1 G1_DRIFT_TERM_M=0.8 \
  "$PY" "$ENTRY" "$TASK" "${COMMON[@]}" --env.commands.motion.motion-file "$M060" \
    --agent.max-iterations 2500 --agent.run-name "${RUN}-s1"
fi

echo "===== STAGE 2/4  tempo 0.75x, 0-40 ms, drift<0.6 m, +2500 (resume s1)  $(date -Is) ====="
if S2=$(resolve s2) && [ "$(echo "$S2" | sed 's/.*model_//; s/\.pt$//')" -ge 4900 ] 2>/dev/null; then
  echo "  stage 2 already complete ($S2) — skipping"
else
read -r R1 C1 <<< "$(resolve s1)"; echo "  resume run=$R1 ckpt=$C1"; assert_iter "$C1" 2400
G1_SLOWDOWN=1.3333 G1_CMD_DELAY_MAX_LAG=8  G1_OBS_DELAY_MAX_LAG=2 G1_DRIFT_TERM_M=0.6 \
  "$PY" "$ENTRY" "$TASK" "${COMMON[@]}" --env.commands.motion.motion-file "$M075" \
    --agent.max-iterations 2500 --agent.run-name "${RUN}-s2" \
    --agent.resume True --agent.load-run "$R1" --agent.load-checkpoint "$C1"
fi

echo "===== STAGE 3/4  tempo 0.90x, 0-50 ms, drift<0.6 m, +2250 (resume s2)  $(date -Is) ====="
if S3=$(resolve s3) && [ "$(echo "$S3" | sed 's/.*model_//; s/\.pt$//')" -ge 7150 ] 2>/dev/null; then
  echo "  stage 3 already complete ($S3) — skipping"
else
read -r R2 C2 <<< "$(resolve s2)"; echo "  resume run=$R2 ckpt=$C2"; assert_iter "$C2" 4900
G1_SLOWDOWN=1.1111 G1_CMD_DELAY_MAX_LAG=10 G1_OBS_DELAY_MAX_LAG=2 G1_DRIFT_TERM_M=0.6 \
  "$PY" "$ENTRY" "$TASK" "${COMMON[@]}" --env.commands.motion.motion-file "$M090" \
    --agent.max-iterations 2250 --agent.run-name "${RUN}-s3" \
    --agent.resume True --agent.load-run "$R2" --agent.load-checkpoint "$C2"
fi

echo "===== STAGE 4/4  tempo 1.00x NATIVE, 0-60 ms, drift<0.8 m, +2250 (resume s3)  $(date -Is) ====="
echo "===== (ends at 9500 TOTAL — do NOT extend; late-stage collapse pattern v7/v9) ====="
if S4=$(resolve s4) && [ "$(echo "$S4" | sed 's/.*model_//; s/\.pt$//')" -ge 9400 ] 2>/dev/null; then
  echo "  stage 4 already complete ($S4) — skipping straight to verify"
else
read -r R3 C3 <<< "$(resolve s3)"; echo "  resume run=$R3 ckpt=$C3"; assert_iter "$C3" 7150
G1_SLOWDOWN=1.0 G1_CMD_DELAY_MAX_LAG=12 G1_OBS_DELAY_MAX_LAG=3 G1_DRIFT_TERM_M=0.8 \
  "$PY" "$ENTRY" "$TASK" "${COMMON[@]}" --env.commands.motion.motion-file "$M100" \
    --agent.max-iterations 2250 --agent.run-name "${RUN}-s4" \
    --agent.resume True --agent.load-run "$R3" --agent.load-checkpoint "$C3"
fi

echo "===== VERIFY CHAIN (native npz, correct task, honest gate)  $(date -Is) ====="
export MUJOCO_GL=egl                # verify only — never during training (Warp CUDA clash)
export G1_DRIFT_TERM_M=999          # measure drift, don't clip it (comparable to v5-v9 gates)
export G1_SLOWDOWN=1.0              # native clock for the gate
# Calibrated ankle p95 bars (decision log 2026-07-20): nominal 22 / worst 25 Nm.
export G1_GATE_ANKLE_P95_NOMINAL_NM=22
export G1_GATE_ANKLE_P95_WORST_NM=25
read -r R4 C4 <<< "$(resolve s4)"; assert_iter "$C4" 9400
S4DIR="$LOGDIR/$R4"; mkdir -p "$EXP"
echo "  final stage run dir: $S4DIR (last ckpt $C4)"

echo "  screening last 6 checkpoints (v10 task, 64 envs)..."
"$PY" "$NB/cloud/pick_checkpoint.py" --python "$PY" \
    --gap-check "$NB/cloud/sim_gap_check.py" --rundir "$S4DIR" \
    --motion-file "$M100" --last 6 --num-envs 64 --workdir "$EXP/screen" \
    --task "$TASK" --task-module "$TM" \
    > "$EXP/pick.log" 2>&1 || true
cat "$EXP/pick.log"
if grep -q "every screen failed" "$EXP/pick.log"; then
  echo "!! all screens failed — refusing the blind last-ckpt fallback"; exit 1
fi
CKPT=$(grep '^WINNER ' "$EXP/pick.log" | tail -1 | sed 's/^WINNER //')
[ -f "$CKPT" ] || { echo "!! picker produced no winner"; exit 1; }
echo "  SELECTED ckpt: $CKPT"

# re-export the PICKED checkpoint (NOT mjlab's auto-export of the last one)
"$PY" "$NB/cloud/export_ckpt_onnx.py" --checkpoint "$CKPT" --motion-file "$M100" \
    --out-dir "$EXP" --task "$TASK" --task-module "$TM"
"$PY" "$NB/cloud/sim_gap_check.py" --checkpoint "$CKPT" --motion-file "$M100" \
    --task "$TASK" --task-module "$TM" --num-envs 128 --output-file "$EXP/gap.json"

# calibration-anchor line: score the deployed anchor policy.onnx on the SAME
# harness/motion (cheap: only the 2 gated conditions) so the absolute bars can
# be read against a known-deployed-quality reference. Skipped if no anchor onnx.
ANCHOR_ONNX=${G1_ANCHOR_ONNX:-$NB/exports/anchor/policy.onnx}
if [ -f "$ANCHOR_ONNX" ]; then
  "$PY" "$NB/cloud/sim_gap_check.py" --onnx "$ANCHOR_ONNX" --motion-file "$M100" \
      --task "$TASK" --task-module "$TM" --num-envs 64 \
      --only nominal,delay40ms_push --output-file "$EXP/gap_anchor_calib.json" \
    || echo "  !! anchor calibration line failed (informational only)"
else
  echo "  (no anchor onnx at $ANCHOR_ONNX — set G1_ANCHOR_ONNX to add the calibration line)"
fi

for S in 90001 90011 90021; do
  "$PY" "$NB/cloud/heldout_eval.py" --checkpoint "$CKPT" --motion-file "$M100" \
      --task "$TASK" --task-module "$TM" --seed "$S" --num-envs 256 \
      --output-file "$EXP/heldout_${S}.json" \
    || echo "  !! heldout seed $S failed (gap.json is the hard gate)"
done
echo "===== DONE $(date -Is) — gap.json + heldout in $EXP (selected $CKPT) ====="
echo "  Gate bars: ankle p95 <=22 nominal / <=25 worst (2026-07-20 calibration:"
echo "  real robot measured 15-19 Nm p95 at NATIVE tempo; ~21 Nm sim residual is"
echo "  stabilization effort, not choreography). Other bars unchanged: nominal"
echo "  survival >=99%, drift episode_max_p95 <=1.5 m, 40ms+push survival >=95%. Also judge vs the"
echo "  calibrated anchor row (experiments/REGISTRY.md): anchor == ~70% IRL quality."
echo "  Beat v8 at NATIVE tempo (99.2% nominal / 59.4% @40ms+push; drift stated as max_m:"
echo "  v8 4.31 m vs v11 3.19 m — the live gate is episode_max_p95 <=1.5 m, a different"
echo "  reduction, so do NOT compare v8's 4.31 max_m against the p95 bar)."
echo "  Then pull to laptop, sign, DELETE THE BOX."
