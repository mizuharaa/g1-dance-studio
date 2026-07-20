#!/usr/bin/env bash
# Attempt 7 (v10) — one-command box orchestrator. NATIVE-tempo speed curriculum.
#
# INPUT MOTION: the native grounded CSV — NOT the adaptive/1.8x slowed variants.
#   Default: $NB/motions/thriller_g1_grounded.csv  (override: G1_MOTION_CSV=...)
#   NOTE: the orchestrator may swap in a RE-PREPPED motion (the 2026-07-20
#   choreography-guard cleaning fix means the next extraction keeps the dance
#   hits the old rejector erased) — this script only reads G1_MOTION_CSV, so a
#   swap is a one-env-var change, no edits here.
#
# The box generates the 4 tempo variants itself (retime_motion.py: cubic +
# quaternion slerp, no frame duplication) and converts each to a 50 fps npz, so
# every stage trains the SAME choreography, only the clock differs.
#
# Push first (laptop):
#   scp -P <PORT> -i .secrets/greennode_rsa \
#     data/motions/thriller/thriller_g1_grounded.csv \
#     root@<IP>:/workspace/notebook-data/motions/
#   scp -P <PORT> -i .secrets/greennode_rsa cloud/{sim2real_task_v10.py,sim2real_task_v8.py,\
#     sim2real_task_v7.py,sim2real_task_v6.py,sim2real_task_v5.py,sim2real_task.py,\
#     retime_motion.py,train_v10_curriculum.sh,run_attempt7.sh,pick_checkpoint.py,\
#     sim_gap_check.py,heldout_eval.py,export_ckpt_onnx.py} \
#     root@<IP>:/workspace/notebook-data/cloud/
# Then on the box:  nohup bash $NB/cloud/run_attempt7.sh > $NB/attempt7.out 2>&1 &
#
# $$$ REMINDER: GreenNode bills creation -> deletion. The full run is ~4 stages
# x ~2500 iters + verify. Check progress daily; when exports are pulled and
# signed, DELETE THE BOX. Never leave it idle overnight "just in case".
set -uo pipefail
NB=${NB:-/workspace/notebook-data}
PY=$NB/envs/mjlab/bin/python
CSV=${G1_MOTION_CSV:-$NB/motions/thriller_g1_grounded.csv}
CSV2NPZ=$NB/repos/mjlab/src/mjlab/scripts/csv_to_npz.py
[ -f "$NB/.wandb_key" ] && export WANDB_API_KEY=$(tr -d '[:space:]' < "$NB/.wandb_key")

cd "$NB"   # pin CWD: mjlab writes logs/ relative to it (attempt-6 stage-2 resume lesson)
say() { echo "== $* == $(date -Is)"; }
die() { echo "!! $*"; exit 1; }

say "preflight"
[ -f "$CSV" ] || die "motion CSV missing: $CSV — push it from the laptop (see header)"
head -1 "$CSV" | awk -F, 'NF != 36 { exit 1 }' || die "CSV is not 36-column (3 pos + 4 quat + 29 joints)"
nvidia-smi --query-gpu=utilization.gpu,memory.total --format=csv,noheader || die "no GPU"
FREE_GB=$(df -BG --output=avail "$NB" | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -ge 30 ] || die "only ${FREE_GB} GB free on $NB — need >=30 for 4 stages of ckpts"
[ -n "${WANDB_API_KEY:-}" ] || echo "  (warn: no WANDB_API_KEY — training logs stay local)"
"$PY" -c "import scipy" 2>/dev/null || { say "installing scipy for the retimer"; "$PY" -m pip install -q scipy || echo "  (scipy install failed — retimer falls back to numpy Catmull-Rom)"; }

say "tempo variants: retime native CSV (0.60 / 0.75 / 0.90 / 1.00)"
declare -A SPEED=( [060]=0.60 [075]=0.75 [090]=0.90 [100]=1.00 )
for k in 060 075 090 100; do
  OUT=$NB/motions/thriller_v10_${k}.csv
  if [ ! -f "$OUT" ]; then
    "$PY" "$NB/cloud/retime_motion.py" --input "$CSV" --output "$OUT" \
        --speed "${SPEED[$k]}" --check-json "$NB/motions/thriller_v10_${k}_check.json" \
      || die "retime ${SPEED[$k]}x FAILED its self-check"
  fi
done

say "convert each tempo CSV -> npz (30 -> 50 fps; velocities/FK recomputed per tempo)"
NATIVE_FRAMES=$(wc -l < "$CSV")
for k in 060 075 090 100; do
  NPZ=$NB/motions/thriller_v10_${k}.npz
  if [ ! -f "$NPZ" ]; then
    MUJOCO_GL=egl "$PY" "$CSV2NPZ" --input-file "$NB/motions/thriller_v10_${k}.csv" \
        --output-name "thriller_v10_${k}" --input-fps 30 --output-fps 50 || true
    [ -f /tmp/motion.npz ] && cp /tmp/motion.npz "$NPZ"
  fi
  [ -f "$NPZ" ] || die "npz still absent after conversion: $NPZ"
  FRAMES=$("$PY" - "$NPZ" << 'EOF'
import numpy as np, sys
d = np.load(sys.argv[1], allow_pickle=True)
print(int(d["joint_pos"].shape[0]))
EOF
)
  # expected 50fps frames ~ native_csv_rows / speed * 50/30 (+-2%)
  EXP=$(python3 -c "print(int($NATIVE_FRAMES / ${SPEED[$k]} * 50 / 30))")
  LO=$((EXP * 98 / 100)); HI=$((EXP * 102 / 100))
  [ "$FRAMES" -ge "$LO" ] && [ "$FRAMES" -le "$HI" ] \
    || die "npz $k frame count $FRAMES outside expected $LO-$HI (retime/convert mismatch)"
  echo "  thriller_v10_${k}.npz: $FRAMES frames (~$((FRAMES / 50))s @50fps) OK"
done

say "selfcheck (v10 task, stage-1 clock 0.60x -> G1_SLOWDOWN=1.6667)"
G1_SLOWDOWN=1.6667 G1_OBS_HISTORY=5 G1_PHASE_OBS=${G1_PHASE_OBS:-0} \
  "$PY" "$NB/cloud/sim2real_task_v10.py" --selfcheck || die "selfcheck FAIL"

say "64-env GPU smoke test (stage-1 conditions, 3 iters)"
G1_SLOWDOWN=1.6667 G1_OBS_HISTORY=5 G1_PHASE_OBS=${G1_PHASE_OBS:-0} \
G1_CMD_DELAY_MAX_LAG=4 G1_OBS_DELAY_MAX_LAG=1 G1_DRIFT_TERM_M=0.8 \
  "$PY" "$NB/cloud/sim2real_task_v10.py" Mjlab-Tracking-Flat-Unitree-G1-S2R-V10 \
    --env.scene.num-envs 64 --env.commands.motion.motion-file "$NB/motions/thriller_v10_060.npz" \
    --agent.max-iterations 3 --agent.run-name smoketest-v10 || die "smoke test FAIL"

say "LAUNCH v10 speed curriculum (detached inner call — this script IS the detached wrapper)"
M060=$NB/motions/thriller_v10_060.npz \
M075=$NB/motions/thriller_v10_075.npz \
M090=$NB/motions/thriller_v10_090.npz \
M100=$NB/motions/thriller_v10_100.npz \
  bash "$NB/cloud/train_v10_curriculum.sh"
say "run_attempt7 DONE — pull exports, judge vs the calibrated bars, sign, DELETE THE BOX"
