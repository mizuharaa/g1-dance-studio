#!/usr/bin/env bash
# Attempt 6 (v9) — one-command box orchestrator. Adaptive-warp motion, v8 recipe.
#
# Motion prep is LAPTOP-side this time: the adaptive CSV + scorecard are COMMITTED
# (data/motions/thriller/thriller_g1_grounded_adaptive.csv, 0% over headroom).
# The box only converts CSV -> npz (30 -> 50 fps) and trains. Push first:
#   scp -P <PORT> -i .secrets/greennode_rsa \
#     data/motions/thriller/thriller_g1_grounded_adaptive.csv \
#     data/motions/thriller/thriller_g1_grounded_adaptive_scorecard.json \
#     root@<IP>:/workspace/notebook-data/motions/
#   scp -P <PORT> -i .secrets/greennode_rsa cloud/{sim2real_task_v8.py,train_v9_curriculum.sh,\
#     run_attempt6.sh,pick_checkpoint.py,sim_gap_check.py,heldout_eval.py,export_ckpt_onnx.py} \
#     root@<IP>:/workspace/notebook-data/cloud/
# Then on the box:  nohup bash $NB/cloud/run_attempt6.sh > $NB/attempt6.out 2>&1 &
set -uo pipefail
NB=${NB:-/workspace/notebook-data}
PY=$NB/envs/mjlab/bin/python
CSV=$NB/motions/thriller_g1_grounded_adaptive.csv
SCORECARD=$NB/motions/thriller_g1_grounded_adaptive_scorecard.json
NPZ=$NB/motions/thriller_grounded_adaptive.npz
CSV2NPZ=$NB/repos/mjlab/src/mjlab/scripts/csv_to_npz.py
[ -f "$NB/.wandb_key" ] && export WANDB_API_KEY=$(tr -d '[:space:]' < "$NB/.wandb_key")

cd "$NB"   # pin CWD: mjlab writes logs/ relative to it (attempt-6 stage-2 resume lesson)
say() { echo "== $* == $(date -Is)"; }
die() { echo "!! $*"; exit 1; }

say "preflight"
[ -f "$CSV" ] || die "adaptive CSV missing — push it from the laptop (see header)"
[ -f "$SCORECARD" ] || die "adaptive scorecard missing (carries the time_map)"
python3 - "$SCORECARD" << 'EOF' || die "scorecard says the motion is NOT fully feasible"
import json, sys
s = json.load(open(sys.argv[1]))
assert s["final"]["ankle_over_headroom_pct"] == 0, s["final"]
print(f"  motion feasible: max {s['final']['ankle_tau_max_nm']} Nm, "
      f"{s['duration_ratio']}x duration, native-speed {s['native_speed_fraction']*100:.0f}%")
EOF
# waist windows come from the scorecard time_map — computed here so they can never
# drift out of sync with the motion actually trained on
WINDOWS=$(python3 - "$SCORECARD" << 'EOF'
import json, sys
import bisect
s = json.load(open(sys.argv[1]))
tm = s["time_map"]; src, wrp = tm["source_t"], tm["warped_t"]
def interp(x):
    i = min(max(bisect.bisect_left(src, x), 1), len(src) - 1)
    a, b = src[i-1], src[i]
    f = 0.0 if b == a else (x - a) / (b - a)
    return wrp[i-1] + f * (wrp[i] - wrp[i-1])
print(",".join(f"{interp(a):.2f}-{interp(b):.2f}" for a, b in ((13.0, 18.0), (25.0, 36.0))))
EOF
)
echo "  warped waist windows: $WINDOWS"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader || die "no GPU"

if [ ! -f "$NPZ" ]; then
  say "convert adaptive CSV -> npz (30 -> 50 fps)"
  MUJOCO_GL=egl "$PY" "$CSV2NPZ" --input-file "$CSV" \
      --output-name "$(basename "${NPZ%.npz}")" --input-fps 30 --output-fps 50 || true
  [ -f /tmp/motion.npz ] && cp /tmp/motion.npz "$NPZ"
fi
[ -f "$NPZ" ] || die "motion npz still absent after conversion"
FRAMES=$("$PY" - "$NPZ" << 'EOF'
import numpy as np, sys
d = np.load(sys.argv[1], allow_pickle=True)
print(int(d["joint_pos"].shape[0]))
EOF
)
[ "${FRAMES:-0}" -ge 100 ] || die "motion npz looks empty/short ($FRAMES frames)"
echo "  npz frames: $FRAMES (~$((FRAMES/50))s @50fps)"

say "selfcheck (v8 task, adaptive windows)"
G1_SLOWDOWN=1.0 G1_OBS_HISTORY=5 G1_WAIST_WINDOWS="$WINDOWS" \
  "$PY" "$NB/cloud/sim2real_task_v8.py" --selfcheck || die "selfcheck FAIL"

say "64-env GPU smoke test"
G1_SLOWDOWN=1.0 G1_OBS_HISTORY=5 G1_WAIST_WINDOWS="$WINDOWS" \
  "$PY" "$NB/cloud/sim2real_task_v8.py" Mjlab-Tracking-Flat-Unitree-G1-S2R-V8 \
    --env.scene.num-envs 64 --env.commands.motion.motion-file "$NPZ" \
    --agent.max-iterations 3 --agent.run-name smoketest-v9 || die "smoke test FAIL"

say "LAUNCH v9 curriculum (detached inner call — this script IS the detached wrapper)"
MOTION="$NPZ" G1_WAIST_WINDOWS="$WINDOWS" bash "$NB/cloud/train_v9_curriculum.sh"
say "run_attempt6 DONE — pull exports, judge vs calibrated bar, sign, DELETE THE BOX"
