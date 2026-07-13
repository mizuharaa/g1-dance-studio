#!/usr/bin/env bash
# ON THE BOX — run inside tmux:  tmux new -s train ; bash retrain_v5_box.sh
# Converts the clean Thriller CSV -> training npz, preflights the curriculum resume flags,
# then runs the 3-stage v5 latency curriculum + the verify chain (~5 h). Idempotent-ish.
set -euo pipefail
export NB=${NB:-/workspace/notebook-data}
PY=$NB/envs/mjlab/bin/python
TASK=Mjlab-Tracking-Flat-Unitree-G1-S2R-V5
CSV=$NB/motions/thriller_g1_clean.csv
NPZ=$NB/motions/thriller_clean.npz
cd "$NB"

[ -f "$CSV" ] || { echo "MISSING $CSV — run retrain_send.sh from the laptop first"; exit 1; }

echo "== 1/3  CSV -> npz  (wandb-upload rc may be nonzero offline — that's fine) =="
$PY "$NB/repos/mjlab/src/mjlab/scripts/csv_to_npz.py" \
    --input-file "$CSV" --output-name thriller_clean --input-fps 30 --output-fps 50 || true
cp /tmp/motion.npz "$NPZ"
echo "   npz: $(ls -la "$NPZ")"

echo "== 2/3  preflight: curriculum RESUME flags (must list e.g. --agent.resume / --agent.load-run) =="
if $PY "$NB/cloud/train_sim2real_v5.py" "$TASK" --help 2>/dev/null | grep -i resume; then
  echo "   resume flags found — OK to run the curriculum."
else
  echo "   !! NO resume flags found. STOP: stages 2/3 would restart from scratch (wasted 5 h)."
  echo "   Fix train_v5_curriculum.sh's resume_args() to this mjlab's real flags, then re-run."
  exit 1
fi

echo "== 3/3  TRAIN  (3 latency stages 0-20 -> 0-50 -> 0-60 ms + export/gap/heldout, ~5 h) =="
MOTION="$NPZ" bash "$NB/cloud/train_v5_curriculum.sh"

echo
echo "==================== GATES (read above) ===================="
echo "  PASS if: gap.json survival OK at 40 ms+push  AND  nominal drift < 1 m  AND  heldout >= 99% (3 seeds)."
echo "  Then on the LAPTOP:  bash scripts/retrain_pull.sh <IP> <PORT>"
echo "  Then DELETE this box in the GreenNode console (stop still bills)."
