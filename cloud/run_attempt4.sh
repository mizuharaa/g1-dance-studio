#!/usr/bin/env bash
# ATTEMPT 4 (v7) — one-command, cost-minimal box orchestrator. Train-only (Thriller
# motion exists). Launch ON THE BOX, detached, WITHOUT MUJOCO_GL (egl collides with
# Warp CUDA at the 4096-env reset; the curriculum sets egl only for the verify step):
#     cd $NB && setsid nohup bash cloud/run_attempt4.sh > attempt4.out 2>&1 &
set -uo pipefail

export NB=${NB:-/workspace/notebook-data}
PY=$NB/envs/mjlab/bin/python
ENTRY=$NB/cloud/sim2real_task_v7.py
TASK=Mjlab-Tracking-Flat-Unitree-G1-S2R-V7
NPZ=${NPZ:-$NB/motions/thriller_clean.npz}
CSV=${CSV:-$NB/motions/thriller_g1_clean.csv}

if [ -f "$NB/.wandb_key" ]; then
  export WANDB_API_KEY=$(tr -d '[:space:]' < "$NB/.wandb_key")
else
  export WANDB_MODE=offline
fi

say() { printf '\n\033[1m== %s ==\033[0m %s\n' "$1" "$(date -Is)"; }
die() { printf '\n\033[31m!! PREFLIGHT FAIL: %s\033[0m\n' "$1"; exit 1; }

say "PREFLIGHT (no GPU spend until all pass)"
[ -x "$PY" ] || die "mjlab venv missing at $PY"
[ -f "$ENTRY" ] || die "recipe $ENTRY missing — push cloud/"

if [ ! -f "$NPZ" ]; then
  [ -f "$CSV" ] || die "neither $NPZ nor $CSV present"
  say "convert CSV -> npz"
  "$PY" "$NB/repos/mjlab/src/mjlab/scripts/csv_to_npz.py" \
      --input-file "$CSV" --output-name "$(basename "${NPZ%.npz}")" --input-fps 30 --output-fps 50 || true
  [ -f /tmp/motion.npz ] && cp /tmp/motion.npz "$NPZ"
fi
[ -f "$NPZ" ] || die "motion npz still absent"
FRAMES=$("$PY" - "$NPZ" <<'PY'
import numpy as np, sys
try:
    d = np.load(sys.argv[1])
    print(d["joint_pos"].shape[0] if "joint_pos" in d.files else max((d[k].shape[0] for k in d.files if getattr(d[k],"ndim",0)>=2), default=0))
except Exception:
    print(0)
PY
)
[ "${FRAMES:-0}" -ge 100 ] || die "motion npz looks empty/short ($FRAMES frames)"
echo "  motion: $NPZ ($FRAMES frames)"

if command -v nvidia-smi >/dev/null; then
  UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "  GPU util: ${UTIL}%"
  [ "${UTIL:-0}" -lt 20 ] || die "GPU busy (${UTIL}%) — another job running"
else
  die "no nvidia-smi"
fi
AVAIL_GB=$(df -BG "$NB" | awk 'NR==2{gsub("G","",$4); print $4}')
[ "${AVAIL_GB:-0}" -ge 5 ] || die "low disk (${AVAIL_GB}G)"
echo "  disk free: ${AVAIL_GB}G"

say "recipe selfcheck"
"$PY" "$ENTRY" --selfcheck || die "v7 --selfcheck failed"

say "resume-flag check"
RESUME_HELP=$("$PY" "$ENTRY" "$TASK" --help 2>&1 || true)
case "$RESUME_HELP" in
  *"--agent.resume"*) echo "  --agent.resume present" ;;
  *) die "no --agent.resume flag on this mjlab" ;;
esac

say "PREFLIGHT PASSED — starting v7 curriculum (~2.8 h)"
MOTION="$NPZ" bash "$NB/cloud/train_v7_curriculum.sh"
RC=$?

RUN=$(ls -dt "$NB"/exports/train-thriller_v7ank-* 2>/dev/null | head -1)
say "RESULT (rc=$RC)"
if [ -f "$RUN/gap.json" ]; then
  "$PY" - "$RUN/gap.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); g = d.get("gate", {})
print("  gate pass:", g.get("pass"))
for k, v in g.get("checks", {}).items():
    print(("   PASS " if v else "   FAIL ") + k)
n = d.get("conditions", {}).get("nominal", {}); dr = n.get("drift", {})
ap = n.get("ankle_pitch", {})
print(f"  nominal survival: {n.get('success_rate')}  drift max: {dr.get('max_m')} m  ankle p95: {ap.get('p95_abs')}")
PY
  echo "  artifacts: $RUN"
else
  echo "  no gap.json — training/verify did not complete; see stage logs."
fi
cat <<EOF

======================= COST / TEARDOWN =======================
Box bills creation->deletion. Pull artifacts, then DELETE the instance to stop it.
  1. bash scripts/retrain_pull.sh <IP> <PORT>   2. sign   3. DELETE in console.
===============================================================
EOF
exit $RC
