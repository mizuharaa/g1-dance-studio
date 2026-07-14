#!/usr/bin/env bash
# ATTEMPT 3 — one-command, cost-minimal box orchestrator.
#
# Cost model: the GreenNode box bills creation->deletion (~18,170 VND/h), NOT
# just when busy. So the whole design is: do everything free on the laptop,
# then ONE short unattended box session, delete immediately. This is TRAIN-ONLY
# — Thriller's motion already exists, so NO GVHMR / no 3.7 GB body-model push /
# no extract spend. GVHMR is only needed to add a NEW dance from video.
#
# Run ON THE BOX, detached so an SSH drop can't kill it:
#     cd $NB && setsid nohup bash cloud/run_attempt3.sh > attempt3.out 2>&1 &
#     tail -f $NB/attempt3.out        # or the laptop dashboard / retrain monitor
#
# It PREFLIGHTS (fails in seconds, before any GPU spend) then runs the v6
# curriculum + verify. On success it prints the gate verdict, the accrued cost,
# and the exact DELETE step (deletion is what stops billing — Stop still bills).
set -uo pipefail

export NB=${NB:-/workspace/notebook-data}
PY=$NB/envs/mjlab/bin/python
ENTRY=$NB/cloud/sim2real_task_v6.py
TASK=Mjlab-Tracking-Flat-Unitree-G1-S2R-V6
# motion: prefer an existing npz; else convert the clean CSV the laptop sent.
NPZ=${NPZ:-$NB/motions/thriller_clean.npz}
CSV=${CSV:-$NB/motions/thriller_g1_clean.csv}

say() { printf '\n\033[1m== %s ==\033[0m %s\n' "$1" "$(date -Is)"; }
die() { printf '\n\033[31m!! PREFLIGHT FAIL: %s\033[0m\n' "$1"; exit 1; }

say "PREFLIGHT (no GPU spend until all pass)"

# 0. python + mjlab venv
[ -x "$PY" ] || die "mjlab venv missing at $PY — run cloud/00_bootstrap.sh + mjlab install first"
[ -f "$ENTRY" ] || die "recipe $ENTRY missing — push cloud/ to the box"

# 1. motion npz present + non-trivial; else build it from the clean CSV.
if [ ! -f "$NPZ" ]; then
  [ -f "$CSV" ] || die "neither $NPZ nor $CSV present — send the de-glitched Thriller CSV from the laptop"
  say "convert CSV -> npz (motion npz absent)"
  "$PY" "$NB/repos/mjlab/src/mjlab/scripts/csv_to_npz.py" \
      --input-file "$CSV" --output-name "$(basename "${NPZ%.npz}")" \
      --input-fps 30 --output-fps 50 || true
  [ -f /tmp/motion.npz ] && cp /tmp/motion.npz "$NPZ"
fi
[ -f "$NPZ" ] || die "motion npz still absent after convert"
FRAMES=$("$PY" - "$NPZ" <<'PY'
import numpy as np, sys
try:
    d = np.load(sys.argv[1]); k = "fpos" if "fpos" in d else list(d.keys())[0]
    print(len(d[k]))
except Exception as e:
    print(0)
PY
)
[ "${FRAMES:-0}" -ge 100 ] || die "motion npz looks empty/short ($FRAMES frames)"
echo "  motion: $NPZ ($FRAMES frames)"

# 2. GPU actually free (a fresh 4090 idles ~0%; >20% = someone/thing is using it).
if command -v nvidia-smi >/dev/null; then
  UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "  GPU util: ${UTIL}%"
  [ "${UTIL:-0}" -lt 20 ] || die "GPU busy (${UTIL}%) — another job is running; not stacking a 5 h train on it"
else
  die "no nvidia-smi — not a GPU box"
fi

# 3. disk headroom (checkpoints + logs need a few GB)
AVAIL_GB=$(df -BG "$NB" | awk 'NR==2{gsub("G","",$4); print $4}')
[ "${AVAIL_GB:-0}" -ge 5 ] || die "low disk on $NB (${AVAIL_GB}G free) — clear old logs/exports first"
echo "  disk free: ${AVAIL_GB}G"

# 4. RECIPE selfcheck — asserts every reward/termination key v6 depends on is
#    registered on THIS mjlab (catches an API rename in seconds, not after 5 h).
say "recipe selfcheck"
"$PY" "$ENTRY" --selfcheck || die "v6 --selfcheck failed — fix the recipe before training"

# 5. resume flags exist (stages 2/3 resume; without them they'd silently restart)
say "resume-flag check"
"$PY" "$ENTRY" "$TASK" --help 2>/dev/null | grep -iq -- "--agent.resume" \
  || die "no --agent.resume flag on this mjlab — fix train_v6_curriculum.sh resume args"
echo "  --agent.resume present"

# 6. wandb key (offline is fine; just warn)
[ -f "$NB/.wandb_key" ] && echo "  wandb key present" || echo "  (no wandb key — training logs stay local, fine)"

say "PREFLIGHT PASSED — starting v6 curriculum (~5 h)"
MOTION="$NPZ" bash "$NB/cloud/train_v6_curriculum.sh"
RC=$?

RUN=$(ls -dt "$NB"/exports/train-thriller_v6sk-* 2>/dev/null | head -1)
say "RESULT (rc=$RC)"
if [ -f "$RUN/gap.json" ]; then
  "$PY" - "$RUN/gap.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
g = d.get("gate", {})
print("  gate pass:", g.get("pass"))
for k, v in g.get("checks", {}).items():
    print(("   PASS " if v else "   FAIL ") + k)
n = d.get("conditions", {}).get("nominal", {})
dr = n.get("drift", {})
print(f"  nominal survival: {n.get('success_rate')}  drift max: {dr.get('max_m')} m")
PY
  echo "  artifacts: $RUN"
else
  echo "  no gap.json — training or verify did not complete; see the stage logs above."
fi

cat <<EOF

======================= COST / TEARDOWN =======================
Billing runs creation->deletion. Training is done; the box is now idle but STILL
BILLING. To stop it you must DELETE the instance (Stop still bills):

  1. Pull artifacts to the laptop:   bash scripts/retrain_pull.sh <IP> <PORT>
  2. Sign the verdict:               python pipeline/mjlab_verify.py <exports/...>
  3. DELETE the box in the GreenNode console.  <-- this is what stops the meter.

If the gate PASSED: sign + register the dance, then gantry (robot repaired first).
If it FAILED: read gap.json above — this was attempt 3 of the <=3 budget.
===============================================================
EOF
exit $RC
