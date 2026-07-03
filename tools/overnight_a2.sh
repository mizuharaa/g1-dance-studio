#!/usr/bin/env bash
# Overnight autopilot for Thriller attempt-2: wait for convergence, export, held-out
# gate, sign verdict. Survives session death (runs detached). Leaves a result marker
# at data/policies/thriller_a2/RESULT.txt for the next session to read and stage.
#
# Does NOT auto-promote to ground policy (that needs judgment + main notify) — it does
# the expensive mechanical work so a resumed session only reads the verdict and decides.
set -uo pipefail
KEY="$HOME/g1-dance/.secrets/greennode_ssh_key"
BOX="root@103.245.250.152"; PORT=46936
SSH="ssh -i $KEY -p $PORT -o ConnectTimeout=15 $BOX"
D=/workspace/notebook-data
OUT="$HOME/g1-dance/data/policies/thriller_a2"
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*" >> "$OUT/overnight.log"; }

log "overnight autopilot started; waiting for train-thriller-a2 to finish"

# 1. Wait for attempt-2 to reach a terminal state. Require the status file to say
#    done/failed, confirmed twice — a transient SSH/tmux blip must NOT trigger a
#    premature export (that bug fired the first run at iter 1500).
CONFIRM=0
while true; do
  ST=$($SSH "cat $D/jobs/train-thriller-a2.status.json 2>/dev/null" 2>/dev/null)
  if echo "$ST" | grep -qE '"state":"(done|failed)"'; then
    CONFIRM=$((CONFIRM+1)); log "a2 terminal signal $CONFIRM/2: $ST"
    [ "$CONFIRM" -ge 2 ] && break
  else
    CONFIRM=0
  fi
  sleep 120
done
OUT="$HOME/g1-dance/data/policies/thriller_a2_final"; mkdir -p "$OUT"

# 2. Pick the latest checkpoint of attempt-2.
CKPT=$($SSH "ls -t $D/cloud/logs/rsl_rl/g1_tracking/*_train-thriller-a2/model_*.pt 2>/dev/null | head -1")
[ -n "$CKPT" ] || { log "NO a2 checkpoint found — abort"; echo "FAIL: no checkpoint" > "$OUT/RESULT.txt"; exit 1; }
ITER=$(basename "$CKPT" | sed 's/model_//;s/.pt//')
log "best a2 checkpoint: iter $ITER"

# 3. Export ONNX (correct per-joint action_scale is intrinsic to the mjlab exporter).
$SSH "export MUJOCO_GL=egl WANDB_MODE=disabled; cd /tmp && $D/envs/mjlab/bin/python $D/cloud/export_policy.py '$CKPT' $D/motions/thriller_deploy.npz $D/exports/thriller_a2" >> "$OUT/overnight.log" 2>&1
$SSH "test -f $D/exports/thriller_a2/policy.onnx" || { log "export FAILED"; echo "FAIL: export" > "$OUT/RESULT.txt"; exit 1; }

# 4. Held-out gate on the DEPLOYABLE motion (256 envs, held-out seed, nominal+push).
$SSH "export MUJOCO_GL=egl WANDB_MODE=disabled; cd /tmp && $D/envs/mjlab/bin/python $D/cloud/heldout_eval.py Mjlab-Tracking-Flat-Unitree-G1 --checkpoint '$CKPT' --motion-file $D/motions/thriller_deploy.npz --num-envs 256 --seed 90001 --output-file $D/exports/thriller_a2/heldout_eval.json" >> "$OUT/overnight.log" 2>&1

# 5. Pull artifacts to laptop.
scp -q -i "$KEY" -P "$PORT" "$BOX:$D/exports/thriller_a2/policy.onnx" "$BOX:$D/exports/thriller_a2/heldout_eval.json" "$CKPT" "$OUT/" 2>>"$OUT/overnight.log"

# 6. Sign the verdict locally (binds policy + deployable motion shas).
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate g1dance
python -m pipeline.mjlab_verify \
  --eval-json "$OUT/heldout_eval.json" \
  --policy "$OUT/policy.onnx" \
  --motion "$HOME/g1-dance/data/policies/thriller/thriller_deploy.csv" \
  --out "$OUT/heldout_verdict.json" >> "$OUT/overnight.log" 2>&1

# 7. Extract survival % and write the RESULT marker for the next session.
python - "$OUT/heldout_eval.json" "$OUT/heldout_verdict.json" "$OUT/RESULT.txt" "$ITER" <<'PY'
import json, sys
ev = json.load(open(sys.argv[1]))
try: vd = json.load(open(sys.argv[2]))
except Exception: vd = {}
conds = ev.get("conditions", ev)
def rate(c):
    x = conds.get(c, {}) if isinstance(conds, dict) else {}
    return x.get("success_rate")
nom = rate("nominal") or (ev.get("nominal") or {}).get("success_rate")
push = rate("push") or (ev.get("push") or {}).get("success_rate")
worst = min([r for r in (nom, push) if r is not None] or [0])
verdict = vd.get("verdict", "?")
lines = [
  f"attempt-2 iter {sys.argv[4]}",
  f"nominal_survival={nom}",
  f"push_survival={push}",
  f"worst={worst}  ({worst*100:.1f}%)" if worst else "worst=?",
  f"signed_verdict={verdict}",
  f"GROUND_READY={'YES' if worst and worst>=0.99 else 'NO'} (bar=0.99)",
  "next: if GROUND_READY=YES, generate thriller_deploy for a2, stage at data/policies/thriller/, tell main to rebuild --full bundle. Keep attempt-1 as gantry fallback. Else report honestly (attempt 2 of <=3).",
]
open(sys.argv[3], "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
log "DONE — see RESULT.txt"
