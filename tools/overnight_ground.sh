#!/usr/bin/env bash
# Autopilot for the OBS-RESTRICTED (No-State-Estimation) ground Thriller policy:
# wait for train-thriller-ground to finish, export ONNX (154-dim obs), held-out gate
# on the No-State-Estimation task, sign verdict, pull to laptop. Detached; survives
# session death. Result -> data/policies/thriller_ground/RESULT.txt.
# Ground deployment is a SEPARATE human-supervised staged (tethered-first) session.
set -uo pipefail
KEY="$HOME/g1-dance/.secrets/greennode_ssh_key"
BOX="root@103.245.250.152"; PORT=46936
SSH="ssh -i $KEY -p $PORT -o ConnectTimeout=15 -o BatchMode=yes $BOX"
D=/workspace/notebook-data
TASK="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"
OUT="$HOME/g1-dance/data/policies/thriller_ground"; mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*" >> "$OUT/overnight.log"; }

log "ground autopilot started; waiting for train-thriller-ground (No-State-Estimation)"

# 1. Wait for the job to reach a terminal state, confirmed twice (avoid SSH-blip false trigger).
CONFIRM=0
while true; do
  ST=$($SSH "cat $D/jobs/train-thriller-ground.status.json 2>/dev/null" 2>/dev/null)
  if echo "$ST" | grep -qE '"state":"(done|failed)"'; then
    CONFIRM=$((CONFIRM+1)); log "terminal signal $CONFIRM/2: $ST"
    [ "$CONFIRM" -ge 2 ] && break
  else CONFIRM=0; fi
  sleep 120
done

# 2. Latest checkpoint.
CKPT=$($SSH "ls -t $D/cloud/logs/rsl_rl/g1_tracking/*_train-thriller-ground/model_*.pt 2>/dev/null | head -1")
[ -n "$CKPT" ] || { log "NO checkpoint — abort"; echo "FAIL: no checkpoint" > "$OUT/RESULT.txt"; exit 1; }
ITER=$(basename "$CKPT" | sed 's/model_//;s/.pt//')
log "checkpoint: iter $ITER"

# 3. Export ONNX (obs layout is intrinsic to the No-State-Estimation env cfg -> 154-dim).
$SSH "export MUJOCO_GL=egl WANDB_MODE=disabled; cd /tmp && $D/envs/mjlab/bin/python $D/cloud/export_policy.py '$CKPT' $D/motions/thriller_deploy.npz $D/exports/thriller_ground" >> "$OUT/overnight.log" 2>&1
$SSH "test -f $D/exports/thriller_ground/policy.onnx" || { log "export FAILED"; echo "FAIL: export" > "$OUT/RESULT.txt"; exit 1; }

# 4. Held-out gate on the NO-STATE-ESTIMATION task (obs must match the policy).
$SSH "export MUJOCO_GL=egl WANDB_MODE=disabled; cd /tmp && $D/envs/mjlab/bin/python $D/cloud/heldout_eval.py $TASK --checkpoint '$CKPT' --motion-file $D/motions/thriller_deploy.npz --num-envs 256 --seed 90007 --output-file $D/exports/thriller_ground/heldout_eval.json" >> "$OUT/overnight.log" 2>&1

# 5. Pull artifacts.
scp -q -i "$KEY" -P "$PORT" "$BOX:$D/exports/thriller_ground/policy.onnx" "$BOX:$D/exports/thriller_ground/policy_meta.json" "$BOX:$D/exports/thriller_ground/heldout_eval.json" "$CKPT" "$OUT/" 2>>"$OUT/overnight.log"

# 6. Sign verdict (binds policy + deployable motion csv).
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate g1dance
python -m pipeline.mjlab_verify \
  --eval-json "$OUT/heldout_eval.json" --policy "$OUT/policy.onnx" \
  --motion "$HOME/g1-dance/data/policies/thriller/thriller_deploy.csv" \
  --out "$OUT/heldout_verdict.json" >> "$OUT/overnight.log" 2>&1

# 7. RESULT marker.
python - "$OUT/heldout_eval.json" "$OUT/heldout_verdict.json" "$OUT/RESULT.txt" "$ITER" <<'PY'
import json, sys
ev = json.load(open(sys.argv[1]))
try: vd = json.load(open(sys.argv[2]))
except Exception: vd = {}
conds = ev.get("conditions", ev)
rate = lambda c: (conds.get(c, {}) if isinstance(conds, dict) else {}).get("success_rate")
nom = rate("nominal") or (ev.get("nominal") or {}).get("success_rate")
push = rate("push") or (ev.get("push") or {}).get("success_rate")
worst = min([r for r in (nom, push) if r is not None] or [0])
lines = [
  f"GROUND (No-State-Estimation, 154-dim obs) iter {sys.argv[4]}",
  f"nominal_survival={nom}", f"push_survival={push}  (NOTE: push DR off in training config)",
  f"worst={worst}  ({worst*100:.1f}%)" if worst else "worst=?",
  f"signed_verdict={vd.get('verdict','?')}",
  f"SIM_READY={'YES' if nom and nom>=0.95 else 'NO'} (nominal>=0.95; push may be lower by design)",
  "next: HUMAN-SUPERVISED staged ground session (tethered-first). Update deploy_runtime build_obs to the 154-dim No-State-Estimation layout (drop base_lin_vel + motion_anchor_pos_b) per docs/ground_policy.md. NEVER autonomous on the ground.",
]
open(sys.argv[3], "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
PY
log "DONE — see RESULT.txt"
