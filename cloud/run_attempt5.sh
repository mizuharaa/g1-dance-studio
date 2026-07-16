#!/usr/bin/env bash
# ATTEMPT 5 (v8) — one-command, cost-minimal box orchestrator for the v-chain revamp.
# Layers Agent 0's asymmetric 154-dim no-state-estimation actor + Agent D's CANDIDATE A
# actuation deltas (1.8x feasible reference, velocity-honest ankle clamp, ankle
# soft-barrier, ankle action-rate, waist slack). Launch ON THE BOX, detached, WITHOUT
# MUJOCO_GL (egl collides with Warp CUDA at the 4096-env reset; the curriculum sets egl
# only for the verify step):
#     cd $NB && setsid nohup bash cloud/run_attempt5.sh > attempt5.out 2>&1 &
set -uo pipefail

export NB=${NB:-/workspace/notebook-data}
PY=$NB/envs/mjlab/bin/python
ENTRY=$NB/cloud/sim2real_task_v8.py
TASK=Mjlab-Tracking-Flat-Unitree-G1-S2R-V8

# --------------------------------------------------------------------------------------
# MOTION PREP CHAIN (v8) — the final training motion is BOTH slowed AND grounded.
#
#   ground(thriller_g1_clean.csv) -> thriller_g1_grounded.csv        [pipeline/grounding,
#                                                                      OWNED + COMMITTED]
#   repair 1.8x (tools/motion_repair.py --apply-factor $G1_SLOWDOWN)  -> _repaired csv
#   csv_to_npz (--input-fps 30 --output-fps 50)                       -> training npz
#
# WHY REGENERATE: the committed experiments/motion_feasibility/thriller_g1_repaired_1p8x.csv
# was repaired from the UN-grounded thriller_g1_clean.csv (verified: its scorecard
# "source" = data/motions/thriller/thriller_g1_clean.csv). Agent D's 1.8x is the right
# slowdown, but the FINAL motion must also be grounded (grounding fix landed in
# pipeline/grounding.py; grounded Thriller = data/motions/thriller/thriller_g1_grounded.csv).
# So we repair the GROUNDED csv here, not reuse the un-grounded repaired csv.
#
# grounding.py is OWNED/COMMITTED (do not modify): we consume its committed output
# thriller_g1_grounded.csv as the ground step. If that file is absent on the box, produce
# it on the laptop via pipeline/prep_motion.py and push it (grounding is not re-run here).
# G1_SLOWDOWN selects the factor (1.8 default; 2.0/2.5 fallbacks) end-to-end.
# --------------------------------------------------------------------------------------
export G1_SLOWDOWN=${G1_SLOWDOWN:-1.8}
SLOWTAG=$(printf '%s' "$G1_SLOWDOWN" | tr '.' 'p')   # 1.8 -> 1p8
GROUNDED=${GROUNDED:-$NB/motions/thriller_g1_grounded.csv}
REPAIRED=${REPAIRED:-$NB/motions/thriller_g1_grounded_repaired_${SLOWTAG}x.csv}
NPZ=${NPZ:-$NB/motions/thriller_grounded_repaired_${SLOWTAG}x.npz}
CSV2NPZ=$NB/repos/mjlab/src/mjlab/scripts/csv_to_npz.py
MOTION_REPAIR=$NB/tools/motion_repair.py

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

# --- motion prep: ground (committed) -> repair 1.8x -> csv_to_npz ---
if [ ! -f "$NPZ" ]; then
  [ -f "$GROUNDED" ] || die "grounded CSV $GROUNDED absent — produce via pipeline/prep_motion.py on the laptop and push it (grounding is owned/committed, not re-run here)"
  if [ ! -f "$REPAIRED" ]; then
    [ -f "$MOTION_REPAIR" ] || die "tools/motion_repair.py absent at $MOTION_REPAIR — push tools/"
    say "repair grounded CSV at ${G1_SLOWDOWN}x (tools/motion_repair.py --apply-factor)"
    "$PY" "$MOTION_REPAIR" "$GROUNDED" --fps 30 --apply-factor "$G1_SLOWDOWN" \
        --out "$REPAIRED" --scorecard "${REPAIRED%.csv}_scorecard.json" \
      || die "motion_repair failed on the grounded CSV"
  fi
  [ -f "$REPAIRED" ] || die "repaired CSV $REPAIRED still absent after repair"
  say "convert repaired+grounded CSV -> npz (30 -> 50 fps)"
  "$PY" "$CSV2NPZ" --input-file "$REPAIRED" \
      --output-name "$(basename "${NPZ%.npz}")" --input-fps 30 --output-fps 50 || true
  [ -f /tmp/motion.npz ] && cp /tmp/motion.npz "$NPZ"
fi
[ -f "$NPZ" ] || die "motion npz still absent after prep chain"
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
echo "  motion: $NPZ ($FRAMES frames, ${G1_SLOWDOWN}x grounded+repaired)"

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

say "recipe selfcheck (asserts 154-dim actor, asymmetric split, ankle clamp, slowdown)"
G1_SLOWDOWN="$G1_SLOWDOWN" "$PY" "$ENTRY" --selfcheck || die "v8 --selfcheck failed (see output above — likely the asymmetric actor/critic split or a reward key)"

say "resume-flag check"
RESUME_HELP=$("$PY" "$ENTRY" "$TASK" --help 2>&1 || true)
case "$RESUME_HELP" in
  *"--agent.resume"*) echo "  --agent.resume present" ;;
  *) die "no --agent.resume flag on this mjlab" ;;
esac

# STACK SMOKE TEST — the one gate --selfcheck can't do: proves the GPU physics actually
# STEPS with the v8 cfg (catches mujoco-warp/warp/torch drift + GL/CUDA clashes AND any
# v8-specific runtime break: the ankle action-rate action index, the effort-DR regexes,
# the waist-gated reward shapes) in ~2 min, BEFORE the ~2.8 h run.
say "stack smoke test (64-env/2-iter GPU — catches version/CUDA/GL + v8 runtime breaks)"
G1_SLOWDOWN="$G1_SLOWDOWN" bash "$NB/cloud/smoke_test.sh" "$ENTRY" "$TASK" "$NPZ" \
  || die "GPU physics smoke test FAILED — NOT starting the real run. Either a mujoco-warp/warp/torch drift (reinstall from cloud/env_lock/requirements.lock.txt) or a v8 cfg break (check the smoke log for a shape/KeyError in the new ankle/waist/effort terms)."

say "PREFLIGHT PASSED — starting v8 curriculum (~2.8 h)"
MOTION="$NPZ" G1_SLOWDOWN="$G1_SLOWDOWN" bash "$NB/cloud/train_v8_curriculum.sh"
RC=$?

RUN=$(ls -dt "$NB"/exports/train-thriller_v8s2r-* 2>/dev/null | head -1)
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
Reminder: v8 actor obs is 154-dim (estimator-free). After signing, the DEPLOY wave
deletes the dead odometry path (pipeline/leg_odometry.py + deploy_runtime.build_obs_odom)
— see the TODO in cloud/sim2real_task_v8.py. Do NOT touch deploy code in this wave.
===============================================================
EOF
exit $RC
