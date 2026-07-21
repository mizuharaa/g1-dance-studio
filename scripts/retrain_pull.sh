#!/usr/bin/env bash
# LAPTOP step 3 — pull the trained artifacts, md5-check, and print the exact sign command.
# Usage:  bash scripts/retrain_pull.sh <IP> <PORT>
set -euo pipefail
IP=${1:?usage: bash scripts/retrain_pull.sh <IP> <PORT>}
PORT=${2:?usage: bash scripts/retrain_pull.sh <IP> <PORT>}
cd "$(dirname "$0")/.."
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate g1dance; set -u
KEY="$PWD/.secrets/greennode_rsa"
# TAG = policy dir under data/policies/ (also the export glob prefix); NAME = dance name
# shown in the app. Override per run:  TAG=thriller_v8s2r NAME="Thriller — v8" bash ...
TAG="${TAG:-thriller_v5fid}"
DST="data/policies/$TAG"
DANCE_NAME="${NAME:-$TAG}"
mkdir -p "$DST"

echo "== pull exports (policy.onnx, policy_meta.json, gap.json, heldout_*.json, *_deploy.npz) =="
scp -P "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
    'root@'"$IP"':/workspace/notebook-data/exports/train-'"$TAG"'-*/*' "$DST"/
# Explicitly pull THIS policy's OWN staged reference motion (export_policy.py now stages the
# training npz into exports/ as *_deploy.npz). Without it the preview fell back to the shared,
# wrong-lineage thriller_deploy.npz for BOTH the reference pane and the policy command input
# (audit 2026-07-21 finding C). The exports/* glob above usually already grabs it; this line
# makes the intent explicit and survives an exports-layout change. Non-fatal if absent.
echo "== pull this policy's own reference motion (*_deploy.npz) =="
scp -P "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
    'root@'"$IP"':/workspace/notebook-data/exports/train-'"$TAG"'-*/*_deploy.npz' "$DST"/ || true
echo "== pull the deploy CSV (needed for signing + preview-motion fallback) =="
scp -P "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
    'root@'"$IP"':/workspace/notebook-data/motions/thriller_g1_clean.csv' "$DST"/ || true

echo "== md5 (note it — must match the box copy) =="
md5sum "$DST"/policy.onnx

echo
echo "PULLED into $DST :"
ls -1 "$DST"

# AUTO-PUBLISH TO THE FRONTEND — register/update a dance for this policy AND render its
# honest sim preview so it shows up in Dance & Stats + Simulation with NO manual attach.
# Foreground render (a few minutes, no GPU). Robust: never aborts the pull — on any error
# the dance is still registered and the UI can re-render on demand. (set +e around it so
# `set -e` above can't kill the pull if publish returns non-zero.)
echo
echo "== publish to frontend (register dance + render sim preview, faithful model) =="
set +e
python -m pipeline.publish_policy "$DST" --name "$DANCE_NAME"
echo "   (publish rc=$? — non-zero is non-fatal; the dance is registered regardless)"
set -e

echo
echo "== NOW SIGN (fill <heldout.json> with the heldout verdict just pulled) =="
echo "  python pipeline/mjlab_verify.py \\"
echo "      --eval-json $DST/<heldout.json> \\"
echo "      --policy    $DST/policy.onnx \\"
echo "      --motion    $DST/thriller_g1_clean.csv \\"
echo "      --out       $DST/heldout_verdict.json"
echo
echo "Then in the app (Shows): attach the policy -> Promote (gate needs the signed >=99% verdict)."
echo "Then DELETE the box in the GreenNode console."
