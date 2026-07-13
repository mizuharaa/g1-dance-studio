#!/usr/bin/env bash
# LAPTOP step 3 — pull the trained artifacts, md5-check, and print the exact sign command.
# Usage:  bash scripts/retrain_pull.sh <IP> <PORT>
set -euo pipefail
IP=${1:?usage: bash scripts/retrain_pull.sh <IP> <PORT>}
PORT=${2:?usage: bash scripts/retrain_pull.sh <IP> <PORT>}
cd "$(dirname "$0")/.."
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate g1dance; set -u
KEY="$PWD/.secrets/greennode_rsa"
DST="data/policies/thriller_v5fid"
mkdir -p "$DST"

echo "== pull exports (policy.onnx, policy_meta.json, gap.json, heldout_*.json) =="
scp -P "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
    'root@'"$IP"':/workspace/notebook-data/exports/train-thriller_v5fid-*/*' "$DST"/
echo "== pull the deploy CSV (needed for signing) =="
scp -P "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
    'root@'"$IP"':/workspace/notebook-data/motions/thriller_g1_clean.csv' "$DST"/ || true

echo "== md5 (note it — must match the box copy) =="
md5sum "$DST"/policy.onnx

echo
echo "PULLED into $DST :"
ls -1 "$DST"
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
