#!/usr/bin/env bash
# LAPTOP step 1 — clean + vet the Thriller motion, then push it + the box runner to the box.
# Usage:  bash scripts/retrain_send.sh <IP> <PORT>
set -euo pipefail
IP=${1:?usage: bash scripts/retrain_send.sh <IP> <PORT>}
PORT=${2:?usage: bash scripts/retrain_send.sh <IP> <PORT>}
cd "$(dirname "$0")/.."
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate g1dance; set -u
KEY="$PWD/.secrets/greennode_rsa"
SSH="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

echo "== 1/3  clean the motion (de-glitch: jerk ~/21) =="
python -m pipeline.prep_motion --in data/motions/thriller/thriller_g1.csv \
                               --out data/motions/thriller/thriller_g1_clean.csv

echo "== 2/3  vet — aborts here if the motion fails the hard gate =="
python pipeline/vet_motion.py data/motions/thriller/thriller_g1_clean.csv --json | tail -8

echo "== 3/3  send CSV + box runner to the box =="
scp -P "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
    data/motions/thriller/thriller_g1_clean.csv root@"$IP":/workspace/notebook-data/motions/
scp -P "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
    cloud/retrain_v5_box.sh root@"$IP":/workspace/notebook-data/

echo
echo "DONE. Next — start training on the box:"
echo "  $SSH -p $PORT root@$IP"
echo "  tmux new -s train"
echo "  bash /workspace/notebook-data/retrain_v5_box.sh"
