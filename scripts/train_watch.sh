#!/usr/bin/env bash
# Live-watch the v5 training on the GPU box.
# Usage:  bash scripts/train_watch.sh <IP> <PORT>      (Ctrl-C to stop the live tail)
IP=${1:?usage: bash scripts/train_watch.sh <IP> <PORT>}
PORT=${2:?usage: bash scripts/train_watch.sh <IP> <PORT>}
KEY="$(cd "$(dirname "$0")/.." && pwd)/.secrets/greennode_rsa"
ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -p "$PORT" root@"$IP" bash -s <<'REMOTE'
NB=/workspace/notebook-data
echo "===== GPU ====="; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo "===== stage running now ====="; pgrep -af 'train_sim2real' | grep -oE 'run-name [^ ]+' | tail -1 || echo 'no train proc'
echo "===== progress so far (reward must CLIMB; watch motion_global_root_pos) ====="
grep -iE 'RESUME STAGE|Learning iteration|Mean reward|motion_global_root_pos|Iteration time|ETA|GATES' "$NB/logs/resume_v5.log" 2>/dev/null | tail -10
echo; echo "===== LIVE LOG (Ctrl-C to stop) ====="; tail -f "$NB/logs/resume_v5.log"
REMOTE
