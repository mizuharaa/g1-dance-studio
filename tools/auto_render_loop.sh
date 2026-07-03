#!/usr/bin/env bash
# Every ~20 min: render each running training job's latest checkpoint on the box
# and pull the MP4 to the laptop. Skips a job if that iteration was already rendered.
set -uo pipefail
KEY=$HOME/g1-dance/.secrets/greennode_ssh_key
BOX="root@103.245.250.152"; PORT=46936
SSH="ssh -i $KEY -p $PORT -o ConnectTimeout=15 $BOX"
LOCAL=$HOME/g1-dance/data/previews/progress
mkdir -p "$LOCAL"
declare -A MOTION=( [train-dance2-long]=dance2_long.npz )
while true; do
  for JOB in "${!MOTION[@]}"; do
    $SSH "tmux has-session -t job-$JOB 2>/dev/null" || continue
    OUT=$($SSH "export PATH=/workspace/notebook-data/bin:\$PATH; D=/workspace/notebook-data; tmux new-session -d -s r-$JOB \"bash \$D/cloud/render_progress.sh $JOB \$D/motions/${MOTION[$JOB]} > \$D/previews_progress/r-$JOB.out 2>&1\"; until ! tmux has-session -t r-$JOB 2>/dev/null; do sleep 10; done; cat \$D/previews_progress/r-$JOB.out" 2>/dev/null)
    case "$OUT" in
      *.mp4) F=$(basename "$OUT"); [ -f "$LOCAL/$F" ] || { scp -q -i $KEY -P $PORT "$BOX:$OUT" "$LOCAL/" && echo "PULLED $F"; } ;;
    esac
  done
  sleep 1200
done
