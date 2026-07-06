#!/usr/bin/env bash
# Queue the acro training behind the running v3 program (poll, don't preempt).
#
# Runs ON the box as its own run_job.sh job ("acro-launcher"): polls the live
# tmux sessions every 3 min and, as soon as <= MAX_CONCURRENT train jobs are
# alive (v3a/v3b finish first, leaving v3c+v3d), starts train-acro-1 plus its
# autopilot, then exits. Counts LIVE tmux sessions (job-train-*), not status
# files — stale status JSONs from before a box restart say "running" forever.
#
#   bash cloud/run_job.sh start acro-launcher -- \
#     "bash /workspace/notebook-data/cloud/launch_acro_when_free.sh [motion.npz]"
set -u
NB=/workspace/notebook-data
export PATH="$NB/bin:$PATH"   # tmux lives in $NB/bin
MOTION="${1:-$NB/motions/acro_backflip.npz}"
MAX_CONCURRENT=2
ITERS=10000
RUN_NAME=train-acro-1

[ -s "$MOTION" ] || { echo "FATAL: motion npz missing: $MOTION"; exit 1; }

echo "$(date -Is) acro-launcher: waiting for <=${MAX_CONCURRENT} live train jobs"
while :; do
    n=$(tmux ls 2>/dev/null | grep -c '^job-train-') || n=0
    echo "$(date -Is) live train jobs: $n"
    [ "$n" -le "$MAX_CONCURRENT" ] && break
    sleep 180
done

echo "$(date -Is) slot free — launching $RUN_NAME"
bash "$NB/cloud/run_job.sh" start "$RUN_NAME" -- \
    "cd $NB && MUJOCO_GL=egl WANDB_API_KEY=\$(cat .wandb_key) \
     ./envs/mjlab/bin/python cloud/train_dynamic_skills.py \
     Mjlab-Tracking-Flat-Unitree-G1-Acro \
     --env.commands.motion.motion-file $MOTION \
     --env.scene.num-envs 4096 --agent.max-iterations $ITERS \
     --agent.run-name $RUN_NAME --video False" || exit 1

bash "$NB/cloud/run_job.sh" start acro-autopilot -- \
    "cd $NB && ./envs/mjlab/bin/python cloud/autopilot_acro.py $RUN_NAME $MOTION" || exit 1

echo "$(date -Is) launched $RUN_NAME + acro-autopilot"
