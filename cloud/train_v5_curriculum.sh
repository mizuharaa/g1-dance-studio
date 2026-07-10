#!/usr/bin/env bash
# Lane E: v5 "fidelity" retrain — 3-stage latency curriculum via resume.
# Run ON THE BOX inside tmux:  tmux new -s train
#   MOTION=/workspace/notebook-data/motions/<lane-b-phase2>.npz bash cloud/train_v5_curriculum.sh
#
# Stages (why: lat80 proved 0-80 ms from step 0 destroys station-keeping; the
# policy must learn the dance FIRST, then harden):
#   1. 0-20 ms cmd / 0-20 ms obs, 4000 iters  (proven-stable regime)
#   2. 0-50 ms cmd / 0-40 ms obs, +3000 iters (resume stage-1)
#   3. 0-60 ms cmd / 0-60 ms obs, +3000 iters (resume stage-2; 60 not 80 —
#      sim PD already models mechanical lag, 80 double-counts it)
set -euo pipefail

NB=${NB:-/workspace/notebook-data}
PY=$NB/envs/mjlab/bin/python
TASK=Mjlab-Tracking-Flat-Unitree-G1-S2R-V5
MOTION=${MOTION:?set MOTION=/path/to/lane-b-phase2.npz}
RUN=train-thriller_v5fid-$(date +%m%d)
COMMON=(--env.scene.num-envs 4096 --env.commands.motion.motion-file "$MOTION")

# ponytail: resume flag names UNVERIFIED on this mjlab version — before stage 2,
# confirm with: $PY $NB/cloud/train_sim2real_v5.py $TASK --help | grep -i resume
# (rsl_rl convention: --agent.resume True --agent.load-run <run> --agent.load-checkpoint <ckpt>)
resume_args() { echo "--agent.resume True --agent.load-run $1"; }

echo "== stage 1/3: 0-20 ms, 4000 iters =="
G1_CMD_DELAY_MAX_LAG=4  G1_OBS_DELAY_MAX_LAG=1 \
  $PY $NB/cloud/train_sim2real_v5.py $TASK "${COMMON[@]}" \
      --agent.max-iterations 4000 --agent.run-name "${RUN}-s1"

echo "== stage 2/3: 0-50 ms, +3000 iters (resume s1) =="
G1_CMD_DELAY_MAX_LAG=10 G1_OBS_DELAY_MAX_LAG=2 \
  $PY $NB/cloud/train_sim2real_v5.py $TASK "${COMMON[@]}" \
      --agent.max-iterations 3000 --agent.run-name "${RUN}-s2" $(resume_args "${RUN}-s1")

echo "== stage 3/3: 0-60 ms, +3000 iters (resume s2) =="
G1_CMD_DELAY_MAX_LAG=12 G1_OBS_DELAY_MAX_LAG=3 \
  $PY $NB/cloud/train_sim2real_v5.py $TASK "${COMMON[@]}" \
      --agent.max-iterations 3000 --agent.run-name "${RUN}-s3" $(resume_args "${RUN}-s2")

echo "== training done — verify chain (gates: 40ms+push survival AND nominal drift <1m) =="
CKPT=$(ls -t $NB/logs/rsl_rl/g1_tracking/*${RUN}-s3*/model_*.pt | head -1)
echo "checkpoint: $CKPT"
$PY $NB/cloud/export_policy.py "$CKPT" "$MOTION" "$NB/exports/${RUN}"
$PY $NB/cloud/sim_gap_check.py --checkpoint "$CKPT" --motion-file "$MOTION" \
    --num-envs 128 --output-file "$NB/exports/${RUN}/gap.json"
for SEED in 90001 90011 90021; do
  $PY $NB/cloud/heldout_eval.py $TASK --checkpoint "$CKPT" --seed $SEED --num-envs 256
done
echo "PULL artifacts + gap.json to the laptop, sign (pipeline/mjlab_verify.py), then DELETE THE BOX."
