#!/usr/bin/env bash
# 2-MINUTE WHOLE-STACK SMOKE TEST — "does mjlab actually STEP physics on this box?"
#
# --selfcheck only builds the reward/termination cfg on the CPU. It CANNOT catch the
# failures that actually cost us 3-hour runs, because those happen on the GPU at the
# first env reset / first step:
#   * mujoco-warp / warp-lang version drift  -> device-side assert (CUDA error 710)
#   * torch cu-version vs Warp CUDA mismatch  -> illegal memory access (CUDA 700)
#   * MUJOCO_GL / GL-context clash with Warp   -> illegal memory access
#
# This runs a tiny 64-env / 2-iteration training. If it reaches a learning iteration
# the whole GPU stack works; if it hits a CUDA/Warp error the stack is broken and you
# must NOT start the real run. ~1-2 min of GPU. Exit 0 = PASS, 1 = FAIL/broken.
#
# Usage: smoke_test.sh <entry.py> <task_id> <motion.npz>
set -uo pipefail
NB=${NB:-/workspace/notebook-data}
ENTRY=${1:?entry .py (e.g. cloud/sim2real_task_v7.py)}
TASK=${2:?task id}
MOTION=${3:?motion .npz}
PY=$NB/envs/mjlab/bin/python
LOG=$(mktemp)

echo "[smoke] 64-env / 2-iter physics-step test on $TASK ..."
unset MUJOCO_GL   # training must run WITHOUT a GL backend (egl clashes with Warp CUDA)
# ENTRY=STOCK -> the built-in trainer (no recipe file needed, e.g. at provision time);
# else a recipe .py that registers its own task.
if [ "$ENTRY" = "STOCK" ]; then CMD=("$PY" -m mjlab.scripts.train "$TASK"); else CMD=("$PY" "$ENTRY" "$TASK"); fi
timeout 300 "${CMD[@]}" \
    --env.scene.num-envs 64 --env.commands.motion.motion-file "$MOTION" \
    --agent.max-iterations 2 --agent.run-name smoketest > "$LOG" 2>&1 || true

if grep -qiE "illegal memory|device-side assert|CUDA error 7|Cannot initialize a EGL|Traceback \(most" "$LOG"; then
    echo "[smoke] FAIL — the GPU physics stack is BROKEN on this box:"
    grep -iE "illegal memory|device-side assert|CUDA error|EGL|Error:" "$LOG" | head -4
    echo "[smoke] Most likely cause: mujoco-warp/warp/torch drifted off the known-good."
    echo "[smoke] Fix: reinstall from cloud/env_lock/requirements.lock.txt, then re-run this."
    rm -f "$LOG"; exit 1
fi
if grep -qE "Learning iteration" "$LOG"; then
    echo "[smoke] PASS — mjlab reached a learning iteration; the stack steps physics."
    rm -f "$LOG"; exit 0
fi
echo "[smoke] INCONCLUSIVE — no iteration and no known error signature. Last lines:"
tail -6 "$LOG"; rm -f "$LOG"; exit 1
