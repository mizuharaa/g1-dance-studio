#!/usr/bin/env bash
# Re-run the v3-program gap checks that silently never ran (2026-07-06):
# sim_gap_check.py's argv shim ate the literal stock task string after --task,
# so every stock-task variant (v3a/v3d/v4) has arm metrics but NO gate data —
# their RESULT.txt "gate=FAIL" means "gap_check.json missing", not a real fail.
# Shim fixed in cloud/sim_gap_check.py; this backfills the four missing gates.
# Outputs are KEPT (autopilot retention bug bypassed): exports/<v>/<tag>/gap_check.json
#
#   bash cloud/run_job.sh start gap-backfill -- "bash /workspace/notebook-data/cloud/rerun_gap_checks.sh"
set -u
NB=/workspace/notebook-data
PY=$NB/envs/mjlab/bin/python
STOCK=Mjlab-Tracking-Flat-Unitree-G1
DEPLOY=$NB/motions/thriller_deploy.npz
SHARP=$NB/motions/thriller_deploy_v2_sharp.npz

run_one() { # name checkpoint motion outdir
    local name="$1" ckpt="$2" motion="$3" out="$4"
    echo "=== $(date -Is) gap_check $name ==="
    [ -s "$ckpt" ] || { echo "SKIP $name: checkpoint missing: $ckpt"; return; }
    mkdir -p "$out"
    MUJOCO_GL=egl $PY $NB/cloud/sim_gap_check_v3.py \
        --checkpoint "$ckpt" --motion-file "$motion" \
        --task $STOCK --num-envs 128 \
        --output-file "$out/gap_check.json" \
        && echo "OK $name -> $out/gap_check.json" \
        || echo "FAIL $name rc=$?"
}

V3A_RUN=$(ls -d $NB/logs/rsl_rl/g1_tracking/*train-thriller-v3a* | tail -1)
V3D_RUN=$(ls -d $NB/logs/rsl_rl/g1_tracking/*train-thriller-v3d* | tail -1)
V4_RUN=$(ls -d $NB/logs/rsl_rl/g1_tracking/*train-thriller-v4* | tail -1)

run_one v3a-last "$V3A_RUN/model_4999.pt" "$DEPLOY" "$NB/exports/thriller_v3a/last"
run_one v3d-last "$V3D_RUN/model_4999.pt" "$SHARP"  "$NB/exports/thriller_v3d/last"
run_one v4-mid   "$V4_RUN/model_2500.pt"  "$SHARP"  "$NB/exports/thriller_v34/mid"
V4_LAST=$(ls $V4_RUN/model_*.pt | sort -t_ -k2 -n | tail -1)
run_one v4-last  "$V4_LAST"               "$SHARP"  "$NB/exports/thriller_v34/last"

echo "=== $(date -Is) gap backfill complete ==="
