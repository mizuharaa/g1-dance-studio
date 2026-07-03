#!/usr/bin/env bash
# Training-stack provisioning: BeyondMimic (whole_body_tracking) under Isaac
# Lab 2.1.0 — with an honest, reported failure path, because the GreenNode
# image is FIXED at PyTorch 2.5.1/CUDA 12.4 and Isaac Sim is picky (needs
# python 3.10 + GLIBC>=2.34 + ~30 GB disk). If Isaac Lab can't work here, the
# report tells the laptop to switch to the mjlab fallback (architecture's
# bounded fallback, decision 2026-06-12).
#
# Writes $NB_DATA/reports/training_stack.json either way. Idempotent.
#
# Usage:   bash 20_training.sh
cd "$(dirname "$0")" || exit 1
# shellcheck source=lib.sh
. ./lib.sh
layout

REPORT="$NB_DATA/reports/training_stack.json"
WBT="$NB_DATA/repos/whole_body_tracking"
ISAACLAB="$NB_DATA/repos/IsaacLab"

report() { # report <status> <detail>
    printf '{"status": "%s", "detail": "%s", "at": "%s"}\n' \
        "$1" "$2" "$(date -Is)" > "$REPORT"
    log "report: $1 — $2"
}

# -- preflight ------------------------------------------------------------------
PYV="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo 0)"
FREE_GB="$(python3 -c 'import shutil; print(shutil.disk_usage("/").free // 10**9)')"
log "preflight: python $PYV, glibc $GLIBC, ${FREE_GB} GB free, GPU: $(gpu_check)"

if [ "$FREE_GB" -lt 40 ]; then
    report "blocked" "only ${FREE_GB} GB free — resize block storage to >=150 GB before installing Isaac Lab"
    exit 1
fi

# -- whole_body_tracking (BeyondMimic) — needed by BOTH isaac and mjlab paths ----
if [ ! -d "$WBT/.git" ]; then
    log "cloning whole_body_tracking (BeyondMimic)"
    git clone --depth 1 https://github.com/HybridRobotics/whole_body_tracking "$WBT"
fi

# -- attempt: Isaac Lab 2.1.0 ---------------------------------------------------
# Isaac Sim 4.5 ships cp310-only wheels; the image's conda python is 3.11 but
# /usr/bin/python3 is 3.10 — build this env on it (isolated, pip-bootstrapped).
VENV="$(ensure_venv310 isaaclab)"
PY="$VENV/bin/python"

if "$PY" -c 'import isaaclab' 2>/dev/null; then
    log "Isaac Lab already importable — skipping install"
else
    log "installing Isaac Sim 4.5 pip wheels (~10 GB, one-time; cached on mount)"
    if ! "$PY" -m pip install -q "isaacsim[all,extscache]==4.5.0" \
            --extra-index-url https://pypi.nvidia.com 2> "$NB_DATA/logs/isaacsim_pip.err"; then
        report "isaac_failed" "isaacsim 4.5.0 wheels rejected on this image (python $PYV, glibc $GLIBC) — see logs/isaacsim_pip.err; USE MJLAB FALLBACK (bash 20_training.sh mjlab)"
        [ "${1:-}" = "mjlab" ] || exit 1
    else
        if [ ! -d "$ISAACLAB/.git" ]; then
            git clone --depth 1 --branch v2.1.0 https://github.com/isaac-sim/IsaacLab "$ISAACLAB"
        fi
        log "installing Isaac Lab 2.1.0"
        # isaaclab.sh expects `python` on PATH (activate the venv) and a sane
        # terminal (TERM=dumb — it dies on tput codes when run without a TTY)
        ( cd "$ISAACLAB" && . "$VENV/bin/activate" && TERM=dumb ./isaaclab.sh --install none ) \
            2> "$NB_DATA/logs/isaaclab_install.err" \
            || { report "isaac_failed" "IsaacLab install script failed — see logs/isaaclab_install.err; USE MJLAB FALLBACK"; exit 1; }
        "$PY" -m pip install -q -e "$WBT/source/whole_body_tracking" \
            || log "WARNING: whole_body_tracking install into isaac env failed"
        report "isaac_ready" "Isaac Lab 2.1.0 + whole_body_tracking installed"
    fi
fi

# -- fallback: mjlab (run with: bash 20_training.sh mjlab) ------------------------
if [ "${1:-}" = "mjlab" ]; then
    VENV_MJ="$(ensure_venv mjlab)"
    log "installing mjlab fallback"
    if "$VENV_MJ/bin/python" -m pip install -q mjlab; then
        report "mjlab_ready" "mjlab fallback installed (bounded fallback per architecture)"
    else
        report "failed" "both Isaac Lab and mjlab installs failed — needs interactive debugging"
    fi
fi

log "training-stack provisioning finished — read $REPORT"
