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
if [ "${1:-}" = "mjlab" ]; then
    log "mjlab requested explicitly - skipping Isaac Lab attempt"
else


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

fi

# -- fallback: mjlab (run with: bash 20_training.sh mjlab) ------------------------
# mjlab runs in an ISOLATED venv (NOT --system-site-packages). A system-site venv
# inherits the base image's /opt/conda packages, and on a compute-only GreenNode
# image those are incompatible with mjlab (2026-07-08: libstdc++/matplotlib +
# scipy `sph_legendre_p` ufunc conflicts broke the convert stage, one after another).
# Isolated => mjlab brings its own mutually-consistent numpy/scipy/matplotlib/torch
# (manylinux wheels that work with the system libstdc++), so it is image-independent.
if [ "${1:-}" = "mjlab" ]; then
    VENV_MJ="$NB_DATA/envs/mjlab"
    if [ ! -x "$VENV_MJ/bin/python" ]; then
        log "creating isolated mjlab venv (no system-site-packages)"
        /opt/conda/bin/python -m venv "$VENV_MJ" || die "mjlab venv creation failed"
        "$VENV_MJ/bin/python" -m pip install -q --upgrade pip
    fi
    log "installing mjlab (isolated: pulls consistent torch/numpy/scipy/matplotlib)"
    if "$VENV_MJ/bin/python" -m pip install -q mjlab; then
        # CRITICAL (2026-07-14): bare `pip install mjlab` leaves deps UNPINNED, so it
        # pulls whatever is newest — which broke training with mujoco-warp 3.10.0.2 +
        # warp-lang 1.15.0 (device-side assert / CUDA error 700 at the first env reset).
        # mjlab v1.5.0's uv.lock pins mujoco-warp==3.10.0.1 + warp-lang==1.14.0 + torch
        # from the cu128 index. Pin them back to the TESTED combo, or a fresh box dies.
        log "pinning physics libs to mjlab v1.5.0 lock (mujoco-warp 3.10.0.1 / warp 1.14.0)"
        "$VENV_MJ/bin/python" -m pip install -q \
            "mujoco-warp==3.10.0.1" "warp-lang==1.14.0" \
            --extra-index-url https://pypi.nvidia.com \
            || log "WARNING: physics-lib pin failed — training may crash at reset"
        # torch: mjlab needs the cu128 (CUDA 12.8) build; the default index gives cu130
        # (CUDA 13), which Warp 1.14 can't interop with. Force the cu128 wheel.
        "$VENV_MJ/bin/python" -c 'import torch,sys; sys.exit(0 if "cu128" in torch.__version__ else 1)' 2>/dev/null \
            || "$VENV_MJ/bin/python" -m pip install -q --force-reinstall "torch>=2.7.0" \
                 --index-url https://download.pytorch.org/whl/cu128 \
            || log "WARNING: torch cu128 pin failed"
        # This GreenNode image is compute-only: no GL runtime and no NVIDIA EGL
        # (NVIDIA_DRIVER_CAPABILITIES unset). mjlab imports PyOpenGL EGL at load, so
        # install the GLVND loaders (libEGL.so.1, libGL.so.1, ...). They land in
        # /opt/conda/lib, which the training scripts add to LD_LIBRARY_PATH.
        log "installing GLVND GL loaders (libEGL/libGL) for headless mjlab import"
        /opt/conda/bin/conda install -y -c conda-forge libglvnd libegl libgl libglx libopengl \
            >/dev/null 2>&1 || log "WARNING: GL loader install failed — convert/render may break"
        # The app calls csv_to_npz at repos/mjlab/src/mjlab/scripts (source-repo layout),
        # but pip installs mjlab into site-packages. Bridge the two with a symlink.
        PYVER="$("$VENV_MJ/bin/python" -c 'import sys;print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
        mkdir -p "$NB_DATA/repos/mjlab/src"
        ln -sfn "$VENV_MJ/lib/$PYVER/site-packages/mjlab" "$NB_DATA/repos/mjlab/src/mjlab"
        report "mjlab_ready" "mjlab installed (isolated venv + GLVND loaders + repo-path shim)"
    else
        report "failed" "both Isaac Lab and mjlab installs failed — needs interactive debugging"
    fi
fi

log "training-stack provisioning finished — read $REPORT"
