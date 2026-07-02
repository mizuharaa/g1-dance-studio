#!/usr/bin/env bash
# Shared helpers for GreenNode notebook provisioning. Sourced by the numbered
# scripts. Design constraints (docs/GREENNODE_SETUP.md):
#   * block storage is EPHEMERAL — wiped on every Stop;
#   * only the Network Volume mount (default /workspace/notebook-data) persists;
#   * fixed image: PyTorch 2.5.1 / CUDA 12.4, no root guarantees.
# Therefore: everything valuable lives under $NB_DATA, and every script must be
# safe to re-run after a stop/start (idempotent, cheap when already done).

set -euo pipefail

# The persistent mount. GreenNode's documented example is /workspace/notebook-data;
# override with NB_DATA=... if the mount folder was named differently.
NB_DATA="${NB_DATA:-/workspace/notebook-data}"

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { log "ERROR: $*"; exit 1; }

require_mount() {
    [ -d "$NB_DATA" ] || die "$NB_DATA does not exist — is the Network Volume \
mounted? (set NB_DATA=<mount folder> if it has another name)"
}

layout() {
    require_mount
    mkdir -p "$NB_DATA"/{repos,envs,jobs,artifacts,logs,reports,cache/pip,bin,body_models}
}

# Route caches to the persistent mount so re-provisioning after a Stop is fast.
export PIP_CACHE_DIR="$NB_DATA/cache/pip"
export HF_HOME="$NB_DATA/cache/huggingface"
export PATH="$NB_DATA/bin:$PATH"

# Package installs: try apt with sudo, then plain apt (root images), else report.
apt_try() {
    if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        sudo apt-get install -y --no-install-recommends "$@" && return 0
    fi
    if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null 2>&1; then
        apt-get install -y --no-install-recommends "$@" && return 0
    fi
    return 1
}

# tmux is essential (long trainings must survive the browser tab / laptop).
ensure_tmux() {
    command -v tmux >/dev/null 2>&1 && return 0
    log "tmux missing — trying apt"
    apt_try tmux && return 0
    log "no apt access — installing a static tmux build into $NB_DATA/bin"
    # nelsonenzo/tmux-appimage provides a self-contained static binary
    if curl -fsSL -o "$NB_DATA/bin/tmux" \
        "https://github.com/nelsonenzo/tmux-appimage/releases/latest/download/tmux.appimage"; then
        chmod +x "$NB_DATA/bin/tmux"
    else
        die "could not install tmux (apt denied, static download failed)"
    fi
}

# Fresh venv that reuses the image's torch 2.5.1/cu124 via system-site-packages
# (Isaac Lab & GVHMR both want torch; re-downloading 3 GB per env is waste).
ensure_venv() { # ensure_venv <name>
    local venv="$NB_DATA/envs/$1"
    [ -x "$venv/bin/python" ] || python3 -m venv --system-site-packages "$venv"
    echo "$venv"
}

gpu_check() {
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null \
        || echo "NO GPU VISIBLE"
}
