#!/usr/bin/env bash
# GVHMR (video -> SMPL-X human motion) provisioning on the GreenNode notebook.
# Idempotent; re-run after every instance Stop (venv + repo live on the
# persistent mount, so re-runs are mostly no-ops).
#
# Prereqs: 00_bootstrap.sh done; body models synced from the laptop into
# $NB_DATA/body_models/ (the laptop app's cloud sync does this — they are
# license-gated and never downloaded here).
#
# Usage:   bash 10_gvhmr.sh
cd "$(dirname "$0")" || exit 1
# shellcheck source=lib.sh
. ./lib.sh
layout

REPO="$NB_DATA/repos/GVHMR"
VENV="$(ensure_venv gvhmr)"
PY="$VENV/bin/python"
log "venv: $VENV (torch from image via system-site-packages)"

# -- repo ------------------------------------------------------------------
if [ ! -d "$REPO/.git" ]; then
    log "cloning GVHMR"
    git clone --depth 1 https://github.com/zju3dv/GVHMR "$REPO"
else
    log "GVHMR repo present"
fi

# -- python deps -------------------------------------------------------------
log "installing GVHMR python deps (cached under the mount)"
"$PY" -m pip install -q -e "$REPO" 2>&1 | tail -2 || die "GVHMR pip install failed"

# -- body models (synced from laptop, license-gated) --------------------------
BM="$NB_DATA/body_models"
CKPT_BM="$REPO/inputs/checkpoints/body_models"
if [ -f "$BM/smpl/SMPL_NEUTRAL.pkl" ] && [ -f "$BM/smplx/SMPLX_NEUTRAL.npz" ]; then
    mkdir -p "$CKPT_BM"
    ln -sfn "$BM/smpl" "$CKPT_BM/smpl"
    ln -sfn "$BM/smplx" "$CKPT_BM/smplx"
    log "body models linked into GVHMR inputs"
else
    log "WARNING: body models not synced yet ($BM) — run the laptop app's"
    log "         cloud sync (or scp data/body_models) before extracting."
fi

# -- pretrained checkpoints ----------------------------------------------------
# GVHMR publishes checkpoints on HuggingFace (mirror: camenduru/GVHMR).
# ~2 GB, cached under $HF_HOME on the mount.
CKPTS="$REPO/inputs/checkpoints"
if [ ! -e "$CKPTS/gvhmr/gvhmr_siga24_release.ckpt" ]; then
    log "fetching GVHMR checkpoints from HuggingFace (one-time, ~2 GB)"
    "$PY" - <<'EOF' || echo "WARNING: checkpoint download failed — laptop can sync them instead (see report)"
from huggingface_hub import snapshot_download
import os
dest = os.path.expandvars("$NB_DATA/repos/GVHMR/inputs/checkpoints")
snapshot_download(repo_id="camenduru/GVHMR", local_dir=dest,
                  allow_patterns=["gvhmr/*", "hmr2/*", "vitpose/*", "dpvo/*", "yolo/*"])
print("checkpoints ready:", dest)
EOF
else
    log "GVHMR checkpoints present"
fi

# -- smoke test -----------------------------------------------------------------
log "smoke test: import + CUDA visibility"
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
EOF
log "GVHMR provisioning done"
