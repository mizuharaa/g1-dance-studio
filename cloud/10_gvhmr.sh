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
# GVHMR needs its own pinned stack (torch 2.3.0+cu121 + pytorch3d cp310 wheel),
# which conflicts with the image's torch 2.5.1/py3.11 — isolated py3.10 venv.
VENV="$(ensure_venv310 gvhmr)"
PY="$VENV/bin/python"
log "venv: $VENV (isolated py3.10, GVHMR-pinned torch)"

# -- repo ------------------------------------------------------------------
if [ ! -d "$REPO/.git" ]; then
    log "cloning GVHMR"
    git clone --depth 1 https://github.com/zju3dv/GVHMR "$REPO"
else
    log "GVHMR repo present"
fi

# -- python deps -------------------------------------------------------------
# GVHMR's setup.py declares no dependencies — the real ones are pinned in
# requirements.txt (incl. its own torch and a matching pytorch3d wheel).
log "installing GVHMR python deps from requirements.txt (first run ~10 min)"
# chumpy and cython_bbox (SMPL-era packages) predate modern pip build
# isolation and fail to even build metadata. cython_bbox additionally needs a
# C compiler the box doesn't have — and NOTHING in GVHMR imports it (verified
# by grep 2026-07-03), so it is skipped outright. chumpy is pure-python and
# needed only to unpickle legacy SMPL pkl files.
grep -vE "^(chumpy|cython_bbox)" "$REPO/requirements.txt" > /tmp/gvhmr-reqs.txt
"$PY" -m pip install -q "numpy==1.23.5" "setuptools>=68,<70" Cython 2>&1 | tail -1
"$PY" -m pip install -q -r /tmp/gvhmr-reqs.txt 2>&1 | tail -3 \
    || die "GVHMR requirements install failed"
"$PY" -m pip install -q --no-build-isolation chumpy 2>&1 | tail -2 \
    || die "chumpy install failed (even with --no-build-isolation)"
"$PY" -m pip install -q -e "$REPO" --no-deps 2>&1 | tail -2 \
    || die "GVHMR pip install failed"

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
    log "fetching GVHMR checkpoints from HuggingFace (one-time, ~5 GB)"
    # NB_DATA must reach the python child as a real env var — expandvars on an
    # unexported name silently keeps the literal '$NB_DATA' and the download
    # lands in a directory literally named '$NB_DATA' (bit us 2026-07-03).
    export NB_DATA
    "$PY" - <<'EOF' || echo "WARNING: checkpoint download failed — laptop can sync them instead (see report)"
from huggingface_hub import snapshot_download
import os
dest = os.path.join(os.environ["NB_DATA"], "repos/GVHMR/inputs/checkpoints")
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
