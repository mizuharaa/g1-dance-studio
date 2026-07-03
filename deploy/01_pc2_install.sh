#!/usr/bin/env bash
# One-time install of the motion-tracking controller stack on PC2 (Jetson).
# DRY-RUN by default. Real execution needs BOTH:
#   export CONFIRMED_BY_HUMAN=alois
#   ./01_pc2_install.sh --yes-install
# Installs software only — never starts a controller, never moves motors.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=lib.sh
source ./lib.sh

DRY_RUN=1
[ "${1:-}" = "--yes-install" ] && DRY_RUN=0

require_human
[ "$DRY_RUN" = "0" ] && check_robot_reachable
log "mode: $([ "$DRY_RUN" = 1 ] && echo DRY-RUN || echo LIVE) target: ${PC2_USER}@${PC2_HOST}"

# 1. workspace + bundle dir on PC2 (idempotent)
pc2 "mkdir -p ${PC2_WORKDIR}/bundles"

# 2. docker present? (Jetson images ship docker; verify, do not install system pkgs)
pc2 "docker --version" || die "docker missing on PC2 — resolve manually on robot day"

# 3. pull the controller image (large; robot LAN has no internet — if the pull
#    fails, save/load it through the laptop: see docs/ROBOT_DAY_RUNBOOK.md step 2b)
pc2 "docker image inspect ${CONTROLLER_IMAGE} >/dev/null 2>&1 || docker pull ${CONTROLLER_IMAGE}"

# 4. clone the controller (exact launch file names are confirmed on robot day —
#    the repo layout is pinned in docs/ROBOT_DAY_RUNBOOK.md step 3)
pc2 "[ -d ${PC2_WORKDIR}/motion_tracking_controller ] || git clone ${CONTROLLER_REPO} ${PC2_WORKDIR}/motion_tracking_controller"

log "install steps issued. NOTHING was started. Next: 02_push_bundle.sh"
