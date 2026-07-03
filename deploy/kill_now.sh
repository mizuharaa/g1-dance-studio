#!/usr/bin/env bash
# INSTANT KILL: stop the controller container immediately.
# This is the software abort path (safety-positive) — it needs the session env
# var but NO extra flags and NO dry-run: export CONFIRMED_BY_HUMAN=alois once at
# session start (runbook step 0) and this is then a single keystroke away.
# The hardware e-stop in the operator's hand ALWAYS takes precedence over this.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=lib.sh
source ./lib.sh

require_human
# shellcheck disable=SC2034  # read by pc2() in lib.sh
DRY_RUN=0
log "KILL: stopping g1dance-controller on PC2 NOW"
pc2 "docker kill g1dance-controller 2>/dev/null; docker rm -f g1dance-controller 2>/dev/null; true"
log "kill issued. Robot low-level falls back to damping. Verify posture before approaching."
