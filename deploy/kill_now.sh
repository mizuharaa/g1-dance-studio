#!/usr/bin/env bash
# INSTANT KILL: stop the controller container immediately.
#
# Safety review #2/#10: an emergency STOP must never be gated on an arming
# credential. Stopping the controller is always safety-positive, so this script
# fires from ANY shell with NO env var and NO flags — grab any terminal and run it.
# The hardware remote (B-damping) in the operator's hand ALWAYS takes precedence.
#
# Safety review #1: we no longer *assert* the robot "falls back to damping". We
# SIGTERM first (docker stop) so the controller has a bounded window to command its
# own damping posture, then SIGKILL as a fallback. The actual command-loss -> damping
# behavior MUST be verified empirically on the gantry before ground use (runbook step 3a).
set -uo pipefail
cd "$(dirname "$0")" || exit 1
# shellcheck source=lib.sh
source ./lib.sh

# NOTE: no require_human here, by design (#2/#10). DRY_RUN stays 0 — a kill is not a drill.
# shellcheck disable=SC2034  # read by pc2() in lib.sh
DRY_RUN=0
STOP_TIMEOUT="${STOP_TIMEOUT:-2}"  # seconds for the controller to damp before SIGKILL
log "KILL: stopping g1dance-controller on PC2 NOW (SIGTERM, ${STOP_TIMEOUT}s grace, then SIGKILL)"
# docker stop = SIGTERM then SIGKILL after -t; the explicit docker kill covers a hung stop.
pc2 "docker stop -t ${STOP_TIMEOUT} g1dance-controller 2>/dev/null; docker kill g1dance-controller 2>/dev/null; docker rm -f g1dance-controller 2>/dev/null; true"
log "kill issued. DO NOT assume a safe posture — VISUALLY verify the robot before approaching,"
log "and keep the remote e-stop in hand (it is the only guaranteed stop until #3a is verified)."
