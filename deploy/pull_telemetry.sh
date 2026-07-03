#!/usr/bin/env bash
# Pull a stage's robot telemetry from PC2 to the laptop for sim-vs-real analysis.
# The controller (10_gantry_test.sh) mounts ${PC2_WORKDIR}/telemetry/<dance>/<stage>
# into the container as /telemetry and passes TELEMETRY_DIR=/telemetry so the controller
# can log joint pos/vel (commanded vs actual), IMU, and tracking error there.
# Read-only pull; never starts/stops anything. DRY-RUN prints the scp it would run.
#   export CONFIRMED_BY_HUMAN=alois
#   ./pull_telemetry.sh --dance thriller --stage gantry [--yes-pull]
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=lib.sh
source ./lib.sh

DANCE="" STAGE="" DRY_RUN=1
while [ $# -gt 0 ]; do
    case "$1" in
        --dance) DANCE="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --yes-pull) DRY_RUN=0; shift ;;
        *) die "unknown arg $1" ;;
    esac
done
[ -n "$DANCE" ] && [ -n "$STAGE" ] || die "usage: pull_telemetry.sh --dance <name> --stage <stage> [--yes-pull]"
printf '%s' "$DANCE" | grep -Eq '^[A-Za-z0-9_-]{1,64}$' || die "refusing: bad --dance"
printf '%s' "$STAGE" | grep -Eq '^[A-Za-z0-9-]{1,32}$' || die "refusing: bad --stage"
require_human

REMOTE="${PC2_WORKDIR}/telemetry/${DANCE}/${STAGE}"
LOCAL="$(dirname "$0")/../data/telemetry/${DANCE}/${STAGE}"
mkdir -p "$LOCAL"
if [ "$DRY_RUN" = 1 ]; then
    log "DRY-RUN scp> ${PC2_USER}@${PC2_HOST}:${REMOTE}/*  ->  ${LOCAL}/"
    log "add --yes-pull to actually pull (read-only, safe)."
else
    check_robot_reachable
    scp -o BatchMode=yes -r "${PC2_USER}@${PC2_HOST}:${REMOTE}/"* "${LOCAL}/" 2>/dev/null \
        || log "no telemetry at ${REMOTE} yet (controller may not have logged this stage)"
    log "pulled -> ${LOCAL}  (compare commanded vs actual joint tracking against the sim)"
fi
