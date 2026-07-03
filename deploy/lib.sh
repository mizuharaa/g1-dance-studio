#!/usr/bin/env bash
# Shared guardrails for every robot-facing script in deploy/.
#
# HARD RULE (CLAUDE.md + PROJECT_STATE): nothing executes against the robot
# unless a human explicitly arms it. Two independent interlocks:
#   1. env  CONFIRMED_BY_HUMAN=alois   (set per-session, by a person)
#   2. each script's own explicit flags (see its --help)
# Scripts default to DRY-RUN: they print what they would do.
# shellcheck disable=SC2034  # consumed by the scripts that source this file

PC2_HOST="192.168.123.164"
PC2_USER="unitree"
LAPTOP_WIRED_IP="192.168.123.2"
CONTROLLER_IMAGE="qiayuanl/unitree:jazzy"
CONTROLLER_REPO="https://github.com/qiayuanl/motion_tracking_controller"
PC2_WORKDIR="/home/unitree/g1dance"

die() { echo "ABORT: $*" >&2; exit 1; }
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

require_human() {
    [ "${CONFIRMED_BY_HUMAN:-}" = "alois" ] ||
        die "refusing: export CONFIRMED_BY_HUMAN=alois required (a human must set this, per safety policy)"
}

# pc2 <cmd...> — run on the Jetson. Honors DRY_RUN=1 (default in every script
# until --arm/--yes-* flags flip it).
pc2() {
    if [ "${DRY_RUN:-1}" = "1" ]; then
        log "DRY-RUN pc2> $*"
    else
        # BatchMode: never hang on password prompts; robot LAN only.
        ssh -o BatchMode=yes -o ConnectTimeout=5 "${PC2_USER}@${PC2_HOST}" "$@"
    fi
}

check_robot_reachable() {
    ping -c1 -W2 "$PC2_HOST" >/dev/null 2>&1 || die "PC2 ${PC2_HOST} unreachable — is the robot on and the laptop on robot-lan?"
}
