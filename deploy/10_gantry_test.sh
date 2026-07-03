#!/usr/bin/env bash
# Gantry-first controller test — the ONLY script that can start the controller.
# Interlocks (ALL required, in addition to CONFIRMED_BY_HUMAN=alois):
#   --dance <name>        which pushed bundle
#   --gantry-confirmed    robot is HANGING, feet just off ground, straps checked
#   --estop-confirmed     operator holds the remote e-stop, area clear
#   --arm                 actually execute (otherwise DRY-RUN printout)
# Even when armed: the controller container starts in DAMPING HOLD — it does NOT
# play the motion. Playback needs the operator's start sequence ON the remote,
# per docs/ROBOT_DAY_RUNBOOK.md step 6. Abort at any point: deploy/kill_now.sh
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=lib.sh
source ./lib.sh

DANCE="" GANTRY=0 ESTOP=0 DRY_RUN=1 STAGE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dance) DANCE="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --gantry-confirmed) GANTRY=1; shift ;;
        --estop-confirmed) ESTOP=1; shift ;;
        --arm) DRY_RUN=0; shift ;;
        *) die "unknown arg $1" ;;
    esac
done
[ -n "$DANCE" ] || die "usage: 10_gantry_test.sh --dance <name> --stage gantry|ground --estop-confirmed [--gantry-confirmed] [--arm]"
# finding #31: no shell metacharacters reach the remote ssh/docker command
printf '%s' "$DANCE" | grep -Eq '^[A-Za-z0-9_-]{1,64}$' || die "refusing: --dance must match ^[A-Za-z0-9_-]{1,64}$"
# finding #17/#30: gantry and ground are DIFFERENT safety states — force an explicit choice.
case "$STAGE" in
    gantry) PHRASE="FEET OFF GROUND" ;;
    ground) PHRASE="GROUND SAFETY LINE SET" ;;
    *) die "refusing: --stage must be 'gantry' (feet off ground) or 'ground' (safety line taut)" ;;
esac

require_human
[ "$ESTOP" = 1 ] || die "refusing: --estop-confirmed not given (e-stop in operator's hand)"
if [ "$STAGE" = gantry ]; then
    [ "$GANTRY" = 1 ] || die "refusing: --gantry-confirmed not given (robot must hang, feet off ground)"
fi

if [ "$DRY_RUN" = 0 ]; then
    # finding #16: the arming phrase must be TYPED at a real terminal, not piped in.
    [ -t 0 ] || die "refusing: confirmation must be typed at an interactive terminal (no pipe/heredoc)"
    check_robot_reachable
    pc2 "test -f ${PC2_WORKDIR}/bundles/${DANCE}/bundle.json" || die "bundle ${DANCE} not on PC2 — run 02_push_bundle.sh"
    # finding #3: verify the pushed start script really pins damping before we start it.
    pc2 "grep -q 'START_MODE.*!=.*damping' ${PC2_WORKDIR}/bundles/${DANCE}/start_controller_damping_hold.sh" \
        || die "refusing: pushed start script does not assert START_MODE=damping"
    pc2 "test -f ${PC2_WORKDIR}/bundles/${DANCE}/LAUNCH_LINE_VERIFIED" \
        || die "refusing: LAUNCH_LINE_VERIFIED missing on PC2 (runbook step 3)"
    echo ""
    echo "  STAGE: ${STAGE}   FINAL PHYSICAL CHECK (answer on the keyboard, not from memory):"
    if [ "$STAGE" = gantry ]; then
        echo "   - robot hanging, feet ~5 cm off ground, straps rated and locked?"
    else
        echo "   - safety line TAUT (partial-weight support), ground stability gate PASSED?"
        echo "   - 2 m radius clear, floor as vetted (hard flat)?"
    fi
    echo "   - nobody within arm radius of the robot?"
    echo "   - e-stop held by the operator, tested today?"
    read -r -p "  type exactly '${PHRASE}' to continue: " ANSWER </dev/tty
    [ "$ANSWER" = "$PHRASE" ] || die "confirmation phrase mismatch"
fi

log "mode: $([ "$DRY_RUN" = 1 ] && echo DRY-RUN || echo LIVE) dance=${DANCE}"

# Start the controller container in DAMPING HOLD. It loads the policy but holds
# damping until the operator's remote start sequence (runbook step 6).
# NOTE robot-day verification: exact launch entrypoint inside the image is
# confirmed against the controller README on the day (runbook step 3) — this
# wrapper pins the contract: damping on start, motion armed only by operator.
pc2 "docker run -d --rm --name g1dance-controller --network host --privileged \
    -v ${PC2_WORKDIR}/bundles/${DANCE}:/bundle:ro \
    ${CONTROLLER_IMAGE} \
    /bundle/start_controller_damping_hold.sh"

log "controller container start issued (DAMPING HOLD)."
log "watch:  ssh ${PC2_USER}@${PC2_HOST} docker logs -f g1dance-controller"
log "abort:  deploy/kill_now.sh  (stops the container). The remote e-stop (B-damping)"
log "        in your hand is the ONLY guaranteed stop until command-loss->damping is"
log "        verified on the gantry (runbook step 3a). Do NOT assume a safe posture."
