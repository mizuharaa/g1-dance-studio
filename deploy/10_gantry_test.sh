#!/usr/bin/env bash
# Staged robot-day controller test — the ONLY script that can start the controller.
# A FULL DAY is a STAGED PROGRESSION with a MANDATORY gate between every stage.
# More time does NOT mean skipping a gate — it means doing every stage thoroughly.
#
#   --stage gantry           robot HANGING, feet off ground            (needs --gantry-confirmed)
#   --stage ground-tethered  on ground, safety line TAUT, partial wt   (needs gantry passed)
#   --stage ground-free      slack line, self-balances, full fall risk (HARD gate, see below)
#   --stage push-test        gentle shoves, after ground-free is clean
#
# ALWAYS required (every stage): CONFIRMED_BY_HUMAN=alois, --estop-confirmed, --arm to execute,
# and the stage's typed phrase at an interactive terminal. Controller ALWAYS starts in
# DAMPING HOLD; motion playback is armed only by the operator's remote sequence.
#
# GROUND-FREE hard gate (where real fall risk lives) additionally requires:
#   --gantry-passed --tethered-passed  (prior stages were clean — operator attests)
#   --kill-damping-confirmed           (operator SAW kill->damping at gantry step 3a)
#   --estimator-verified               (onboard DLIO state-estimator confirmed sane)
#   a SHOW-READY (scope=full, >=99%) bundle  OR  --informed-override (loud, sub-99%)
#
# Abort at any point: deploy/kill_now.sh + the remote e-stop (B-damping) in your hand.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=lib.sh
source ./lib.sh

DANCE="" STAGE="" ESTOP=0 DRY_RUN=1
GANTRY=0 TETHER=0
GANTRY_PASSED=0 TETHERED_PASSED=0 FREE_PASSED=0
KILL_DAMPING=0 ESTIMATOR=0 OVERRIDE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dance) DANCE="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --estop-confirmed) ESTOP=1; shift ;;
        --gantry-confirmed) GANTRY=1; shift ;;
        --tether-taut-confirmed) TETHER=1; shift ;;
        --gantry-passed) GANTRY_PASSED=1; shift ;;
        --tethered-passed) TETHERED_PASSED=1; shift ;;
        --free-passed) FREE_PASSED=1; shift ;;
        --kill-damping-confirmed) KILL_DAMPING=1; shift ;;
        --estimator-verified) ESTIMATOR=1; shift ;;
        --informed-override) OVERRIDE=1; shift ;;
        --arm) DRY_RUN=0; shift ;;
        *) die "unknown arg $1" ;;
    esac
done
[ -n "$DANCE" ] || die "usage: 10_gantry_test.sh --dance <name> --stage <gantry|ground-tethered|ground-free|push-test> --estop-confirmed [stage flags] [--arm]"
# finding #31: no shell metacharacters reach the remote ssh/docker command
printf '%s' "$DANCE" | grep -Eq '^[A-Za-z0-9_-]{1,64}$' || die "refusing: --dance must match ^[A-Za-z0-9_-]{1,64}$"

# finding #17/#30: each stage is a DIFFERENT safety state — a distinct typed phrase forces
# an explicit, conscious choice (a tired all-day operator cannot muscle-memory past it).
case "$STAGE" in
    gantry)          PHRASE="FEET OFF GROUND" ;;
    ground-tethered) PHRASE="TETHER TAUT PARTIAL WEIGHT" ;;
    ground-free)     PHRASE="GROUND FREE FULL FALL RISK" ;;
    push-test)       PHRASE="PUSH TEST BEGIN" ;;
    *) die "refusing: --stage must be gantry | ground-tethered | ground-free | push-test" ;;
esac

require_human
[ "$ESTOP" = 1 ] || die "refusing: --estop-confirmed not given (e-stop in operator's hand)"

BUNDLE="bundles/${DANCE}"
[ -f "${BUNDLE}/bundle.json" ] || die "no local bundle ${BUNDLE}/bundle.json — run gen_config.py"
SCOPE=$(python3 -c "import json;print(json.load(open('${BUNDLE}/bundle.json')).get('scope','full'))")
[ "$SCOPE" = rehearsal ] && die "refusing: rehearsal bundle is never runnable on a robot"

# ---- MANDATORY per-stage entry gates (the whole point of a staged day) --------------
case "$STAGE" in
gantry)
    [ "$GANTRY" = 1 ] || die "refusing: --gantry-confirmed not given (robot must hang, feet off ground)"
    ;;
ground-tethered)
    [ "$GANTRY_PASSED" = 1 ] || die "GATE: --gantry-passed required — gantry stage (incl. step-3a kill->damping) must be CLEAN first"
    [ "$TETHER" = 1 ] || die "refusing: --tether-taut-confirmed not given (safety line must bear partial weight)"
    # sub-99% (gantry-only) policy is acceptable here IF gantry passed — line still catches a fall.
    ;;
ground-free)
    # This is where real fall risk lives — the hard gate.
    [ "$GANTRY_PASSED" = 1 ]   || die "GATE: --gantry-passed required (gantry must be clean)"
    [ "$TETHERED_PASSED" = 1 ] || die "GATE: --tethered-passed required (ground-tethered must be clean)"
    [ "$TETHER" = 1 ]          || die "refusing: --tether-taut-confirmed not given (keep the line rigged, just slack)"
    [ "$KILL_DAMPING" = 1 ]    || die "GATE: --kill-damping-confirmed required — you must have SEEN kill->damping work on the gantry (step 3a). Until then the remote e-stop is the only stop; do NOT go slack-line."
    [ "$ESTIMATOR" = 1 ]       || die "GATE: --estimator-verified required — confirm the onboard DLIO state-estimator is sane (base_lin_vel matters on the ground, unlike the gantry)"
    if [ "$SCOPE" != full ]; then
        [ "$OVERRIDE" = 1 ] || die "GATE: bundle is scope=${SCOPE} (NOT the >=99% show-ready gate). ground-free needs a show-ready bundle. If you are knowingly testing a sub-99% policy free-standing, pass --informed-override (accepts higher fall risk)."
        log "############################################################"
        log "## WARNING: ground-free with a SUB-99% (scope=${SCOPE}) policy via --informed-override."
        log "## Higher fall risk. Keep the shove gentle, hand on the e-stop, expect to abort."
        log "############################################################"
    fi
    ;;
push-test)
    [ "$FREE_PASSED" = 1 ] || die "GATE: --free-passed required — ground-free must be clean AND repeated before any shove test"
    if [ "$SCOPE" != full ] && [ "$OVERRIDE" != 1 ]; then
        die "GATE: push-test on a sub-99% policy needs --informed-override (real fall risk)"
    fi
    ;;
esac

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
    case "$STAGE" in
        gantry)          echo "   - robot hanging, feet ~5 cm off ground, straps rated and locked?" ;;
        ground-tethered) echo "   - safety line TAUT (partial-weight support), robot cannot hit the floor?"
                         echo "   - 2 m radius clear, floor as vetted (hard flat)?" ;;
        ground-free)     echo "   - safety line SLACK but rigged, robot self-balancing, area 2 m clear?"
                         echo "   - you SAW kill->damping work on the gantry today?" ;;
        push-test)       echo "   - ground-free already clean AND repeated, shove will be GENTLE and expected?" ;;
    esac
    echo "   - nobody within arm radius of the robot?"
    echo "   - e-stop held by the operator, tested today?"
    read -r -p "  type exactly '${PHRASE}' to continue: " ANSWER </dev/tty
    [ "$ANSWER" = "$PHRASE" ] || die "confirmation phrase mismatch"
fi

log "mode: $([ "$DRY_RUN" = 1 ] && echo DRY-RUN || echo LIVE) stage=${STAGE} scope=${SCOPE} dance=${DANCE}"

# Telemetry: mount a writable dir so the controller can log joint pos/vel tracking, IMU,
# and commanded-vs-actual for sim-vs-real comparison. Pulled afterwards by pull_telemetry.sh.
TELE="${PC2_WORKDIR}/telemetry/${DANCE}/${STAGE}"
pc2 "mkdir -p ${TELE}"

# Start the controller container in DAMPING HOLD. It loads the policy but holds damping
# until the operator's remote start sequence (runbook step 6). Exact launch entrypoint is
# confirmed against the controller README on the day (runbook step 3).
pc2 "docker run -d --rm --name g1dance-controller --network host --privileged \
    -v ${PC2_WORKDIR}/bundles/${DANCE}:/bundle:ro \
    -v ${TELE}:/telemetry:rw \
    -e TELEMETRY_DIR=/telemetry -e STAGE=${STAGE} \
    ${CONTROLLER_IMAGE} \
    /bundle/start_controller_damping_hold.sh"

log "controller container start issued (DAMPING HOLD). stage=${STAGE}"
log "watch:      ssh ${PC2_USER}@${PC2_HOST} docker logs -f g1dance-controller"
log "telemetry:  deploy/pull_telemetry.sh --dance ${DANCE} --stage ${STAGE}  (after the run)"
log "abort:      deploy/kill_now.sh   + the remote e-stop (B-damping) in your hand is the"
log "            ONLY guaranteed stop until command-loss->damping is proven on the gantry."
log "GATE:       when this stage is CLEAN, re-run the NEXT stage with its --<stage>-passed"
log "            attestation. Do NOT advance on a single marginal run — repeat first."
