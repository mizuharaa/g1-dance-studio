#!/usr/bin/env bash
# ============================================================================
# INVESTOR DEMO LAUNCHER — Thriller, v3e policy, real track via laptop/aux.
# This is the EXACT proven path that ran clean 2589/2589 ticks with music on
# 2026-07-07 (telemetry 20260707-103403). Nothing experimental. No untested
# runtime code. Ends with the proven smooth ramp-to-damping handoff.
#
#   bash demo.sh
#
# Sequence after you press ENTER:
#   1. robot moves to the ready pose (~4 s)
#   2. dances Thriller to the real track (music auto-fires on the beat, ~50 s)
#   3. smooth ramp to damping, hands back to onboard balance
# ============================================================================
set -u
cd "$(dirname "$0")" || exit 1

RED=$'\e[1;31m'; GRN=$'\e[1;32m'; YEL=$'\e[1;33m'; RST=$'\e[0m'

cat <<BANNER

${GRN}==================== G1 THRILLER — INVESTOR DEMO ====================${RST}
  policy : v3e  (show-ready, sha e68335aa..)
  audio  : real Thriller track -> laptop/aux speaker (auto-synced)
  proven : ran clean full-dance with music earlier today
${RED}  SAFETY: the damping remote is the ONLY stop. Keep it in your hand.${RST}
${GRN}====================================================================${RST}

BANNER

# Pre-flight (no robot contact) — fail loud before anyone is on stage.
ping -c1 -W2 192.168.123.164 >/dev/null 2>&1 \
  || { echo "${RED}ABORT: robot PC2 (192.168.123.164) unreachable — check the robot-lan cable.${RST}"; exit 1; }
for f in data/policies/thriller/policy.onnx data/policies/thriller/thriller_deploy.npz \
         data/dances/20260704-18f65bbd/audio/music.wav; do
  [ -s "$f" ] || { echo "${RED}ABORT: missing $f${RST}"; exit 1; }
done
echo "${GRN}pre-flight OK${RST} — robot reachable, policy + real audio present."
echo
echo "${YEL}CONFIRM BEFORE PRESSING ENTER:${RST}"
echo "   [ ] robot standing on the tether, tether taut enough to catch a fall"
echo "   [ ] damping remote in your hand, thumb ready"
echo "   [ ] 2 m area clear (robot may catch-step ~1 m rightward at the end)"
echo
read -r -p "Press ENTER to run the show (Ctrl-C to abort) ... " _

exec env \
  AUDIO_MODE=laptop \
  AUDIO_LATENCY_COMP=0.0 \
  ARM_ACTION_CAP_SCALE=2.2 \
  CONFIRMED_BY_HUMAN=alois \
  bash tools/show_run.sh
