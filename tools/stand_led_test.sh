#!/usr/bin/env bash
# Stand-handoff validation with a HEAD-LED cue so the operator (watching the robot,
# not the terminal) can SEE the handoff moment:
#   BLUE  = set before the run: our controller is (or will be) holding the robot.
#   GREEN = fired the instant the runtime prints "handoff complete" (onboard 'ai'
#           has just taken over). Watch the FEET when it turns green — planted = clean.
#
# LED is the DDS "voice" service (AudioClient.LedControl), independent of the motion
# lowcmd loop. The blue-set process runs+exits BEFORE deploy_runtime inits its DDS;
# the green-set fires AFTER the lowcmd loop is done (handoff-complete line), so there
# is no 50Hz control contention.
#
#   bash tools/stand_led_test.sh
set -u
cd "$(dirname "$0")/.." || exit 1
PY="$HOME/miniconda3/envs/tv/bin/python"
IFACE=enp0s31f6
ISO=data/policies/thriller_standhold_iso

led() { # r g b  — set the head LED and exit (own short-lived DDS participant)
  "$PY" - "$IFACE" "$1" "$2" "$3" <<'PYEOF' 2>/dev/null
import sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
iface, r, g, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
ChannelFactoryInitialize(0, iface)
c = AudioClient(); c.SetTimeout(3.0); c.Init()
c.LedControl(r, g, b)
PYEOF
}

echo ">> LED BLUE (our controller will hold the robot)"
led 0 0 255

echo ">> running stand-handoff test (10s settle + 3s deliberate hold, then handoff)"
while IFS= read -r line; do
  printf '%s\n' "$line"
  case "$line" in
    *"handoff complete"*)
      led 0 255 0        # GREEN — onboard just took over; watch the feet
      echo ">> LED GREEN — onboard balance has the robot. Planted = clean handoff."
      ;;
    "STOP:"*|*" STOP:"*)
      led 255 0 0        # RED — a fault/abort damped the robot
      ;;
  esac
done < <(HANDOFF_HOLD_S=3.0 CONFIRMED_BY_HUMAN=alois "$PY" -u -m pipeline.deploy_runtime \
           --mode ground-run-legodom --exit stand --max-secs 13 --i-will-watch-the-robot \
           --policy "$ISO/policy.onnx" --meta "$ISO/policy_meta.json" \
           --motion-npz "$ISO/thriller_deploy.npz" 2>&1)

sleep 4; led 0 0 0       # LED off a few seconds after the handoff
echo ">> done"
