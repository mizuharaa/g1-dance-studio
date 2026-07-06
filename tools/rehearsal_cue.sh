#!/usr/bin/env bash
# Music-cued rehearsal run: launches the full dance and prints a loud PLAY cue at
# the music-start moment (policy tick0 + 4.0s = 2.5s activation ramp + 1.5s standing
# lead-in). A human presses play on a phone/speaker at the cue; expected sync error
# ~ +-0.3s (human reaction). Proper laptop audio is blocked on missing sof-arl.ri
# firmware (see PROJECT_STATE 2026-07-06).
set -u
cd "$(dirname "$0")/.." || exit 1
CUE_LEAD=${CUE_LEAD:-0.4}   # cue this many seconds EARLY to absorb human reaction time
"$HOME/miniconda3/envs/tv/bin/python" -u -m pipeline.deploy_runtime \
  --mode ground-run-legodom --max-secs 52 --i-will-watch-the-robot "$@" 2>&1 \
| while IFS= read -r line; do
    printf '%s\n' "$line"
    case "$line" in
      *"starting leg-odometry policy"*)
        ( sleep "$(echo "4.0 - $CUE_LEAD" | bc)"
          for i in 1 2 3; do printf '\a'; done
          echo ""
          echo "██████████████████████████████████████"
          echo "██  ▶▶▶  PLAY MUSIC NOW  ◀◀◀        ██"
          echo "██████████████████████████████████████"
        ) &
        ;;
    esac
  done
