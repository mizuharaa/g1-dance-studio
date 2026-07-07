#!/usr/bin/env bash
# SHOW RUNNER — full dance with synchronized music/cue (rehearsal_cue.sh successor).
#
# The operator watches the ROBOT, not the terminal (2026-07-06 rehearsal finding), so
# the cue comes from the robot itself:
#   AUDIO_MODE=robot   music out of the G1's own speaker (the product mode)
#   AUDIO_MODE=led     G1 head LED: blue countdown flashes, GREEN = press play
#   AUDIO_MODE=laptop  play via laptop speakers (needs SOF firmware fix — docs/SHOW_AUDIO.md)
#   AUDIO_MODE=banner  legacy terminal bell + banner (default; zero robot audio deps)
#
# Timeline contract: music starts at policy tick0 + 4.0 s (2.5 s activation ramp +
# 1.5 s standing lead-in). tick0 is anchored the instant the runtime prints its
# "starting leg-odometry policy" line; we capture date +%s.%N right there and hand it
# to pipeline/show_audio.py so process-spawn time cannot skew the cue.
#
# Knobs (env):
#   DANCE_ID            dance record with attached audio (default: Thriller)
#   AUDIO_LATENCY_COMP  seconds to start robot/laptop audio EARLY (playback chain
#                       startup latency; calibrate per docs/SHOW_AUDIO.md), default 0
#   AUDIO_VOLUME        robot speaker volume 0-100 (optional)
#   CUE_LEAD            banner mode: fire early by this (human reaction), default 0.4
#   SHOW_VIDEO          if set, path to a side-by-side comparison video (reference|sim);
#                       played FULL-SCREEN on the external display at tick0 via
#                       tools/show_display.py, in sync with the robot + music. The video
#                       carries its own 4.0s pre-dance lead-in, so it starts at tick0
#                       (NOT tick0+4 like the music). Empty/unset = no video (unchanged).
#   SHOW_DISPLAY        xrandr output for the video (e.g. HDMI-1); default auto-picks a
#                       non-primary monitor, else falls back to the primary/full-screen.
#
# Abort: any runtime STOP/exit kills the cue helper -> its SIGTERM handler sends
# PlayStop / LED off immediately, and the video player is terminated the same way.
# Same guarantee rehearsal_cue never had.
set -u
cd "$(dirname "$0")/.." || exit 1

AUDIO_MODE=${AUDIO_MODE:-banner}
DANCE_ID=${DANCE_ID:-20260704-18f65bbd}
AUDIO_LATENCY_COMP=${AUDIO_LATENCY_COMP:-0.0}
CUE_LEAD=${CUE_LEAD:-0.4}
# End-of-run handoff, passed through to deploy_runtime --exit. DEFAULT "damp" reproduces
# the frozen proven ramp-to-damping — demo.sh does NOT set EXIT_MODE, so the demo path is
# byte-for-byte unchanged. Set EXIT_MODE=stand to opt into the stand-and-hand-back handoff
# (only honored if the dance motion ends standing; UNVALIDATED on hardware — tethered test
# with the user present required for first live use).
EXIT_MODE=${EXIT_MODE:-damp}
PY=${PY:-"$HOME/miniconda3/envs/tv/bin/python"}

case "$AUDIO_MODE" in robot|led|laptop|banner) ;; *)
  echo "AUDIO_MODE must be robot|led|laptop|banner (got '$AUDIO_MODE')" >&2; exit 2 ;;
esac

CUE_PID=""
VIDEO_PID=""
stop_cue() {
  if [ -n "$CUE_PID" ] && kill -0 "$CUE_PID" 2>/dev/null; then
    kill -TERM "$CUE_PID" 2>/dev/null   # -> show_audio handler: PlayStop + LED off
    wait "$CUE_PID" 2>/dev/null
  fi
  CUE_PID=""
  if [ -n "$VIDEO_PID" ] && kill -0 "$VIDEO_PID" 2>/dev/null; then
    kill -TERM "$VIDEO_PID" 2>/dev/null   # -> show_display handler: player terminate()
    wait "$VIDEO_PID" 2>/dev/null
  fi
  VIDEO_PID=""
}
trap stop_cue EXIT INT TERM

echo "SHOW RUN: dance=$DANCE_ID audio=$AUDIO_MODE latency_comp=${AUDIO_LATENCY_COMP}s"

# Process substitution (not a pipe) so this loop runs in THIS shell and CUE_PID/traps work.
while IFS= read -r line; do
  printf '%s\n' "$line"
  case "$line" in
    *"starting leg-odometry policy"*)
      T0=$(date +%s.%N)     # tick0 anchor — captured the moment the line appeared
      AUDIO_LATENCY_COMP="$AUDIO_LATENCY_COMP" \
      "$PY" -u -m pipeline.show_audio cue \
        --mode "$AUDIO_MODE" --dance-id "$DANCE_ID" --t0-epoch "$T0" \
        --cue-lead "$CUE_LEAD" ${AUDIO_VOLUME:+--volume "$AUDIO_VOLUME"} &
      CUE_PID=$!
      # Parallel VIDEO cue on the SAME tick0 anchor: the side-by-side already bakes in
      # the 4.0s pre-dance lead-in, so it starts at tick0 (not tick0+4) to land with the
      # robot; its own lead-in then keeps it aligned with the music at tick0+4.
      if [ -n "${SHOW_VIDEO:-}" ]; then
        "$PY" -u tools/show_display.py "$SHOW_VIDEO" --at-epoch "$T0" \
          ${SHOW_DISPLAY:+--display "$SHOW_DISPLAY"} &
        VIDEO_PID=$!
      fi
      ;;
    "STOP:"*|*" STOP:"*)    # runtime abort line -> silence the show immediately
      stop_cue
      ;;
  esac
done < <("$PY" -u -m pipeline.deploy_runtime \
           --mode ground-run-legodom --max-secs "${MAX_SECS:-52}" --exit "$EXIT_MODE" \
           --i-will-watch-the-robot "$@" 2>&1)

# Runtime finished (clean end or abort): music must never outlive the motion by more
# than the show tail; in robot/laptop modes the helper exits on its own after the
# track ends, but a still-running helper here means the run ended early -> stop it.
stop_cue
