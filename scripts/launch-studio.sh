#!/usr/bin/env bash
# G1 Dance Studio — desktop-shortcut launcher.
#
# Clicking the shortcut should ALWAYS give a fresh, working window running the
# current code. So: if a previous instance is holding the port (a stale server
# would otherwise make desktop.py refuse to start, or serve out-of-date code),
# retire it first — by the LISTENER's PID, never `pkill -f` (that has matched
# this very script / the calling shell before). Then launch the app.
set -u
PORT=8735
APP="$HOME/g1-dance"
LOG="$HOME/.cache/g1-dance-studio.log"
mkdir -p "$(dirname "$LOG")"

# free the port if a prior instance holds it
PID=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
if [ -n "${PID:-}" ]; then
  kill "$PID" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8; do
    ss -ltnH "sport = :$PORT" 2>/dev/null | grep -q . || break
    sleep 0.5
  done
fi

# activate the env the app needs (PySide6 Qt dlopens libxcb-cursor from the env)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate g1dance
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$APP" || exit 1
# log to a file so a shortcut launch (no terminal) still leaves a trace to debug
exec python "$APP/ui/desktop.py" "$@" >>"$LOG" 2>&1
