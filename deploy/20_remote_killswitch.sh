#!/usr/bin/env bash
# ARM the software remote kill-switch (L2+B -> damp) in THIS terminal.
# Run BEFORE any motion, keep it visible the whole session. See
# pipeline/remote_estop.py header for why the factory chord is inert in dev mode.
#
# Usage: deploy/20_remote_killswitch.sh [iface]   (default: robot-lan-usb NIC)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
IFACE="${1:-${ROBOT_IFACE:-enx000ec6c3d44a}}"
PY=~/miniconda3/envs/tv/bin/python
[ -x "$PY" ] || { echo "tv env python not found at $PY"; exit 1; }
exec "$PY" deploy/remote_killswitch.py --iface "$IFACE"
