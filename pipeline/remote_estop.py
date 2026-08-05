"""Remote e-stop decoding for the G1 in CUSTOM/dev control mode.

WHY THIS EXISTS (hardware finding, 2026-08-05): while our controller (laptop
deploy_runtime or the PC2 docker controller) holds low-level control, the
FACTORY handler for the remote's L2+B damping chord is suspended — the remote
is INERT as a stop. But the remote's raw state still arrives on the wire: the
unitree_hg LowState carries a 40-byte `wireless_remote` buffer (buttons in
bytes [2]/[3], layout per the official SDK example
unitree_sdk2_python/example/wireless_controller/wireless_controller.py).

So we watch the chord OURSELVES and damp through our own proven path:
  - in-loop: pipeline/deploy_runtime.read_state() feeds every LowState through
    RemoteKill; a debounced L2+B raises RemoteKillRequested, which the motion
    modes' except/finally turns into _finalize_and_exit -> _damp_burst FIRST.
  - out-of-loop (hung process / PC2 docker controller): deploy/remote_killswitch.py
    is an INDEPENDENT laptop process with its own DDS subscription that fires
    deploy/kill_now.sh + SIGTERMs any local runtime on the same chord.

HONEST LIMITS: both layers need the DDS bus (wired LAN) alive and need the
robot's MCU to still be forwarding remote state into LowState. The POWER
SWITCH is the only hardware-guaranteed stop in dev mode. Validate the chord
end-to-end on the gantry (checklist §3a) before any motion, every robot day.

Button bit layout (from the SDK example, G1/unitree_hg):
  byte2: R1=b0 L1=b1 Start=b2 Select=b3 R2=b4 L2=b5 F1=b6 F3=b7
  byte3: A=b0  B=b1  X=b2   Y=b3     Up=b4 Right=b5 Down=b6 Left=b7
"""
from __future__ import annotations

import os
import time

# consecutive samples of the chord required to fire. At the 50 Hz control rate
# 3 ticks = 60 ms — debounces radio glitches without adding meaningful latency.
KILL_TICKS = int(os.environ.get("REMOTE_ESTOP_TICKS", "3"))
ENABLED = os.environ.get("REMOTE_ESTOP", "1") == "1"


class RemoteKillRequested(Exception):
    """Operator pressed the damping chord (L2+B) on the remote."""


def decode_buttons(wireless_remote) -> dict:
    """Decode the 40-byte LowState.wireless_remote button bytes -> {name: 0/1}.
    Tolerates any indexable byte container; returns all-zeros if too short
    (remote off / no data yet)."""
    try:
        b2, b3 = int(wireless_remote[2]) & 0xFF, int(wireless_remote[3]) & 0xFF
    except (IndexError, TypeError, ValueError):
        return {k: 0 for k in ("R1", "L1", "Start", "Select", "R2", "L2", "F1",
                               "F3", "A", "B", "X", "Y", "Up", "Right", "Down", "Left")}
    return {
        "R1": (b2 >> 0) & 1, "L1": (b2 >> 1) & 1, "Start": (b2 >> 2) & 1,
        "Select": (b2 >> 3) & 1, "R2": (b2 >> 4) & 1, "L2": (b2 >> 5) & 1,
        "F1": (b2 >> 6) & 1, "F3": (b2 >> 7) & 1,
        "A": (b3 >> 0) & 1, "B": (b3 >> 1) & 1, "X": (b3 >> 2) & 1,
        "Y": (b3 >> 3) & 1, "Up": (b3 >> 4) & 1, "Right": (b3 >> 5) & 1,
        "Down": (b3 >> 6) & 1, "Left": (b3 >> 7) & 1,
    }


def chord_down(wireless_remote) -> bool:
    """True while L2+B are BOTH held."""
    btn = decode_buttons(wireless_remote)
    return bool(btn["L2"] and btn["B"])


class RemoteKill:
    """Debounced L2+B detector. Feed every LowState's wireless_remote buffer;
    update() returns True once the chord has been held KILL_TICKS consecutive
    samples. Releasing the chord resets the counter."""

    def __init__(self, ticks: int = KILL_TICKS):
        self.ticks = max(1, int(ticks))
        self._count = 0
        self.fired_at: float | None = None

    def update(self, wireless_remote) -> bool:
        if chord_down(wireless_remote):
            self._count += 1
        else:
            self._count = 0
        if self._count >= self.ticks:
            if self.fired_at is None:
                self.fired_at = time.time()
            return True
        return False

    def check(self, wireless_remote) -> None:
        """update() but raising — the deploy_runtime in-loop hook."""
        if self.update(wireless_remote):
            raise RemoteKillRequested(
                f"remote L2+B held {self._count} consecutive samples — damping now")
