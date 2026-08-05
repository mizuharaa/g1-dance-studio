#!/usr/bin/env python3
"""Independent remote kill-switch watchdog (laptop-side). ARM THIS IN ITS OWN
TERMINAL BEFORE ANY MOTION — see docs/ROBOT_DAY_CHECKLIST.md.

WHY: in custom/dev control the factory L2+B damping handler is INERT
(hardware finding 2026-08-05). This process is a SEPARATE OS process with its
own DDS subscription to rt/lowstate, so it works even when:
  - the policy runs in the PC2 docker controller (laptop runtime not involved);
  - the laptop deploy_runtime is hung/GIL-stuck (its in-loop check can't run).

ON L2+B (debounced):
  1. SIGTERM every local pipeline/deploy_runtime process (its handler damps).
  2. Fire deploy/kill_now.sh (docker stop -> the PC2 controller's own
     SIGTERM-damp window, then SIGKILL).
  3. Send LocoClient.Damp() via the SDK RPC — this is what stops the robot
     when the FACTORY controller is in charge (normal running mode), where
     steps 1-2 have nothing to kill (observed 2026-08-05: "kill issued" but
     the factory controller kept obeying the sticks).
All actions are idempotent and safe when their target isn't running/active.

HONEST LIMITS: needs the wired LAN + DDS alive and the MCU still forwarding
remote state into LowState. The POWER SWITCH remains the only
hardware-guaranteed stop. Validate end-to-end on the gantry (checklist §3a)
before trusting it: press L2+B, watch THIS terminal print KILL and the robot
damp.

Run (tv env has cyclonedds; SDK path injected below):
  deploy/20_remote_killswitch.sh [iface]     # wrapper, recommended
  # or directly:
  ~/miniconda3/envs/tv/bin/python deploy/remote_killswitch.py --iface enx000ec6c3d44a
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
SDK_PATH = os.environ.get(
    "UNITREE_SDK_PATH", "/home/alois/meta-quest-teleoperate/unitree_sdk2_python")
sys.path.insert(0, SDK_PATH)

from pipeline.remote_estop import RemoteKill, decode_buttons  # noqa: E402


def _local_runtime_pids() -> list[int]:
    out = subprocess.run(
        ["pgrep", "-f", "[d]eploy_runtime"], capture_output=True, text=True)
    return [int(p) for p in out.stdout.split()] if out.returncode == 0 else []


_LOCO = None


def _damp_via_sdk() -> None:
    """Damp the FACTORY controller (normal running mode) via the loco RPC.
    Best-effort: when OUR controller holds low-level control the loco service
    is released and this may time out — steps 1-2 own that case."""
    global _LOCO
    try:
        if _LOCO is None:
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
            _LOCO = LocoClient()
            _LOCO.SetTimeout(1.0)
            _LOCO.Init()
        _LOCO.Damp()
        _LOCO.Damp()   # fire twice — an e-stop RPC deserves redundancy
        print("  LocoClient.Damp() sent (factory-controller stop)", flush=True)
    except Exception as exc:  # noqa: BLE001 — never let one layer block the others
        print(f"  LocoClient.Damp() unavailable ({exc}) — expected when OUR "
              "controller holds low-level control.", flush=True)


def fire(reason: str) -> None:
    print(f"\n{'!' * 60}\nKILL: {reason}\n{'!' * 60}", flush=True)
    # factory-mode damp FIRST — it acts in milliseconds; the docker stop takes seconds
    _damp_via_sdk()
    for pid in _local_runtime_pids():
        try:
            os.kill(pid, signal.SIGTERM)   # runtime's handler -> damp + os._exit
            print(f"  SIGTERM -> local deploy_runtime pid {pid}", flush=True)
        except ProcessLookupError:
            pass
    kill_sh = REPO / "deploy" / "kill_now.sh"
    print(f"  firing {kill_sh} (PC2 controller docker stop)", flush=True)
    subprocess.run(["bash", str(kill_sh)], check=False)
    print("  kill issued. VISUALLY verify the robot is still before approaching.",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iface", default=os.environ.get("ROBOT_IFACE", "enx000ec6c3d44a"),
                    help="wired NIC on the robot LAN (192.168.123.x)")
    ap.add_argument("--once", action="store_true",
                    help="exit after firing once (default: keep watching, re-fire on a new chord)")
    args = ap.parse_args()

    from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,  # noqa: E402
                                             ChannelSubscriber)
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_  # noqa: E402

    ChannelFactoryInitialize(0, args.iface)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()

    print(f"remote killswitch ARMED on {args.iface} — press L2+B on the remote to damp.\n"
          "This is a WATCHDOG; keep this terminal visible. Ctrl-C disarms (loudly).",
          flush=True)
    kill = RemoteKill()
    fired = False
    last_beat = 0.0
    last_msg_at = None
    try:
        while True:
            msg = sub.Read(1.0)
            now = time.time()
            if msg is None:
                if last_msg_at and now - last_msg_at > 3:
                    print(f"WARN: no LowState for {now - last_msg_at:.0f}s — "
                          "robot off / LAN down: killswitch is BLIND.", flush=True)
                    last_msg_at = now  # rate-limit the warning
                continue
            last_msg_at = now
            wr = getattr(msg, "wireless_remote", None)
            if kill.update(wr):
                if not fired:
                    fire("remote L2+B chord (debounced)")
                    fired = True
                    if args.once:
                        return 0
            elif fired and not any(decode_buttons(wr).values()):
                fired = False   # chord fully released -> re-arm
                print("chord released — killswitch re-armed.", flush=True)
            if now - last_beat > 5:
                btn = decode_buttons(wr)
                held = [k for k, v in btn.items() if v]
                print(f"[alive {time.strftime('%H:%M:%S')}] remote ok"
                      f"{' — held: ' + '+'.join(held) if held else ''}", flush=True)
                last_beat = now
    except KeyboardInterrupt:
        print("\nDISARMED (Ctrl-C). The software L2+B stop is NO LONGER watching. "
              "Only kill_now.sh and the power switch remain.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
