#!/usr/bin/env python3
"""Robot-day tool: verify the real robot's standby joint positions match the
sim's default_joint_pos (from policy_meta.json) BEFORE running any policy.

WHY (this is a fall risk): the trained policy assumes the robot's joint zeros are
the sim defaults. If the real encoders/offsets differ, every joint target is
wrong from frame 0 — the 2.5 s activation ramp starts from the wrong place and the
whole motion is off. A few degrees is fine; tens of degrees means STOP and
recalibrate before you run anything.

RUN THIS on robot day, with the robot ON, in STANDBY (damping hold), FEET OFF THE
GROUND on the gantry, on the laptop that talks to the robot over the wired LAN.
It only READS LowState — it never commands a motor.

    # in the env that has unitree_sdk2py + CycloneDDS configured (see ~/robot):
    #   conda activate tv         (per ~/robot/RUNBOOK.md)
    python deploy/check_joint_calibration.py \
        --meta data/policies/thriller/policy_meta.json \
        --iface enp0s31f6 --threshold-deg 8

Exit code 0 = all joints within threshold (GO). Non-zero = a joint is off (NO-GO
until resolved). If unitree_sdk2py isn't importable, it prints how to run it in the
~/robot env and exits non-zero (treat as "not verified", i.e. do NOT skip).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def load_default_pose(meta_path: Path) -> dict[str, float]:
    """Pull the ordered default_joint_pos (radians) out of policy_meta.json.

    Accepts either a {joint_name: value} dict or a list aligned to a joint_order
    list — whatever the exporter wrote. Returns name->radians.
    """
    meta = json.loads(meta_path.read_text())
    default = (meta.get("default_joint_pos") or meta.get("default_joint_pos_rad")
               or meta.get("default_qpos"))
    if default is None:
        raise SystemExit(f"no default_joint_pos[_rad] in {meta_path}")
    if isinstance(default, dict):
        return {k: float(v) for k, v in default.items()}
    order = meta.get("joint_order") or meta.get("joint_order_29dof") or meta.get("joint_names")
    if not order or len(order) != len(default):
        raise SystemExit("default_joint_pos is a list but joint_order is missing/mismatched")
    return {name: float(v) for name, v in zip(order, default)}


def read_lowstate_q(iface: str, timeout_s: float) -> list[float]:
    """Read one LowState over CycloneDDS and return the 29 motor positions (rad).

    Uses unitree_sdk2py exactly as ~/robot's working teleop stack does. Kept in a
    function so a missing SDK degrades to a clear message rather than a traceback.
    """
    try:
        from unitree_sdk2py.core.channel import (  # type: ignore
            ChannelFactoryInitialize, ChannelSubscriber)
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_  # type: ignore
    except Exception as exc:  # noqa: BLE001 - want the friendly message
        raise SystemExit(
            "unitree_sdk2py not importable (%s).\n"
            "Run this in the env that has it — per ~/robot/RUNBOOK.md:\n"
            "    conda activate tv   # laptop env with unitree_sdk2python + CycloneDDS\n"
            "and make sure the wired LAN is up (cat /sys/class/net/%s/carrier == 1)."
            % (exc, iface)
        )

    ChannelFactoryInitialize(0, iface)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()
    msg = sub.Read(int(timeout_s * 1000))
    if msg is None:
        raise SystemExit(
            "no LowState received on %s within %.1fs — robot off, wrong iface, or "
            "LAN down (check `ping -c2 192.168.123.164` and the carrier). NO-GO."
            % (iface, timeout_s))
    return [float(m.q) for m in msg.motor_state[:29]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="policy_meta.json with default_joint_pos")
    ap.add_argument("--iface", default="enp0s31f6", help="wired LAN iface to the robot")
    ap.add_argument("--threshold-deg", type=float, default=8.0,
                    help="per-joint tolerance; above this = NO-GO")
    ap.add_argument("--timeout-s", type=float, default=3.0)
    a = ap.parse_args()

    default = load_default_pose(Path(a.meta))
    q = read_lowstate_q(a.iface, a.timeout_s)

    # Compare in policy_meta's joint order. We need the ordered names; if default
    # was a dict we still need the LowState index order — the exporter's joint_order
    # is authoritative and must match the SDK's motor index order (documented in
    # policy_meta). If names are absent, fall back to positional.
    meta = json.loads(Path(a.meta).read_text())
    order = (meta.get("joint_order") or meta.get("joint_order_29dof")
             or meta.get("joint_names") or list(default.keys()))
    if len(order) != len(q):
        raise SystemExit(
            f"joint count mismatch: policy_meta has {len(order)}, LowState has {len(q)}")

    thr = math.radians(a.threshold_deg)
    worst = 0.0
    bad = []
    print(f"{'joint':<28} {'sim(deg)':>9} {'real(deg)':>9} {'delta':>8}")
    for name, real in zip(order, q):
        exp = default[name] if isinstance(default, dict) and name in default else default.get(name, 0.0) if isinstance(default, dict) else 0.0
        d = real - exp
        worst = max(worst, abs(d))
        flag = "  <-- OFF" if abs(d) > thr else ""
        if abs(d) > thr:
            bad.append(name)
        print(f"{name:<28} {math.degrees(exp):>9.1f} {math.degrees(real):>9.1f} "
              f"{math.degrees(d):>7.1f}{flag}")

    print("-" * 60)
    if bad:
        print(f"NO-GO: {len(bad)} joint(s) exceed {a.threshold_deg:.0f} deg "
              f"(worst {math.degrees(worst):.1f} deg): {', '.join(bad)}")
        print("Do NOT run the policy. Recalibrate/re-zero the robot (see ~/robot) or "
              "investigate the offset before activation.")
        return 2
    print(f"GO: all joints within {a.threshold_deg:.0f} deg "
          f"(worst {math.degrees(worst):.1f} deg). Standby pose matches the sim default.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
