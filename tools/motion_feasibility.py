#!/usr/bin/env python
"""Motion feasibility analysis for a G1 motion CSV (LAFAN1, 30 fps) — AGENT B Phase 2.

CSV rows = [root_pos(3), root_rot xyzw(4), dof_pos(29)].

Reports per-joint velocity vs the motor-class limit and the torque HEADROOM (how close
the motion rides to saturation), and flags a motion that would exceed the limit. A
`retime_to_margin` helper time-warps only the over-limit segments.

IMPORTANT finding (2026-07-10, data/telemetry/feasibility_20260710/): the CURRENT Thriller
deploy motion is ALREADY velocity-feasible (peak ~8.5 < 9.4 rad/s, 0% frames over), and a
sandbox A/B (tools/sim_sandbox) showed that SLOWING it does NOT raise the policy's achieved
fraction (79.7% at 1.0x/1.25x/1.5x). i.e. the "robot does 60-70%" is a POLICY-TRACKING gap,
not a motion-feasibility one — the fix is the Lane-E retrain (arm-scoped reward), not retiming.
This tool remains useful for (a) the vet feasibility gate and (b) FUTURE raw retargets, which
DO exceed the limit (~30% of frames) before the Lane-B clamp/de-glitch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FPS = 30.0
MOTOR_VEL_LIMIT = 3.0 * np.pi        # rad/s, G1 motor-class limit (matches vet_motion)
COMFORT_MARGIN = 0.70                # "comfortable" = ride under 70% of the limit


def feasibility(motion: np.ndarray, fps: float = FPS,
                vel_limit: float = MOTOR_VEL_LIMIT) -> dict:
    j = motion[:, 7:]
    vel = np.abs(np.diff(j, axis=0)) * fps               # (N-1, 29)
    peak = vel.max(axis=0)
    over = (vel > vel_limit).mean(axis=0)
    any_over = float((vel > vel_limit).any(axis=1).mean())
    worst = np.argsort(peak)[::-1][:8]
    return {
        "frames": int(len(motion)),
        "vel_limit_rad_s": round(vel_limit, 2),
        "peak_vel_rad_s": round(float(peak.max()), 2),
        "headroom_frac": round(float(1 - peak.max() / vel_limit), 3),   # <0 => infeasible
        "frames_over_limit_pct": round(100 * any_over, 2),
        "feasible": bool(peak.max() <= vel_limit),
        "comfortable": bool(peak.max() <= COMFORT_MARGIN * vel_limit),
        "worst_joints": [{"dof": int(k), "peak_rad_s": round(float(peak[k]), 2),
                          "pct_over": round(100 * float(over[k]), 2)} for k in worst],
    }


def retime_to_margin(motion: np.ndarray, fps: float = FPS,
                     vel_limit: float = MOTOR_VEL_LIMIT,
                     margin: float = COMFORT_MARGIN) -> tuple[np.ndarray, dict]:
    """Time-warp the WHOLE motion just enough that the peak joint velocity rides at
    `margin*limit`. (Segment-local warping is possible but a global factor is the honest
    minimal change; per the 2026-07-10 finding this does NOT help the current policy, so it
    is opt-in.) Returns (retimed, info)."""
    j = motion[:, 7:]
    peak = (np.abs(np.diff(j, axis=0)) * fps).max()
    target = margin * vel_limit
    factor = max(1.0, float(peak / target))              # >=1 : slow down
    if factor <= 1.0 + 1e-6:
        return motion.copy(), {"retimed": False, "factor": 1.0, "reason": "already comfortable"}
    T = len(motion); newT = int(round(T * factor))
    xi = np.linspace(0, T - 1, newT)
    out = np.stack([np.interp(xi, np.arange(T), motion[:, c])
                    for c in range(motion.shape[1])], axis=1)
    # renormalise the quaternion columns after per-component interp
    q = out[:, 3:7]; out[:, 3:7] = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    return out, {"retimed": True, "factor": round(factor, 3),
                 "new_frames": newT, "peak_before_rad_s": round(float(peak), 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", type=Path)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    m = np.loadtxt(args.csv, delimiter=",")
    rep = feasibility(m, args.fps)
    rep["file"] = str(args.csv)
    print(f"== {args.csv} ==")
    print(f"  peak vel {rep['peak_vel_rad_s']} rad/s (limit {rep['vel_limit_rad_s']}), "
          f"headroom {rep['headroom_frac']*100:.0f}%, {rep['frames_over_limit_pct']}% frames over")
    print(f"  FEASIBLE: {rep['feasible']}   COMFORTABLE (<70% limit): {rep['comfortable']}")
    print("  worst joints (dof, peak rad/s, %over): "
          + ", ".join(f"{w['dof']}:{w['peak_rad_s']}/{w['pct_over']}%" for w in rep['worst_joints'][:5]))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rep, indent=2))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
