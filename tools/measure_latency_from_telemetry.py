#!/usr/bin/env python3
"""Estimate the effective command->response latency from a deploy telemetry npz.

The policy fights whatever delay sits between "action computed from obs at tick k"
and "that action visibly moving the joints". We recover it directly from logged data
by cross-correlating each joint's COMMANDED target trajectory against its MEASURED
position, over a lag sweep, and taking the lag that maximizes correlation. Reported
per-joint and aggregated (leg joints matter most for balance).

This is passive (no robot), and it is the real number to set sim delay-randomization.
"""
import sys
import numpy as np

def main(path):
    d = np.load(path)
    q = d["q"]; target = d["target"]; t = d["t"]; jo = list(map(str, d["joint_order"]))
    dt = float(np.median(np.diff(t)))
    N = len(t)
    # restrict to the dancing region (skip the move-to-default settle at the very start)
    lo = int(3.0 / dt)
    q = q[lo:]; target = target[lo:]
    maxlag = int(round(0.12 / dt))  # sweep 0..120ms

    leg = [i for i, n in enumerate(jo) if any(s in n for s in
           ("hip", "knee", "ankle"))]

    def best_lag(i):
        a = target[:, i] - target[:, i].mean()
        b = q[:, i] - q[:, i].mean()
        if a.std() < 1e-4 or b.std() < 1e-4:
            return None, 0.0
        best, bestc = 0, -2
        for L in range(0, maxlag + 1):
            # target leads response by L ticks: corr(target[:-L], q[L:])
            if L == 0:
                aa, bb = a, b
            else:
                aa, bb = a[:-L], b[L:]
            c = np.corrcoef(aa, bb)[0, 1]
            if c > bestc:
                bestc, best = c, L
        return best, bestc

    print(f"file: {path}")
    print(f"dt={dt*1000:.1f}ms  ticks(dance)={len(q)}  lag sweep 0..{maxlag*dt*1000:.0f}ms")
    print(f"{'joint':28s} {'lag_ms':>7s} {'corr':>6s} {'range_rad':>9s}")
    leg_lags = []
    for i in range(len(jo)):
        L, c = best_lag(i)
        if L is None:
            continue
        rng = float(target[:, i].max() - target[:, i].min())
        tag = " <-leg" if i in leg else ""
        if i in leg and rng > 0.2:
            leg_lags.append((L * dt * 1000, rng))
        if rng > 0.15:
            print(f"{jo[i]:28s} {L*dt*1000:7.1f} {c:6.2f} {rng:9.2f}{tag}")
    if leg_lags:
        lags = np.array([x[0] for x in leg_lags])
        wts = np.array([x[1] for x in leg_lags])
        wmean = float((lags * wts).sum() / wts.sum())
        print(f"\nLEG joints (range>0.2rad): n={len(leg_lags)}  "
              f"lag min/median/max = {lags.min():.0f}/{np.median(lags):.0f}/{lags.max():.0f} ms  "
              f"range-weighted mean = {wmean:.0f} ms")
        print("=> effective command->response delay the balance policy experiences")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "data/telemetry/20260709-192744_ground-run-legodom.npz")
