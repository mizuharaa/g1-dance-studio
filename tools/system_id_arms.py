#!/usr/bin/env python3
"""Per-arm-joint system-ID from ground-run telemetry (q vs target dynamics).

Context (2026-07-06 dance-quality program): the promoted s2r-b policy dances the
full Thriller on hardware but the arms track ~2x worse than sim (arm RMS 13.2 deg,
wrist lag 100-160 ms). Leading hypothesis: soft trained arm gains (kp 14.3-16.8)
against a real arm plant with friction/inertia sim's ideal actuator lacks — the
hardware-proven teleop drives the SAME motors at kp 80/40.

This tool estimates, per arm joint (indices 15-28), from deploy_runtime Telemetry
npz files (q, dq, target, tau_est, kp, kd @ 50 Hz):

  * lag_ms        — cross-correlation lag of q behind the commanded target
  * amp_ratio     — regression slope of q on the lag-shifted target (1.0 = full
                    amplitude; <1 = the joint under-swings its command)
  * track_rms_deg — RMS(target - q), the plant-level tracking error
  * coulomb_nm    — Coulomb friction from fitting delivered torque
                    tau_est ~= J*qdd + b*dq + c*sign(dq) + g0 on moving samples
  * viscous_nms   — the b of the same fit
  * stiction_nm   — |commanded PD torque| while the joint is STUCK (|dq|<0.05)
                    but the target is moving (|dtarget/dt|>0.3): median/p95 —
                    an independent bound on breakaway friction
  * delivery      — tau_est vs kp*err-kd*dq slope (same as capture_stage0 [d])

Aggregates across runs, then prints DR + gain-scale recommendations for the v3b
(arm-plant realism) retrain variant. Writes data/reports/system_id_20260706.json.

Usage:
  python tools/system_id_arms.py data/telemetry/20260706-11*_ground-run-legodom.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "data/reports/system_id_20260706.json"

ARM_IDX = list(range(15, 29))
GROUPS = {
    "shoulder": ("shoulder_pitch", "shoulder_roll", "shoulder_yaw"),
    "elbow": ("elbow",),
    "wrist_roll": ("wrist_roll",),
    "wrist_pitch_yaw": ("wrist_pitch", "wrist_yaw"),
}
MAX_LAG_TICKS = 20          # 400 ms search window @ 50 Hz
STUCK_DQ = 0.05             # rad/s: "not moving"
MOVING_TGT = 0.3            # rad/s: "commanded to move"
MOVING_DQ = 0.10            # rad/s: samples used for the friction fit


def smooth(x, k=5):
    kern = np.ones(k) / k
    return np.convolve(x, kern, mode="same")


def best_lag(sig_ref, sig, max_lag):
    """Lag (ticks >=0) at which sig best correlates with sig_ref shifted back."""
    a = sig_ref - sig_ref.mean()
    b = sig - sig.mean()
    best, best_r = 0, -np.inf
    for lag in range(0, max_lag + 1):
        aa = a[: len(a) - lag] if lag else a
        bb = b[lag:]
        d = np.sqrt((aa ** 2).sum() * (bb ** 2).sum())
        r = float((aa * bb).sum() / d) if d > 0 else 0.0
        if r > best_r:
            best_r, best = r, lag
    return best, best_r


def analyze_joint(t, q, dq, tgt, kp, kd, tau):
    dt = float(np.median(np.diff(t)))
    out = {}
    # --- lag + amplitude ---
    lag, r = best_lag(tgt, q, MAX_LAG_TICKS)
    out["lag_ms"] = lag * dt * 1000.0
    out["lag_corr"] = r
    tgt_s = tgt[: len(tgt) - lag] if lag else tgt
    q_s = q[lag:]
    a = tgt_s - tgt_s.mean()
    b = q_s - q_s.mean()
    var = float((a ** 2).sum())
    out["amp_ratio"] = float((a * b).sum() / var) if var > 1e-9 else float("nan")
    out["track_rms_deg"] = float(np.degrees(np.sqrt(np.mean((tgt - q) ** 2))))
    out["cmd_range_deg"] = float(np.degrees(tgt.max() - tgt.min()))
    # --- friction fit on moving samples: tau ~ J*qdd + b*dq + c*sign(dq) + g0 ---
    dq_f = smooth(dq)
    qdd = np.gradient(dq_f, dt)
    m = np.abs(dq_f) > MOVING_DQ
    out["n_moving"] = int(m.sum())
    if m.sum() > 100:
        X = np.column_stack([qdd[m], dq_f[m], np.sign(dq_f[m]), np.ones(m.sum())])
        coef, *_ = np.linalg.lstsq(X, tau[m], rcond=None)
        pred = X @ coef
        ss = float(np.sum((tau[m] - tau[m].mean()) ** 2))
        out["inertia_est"] = float(coef[0])
        out["viscous_nms"] = float(coef[1])
        out["coulomb_nm"] = float(abs(coef[2]))
        out["friction_fit_r2"] = float(1 - np.sum((tau[m] - pred) ** 2) / ss) if ss > 0 else float("nan")
    else:
        out["coulomb_nm"] = None
    # --- stiction: PD torque applied while stuck but commanded to move ---
    dtgt = np.gradient(smooth(tgt), dt)
    stuck = (np.abs(dq_f) < STUCK_DQ) & (np.abs(dtgt) > MOVING_TGT)
    out["n_stuck"] = int(stuck.sum())
    if stuck.sum() > 20:
        pd_cmd = np.abs(kp * (tgt - q) - kd * dq)[stuck]
        out["stiction_med_nm"] = float(np.median(pd_cmd))
        out["stiction_p95_nm"] = float(np.percentile(pd_cmd, 95))
    # --- delivery (same definition as capture_stage0 [d]) ---
    x = kp * (tgt - q) - kd * dq
    sxx = float(np.dot(x, x))
    cmd_rms = float(np.sqrt(np.mean(x ** 2)))
    out["cmd_rms_nm"] = cmd_rms
    out["delivery"] = float(np.dot(x, tau) / sxx) if (cmd_rms > 0.3 and sxx > 0) else None
    return out


def main(paths):
    per_run = {}
    names = None
    for p in paths:
        d = dict(np.load(p, allow_pickle=False))
        names = [str(x) for x in np.asarray(d["joint_order"]).tolist()]
        kp = np.asarray(d["kp"], float)
        kd = np.asarray(d["kd"], float)
        t = np.asarray(d["t"], float)
        res = {}
        for i in ARM_IDX:
            res[names[i]] = analyze_joint(
                t, d["q"][:, i], d["dq"][:, i], d["target"][:, i],
                kp[i], kd[i], d["tau_est"][:, i])
        per_run[Path(p).name] = res

    # aggregate: mean across runs per joint
    agg = {}
    for i in ARM_IDX:
        n = names[i]
        vals = [per_run[r][n] for r in per_run]
        keys = ("lag_ms", "amp_ratio", "track_rms_deg", "coulomb_nm",
                "viscous_nms", "stiction_med_nm", "stiction_p95_nm", "delivery")
        agg[n] = {k: (float(np.mean([v[k] for v in vals if v.get(k) is not None]))
                      if any(v.get(k) is not None for v in vals) else None)
                  for k in keys}
        agg[n]["n_runs"] = len(vals)

    hdr = (f"{'joint':<28}{'lag_ms':>7}{'amp':>6}{'rms_deg':>8}{'coul_Nm':>8}"
           f"{'visc':>6}{'stic_med':>9}{'stic_p95':>9}{'deliv':>6}")
    print(hdr)
    print("-" * len(hdr))
    for n, s in agg.items():
        def f(k, fmt="%.2f"):
            return (fmt % s[k]) if s.get(k) is not None else "--"
        print(f"{n:<28}{f('lag_ms','%.0f'):>7}{f('amp_ratio'):>6}"
              f"{f('track_rms_deg'):>8}{f('coulomb_nm'):>8}{f('viscous_nms'):>6}"
              f"{f('stiction_med_nm'):>9}{f('stiction_p95_nm'):>9}{f('delivery'):>6}")

    # group rollups + recommendations for the v3b variant
    groups = {}
    for g, pats in GROUPS.items():
        joints = [n for n in agg if any(p in n for p in pats)]
        gv = lambda k: [agg[n][k] for n in joints if agg[n].get(k) is not None]  # noqa: E731
        groups[g] = {
            "joints": joints,
            "lag_ms_mean": float(np.mean(gv("lag_ms"))) if gv("lag_ms") else None,
            "lag_ms_max": float(np.max(gv("lag_ms"))) if gv("lag_ms") else None,
            "amp_ratio_min": float(np.min(gv("amp_ratio"))) if gv("amp_ratio") else None,
            "coulomb_nm_range": [float(np.min(gv("coulomb_nm"))), float(np.max(gv("coulomb_nm")))]
            if gv("coulomb_nm") else None,
            "stiction_med_nm_range": [float(np.min(gv("stiction_med_nm"))),
                                      float(np.max(gv("stiction_med_nm")))]
            if gv("stiction_med_nm") else None,
            "delivery_min": float(np.min(gv("delivery"))) if gv("delivery") else None,
        }
    print("\nGROUP ROLLUP")
    for g, s in groups.items():
        print(f"  {g:<16} lag mean/max {s['lag_ms_mean'] and '%.0f' % s['lag_ms_mean']}/"
              f"{s['lag_ms_max'] and '%.0f' % s['lag_ms_max']} ms   "
              f"amp_min {s['amp_ratio_min'] and '%.2f' % s['amp_ratio_min']}   "
              f"coulomb {s['coulomb_nm_range']}   stiction_med {s['stiction_med_nm_range']}   "
              f"delivery_min {s['delivery_min'] and '%.2f' % s['delivery_min']}")

    out = {"runs": list(per_run), "per_run": per_run, "aggregate": agg, "groups": groups}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {OUT_JSON}")


if __name__ == "__main__":
    main(sys.argv[1:])
