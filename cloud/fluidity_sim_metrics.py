#!/usr/bin/env python
"""Fluidity metrics for a sim rollout trace (cloud/sim_trace_dump.py output).

Produces the v3/v4 decision-table numbers with the SAME math as
tools/fluidity_forensics.py (band_filter / band_rms_by_group /
action_rate_stats / lag_amp copied verbatim — the tool has no CLI, its
helpers are its reusable surface):

  * 2-10 Hz leg ACTION band RMS  — the chatter metric; decision bar <= 0.20
    (s2r-b's level; lower is better)
  * per-group action-rate mean/p95 |da|
  * leg amplitude ratio + lag vs the trace's own reference — bar > 0.5
  * per-group tracking RMS deg (arms number should agree with
    arm_tracking_eval.py within eval noise)
  * base ang-vel roll/pitch band RMS vs the reference pelvis demand (wobble)

Usage:
  ./envs/mjlab/bin/python cloud/fluidity_sim_metrics.py <sim_trace.npz> <out.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

FS = 50.0
DT = 1.0 / FS
BANDS = {"motion_0-2Hz": (None, 2.0), "wobble_2-10Hz": (2.0, 10.0),
         "buzz_10-25Hz": (10.0, None)}
MAX_LAG_TICKS = 20


# ---- helpers copied verbatim from tools/fluidity_forensics.py -----------------
def band_filter(x, lo, hi, fs=FS):
    ny = fs / 2.0
    if lo is None:
        b, a = signal.butter(4, hi / ny, "low")
    elif hi is None:
        b, a = signal.butter(4, lo / ny, "high")
    else:
        b, a = signal.butter(4, [lo / ny, hi / ny], "band")
    return signal.filtfilt(b, a, x, axis=0)


def rms(x, axis=None):
    return float(np.sqrt(np.mean(np.square(x), axis=axis))) if axis is None \
        else np.sqrt(np.mean(np.square(x), axis=axis))


def best_lag(sig_ref, sig, max_lag=MAX_LAG_TICKS):
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


def lag_amp(tgt, q):
    lag, r = best_lag(tgt, q)
    tgt_s = tgt[: len(tgt) - lag] if lag else tgt
    q_s = q[lag:]
    a = tgt_s - tgt_s.mean()
    b = q_s - q_s.mean()
    var = float((a ** 2).sum())
    amp = float((a * b).sum() / var) if var > 1e-9 else float("nan")
    return lag * DT * 1e3, r, amp


def action_rate_stats(actions, idx_groups, mask=None):
    da = np.abs(np.diff(actions, axis=0))
    if mask is not None:
        da = da[mask[1:]]
    out = {}
    for g, idx in idx_groups.items():
        v = da[:, idx]
        out[g] = {"mean": float(v.mean()), "p95": float(np.percentile(v, 95))}
    return out


def band_rms_by_group(x, idx_groups):
    out = {}
    for bname, (lo, hi) in BANDS.items():
        xf = band_filter(x, lo, hi)
        out[bname] = {g: float(rms(xf[:, idx], axis=0).mean())
                      for g, idx in idx_groups.items()}
    return out
# -------------------------------------------------------------------------------


def main(trace_path: str, out_path: str) -> None:
    d = dict(np.load(trace_path, allow_pickle=False))
    names = [str(x) for x in np.asarray(d["joint_names"]).tolist()]
    groups = {
        "legs": [i for i, n in enumerate(names)
                 if ("hip" in n) or ("knee" in n) or ("ankle" in n)],
        "waist": [i for i, n in enumerate(names) if "waist" in n],
        "arms": [i for i, n in enumerate(names)
                 if ("shoulder" in n) or ("elbow" in n) or ("wrist" in n)],
    }
    assert len(groups["legs"]) == 12 and len(groups["arms"]) == 14, groups

    act = d["action"].astype(float)
    q = d["q"].astype(float)
    ref = d["ref_q"].astype(float)
    gyro = d["base_ang_vel"].astype(float)
    ref_w = d["ref_pelvis_ang_vel_w"].astype(float)

    out = {
        "trace": str(trace_path),
        "checkpoint": str(np.asarray(d.get("checkpoint", "?")).item()),
        "task": str(np.asarray(d.get("task", "?")).item()),
        "motion_file": str(np.asarray(d.get("motion_file", "?")).item()),
        "steps": int(act.shape[0]),
        "terminated_at": int(np.asarray(d["terminated_at"]).reshape(-1)[0]),
        "action_band_rms": band_rms_by_group(act, groups),
        "action_rate": action_rate_stats(act, groups),
        "tracking_rms_deg": {
            g: float(np.degrees(np.sqrt(np.mean((q[:, idx] - ref[:, idx]) ** 2))))
            for g, idx in groups.items()},
    }

    # leg amplitude/lag vs the trace's own reference (the decision-bar number)
    leg_rows = {}
    for j in groups["legs"]:
        lag_ms, corr, amp = lag_amp(ref[:, j], q[:, j])
        leg_rows[names[j]] = {"lag_ms": lag_ms, "corr": corr, "amp_ratio": amp}
    amps = [v["amp_ratio"] for v in leg_rows.values() if np.isfinite(v["amp_ratio"])]
    lags = [v["lag_ms"] for v in leg_rows.values()]
    out["leg_vs_ref"] = {
        "per_joint": leg_rows,
        "amp_ratio_mean": float(np.mean(amps)),
        "amp_ratio_min": float(np.min(amps)),
        "lag_ms_mean": float(np.mean(lags)),
    }

    # wobble: roll/pitch ang-vel band RMS, sim vs reference demand
    wob = {}
    for bname, (lo, hi) in BANDS.items():
        wob[bname] = {
            "sim_rms_rad_s": float(rms(band_filter(gyro[:, :2], lo, hi))),
            "ref_rms_rad_s": float(rms(band_filter(ref_w[:, :2], lo, hi))),
        }
    out["wobble_ang_vel"] = wob

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))

    leg_band = out["action_band_rms"]["wobble_2-10Hz"]["legs"]
    lv = out["leg_vs_ref"]
    print(f"[fluidity] steps={out['steps']} terminated_at={out['terminated_at']}")
    print(f"  LEG 2-10Hz action band RMS = {leg_band:.4f}  (bar <= 0.20, lower better)")
    print(f"  LEG amp ratio vs ref mean/min = {lv['amp_ratio_mean']:.2f}/"
          f"{lv['amp_ratio_min']:.2f}  (bar > 0.5)   lag {lv['lag_ms_mean']:.0f} ms")
    print(f"  track RMS deg: " + "  ".join(
        f"{g} {v:.2f}" for g, v in out["tracking_rms_deg"].items()))
    w = wob["wobble_2-10Hz"]
    print(f"  wobble 2-10Hz ang-vel RMS sim {w['sim_rms_rad_s']:.3f} vs "
          f"ref demand {w['ref_rms_rad_s']:.3f} rad/s")
    print(f"FLUIDITY_LEG_BAND={leg_band:.4f} LEG_AMP={lv['amp_ratio_mean']:.3f}")
    print(f"[INFO] wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
