#!/usr/bin/env python
"""Motion-quality metrics + temporal cleaning for G1 motion CSVs (36-col LAFAN1 layout).

Why: GVHMR estimates pose per-frame (no temporal model), so fast moves produce
frame-to-frame jitter and occasional single-frame outliers (limb flips). Those
survive retargeting and show up as accel/jerk spikes in the deploy CSV — the
"twitch" the operator sees in the preview and a jerky-command risk on hardware.
(Measured 2026-07-10: Thriller vet peak joint vel 56.4 rad/s vs p99 5.8.)

Two halves, importable separately:
  * analyze(motion)      — vel/accel/jerk stats + robust (MAD) accel-spike frames.
  * clean_motion(motion) — accel-spike outlier rejection (same detector as
    analyze, so what we measure is what we fix) + Savitzky-Golay smoothing on
    joints & root position, tangent-space (slerp-aware) SG on the root quaternion.
    Runs in pipeline/prep_motion.py BEFORE the velocity clamp, so the clamp is a
    last-resort guard instead of the only defence.

Savitzky-Golay over One-Euro: this is an offline batch pipeline (no causality
constraint) and SG preserves sharp choreography peaks far better than a causal
low-pass at the same smoothing strength.

CLI:
  python -m tools.motion_quality data/foo.csv            # print report
  python -m tools.motion_quality data/foo.csv --json out.json --clean out.csv --plot out.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FPS = 30.0
# Spike = per-joint accel whose robust z-score (vs that joint's own MAD) exceeds
# this, AND above an absolute floor so a uniformly-gentle motion can't flag noise.
# Derived 2026-07-10 from data/telemetry/motion_quality_20260710: clean LAFAN1
# mocap retargets sit < 6 robust-z; GVHMR outlier frames land at 20-600+.
SPIKE_ROBUST_Z = 10.0
SPIKE_ACCEL_FLOOR = 150.0  # rad/s^2; clean dance p99 accel is ~30-60

# Cleaning defaults (tuned on the repo CSVs, see telemetry dir above):
# SG window 7 @30fps (233 ms) polyorder 3 keeps sharp beats — fidelity RMS
# ~0.01-0.03 rad on the repo dances, sine beats up to ~2.5 Hz pass unblurred.
SG_WINDOW = 7
SG_POLY = 3


def _spike_hit(x: np.ndarray, fps: float) -> np.ndarray:
    """(N-2,D) bool: accel-spike test per sample — robust z vs the column's own
    MAD, plus an absolute floor so gentle motions never flag. hit[i] ~ frame i+1.
    Both analyze() and reject_outliers() use THIS, so what we measure is what
    we fix."""
    aacc = np.abs(np.diff(x, n=2, axis=0)) * fps * fps
    med = np.median(aacc, axis=0)
    mad = np.median(np.abs(aacc - med), axis=0) * 1.4826 + 1e-9
    return ((aacc - med) / mad > SPIKE_ROBUST_Z) & (aacc > SPIKE_ACCEL_FLOOR)


def _spike_mask(x: np.ndarray, fps: float) -> np.ndarray:
    """(N,D) bool sample mask, dilated ±1 frame because a 1-frame impulse
    smears across 3 accel samples."""
    mask = np.zeros(x.shape, dtype=bool)
    mask[1:-1] = _spike_hit(x, fps)
    return binary_dilation(mask, structure=np.ones((3, 1), dtype=bool))


def _derivs(dof: np.ndarray, fps: float):
    vel = np.diff(dof, axis=0) * fps
    acc = np.diff(vel, axis=0) * fps
    jerk = np.diff(acc, axis=0) * fps
    return vel, acc, jerk


def analyze(motion: np.ndarray, fps: float = FPS) -> dict:
    """Vel/accel/jerk profile + spike frames for a (N,36) motion array."""
    dof = motion[:, 7:]
    vel, acc, jerk = _derivs(dof, fps)
    aacc = np.abs(acc)
    hit = _spike_hit(dof, fps)
    spike_frames = np.flatnonzero(hit.any(axis=1)) + 1  # hit[i] ~ frame i+1
    per_joint = hit.sum(axis=0)
    worst = np.argsort(per_joint)[::-1][:5]
    return {
        "frames": int(len(motion)),
        "vel_peak_rad_s": round(float(np.abs(vel).max()), 2),
        "vel_p99_rad_s": round(float(np.percentile(np.abs(vel), 99)), 2),
        "accel_peak_rad_s2": round(float(aacc.max()), 1),
        "accel_p99_rad_s2": round(float(np.percentile(aacc, 99)), 1),
        "jerk_peak_rad_s3": round(float(np.abs(jerk).max()), 0),
        "jerk_p99_rad_s3": round(float(np.percentile(np.abs(jerk), 99)), 0),
        "spike_frames": [int(i) for i in spike_frames],
        "spike_frame_count": int(len(spike_frames)),
        "spike_timestamps_s": [round(float(i) / fps, 2) for i in spike_frames],
        "worst_joints": [
            {"dof_index": int(j), "spikes": int(per_joint[j])}
            for j in worst if per_joint[j] > 0
        ],
    }


def reject_outliers(x: np.ndarray, fps: float = FPS) -> tuple[np.ndarray, int]:
    """Remove accel-spike outliers per column of (N,D) by cubic interpolation
    across the flagged samples (glitches sit ON fast curved moves, so linear
    interp under-cuts the arc). Returns (cleaned, n_frames_touched). Rolling-
    median (hampel) was tried first but its window MAD inflates on legitimately
    fast joints and misses flips there; the accel detector is speed-invariant."""
    from scipy.interpolate import CubicSpline
    mask = _spike_mask(x, fps)
    out = x.copy()
    idx = np.arange(len(x))
    for d in np.flatnonzero(mask.any(axis=0)):
        bad = mask[:, d]
        good = ~bad
        if good.sum() < 4:
            continue  # nothing to anchor interpolation on
        out[bad, d] = CubicSpline(idx[good], x[good, d])(idx[bad])
    return out, int(mask.any(axis=1).sum())


def smooth_quat(quat: np.ndarray, window: int = SG_WINDOW,
                poly: int = SG_POLY) -> np.ndarray:
    """Slerp-aware SG smoothing of an (N,4) xyzw quaternion track: neighbours are
    mapped to the tangent space at each frame (rotvec of relative rotation), the
    SG value-kernel is applied there, and the result mapped back. No naive
    per-component filtering."""
    q = quat.copy()
    # hemisphere continuity first (q and -q are the same rotation)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    n, h = len(q), window // 2
    if n < window:
        return q
    coeffs = savgol_coeffs(window, poly, use="dot")
    R = Rotation.from_quat(q)
    out = q.copy()
    for i in range(h, n - h):  # edges stay raw (a pad frame there is static anyway)
        rel = (R[i].inv() * R[i - h : i + h + 1]).as_rotvec()
        out[i] = (R[i] * Rotation.from_rotvec(coeffs @ rel)).as_quat()
    return out


def clean_motion(motion: np.ndarray, fps: float = FPS) -> tuple[np.ndarray, dict]:
    """Outlier rejection + temporal smoothing on a (N,36) motion.
    Joints & root xyz: hampel then Savitzky-Golay. Root quat: tangent-space SG.
    Returns (cleaned, info) with before/after jerk and a fidelity delta."""
    before = analyze(motion, fps)
    out = motion.copy()
    cols = np.concatenate([out[:, 0:3], out[:, 7:]], axis=1)
    cols, n_outliers = reject_outliers(cols, fps)
    if len(out) >= SG_WINDOW:
        cols = savgol_filter(cols, SG_WINDOW, SG_POLY, axis=0, mode="interp")
    out[:, 0:3], out[:, 7:] = cols[:, :3], cols[:, 3:]
    out[:, 3:7] = smooth_quat(out[:, 3:7])
    after = analyze(out, fps)
    info = {
        "outlier_frames_replaced": n_outliers,
        "jerk_peak_before": before["jerk_peak_rad_s3"],
        "jerk_peak_after": after["jerk_peak_rad_s3"],
        "jerk_p99_before": before["jerk_p99_rad_s3"],
        "jerk_p99_after": after["jerk_p99_rad_s3"],
        "spike_frames_before": before["spike_frame_count"],
        "spike_frames_after": after["spike_frame_count"],
        # tracking fidelity vs raw (excluding the outlier frames we meant to move)
        "dof_rms_delta_rad": round(float(
            np.sqrt(np.mean((out[:, 7:] - motion[:, 7:]) ** 2))), 4),
        "dof_p99_delta_rad": round(float(
            np.percentile(np.abs(out[:, 7:] - motion[:, 7:]), 99)), 4),
    }
    return out, info


def _plot(motion, cleaned, report, out_png, fps=FPS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    joints = [w["dof_index"] for w in report["worst_joints"]][:3] or [0]
    frames = report["spike_frames"]
    c = frames[0] if frames else len(motion) // 2
    lo, hi = max(0, c - 45), min(len(motion), c + 45)
    t = np.arange(lo, hi) / fps
    fig, axes = plt.subplots(len(joints), 1, figsize=(9, 2.6 * len(joints)),
                             squeeze=False, sharex=True)
    for ax, j in zip(axes[:, 0], joints):
        ax.plot(t, motion[lo:hi, 7 + j], label="raw", lw=1, alpha=0.7)
        ax.plot(t, cleaned[lo:hi, 7 + j], label="cleaned", lw=1.2)
        for f in frames:
            if lo <= f < hi:
                ax.axvline(f / fps, color="red", alpha=0.15)
        ax.set_ylabel(f"dof {j} (rad)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", type=Path)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--json", type=Path, help="write the analysis report here")
    ap.add_argument("--clean", type=Path, help="write a cleaned CSV here")
    ap.add_argument("--plot", type=Path, help="before/after plot around worst spike")
    args = ap.parse_args()

    from pipeline.motion_io import load_motion_csv
    motion = load_motion_csv(args.csv)
    report = analyze(motion, args.fps)
    report["file"] = str(args.csv)

    if args.clean or args.plot:
        cleaned, info = clean_motion(motion, args.fps)
        report["clean"] = info
        if args.clean:
            args.clean.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(args.clean, cleaned, delimiter=",")
        if args.plot:
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            _plot(motion, cleaned, report, args.plot, args.fps)

    text = json.dumps(report, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
