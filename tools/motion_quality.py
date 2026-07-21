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

# Choreography guard (audited 2026-07-20, experiments/clean_motion_audit_20260720):
# a k-frame position impulse contaminates k+2 consecutive accel samples, so a
# 1-2 frame GVHMR flip yields a core run of 3-4 flagged samples. A genuine
# velocity-feasible dance "hit" (>=200 ms) spans >=5. Flagged runs LONGER than
# this are coherent motion, not impulses — they are protected from rejection
# (SG smoothing and the downstream velocity clamp still apply to them).
MAX_GLITCH_CORE = 4
# Second guard criterion (2026-07-21, the 8.4 rad/s Thriller knee-snap lesson):
# even a SHORT flagged run is only a glitch if the sample deviates from its
# neighbour-spline by many times the joint's local motion scale. A genuine
# impulsive snap sits ON the motion (deviation ~1-2x the local per-frame delta);
# GVHMR limb flips deviate 10-600x. Short runs below this ratio are protected.
GLITCH_DEV_RATIO = 3.0


def _spike_hit(x: np.ndarray, fps: float) -> np.ndarray:
    """(N-2,D) bool: accel-spike test per sample — robust z vs the column's own
    MAD, plus an absolute floor so gentle motions never flag. hit[i] ~ frame i+1.
    Both analyze() and reject_outliers() use THIS, so what we measure is what
    we fix."""
    aacc = np.abs(np.diff(x, n=2, axis=0)) * fps * fps
    med = np.median(aacc, axis=0)
    mad = np.median(np.abs(aacc - med), axis=0) * 1.4826 + 1e-9
    return ((aacc - med) / mad > SPIKE_ROBUST_Z) & (aacc > SPIKE_ACCEL_FLOOR)


def _spike_mask(x: np.ndarray, fps: float,
                info: dict | None = None) -> np.ndarray:
    """(N,D) bool sample mask of REJECTABLE glitch samples, dilated ±1 frame
    because a 1-frame impulse smears across 3 accel samples.

    Choreography guard: contiguous flagged runs whose core exceeds
    MAX_GLITCH_CORE samples are coherent fast MOTION (a real dance hit), not a
    1-2 frame estimator flip — those are dropped from the mask (protected).
    If ``info`` is given, per-run bookkeeping is appended to
    info["rejected_runs"] / info["protected_runs"] as (frame_start, frame_end,
    dof_column) tuples on the frame clock."""
    from scipy.interpolate import CubicSpline
    from scipy.ndimage import binary_closing
    core = np.zeros(x.shape, dtype=bool)
    core[1:-1] = _spike_hit(x, fps)
    idx = np.arange(len(x))
    for d in np.flatnonzero(core.any(axis=0)):
        col = core[:, d]
        # one physical event = one run: a smooth pulse's |accel| crosses zero
        # mid-rise/fall, splitting its core into fragments — close gaps <=2
        # samples before judging run length.
        closed = binary_closing(col, structure=np.ones(3, dtype=bool))
        edges = np.flatnonzero(np.diff(closed.astype(int)))
        starts = list(edges[closed[edges + 1]] + 1) + ([0] if closed[0] else [])
        for s in sorted(starts):
            e = s
            while e + 1 < len(closed) and closed[e + 1]:
                e += 1
            run = (int(s), int(e), int(d))
            if e - s + 1 > MAX_GLITCH_CORE:
                col[s:e + 1] = False
                if info is not None:
                    info.setdefault("protected_runs", []).append(run)
                continue
            # short run: glitch only if it deviates from the neighbour-spline
            # by >= GLITCH_DEV_RATIO x the joint's local per-frame motion scale
            good = np.ones(len(x), dtype=bool)
            good[max(0, s - 1):e + 2] = False
            lo, hi = max(0, s - 30), min(len(x), e + 31)
            scale = np.percentile(np.abs(np.diff(x[lo:hi, d])), 95) + 1e-6
            if good.sum() >= 4:
                spl = CubicSpline(idx[good], x[good, d])(idx[s:e + 1])
                dev = float(np.max(np.abs(x[s:e + 1, d] - spl)))
            else:
                dev = float("inf")
            if dev / scale >= GLITCH_DEV_RATIO:
                col[s:e + 1] = True   # true flip: reject whole event incl. gaps
                if info is not None:
                    info.setdefault("rejected_runs", []).append(run)
            else:
                col[s:e + 1] = False  # choreography-scale snap: protect
                if info is not None:
                    info.setdefault("protected_runs", []).append(run)
    return binary_dilation(core, structure=np.ones((3, 1), dtype=bool))


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


def reject_outliers(x: np.ndarray, fps: float = FPS,
                    info: dict | None = None) -> tuple[np.ndarray, int]:
    """Remove accel-spike outliers per column of (N,D) by cubic interpolation
    across the flagged samples (glitches sit ON fast curved moves, so linear
    interp under-cuts the arc). Returns (cleaned, n_frames_touched). Rolling-
    median (hampel) was tried first but its window MAD inflates on legitimately
    fast joints and misses flips there; the accel detector is speed-invariant.
    Coherent fast moves are protected by _spike_mask's choreography guard;
    pass ``info`` to receive rejected/protected run bookkeeping."""
    from scipy.interpolate import CubicSpline
    mask = _spike_mask(x, fps, info)
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


# joint-group columns within the 29-dof block (G1 LAFAN1 order)
_GROUPS = {"legs": list(range(12)), "waist": [12, 13, 14],
           "arms": list(range(15, 29))}
RETENTION_WARN = 0.90     # amp retention below this in any 5 s tile => warn


def _fidelity_retention(raw: np.ndarray, cleaned: np.ndarray,
                        fps: float) -> dict:
    """Automated choreography-preservation report: per-joint-group amplitude
    retention (cleaned/raw peak-to-peak), overall and worst 5 s tile, plus
    peak-velocity retention. Replaces eye-checking every move: any tile whose
    amplitude retention drops below RETENTION_WARN is listed in ``warnings``."""
    n = len(raw)
    tile = max(int(5 * fps), 1)
    res: dict = {"warnings": []}
    for g, jj in _GROUPS.items():
        jj = [j for j in jj if j < raw.shape[1]]
        if not jj:
            continue
        amps_r = np.ptp(raw[:, jj], axis=0)
        keep = amps_r > 0.05                      # ignore idle joints
        amp = (float(np.mean(np.ptp(cleaned[:, jj], axis=0)[keep]
                             / amps_r[keep])) if keep.any() else 1.0)
        v_r = np.abs(np.diff(raw[:, jj], axis=0)).max() * fps
        v_c = np.abs(np.diff(cleaned[:, jj], axis=0)).max() * fps
        worst_tile, worst_t0 = 1.0, 0.0
        for s in range(0, n - tile // 2, tile):
            e = min(n, s + tile)
            a_r = np.ptp(raw[s:e, jj], axis=0)
            k = a_r > 0.05
            if not k.any():
                continue
            r = float(np.mean(np.ptp(cleaned[s:e, jj], axis=0)[k] / a_r[k]))
            if r < worst_tile:
                worst_tile, worst_t0 = r, s / fps
        pv_ret = float(v_c / max(v_r, 1e-9))
        res[g] = {"amp_retention": round(amp, 3),
                  "peakvel_retention": round(pv_ret, 3),
                  "worst_5s_tile": {"t0_s": round(worst_t0, 1),
                                    "amp_retention": round(worst_tile, 3)}}
        if worst_tile < RETENTION_WARN:
            res["warnings"].append(
                f"{g}: amplitude retention {worst_tile:.2f} in the 5s tile at "
                f"{worst_t0:.1f}s (< {RETENTION_WARN}) — cleaning may be "
                f"eating choreography there; inspect before training")
        # peak-velocity retention catches SG-blunted snaps that amplitude misses
        # (audit finding A): a sharp hit keeps its range but loses its speed.
        if pv_ret < RETENTION_WARN:
            res["warnings"].append(
                f"{g}: peak-velocity retention {pv_ret:.2f} (< {RETENTION_WARN}) "
                f"— smoothing is blunting sharp hits; check the SG/protected-run "
                f"blend before training")
    return res


# Velocity-aware SG exemption band (rad/s). Below VEL_LO the signal is jitter →
# full SG; above VEL_HI it is genuine choreography (a hit/snap; the deploy limit
# is 9.4 rad/s) → keep the un-smoothed value; smoothstep between. Jitter on the
# repo dances sits well under 2 rad/s; real Thriller snaps reach 7-9 rad/s.
VEL_LO, VEL_HI = 2.5, 4.5
N_ROOT = 3    # cols 0:3 are root xyz (metres, different scale) — flag-only protect


def _reblend_protected(raw: np.ndarray, sg: np.ndarray, protected_runs,
                       fps: float, ramp: int = 2) -> np.ndarray:
    """Keep genuine sharp motion out of the SG low-pass (audit finding A,
    2026-07-21): SG(window 7) blunts fast hits ~29% even when they are NOT
    flagged as glitches, because a fast joint's own MAD hides the snap from the
    spike detector. Weight W (1=raw, 0=SG) per column is the max of:
      * flagged protected runs (cosine-tapered ±ramp frames), and
      * a velocity gate on the JOINT columns — smoothstep of the raw per-frame
        speed across [VEL_LO, VEL_HI] rad/s, so choreographic speed is preserved
        and low-speed jitter is still smoothed. Root xyz uses flag-only protect.
    Glitches are already removed by reject_outliers before SG, and the downstream
    velocity clamp still bounds anything over the actuator limit."""
    N = len(sg)
    W = np.zeros(sg.shape)
    # velocity gate (joints only)
    if N >= 2:
        v = np.zeros_like(raw)
        v[1:] = np.abs(raw[1:] - raw[:-1]) * fps
        v[0] = v[1]
        t = np.clip((v[:, N_ROOT:] - VEL_LO) / (VEL_HI - VEL_LO), 0.0, 1.0)
        W[:, N_ROOT:] = t * t * (3 - 2 * t)          # smoothstep
    # flagged protected runs (all columns, tapered)
    for (s, e, d) in protected_runs:
        s, e, d = int(s), int(e), int(d)
        W[s:e + 1, d] = 1.0
        for k in range(1, ramp + 1):
            w = 0.5 * (1 + np.cos(np.pi * k / (ramp + 1)))   # 1->0 cosine
            if s - k >= 0:
                W[s - k, d] = max(W[s - k, d], w)
            if e + k < N:
                W[e + k, d] = max(W[e + k, d], w)
    return W * raw + (1.0 - W) * sg


def clean_motion(motion: np.ndarray, fps: float = FPS) -> tuple[np.ndarray, dict]:
    """Outlier rejection + temporal smoothing on a (N,36) motion.
    Joints & root xyz: hampel then Savitzky-Golay. Root quat: tangent-space SG.
    Returns (cleaned, info) with before/after jerk and a fidelity delta."""
    before = analyze(motion, fps)
    out = motion.copy()
    cols = np.concatenate([out[:, 0:3], out[:, 7:]], axis=1)
    runs: dict = {}
    cols, n_outliers = reject_outliers(cols, fps, info=runs)
    if len(out) >= SG_WINDOW:
        cols_raw = cols.copy()
        cols = savgol_filter(cols, SG_WINDOW, SG_POLY, axis=0, mode="interp")
        # keep genuine sharp choreography (flagged OR just fast) out of the SG blur
        cols = _reblend_protected(cols_raw, cols, runs.get("protected_runs", []), fps)
    out[:, 0:3], out[:, 7:] = cols[:, :3], cols[:, 3:]
    out[:, 3:7] = smooth_quat(out[:, 3:7])
    after = analyze(out, fps)
    info = {
        "outlier_frames_replaced": n_outliers,
        "glitch_runs_rejected": len(runs.get("rejected_runs", [])),
        "choreo_runs_protected": len(runs.get("protected_runs", [])),
        "fidelity": _fidelity_retention(motion[:, 7:], out[:, 7:], fps),
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
