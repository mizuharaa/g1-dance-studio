"""Graceful-degradation REPAIR TOOLBOX for infeasible G1 reference motion.

Implements the user's DECIDED degradation policy, in order (escalate only when the
prior step leaves flags):

  1. GLOBAL TEMPO SLOWDOWN  (primary / default) - one uniform time-scale on the
     whole motion. Torque ~ 1/T^2, so a modest global slowdown clears most flags
     while preserving choreography exactly. Sweep the factor, pick the mildest one
     that clears the majority of ankle flags.
  2. LOCAL TIME-WARP        - slow only the residual still-flagged windows with
     smooth (cosine-ramped) time warps.
  3. AMPLITUDE SCALING      - shrink CoM-sway / joint excursion at flagged beats.
  (track-to-limit clamp, strategy- and missing-DoF substitution are documented in
   PROMPT B; steps 1-3 are what actually move the Thriller under the envelope and
   are implemented here.)

After each step we RE-RUN pipeline.motion_dynamics.analyze and stop as soon as the
ankle demand is under the headroom envelope (or we hit the max tolerated slowdown).

Feasibility metric = the ankle-strategy torque from motion_dynamics (F_z*||ZMP-CoM||),
which is foot-position independent and scales ~1/T^2 under time-scaling. Style
preservation = time-normalized joint-trajectory similarity pre/post (pure global
slowdown preserves the path exactly => ~1.0).

NEVER overwrites the source: writes a new CSV + a scorecard, and the caller
appends a registry row with the new sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import g1_limits as L
from pipeline import motion_dynamics as MD
from pipeline.motion_io import load_motion_csv

FPS = 30.0


# --------------------------------------------------------------------------- #
# quaternion helpers (xyzw, CSV convention)
# --------------------------------------------------------------------------- #
def _slerp(q0, q1, t):
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    d = np.dot(q0, q1)
    if d < 0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / (np.linalg.norm(q) + 1e-12)
    th0 = np.arccos(np.clip(d, -1, 1))
    s = np.sin(th0)
    return (np.sin((1 - t) * th0) / s) * q0 + (np.sin(t * th0) / s) * q1


def _resample(m: np.ndarray, times_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    """Resample a 36-col motion sampled at times_src onto times_dst. Linear on
    root xyz + joints, slerp on the xyzw quat (cols 3:7)."""
    N2 = len(times_dst)
    out = np.empty((N2, 36))
    # linear cols
    lin = np.r_[0:3, 7:36]
    for c in lin:
        out[:, c] = np.interp(times_dst, times_src, m[:, c])
    # quat slerp
    idx = np.searchsorted(times_src, times_dst, side="right") - 1
    idx = np.clip(idx, 0, len(times_src) - 2)
    for k in range(N2):
        i = idx[k]
        span = times_src[i + 1] - times_src[i]
        a = 0.0 if span <= 0 else (times_dst[k] - times_src[i]) / span
        out[k, 3:7] = _slerp(m[i, 3:7], m[i + 1, 3:7], np.clip(a, 0, 1))
    return out


# --------------------------------------------------------------------------- #
# repair operators
# --------------------------------------------------------------------------- #
def global_slowdown(m: np.ndarray, factor: float, fps: float = FPS) -> np.ndarray:
    """Uniform time-scale. factor>1 = slower. Output has round(N*factor) frames at
    the SAME fps (i.e. the dance takes `factor`x longer). Choreography identical,
    every velocity /factor and acceleration /factor^2."""
    N = len(m)
    dur = (N - 1) / fps
    t_src = np.arange(N) / fps
    N2 = max(2, int(round(N * factor)))
    t_dst = np.linspace(0, dur, N2)          # same path, resampled denser
    return _resample(m, t_src, t_dst)


def local_time_warp(m: np.ndarray, windows_s, factor: float, fps: float = FPS,
                    ramp_s: float = 0.4) -> np.ndarray:
    """Slow only the given [start_s,end_s] windows by `factor`, with cosine speed
    ramps of width ramp_s so there are no velocity discontinuities at the seams.
    Monotone time-warp: build a per-source-frame slowdown gain, integrate to a
    warped clock, then resample uniformly on the warped clock."""
    N = len(m)
    t_src = np.arange(N) / fps
    gain = np.ones(N)                        # local time dilation per source frame
    for (s, e) in windows_s:
        for i in range(N):
            ti = t_src[i]
            if s - ramp_s <= ti <= e + ramp_s:
                if ti < s:
                    w = 0.5 * (1 - np.cos(np.pi * (ti - (s - ramp_s)) / ramp_s))
                elif ti > e:
                    w = 0.5 * (1 + np.cos(np.pi * (ti - e) / ramp_s))
                else:
                    w = 1.0
                gain[i] = max(gain[i], 1 + (factor - 1) * w)
    warped = np.concatenate([[0], np.cumsum((gain[1:] + gain[:-1]) / 2) / fps])
    dur = warped[-1]
    N2 = max(2, int(round(dur * fps)))
    t_dst = np.linspace(0, dur, N2)
    # map warped clock -> source time, then sample the source motion there
    src_time_at = np.interp(t_dst, warped, t_src)
    return _resample(m, t_src, src_time_at)


def adaptive_time_warp(m: np.ndarray, fps: float = FPS, margin: float = 0.85,
                       pad_s: float = 0.20, smooth_s: float = 0.30,
                       max_gain: float = 2.6, rounds: int = 6, verbose: bool = True):
    """DEMAND-SHAPED local retiming (v9): slow ONLY the moments whose ankle demand
    exceeds `margin`x the (speed-derated) limit; everything feasible keeps 1.0x speed.

    Per round:
      1. dynamic pass -> per-frame demand tau_i and per-frame limit lim_i
      2. required local gain g_i = clip( sqrt(tau_i / (margin*lim_i)), 1, max_gain )
         (torque ~ 1/T^2 under local time-scaling, validated by the global sweep)
      3. dilate g by +-pad_s (torque at a frame depends on accelerations around it),
         cosine-smooth over `smooth_s` so the tempo ramps musically, never steps
      4. integrate g -> monotone warped clock, resample uniformly at fps
    Iterate (warping shifts accelerations and lowers joint speeds, which also RAISES
    the derated limit) until 0 frames over headroom or `rounds` exhausted.

    Returns (motion, report) — report carries duration ratio, max local gain and the
    fraction of source time left untouched at 1.0x.
    """
    src_dur = (len(m) - 1) / fps
    total_gain_max, rounds_used = 1.0, 0
    kept_native = None
    # source-time of each CURRENT frame (composed across rounds) — lets callers map
    # any source-time landmark (e.g. the waist-slack beat windows) to the warped clock
    orig_t = np.arange(len(m)) / fps
    for r in range(rounds):
        fe = _feas(m, fps, arrays=True)
        arrs = fe.pop("_arrays")
        tau = arrs["ankle_tau_max"]
        need = tau / (margin * L.ANKLE_HEADROOM_NM)
        if (need <= 1.0).all():
            break
        rounds_used = r + 1
        g = np.clip(np.sqrt(np.maximum(need, 1.0)), 1.0, max_gain)
        if kept_native is None:
            kept_native = float((g <= 1.0 + 1e-9).mean())
        # dilation: sliding max over +-pad_s
        w = max(1, int(round(pad_s * fps)))
        gd = np.copy(g)
        for i in range(len(g)):
            gd[i] = g[max(0, i - w):i + w + 1].max()
        # cosine smoothing (hann), keep >=1
        k = max(3, int(round(smooth_s * fps)) | 1)
        hann = np.hanning(k); hann /= hann.sum()
        gs = np.convolve(gd, hann, mode="same")
        gs = np.maximum(gs, 1.0)
        total_gain_max = max(total_gain_max, float(gs.max()))
        # integrate to the warped clock, resample uniform
        t_src = np.arange(len(m)) / fps
        warped = np.concatenate([[0], np.cumsum((gs[1:] + gs[:-1]) / 2) / fps])
        t_dst = np.arange(0, warped[-1], 1.0 / fps)
        src_time_at = np.interp(t_dst, warped, t_src)
        orig_t = np.interp(src_time_at, t_src, orig_t)  # compose the source-time map
        m = _resample(m, t_src, src_time_at)
        if verbose:
            print(f"  round {r+1}: peak need x{np.sqrt(need.max()):.2f}, "
                  f"applied local gain <=x{gs.max():.2f}, "
                  f"duration {src_dur:.1f}s -> {warped[-1]:.1f}s")
    fe = _feas(m, fps)
    report = {
        "op": "adaptive_time_warp", "margin": margin, "pad_s": pad_s,
        "smooth_s": smooth_s, "max_gain": max_gain, "rounds_used": rounds_used,
        "duration_ratio": round(((len(m) - 1) / fps) / src_dur, 3),
        "max_local_gain": round(total_gain_max, 2),
        "native_speed_fraction": (round(kept_native, 3) if kept_native is not None else 1.0),
        # source-time of every warped frame (30 fps grid) — np.interp(src_landmark_s,
        # source_t, warped_t) maps any source-time window onto the warped clock
        "time_map": {"warped_t": [round(float(t), 3) for t in np.arange(len(m)) / fps],
                     "source_t": [round(float(t), 3) for t in orig_t]},
        "final": fe,
    }
    return m, report


def map_source_windows(report: dict, windows_s) -> list:
    """Map source-time [start,end] windows through an adaptive_time_warp report's
    time_map onto the warped clock (for e.g. the v8/v9 waist-slack beat windows)."""
    tm = report["time_map"]
    src, wrp = np.asarray(tm["source_t"]), np.asarray(tm["warped_t"])
    return [[round(float(np.interp(s, src, wrp)), 2),
             round(float(np.interp(e, src, wrp)), 2)] for (s, e) in windows_s]


def amplitude_scale_root(m: np.ndarray, windows_s, scale: float, fps: float = FPS,
                         ramp_s: float = 0.4) -> np.ndarray:
    """Shrink root-XY sway toward its local mean inside flagged windows (reduces
    the CoM excursion that drives ankle demand) while preserving joints + timing."""
    out = m.copy()
    N = len(m)
    t = np.arange(N) / fps
    center = m[:, :2].mean(axis=0)
    for (s, e) in windows_s:
        w = np.zeros(N)
        for i in range(N):
            ti = t[i]
            if s - ramp_s <= ti <= e + ramp_s:
                if ti < s:
                    w[i] = 0.5 * (1 - np.cos(np.pi * (ti - (s - ramp_s)) / ramp_s))
                elif ti > e:
                    w[i] = 0.5 * (1 + np.cos(np.pi * (ti - e) / ramp_s))
                else:
                    w[i] = 1.0
        f = 1 - w[:, None] * (1 - scale)
        out[:, :2] = center + (out[:, :2] - center) * f
    return out


# --------------------------------------------------------------------------- #
# style-preservation metric
# --------------------------------------------------------------------------- #
def style_similarity(m0: np.ndarray, m1: np.ndarray, samples: int = 300) -> float:
    """Time-normalized similarity of the 29 joint-angle trajectories (both motions
    resampled to `samples` points on a common [0,1] clock). 1.0 = identical shape.
    Pure global slowdown preserves the path exactly => ~1.0. Amplitude/warp lower
    it. Reported as the mean over joints of 1 - RMSE/(range+eps)."""
    def norm(m):
        t = np.linspace(0, 1, len(m))
        td = np.linspace(0, 1, samples)
        return np.stack([np.interp(td, t, m[:, 7 + j]) for j in range(29)], axis=1)
    a, b = norm(m0), norm(m1)
    rng = np.maximum(a.max(0) - a.min(0), 1e-3)
    rmse = np.sqrt(((a - b) ** 2).mean(0))
    return float(np.clip(1 - np.mean(rmse / rng), 0, 1))


# --------------------------------------------------------------------------- #
# feasibility summary from the dynamic pass
# --------------------------------------------------------------------------- #
def _feas(m_or_path, fps=FPS, tmp=None, arrays=False):
    """Run the dynamic pass; accept a path or an array (written to a temp csv)."""
    if isinstance(m_or_path, np.ndarray):
        tmp = tmp or (ROOT / "experiments/motion_feasibility/_tmp_repair.csv")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(tmp, m_or_path, delimiter=",")
        path = tmp
    else:
        path = m_or_path
    r = MD.analyze(path, fps=fps)
    d = r["dynamic"]
    out = {
        "frames": r["frames"], "seconds": r["seconds"],
        "ankle_tau_max_nm": d["ankle_tau_max_nm"],
        "ankle_tau_p95_nm": d["ankle_tau_p95_nm"],
        "ankle_over_headroom_pct": d["ankle_frames_over_headroom_pct"],
        "windows": r["ankle_flag_windows_s"],
    }
    if arrays:
        out["_arrays"] = r["_arrays"]  # per-frame demand/limit for adaptive_time_warp
    return out


def sweep(m: np.ndarray, factors, fps=FPS):
    rows = []
    for f in factors:
        mm = m if f == 1.0 else global_slowdown(m, f, fps)
        fe = _feas(mm, fps)
        fe["factor"] = f
        rows.append(fe)
        print(f"  factor {f:.2f}: {fe['seconds']:.1f}s  ankle p95 {fe['ankle_tau_p95_nm']:6.1f}  "
              f"max {fe['ankle_tau_max_nm']:6.1f}  over-headroom {fe['ankle_over_headroom_pct']:5.1f}%")
    return rows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--sweep", type=str, default="1.0,1.3,1.5,1.7,2.0,2.3",
                    help="comma factors for the global-slowdown sweep")
    ap.add_argument("--target-pct", type=float, default=15.0,
                    help="stop escalating once ankle-over-headroom%% <= this")
    ap.add_argument("--max-factor", type=float, default=2.3,
                    help="mildest-first cap on global slowdown the show tolerates")
    ap.add_argument("--out", type=Path, default=None, help="write repaired CSV")
    ap.add_argument("--scorecard", type=Path, default=None)
    ap.add_argument("--apply-factor", type=float, default=None,
                    help="skip the sweep; just apply this global factor")
    ap.add_argument("--local-warp", type=float, default=1.4,
                    help="extra local slowdown applied to residual flagged windows")
    ap.add_argument("--adaptive", action="store_true",
                    help="v9 mode: NO global slowdown — demand-shaped local retiming only "
                         "(adaptive_time_warp); feasible parts keep native 1.0x speed")
    ap.add_argument("--margin", type=float, default=0.85,
                    help="adaptive mode: target = margin x ankle headroom")
    args = ap.parse_args()

    m0 = load_motion_csv(args.csv)
    base = _feas(args.csv, args.fps)
    print(f"SOURCE {args.csv}: {base['seconds']:.1f}s  ankle p95 {base['ankle_tau_p95_nm']} "
          f"max {base['ankle_tau_max_nm']}  over-headroom {base['ankle_over_headroom_pct']}%")

    if args.adaptive:
        print("\n[ADAPTIVE] demand-shaped local retiming (no global slowdown):")
        m, rep = adaptive_time_warp(m0.copy(), fps=args.fps, margin=args.margin)
        f = rep["final"]
        print(f"RESULT: {f['seconds']:.1f}s ({rep['duration_ratio']}x), native-speed "
              f"{rep['native_speed_fraction']*100:.0f}%, max local gain x{rep['max_local_gain']}, "
              f"ankle max {f['ankle_tau_max_nm']} p95 {f['ankle_tau_p95_nm']} "
              f"over {f['ankle_over_headroom_pct']}%")
        if args.out:
            np.savetxt(args.out, m, delimiter=",")
            rep.update(source=str(args.csv), source_sha256=sha256(args.csv),
                       repaired=str(args.out), repaired_sha256=sha256(args.out))
            print(f"wrote {args.out}")
        if args.scorecard:
            args.scorecard.parent.mkdir(parents=True, exist_ok=True)
            args.scorecard.write_text(json.dumps(rep, indent=1))
            print(f"wrote {args.scorecard}")
        return 0 if f["ankle_over_headroom_pct"] == 0 else 1

    ops = []
    if args.apply_factor:
        factor = args.apply_factor
        print(f"\nApplying global slowdown factor {factor}")
    else:
        print("\n[1] GLOBAL SLOWDOWN sweep (mildest that clears the majority):")
        factors = [float(x) for x in args.sweep.split(",")]
        rows = sweep(m0, factors, args.fps)
        # mildest factor whose over-headroom% <= target, capped at max-factor
        ok = [r for r in rows if r["ankle_over_headroom_pct"] <= args.target_pct
              and r["factor"] <= args.max_factor]
        if ok:
            factor = min(r["factor"] for r in ok)
        else:
            # none clears target -> take the largest tolerated factor (best effort)
            factor = min(args.max_factor, max(r["factor"] for r in rows))
        print(f"  -> chosen global factor: {factor:.2f}")

    m = global_slowdown(m0, factor, args.fps) if factor != 1.0 else m0.copy()
    ops.append({"op": "global_slowdown", "factor": factor})
    after_global = _feas(m, args.fps)
    print(f"  after global: ankle p95 {after_global['ankle_tau_p95_nm']} "
          f"max {after_global['ankle_tau_max_nm']}  over-headroom {after_global['ankle_over_headroom_pct']}%")

    # [2] LOCAL TIME-WARP for residual flagged windows
    resid = after_global["windows"]
    if resid and after_global["ankle_over_headroom_pct"] > args.target_pct:
        print(f"\n[2] LOCAL TIME-WARP x{args.local_warp} on {len(resid)} residual window(s)")
        m = local_time_warp(m, resid, args.local_warp, args.fps)
        ops.append({"op": "local_time_warp", "factor": args.local_warp,
                    "windows_s": resid})
        after_local = _feas(m, args.fps)
        print(f"  after local:  ankle p95 {after_local['ankle_tau_p95_nm']} "
              f"max {after_local['ankle_tau_max_nm']}  over-headroom {after_local['ankle_over_headroom_pct']}%")
        final = after_local
    else:
        final = after_global

    style = style_similarity(m0, m)
    print(f"\nSTYLE similarity (joint-trajectory, time-normalized): {style:.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.out, m, delimiter=",")
        print("wrote repaired motion:", args.out, "sha256", sha256(args.out)[:12])

    scorecard = {
        "source": str(args.csv),
        "source_sha256": sha256(args.csv),
        "repaired": str(args.out) if args.out else None,
        "repaired_sha256": sha256(args.out) if args.out else None,
        "repairs_applied": ops,
        "before": base,
        "after": final,
        "style_similarity": round(style, 3),
        "ankle_headroom_nm": L.ANKLE_HEADROOM_NM,
        "torque_speed_model": L.summary()["torque_speed_model"],
    }
    if args.scorecard:
        args.scorecard.parent.mkdir(parents=True, exist_ok=True)
        args.scorecard.write_text(json.dumps(scorecard, indent=2))
        print("wrote scorecard:", args.scorecard)
    return scorecard


if __name__ == "__main__":
    main()
