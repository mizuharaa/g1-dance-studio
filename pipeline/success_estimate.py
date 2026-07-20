"""Pre-training success estimate: rough predicted survival band for a prepared motion.

Runs RIGHT AFTER prep/vet — BEFORE any GPU money is spent — and answers: "if we
train on this motion with the current recipe, roughly what nominal gate survival
should we expect?" It cross-checks the motion with the dynamic feasibility
checker (pipeline.motion_dynamics.analyze) and maps the feasibility metrics to a
predicted nominal-survival band via a calibration table built from historical
(feasibility metrics, observed gate survival) pairs in exports/.

HONESTY CONTRACT (read before trusting a number):
  * The bands are WIDE on purpose. The same motion (the native-speed Thriller
    deploy CSV) spanned 85.9-100% nominal survival across three training
    attempts — at fixed feasibility, survival is dominated by the training
    recipe, not the motion. The band is an envelope over that history.
  * The dynamics checker historically OVER-estimated ankle torque demand by
    6-10x vs real hardware telemetry (decision log 2026-07-20: double-support
    mis-attribution + extraction jitter). Bad-looking metrics are therefore
    conservative: the real robot cleared native-speed windows the model flagged.
  * Calibration n is tiny (5 points, one choreography, sim-gate survival, not
    live-show survival). Treat the output as a cost-gate sanity check, not a
    guarantee.

The calibration table lives in data/calibration/success_calibration.json and is
regenerated with `python -m pipeline.success_estimate --recalibrate` (re-run it
after any motion_dynamics upgrade so the stored metrics match the live checker).

CLI:
    python -m pipeline.success_estimate <motion.csv> [--vet vet.json] [--json out.json]
    python -m pipeline.success_estimate --recalibrate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CALIBRATION_PATH = PROJECT_ROOT / "data" / "calibration" / "success_calibration.json"

# The primary predictor: fraction of frames where the checker says the ankle
# demand exceeds its headroom. It is the metric the historical attempts actually
# varied on (native > 1.8x-slowdown > adaptive; ABSOLUTE values depend on the
# checker version — always compare against a calibration table regenerated with
# the same checker, which is what --recalibrate guarantees).
PRIMARY_METRIC = "dynamic.ankle_frames_over_headroom_pct"

# Beyond the calibrated range we have NO data — decay the band so a motion far
# worse than anything ever trained reads as increasingly risky (pts per %-point
# of ankle-over-headroom beyond the calibrated max).
EXTRAP_LO_SLOPE = 0.6
EXTRAP_HI_SLOPE = 0.3
BAND_FLOOR = 5.0     # never predict below this (we simply don't know)
BAND_CEIL = 99.0     # never promise 100% before training
MIN_BAND_WIDTH = 10.0  # honesty: a tighter band than this is not supported by n=5

# Historical anchors mined from exports/ gap files: (attempt, gap.json relative
# path, motion CSV the attempt trained on, provenance note). v6/v7 and the
# calibration anchor all trained the SAME native-speed deploy CSV — their
# survival spread (85.9-100%) is the recipe-variance evidence behind the wide bands.
ANCHORS = [
    ("train-thriller_v6sk-0714",
     "exports/train-thriller_v6sk-0714/gap.json",
     "data/motions/thriller/thriller_deploy.csv",
     "v6 skills recipe, native speed; nominal survival from 128-episode gate"),
    ("train-thriller_v7ank-0715",
     "exports/train-thriller_v7ank-0715/gap.json",
     "data/motions/thriller/thriller_deploy.csv",
     "v7 ankle-penalty recipe, native speed; best-checkpoint-selected"),
    ("thriller_csv_ankle_penalty-anchor",
     "exports/train-thriller_v8s2r-0716/calibration_anchor_gap.json",
     "data/motions/thriller/thriller_deploy.csv",
     "2026-07-08 anchor policy (live-deployed, ~70% IRL mimicry) re-gated "
     "2026-07-17 via --onnx mode; trained on the native deploy CSV"),
    ("train-thriller_v8s2r-0716",
     "exports/train-thriller_v8s2r-0716/gap.json",
     "data/motions/thriller/thriller_g1_grounded_repaired_1p8x.csv",
     "v8 stage-2-repair recipe on the 1.8x-slowed repaired motion"),
    ("train-thriller_v9adpt-0717",
     "exports/train-thriller_v9adpt-0717/gap.json",
     "data/motions/thriller/thriller_g1_grounded_adaptive.csv",
     "v9 adaptive-slowdown motion (1.53x avg); winner iter 9500"),
]


def _dget(d: dict, dotted: str, default=None):
    """res['dynamic']['ankle_tau_p95_nm'] via 'dynamic.ankle_tau_p95_nm', .get all
    the way down — motion_dynamics is being upgraded concurrently and new/renamed
    keys must degrade gracefully, never crash the estimator."""
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return default if cur is None else cur


def load_calibration(path: str | Path | None = None) -> dict:
    p = Path(path) if path else CALIBRATION_PATH
    with open(p) as f:
        cal = json.load(f)
    if not isinstance(cal.get("rows"), list) or not cal["rows"]:
        raise ValueError(f"calibration table {p} has no rows")
    return cal


def predict_band(x: float, calibration: dict) -> tuple[float, float]:
    """Map the primary metric value x -> (lo, hi) predicted nominal-survival band.

    Pure + deterministic (unit-testable without MuJoCo). Method: group calibration
    rows by their primary-metric value; at each value take the min/max observed
    survival; then take MONOTONE NON-INCREASING envelopes (upper envelope = max
    survival observed at any metric >= x; lower = min observed at any metric <= x)
    so a worse metric can never predict a better band; linearly interpolate
    between grid points, decay beyond the calibrated range, clamp, and enforce a
    minimum width. This is an envelope over observed history, not a regression —
    with n=5 anything tighter would be fiction.
    """
    metric_key = calibration.get("primary_metric", PRIMARY_METRIC)
    pts: dict[float, list[float]] = {}
    for row in calibration["rows"]:
        mx = _dget(row.get("metrics", {}), metric_key.split(".", 1)[-1]
                   if "." in metric_key else metric_key)
        if mx is None:  # metrics dict stores flat keys; try the full dotted name
            mx = row.get("metrics", {}).get(metric_key)
        surv = row.get("nominal_survival_pct")
        if mx is None or surv is None:
            continue
        pts.setdefault(round(float(mx), 2), []).append(float(surv))
    if not pts:
        raise ValueError("calibration rows carry no usable "
                         f"({metric_key}, nominal_survival_pct) pairs")
    xs = sorted(pts)
    lo_raw = [min(pts[v]) for v in xs]
    hi_raw = [max(pts[v]) for v in xs]
    # monotone non-increasing envelopes
    lo_env = [min(lo_raw[: i + 1]) for i in range(len(xs))]          # cummin left
    hi_env = [max(hi_raw[i:]) for i in range(len(xs))]               # cummax right
    x = float(x)
    if x <= xs[0]:
        lo, hi = lo_env[0], hi_env[0]
    elif x >= xs[-1]:
        over = x - xs[-1]
        lo = lo_env[-1] - EXTRAP_LO_SLOPE * over
        hi = hi_env[-1] - EXTRAP_HI_SLOPE * over
    else:
        import bisect
        j = bisect.bisect_right(xs, x)
        f = (x - xs[j - 1]) / (xs[j] - xs[j - 1])
        lo = lo_env[j - 1] + f * (lo_env[j] - lo_env[j - 1])
        hi = hi_env[j - 1] + f * (hi_env[j] - hi_env[j - 1])
    lo = max(BAND_FLOOR, min(lo, BAND_CEIL))
    hi = max(BAND_FLOOR, min(hi, BAND_CEIL))
    if hi < lo:
        hi = lo
    if hi - lo < MIN_BAND_WIDTH:  # widen symmetrically inside [floor, ceil]
        pad = (MIN_BAND_WIDTH - (hi - lo)) / 2
        lo, hi = max(BAND_FLOOR, lo - pad), min(BAND_CEIL, hi + pad)
        if hi - lo < MIN_BAND_WIDTH:  # ran into a clamp — push the other side
            lo = max(BAND_FLOOR, hi - MIN_BAND_WIDTH)
            hi = min(BAND_CEIL, lo + MIN_BAND_WIDTH)
    return lo, hi


def merge_risk_windows(windows, top: int = 3, merge_gap_s: float = 1.5) -> list[dict]:
    """ankle_flag_windows_s ([[start,end],...]) -> the `top` biggest merged windows,
    chronological, labelled for the operator."""
    spans: list[list[float]] = []
    for w in windows or []:
        try:
            s, e = float(w[0]), float(w[1])
        except (TypeError, ValueError, IndexError):
            continue
        if spans and s - spans[-1][1] <= merge_gap_s:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, max(e, s)])
    spans = [p for p in spans if p[1] - p[0] >= 0.2]  # drop single-frame blips
    spans.sort(key=lambda p: p[1] - p[0], reverse=True)
    out = [{"start_s": round(s, 1), "end_s": round(e, 1),
            "duration_s": round(e - s, 1),
            "label": "fast weight shift / high ankle demand"}
           for s, e in spans[:top]]
    out.sort(key=lambda w: w["start_s"])
    return out


def estimate(motion_csv_path: str | Path, vet_report: dict | None = None,
             fps: float = 30.0, calibration_path: str | Path | None = None) -> dict:
    """Full pre-training crosscheck for one prepared motion CSV. Returns the
    estimate dict that gets attached to the stage record / served by the API."""
    from pipeline.motion_dynamics import analyze

    res = analyze(motion_csv_path, fps=fps, ground=True)
    res.pop("_arrays", None)

    metrics = {
        "ankle_frames_over_headroom_pct":
            _dget(res, "dynamic.ankle_frames_over_headroom_pct"),
        "ankle_tau_p95_nm": _dget(res, "dynamic.ankle_tau_p95_nm"),
        "zmp_outside_support_pct": _dget(res, "balance.zmp_outside_support_pct"),
        "vel_frames_over_limit_pct":
            _dget(res, "kinematic.vel_frames_over_limit_pct"),
    }

    cal = load_calibration(calibration_path)
    notes = [
        "Rough pre-training estimate from motion physics; not a guarantee.",
        "The dynamics checker historically OVER-estimated ankle torque 6-10x vs "
        "hardware telemetry (decision log 2026-07-20) — flagged demand is "
        "conservative; the real robot cleared native-speed windows the model "
        "flagged.",
        "At fixed feasibility, gate survival varied 85.9-100% purely with the "
        "training recipe — the band is an envelope over that history.",
    ]

    x = metrics["ankle_frames_over_headroom_pct"]
    if x is None:
        lo = hi = None
        notes.append("primary metric missing from the checker output — no band "
                     "predicted (checker interface changed?)")
    else:
        lo, hi = predict_band(float(x), cal)
        # secondary penalty: velocity-infeasible frames were 0.0% on every
        # calibrated motion — any real amount is outside calibration, mark risk.
        vel_pct = metrics["vel_frames_over_limit_pct"] or 0.0
        if vel_pct > 0.5:
            drop_lo = min(15.0, 2.0 * vel_pct)
            drop_hi = min(8.0, vel_pct)
            lo = max(BAND_FLOOR, lo - drop_lo)
            hi = max(lo, hi - drop_hi)
            notes.append(f"{vel_pct:.1f}% of frames exceed joint velocity limits "
                         "— outside all calibration history; band lowered")

    blockers = []
    if isinstance(vet_report, dict):
        for name, check in (vet_report.get("hard") or {}).items():
            if isinstance(check, dict) and check.get("pass") is False:
                blockers.append(name)
        if blockers:
            notes.append("hard vet failures present — training is blocked until "
                         "they are fixed; the band assumes they get fixed")

    out = {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "motion_csv": str(motion_csv_path),
        "predicted_survival_pct_range": (
            f"{lo:.0f}-{hi:.0f}%" if lo is not None else "unavailable"),
        "predicted_survival_lo_pct": round(lo, 1) if lo is not None else None,
        "predicted_survival_hi_pct": round(hi, 1) if hi is not None else None,
        "confidence": (f"rough — n={len(cal['rows'])} historical points, "
                       "sim-gate based"),
        "primary_metric": {"name": cal.get("primary_metric", PRIMARY_METRIC),
                           "value": x},
        "metrics": metrics,
        "risk_windows": merge_risk_windows(res.get("ankle_flag_windows_s")),
        "hard_blockers": blockers,
        "notes": notes,
        "calibration": {"path": str(calibration_path or CALIBRATION_PATH),
                        "generated_at": cal.get("generated_at"),
                        "n_rows": len(cal["rows"])},
    }
    return out


def recalibrate(out_path: str | Path | None = None) -> dict:
    """Rebuild data/calibration/success_calibration.json from the exports/ gap
    files by running the CURRENT feasibility checker on each anchor's training
    motion. Re-run after every motion_dynamics upgrade."""
    from pipeline.motion_dynamics import analyze

    out_path = Path(out_path) if out_path else CALIBRATION_PATH
    metric_cache: dict[str, dict] = {}
    rows = []
    for attempt, gap_rel, motion_rel, note in ANCHORS:
        gap_path = PROJECT_ROOT / gap_rel
        motion_path = PROJECT_ROOT / motion_rel
        if not gap_path.exists() or not motion_path.exists():
            print(f"recalibrate: SKIP {attempt} (missing "
                  f"{gap_path.name if not gap_path.exists() else motion_path.name})",
                  file=sys.stderr)
            continue
        gap = json.loads(gap_path.read_text())
        surv = _dget(gap, "conditions.nominal.success_rate")
        if surv is None:
            print(f"recalibrate: SKIP {attempt} (no nominal success_rate)",
                  file=sys.stderr)
            continue
        if motion_rel not in metric_cache:
            print(f"recalibrate: analyzing {motion_rel} ...")
            res = analyze(motion_path, fps=30.0, ground=True)
            res.pop("_arrays", None)
            metric_cache[motion_rel] = {
                "ankle_frames_over_headroom_pct":
                    _dget(res, "dynamic.ankle_frames_over_headroom_pct"),
                "ankle_tau_p95_nm": _dget(res, "dynamic.ankle_tau_p95_nm"),
                "ankle_tau_max_nm": _dget(res, "dynamic.ankle_tau_max_nm"),
                "zmp_outside_support_pct":
                    _dget(res, "balance.zmp_outside_support_pct"),
                "vel_frames_over_limit_pct":
                    _dget(res, "kinematic.vel_frames_over_limit_pct"),
            }
        rows.append({
            "attempt": attempt,
            "gap_file": gap_rel,
            "motion": motion_rel,
            "metrics": metric_cache[motion_rel],
            "nominal_survival_pct": round(100.0 * float(surv), 2),
            "provenance": note,
        })
    cal = {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "primary_metric": "ankle_frames_over_headroom_pct",
        "checker": {
            "module": "pipeline.motion_dynamics",
            "note": "metrics computed with the checker as of generated_at; "
                    "regenerate with `python -m pipeline.success_estimate "
                    "--recalibrate` after any checker upgrade",
        },
        "rows": rows,
        "notes": [
            "Survival = nominal-condition success_rate from the 128-episode "
            "sim gate (exports/*/gap.json), NOT live-show survival.",
            "v6/v7/anchor trained the SAME native-speed motion and spanned "
            "85.9-100% survival: at fixed feasibility the recipe dominates — "
            "hence envelope bands, never point predictions.",
            "The checker over-estimated ankle torque 6-10x vs hardware "
            "(decision log 2026-07-20); metric values here are inflated "
            "consistently for ALL rows, so the RELATIVE mapping still holds.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cal, indent=2) + "\n")
    print(f"wrote {out_path} ({len(rows)} rows)")
    return cal


def _print(est: dict) -> None:
    print(f"{est['motion_csv']}")
    print(f"  PREDICTED nominal survival: {est['predicted_survival_pct_range']} "
          f"({est['confidence']})")
    pm = est["primary_metric"]
    print(f"  primary metric {pm['name']} = {pm['value']}")
    for k, v in est["metrics"].items():
        print(f"    {k}: {v}")
    if est["risk_windows"]:
        print("  top risk windows:")
        for w in est["risk_windows"]:
            print(f"    {w['start_s']}-{w['end_s']}s ({w['duration_s']}s) "
                  f"{w['label']}")
    if est["hard_blockers"]:
        print(f"  HARD BLOCKERS (vet): {', '.join(est['hard_blockers'])}")
    for n in est["notes"]:
        print(f"  note: {n}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="prepared motion CSV to estimate")
    ap.add_argument("--vet", type=Path, default=None,
                    help="vet_motion --json report (hard failures -> blockers)")
    ap.add_argument("--json", type=Path, default=None, help="write estimate here")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--recalibrate", action="store_true",
                    help="regenerate the calibration table from exports/ and exit")
    args = ap.parse_args()
    if args.recalibrate:
        recalibrate(args.calibration)
        return
    if not args.csv:
        ap.error("need a motion CSV (or --recalibrate)")
    vet = json.loads(args.vet.read_text()) if args.vet else None
    est = estimate(args.csv, vet_report=vet, fps=args.fps,
                   calibration_path=args.calibration)
    _print(est)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(est, indent=2))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
