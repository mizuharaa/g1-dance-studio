#!/usr/bin/env python
"""Hardware A/B: candidate full-dance telemetry vs the s2r-b baseline runs.

Same math as tools/fluidity_forensics.py (helpers imported from it): per-group
track RMS deg (q - ref, full dance) and the 2-10 Hz pelvis-gyro wobble band.

Usage:
  ~/miniconda3/envs/tv/bin/python tools/hw_ab_compare.py \
      data/telemetry/<candidate>.npz [more_candidate.npz ...] \
      [--ref data/policies/thriller/thriller_deploy.npz]
Baseline runs are the four committed s2r-b full-dance telemetry files.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.fluidity_forensics import GROUPS, RUNS as BASELINE_RUNS, band_filter, rms

FS = 50.0


def load_runs(paths, T):
    runs = []
    for p in paths:
        d = dict(np.load(p, allow_pickle=False))
        assert d["q"].shape[0] == T, f"{Path(p).name}: {d['q'].shape[0]} ticks != ref {T}"
        runs.append((Path(p).name, d))
    return runs


def group_rms(runs, ref_jp):
    out = {}
    for g, idx in GROUPS.items():
        errs = [np.degrees(d["q"][:, idx] - ref_jp[:, idx]) for _, d in runs]
        out[g] = rms(np.concatenate(errs))
    return out


def wobble(runs):
    # roll/pitch gyro 2-10 Hz band RMS — the leg-fluidity hardware metric
    vals = []
    for _, d in runs:
        g = d["gyro"][:, :2].astype(float)
        vals.append(rms(band_filter(g, 2.0, 10.0)))
    return float(np.mean(vals)), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="+")
    ap.add_argument("--ref", default=str(ROOT / "data/policies/thriller/thriller_deploy.npz"))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    ref_jp = np.load(args.ref)["joint_pos"].astype(float)
    T = ref_jp.shape[0]
    cand = load_runs(args.candidates, T)
    base = load_runs([p for p in BASELINE_RUNS if Path(p).exists()], T)

    res = {"ref": args.ref, "n_base_runs": len(base), "n_cand_runs": len(cand)}
    cg, bg = group_rms(cand, ref_jp), group_rms(base, ref_jp)
    cw, cw_each = wobble(cand)
    bw, _ = wobble(base)
    res["track_rms_deg"] = {"candidate": cg, "s2rb_baseline": bg}
    res["gyro_wobble_2_10Hz_rad_s"] = {"candidate_mean": cw, "candidate_each": cw_each,
                                       "s2rb_baseline_mean": bw}

    print(f"ref={Path(args.ref).name}  candidate runs={len(cand)}  baseline runs={len(base)}")
    print(f"{'group':8s} {'candidate':>10s} {'s2r-b':>10s} {'delta':>8s}")
    for g in GROUPS:
        d = cg[g] - bg[g]
        print(f"{g:8s} {cg[g]:10.2f} {bg[g]:10.2f} {d:+8.2f} deg RMS")
    print(f"wobble   {cw:10.3f} {bw:10.3f} {cw-bw:+8.3f} rad/s (2-10Hz gyro, lower=smoother)")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
