#!/usr/bin/env python
"""Build a sharpness-preserving deployable reference from a raw GMR retarget.

Rationale (docs/retarget_fidelity.md): the standard chain clamps EVERY joint to
a blanket 0.9*3*pi ~ 8.48 rad/s (prep_motion._clamp_joint_velocities), which is
2-4x BELOW the G1's true per-motor velocity limits (hips/waist 20-32, shoulders/
elbows/ankles 37, wrists 22 rad/s). On Thriller that clamp blunted 58 real dance
accents by 40-85% and deleted ~97% of the high-frequency velocity energy in the
sharpest section. This tool rebuilds the deployable CSV with:

  1. DESPIKE: single-frame pose spikes (out-and-back within 2 frames, both
     steps > SPIKE_VEL) are estimator glitches -> midpoint-interpolated.
  2. PER-JOINT velocity clamp at MARGIN * the true motor velocity limit
     (sequential delta cap, same algorithm as prep_motion but per-joint and
     with no moving-average smoothing pass).
  3. Show assembly IDENTICAL in timing to pipeline/prep_motion.py (same pad/
     blend/hold frame counts, same standing pose, same FK grounding) — the
     frame count and beat alignment of the show cut are bit-preserved.
  4. Activation ramp: rows are copied verbatim from an existing deploy CSV
     (they end on the same standing pad row), so deploy timing is unchanged.

The output has the SAME number of frames as the reference deploy CSV — music
sync is untouched. Only joint sharpness differs.

Usage:
  python tools/make_sharp_reference.py \
      --raw data/motions/thriller/thriller_g1.csv \
      --deploy data/policies/thriller/thriller_deploy.csv \
      --out data/motions/edits/thriller_deploy_v2_sharp.csv \
      --report data/reports/thriller_v2_sharp.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.prep_motion import (  # noqa: E402
    BLEND_IN_S, BLEND_OUT_S, FPS, HOLD_OUT_S, MODEL_XML, PAD_IN_S,
    _blend, _min_height_fk, _standing_row,
)
from tools.retarget_fidelity_analysis import (  # noqa: E402
    JOINT_NAMES, true_limit,
)

DT = 1.0 / FPS
MARGIN = 0.9          # fraction of the true motor velocity limit to allow
SPIKE_VEL = 8.0       # rad/s — both legs of an out-and-back above this = spike
SPIKE_NET_FRAC = 0.5  # net displacement below this fraction of the step = spike


def despike(dof: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove single-frame out-and-back pose spikes (estimator glitches)."""
    out = dof.copy()
    n_fixed = 0
    step = SPIKE_VEL * DT
    for j in range(out.shape[1]):
        x = out[:, j]
        for i in range(1, len(x) - 1):
            d1, d2 = x[i] - x[i - 1], x[i + 1] - x[i]
            if (abs(d1) > step and abs(d2) > step and d1 * d2 < 0
                    and abs(x[i + 1] - x[i - 1]) < SPIKE_NET_FRAC * max(abs(d1), abs(d2))):
                x[i] = 0.5 * (x[i - 1] + x[i + 1])
                n_fixed += 1
    return out, n_fixed


def clamp_per_joint(dof: np.ndarray, limits: np.ndarray) -> tuple[np.ndarray, int]:
    """Sequential per-frame delta cap at per-joint limits (rad/s)."""
    cap = limits * DT
    out = dof.copy()
    touched = 0
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        over = np.abs(delta) > cap
        if over.any():
            out[i, over] = out[i - 1, over] + np.clip(delta[over], -cap[over], cap[over])
            touched += 1
    return out, touched


def assemble_show(motion: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """pipeline/prep_motion.py assembly (grounding + pad/blend/hold), no clamp."""
    motion = motion.copy()
    zmin = _min_height_fk(motion, model)
    motion[:, 2] -= zmin
    stand_in = _standing_row(model, motion[0])
    stand_out = _standing_row(model, motion[-1])
    return np.vstack([
        np.tile(stand_in, (round(PAD_IN_S * FPS), 1)),
        _blend(stand_in, motion[0], round(BLEND_IN_S * FPS)),
        motion,
        _blend(motion[-1], stand_out, round(BLEND_OUT_S * FPS)),
        np.tile(stand_out, (round(HOLD_OUT_S * FPS), 1)),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path,
                    help="raw (pre-clamp) GMR retarget CSV, 36 cols @30fps")
    ap.add_argument("--deploy", required=True, type=Path,
                    help="existing deploy CSV (supplies the activation ramp + timing reference)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--margin", type=float, default=MARGIN)
    args = ap.parse_args()

    raw = np.loadtxt(args.raw, delimiter=",")
    deploy_old = np.loadtxt(args.deploy, delimiter=",")
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))

    n_dance = len(raw)
    n_pad = round(PAD_IN_S * FPS) + round(BLEND_IN_S * FPS)
    n_show = n_pad + n_dance + round(BLEND_OUT_S * FPS) + round(HOLD_OUT_S * FPS)
    n_ramp = len(deploy_old) - n_show
    assert n_ramp >= 0, "deploy CSV shorter than the assembled show — wrong inputs"

    limits = args.margin * np.array([true_limit(n) for n in JOINT_NAMES])

    dof, n_spikes = despike(raw[:, 7:])
    dof, n_clamped = clamp_per_joint(dof, limits)
    motion = raw.copy()
    motion[:, 7:] = dof

    show = assemble_show(motion, model)
    v2 = np.vstack([deploy_old[:n_ramp], show])
    assert v2.shape == deploy_old.shape, "frame count changed — music sync broken"

    # ramp must land exactly on the new show's first row (same standing pad row)
    seam = np.abs(v2[n_ramp] - deploy_old[n_ramp]).max()
    assert seam < 1e-6, f"ramp/show seam mismatch ({seam:.2e}) — standing row drifted"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out, v2, delimiter=",")

    # ---- metrics + validation
    def vstats(m: np.ndarray) -> dict:
        v = np.abs(np.diff(m[:, 7:], axis=0)) * FPS
        return {"vel_max": round(float(v.max()), 2),
                "vel_p99": round(float(np.percentile(v, 99)), 2)}

    zmin_out = _min_height_fk(v2, model)
    vet = subprocess.run(
        [sys.executable, str(ROOT / "pipeline/vet_motion.py"), str(args.out), "--json"],
        capture_output=True, text=True)
    vet_json = json.loads(vet.stdout) if vet.stdout.strip() else {"error": vet.stderr[-500:]}

    report = {
        "inputs": {"raw": str(args.raw), "deploy_ref": str(args.deploy)},
        "params": {"margin": args.margin, "per_joint_limits_rad_s":
                   {n: round(float(l), 2) for n, l in zip(JOINT_NAMES, limits)}},
        "despiked_cells": n_spikes,
        "clamped_frames": n_clamped,
        "frames": {"out": int(len(v2)), "ref": int(len(deploy_old)),
                   "ramp": int(n_ramp), "show": int(n_show)},
        "old_deploy": vstats(deploy_old),
        "v2_sharp": vstats(v2),
        "fk_min_contact_height_m": round(float(zmin_out), 4),
        "vet": {"pass": vet.returncode == 0, "detail": vet_json},
        "out": str(args.out),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("despiked_cells", "clamped_frames", "old_deploy",
                       "v2_sharp", "fk_min_contact_height_m")}, indent=2))
    print("vet:", "PASS" if vet.returncode == 0 else "FAIL")
    print(f"out -> {args.out}")


if __name__ == "__main__":
    main()
