"""Cross-check: pipeline torque-DEMAND model vs REAL robot measured torque.

Question (user, 2026-07-20): the robot performed the native-speed Thriller live
(the ~70% IRL anchor policy, thriller_csv_ankle_penalty) without falling — yet
pipeline/motion_dynamics.py says the native-speed reference demands ~173 Nm peak
ankle torque (4x over the ~40 Nm envelope) and forced v8/v9 to slow the dance.
Is the demand model (or the GVHMR/landmark extraction feeding it) WRONG?

Method — three independent legs, aligned on the same dance clock:
  A. PREDICTED demand: motion_dynamics.analyze on the exact deployed motion
     (data/motions/thriller/thriller_deploy.csv, native speed, 30 fps).
  B. MEASURED reality: tau_est from the real 50 Hz deploy telemetry of the live
     native-speed runs (2026-07-08 gantry/show + 2026-07-10), ankle pitch L/R.
  C. FIDELITY: |q - target| and amplitude ratio per joint group per window —
     did the robot actually EXECUTE the reference at the flagged beats, or did
     it yield/wash them out (in which case low measured torque does NOT refute
     the demand model)?

Alignment: telemetry runs are stage='1' (dance) from tick 0; motion is the same
lineage (1554 frames * 50/30 = 2590 ticks ~= 2589/2600 recorded). We refine with
a cross-correlation of measured vs reference left-ankle-pitch position and
report the residual lag.

Outputs: crosscheck_result.json + per-run window tables (raw, committed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.motion_dynamics import analyze  # noqa: E402
from pipeline.motion_io import load_motion_csv  # noqa: E402

OUT = Path(__file__).resolve().parent
MOTION = ROOT / "data/motions/thriller/thriller_deploy.csv"
RUNS = [
    "20260708-192839",  # gantry/show day
    "20260708-212451",
    "20260710-141415",  # run cited in the Lane-B feasibility memo
    "20260710-145111",
]
TEL = ROOT / "data/telemetry"
FPS_REF = 30.0
FPS_TEL = 50.0

# joint_order indices (verified identical across runs)
L_AP, R_AP = 4, 10          # ankle pitch
L_AR, R_AR = 5, 11          # ankle roll
LEG_IDX = list(range(12))   # both legs
HIP_KNEE = [0, 1, 3, 6, 7, 9]
ARM_IDX = list(range(15, 29))


def window_slice(t: np.ndarray, w0: float, w1: float) -> np.ndarray:
    return (t >= w0) & (t <= w1)


def amp_ratio(q: np.ndarray, ref: np.ndarray, idx: list[int]) -> float:
    """Achieved/commanded amplitude: std of measured q vs std of reference,
    averaged over joints (guarding tiny-motion joints)."""
    num, den = [], []
    for j in idx:
        s_ref = np.std(ref[:, j])
        if s_ref < 0.03:      # joint basically idle in this window
            continue
        num.append(np.std(q[:, j]))
        den.append(s_ref)
    if not den:
        return float("nan")
    return float(np.mean(np.array(num) / np.array(den)))


def main() -> None:
    # ---- A. predicted demand on the exact deployed native-speed motion -------
    res = analyze(MOTION, fps=FPS_REF, ground=True)
    arr = res.pop("_arrays")
    t_ref = arr["t"]
    demand = arr["ankle_tau_max"]           # Nm per frame (balance-strategy demand)
    windows = res["ankle_flag_windows_s"]
    print(f"[A] predicted (native, {MOTION.name}): "
          f"max {res['dynamic']['ankle_tau_max_nm']} Nm, "
          f"p95 {res['dynamic']['ankle_tau_p95_nm']} Nm, "
          f"{res['dynamic']['ankle_frames_over_headroom_pct']}% frames over, "
          f"windows {windows}")

    ref_m = load_motion_csv(MOTION)
    ref_joints_30 = ref_m[:, 7:]            # (N,29) reference joint angles

    out = {
        "motion": str(MOTION),
        "predicted": {k: res["dynamic"][k] for k in
                      ("ankle_tau_max_nm", "ankle_tau_p95_nm",
                       "ankle_frames_over_headroom_pct")},
        "ankle_flag_windows_s": windows,
        "runs": {},
    }

    # analysis windows: the flagged beats + a calm control window
    probe = [tuple(w) for w in windows if (w[1] - w[0]) >= 0.5]
    calm = [(2.0, 8.0)]

    for run in RUNS:
        p = TEL / f"{run}_ground-run-legodom.npz"
        d = np.load(p, allow_pickle=True)
        q, tgt, tau = d["q"], d["target"], np.abs(d["tau_est"])
        t = d["t"] - d["t"][0]
        n = len(t)

        # ---- alignment refine: xcorr measured vs reference left ankle pitch --
        # resample reference to 50 Hz over the run length
        t50 = np.arange(int(len(ref_joints_30) * FPS_TEL / FPS_REF)) / FPS_TEL
        ref50 = np.empty((len(t50), 29))
        tr = np.arange(len(ref_joints_30)) / FPS_REF
        for j in range(29):
            ref50[:, j] = np.interp(t50, tr, ref_joints_30[:, j])
        m = min(n, len(t50))
        a = q[:m, L_AP] - q[:m, L_AP].mean()
        b = ref50[:m, L_AP] - ref50[:m, L_AP].mean()
        lags = np.arange(-100, 101)          # +-2 s
        xc = [np.dot(a[max(0, -k):m - max(0, k)], b[max(0, k):m - max(0, -k)])
              for k in lags]
        lag = int(lags[int(np.argmax(xc))])  # ticks; ref time = (i - lag)/50
        lag_s = lag / FPS_TEL

        # measured torque stats over the aligned dance span
        span = window_slice(t, 0, m / FPS_TEL)
        ap = np.maximum(tau[:, L_AP], tau[:, R_AP])
        run_row = {
            "ticks": int(n),
            "xcorr_lag_s": round(lag_s, 2),
            "ankle_pitch_measured_nm": {
                "mean": round(float(ap[span].mean()), 2),
                "p95": round(float(np.percentile(ap[span], 95)), 2),
                "max": round(float(ap[span].max()), 2),
            },
            "windows": {},
        }

        for (w0, w1), tag in [*[(w, "FLAG") for w in probe],
                              *[(w, "calm") for w in calm]]:
            # dance-time window -> telemetry ticks (shift by lag)
            s = window_slice(t - lag_s, w0, w1)
            if s.sum() < 10:
                continue
            # matching reference ticks for fidelity
            i0 = max(0, int((w0 + lag_s) * FPS_TEL))
            i1 = min(m, int((w1 + lag_s) * FPS_TEL))
            r0 = max(0, int(w0 * FPS_TEL))
            r1 = r0 + (i1 - i0)
            qw, tw = q[i0:i1], tgt[i0:i1]
            rw = ref50[r0:min(r1, len(ref50))]
            k = min(len(qw), len(rw))
            qw, tw, rw = qw[:k], tw[:k], rw[:k]
            # predicted demand in this window (30 fps clock)
            dm = window_slice(t_ref, w0, w1)
            row = {
                "tag": tag,
                "predicted_demand_nm": {
                    "p95": round(float(np.percentile(demand[dm], 95)), 1),
                    "max": round(float(demand[dm].max()), 1),
                },
                "measured_ankle_pitch_nm": {
                    "mean": round(float(ap[s].mean()), 2),
                    "p95": round(float(np.percentile(ap[s], 95)), 2),
                    "max": round(float(ap[s].max()), 2),
                },
                "tracking_err_deg": {
                    "legs_mean": round(float(np.degrees(
                        np.abs(qw[:, LEG_IDX] - tw[:, LEG_IDX]).mean())), 2),
                    "legs_p95": round(float(np.degrees(np.percentile(
                        np.abs(qw[:, LEG_IDX] - tw[:, LEG_IDX]), 95))), 2),
                    "arms_mean": round(float(np.degrees(
                        np.abs(qw[:, ARM_IDX] - tw[:, ARM_IDX]).mean())), 2),
                },
                "amplitude_ratio_vs_reference": {
                    "legs": round(amp_ratio(qw, rw, LEG_IDX), 3),
                    "hip_knee": round(amp_ratio(qw, rw, HIP_KNEE), 3),
                    "arms": round(amp_ratio(qw, rw, ARM_IDX), 3),
                },
            }
            run_row["windows"][f"{w0}-{w1}s"] = row
        out["runs"][run] = run_row
        print(f"[B/C] {run}: lag {lag_s:+.2f}s, ankle p95 "
              f"{run_row['ankle_pitch_measured_nm']['p95']} Nm, max "
              f"{run_row['ankle_pitch_measured_nm']['max']} Nm")

    (OUT / "crosscheck_result.json").write_text(json.dumps(out, indent=2))
    print("wrote", OUT / "crosscheck_result.json")


if __name__ == "__main__":
    main()
