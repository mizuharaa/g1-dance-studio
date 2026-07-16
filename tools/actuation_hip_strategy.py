"""HIP-STRATEGY REDISTRIBUTION ANALYSIS (Agent D — actuation/control).

QUESTION THIS ANSWERS
---------------------
Agent B proved a global 2.5x slowdown makes Thriller feasible under a PURE
ANKLE strategy (support ankle carries the whole CoM->ZMP moment). 2.5x (~2 min,
half speed) is slow for the show. The user's #1 priority: find the MILDEST safe
slowdown when the HIPS/TORSO help balance (hip strategy), because the G1's hips
have far more torque headroom (hip-roll 139 Nm, hip-pitch 88 Nm each) than the
ankles (~40 Nm usable).

PHYSICS (reuses pipeline.motion_dynamics' centroidal machinery verbatim)
------------------------------------------------------------------------
motion_dynamics computes the support-ankle balance moment as

    tau_ankle(t) = F_z(t) * || ZMP(t) - CoM_xy(t) ||          (ankle-only)

where ZMP already contains the reference's own rate-of-change of centroidal
angular momentum Hdot (the ZMP formula subtracts Hdot_y/F_z in x, adds
Hdot_x/F_z in y). Splitting into axes:

    tau_x(t) = F_z * |ZMP_x - CoM_x|   (sagittal, carried by ankle_PITCH)
    tau_y(t) = F_z * |ZMP_y - CoM_y|   (lateral,  carried by ankle_ROLL)

HIP STRATEGY = deliberately inject EXTRA centroidal angular-momentum rate beyond
the reference so the required ZMP moves back toward the foot, unloading the
ankle. Extra Hdot has torque units (Nm), so in TORQUE space the substitution is
a clean subtraction:

    tau_x_hip(t) = max(0, tau_x(t) - dHdot_sag(t))
    tau_y_hip(t) = max(0, tau_y(t) - dHdot_lat(t))
    tau_ankle_hip(t) = sqrt(tau_x_hip^2 + tau_y_hip^2)

  * sagittal Hdot_y is produced by hip_PITCH (88 Nm x2) + waist_pitch (50 Nm)
  * lateral  Hdot_x is produced by hip_ROLL  (139 Nm x2) + waist_roll (50 Nm)

WHY HIP STRATEGY DOES NOT FULLY REPLACE THE SLOWDOWN (the honest bound)
----------------------------------------------------------------------
The trunk is a bounded flywheel: it cannot counter-rotate forever, so hip
strategy can only cancel the *transient* (fast) part of the ankle demand, not
the *sustained* (quasi-static lean) part. We therefore split the ankle demand
into a sustained component (moving average over W_HIP, the trunk counter-rotate
timescale) and a transient component; the hip can remove up to C_HIP Nm of the
transient but can NEVER pull the demand below the sustained lean:

    tau_ankle_hip(t) = max( tau_sustained(t), tau_ref(t) - C_HIP )

C_HIP (Nm) = how much angular-momentum RATE the trunk can realistically inject.
We report a BAND, since the exact value needs GPU sim confirmation:
    conservative 40, moderate 70, aggressive 100 Nm.
(Torque-headroom ceiling from the limbs is ~90 Nm sagittal / ~130 Nm lateral, so
100 Nm is the physical upper bound; 40 Nm is a single-axis, excursion-limited
floor.)

OUTPUT: per-factor {max, p95, %>40} of the hip-assisted ankle demand, and the
mildest factor that keeps the hip-assisted MAX <= 40 Nm, for each C_HIP.
Every number is reproduced from the reference CSV on CPU; raw JSON is written.
"""
from __future__ import annotations

import argparse
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
from tools.motion_repair import global_slowdown

G = 9.81

# joint indices for the hip-assist torque budget
HIP_PITCH = [L.IDX["left_hip_pitch_joint"], L.IDX["right_hip_pitch_joint"]]
HIP_ROLL = [L.IDX["left_hip_roll_joint"], L.IDX["right_hip_roll_joint"]]
WAIST_PITCH = [L.IDX["waist_pitch_joint"]]
WAIST_ROLL = [L.IDX["waist_roll_joint"]]


def centroidal_profile(m: np.ndarray, fps: float) -> dict:
    """Replicate motion_dynamics.analyze's centroidal computation but EXPOSE the
    per-axis pieces (F_z, CoM_xy, ZMP_xy, per-joint limb torque) that the ankle-
    only pass folds into one scalar. Identical smoothing/central-diff as MD so the
    ankle magnitude matches Agent B's number to rounding."""
    import mujoco
    from pipeline.grounding import ground_motion

    model = L.build_model()
    data = mujoco.MjData(model)
    m, _ = ground_motion(m, model)
    q = MD._csv_to_qpos(m)
    N = len(q)
    dt = 1.0 / fps
    nv = model.nv
    Mtot = float(model.body_subtreemass[1])

    qvel = np.zeros((N, nv))
    for i in range(N):
        a = max(i - 1, 0); b = min(i + 1, N - 1)
        dv = np.zeros(nv)
        mujoco.mj_differentiatePos(model, dv, dt * (b - a), q[a], q[b])
        qvel[i] = dv
    qacc = np.zeros((N, nv))
    qacc[1:-1] = (qvel[2:] - qvel[:-2]) / (2 * dt)
    jvel = qvel[:, 6:]

    com = np.zeros((N, 3)); angmom = np.zeros((N, 3))
    for i in range(N):
        data.qpos[:] = q[i]; data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        mujoco.mj_subtreeVel(model, data)
        com[i] = data.subtree_com[1]
        angmom[i] = data.subtree_angmom[1]

    com_s = MD._sg_smooth(com); angmom_s = MD._sg_smooth(angmom)
    com_acc = np.zeros((N, 3))
    com_acc[1:-1] = (com_s[2:] - 2 * com_s[1:-1] + com_s[:-2]) / (dt * dt)
    Hdot = np.zeros((N, 3))
    Hdot[1:-1] = (angmom_s[2:] - angmom_s[:-2]) / (2 * dt)
    com = com_s

    Fz = Mtot * (com_acc[:, 2] + G)
    Fz_safe = np.where(np.abs(Fz) < 1e-6, 1e-6, Fz)
    zmp = np.zeros((N, 2))
    zmp[:, 0] = (Mtot * (com_acc[:, 2] + G) * com[:, 0]
                 - Mtot * com_acc[:, 0] * com[:, 2] - Hdot[:, 1]) / Fz_safe
    zmp[:, 1] = (Mtot * (com_acc[:, 2] + G) * com[:, 1]
                 - Mtot * com_acc[:, 1] * com[:, 2] + Hdot[:, 0]) / Fz_safe

    # per-axis ankle-only balance torque (matches MD's scalar in magnitude)
    Fz_abs = np.abs(Fz)
    tau_x = MD._sg_smooth(Fz_abs * np.abs(zmp[:, 0] - com[:, 0]), 5)
    tau_y = MD._sg_smooth(Fz_abs * np.abs(zmp[:, 1] - com[:, 1]), 5)
    for arr in (tau_x, tau_y):
        arr[0] = arr[1]; arr[-1] = arr[-2]

    # per-joint reference limb torque (for the torque-headroom ceiling)
    tau_limb = np.zeros((N, L.N_JOINTS))
    for i in range(N):
        data.qpos[:] = q[i]; data.qvel[:] = qvel[i]; data.qacc[:] = qacc[i]
        mujoco.mj_inverse(model, data)
        tau_limb[i] = data.qfrc_inverse[6:].copy()
    tau_limb[0] = tau_limb[1]; tau_limb[-1] = tau_limb[-2]

    return {"N": N, "fps": fps, "Mtot": Mtot,
            "tau_x": tau_x, "tau_y": tau_y,
            "tau_ankle": np.sqrt(tau_x ** 2 + tau_y ** 2),
            "tau_limb": np.abs(tau_limb), "jvel": jvel}


def _moving_avg(x: np.ndarray, win_frames: int) -> np.ndarray:
    if win_frames <= 1:
        return x.copy()
    k = np.ones(win_frames) / win_frames
    return np.convolve(np.pad(x, win_frames // 2, mode="edge"), k, mode="same")[
        win_frames // 2: win_frames // 2 + len(x)]


def hip_assist(prof: dict, c_hip: float, w_hip_s: float) -> np.ndarray:
    """Hip-assisted ankle demand per frame (Nm). Per-axis: subtract the trunk's
    transient-cancellation authority split proportional to each axis' torque
    ceiling, but never pull an axis below its own sustained (moving-average) lean.

    c_hip = total trunk angular-momentum-rate authority (Nm), split sagittal:lateral
    by the joint-effort ceilings (~90:130). w_hip_s = trunk counter-rotate timescale."""
    fps = prof["fps"]
    win = max(1, int(round(w_hip_s * fps)))
    # split the trunk authority between the two planes by their effort ceilings
    sag_ceiling = 2 * L.EFFORT_LIMIT_NM[HIP_PITCH[0]] + L.EFFORT_LIMIT_NM[WAIST_PITCH[0]]   # 226
    lat_ceiling = 2 * L.EFFORT_LIMIT_NM[HIP_ROLL[0]] + L.EFFORT_LIMIT_NM[WAIST_ROLL[0]]      # 328
    c_sag = c_hip * sag_ceiling / (sag_ceiling + lat_ceiling)
    c_lat = c_hip * lat_ceiling / (sag_ceiling + lat_ceiling)

    out = {}
    for ax, c in (("tau_x", c_sag), ("tau_y", c_lat)):
        tau = prof[ax]
        sustained = _moving_avg(tau, win)
        # hip cancels up to c off the peak; never below the sustained lean
        out[ax] = np.maximum(sustained, tau - c)
    return np.sqrt(out["tau_x"] ** 2 + out["tau_y"] ** 2)


def summarize(tau: np.ndarray, headroom: float = L.ANKLE_HEADROOM_NM) -> dict:
    return {"max": round(float(tau.max()), 2),
            "p95": round(float(np.percentile(tau, 95)), 2),
            "pct_over": round(100.0 * float((tau > headroom).mean()), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--factors", default="1.0,1.3,1.5,1.7,1.9,2.0,2.5")
    ap.add_argument("--c-hip", default="0,40,70,100",
                    help="trunk angular-momentum-rate authority band (Nm)")
    ap.add_argument("--w-hip", type=float, default=0.4,
                    help="trunk counter-rotate timescale (s)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    m0 = load_motion_csv(args.csv)
    factors = [float(x) for x in args.factors.split(",")]
    c_hips = [float(x) for x in args.c_hip.split(",")]

    HARD_CLAMP = 50.0   # ankle effort_limit_sim (flat); brief touches OK, sustained not
    MAX_OVER_PCT = 3.0  # tolerate <=3% of frames in the 40-50 Nm transient band

    def strict_ok(s):     # motor-protective: every frame under the 40 Nm usable cap
        return s["max"] <= L.ANKLE_HEADROOM_NM

    def practical_ok(s):  # thermal/show: sustained p95<=40, brief peaks<=hard clamp
        return (s["p95"] <= L.ANKLE_HEADROOM_NM and s["max"] <= HARD_CLAMP
                and s["pct_over"] <= MAX_OVER_PCT)

    print(f"SOURCE {args.csv}  hips-help model  W_hip={args.w_hip}s  "
          f"ankle usable {L.ANKLE_HEADROOM_NM} Nm  hard clamp {HARD_CLAMP} Nm\n")
    header = "factor  dur(s) " + "  ".join(
        f"C={int(c):>3}[max/p95/%>40]" for c in c_hips)
    print(header)

    results = {"source": args.csv, "w_hip_s": args.w_hip,
               "ankle_headroom_nm": L.ANKLE_HEADROOM_NM, "hard_clamp_nm": HARD_CLAMP,
               "practical_criterion": "p95<=40 & max<=50 & pct_over40<=3",
               "c_hip_band_nm": c_hips, "factors": {}}
    mildest_strict = {c: None for c in c_hips}
    mildest_practical = {c: None for c in c_hips}

    for f in factors:
        m = m0 if f == 1.0 else global_slowdown(m0, f, args.fps)
        prof = centroidal_profile(m, args.fps)
        dur = round(prof["N"] / args.fps, 1)
        row = f"{f:4.2f}  {dur:6.1f} "
        results["factors"][f] = {"dur_s": dur, "by_c_hip": {}}
        for c in c_hips:
            tau = hip_assist(prof, c, args.w_hip)
            s = summarize(tau)
            results["factors"][f]["by_c_hip"][c] = s
            row += f"  {s['max']:5.1f}/{s['p95']:4.1f}/{s['pct_over']:4.1f}"
            if strict_ok(s) and mildest_strict[c] is None:
                mildest_strict[c] = f
            if practical_ok(s) and mildest_practical[c] is None:
                mildest_practical[c] = f
        print(row)

    def _fmt(d, c):
        return f">{factors[-1]}x" if d[c] is None else f"{d[c]}x"

    print("\nMILDEST SAFE SLOWDOWN by criterion:")
    print(f"  {'trunk authority':<30} {'STRICT(max<=40)':<16} PRACTICAL(p95<=40,max<=50)")
    for c in c_hips:
        tag = ("ankle-only (Agent B)" if c == 0 else f"C_HIP={int(c)} Nm")
        print(f"  {tag:<30} {_fmt(mildest_strict, c):<16} {_fmt(mildest_practical, c)}")
    results["mildest_strict_factor"] = {c: mildest_strict[c] for c in c_hips}
    results["mildest_practical_factor"] = {c: mildest_practical[c] for c in c_hips}

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        print("\nwrote", args.json)


if __name__ == "__main__":
    main()
