#!/usr/bin/env python3
"""Fluidity forensics — turn "the legs look clunky / unstable" into numbers.

Context (2026-07-06): the promoted s2r-b policy dances the full Thriller on
hardware, but the user reports legs that don't replicate the reference and an
overall clunky/unstable look, worse than the sim rollout. This tool quantifies
WHERE the fluidity dies and attributes it to causes, from the four full-dance
telemetry runs vs the deployed reference:

  1. TRACKING   per-joint RMS/p95 |q - ref| (deg) per dance section, legs vs
                arms vs waist; decomposed into PLANT error (target - q, what the
                motors fail to do) and POLICY error (target - ref, what the
                policy chose not to command). Sim baseline (box eval of the same
                checkpoint, same metric) is merged in for a sim-vs-HW column.
  2. JITTER     action-rate |a_t - a_{t-1}| per group for four traces:
                reference demand (the action a perfect tracker would need),
                clean-obs ONNX replay (policy-intrinsic jitter), leg-odom ONNX
                replay (adds the estimator's obs artifacts), and hardware.
                Velocity + jerk band split (0-2 / 2-10 / 10-25 Hz) of measured
                q vs the reference — where HW has energy the dance doesn't.
  3. WOBBLE     gyro roll/pitch band RMS during quasi-stance vs stepping
                (phases classified from reference foot heights), compared with
                the reference pelvis angular velocity (the choreography's own
                body rotation demand). 2-10 Hz band = the "unstable look".
  4. CAUSES     leg plant lag/amplitude (same estimator as system_id_arms),
                action-jitter vs stepping-phase correlation (leg-odom-suspect
                windows), and reference style loss legs vs arms (raw retarget
                vs deployed CSV: peak velocity + HF energy kept).

Reads (never writes): data/telemetry/20260706-{114445,115004,115456,133905}_*,
data/policies/thriller/{thriller_deploy.npz,policy_meta.json,policy.onnx},
data/motions/thriller/thriller_g1.csv, data/policies/thriller/thriller_deploy.csv.

Writes: data/reports/fluidity_forensics.json (+ console summary).

Run:  ~/miniconda3/envs/g1dance/bin/python tools/fluidity_forensics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_JSON = ROOT / "data/reports/fluidity_forensics.json"
RUNS = [
    ROOT / "data/telemetry/20260706-114445_ground-run-legodom.npz",
    ROOT / "data/telemetry/20260706-115004_ground-run-legodom.npz",
    ROOT / "data/telemetry/20260706-115456_ground-run-legodom.npz",
    ROOT / "data/telemetry/20260706-133905_ground-run-legodom.npz",
]
REF_NPZ = ROOT / "data/policies/thriller/thriller_deploy.npz"
RAW_CSV = ROOT / "data/motions/thriller/thriller_g1.csv"       # 30 fps raw retarget
DEPLOY_CSV = ROOT / "data/policies/thriller/thriller_deploy.csv"  # 30 fps deployed
SIM_BASELINE = Path("/tmp/claude-1000/-home-alois-g1-dance/"
                    "9308416e-0894-4c9c-a2df-535bed1144aa/scratchpad/"
                    "sim_tracking_s2rb_baseline.json")  # box eval, copied read-only

FS = 50.0
DT = 1.0 / FS
SECTIONS = {          # deploy-npz timeline (includes the 2.5 s activation ramp)
    "0-13s": (0.0, 13.0),
    "13-17.5s": (13.0, 17.5),
    "17.5-25s": (17.5, 25.0),
    "25-36s": (25.0, 36.0),
    "36-40s": (36.0, 40.0),
    "40-49.5s": (40.0, 49.5),
    "49.5-end": (49.5, 60.0),
    "full": (0.0, 60.0),
}
LEG_IDX = list(range(0, 12))
WAIST_IDX = list(range(12, 15))
ARM_IDX = list(range(15, 29))
GROUPS = {"legs": LEG_IDX, "waist": WAIST_IDX, "arms": ARM_IDX}
BANDS = {"motion_0-2Hz": (None, 2.0), "wobble_2-10Hz": (2.0, 10.0),
         "buzz_10-25Hz": (10.0, None)}
PELVIS = 0
FOOT_BODIES = (6, 12)      # left/right ankle_roll_link (lowest mean-z bodies)
FOOT_LIFT_M = 0.04         # foot z above its 5th-pct baseline => leg in flight
STEP_DILATE_S = 0.3        # widen stepping windows (odometry recovers slowly)
MAX_LAG_TICKS = 20         # 400 ms lag search (same as system_id_arms)

# 30 fps CSV layout: root pos (3) + root quat (4) + 29 joints
CSV_JOINT0 = 7
DEPLOY_DANCE_ROW0 = 75 + 45      # ramp (75) + prep pad (45) -> raw frame 0
CSV_FS = 30.0


# ---------------------------------------------------------------- helpers
def band_filter(x, lo, hi, fs=FS):
    """Zero-phase band-limited copy of x along axis 0."""
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


def sec_mask(n, lo, hi):
    t = np.arange(n) * DT
    return (t >= lo) & (t < hi)


def best_lag(sig_ref, sig, max_lag=MAX_LAG_TICKS):
    """Lag (ticks >= 0) at which sig best correlates with sig_ref (from system_id_arms)."""
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


# ---------------------------------------------------------------- ONNX replays
def run_replays(ref_npz):
    """Perfect-tracking ONNX replays through the deploy obs paths.

    Returns dict name -> action trace [T,29]:
      clean  : obs built from npz truth (pelvis vel/quat/gyro, torso disp)
               = the policy's INTRINSIC output on ideal observations
      legodom: obs built with the deploy LegOdometry estimator running on the
               same perfect tracking = adds the estimator's obs artifacts
    """
    import onnxruntime as ort
    import pipeline.deploy_runtime as dr
    from pipeline.leg_odometry import LegOdometry

    meta = dr.Meta(dr.DEFAULT_META)
    ref = dr.Reference(dr.DEFAULT_MOTION)
    d = np.load(ref_npz)
    bq = d["body_quat_w"]
    ba = d["body_ang_vel_w"]
    bv = d["body_lin_vel_w"]
    sess = ort.InferenceSession(str(dr.DEFAULT_POLICY),
                                providers=["CPUExecutionProvider"])
    out = {}

    # -- clean-obs (ideal estimator) --------------------------------------
    last = np.zeros(meta.n)
    acts = np.zeros((ref.T, meta.n))
    for t in range(ref.T):
        imu = bq[t, PELVIS]
        R = dr.quat_wxyz_to_mat(imu)
        gyro = R.T @ ba[t, PELVIS]
        robot_disp = ref.at(t)[2] - ref.apos[0]        # torso disp = truth
        v_world = bv[t, PELVIS]                        # pelvis vel = truth
        obs, _ = dr.build_obs_odom(meta, ref, ref.jp[t], ref.jv[t], imu, gyro,
                                   last, t, robot_disp, v_world)
        a = dr.run_policy(sess, obs, t)
        acts[t] = a
        last = a
    out["clean"] = acts

    # -- leg-odom estimator in the loop (mirrors tools/sim_ground_legodom) --
    odo = LegOdometry(list(meta.joint_order))
    last = np.zeros(meta.n)
    acts = np.zeros((ref.T, meta.n))
    R0 = dr.quat_wxyz_to_mat(bq[0, PELVIS])
    h0 = odo.estimate(ref.jp[0], ref.jv[0], R0, R0.T @ ba[0, PELVIS])[1]
    for t in range(ref.T):
        imu = bq[t, PELVIS]
        R = dr.quat_wxyz_to_mat(imu)
        gyro = R.T @ ba[t, PELVIS]
        v_body, h_est, _ = odo.estimate(ref.jp[t], ref.jv[t], R, gyro)
        v_world = R @ v_body
        rd = ref.at(t)[2] - ref.apos[0]
        robot_disp = np.array([rd[0], rd[1], h_est - h0])
        obs, _ = dr.build_obs_odom(meta, ref, ref.jp[t], ref.jv[t], imu, gyro,
                                   last, t, robot_disp, v_world)
        a = dr.run_policy(sess, obs, t)
        acts[t] = a
        last = a
    out["legodom"] = acts
    return out, meta


# ---------------------------------------------------------------- analyses
def tracking_tables(runs, ref_jp, names):
    """Per-joint / per-group RMS + p95 of |q-ref|, plant (tgt-q), policy (tgt-ref)."""
    T = ref_jp.shape[0]
    per_joint = {}
    per_group = {}
    for j, n in enumerate(names):
        per_joint[n] = {}
        for sname, (lo, hi) in SECTIONS.items():
            m = sec_mask(T, lo, hi)
            errs = [np.degrees(r["q"][m, j] - ref_jp[m, j]) for r in runs]
            e = np.concatenate(errs)
            per_joint[n][sname] = {"rms_deg": rms(e),
                                   "p95_deg": float(np.percentile(np.abs(e), 95))}
    for g, idx in GROUPS.items():
        per_group[g] = {}
        for sname, (lo, hi) in SECTIONS.items():
            m = sec_mask(T, lo, hi)
            tot = {"track": [], "plant": [], "policy": []}
            for r in runs:
                tot["track"].append(np.degrees(r["q"][m][:, idx] - ref_jp[m][:, idx]))
                tot["plant"].append(np.degrees(r["target"][m][:, idx] - r["q"][m][:, idx]))
                tot["policy"].append(np.degrees(r["target"][m][:, idx] - ref_jp[m][:, idx]))
            per_group[g][sname] = {
                "track_rms_deg": rms(np.concatenate(tot["track"])),
                "track_p95_deg": float(np.percentile(np.abs(np.concatenate(tot["track"])), 95)),
                "plant_rms_deg": rms(np.concatenate(tot["plant"])),
                "policy_rms_deg": rms(np.concatenate(tot["policy"])),
            }
    return per_joint, per_group


def action_rate_stats(actions, idx_groups, mask=None):
    """mean/p95 of |a_t - a_{t-1}| summed stats per group; optional tick mask."""
    da = np.abs(np.diff(actions, axis=0))
    if mask is not None:
        da = da[mask[1:]]
    out = {}
    for g, idx in idx_groups.items():
        v = da[:, idx]
        out[g] = {"mean": float(v.mean()), "p95": float(np.percentile(v, 95))}
    return out


def band_rms_by_group(x, idx_groups):
    """Band RMS (per group, mean over joints) of a [T,29] signal."""
    out = {}
    for bname, (lo, hi) in BANDS.items():
        xf = band_filter(x, lo, hi)
        out[bname] = {g: float(rms(xf[:, idx], axis=0).mean())
                      for g, idx in idx_groups.items()}
    return out


def phase_masks(ref_npz):
    """Three phases from REFERENCE foot heights (same mask for every run):
       ramp         0-2.5 s activation ramp (robot should be near-still)
       quasi_stance both feet planted (z < base+3 cm), after the ramp
       swing        at least one foot in real flight (z > base+6 cm)
    The in-between (heel raises / weight shifts) belongs to neither. Thriller is
    step-heavy: swing covers most of the dance."""
    d = np.load(ref_npz)
    z = d["body_pos_w"][:, FOOT_BODIES, 2]
    base = np.percentile(z, 5, axis=0)
    T = z.shape[0]
    t = np.arange(T) * DT
    ramp = t < 2.5
    planted = (z < base + 0.03).all(axis=1) & ~ramp
    swing = (z > base + FOOT_LIFT_M + 0.02).any(axis=1)
    struct = np.ones(int(2 * STEP_DILATE_S * FS) + 1, bool)
    swing = binary_dilation(swing, structure=struct) & ~ramp
    planted = planted & ~swing
    return {"ramp": ramp, "quasi_stance": planted, "swing": swing}


def gyro_wobble(runs, ref_npz, phases):
    """Roll/pitch gyro band RMS per phase + per section, vs reference pelvis demand."""
    d = np.load(ref_npz)
    # reference pelvis angular velocity in the pelvis frame (the demanded 'gyro')
    import pipeline.deploy_runtime as dr
    Rt = np.array([dr.quat_wxyz_to_mat(qq).T for qq in d["body_quat_w"][:, PELVIS]])
    ref_gyro = np.einsum("tij,tj->ti", Rt, d["body_ang_vel_w"][:, PELVIS])
    T = ref_gyro.shape[0]
    out = {}
    for bname, (lo, hi) in BANDS.items():
        row = {}
        hw_f = [band_filter(r["gyro"][:, :2], lo, hi) for r in runs]
        rf_f = band_filter(ref_gyro[:, :2], lo, hi)
        for pname, pmask in phases.items():
            hw = [rms(h[pmask]) for h in hw_f]
            row[pname] = {"hw_rms_rad_s": float(np.mean(hw)),
                          "hw_per_run": [float(v) for v in hw],
                          "ref_rms_rad_s": float(rms(rf_f[pmask]))}
        row["by_section"] = {}
        for sname, (lo_s, hi_s) in SECTIONS.items():
            m = sec_mask(T, lo_s, hi_s)
            row["by_section"][sname] = {
                "hw_rms_rad_s": float(np.mean([rms(h[m]) for h in hw_f])),
                "ref_rms_rad_s": float(rms(rf_f[m]))}
        out[bname] = row
    out["phase_fractions"] = {p: float(m.mean()) for p, m in phases.items()}

    # Coherence split: of the HW roll/pitch gyro power in each band, how much is
    # linearly explained by the reference pelvis rotation (performed choreography)
    # vs incoherent (added wobble)? incoherent_rms = sqrt(sum (1-C^2?) ... uses
    # magnitude-squared coherence Cxy directly: P_incoh = (1-Cxy)*Pxx.
    coh = {}
    for bname, (lo, hi) in BANDS.items():
        lo_f = lo if lo is not None else 0.05
        hi_f = hi if hi is not None else FS / 2 - 0.5
        vals_tot, vals_incoh = [], []
        for r in runs:
            tot_b, inc_b = 0.0, 0.0
            for ax in range(2):
                f, Cxy = signal.coherence(r["gyro"][:, ax], ref_gyro[:, ax],
                                          fs=FS, nperseg=256)
                _, Pxx = signal.welch(r["gyro"][:, ax], fs=FS, nperseg=256)
                m = (f >= lo_f) & (f < hi_f)
                df = f[1] - f[0]
                tot_b += float((Pxx[m]).sum() * df)
                inc_b += float(((1.0 - Cxy[m]) * Pxx[m]).sum() * df)
            vals_tot.append(tot_b)
            vals_incoh.append(inc_b)
        coh[bname] = {
            "hw_rms_rad_s": float(np.mean(np.sqrt(vals_tot))),
            "incoherent_rms_rad_s": float(np.mean(np.sqrt(vals_incoh))),
            "incoherent_power_frac": float(np.mean(np.array(vals_incoh) /
                                                   np.maximum(vals_tot, 1e-12))),
        }
    out["coherence_split_full_run"] = coh
    return out


def leg_lag_table(runs, names, ref_jp):
    """Two lags per joint:
       vs_target — q behind the PD command (plant response). CAVEAT for legs: the
                   policy swings leg targets as TORQUE levers (kp*err), so low
                   corr/amp there is control style, not only plant softness.
       vs_ref    — q behind the choreography (the timing the audience sees)."""
    out = {}
    for j in LEG_IDX + ARM_IDX:
        n = names[j]
        vt, vr = [], []
        for r in runs:
            vt.append(lag_amp(r["target"][:, j], r["q"][:, j]))
            vr.append(lag_amp(ref_jp[:, j], r["q"][:, j]))
        out[n] = {
            "vs_target": {"lag_ms": float(np.mean([v[0] for v in vt])),
                          "lag_corr": float(np.mean([v[1] for v in vt])),
                          "amp_ratio": float(np.mean([v[2] for v in vt]))},
            "vs_ref": {"lag_ms": float(np.mean([v[0] for v in vr])),
                       "lag_corr": float(np.mean([v[1] for v in vr])),
                       "amp_ratio": float(np.mean([v[2] for v in vr]))},
        }
    return out


def step_burst_correlation(runs, replays, ref_rate_demand, step):
    """Does leg action jitter fire in the leg-odom-suspect (stepping) windows?"""
    k = int(0.2 * FS)  # 0.2 s smoothing
    kern = np.ones(k) / k

    def leg_rate(actions):
        da = np.abs(np.diff(actions, axis=0))[:, LEG_IDX].mean(axis=1)
        return np.convolve(da, kern, mode="same")

    sm = step[1:].astype(float)
    rows = {}
    traces = {"hw_run%d" % i: r["action"] for i, r in enumerate(runs, 1)}
    traces["replay_clean"] = replays["clean"]
    traces["replay_legodom"] = replays["legodom"]
    traces["ref_demand"] = ref_rate_demand
    for name, act in traces.items():
        x = leg_rate(act)
        r = float(np.corrcoef(x, sm)[0, 1])
        ratio = float(x[step[1:]].mean() / x[~step[1:]].mean())
        rows[name] = {"corr_with_stepping": r, "stepping_over_stance_ratio": ratio}
    return rows


def reference_style_loss():
    """Raw retarget vs deployed CSV (30 fps): what the front-end took from the LEGS."""
    raw = np.loadtxt(RAW_CSV, delimiter=",")[:, CSV_JOINT0:CSV_JOINT0 + 29]
    dep_full = np.loadtxt(DEPLOY_CSV, delimiter=",")[:, CSV_JOINT0:CSV_JOINT0 + 29]
    dep = dep_full[DEPLOY_DANCE_ROW0:DEPLOY_DANCE_ROW0 + raw.shape[0]]
    v_raw = np.diff(raw, axis=0) * CSV_FS
    v_dep = np.diff(dep, axis=0) * CSV_FS

    def hf_energy(v):  # 2-10 Hz velocity energy at 30 fps
        ny = CSV_FS / 2.0
        b, a = signal.butter(4, [2.0 / ny, 10.0 / ny], "band")
        return np.square(signal.filtfilt(b, a, v, axis=0)).sum(axis=0)

    e_raw, e_dep = hf_energy(v_raw), hf_energy(v_dep)
    out = {}
    for g, idx in GROUPS.items():
        out[g] = {
            "peak_vel_raw_rad_s": float(np.abs(v_raw[:, idx]).max()),
            "peak_vel_deploy_rad_s": float(np.abs(v_dep[:, idx]).max()),
            "hf_2_10Hz_energy_kept": float(e_dep[idx].sum() / e_raw[idx].sum()),
        }
    return out


# ---------------------------------------------------------------- main
def main():
    ref = np.load(REF_NPZ)
    ref_jp = ref["joint_pos"].astype(float)
    ref_jv = ref["joint_vel"].astype(float)
    T = ref_jp.shape[0]

    runs = []
    for p in RUNS:
        d = dict(np.load(p, allow_pickle=False))
        assert d["q"].shape[0] == T, f"{p.name}: {d['q'].shape[0]} ticks != ref {T}"
        runs.append(d)
    names = [str(x) for x in np.asarray(runs[0]["joint_order"]).tolist()]

    import pipeline.deploy_runtime as dr
    meta = dr.Meta(dr.DEFAULT_META)
    # reference-demand action: the action a perfect plant would need
    a_demand = (ref_jp - meta.default[None, :]) / meta.action_scale[None, :]

    print("== ONNX replays (perfect tracking; clean obs vs leg-odom obs) ==")
    replays, _ = run_replays(REF_NPZ)

    phases = phase_masks(REF_NPZ)
    step = phases["swing"]

    print("== tracking ==")
    per_joint, per_group = tracking_tables(runs, ref_jp, names)

    print("== jitter ==")
    jitter = {
        "action_rate": {
            "ref_demand": action_rate_stats(a_demand, GROUPS),
            "replay_clean": action_rate_stats(replays["clean"], GROUPS),
            "replay_legodom": action_rate_stats(replays["legodom"], GROUPS),
            "hw": {g: {"mean": float(np.mean([action_rate_stats(r["action"], GROUPS)[g]["mean"]
                                              for r in runs])),
                       "p95": float(np.mean([action_rate_stats(r["action"], GROUPS)[g]["p95"]
                                             for r in runs]))}
                   for g in GROUPS},
            "hw_stepping": {g: {"mean": float(np.mean(
                [action_rate_stats(r["action"], GROUPS, mask=step)[g]["mean"] for r in runs]))}
                for g in GROUPS},
            "hw_stance": {g: {"mean": float(np.mean(
                [action_rate_stats(r["action"], GROUPS, mask=~step)[g]["mean"] for r in runs]))}
                for g in GROUPS},
        },
        "action_band_rms": {
            "ref_demand": band_rms_by_group(a_demand, GROUPS),
            "replay_clean": band_rms_by_group(replays["clean"], GROUPS),
            "replay_legodom": band_rms_by_group(replays["legodom"], GROUPS),
            "hw": {b: {g: float(np.mean([band_rms_by_group(r["action"], GROUPS)[b][g]
                                         for r in runs])) for g in GROUPS}
                   for b in BANDS},
        },
        "velocity_band_rms_rad_s": {
            "ref": band_rms_by_group(ref_jv, GROUPS),
            "hw": {},
        },
        "jerk_band_rms_rad_s3": {"ref": {}, "hw": {}},
    }
    hw_v = [band_rms_by_group(r["dq"], GROUPS) for r in runs]
    for b in BANDS:
        jitter["velocity_band_rms_rad_s"]["hw"][b] = {
            g: float(np.mean([h[b][g] for h in hw_v])) for g in GROUPS}
    # jerk = d2(dq)/dt2, band-filtered
    def jerk(dq):
        return np.gradient(np.gradient(dq, DT, axis=0), DT, axis=0)
    jr = jerk(ref_jv)
    hw_j = [band_rms_by_group(jerk(r["dq"]), GROUPS) for r in runs]
    ref_j = band_rms_by_group(jr, GROUPS)
    for b in BANDS:
        jitter["jerk_band_rms_rad_s3"]["ref"][b] = ref_j[b]
        jitter["jerk_band_rms_rad_s3"]["hw"][b] = {
            g: float(np.mean([h[b][g] for h in hw_j])) for g in GROUPS}

    print("== wobble ==")
    wobble = gyro_wobble(runs, REF_NPZ, phases)

    print("== lag ==")
    lag = leg_lag_table(runs, names, ref_jp)

    print("== bursts ==")
    bursts = step_burst_correlation(runs, replays, a_demand, step)

    print("== reference style ==")
    style = reference_style_loss()

    sim_baseline = json.loads(SIM_BASELINE.read_text()) if SIM_BASELINE.exists() else None

    out = {
        "runs": [p.name for p in RUNS],
        "sections_s": {k: list(v) for k, v in SECTIONS.items()},
        "phase_fractions": {p: float(m.mean()) for p, m in phases.items()},
        "tracking_per_joint": per_joint,
        "tracking_per_group": per_group,
        "sim_tracking_baseline": sim_baseline,
        "jitter": jitter,
        "wobble": wobble,
        "plant_lag": lag,
        "burst_correlation": bursts,
        "reference_style_loss": style,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {OUT_JSON}")

    # ---------- console summary ----------
    print("\nTRACKING RMS deg (4-run agg)  [track | plant | policy]")
    for g in GROUPS:
        row = " ".join(f"{s}:{per_group[g][s]['track_rms_deg']:5.1f}"
                       for s in ("0-13s", "13-17.5s", "25-36s", "40-49.5s", "full"))
        f = per_group[g]["full"]
        print(f"  {g:<6} {row}   plant {f['plant_rms_deg']:.1f} policy {f['policy_rms_deg']:.1f}")
    print("\nWorst leg joints (full, RMS deg):")
    legs = sorted(((per_joint[names[j]]["full"]["rms_deg"], names[j]) for j in LEG_IDX),
                  reverse=True)
    for v, n in legs[:6]:
        print(f"  {n:<26} {v:5.1f}")
    print("\nACTION-RATE mean |da| (legs): demand "
          f"{jitter['action_rate']['ref_demand']['legs']['mean']:.4f}  clean "
          f"{jitter['action_rate']['replay_clean']['legs']['mean']:.4f}  legodom "
          f"{jitter['action_rate']['replay_legodom']['legs']['mean']:.4f}  HW "
          f"{jitter['action_rate']['hw']['legs']['mean']:.4f}")
    w = wobble["wobble_2-10Hz"]
    print("GYRO roll/pitch 2-10 Hz RMS (rad/s), HW vs ref demand:")
    for p in ("ramp", "quasi_stance", "swing"):
        print(f"  {p:<13} hw {w[p]['hw_rms_rad_s']:.3f}  ref {w[p]['ref_rms_rad_s']:.3f}"
              f"  (frac {wobble['phase_fractions'][p]:.2f})")
    print("  by section: " + "  ".join(
        f"{s} hw {w['by_section'][s]['hw_rms_rad_s']:.3f}/ref {w['by_section'][s]['ref_rms_rad_s']:.3f}"
        for s in ("0-13s", "13-17.5s", "25-36s", "40-49.5s")))


if __name__ == "__main__":
    main()
