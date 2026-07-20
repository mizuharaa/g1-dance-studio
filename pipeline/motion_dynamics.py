"""Contact-aware dynamic (+ kinematic) feasibility pass for a G1 reference motion.

Computes the per-joint torque the reference DEMANDS via full floating-base
inverse dynamics with an explicit ground-reaction term:

    tau = M(q)*qacc + C(q,qvel)*qvel + g(q) - sum_i (J_i^T f_i)      (joints 6..)

where f_i is the ground-reaction force assigned to stance foot i. This replaces
the 2026-07-16 "ankle-strategy" model (F_z * ||ZMP - CoM|| billed to one ankle),
which was FALSIFIED against real robot telemetry on 2026-07-20
(experiments/torque_crosscheck_20260720): it predicted ankle p95 114 Nm for the
native-speed Thriller while the real robot measured 15-19 Nm executing it live.
Its three sins, all fixed here:

  1. DOUBLE SUPPORT: 83% of the dance is double-support; a fast weight shift
     moves the ZMP between the feet by LOAD REDISTRIBUTION (nearly free for the
     ankles). Here the net GRF is split between stance feet by the ZMP lever
     rule, and each foot's centre of pressure is clamped to ITS OWN sole — each
     ankle is billed only for its own share at its own lever arm. Demanded ZMP
     outside the support polygon is reported as a "requires stepping / repair"
     window, NOT as fictitious torque (no ankle can exceed ~Mg*toe ~ 34 Nm).
  2. GRAVITY THROUGH THE STANCE LEG: the old base-floated mj_inverse (contact
     disabled, reaction lumped into the base residual) reported only the torque
     to swing each limb — a deep squat's SUSTAINED knee torque was invisible.
     The J^T f term routes body weight through the stance-leg knee/hip/ankle,
     so slow-but-loaded poses are now correctly expensive and fast-but-unloaded
     swings correctly cheap ("high velocity" is NOT "high torque").
  3. ESTIMATOR JITTER: GVHMR/GMR retargets carry frame noise that explodes
     through double differentiation (calm-section demand read 56 Nm vs 12 real).
     Velocities, CoM and angular momentum are zero-phase low-passed at
     LOWPASS_HZ before differencing (dance content sits below ~5 Hz).

Per-joint limits come from pipeline.g1_limits: per-MOTOR-CLASS effort limits
(139 knee/hip-roll, 88 hip-pitch/yaw/waist-yaw, 50 ankle/waist-RP, 25 arm,
5 wrist-PY — NOT one number for 29 joints) with a flat-until-knee torque-speed
derate. Also reported per joint: RMS torque, peak mechanical power |tau*w|,
and saturation duration (% time above SATURATION_FRAC of the envelope).

Validation anchors (2026-07-20, native-speed live runs, tau_est p95):
knee ~31-34 Nm, hip-roll ~27-29, hip-pitch ~19-21, ankle-pitch ~13-16.
A correct demand model for the same motion must land in this decade — the old
one missed by 6-10x. Runs on CPU. See --help for the CLI.
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
from pipeline.motion_io import load_motion_csv

CSV_FPS = 30.0
G = 9.81
# Foot support rectangle around the ankle_roll body origin, in the foot yaw
# frame (approx G1 sole: ~0.10 m toe / 0.07 m heel / 0.03 m half-width).
FOOT_TOE, FOOT_HEEL, FOOT_HALFW = 0.10, 0.07, 0.03
STANCE_BAND = 0.06        # m, foot within this of the lower foot => stance candidate
STANCE_VEL = 0.25         # m/s, reference foot speed above this => swing (not stance)
FLOATY_FOOT_Z = 0.10      # lower foot above this = reference not grounded (advisory)
FLIGHT_Z = 0.25           # both ankle origins above this = genuine airborne
LOWPASS_HZ = 6.0          # zero-phase Butterworth on vel/CoM/angmom pre-diff
SATURATION_FRAC = 0.90    # "saturated" = |tau| above this fraction of envelope
SMOOTH_WIN = 9            # legacy SG window (still used for ZMP inputs)


def _sg_smooth(x: np.ndarray, window: int = SMOOTH_WIN, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing (numpy only) along axis 0."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return _sg_smooth(x[:, None], window, poly)[:, 0]
    N = len(x)
    if window % 2 == 0:
        window += 1
    if N < window:
        return x.copy()
    half = window // 2
    j = np.arange(-half, half + 1)
    A = np.vander(j, poly + 1, increasing=True)
    coef = np.linalg.pinv(A)[0]
    xp = np.pad(x, ((half, half), (0, 0)), mode="edge")
    out = np.empty_like(x)
    for i in range(N):
        out[i] = coef @ xp[i:i + window]
    return out


def _lowpass(x: np.ndarray, fps: float, hz: float = LOWPASS_HZ) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth low-pass along axis 0 (kills estimator
    jitter before differentiation; dance choreography lives below ~5 Hz)."""
    from scipy.signal import butter, filtfilt
    if len(x) < 16 or hz >= fps / 2:
        return np.asarray(x, dtype=float)
    b, a = butter(2, hz / (fps / 2))
    return filtfilt(b, a, x, axis=0)


def _csv_to_qpos(m: np.ndarray) -> np.ndarray:
    """CSV (N,36: xyz | quat xyzw | 29 joints) -> mujoco qpos (xyz | wxyz | 29)."""
    q = np.empty_like(m)
    q[:, 0:3] = m[:, 0:3]
    q[:, 3] = m[:, 6]
    q[:, 4:7] = m[:, 3:6]
    q[:, 7:] = m[:, 7:]
    return q


def analyze(csv_path: str | Path, fps: float = CSV_FPS, ground: bool = True) -> dict:
    """Run the kinematic + contact-aware dynamic feasibility pass. Returns a
    dict with summary flags and (under "_arrays") per-frame arrays for repair.
    ``ground`` re-references the motion so the lowest geom sits at z=0 first."""
    import mujoco

    m = load_motion_csv(csv_path)
    model = L.build_model()                # contact disabled, armatures patched
    data = mujoco.MjData(model)

    if ground:
        from pipeline.grounding import ground_motion
        m, _ = ground_motion(m, model)

    q = _csv_to_qpos(m)
    N = len(q)
    dt = 1.0 / fps
    nv = model.nv
    Mtot = float(model.body_subtreemass[1])

    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("left_ankle_roll_link", "right_ankle_roll_link")]

    # --- velocities (tangent space, low-passed) and accelerations --------------
    qvel = np.zeros((N, nv))
    for i in range(N):
        a = max(i - 1, 0)
        b = min(i + 1, N - 1)
        dv = np.zeros(nv)
        mujoco.mj_differentiatePos(model, dv, dt * (b - a), q[a], q[b])
        qvel[i] = dv
    qvel = _lowpass(qvel, fps)
    qacc = np.zeros((N, nv))
    qacc[1:-1] = (qvel[2:] - qvel[:-2]) / (2 * dt)
    jvel = qvel[:, 6:]

    # --- per-frame CoM / angmom / foot kinematics -------------------------------
    com = np.zeros((N, 3))
    angmom = np.zeros((N, 3))
    foot_pos = np.zeros((N, 2, 3))
    foot_yaw = np.zeros((N, 2))
    for i in range(N):
        data.qpos[:] = q[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, data)
        mujoco.mj_subtreeVel(model, data)
        com[i] = data.subtree_com[1]
        angmom[i] = data.subtree_angmom[1]
        for k, bid in enumerate(feet):
            foot_pos[i, k] = data.xpos[bid]
            R = data.xmat[bid].reshape(3, 3)
            foot_yaw[i, k] = np.arctan2(R[1, 0], R[0, 0])

    com_s = _lowpass(_sg_smooth(com), fps)
    angmom_s = _lowpass(_sg_smooth(angmom), fps)
    com_acc = np.zeros((N, 3))
    com_acc[1:-1] = (com_s[2:] - 2 * com_s[1:-1] + com_s[:-2]) / (dt * dt)
    Hdot = np.zeros((N, 3))
    Hdot[1:-1] = (angmom_s[2:] - angmom_s[:-2]) / (2 * dt)

    # --- net GRF + ZMP (multibody formula, z=0 floor) ---------------------------
    Fz = Mtot * (com_acc[:, 2] + G)
    Fz_safe = np.where(np.abs(Fz) < 1e-6, 1e-6, Fz)
    zmp = np.zeros((N, 2))
    zmp[:, 0] = (Fz * com_s[:, 0] - Mtot * com_acc[:, 0] * com_s[:, 2]
                 - Hdot[:, 1]) / Fz_safe
    zmp[:, 1] = (Fz * com_s[:, 1] - Mtot * com_acc[:, 1] * com_s[:, 2]
                 + Hdot[:, 0]) / Fz_safe
    F_net = np.stack([Mtot * com_acc[:, 0], Mtot * com_acc[:, 1], Fz], axis=1)

    # --- stance detection (height band + reference foot speed, hysteresis-free
    # but velocity-gated so a sliding/swinging foot is not billed as support) ---
    foot_vel = np.zeros((N, 2))
    fp_s = _lowpass(foot_pos.reshape(N, 6), fps).reshape(N, 2, 3)
    foot_vel[1:-1] = np.linalg.norm(
        (fp_s[2:, :, :2] - fp_s[:-2, :, :2]) / (2 * dt), axis=2)
    lower_z = foot_pos[:, :, 2].min(axis=1)
    in_band = foot_pos[:, :, 2] <= (lower_z[:, None] + STANCE_BAND)
    slow = foot_vel <= STANCE_VEL
    cand = in_band & (slow | ~(in_band & slow).any(axis=1, keepdims=True))
    # if neither foot is both low and slow, fall back to the lower foot
    none = ~cand.any(axis=1)
    cand[none, np.argmin(foot_pos[none][:, :, 2], axis=1)] = True
    flight = foot_pos[:, :, 2].min(axis=1) > FLIGHT_Z
    floaty = lower_z > FLOATY_FOOT_Z

    # --- per-foot load share + per-foot CoP clamped to its own sole -------------
    share = np.zeros((N, 2))
    cop_w = np.zeros((N, 2, 3))            # world CoP point per foot (z at sole)
    zmp_deficit = np.zeros(N)              # m, ZMP outside what the feet can cover
    zmp_margin = np.zeros(N)
    stance_code = np.full(N, -1)           # 0 L, 1 R, 2 double, -1 flight
    for i in range(N):
        if flight[i]:
            continue
        c0, c1 = cand[i]
        if c0 and c1:
            stance_code[i] = 2
            a, b = foot_pos[i, 0, :2], foot_pos[i, 1, :2]
            ab = b - a
            den = float(ab @ ab)
            s = float(np.clip(((zmp[i] - a) @ ab) / max(den, 1e-9), 0.0, 1.0))
            share[i] = [1.0 - s, s]
        else:
            k = 0 if c0 else 1
            stance_code[i] = k
            share[i, k] = 1.0
        for k in (0, 1):
            if share[i, k] <= 0.0:
                continue
            # clamp the global ZMP into foot k's sole rectangle (foot yaw frame)
            cy, sy = np.cos(foot_yaw[i, k]), np.sin(foot_yaw[i, k])
            Rk = np.array([[cy, -sy], [sy, cy]])
            local = Rk.T @ (zmp[i] - foot_pos[i, k, :2])
            clamped = np.array([np.clip(local[0], -FOOT_HEEL, FOOT_TOE),
                                np.clip(local[1], -FOOT_HALFW, FOOT_HALFW)])
            cop_w[i, k, :2] = foot_pos[i, k, :2] + Rk @ clamped
            cop_w[i, k, 2] = 0.0
        # achievable ZMP = share-weighted combination of per-foot clamped CoPs
        ach = share[i, 0] * cop_w[i, 0, :2] + share[i, 1] * cop_w[i, 1, :2]
        zmp_deficit[i] = float(np.linalg.norm(zmp[i] - ach))
        zmp_margin[i] = _support_margin(foot_pos[i, :, :2], foot_yaw[i],
                                        cand[i], zmp[i])

    # --- inverse dynamics with the contact term: tau = r - sum_i (J_i^T f_i) ----
    tau = np.zeros((N, L.N_JOINTS))
    jacp = np.zeros((3, nv))
    for i in range(N):
        data.qpos[:] = q[i]
        data.qvel[:] = qvel[i]
        data.qacc[:] = qacc[i]
        mujoco.mj_inverse(model, data)
        t_i = data.qfrc_inverse[6:].copy()
        if not flight[i]:
            for k in (0, 1):
                if share[i, k] <= 0.0:
                    continue
                f_k = share[i, k] * F_net[i]
                mujoco.mj_jac(model, data, jacp, None, cop_w[i, k], feet[k])
                t_i -= (jacp.T @ f_k)[6:]
        tau[i] = t_i
    tau[0] = tau[1]
    tau[-1] = tau[-2]
    tau = _sg_smooth(tau, 5)              # de-fuzz endpoint diffs only

    # --- envelope comparison (per-joint, speed-derated) --------------------------
    lim = L.flag_limit(jvel)
    lim_ankle_capped = lim.copy()
    lim_ankle_capped[:, L.ANKLE_IDX] = np.minimum(
        lim_ankle_capped[:, L.ANKLE_IDX], L.ANKLE_HEADROOM_NM)
    ratio = np.abs(tau) / np.maximum(lim_ankle_capped, 1e-9)
    binding_ratio = ratio.max(axis=1)
    over = ratio > 1.0
    power = np.abs(tau * jvel)
    envelope = L.effective_torque_limit(jvel)
    saturated = np.abs(tau) > SATURATION_FRAC * np.maximum(envelope, 1e-9)

    ankle_abs = np.abs(tau[:, L.ANKLE_IDX])
    ankle_max_per_frame = ankle_abs.max(axis=1)

    # legacy diagnostic: the old single-ankle-strategy upper bound (kept ONLY to
    # quantify how much the old model over-billed; never used for flagging)
    naive = np.abs(Fz) * np.linalg.norm(zmp - com_s[:, :2], axis=1)

    # --- kinematic checks --------------------------------------------------------
    joints = m[:, 7:]
    pos_lo, pos_hi = L.POS_LO, L.POS_HI
    pos_viol = (np.clip(pos_lo - joints, 0, None) + np.clip(joints - pos_hi, 0, None)
                if pos_lo is not None else np.zeros_like(joints))
    vel_over = np.abs(jvel) > L.VELOCITY_LIMIT
    jacc = qacc[:, 6:]

    flag_frames = np.where(over.any(axis=1))[0]
    ankle_flag_frames = np.where((ratio[:, L.ANKLE_IDX] > 1.0).any(axis=1))[0]
    stepping_frames = np.where(zmp_deficit > 0.02)[0]

    res = {
        "file": str(csv_path),
        "frames": int(N),
        "fps": fps,
        "seconds": round(N / fps, 2),
        "total_mass_kg": round(Mtot, 2),
        "model": "contact-aware floating-base ID (per-foot ZMP load share, "
                 f"{LOWPASS_HZ:.0f} Hz zero-phase low-pass)",
        "torque_speed_model": L.summary()["torque_speed_model"],
        "ankle_headroom_nm": L.ANKLE_HEADROOM_NM,
        "dynamic": {
            "ankle_tau_max_nm": round(float(ankle_max_per_frame.max()), 2),
            "ankle_tau_p95_nm": round(float(np.percentile(ankle_max_per_frame, 95)), 2),
            "ankle_frames_over_headroom_pct":
                round(100.0 * len(ankle_flag_frames) / N, 2),
            "any_joint_frames_over_pct": round(100.0 * over.any(axis=1).mean(), 2),
            "binding_ratio_p95": round(float(np.percentile(binding_ratio, 95)), 3),
            "binding_ratio_max": round(float(binding_ratio.max()), 3),
            "per_joint_tau_max_nm": [round(float(v), 2) for v in np.abs(tau).max(axis=0)],
            "per_joint_tau_rms_nm": [round(float(v), 2)
                                     for v in np.sqrt(np.mean(tau ** 2, axis=0))],
            "per_joint_power_peak_w": [round(float(v), 1) for v in power.max(axis=0)],
            "per_joint_saturation_pct": [round(100.0 * float(v), 2)
                                         for v in saturated.mean(axis=0)],
            "per_joint_effort_limit_nm": L.EFFORT_LIMIT_NM.tolist(),
            "naive_ankle_strategy_p95_nm (legacy diagnostic)":
                round(float(np.percentile(naive, 95)), 1),
        },
        "balance": {
            "frames_flight_pct": round(100.0 * float(flight.mean()), 2),
            "floaty_feet_pct": round(100.0 * float(floaty.mean()), 2),
            "double_support_pct": round(100.0 * float((stance_code == 2).mean()), 2),
            "zmp_outside_support_pct": round(100.0 * float((zmp_margin < 0).mean()), 2),
            "zmp_margin_min_m": round(float(zmp_margin.min()), 3),
            "zmp_deficit_max_m": round(float(zmp_deficit.max()), 3),
            "requires_stepping_pct": round(100.0 * len(stepping_frames) / N, 2),
        },
        "kinematic": {
            "pos_worst_violation_rad": round(float(pos_viol.max()), 4),
            "vel_frames_over_limit_pct": round(100.0 * vel_over.any(axis=1).mean(), 2),
            "vel_peak_rad_s": round(float(np.abs(jvel).max()), 2),
            "accel_peak_rad_s2": round(float(np.abs(jacc).max()), 1),
        },
        # flagged windows: ANY joint over its speed-derated envelope (primary),
        # plus stepping-required windows (ZMP not coverable by the stance feet)
        "ankle_flag_windows_s": _windows(flag_frames, fps),
        "requires_stepping_windows_s": _windows(stepping_frames, fps),
    }
    res["_arrays"] = {
        "t": (np.arange(N) / fps),
        "tau": tau,
        "tau_abs": np.abs(tau),
        "flag_limit": lim_ankle_capped,
        "binding_ratio": binding_ratio,
        "ankle_tau_max": ankle_max_per_frame,
        "power": power,
        "zmp": zmp,
        "zmp_margin": zmp_margin,
        "zmp_deficit": zmp_deficit,
        "stance_code": stance_code,
        "share": share,
        "jvel": jvel,
    }
    return res


def _foot_corners(center_xy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[FOOT_TOE, FOOT_HALFW], [FOOT_TOE, -FOOT_HALFW],
                      [-FOOT_HEEL, -FOOT_HALFW], [-FOOT_HEEL, FOOT_HALFW]])
    R = np.array([[c, -s], [s, c]])
    return center_xy + local @ R.T


def _support_margin(fpos_xy, fyaw, contact, zmp_xy) -> float:
    """Signed distance of the ZMP to the support-polygon boundary. >0 inside."""
    pts = []
    for k in (0, 1):
        if contact[k]:
            pts.append(_foot_corners(fpos_xy[k], fyaw[k]))
    if not pts:
        return -1.0
    P = np.vstack(pts)
    return _point_in_hull_margin(zmp_xy, P)


def _point_in_hull_margin(pt, pts) -> float:
    hull = _convex_hull(pts)
    if len(hull) < 3:
        return -np.linalg.norm(pt - pts.mean(axis=0))
    n = len(hull)
    inside = True
    min_edge = np.inf
    for i in range(n):
        a = hull[i]
        b = hull[(i + 1) % n]
        e = b - a
        nrm = np.array([-e[1], e[0]])
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        d = np.dot(pt - a, nrm)
        if d > 0:
            inside = False
        min_edge = min(min_edge, abs(d))
    return min_edge if inside else -min_edge


def _convex_hull(pts):
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _windows(frames, fps, gap=3):
    """Contiguous runs of flagged frame indices -> [start_s, end_s] windows."""
    if len(frames) == 0:
        return []
    out = []
    s = frames[0]
    prev = frames[0]
    for f in frames[1:]:
        if f - prev > gap:
            out.append([round(s / fps, 2), round(prev / fps, 2)])
            s = f
        prev = f
    out.append([round(s / fps, 2), round(prev / fps, 2)])
    return out


def _print(res):
    d, b, k = res["dynamic"], res["balance"], res["kinematic"]
    print(f"{res['file']}: {res['frames']} frames, {res['seconds']}s, "
          f"mass {res['total_mass_kg']} kg  [{res['model']}]")
    print(f"  DYNAMIC  ankle tau max {d['ankle_tau_max_nm']} Nm "
          f"(headroom {res['ankle_headroom_nm']}), p95 {d['ankle_tau_p95_nm']} Nm; "
          f"binding ratio p95 {d['binding_ratio_p95']} max {d['binding_ratio_max']}; "
          f"{d['any_joint_frames_over_pct']}% frames any-joint-over "
          f"(old naive ankle model p95: {d['naive_ankle_strategy_p95_nm (legacy diagnostic)']} Nm)")
    print(f"  BALANCE  double-support {b['double_support_pct']}%, flight "
          f"{b['frames_flight_pct']}%, floaty {b['floaty_feet_pct']}%, "
          f"ZMP-outside {b['zmp_outside_support_pct']}%, "
          f"stepping-required {b['requires_stepping_pct']}% "
          f"(deficit max {b['zmp_deficit_max_m']} m)")
    print(f"  KINEMATIC pos_viol {k['pos_worst_violation_rad']} rad, "
          f"vel_over {k['vel_frames_over_limit_pct']}%, "
          f"vel_peak {k['vel_peak_rad_s']} rad/s")
    print(f"  FLAG WINDOWS (s): {res['ankle_flag_windows_s']}")
    print(f"  STEPPING WINDOWS (s): {res['requires_stepping_windows_s']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--fps", type=float, default=CSV_FPS)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--npz", type=Path, default=None,
                    help="dump per-frame arrays (t, tau, binding_ratio, zmp,...)")
    ap.add_argument("--no-ground", action="store_true")
    args = ap.parse_args()
    res = analyze(args.csv, fps=args.fps, ground=not args.no_ground)
    arrays = res.pop("_arrays")
    _print(res)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2))
        print("wrote", args.json)
    if args.npz:
        args.npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.npz, **arrays)
        print("wrote", args.npz)


if __name__ == "__main__":
    main()
