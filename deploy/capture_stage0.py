#!/usr/bin/env python3
"""Stage-0 READ-ONLY robot measurement capture + offline analysis.

WHY (docs/first_principles_audit.md §4/§6): the project's load-bearing hardware
numbers are single-sample, uncommitted, or assumed — ankle ~15 Nm (1 run, dirty
window), "onboard controller stands at ~0 Nm ankle" (assumed, never measured),
thermal 22.5 C/min (one 34-s datapoint), obs latency (never measured). This tool
captures all of them cleanly, with provenance, at the next robot session.

READ-ONLY GUARANTEE (hard rule): this tool creates DDS SUBSCRIBERS ONLY.
It never constructs a publisher, never sends LowCmd, never touches the
MotionSwitcherClient. It is explicitly safe to run while the ONBOARD controller
is standing the robot — that is the primary use case (audit §4 exp #5).

Usage (in the `tv` conda env, robot powered, laptop on the robot LAN):

    python deploy/capture_stage0.py --minutes 3 --label onboard-standby
    python deploy/capture_stage0.py --analyze data/telemetry/<file>.npz   # offline

What it answers:
  a. OBS STALENESS / LATENCY (audit §4 exp #4): wall-clock read intervals,
     LowState tick advance, tick-vs-wall drift -> p50/p95/max staleness in ms,
     and whether obs latency beyond ~2 ms exists at all.
     LowState tick field: `tick: types.uint32` in the unitree_hg IDL
     (~/robot/unitree_sdk2_python/unitree_sdk2py/idl/unitree_hg/msg/dds_/
     _LowState_.py:28; nominally 1 ms per count — the tool measures the unit
     empirically against the wall clock rather than assuming it).
  b. STANDING TORQUE BASELINE (audit §4 exp #5, §6.1/§6.4): per-joint tau_est
     mean/RMS/p95 for the 12 leg motors, ankle_pitch L/R highlighted —
     calibrates tau_est semantics against a known stance and tests the assumed
     "onboard ~0 Nm ankle" (healthy prediction: 3-6 Nm, <2 C/min).
  c. THERMAL (audit §6.5/§6.7): per-motor temperature start/end/rate (C/min),
     hottest + fastest-heating motor NAMED (the old "~20 Nm continuous" never
     printed the argmax).
  d. KP*ERR IDENTITY CHECK (audit §4 exp #3, §6.1/§6.3), via --analyze on a
     deploy_runtime Telemetry run npz: per-joint regression slope of
     tau_est vs kp*(target-q) - kd*dq -> torque delivery ratio (the audit's
     "knee delivers 0.2-0.4x" question). Fully offline.

Output: data/telemetry/<stamp>_stage0_<label>.npz (+ <same>.analysis.json).
Imports cleanly WITHOUT the Unitree SDK (all SDK imports are lazy) so the
analysis path runs anywhere, and tests run offline.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TELEMETRY_DIR = Path(os.environ.get("TELEMETRY_DIR", str(ROOT / "data" / "telemetry")))
SAMPLE_HZ = 50.0
N_MOTORS = 29
DEFAULT_IFACE = "enp0s31f6"
ODOM_TOPIC = "rt/odommodestate"

# LowState motor index == joint index in the project-wide 29-DoF order (same
# convention deploy_runtime.read_state uses; matches policy_meta.json joint_order_29dof).
G1_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
LEG_IDX = list(range(12))
ANKLE_PITCH_IDX = (4, 10)  # left/right ankle_pitch


# =============================== analysis (pure) ===============================
# Everything below is numpy-only and runs offline on recorded npz files.

def _pctl(x) -> dict:
    x = np.asarray(x, float)
    if x.size == 0:
        return {"p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {"p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)),
            "max": float(np.max(x))}


def staleness_stats(t_mono, tick) -> dict:
    """Audit §4 exp #4 — obs staleness/latency from a stage-0 capture.

    t_mono: monotonic receive times (s) of consecutive 50 Hz reads.
    tick:   LowState_.tick per read (uint32, nominally 1 ms/count; unit is
            estimated empirically here, never assumed).

    Reports (all in ms): wall-clock read-interval distribution, msg-tick advance
    distribution, and the tick-vs-wall STALENESS — the residual of receive time
    against a linear fit on the (unwrapped) tick, re-zeroed so the freshest
    observed read defines ~0. Clock offset between robot and laptop cancels;
    clock-rate drift is removed by the fit and reported as ppm.
    """
    t = np.asarray(t_mono, float)
    k = np.asarray(tick, float)
    out = {"n": int(t.size)}
    if t.size < 3:
        out["error"] = "too few samples for staleness stats"
        return out
    dk = np.diff(k)
    dk = np.where(dk < 0, dk + 2.0 ** 32, dk)        # uint32 wraparound
    k_un = np.concatenate([[0.0], np.cumsum(dk)])     # unwrapped ticks from start
    wall_ms = (t - t[0]) * 1000.0
    out["duration_s"] = float(t[-1] - t[0])
    out["wall_read_interval_ms"] = _pctl(np.diff(wall_ms))
    # Empirical tick unit (ms of wall time per tick count). ~1.0 => tick is in ms.
    unit = float(wall_ms[-1] / k_un[-1]) if k_un[-1] > 0 else float("nan")
    out["tick_unit_ms_est"] = unit
    unit_ok = np.isfinite(unit) and unit > 0
    out["tick_advance_ms"] = _pctl(dk * (unit if unit_ok else 1.0))
    # Same LowState read twice in a row (we outpaced the publisher / stale reread).
    out["repeated_tick_fraction"] = float(np.mean(dk == 0))
    if unit_ok:
        A = np.vstack([k_un, np.ones(k_un.size)]).T
        coef, *_ = np.linalg.lstsq(A, wall_ms, rcond=None)
        resid = wall_ms - A @ coef
        stale = resid - resid.min()   # freshest observed read ~= 0 staleness
        out["staleness_ms"] = _pctl(stale)
        # drift of the robot tick clock vs the laptop wall clock, vs nominal 1 ms/tick
        out["tick_rate_vs_wall_ppm"] = float((coef[0] - 1.0) * 1e6)
        out["latency_beyond_2ms"] = bool(out["staleness_ms"]["p95"] > 2.0)
    else:
        out["error"] = "tick never advanced — cannot compute staleness"
    return out


def torque_summary(tau_est, joint_order=None) -> dict:
    """Audit §4 exp #5 — per-joint tau_est mean/RMS/p95(|.|), legs highlighted."""
    tau = np.atleast_2d(np.asarray(tau_est, float))
    names = list(joint_order) if joint_order is not None else G1_JOINT_ORDER
    per = {}
    for i in range(min(tau.shape[1], len(names))):
        col = tau[:, i]
        per[names[i]] = {
            "mean_nm": float(np.mean(col)),
            "rms_nm": float(np.sqrt(np.mean(col ** 2))),
            "p95_abs_nm": float(np.percentile(np.abs(col), 95)),
        }
    legs = {names[i]: per[names[i]] for i in LEG_IDX if i < len(names) and names[i] in per}
    ankles = {n: s for n, s in per.items() if "ankle_pitch" in n}
    return {"n_samples": int(tau.shape[0]), "per_joint": per,
            "legs": legs, "ankle_pitch": ankles}


def thermal_summary(temp, t, joint_order=None) -> dict:
    """Audit §6.5/§6.7 — per-motor temperature start/end/rate; argmax NAMED."""
    temp = np.atleast_2d(np.asarray(temp, float))
    t = np.asarray(t, float)
    names = list(joint_order) if joint_order is not None else G1_JOINT_ORDER
    dur_min = float(t[-1] - t[0]) / 60.0 if t.size >= 2 else float("nan")
    k = max(1, min(5, temp.shape[0] // 2))   # average a few samples at each end
    start = temp[:k].mean(axis=0)
    end = temp[-k:].mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = (end - start) / dur_min if dur_min and dur_min > 0 else np.full_like(start, np.nan)
    per = {}
    for i in range(min(temp.shape[1], len(names))):
        per[names[i]] = {"start_C": float(start[i]), "end_C": float(end[i]),
                         "rate_C_per_min": float(rate[i])}
    i_hot = int(np.nanargmax(end)) if np.any(np.isfinite(end)) else 0
    i_fast = int(np.nanargmax(rate)) if np.any(np.isfinite(rate)) else 0
    return {
        "duration_min": dur_min,
        "per_motor": per,
        "hottest_motor": {"name": names[i_hot], "end_C": float(end[i_hot]),
                          "rate_C_per_min": float(rate[i_hot])},
        "fastest_heating_motor": {"name": names[i_fast], "end_C": float(end[i_fast]),
                                  "rate_C_per_min": float(rate[i_fast])},
    }


# Below this commanded-PD-torque RMS the delivery ratio is numerically meaningless
# (dividing noise by noise) — report it as undefined instead of a garbage slope.
MIN_CMD_RMS_NM = 0.5


def kp_err_delivery(q, dq, tau_est, target, kp, kd, joint_order=None) -> dict:
    """Audit §4 exp #3 / §6.1/§6.3 — per-joint torque DELIVERY ratio.

    The firmware torque law is tau = kp*(q_des - q) + kd*(0 - dq) [+ tau_ff].
    Regress measured tau_est against the commanded PD torque x = kp*err - kd*dq
    (least squares through the origin): slope ~1.0 => the motor delivers what was
    commanded and tau_est is joint-space consistent; the audit's open question is
    whether the knee delivers only 0.2-0.4x. Also reports R^2 and the commanded
    RMS so a contaminated/unloaded joint can't masquerade as a delivery deficit.
    """
    q = np.atleast_2d(np.asarray(q, float))
    dq = np.atleast_2d(np.asarray(dq, float))
    tau = np.atleast_2d(np.asarray(tau_est, float))
    tgt = np.atleast_2d(np.asarray(target, float))
    kp = np.asarray(kp, float).reshape(-1)
    kd = np.asarray(kd, float).reshape(-1)
    names = list(joint_order) if joint_order is not None else G1_JOINT_ORDER
    x = kp[None, :] * (tgt - q) - kd[None, :] * dq   # commanded PD torque
    out = {}
    for i in range(min(tau.shape[1], len(names))):
        xi, yi = x[:, i], tau[:, i]
        cmd_rms = float(np.sqrt(np.mean(xi ** 2)))
        row = {"cmd_rms_nm": cmd_rms,
               "tau_rms_nm": float(np.sqrt(np.mean(yi ** 2))),
               "n": int(xi.size)}
        sxx = float(np.dot(xi, xi))
        if cmd_rms < MIN_CMD_RMS_NM or sxx <= 0.0:
            row["ratio"] = None
            row["note"] = f"commanded PD torque ~0 (rms {cmd_rms:.2f} Nm) — ratio undefined"
        else:
            ratio = float(np.dot(xi, yi) / sxx)
            resid = yi - ratio * xi
            ss_tot = float(np.sum((yi - yi.mean()) ** 2))
            row["ratio"] = ratio
            row["r2"] = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan")
        out[names[i]] = row
    return out


# ============================ npz analysis + report =============================

def _names_from(d) -> list:
    if "joint_order" in d:
        return [str(x) for x in np.asarray(d["joint_order"]).tolist()]
    return G1_JOINT_ORDER


def _fmt_stats(s: dict) -> str:
    return f"p50 {s['p50']:7.2f}  p95 {s['p95']:7.2f}  max {s['max']:7.2f}"


def _effective_gains(d, run_meta):
    """Gains ACTUALLY commanded during the recorded window. Telemetry stores the
    TRAINED kp/kd, but stand-hold streams at APPROACH_KP_SCALE x and legodom may
    boost the sagittal leg gains — regressing against the stored gains there would
    fabricate a delivery deficit (the exact audit failure mode). Returns
    (kp, kd, notes)."""
    kp = np.asarray(d["kp"], float).copy()
    kd = np.asarray(d["kd"], float).copy()
    rm = run_meta or {}
    notes = []
    mode = rm.get("mode", "")
    if mode == "stand-hold":
        s = float(rm.get("approach_kp_scale", 1.0))
        if s != 1.0:
            kp *= s
            kd *= s
            notes.append(f"stand-hold: gains scaled x{s:g} (approach gains) for the check")
    elif mode == "ground-run-legodom":
        s = float(rm.get("ground_leg_kp_scale", 1.0))
        if s != 1.0:
            idx = [0, 3, 4, 6, 9, 10]   # deploy_runtime.LEG_JOINT_IDX (sagittal boost)
            kp[idx] *= s
            kd[idx] *= s
            notes.append(f"ground-run-legodom: sagittal leg gains scaled x{s:g}")
        s_arm = float(rm.get("arm_ground_kp_scale", 1.0))
        if s_arm != 1.0:
            idx = list(range(15, 29))   # deploy_runtime ARM boost: shoulder/elbow/wrist
            kp[idx] *= s_arm
            kd[idx] *= s_arm
            notes.append(f"ground-run-legodom: ARM gains scaled x{s_arm:g} (kp and kd)")
    if rm.get("gravity_ff"):
        notes.append("WARNING: run used GRAVITY_FF tau feedforward (not recorded) — "
                     "delivery ratios include the unmodeled FF term")
    return kp, kd, notes


def analyze_npz(path, print_fn=print) -> dict:
    """Analyze a recorded npz — a stage-0 capture OR a deploy_runtime Telemetry
    run npz — printing the ANALYSIS SUMMARY and writing <file>.analysis.json.
    Sections apply per available fields; fully offline (no SDK, no robot)."""
    path = Path(path)
    d = dict(np.load(path, allow_pickle=False))
    names = _names_from(d)
    # time base: stage-0 records t_mono (monotonic); Telemetry records t (wall).
    t = np.asarray(d["t_mono"] if "t_mono" in d else d.get("t"), float) \
        if ("t_mono" in d or "t" in d) else None
    summary: dict = {"file": str(path), "n_rows": int(t.size) if t is not None else 0}
    if "run_meta_json" in d:
        try:
            summary["run_meta"] = json.loads(str(np.asarray(d["run_meta_json"]).item()))
        except Exception:
            pass

    print_fn("=" * 74)
    print_fn(f"ANALYSIS SUMMARY — {path.name} ({summary['n_rows']} rows)")
    print_fn("=" * 74)

    # a) obs staleness — needs the raw LowState tick (stage-0 captures only;
    #    Telemetry's 'tick' is the 50 Hz LOOP index, not the robot clock).
    if "lowstate_tick" in d and t is not None:
        st = staleness_stats(t, d["lowstate_tick"])
        summary["staleness"] = st
        print_fn("\n[a] OBS STALENESS / LATENCY (audit §4 exp #4)")
        if "error" in st:
            print_fn(f"  {st['error']}")
        else:
            print_fn(f"  reads: {st['n']} over {st['duration_s']:.1f}s   "
                     f"tick unit est: {st['tick_unit_ms_est']:.4f} ms/count "
                     f"(LowState_.tick, nominally 1 ms)")
            print_fn(f"  wall read interval (ms):   {_fmt_stats(st['wall_read_interval_ms'])}")
            print_fn(f"  msg tick advance   (ms):   {_fmt_stats(st['tick_advance_ms'])}")
            print_fn(f"  staleness vs tick  (ms):   {_fmt_stats(st['staleness_ms'])}")
            print_fn(f"  repeated-tick reads: {st['repeated_tick_fraction']*100:.1f}%   "
                     f"tick-clock drift vs wall: {st['tick_rate_vs_wall_ppm']:+.0f} ppm")
            verdict = ("obs latency BEYOND ~2 ms EXISTS (p95 staleness "
                       f"{st['staleness_ms']['p95']:.2f} ms) — center latency DR on this"
                       if st["latency_beyond_2ms"] else
                       "NO obs latency beyond ~2 ms (p95 staleness "
                       f"{st['staleness_ms']['p95']:.2f} ms) — latency DR stays hygiene-only")
            print_fn(f"  VERDICT: {verdict}")
            print_fn("  (staleness = spread of receive-time vs the robot tick clock, "
                     "min-shifted to 0; it\n   includes this 50 Hz reader's own "
                     "scheduling jitter — the deploy loop has the same.)")

    # b) standing torque baseline
    if "tau_est" in d:
        ts = torque_summary(d["tau_est"], names)
        summary["torque"] = ts
        print_fn("\n[b] STANDING TORQUE BASELINE — tau_est, 12 leg motors "
                 "(audit §4 exp #5, §6.1/§6.4)")
        print_fn(f"  {'joint':<28}{'mean':>8}{'rms':>8}{'p95|.|':>8}  (Nm)")
        for n, s in ts["legs"].items():
            mark = "  <== ankle_pitch" if "ankle_pitch" in n else ""
            print_fn(f"  {n:<28}{s['mean_nm']:>8.2f}{s['rms_nm']:>8.2f}"
                     f"{s['p95_abs_nm']:>8.2f}{mark}")
        if ts["ankle_pitch"]:
            worst = max(abs(s["mean_nm"]) for s in ts["ankle_pitch"].values())
            print_fn(f"  ankle_pitch |mean| worst side: {worst:.2f} Nm  "
                     "(assumed-onboard-baseline claim was ~0; healthy prediction 3-6)")

    # c) thermal
    if "temp" in d and t is not None and t.size >= 2:
        th = thermal_summary(d["temp"], t, names)
        summary["thermal"] = th
        print_fn(f"\n[c] THERMAL — {th['duration_min']:.1f} min (audit §6.5/§6.7)")
        for n in [names[i] for i in LEG_IDX if i < len(names)]:
            s = th["per_motor"].get(n)
            if s:
                print_fn(f"  {n:<28}{s['start_C']:>7.1f} -> {s['end_C']:>5.1f} C   "
                         f"{s['rate_C_per_min']:>+6.2f} C/min")
        hm, fm = th["hottest_motor"], th["fastest_heating_motor"]
        print_fn(f"  HOTTEST: {hm['name']} at {hm['end_C']:.1f} C   "
                 f"FASTEST-HEATING: {fm['name']} at {fm['rate_C_per_min']:+.2f} C/min")

    # d) kp*err identity / delivery ratio — needs a commanded target (motion runs)
    if all(k in d for k in ("q", "dq", "tau_est", "target", "kp", "kd")):
        kp_eff, kd_eff, gain_notes = _effective_gains(d, summary.get("run_meta"))
        dl = kp_err_delivery(d["q"], d["dq"], d["tau_est"], d["target"],
                             kp_eff, kd_eff, names)
        summary["delivery"] = dl
        summary["delivery_gain_notes"] = gain_notes
        if "stage" in d:
            vals, counts = np.unique(np.asarray(d["stage"]), return_counts=True)
            summary["stage_composition"] = {int(v): int(c) for v, c in zip(vals, counts)}
        print_fn("\n[d] KP*ERR IDENTITY / TORQUE DELIVERY RATIO (audit §4 exp #3, §6.1/§6.3)")
        print_fn("  tau_est ~= ratio * [kp*(target-q) - kd*dq];  1.0 = full delivery")
        for note in gain_notes:
            print_fn(f"  NOTE: {note}")
        print_fn(f"  {'joint':<28}{'ratio':>7}{'R^2':>7}{'cmd_rms':>9}{'tau_rms':>9}")
        for n, s in dl.items():
            i = names.index(n)
            mark = "  <== leg" if i in LEG_IDX else ""
            if s["ratio"] is None:
                print_fn(f"  {n:<28}{'--':>7}{'--':>7}{s['cmd_rms_nm']:>9.2f}"
                         f"{s['tau_rms_nm']:>9.2f}  (cmd ~0){mark}")
            else:
                flag = "  <-- DEFICIT?" if (i in LEG_IDX and s["ratio"] < 0.6) else ""
                print_fn(f"  {n:<28}{s['ratio']:>7.2f}{s.get('r2', float('nan')):>7.2f}"
                         f"{s['cmd_rms_nm']:>9.2f}{s['tau_rms_nm']:>9.2f}{mark}{flag}")
        knees = {n: s for n, s in dl.items() if "knee" in n and s["ratio"] is not None}
        if knees:
            kmin = min(s["ratio"] for s in knees.values())
            print_fn(f"  KNEE delivery (audit's 0.2-0.4x question): worst side {kmin:.2f}x")

    # odom presence (stage-0 captures record it when published)
    if "odom_ok" in d:
        ok = np.asarray(d["odom_ok"], bool)
        summary["odom"] = {"fraction_present": float(np.mean(ok)) if ok.size else 0.0}
        print_fn(f"\n[odom] {ODOM_TOPIC}: present on {summary['odom']['fraction_present']*100:.0f}% "
                 "of reads (non-fatal either way)")

    out_path = path.with_suffix(".analysis.json")
    try:
        out_path.write_text(json.dumps(summary, indent=2))
        print_fn(f"\nanalysis JSON written: {out_path}")
    except Exception as e:  # noqa: BLE001 — the printed summary already exists
        print_fn(f"\nWARN: could not write analysis JSON: {e}")
    print_fn("=" * 74)
    return summary


# ================================ capture (robot) ===============================

class Stage0Recorder:
    """Append-only per-tick rows -> npz. Mirrors deploy_runtime.Telemetry's design
    rules: add() never raises and does no disk I/O; save() runs once at exit."""

    KEYS = ("t_wall", "t_mono", "lowstate_tick", "q", "dq", "tau_est", "temp",
            "imu_quat", "gyro", "accel", "odom_pos", "odom_vel", "odom_stamp", "odom_ok")

    def __init__(self, label: str, iface: str, minutes: float):
        self.rows = {k: [] for k in self.KEYS}
        safe = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-") or "capture"
        self.path = TELEMETRY_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_stage0_{safe}.npz"
        self.meta = {
            "tool": "deploy/capture_stage0.py",
            "purpose": ("stage-0 READ-ONLY measurement: obs staleness (audit §4#4), "
                        "standing tau_est baseline (§4#5), thermal rate (§6.5/§6.7)"),
            "label": label, "iface": iface, "minutes": minutes,
            "read_only": True, "sample_hz": SAMPLE_HZ,
            "topics": ["rt/lowstate", f"{ODOM_TOPIC} (optional)"],
            "lowstate_tick_field": ("LowState_.tick (uint32, unitree_hg IDL "
                                    "_LowState_.py:28, nominally 1 ms/count)"),
        }

    def add(self, msg, t_wall, t_mono, odom=None):
        try:
            ms = msg.motor_state
            r = self.rows
            r["t_wall"].append(float(t_wall))
            r["t_mono"].append(float(t_mono))
            r["lowstate_tick"].append(int(msg.tick))
            r["q"].append([float(ms[i].q) for i in range(N_MOTORS)])
            r["dq"].append([float(ms[i].dq) for i in range(N_MOTORS)])
            r["tau_est"].append([float(ms[i].tau_est) for i in range(N_MOTORS)])
            # MotorState_.temperature is int16[2]; channel 0, same as Telemetry.
            r["temp"].append([float(np.atleast_1d(np.asarray(ms[i].temperature,
                                                             dtype=float))[0])
                              for i in range(N_MOTORS)])
            imu = msg.imu_state
            r["imu_quat"].append([float(v) for v in imu.quaternion])   # wxyz
            r["gyro"].append([float(v) for v in imu.gyroscope])
            r["accel"].append([float(v) for v in imu.accelerometer])
            if odom is None:
                r["odom_ok"].append(False)
                r["odom_pos"].append([np.nan] * 3)
                r["odom_vel"].append([np.nan] * 3)
                r["odom_stamp"].append(np.nan)
            else:
                pos, vel, stamp = odom
                r["odom_ok"].append(True)
                r["odom_pos"].append([float(v) for v in pos])
                r["odom_vel"].append([float(v) for v in vel])
                r["odom_stamp"].append(float(stamp))
        except Exception:
            pass  # capture must never disturb / crash mid-session

    def save(self):
        if not self.rows["t_mono"]:
            print("no samples captured — nothing to save.")
            return None
        TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.asarray(v) for k, v in self.rows.items()}
        arrays["joint_order"] = np.array(G1_JOINT_ORDER)
        arrays["run_meta_json"] = np.array(json.dumps(self.meta))
        np.savez_compressed(self.path, **arrays)
        print(f"capture saved: {self.path} ({len(self.rows['t_mono'])} rows)")
        return self.path


def _try_odom_subscriber():
    """Optional rt/odommodestate subscriber — NON-FATAL if absent (audit expects it
    to freeze/vanish outside onboard control; presence is itself a data point)."""
    try:
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
        sub = ChannelSubscriber(ODOM_TOPIC, SportModeState_)
        sub.Init()
        return sub
    except Exception as e:  # noqa: BLE001 — optional topic
        print(f"{ODOM_TOPIC} subscriber unavailable (non-fatal): {e}")
        return None


def _read_odom_once(sub):
    """One non-blocking-ish odom read. NOTE: ChannelSubscriber.Read takes SECONDS
    (deploy_runtime.read_state got this right; read_odom's int(s*1000) is the old
    ms bug — do not copy it). Returns (pos, vel, stamp_s) or None."""
    if sub is None:
        return None
    try:
        msg = sub.Read(0.002)
        if msg is None:
            return None
        st = getattr(msg, "stamp", None)
        stamp = (float(getattr(st, "sec", 0)) + float(getattr(st, "nanosec", 0)) * 1e-9
                 if st else 0.0)
        return (np.array(list(msg.position), float),
                np.array(list(msg.velocity), float), stamp)
    except Exception:  # noqa: BLE001 — optional topic
        return None


def capture(minutes: float, iface: str, label: str) -> int:
    print("=" * 74)
    print("STAGE-0 CAPTURE — STRICTLY READ-ONLY (subscribers only; no publishers,")
    print("no MotionSwitcher, no commands). Safe while the ONBOARD controller stands")
    print(f"the robot. {minutes:g} min at {SAMPLE_HZ:.0f} Hz on {iface}, label '{label}'.")
    print("Ctrl-C stops early — partial data is kept and analyzed.")
    print("=" * 74)
    # Lazy SDK bring-up: reuse deploy_runtime's DDS + subscriber + read_state
    # (drain-to-latest, SECONDS timeout). deploy_runtime imports only numpy at
    # module level, so this stays SDK-free until here.
    from pipeline import deploy_runtime as drt
    try:
        drt.make_dds(iface)
        sub = drt.lowstate_subscriber()
    except ImportError as e:
        raise SystemExit(f"unitree_sdk2py not available ({e}) — run this in the `tv` "
                         "conda env on the robot laptop. (--analyze works anywhere.)")
    odom_sub = _try_odom_subscriber()
    rec = Stage0Recorder(label, iface, minutes)
    n = max(1, int(minutes * 60.0 * SAMPLE_HZ))
    dt = 1.0 / SAMPLE_HZ
    print(f"waiting for first LowState (2 s timeout) ... target {n} samples.")
    code = 0
    try:
        next_t = time.monotonic()
        for i in range(n):
            # read_state: raises SystemExit if no LowState within timeout (NO-GO).
            _q, _dq, _quat, _gyro, msg = drt.read_state(sub, timeout_s=2.0)
            odom = _read_odom_once(odom_sub)
            rec.add(msg, time.time(), time.monotonic(), odom)
            if i % int(10 * SAMPLE_HZ) == 0:   # console heartbeat every ~10 s
                tau = rec.rows["tau_est"][-1]
                tmp = rec.rows["temp"][-1]
                print(f"  [{i/SAMPLE_HZ:5.0f}s] ankle_pitch tau L/R = "
                      f"{tau[ANKLE_PITCH_IDX[0]]:+6.2f}/{tau[ANKLE_PITCH_IDX[1]]:+6.2f} Nm   "
                      f"max temp = {max(tmp):.0f} C "
                      f"({G1_JOINT_ORDER[int(np.argmax(tmp))]})   "
                      f"odom={'yes' if odom else 'no'}", flush=True)
            next_t += dt
            time.sleep(max(0.0, next_t - time.monotonic()))
        print("capture complete.")
    except KeyboardInterrupt:
        print("\nCtrl-C — stopping capture early (partial data kept).")
    except SystemExit as e:
        print(f"\ncapture aborted: {e}")
        code = 2
    finally:
        path = rec.save()
        if path is not None:
            try:
                analyze_npz(path)
            except Exception as e:  # noqa: BLE001 — raw npz is already on disk
                print(f"analysis failed (raw npz kept): {e}")
                code = code or 1
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="READ-ONLY stage-0 robot measurement capture + offline analysis "
                    "(docs/first_principles_audit.md §4/§6).")
    ap.add_argument("--minutes", type=float, default=3.0,
                    help="capture duration (default 3)")
    ap.add_argument("--iface", default=DEFAULT_IFACE,
                    help=f"network interface for DDS (default {DEFAULT_IFACE})")
    ap.add_argument("--label", default="onboard-standby",
                    help="tag baked into the npz filename + run_meta")
    ap.add_argument("--analyze", metavar="NPZ", default=None,
                    help="offline: analyze an existing npz (stage-0 capture or "
                         "deploy_runtime telemetry run). No robot / SDK needed.")
    a = ap.parse_args(argv)
    if a.analyze:
        analyze_npz(Path(a.analyze))
        return 0
    return capture(a.minutes, a.iface, a.label)


if __name__ == "__main__":
    sys.exit(main())
