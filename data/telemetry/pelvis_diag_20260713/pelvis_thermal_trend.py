#!/usr/bin/env python3
"""
pelvis_thermal_trend.py  --  READ-ONLY telemetry analysis for the 2026-07-13 pelvis
fault diagnosis (fishy/burning smell, robot powered off).

Loads every per-run telemetry .npz under data/telemetry/ (the deploy runtime records
per-tick `temp` = per-MOTOR winding/driver temperature for all 29 joints, plus tau_est,
q, dq, IMU quat/gyro; see pipeline/deploy_runtime.py Telemetry).

IMPORTANT MEASUREMENT NOTE: the .npz telemetry contains NO IMU/pelvis-electronics
temperature channel (only imu_quat + gyro). Today's IMU=80 C reading came from a LIVE
rt/lowstate DDS scan, not from this history. So this script can only trend the 29 MOTOR
temperatures. The pelvis-area motors we can trend are the 3 waist motors
(waist_yaw/roll/pitch) and, as proxies for lower-torso load, the 6 hip motors.

Outputs:
  - per-run end/max motor temps + torque, chronological
  - run-over-run creep for the waist + hip joints (is any pelvis motor trending hotter?)
  - sustained-high-torque (bind) detector per joint
  - the 20-min stage0 thermal-soak curve for waist/hip joints (steady-state temp + rate)
All numbers are printed with the source .npz filename so every claim is traceable.
"""
import numpy as np, glob, os, json, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TELDIR = os.path.abspath(os.path.join(HERE, ".."))   # data/telemetry

WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
HIPS  = ["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
         "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint"]
PELVIS_AREA = WAIST + HIPS

def load(f):
    d = np.load(f, allow_pickle=True)
    jo = [str(x) for x in d["joint_order"]]
    temp = np.asarray(d["temp"], float)          # (N,29)
    tau  = np.asarray(d["tau_est"], float)        # (N,29)
    # duration
    if "t" in d.files:
        t = np.asarray(d["t"], float)
    elif "t_wall" in d.files:
        t = np.asarray(d["t_wall"], float)
    else:
        t = np.arange(temp.shape[0], dtype=float) * 0.02
    dur = float(t[-1] - t[0]) if t.size > 1 else 0.0
    meta = {}
    if "run_meta_json" in d.files:
        try: meta = json.loads(str(d["run_meta_json"]))
        except Exception: meta = {}
    return dict(file=os.path.basename(f), jo=jo, temp=temp, tau=tau, dur=dur, n=temp.shape[0], meta=meta)

def col(run, joint):
    return run["jo"].index(joint)

def main():
    files = sorted(glob.glob(os.path.join(TELDIR, "*.npz")))
    runs = []
    for f in files:
        try:
            runs.append(load(f))
        except Exception as e:
            print(f"[skip] {os.path.basename(f)}: {e}")
    print(f"# Loaded {len(runs)} telemetry runs from {TELDIR}\n")

    # ---- 1. Per-run summary (chronological) -------------------------------------
    print("="*118)
    print("1. PER-RUN SUMMARY  (motor temps in C; tau in Nm). end=last-tick temp, max=peak over run.")
    print("   Columns focus on pelvis-area motors: waist_yaw/roll/pitch + hottest hip.")
    print("="*118)
    hdr = f"{'file':42s} {'dur_s':>6s} {'rows':>6s} | {'body_max':>8s} {'body_hot_joint':>16s} | {'wY':>4s} {'wR':>4s} {'wP':>4s} | {'hip_max':>7s}"
    print(hdr); print("-"*118)
    for r in runs:
        endrow = r["temp"][-1]
        maxrow = r["temp"].max(axis=0)
        body_max = maxrow.max(); body_hot = r["jo"][int(maxrow.argmax())].replace("_joint","")
        wy = maxrow[col(r,"waist_yaw_joint")]; wr = maxrow[col(r,"waist_roll_joint")]; wp = maxrow[col(r,"waist_pitch_joint")]
        hipmax = max(maxrow[col(r,j)] for j in HIPS)
        print(f"{r['file']:42s} {r['dur']:6.0f} {r['n']:6d} | {body_max:8.0f} {body_hot:>16s} | {wy:4.0f} {wr:4.0f} {wp:4.0f} | {hipmax:7.0f}")

    # ---- 2. Run-over-run creep for pelvis-area joints ---------------------------
    # Use only the longer "working" runs (dur>=30s) so warmup is comparable.
    print("\n" + "="*118)
    print("2. RUN-OVER-RUN CREEP  (peak temp per run, chronological) for pelvis-area motors.")
    print("   Question: is any waist/hip motor trending hotter run-over-run (a developing bind/bearing issue)?")
    print("   Only runs with dur>=25s included (comparable warmup).")
    print("="*118)
    long_runs = [r for r in runs if r["dur"] >= 25]
    print(f"{'file':42s} {'dur':>4s} | " + " ".join(f"{j.replace('_joint','').replace('left_','L_').replace('right_','R_'):>10s}" for j in PELVIS_AREA))
    print("-"*118)
    series = {j: [] for j in PELVIS_AREA}
    for r in long_runs:
        maxrow = r["temp"].max(axis=0)
        vals = [maxrow[col(r,j)] for j in PELVIS_AREA]
        for j,v in zip(PELVIS_AREA, vals): series[j].append(v)
        print(f"{r['file']:42s} {r['dur']:4.0f} | " + " ".join(f"{v:10.0f}" for v in vals))
    print("-"*118)
    print("TREND (peak temp: first long-run -> last long-run, and linear slope per run):")
    for j in PELVIS_AREA:
        s = np.array(series[j], float)
        if s.size >= 2:
            slope = np.polyfit(np.arange(s.size), s, 1)[0]
            print(f"  {j:26s} first={s[0]:4.0f}C  last={s[-1]:4.0f}C  min={s.min():4.0f}  max={s.max():4.0f}  slope={slope:+5.2f} C/run")

    # ---- 3. Sustained high torque (bind) detector -------------------------------
    print("\n" + "="*118)
    print("3. SUSTAINED-TORQUE / BIND DETECTOR  (per joint, over all runs).")
    print("   A mechanical bind or rubbing shows as a joint holding high |tau| for a large fraction of ticks.")
    print("   Reported: worst run's p95|tau| and %ticks>|5Nm| for each pelvis-area joint; plus body-wide worst.")
    print("="*118)
    # body-wide: which joint had the single highest sustained torque fraction in any run
    worst = []  # (frac, p95, joint, file)
    for r in runs:
        if r["dur"] < 5: continue
        p95 = np.percentile(np.abs(r["tau"]), 95, axis=0)
        frac5 = (np.abs(r["tau"]) > 5.0).mean(axis=0)
        for i,j in enumerate(r["jo"]):
            worst.append((frac5[i], p95[i], j, r["file"]))
    worst.sort(reverse=True)
    print("Top 12 (joint, run) by fraction of ticks holding |tau|>5 Nm (sustained load):")
    for frac,p95,j,f in worst[:12]:
        print(f"  {j:26s} frac>5Nm={frac*100:5.1f}%  p95|tau|={p95:6.1f}Nm   in {f}")
    print("\nPelvis-area joints specifically (worst run each):")
    for j in PELVIS_AREA:
        cand = [w for w in worst if w[2]==j]
        if cand:
            frac,p95,_,f = max(cand)
            print(f"  {j:26s} worst frac>5Nm={frac*100:5.1f}%  p95|tau|={p95:6.1f}Nm   in {f}")

    # ---- 4. 20-min thermal soak curve (stage0) ----------------------------------
    print("\n" + "="*118)
    print("4. LONG THERMAL SOAK (stage0 post-session-watch, ~20 min) — waist/hip steady-state + rise rate.")
    print("   This is the best available proxy for how hot the LOWER TORSO gets under sustained standing load.")
    print("="*118)
    soak = [r for r in runs if "stage0" in r["file"] and r["n"] >= 9000]
    for r in sorted(soak, key=lambda x: -x["n"]):
        print(f"\n-- {r['file']}  (rows={r['n']}, dur={r['dur']:.0f}s, purpose={r['meta'].get('purpose','?')[:60]})")
        for j in ["waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
                  "left_hip_roll_joint","right_hip_roll_joint","left_ankle_pitch_joint"]:
            c = col(r,j); s = r["temp"][:,c]
            start = s[:50].mean(); end = s[-50:].mean(); peak = s.max()
            rate = (end-start)/(r["dur"]/60.0) if r["dur"]>0 else 0.0
            print(f"    {j:26s} start={start:4.0f}C  end={end:4.0f}C  peak={peak:4.0f}C  rise={end-start:+4.0f}C  rate={rate:+5.1f}C/min")

    # ---- 5. Body-wide hottest-motor-ever + today's context ----------------------
    print("\n" + "="*118)
    print("5. BODY-WIDE HOTTEST MOTOR EVER RECORDED (across all runs) + context vs today.")
    print("="*118)
    allmax = []
    for r in runs:
        mr = r["temp"].max(axis=0)
        i = int(mr.argmax())
        allmax.append((mr[i], r["jo"][i], r["file"]))
    allmax.sort(reverse=True)
    print("Top 10 hottest single-motor peaks in history:")
    for v,j,f in allmax[:10]:
        print(f"  {v:4.0f}C  {j:26s}  {f}")
    print("\nNOTE: none of these is the IMU/pelvis electronics — the .npz has no IMU-temp channel.")
    print("Today's LIVE scan: hottest MOTOR = left_ankle_pitch 64C; IMU(pelvis) = 80C (hotter than any motor).")

if __name__ == "__main__":
    main()
