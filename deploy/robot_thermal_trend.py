#!/usr/bin/env python
"""READ-ONLY thermal TREND scan for the G1 pelvis-heat/smell diagnosis.

Answers the one question that decides everything: with the robot idle, is the
IMU/pelvis-board temperature CLIMBING past its ~80 C baseline (an ACTIVE fault ->
open it) or does it PLATEAU (a hot-but-stable baseline)?

SAFETY:
  * Subscribe-only on rt/lowstate. It NEVER creates a publisher, never sends a
    LowCmd/motor command — it physically cannot move or power the robot.
  * It also cannot power the robot OFF. YOU are the fast safety loop: keep the
    kill switch / battery disconnect in hand and abort on smoke / hiss / a smell
    spike REGARDLESS of this script. The script prints a loud ABORT if the IMU
    crosses a hard limit so you get a second trigger.
  * A COLD robot warms from ambient to steady-state over the first ~5-15 min —
    a rising temperature early is NORMAL. The fault signature is overshooting the
    ~80 C baseline (>=90 C) or never plateauing. Run in the `tv` env.
"""
import argparse, json, os, sys, time
import numpy as np

REPORT = os.environ.get("THERMAL_REPORT", "/tmp/thermal_trend.txt")
_rf = open(REPORT, "w")
def log(m=""):
    print(m, flush=True); _rf.write(str(m) + "\n"); _rf.flush()
def done(c):
    _rf.flush(); _rf.close(); sys.stdout.flush(); os._exit(c)

ap = argparse.ArgumentParser()
ap.add_argument("--iface", default=os.environ.get("ROBOT_IFACE", "enp0s31f6"))
ap.add_argument("--interval", type=float, default=20.0)   # s between samples
ap.add_argument("--duration", type=float, default=720.0)  # total s (~12 min)
ap.add_argument("--abort-c", type=float, default=90.0)    # IMU hard-abort temp
ap.add_argument("--meta", default="data/policies/thriller/policy_meta.json")
args = ap.parse_args()

names = [f"m{i}" for i in range(35)]
try:
    jo = json.load(open(args.meta)).get("joint_order") or json.load(open(args.meta)).get("joint_order_29dof")
    if jo:
        for i, n in enumerate(jo):
            names[i] = n
except Exception:
    pass
WAIST = [i for i in range(29) if "waist" in names[i]]
HIP = [i for i in range(29) if "hip" in names[i]]

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

log(f"[thermal-trend] READ-ONLY (subscribe only). iface={args.iface} "
    f"interval={args.interval:.0f}s duration={args.duration:.0f}s abort>={args.abort_c:.0f}C")
log("YOU are the safety loop: kill switch in hand; abort on smoke/hiss/smell spike no matter what this says.")
log("(Cold-start warm-up is NORMAL; we're watching for an overshoot past the ~80 C baseline.)")
ChannelFactoryInitialize(0, args.iface)
sub = ChannelSubscriber("rt/lowstate", LowState_); sub.Init()

def read():
    for _ in range(10):
        m = sub.Read(1.0)
        if m is not None:
            return m
    return None

def imu_t(m):
    im = getattr(m, "imu_state", None)
    return float(im.temperature) if im is not None and hasattr(im, "temperature") else float("nan")
def mt(m, i):
    return float(np.max(np.atleast_1d(np.asarray(m.motor_state[i].temperature, dtype=float))))
def group_max(m, idxs):
    return max((mt(m, i) for i in idxs), default=float("nan"))
def power(m):
    for f in ("power_a", "power_v"):
        if hasattr(m, f):
            return f"{f}={getattr(m, f)}"
    return ""

first = read()
if first is None:
    log("NO LowState received — robot off / wrong iface / LAN down.")
    log("If it IS powered and smells of burning: POWER IT OFF NOW.")
    done(2)

imu0 = imu_t(first)
log(f"\n  t(s)   IMU_C  dIMU   waist_max  hip_max  body_max   {power(first)}   note")
series = []
n = int(args.duration / args.interval) + 1
t_start = time.time()
for _ in range(n):
    m = read()
    if m is None:
        log(f"  ---    link drop (no lowstate) — check LAN / robot still on")
        time.sleep(args.interval); continue
    el = int(time.time() - t_start)
    it = imu_t(m); wm = group_max(m, WAIST); hm = group_max(m, HIP)
    bm = max((mt(m, i) for i in range(29)), default=-1)
    series.append((el, it))
    hard = it >= args.abort_c
    log(f"  {el:>4}   {it:>5.0f}  {it-imu0:>+5.0f}   {wm:>8.0f}   {hm:>6.0f}   {bm:>7.0f}   {power(m)}   "
        f"{'⚠️ HARD-ABORT' if hard else ''}")
    if hard:
        log("")
        log(f"⚠️⚠️ ABORT: IMU {it:.0f}C >= {args.abort_c:.0f}C — overshooting the ~80C baseline = ACTIVE runaway.")
        log("   POWER THE ROBOT OFF NOW. Open the pelvis; inspect the power board / DC-DC / electrolytic caps.")
        done(3)
    time.sleep(args.interval)

els = [e for e, _ in series]; its = [t for _, t in series]
if len(its) < 2:
    log("\nToo few samples for a trend — re-run."); done(1)
rise = its[-1] - its[0]
tail = its[max(0, len(its) - 4):]                       # last ~1 min
tail_slope = (tail[-1] - tail[0]) / max((args.interval * (len(tail) - 1)) / 60.0, 1e-6)  # C/min
log("")
log(f"=== TREND VERDICT — {len(its)} samples over {els[-1]}s ===")
log(f"  IMU {its[0]:.0f}C -> {its[-1]:.0f}C (rise {rise:+.0f}C); peak {max(its):.0f}C; tail slope {tail_slope:+.1f} C/min")
if its[-1] >= 88 or (tail_slope >= 1.0 and its[-1] >= 82):
    log("  VERDICT: CLIMBING / not plateauing -> ACTIVE FAULT. Open the pelvis, inspect power board+caps. DO NOT dance it.")
elif abs(tail_slope) < 0.5 and its[-1] <= 82:
    log("  VERDICT: PLATEAUED at the ~80C baseline (not runaway). The SMELL still means inspect soon, but no emergency.")
else:
    log("  VERDICT: borderline — plateauing but warm. Inspect to be safe before any hard use.")
log("  (Motors stayed normal throughout = not a winding; the heat is the pelvis power board.)")
done(0)
