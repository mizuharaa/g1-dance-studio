#!/usr/bin/env python
"""READ-ONLY full-body thermal/fault scan of the Unitree G1 over DDS.

SAFETY: ONLY subscribes to rt/lowstate. It never creates a publisher and never sends a
LowCmd or any motor command — it cannot move the robot. Run in the `tv` env.
Writes the report to REPORT (below) AND stdout, then os._exit to avoid the DDS teardown hang.
"""
import argparse, json, os, sys
import numpy as np

REPORT = os.environ.get("THERMAL_REPORT", "/tmp/thermal_report.txt")
_rf = open(REPORT, "w")
def log(msg=""):
    print(msg, flush=True)
    _rf.write(str(msg) + "\n"); _rf.flush()
def done(code):
    _rf.flush(); _rf.close(); sys.stdout.flush(); os._exit(code)

ap = argparse.ArgumentParser()
ap.add_argument("--iface", default=os.environ.get("ROBOT_IFACE", "enp0s31f6"))
ap.add_argument("--samples", type=int, default=15)
ap.add_argument("--meta", default="data/policies/thriller/policy_meta.json")
args = ap.parse_args()

names = [f"motor_{i}" for i in range(35)]
try:
    m = json.load(open(args.meta))
    jo = m.get("joint_order") or m.get("joint_order_29dof")
    if jo:
        for i, n in enumerate(jo):
            names[i] = n
except Exception:
    pass

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

log(f"[thermal-diag] READ-ONLY (subscribe only). iface={args.iface}. Reading rt/lowstate ...")
ChannelFactoryInitialize(0, args.iface)
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init()

last, maxtemp, got = None, {}, 0
for _ in range(args.samples):
    msg = sub.Read(1.0)
    if msg is None:
        continue
    last, got = msg, got + 1
    for i in range(35):
        try:
            t = np.atleast_1d(np.asarray(msg.motor_state[i].temperature, dtype=float))
            maxtemp[i] = max(maxtemp.get(i, -999.0), float(np.max(t)))
        except Exception:
            pass

if last is None:
    log("NO LowState received in the window — robot off / wrong iface / LAN down.")
    log("If it IS powered and smells of burning: POWER IT OFF NOW.")
    done(2)

log(f"[thermal-diag] {got}/{args.samples} samples. mode_machine={getattr(last,'mode_machine','?')} tick={getattr(last,'tick','?')}")
for f in ("power_v", "power_a", "bms_state", "bms"):
    if hasattr(last, f):
        log(f"  {f}: {getattr(last, f)}")
imu = getattr(last, "imu_state", None)
if imu is not None and hasattr(imu, "temperature"):
    log(f"  IMU temp: {imu.temperature} C")
log("")
log(f"{'idx':>3} {'joint':<24}{'tC0':>5}{'tC1':>5}{'maxC':>6}{'tau_Nm':>8}{'q_deg':>8}{'dq':>7}{'lost':>6}  flag")
hot = []
for i in range(29):
    ms = last.motor_state[i]
    t = np.atleast_1d(np.asarray(ms.temperature, dtype=float))
    t0, t1 = float(t[0]), float(t[-1])
    tmax = max(t0, t1, maxtemp.get(i, -999.0))
    tau, q, dq = float(ms.tau_est), float(ms.q), float(ms.dq)
    lost = getattr(ms, "lost", "?")
    flag = ""
    if tmax >= 90: flag = "CRITICAL-HOT"; hot.append((i, tmax))
    elif tmax >= 75: flag = "HOT"; hot.append((i, tmax))
    elif tmax >= 60: flag = "warm"
    if abs(tau) > 30: flag += " HIGH-TORQUE"
    log(f"{i:>3} {names[i]:<24}{t0:>5.0f}{t1:>5.0f}{tmax:>6.0f}{tau:>8.2f}{np.degrees(q):>8.1f}{dq:>7.2f}{str(lost):>6}  {flag}")
log("")
allmax = max(maxtemp.values()) if maxtemp else -1
if hot:
    log("HOT MOTORS (>=75C): " + ", ".join(f"{names[i]}={t:.0f}C" for i, t in sorted(hot, key=lambda x: -x[1])))
    log("Winding temps near/over thermal limit -> likely source of the smell. POWER OFF, let cool,")
    log("check the hot joint for a mechanical bind or a stuck/fighting pose (high tau).")
else:
    log(f"No motor >=75C (hottest {allmax:.0f}C).")
    log("If you still smell burning with no hot motor, suspect electronics/connector/battery,")
    log("not a winding -> power off and inspect PCBs / battery / wiring rather than the joints.")
done(0)
