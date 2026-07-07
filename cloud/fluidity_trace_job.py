#!/usr/bin/env python
"""Per-variant fluidity trace job (decision-table numbers).

argv[1] = v3a | v3b | v3c | v3d | v4 | s2rb

For a variant: waits until its autopilot has written
exports/thriller_<tail>/RESULT.txt, parses the winning checkpoint + eval task,
rolls out a 1-env nominal full-motion trace (cloud/sim_trace_dump.py), runs
cloud/fluidity_sim_metrics.py, writes exports/thriller_<tail>/fluidity.json
and APPENDS the headline numbers to RESULT.txt.

For s2rb: runs immediately on the promoted checkpoint (model_4999, stock task,
old reference) -> reports/fluidity_s2rb_baseline.json — the baseline row.

Run:  bash cloud/run_job.sh start <tail>-fluidity -- \
  "cd /workspace/notebook-data && MUJOCO_GL=egl ./envs/mjlab/bin/python cloud/fluidity_trace_job.py <tail>"
"""
import json
import os
import re
import subprocess
import sys
import time

NB = "/workspace/notebook-data"
PY = f"{NB}/envs/mjlab/bin/python"
TAIL = sys.argv[1] if len(sys.argv) > 1 else "s2rb"

OLD_MOTION = f"{NB}/motions/thriller_deploy.npz"
SHARP_MOTION = f"{NB}/motions/thriller_deploy_v2_sharp.npz"
STOCK_TASK = "Mjlab-Tracking-Flat-Unitree-G1"
S2RB_CKPT = (f"{NB}/logs/rsl_rl/g1_tracking/"
             "2026-07-04_17-04-58_train-thriller-s2r-b/model_4999.pt")


def log(m):
    print(m, flush=True)


def run(args, step):
    env = dict(os.environ, MUJOCO_GL="egl")
    r = subprocess.run(args, env=env, capture_output=True, text=True)
    log(f"[{step}] rc={r.returncode}\n" + r.stdout[-3000:] + r.stderr[-1500:])
    return r.returncode == 0, r.stdout


def main():
    if TAIL == "s2rb":
        trace = f"{NB}/reports/sim_trace_s2rb.npz"
        out_json = f"{NB}/reports/fluidity_s2rb_baseline.json"
        ok, _ = run([PY, f"{NB}/cloud/sim_trace_dump.py", "--checkpoint", S2RB_CKPT,
                     "--motion-file", OLD_MOTION, "--task", STOCK_TASK,
                     "--out", trace], "trace")
        if not ok:
            sys.exit(1)
        ok, _ = run([PY, f"{NB}/cloud/fluidity_sim_metrics.py", trace, out_json],
                    "metrics")
        sys.exit(0 if ok else 1)

    # v4's autopilot exported under thriller_v34 (patched-variant naming)
    dir_tail = {"v4": "v34"}.get(TAIL, TAIL)
    out_dir = f"{NB}/exports/thriller_{dir_tail}"
    result = f"{out_dir}/RESULT.txt"
    motion = SHARP_MOTION if TAIL in ("v3d", "v3e", "v4") else OLD_MOTION

    log(f"waiting for {result} ...")
    while not os.path.exists(result):
        time.sleep(300)
    txt = open(result).read()
    m = re.search(r"^checkpoint=(.+)$", txt, re.M)
    if not m:
        log(f"no checkpoint in RESULT.txt (verdict line: {txt.splitlines()[0]}) — nothing to trace")
        sys.exit(1)
    ckpt = m.group(1).strip()
    mt = re.search(r"eval_task=(\S+)", txt)
    task = mt.group(1).strip() if mt else STOCK_TASK

    trace = f"{out_dir}/sim_trace.npz"
    out_json = f"{out_dir}/fluidity.json"
    ok, _ = run([PY, f"{NB}/cloud/sim_trace_dump.py", "--checkpoint", ckpt,
                 "--motion-file", motion, "--task", task, "--out", trace], "trace")
    if not ok:
        sys.exit(1)
    ok, stdout = run([PY, f"{NB}/cloud/fluidity_sim_metrics.py", trace, out_json],
                     "metrics")
    if not ok:
        sys.exit(1)

    # append the headline to RESULT.txt (baseline for comparison if present)
    base_line = ""
    try:
        b = json.load(open(f"{NB}/reports/fluidity_s2rb_baseline.json"))
        base_line = (f" (s2r-b baseline {b['action_band_rms']['wobble_2-10Hz']['legs']:.4f}"
                     f" / amp {b['leg_vs_ref']['amp_ratio_mean']:.2f})")
    except Exception:  # noqa: BLE001
        pass
    head = [ln for ln in stdout.splitlines() if ln.startswith("FLUIDITY_LEG_BAND=")]
    with open(result, "a") as f:
        f.write(f"\nfluidity={out_json}\n")
        if head:
            f.write(head[0] + base_line + "\n")
        f.write("fluidity bars: leg 2-10Hz band <= 0.20 (lower better); "
                "leg amp ratio > 0.5 preferred\n")
    log("appended fluidity headline to RESULT.txt")


if __name__ == "__main__":
    main()
