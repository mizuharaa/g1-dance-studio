#!/usr/bin/env python
"""Autopilot for the acro (dynamic-skills) training: train-acro-1.

When the training job finishes: export ONNX, run the acro landing eval
(cloud/acro_eval.py: survival + rotation-completed + upright-at-end, plus the
peak-torque/velocity/impact numbers for the hardware-risk memo), render the
rollout mp4 on the acro task, and write exports/acro1/RESULT.txt.

If the final checkpoint does not land, the mid checkpoint is also evaluated
(attempt-2 lesson). Non-destructive: nothing is staged into deploy dirs; acro
artifacts never enter the show pipeline (see cloud/dynamic_skills_task.py
vet-gate note).

Run on the box:  bash cloud/run_job.sh start acro-autopilot -- \
  "cd /workspace/notebook-data && ./envs/mjlab/bin/python cloud/autopilot_acro.py train-acro-1 motions/<acro>.npz"
"""
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

NB = "/workspace/notebook-data"
PY = f"{NB}/envs/mjlab/bin/python"
JOB_NAME = sys.argv[1] if len(sys.argv) > 1 else "train-acro-1"
MOTION = sys.argv[2] if len(sys.argv) > 2 else f"{NB}/motions/acro_backflip.npz"
if not os.path.isabs(MOTION):
    MOTION = f"{NB}/{MOTION}"
JOB_STATUS = f"{NB}/jobs/{JOB_NAME}.status.json"
OUT = f"{NB}/exports/acro1"
TASK = "Mjlab-Tracking-Flat-Unitree-G1-Acro"
LAND_BAR = 0.90  # landed_rate for VERDICT=LANDED (64 randomized envs)


def log(msg):
    print(msg, flush=True)


def write_result(lines):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/RESULT.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))


def run(args, tag, step):
    env = dict(os.environ, MUJOCO_GL="egl", WANDB_MODE="disabled")
    r = subprocess.run(args, env=env, capture_output=True, text=True)
    log(f"[{step} {tag}] rc={r.returncode}\n" + r.stdout[-4000:] + r.stderr[-2000:])
    return r.returncode == 0


def motion_frames():
    d = np.load(MOTION)
    return int(d["joint_pos"].shape[0])


def evaluate(ckpt, tag):
    d = f"{OUT}/{tag}"
    os.makedirs(d, exist_ok=True)
    if not run([PY, f"{NB}/cloud/export_policy.py", ckpt, MOTION, d], tag, "export"):
        return None
    run([PY, f"{NB}/cloud/acro_eval.py",
         "--checkpoint", ckpt, "--motion-file", MOTION,
         "--task", TASK, "--num-envs", "64",
         "--output-file", f"{d}/acro_eval.json"], tag, "acro_eval")
    try:
        res = json.load(open(f"{d}/acro_eval.json"))
    except Exception as e:  # noqa: BLE001
        log(f"[{tag}] no acro_eval.json: {e}")
        return None
    res["_ckpt"], res["_tag"] = ckpt, tag
    return res


def landed_rate(res):
    try:
        return float(res["success"]["landed_rate"])
    except (KeyError, TypeError):
        return 0.0


def summarize(res):
    s = res.get("success", {})
    ref = res.get("reference", {})
    tq = res.get("peak_torque_nm_by_group", {})
    knee = tq.get("knee", {})
    ankle = tq.get("ankle", {})
    hip = tq.get("hip", {})
    imp = res.get("landing_impact", {})
    return (f"tag={res['_tag']} landed={s.get('landed')}/{res.get('num_envs')} "
            f"(survived={s.get('survived')} rot={s.get('rotation_ok')} "
            f"upright={s.get('upright_ok')}) "
            f"ref_rot={ref.get('total_rotation_rev')}rev flight={ref.get('flight_s')}s "
            f"peak_tau hip={hip.get('max')}/{hip.get('rating')} "
            f"knee={knee.get('max')}/{knee.get('rating')} "
            f"ankle={ankle.get('max')}/{ankle.get('rating')}Nm "
            f"impact={imp.get('peak_base_decel_m_s2')}m/s2")


def main():
    log(f"autopilot_acro: waiting for {JOB_STATUS} ...")
    while True:
        try:
            st = json.load(open(JOB_STATUS))["state"]
        except Exception:  # noqa: BLE001
            st = "missing"
        if st == "done":
            break
        if st == "failed":
            write_result(["VERDICT=TRAIN_FAILED", f"see {NB}/jobs/{JOB_NAME}.log"])
            sys.exit(1)
        time.sleep(120)
    log("training done — evaluating checkpoints")

    runs = sorted(glob.glob(f"{NB}/logs/rsl_rl/g1_tracking/*{JOB_NAME}*")
                  + glob.glob(f"{NB}/cloud/logs/rsl_rl/g1_tracking/*{JOB_NAME}*"),
                  key=os.path.getmtime)
    if not runs:
        write_result(["VERDICT=NO_RUN_DIR"])
        sys.exit(1)
    run_dir = runs[-1]
    ckpts = sorted(glob.glob(f"{run_dir}/model_*.pt"),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if not ckpts:
        write_result(["VERDICT=NO_CHECKPOINTS", f"run={run_dir}"])
        sys.exit(1)

    results = []
    res_last = evaluate(ckpts[-1], "last")
    if res_last:
        results.append(res_last)
    if landed_rate(res_last) < LAND_BAR and len(ckpts) > 2:
        res_mid = evaluate(ckpts[len(ckpts) // 2], "mid")
        if res_mid:
            results.append(res_mid)
    if not results:
        write_result(["VERDICT=EVAL_FAILED", f"run={run_dir}"])
        sys.exit(1)

    best = max(results, key=landed_rate)
    steps = motion_frames() - 2
    render_ok = run([PY, f"{NB}/cloud/headless_render_acro.py", best["_ckpt"], MOTION,
                     f"{OUT}/rollout_acro1.mp4", str(steps), TASK],
                    best["_tag"], "render")

    lr = landed_rate(best)
    verdict = ("LANDED" if lr >= LAND_BAR else
               "PARTIAL" if lr > 0 else "FAILED")
    lines = [
        f"VERDICT={verdict} landed_rate={lr:.3f} (bar {LAND_BAR})",
        f"task={TASK} motion={MOTION}",
        f"checkpoint={best['_ckpt']}",
        f"onnx={OUT}/{best['_tag']}/policy.onnx",
        f"acro_eval={OUT}/{best['_tag']}/acro_eval.json",
        f"render={OUT}/rollout_acro1.mp4 ({'ok' if render_ok else 'FAILED'})",
    ] + [summarize(x) for x in results] + [
        "",
        "SIM-ONLY RESULT. No hardware use without the separate human decision",
        "documented in docs/DYNAMIC_SKILLS.md (hardware-risk memo) — this policy",
        "was trained WITHOUT pushes and WITHOUT torque/thermal penalties.",
    ]
    write_result(lines)


if __name__ == "__main__":
    main()
