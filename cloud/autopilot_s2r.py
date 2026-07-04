"""Autopilot for the sim2real retrain (train-thriller-s2r).

Waits for the training job, exports ONNX, runs the full-motion sim_gap_check
(v2 gates: survival nominal/worst, ankle mean/p95/RMS, mpkpe), and writes a
RESULT.txt verdict. If the final checkpoint fails the gate, the mid checkpoint
is also evaluated (attempt-2 lesson: mid checkpoints can beat the final).

Non-destructive: everything lands under exports/thriller_s2r/; nothing is
staged into deploy dirs — staging + visual render happen laptop-side after a
human-visible verdict.

Run on the box:  bash cloud/run_job.sh start s2r-autopilot -- \
  "cd /workspace/notebook-data && ./envs/mjlab/bin/python cloud/autopilot_s2r.py"
"""
import glob
import json
import os
import subprocess
import sys
import time

import sys as _sys

NB = "/workspace/notebook-data"
PY = f"{NB}/envs/mjlab/bin/python"
# argv[1] = train job name (default train-thriller-s2r); OUT dir derives from it.
JOB_NAME = _sys.argv[1] if len(_sys.argv) > 1 else "train-thriller-s2r"
JOB_STATUS = f"{NB}/jobs/{JOB_NAME}.status.json"
OUT = f"{NB}/exports/{JOB_NAME.replace('train-thriller-', 'thriller_').replace('-', '_')}"
MOTION = f"{NB}/motions/thriller_deploy.npz"


def log(msg):
    print(msg, flush=True)


def write_result(lines):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/RESULT.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))


def evaluate(ckpt, tag):
    d = f"{OUT}/{tag}"
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ, MUJOCO_GL="egl")
    r = subprocess.run([PY, f"{NB}/cloud/export_policy.py", ckpt, MOTION, d],
                       env=env, capture_output=True, text=True)
    log(f"[export {tag}] rc={r.returncode}\n" + r.stdout[-1500:] + r.stderr[-1500:])
    if r.returncode != 0:
        return None
    r = subprocess.run([PY, f"{NB}/cloud/sim_gap_check.py",
                        "--checkpoint", ckpt, "--motion-file", MOTION,
                        "--num-envs", "128",
                        "--output-file", f"{d}/gap_check.json"],
                       env=env, capture_output=True, text=True)
    log(f"[gap_check {tag}] rc={r.returncode}\n" + r.stdout[-6000:] + r.stderr[-2000:])
    try:
        res = json.load(open(f"{d}/gap_check.json"))
        res["_ckpt"], res["_tag"] = ckpt, tag
        return res
    except Exception as e:  # noqa: BLE001
        log(f"[gap_check {tag}] could not read json: {e}")
        return None


def worst_survival(res):
    conds = res.get("conditions", {})
    names = [n for n in ("delay40ms_push", "delay20ms_push", "delay40ms") if n in conds]
    if not names:
        return 0.0
    return min(conds[n]["success_rate"] for n in names)


def summarize(res):
    nom = res["conditions"].get("nominal", {})
    ap = nom.get("ankle_pitch", {})
    return (f"tag={res['_tag']} gate={'PASS' if (res.get('gate') or {}).get('pass') else 'FAIL'} "
            f"nominal_survival={nom.get('success_rate'):.3f} mpkpe={nom.get('mpkpe_m'):.3f} "
            f"ankle mean={ap.get('mean_abs'):.2f} rms={ap.get('rms_abs'):.2f} Nm "
            f"worst_survival={worst_survival(res):.3f}")


def main():
    log(f"waiting for training job ({JOB_STATUS}) ...")
    while True:
        try:
            st = json.load(open(JOB_STATUS))["state"]
        except Exception:  # noqa: BLE001
            st = "missing"
        if st == "done":
            break
        if st == "failed":
            write_result(["VERDICT=TRAIN_FAILED",
                          f"see {NB}/jobs/{JOB_NAME}.log"])
            sys.exit(1)
        time.sleep(60)
    log("training done — evaluating checkpoints")

    # train.py resolves its log root from the invocation context: a1/a2 landed under
    # cloud/logs/, the s2r wrapper lands under logs/ — search both.
    runs = sorted(glob.glob(f"{NB}/logs/rsl_rl/g1_tracking/*{JOB_NAME}*")
                  + glob.glob(f"{NB}/cloud/logs/rsl_rl/g1_tracking/*{JOB_NAME}*"),
                  key=os.path.getmtime)
    if not runs:
        write_result(["VERDICT=NO_RUN_DIR"])
        sys.exit(1)
    run = runs[-1]
    ckpts = sorted(glob.glob(f"{run}/model_*.pt"),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if not ckpts:
        write_result(["VERDICT=NO_CHECKPOINTS", f"run={run}"])
        sys.exit(1)

    results = []
    res_last = evaluate(ckpts[-1], "last")
    if res_last:
        results.append(res_last)
    if not (res_last and (res_last.get("gate") or {}).get("pass")) and len(ckpts) > 2:
        res_mid = evaluate(ckpts[len(ckpts) // 2], "mid")
        if res_mid:
            results.append(res_mid)

    if not results:
        write_result(["VERDICT=EVAL_FAILED", f"run={run}"])
        sys.exit(1)

    best = max(results, key=lambda r: (bool((r.get("gate") or {}).get("pass")),
                                       worst_survival(r)))
    gate_pass = bool((best.get("gate") or {}).get("pass"))
    lines = [
        f"VERDICT={'GATE_PASS' if gate_pass else 'GATE_FAIL'}",
        f"checkpoint={best['_ckpt']}",
        f"onnx={OUT}/{best['_tag']}/policy.onnx",
        f"gap_check={OUT}/{best['_tag']}/gap_check.json",
    ] + [summarize(r) for r in results] + [
        "",
        "NEXT (laptop): if GATE_PASS -> pull onnx + gap_check.json, stage under",
        "data/policies/thriller_s2r/ (reuse policy_meta.json — same gains/scales/obs),",
        "render a rollout video for visual sign-off, then plan ONE tethered HW test",
        "(human present, damping remote in hand). If GATE_FAIL -> read per-section",
        "stats in gap_check.json; the failing sections drive the attempt-2 delta",
        "(targeted choreography edit or reward re-weighting).",
    ]
    write_result(lines)


if __name__ == "__main__":
    main()
