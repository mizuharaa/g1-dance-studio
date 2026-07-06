"""Autopilot for the v3 dance-quality retrains (train-thriller-v3{a,b,c}).

Per variant, when its training job finishes: export ONNX, run the sim gap
check (gate v3, stock harness — V3B on the V3B-GAPEVAL task so it is evaluated
on the arm plant the deploy contract gives it), run the ARM joint-space
tracking eval (the dance-quality metric: arm-group RMS vs reference must BEAT
the s2r-b sim baseline in reports/arm_tracking_s2rb_baseline.json), render a
rollout mp4, and write exports/thriller_v3<x>/RESULT.txt.

If the final checkpoint fails the gate, the mid checkpoint is also evaluated
(attempt-2 lesson). Non-destructive: nothing is staged into deploy dirs.

Run on the box:  bash cloud/run_job.sh start v3a-autopilot -- \
  "cd /workspace/notebook-data && ./envs/mjlab/bin/python cloud/autopilot_v3.py train-thriller-v3a"
"""
import glob
import json
import os
import subprocess
import sys
import time

NB = "/workspace/notebook-data"
PY = f"{NB}/envs/mjlab/bin/python"
JOB_NAME = sys.argv[1] if len(sys.argv) > 1 else "train-thriller-v3a"
VARIANT = JOB_NAME.rsplit("v3", 1)[-1][:1]  # "a" | "b" | "c" | "d"
JOB_STATUS = f"{NB}/jobs/{JOB_NAME}.status.json"
OUT = f"{NB}/exports/thriller_v3{VARIANT}"
# v3d trains/tracks the SHARP reference (retarget-fidelity fix: prep_motion's
# blanket 8.48 rad/s velocity clamp blunted 58 dance accents; docs/
# retarget_fidelity.md) — its gate, arm metric and render MUST use that npz,
# and its s2r-b baseline is recomputed against the sharp reference too.
if VARIANT in ("d", "e"):  # v3e = v3c recipe (10k) x SHARP reference follow-up
    MOTION = f"{NB}/motions/thriller_deploy_v2_sharp.npz"
    BASELINE = f"{NB}/reports/arm_tracking_s2rb_baseline_sharp.json"
else:
    MOTION = f"{NB}/motions/thriller_deploy.npz"
    BASELINE = f"{NB}/reports/arm_tracking_s2rb_baseline.json"

STOCK_TASK = "Mjlab-Tracking-Flat-Unitree-G1"
GAPEVAL_TASK = "Mjlab-Tracking-Flat-Unitree-G1-S2R-V3B-GAPEVAL"
EVAL_TASK = GAPEVAL_TASK if VARIANT == "b" else STOCK_TASK
RENDER_STEPS = 2650  # full 51.8 s dance + tail

V3B_DEPLOY_NOTE = (
    "DEPLOY CONTRACT (V3B): trained with arm actuator kp/kd x2.5. Deploy ONLY with "
    "ARM_GROUND_KP_SCALE=2.5 (pipeline/deploy_runtime.py ground-run-legodom) or a "
    "policy_meta.json with the 14 arm joints' kp/kd pre-multiplied by 2.5. "
    "At unscaled gains this policy reproduces the soft-arm gap it was trained to avoid.")


def log(msg):
    print(msg, flush=True)


def write_result(lines):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/RESULT.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))


def run(args, tag, step):
    env = dict(os.environ, MUJOCO_GL="egl")
    r = subprocess.run(args, env=env, capture_output=True, text=True)
    log(f"[{step} {tag}] rc={r.returncode}\n" + r.stdout[-4000:] + r.stderr[-2000:])
    return r.returncode == 0


def evaluate(ckpt, tag):
    d = f"{OUT}/{tag}"
    os.makedirs(d, exist_ok=True)
    if not run([PY, f"{NB}/cloud/export_policy.py", ckpt, MOTION, d], tag, "export"):
        return None
    run([PY, f"{NB}/cloud/sim_gap_check_v3.py",
         "--checkpoint", ckpt, "--motion-file", MOTION,
         "--task", EVAL_TASK, "--num-envs", "128",
         "--output-file", f"{d}/gap_check.json"], tag, "gap_check")
    run([PY, f"{NB}/cloud/arm_tracking_eval.py",
         "--checkpoint", ckpt, "--motion-file", MOTION,
         "--task", EVAL_TASK, "--num-envs", "64",
         "--output-file", f"{d}/arm_tracking.json"], tag, "arm_tracking")
    res = {"_ckpt": ckpt, "_tag": tag}
    try:
        res["gap"] = json.load(open(f"{d}/gap_check.json"))
    except Exception as e:  # noqa: BLE001
        log(f"[{tag}] no gap_check.json: {e}")
        res["gap"] = None
    try:
        res["arm"] = json.load(open(f"{d}/arm_tracking.json"))
    except Exception as e:  # noqa: BLE001
        log(f"[{tag}] no arm_tracking.json: {e}")
        res["arm"] = None
    if res["gap"] is None and res["arm"] is None:
        return None
    return res


def gate_pass(res):
    return bool(((res.get("gap") or {}).get("gate") or {}).get("pass"))


def arm_rms(res):
    try:
        return float(res["arm"]["groups"]["arms_all"]["rms_deg"])
    except (KeyError, TypeError):
        return None


def summarize(res, baseline_rms):
    parts = [f"tag={res['_tag']}"]
    gap = res.get("gap") or {}
    nom = (gap.get("conditions") or {}).get("nominal") or {}
    ap = nom.get("ankle_pitch") or {}
    parts.append(f"gate={'PASS' if gate_pass(res) else 'FAIL'}")
    if nom:
        parts.append(f"nominal_survival={nom.get('success_rate'):.3f}")
        parts.append(f"rr_mpkpe={nom.get('mpkpe_root_rel_m'):.3f}")
        parts.append(f"drift_max={(nom.get('drift') or {}).get('max_m'):.2f}")
        if ap.get("mean_abs") is not None:
            parts.append(f"ankle mean={ap['mean_abs']:.2f}/rms={ap['rms_abs']:.2f}Nm")
    r = arm_rms(res)
    if r is not None:
        beat = "" if baseline_rms is None else \
            (" BEATS" if r < baseline_rms else " MISSES") + f" s2r-b {baseline_rms:.2f}"
        parts.append(f"ARM_RMS={r:.2f}deg{beat}")
    return " ".join(parts)


def main():
    log(f"autopilot_v3: variant={VARIANT} eval_task={EVAL_TASK} "
        f"waiting for {JOB_STATUS} ...")
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

    baseline_rms = None
    try:
        baseline_rms = float(json.load(open(BASELINE))["groups"]["arms_all"]["rms_deg"])
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: no s2r-b arm baseline ({e}) — ARM_RMS reported without comparison")

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
    if not (res_last and gate_pass(res_last)) and len(ckpts) > 2:
        res_mid = evaluate(ckpts[len(ckpts) // 2], "mid")
        if res_mid:
            results.append(res_mid)
    if not results:
        write_result(["VERDICT=EVAL_FAILED", f"run={run_dir}"])
        sys.exit(1)

    best = max(results, key=lambda r: (gate_pass(r),
                                       -(arm_rms(r) if arm_rms(r) is not None else 1e9)))
    render_ok = run([PY, f"{NB}/cloud/headless_render_v3.py", best["_ckpt"], MOTION,
                     f"{OUT}/rollout_v3{VARIANT}.mp4", str(RENDER_STEPS), EVAL_TASK],
                    best["_tag"], "render")

    gp = gate_pass(best)
    r = arm_rms(best)
    beats = (r is not None and baseline_rms is not None and r < baseline_rms)
    verdict = ("WIN" if (gp and beats) else
               "GATE_PASS_ARM_MISS" if gp else "GATE_FAIL")
    lines = [
        f"VERDICT={verdict}",
        f"variant=v3{VARIANT} eval_task={EVAL_TASK}",
        f"checkpoint={best['_ckpt']}",
        f"onnx={OUT}/{best['_tag']}/policy.onnx",
        f"gap_check={OUT}/{best['_tag']}/gap_check.json",
        f"arm_tracking={OUT}/{best['_tag']}/arm_tracking.json",
        f"render={OUT}/rollout_v3{VARIANT}.mp4 ({'ok' if render_ok else 'FAILED'})",
        f"s2rb_arm_baseline={baseline_rms}",
    ] + [summarize(x, baseline_rms) for x in results]
    if VARIANT == "b":
        lines += ["", V3B_DEPLOY_NOTE]
    lines += [
        "",
        "NEXT (laptop): compare RESULT.txt across v3a/v3b/v3c (decision matrix in",
        "cloud/V3_PROGRAM.md): gate v3 pass AND arm RMS < s2r-b baseline AND render",
        "visually crisp. Winner -> pull onnx+jsons, stage under data/policies/ as a",
        "CANDIDATE (never overwrite data/policies/thriller/), then the usual: render",
        "sign-off, held-out exams, ONE tethered HW test with telemetry, re-measure",
        "hardware arm RMS vs the 13.2 deg baseline.",
    ]
    write_result(lines)


if __name__ == "__main__":
    main()
