"""Cloud-backed stage implementations: the video->policy path on the GPU box.

These stages wire the app's job model to the flow that produced the show-ready
Thriller (PROJECT_STATE 2026-07-05..06), so a new dance video runs end to end
with only the documented human gates (vet/preview approval, promotion, robot day):

  extract   push 30 fps clip -> GVHMR on the box (~9 min) -> pull hmr4d_results.pt
  train     push <slug>_deploy.csv -> csv_to_npz (mjlab) -> train_sim2real
            (task Mjlab-Tracking-Flat-Unitree-G1-Sim2Real + the s2r-b root-pos
            delta) -> export ONNX -> pull {policy.onnx, policy_meta.json,
            <slug>_deploy.csv/.npz} into data/policies/<slug>/
  verify    sim_gap_check (v3 gates, in cloud/sim_gap_check.py) -> 3x held-out
            exams (disjoint seeds) -> signed sim_exam/v1 verdicts
            (pipeline/mjlab_verify.py) -> register-or-update the dance, attach
            the policy, record the runs -> dance is SIM-VERIFIED
  export    deploy-consumption-contract audit + music attach (if
            data/audio/<slug>/music.* exists). Promotion to show-ready stays a
            HUMAN action in the Shows UI, as does everything robot-facing.

Long box work never holds the worker thread: stages record where they are in
stage meta (reboot-safe) and raise StageBlocked with an honest message; blocked
stages carrying a `retry_after_s` hint are re-queued by the server's poll loop.

All box interaction goes through the thin helpers at the top of this module
(_box_run/_push/_pull/_job_status/_log_tail/_start_script_job) — tests
monkeypatch these; nothing here is allowed to touch the robot.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from .. import cloud, monitor
from ..config import DATA_DIR, PROJECT_ROOT
from ..store import Job
from .base import Reporter, SkipStage, StageBlocked

# ---- box layout (mirrors logs/jobs.md / cloud/README.md) --------------------------
NB = "/workspace/notebook-data"
PY_MJLAB = f"{NB}/envs/mjlab/bin/python"
PY_GVHMR = f"{NB}/envs/gvhmr/bin/python"
CSV_TO_NPZ = f"{NB}/repos/mjlab/src/mjlab/scripts/csv_to_npz.py"

CLOUD_MSG = "waiting on cloud GPU (configure it in Studio → Cloud GPU, then Retry)"
POLL_SECS = 120          # how soon the server's poll loop should re-check a box job

POLICIES_DIR = DATA_DIR / "policies"
AUDIO_DIR = DATA_DIR / "audio"
# Canonical policy interface sidecar template (identical to the promoted Thriller's
# policy_meta.json — PD gains / action scales / obs order are task-level, not
# per-policy; see docs/mjlab_policy_interface.json provenance note).
POLICY_INTERFACE = PROJECT_ROOT / "docs" / "mjlab_policy_interface.json"

# ---- per-dance knobs (drop a dance.yaml/dance.json in the job dir to override) -----
# Defaults are the PROMOTED Thriller recipe: sim2real task v2 + the s2r-b delta
# (motion_global_root_pos 0.5->1.0, the drift fix), 4096 envs, 5000-iter cap,
# 3 held-out exams at 256 envs on de-correlated seeds, gap gate at 128 envs.
DEFAULT_PARAMS: dict = {
    "task": "Mjlab-Tracking-Flat-Unitree-G1-Sim2Real",
    "eval_task": "Mjlab-Tracking-Flat-Unitree-G1",   # exams ran on the stock task id
    "num_envs": 4096,
    "iterations": 5000,
    # extra tyro args appended to the train command; the default carries the
    # s2r-b winning delta. Note: per-joint action caps at DEPLOY time are a
    # runtime knob (pipeline/deploy_runtime.py), not a training parameter —
    # calibrate them from telemetry on robot day, not here.
    "extra_train_args": ["--env.rewards.motion_global_root_pos.weight", "1.0"],
    "heldout_seeds": [90001, 90011, 90021],
    "heldout_num_envs": 256,
    "gap_check_num_envs": 128,
    # Motion window override (seconds into the retargeted motion). None = the
    # vet gate's longest deployable window (the normal case).
    "window_start_s": None,
    "window_end_s": None,
    # GMR joint-velocity clamp during retarget (recipe doc §2) — leave on.
    "velocity_limit": True,
}


def load_params(job: Job) -> dict:
    """DEFAULT_PARAMS overlaid with job.input['params'] and then the first of
    dance.yaml / dance.yml / dance.json found in the job dir. Unknown keys are a
    hard error (a typo must not silently train with defaults)."""
    params = dict(DEFAULT_PARAMS)
    override: dict = dict(job.input.get("params") or {})
    for fname in ("dance.yaml", "dance.yml", "dance.json"):
        f = job.dir / fname
        if f.exists():
            if fname.endswith(".json"):
                file_over = json.loads(f.read_text() or "{}")
            else:
                import yaml
                file_over = yaml.safe_load(f.read_text()) or {}
            if not isinstance(file_over, dict):
                raise RuntimeError(f"{fname} must contain a mapping of param: value")
            override.update(file_over)
            break
    unknown = set(override) - set(params)
    if unknown:
        raise RuntimeError(f"unknown dance param(s) {sorted(unknown)} — "
                           f"valid keys: {sorted(params)}")
    params.update(override)
    return params


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "dance").lower()).strip("_")
    return s or "dance"


def _suffix(job: Job) -> str:
    return job.id.rsplit("-", 1)[-1]


# ---- thin box helpers (tests monkeypatch these) ------------------------------------

def _require_cloud() -> None:
    if not cloud.load_config().get("transport"):
        raise StageBlocked(CLOUD_MSG)


def _blocked(msg: str, retry_s: float | None = POLL_SECS) -> StageBlocked:
    """A StageBlocked that the server poll loop will retry after `retry_s`."""
    e = StageBlocked(msg)
    if retry_s:
        e.retry_after_s = float(retry_s)
    return e


def _box_run(cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    """cloud.run with connectivity failures mapped to (retryable) StageBlocked."""
    try:
        rc, out, err = cloud.run(cmd, timeout=timeout)
    except ValueError as e:                    # transport not configured
        raise StageBlocked(str(e))
    except Exception as e:                     # timeout / network / ssh missing
        raise _blocked(f"box unreachable: {type(e).__name__}: {e}"[:250])
    if rc == 255:                              # ssh-level connection failure
        raise _blocked(f"box unreachable (ssh): {(err or out).strip()[-200:]}")
    return rc, out, err


_TRANSIENT_SCP = ("timed out", "connection", "route to host", "could not resolve",
                  "broken pipe")


def _transfer(fn, *args, **kwargs):
    """push/pull with connectivity failures mapped to retryable StageBlocked;
    real errors (missing file, auth) fail the stage honestly."""
    try:
        return fn(*args, **kwargs)
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as e:
        text = str(e).lower()
        if isinstance(e, subprocess.TimeoutExpired) or \
                any(t in text for t in _TRANSIENT_SCP):
            raise _blocked(f"box transfer failed (will retry): {e}"[:250])
        raise


def _push(local: Path | str, remote: str, timeout: int = 1800) -> None:
    _transfer(cloud.push_file, local, remote, timeout=timeout)


def _pull(remote: str, local: Path | str, timeout: int = 1800) -> Path:
    return _transfer(cloud.pull_file, remote, local, timeout=timeout)


def _job_status(name: str) -> dict | None:
    """Parsed run_job.sh status JSON, or None if not present/readable yet."""
    rc, out, _ = _box_run(f"cat {NB}/jobs/{name}.status.json 2>/dev/null")
    if rc != 0 or not out.strip():
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def _log_tail(name: str, n: int = 80) -> str:
    rc, out, _ = _box_run(f"tail -n {n} {NB}/jobs/{name}.log 2>/dev/null")
    return out if rc == 0 else ""


def _start_script_job(name: str, script: str) -> None:
    """Write the job body as a script on the box (no tmux quoting hazards), then
    launch it detached through cloud/run_job.sh (status.json + log, reboot-safe)."""
    path = f"{NB}/jobs/scripts/{name}.sh"
    rc, out, err = _box_run(
        f"mkdir -p {NB}/jobs/scripts && cat > {path} <<'G1_EOF'\n{script}\nG1_EOF")
    if rc != 0:
        raise RuntimeError(f"could not stage job script {name}: {(err or out)[-300:]}")
    rc, out, err = _box_run(
        f'cd {NB} && bash cloud/run_job.sh start {name} -- "bash {path}"')
    if rc != 0:
        if "already running" in (out + err):
            return                      # resume path: the job survived our restart
        raise RuntimeError(f"could not start box job {name}: {(err or out)[-300:]}")


def _remote_first(glob_expr: str) -> str | None:
    """Newest path matching a glob on the box (ls -dt … | head -1)."""
    rc, out, _ = _box_run(f"ls -dt {glob_expr} 2>/dev/null | head -1")
    line = out.strip().splitlines()[0].strip() if out.strip() else ""
    return line or None


def _drive_box_job(job: Job, st, name: str, script: str, *, desc: str,
                   report: Reporter, running=None) -> None:
    """Start-once + poll one run_job.sh job. Returns only when the job is done;
    raises StageBlocked (poll-retryable) while running, RuntimeError on failure."""
    started_key = f"started:{name}"
    if not st.meta.get(started_key):
        _start_script_job(name, script)
        st.meta[started_key] = time.time()
        job.log(f"box job '{name}' started — {desc}")
        raise _blocked(f"{desc} — started on the box")
    status = _job_status(name)
    if status is None or status.get("state") == "running":
        progress, msg = running() if running else (None, f"{desc} — running on the box")
        if progress is not None:
            report(progress, msg)
        raise _blocked(msg)
    if status.get("state") != "done":
        tail = _log_tail(name)
        st.meta.pop(started_key, None)      # Retry restarts the box job cleanly
        raise RuntimeError(f"box job '{name}' failed ({desc}) — log tail:\n"
                           f"{tail[-600:]}")


# ---- box job scripts ---------------------------------------------------------------

GVHMR_SCRIPT = """set -e
cd {nb}/repos/GVHMR
{py} tools/demo/demo.py --video {video} --output_root {nb}/artifacts/gvhmr -s
test -s {nb}/artifacts/gvhmr/{stem}/hmr4d_results.pt
echo GVHMR_OK {stem}"""

# csv_to_npz writes /tmp/motion.npz then uploads to W&B; offline the upload may
# exit nonzero AFTER the save, so tolerate its rc and gate on the copied npz.
CONVERT_SCRIPT = """set -e
cd {nb}
export LD_LIBRARY_PATH=/opt/conda/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}
export MUJOCO_GL=egl WANDB_MODE=offline
rm -f /tmp/motion.npz
{py} {converter} --input-file {csv} --output-name {name} \
  --input-fps 30 --output-fps 50 \
  || echo "csv_to_npz rc nonzero (offline wandb upload) — checking npz anyway"
cp /tmp/motion.npz {npz}
test -s {npz}
echo CONVERT_OK {npz}"""

TRAIN_SCRIPT = """set -e
cd {nb}
export LD_LIBRARY_PATH=/opt/conda/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}
export MUJOCO_GL=egl
for f in .wandb_key .wandb.key; do [ -f "$f" ] && export WANDB_API_KEY="$(cat $f)"; done
[ -n "${{WANDB_API_KEY:-}}" ] || export WANDB_MODE=offline
{py} cloud/train_sim2real.py {task} \
  --env.scene.num-envs {num_envs} \
  --env.commands.motion.motion-file {npz} \
  --agent.max-iterations {iterations} \
  --agent.run-name {run_name} \
  --video False {extra}"""

EXPORT_SCRIPT = """set -e
export LD_LIBRARY_PATH=/opt/conda/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}
export MUJOCO_GL=egl WANDB_MODE=disabled
mkdir -p {exports}
cd /tmp
{py} {nb}/cloud/export_policy.py '{ckpt}' {npz} {exports}
test -s {exports}/policy.onnx
echo EXPORT_OK {exports}/policy.onnx"""

GAP_SCRIPT = """set -e
cd {nb}
export LD_LIBRARY_PATH=/opt/conda/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}
export MUJOCO_GL=egl WANDB_MODE=disabled
{py} cloud/sim_gap_check.py --checkpoint '{ckpt}' --motion-file {npz} \
  --num-envs {num_envs} --output-file {out}"""

EXAM_SCRIPT = """set -e
cd {nb}
export LD_LIBRARY_PATH=/opt/conda/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}
export MUJOCO_GL=egl WANDB_MODE=disabled
{py} cloud/heldout_eval.py {task} --checkpoint '{ckpt}' --motion-file {npz} \
  --num-envs {num_envs} --seed {seed} --output-file {out}"""


# ---- extract: video -> GVHMR (box) -> hmr4d_results.pt -----------------------------

def _reencode_30fps(src: Path, dst: Path) -> None:
    """Normalize the clip to constant 30 fps (GVHMR + the 30 fps retarget contract;
    the Thriller source was VFR and needed exactly this). Audio is dropped — music
    is attached from the original file at the export stage."""
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vf", "fps=30", "-an",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(dst)],
        capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg 30 fps re-encode failed: {proc.stderr[-400:]}")


class ExtractStage:
    """Video intake validation (local, fail-fast) then GVHMR on the GPU box."""

    name = "extract"

    def run(self, job: Job, report: Reporter) -> None:
        if (job.dir / "input.csv").exists():
            raise SkipStage("input is already robot motion (CSV) — no video to extract")
        # Legacy/manual jobs: if the retarget output already exists (e.g. the
        # 2026-07-03 Thriller job, retargeted by hand), extraction has nothing
        # left to provide — never burn box GPU re-extracting on a restart requeue.
        if job.stages["retarget"].state == "done":
            raise SkipStage("retarget already complete — extraction not needed")
        st = job.stages[self.name]
        out_dir = job.stage_dir(self.name)

        if "video" not in st.meta:
            from ..video_probe import validate
            report(0.05, "checking video file")
            st.meta["video"] = validate(job.dir / "input.mp4")  # raises w/ reason
            for adv in st.meta["video"]["advisories"]:
                job.log(f"extract: ADVISORY — {adv}")
            job.log(f"extract: video ok — {st.meta['video']['duration_s']}s "
                    f"{st.meta['video']['width']}x{st.meta['video']['height']} "
                    f"@ {st.meta['video']['fps']}fps")

        clip = out_dir / "input_30fps.mp4"
        if not clip.exists():
            report(0.1, "normalizing video to constant 30 fps")
            _reencode_30fps(job.dir / "input.mp4", clip)
            job.log("extract: re-encoded to constant 30 fps for GVHMR")

        # Capture the source video's OWN soundtrack for the show music. The 30 fps
        # re-encode above stays silent on purpose (GVHMR needs silent video); this
        # pulls the audio track to the job dir, and the retarget window trims it to
        # the danced span at export (see _prepare_windowed_music). A silent source
        # simply stays silent — the operator can attach a music file later. Failures
        # here are never fatal (audio is presentation-only).
        if "source_audio" not in st.meta:
            try:
                from .. import audio as audio_mod
                src_video = job.dir / "input.mp4"
                if audio_mod.has_audio(src_video):
                    src_audio = job.dir / "source_audio.wav"
                    audio_mod.extract_audio(src_video, src_audio)
                    st.meta["source_audio"] = str(src_audio)
                    job.log(f"extract: captured source audio -> {src_audio.name}")
                else:
                    st.meta["source_audio"] = None
                    job.log("extract: source video has no audio track — dance "
                            "stays silent (attach music later on the Shows page)")
            except Exception as e:  # noqa: BLE001 — audio is presentation-only
                st.meta["source_audio"] = None
                job.log(f"extract: source-audio capture skipped (non-fatal): {e}")

        _require_cloud()
        stem = f"{_slug(job.name)}_{_suffix(job)}"
        box_video = f"{NB}/videos_in/{stem}.mp4"
        if not st.meta.get("pushed"):
            report(0.15, "uploading video to the GPU box")
            _box_run(f"mkdir -p {NB}/videos_in", timeout=30)   # a fresh box may lack it -> scp "No such file"
            _push(clip, box_video)
            st.meta["pushed"] = True
            job.log(f"extract: pushed clip -> {box_video}")

        jobname = f"gvhmr-{stem}"
        script = GVHMR_SCRIPT.format(nb=NB, py=PY_GVHMR, video=box_video, stem=stem)
        _drive_box_job(job, st, jobname, script, report=report,
                       desc="GVHMR pose extraction (box GPU, ~9 min for a 45 s clip)",
                       running=lambda: (0.4, "GVHMR pose extraction running on the box"))

        report(0.85, "downloading extracted human motion")
        pred = _pull(f"{NB}/artifacts/gvhmr/{stem}/hmr4d_results.pt",
                     out_dir / "hmr4d_results.pt")
        st.meta["pred"] = str(pred)
        st.meta.pop("poll_after", None)
        job.log("extract: pulled hmr4d_results.pt")
        report(1.0, "human motion extracted (GVHMR)")


# ---- train: prepped CSV -> npz -> train_sim2real -> ONNX + meta + artifacts --------

class TrainStage:
    name = "train"

    def run(self, job: Job, report: Reporter) -> None:
        st = job.stages[self.name]
        rmeta = job.stages["retarget"].meta
        deploy_csv = rmeta.get("deploy_csv")
        if not deploy_csv or not Path(deploy_csv).exists():
            raise StageBlocked("needs the prepped deployable motion "
                               "(retarget stage incomplete)")
        # HUMAN GATE (architecture §5): the operator watches the MuJoCo preview
        # before any GPU spend. POST /api/jobs/{id}/approve-train flips this.
        if not st.meta.get("approved"):
            raise StageBlocked(
                "waiting for human approval — review the motion preview, then "
                "click 'Approve training' (2-3 h GPU run)")
        _require_cloud()
        params = load_params(job)
        slug, suffix = _slug(job.name), _suffix(job)
        # STAND-END: the trained/deployed motion must FINISH at the standby pose so
        # the dance ends STANDING and deploy_runtime --exit stand can hand back to
        # onboard balance (validated on hardware 2026-07-07). RetargetStage builds
        # the deploy CSV with only the 2.5 s activation ramp; here we rebuild it
        # from the show CSV WITH the return-to-standing tail (deploy_ramp stand_end)
        # and train + deploy THAT. Idempotent (keyed on the file); falls back to the
        # retarget deploy CSV when no show CSV is on record (e.g. a hand-retargeted
        # legacy job). Only writes into this job's own train stage dir — the golden
        # data/policies/thriller deploy CSV is never touched.
        show_csv = rmeta.get("show_csv")
        if show_csv and Path(show_csv).exists():
            standend_csv = job.stage_dir(self.name) / f"{slug}_deploy.csv"
            if not standend_csv.exists():
                from ..deploy_ramp import make_deploy_csv
                st.meta["deploy_ramp"] = make_deploy_csv(
                    Path(show_csv), standend_csv, stand_end=True)
                st.meta["stand_end"] = True
                job.log("train: rebuilt deployable WITH return-to-standing tail — "
                        f"{st.meta['deploy_ramp']['seconds']}s, ends "
                        f"{st.meta['deploy_ramp']['final_max_delta_rad']:.3f} rad "
                        "from the standby pose (ends standing)")
            deploy_csv = str(standend_csv)
        run_name = f"train-{slug}-{suffix}"
        box_csv = f"{NB}/motions/{slug}_deploy.csv"
        box_npz = f"{NB}/motions/{slug}_deploy.npz"
        exports = f"{NB}/exports/app_{slug}_{suffix}"
        st.meta.setdefault("run_name", run_name)
        st.meta.setdefault("box_npz", box_npz)
        st.meta.setdefault("exports", exports)
        phase = st.meta.setdefault("phase", "push")

        if phase == "push":
            report(0.02, "uploading deployable motion CSV to the box")
            _push(deploy_csv, box_csv)
            job.log(f"train: pushed {Path(deploy_csv).name} -> {box_csv}")
            phase = st.meta["phase"] = "convert"

        if phase == "convert":
            script = CONVERT_SCRIPT.format(nb=NB, py=PY_MJLAB, converter=CSV_TO_NPZ,
                                           csv=box_csv, name=f"{slug}_deploy",
                                           npz=box_npz)
            _drive_box_job(job, st, f"convert-{slug}-{suffix}", script, report=report,
                           desc="converting motion CSV -> npz (mjlab, box GPU)",
                           running=lambda: (0.06, "csv_to_npz running on the box"))
            job.log(f"train: motion converted -> {box_npz}")
            phase = st.meta["phase"] = "train"

        if phase == "train":
            extra = " ".join(str(a) for a in params["extra_train_args"])
            script = TRAIN_SCRIPT.format(nb=NB, py=PY_MJLAB, task=params["task"],
                                         num_envs=params["num_envs"], npz=box_npz,
                                         iterations=params["iterations"],
                                         run_name=run_name, extra=extra)

            def running():
                info = monitor.parse_job_log(run_name, _log_tail(run_name, 120))
                it, mx = info.get("iteration"), info.get("max_iteration")
                if it and mx:
                    frac = it / mx
                    rew = info.get("mean_reward")
                    return (0.1 + 0.75 * frac,
                            f"training {run_name}: iter {it}/{mx} ({frac:.0%})"
                            + (f", reward {rew:.1f}" if rew is not None else ""))
                return (0.1, f"training {run_name}: starting up on the box")

            _drive_box_job(job, st, run_name, script, report=report,
                           desc=f"sim2real training ({params['iterations']} iters, "
                                "~2-3 h)", running=running)
            job.log(f"train: {run_name} converged/completed on the box")
            phase = st.meta["phase"] = "export"

        if phase == "export":
            if not st.meta.get("checkpoint"):
                run_dir = _remote_first(
                    f"{NB}/logs/rsl_rl/g1_tracking/*{run_name}* "
                    f"{NB}/cloud/logs/rsl_rl/g1_tracking/*{run_name}*")
                if not run_dir:
                    raise RuntimeError(f"no training run dir found for {run_name}")
                ckpt = _remote_first(f"{run_dir}/model_*.pt")
                if not ckpt:
                    raise RuntimeError(f"no checkpoints in {run_dir}")
                st.meta["run_dir"], st.meta["checkpoint"] = run_dir, ckpt
                job.log(f"train: exporting checkpoint {ckpt}")
            script = EXPORT_SCRIPT.format(nb=NB, py=PY_MJLAB, exports=exports,
                                          ckpt=st.meta["checkpoint"], npz=box_npz)
            _drive_box_job(job, st, f"export-{slug}-{suffix}", script, report=report,
                           desc="exporting policy to ONNX (box)",
                           running=lambda: (0.9, "ONNX export running on the box"))
            phase = st.meta["phase"] = "pull"

        if phase == "pull":
            report(0.95, "downloading policy artifacts")
            pol_dir = POLICIES_DIR / slug
            pol_dir.mkdir(parents=True, exist_ok=True)
            _pull(f"{exports}/policy.onnx", pol_dir / "policy.onnx")
            _pull(box_npz, pol_dir / f"{slug}_deploy.npz")
            shutil.copyfile(deploy_csv, pol_dir / f"{slug}_deploy.csv")
            meta = json.loads(POLICY_INTERFACE.read_text())
            meta.update({
                "task": params["task"],
                "dance": slug,
                "trained_run": run_name,
                "exported_from_checkpoint": st.meta["checkpoint"],
                "generated_by": "app TrainStage (policy-independent PD/obs contract "
                                "from docs/mjlab_policy_interface.json)",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
            (pol_dir / "policy_meta.json").write_text(json.dumps(meta, indent=2))
            st.meta["policy_dir"] = str(pol_dir)
            st.meta["phase"] = "done"
            st.meta.pop("poll_after", None)
            job.log(f"train: artifacts staged -> {pol_dir} "
                    "(policy.onnx, policy_meta.json, deploy csv/npz)")
            report(1.0, f"policy trained + staged in {pol_dir}")


# ---- verify: gap gate + 3x held-out exams + signed verdicts + dance record ---------

class VerifyStage:
    name = "verify"

    def run(self, job: Job, report: Reporter) -> None:
        st = job.stages[self.name]
        t = job.stages["train"].meta
        if t.get("phase") != "done":
            raise StageBlocked("needs a trained policy first")
        _require_cloud()
        params = load_params(job)
        slug, suffix = _slug(job.name), _suffix(job)
        pol_dir = Path(t["policy_dir"])
        ckpt, box_npz, exports = t["checkpoint"], t["box_npz"], t["exports"]
        out_dir = job.stage_dir(self.name)
        phase = st.meta.setdefault("phase", "gap")

        if phase == "gap":
            script = GAP_SCRIPT.format(nb=NB, py=PY_MJLAB, ckpt=ckpt, npz=box_npz,
                                       num_envs=params["gap_check_num_envs"],
                                       out=f"{exports}/gap_check.json")
            _drive_box_job(job, st, f"gap-{slug}-{suffix}", script, report=report,
                           desc="sim-gap gate v3 (full-motion, 7 conditions)",
                           running=lambda: (0.15, "sim-gap gate running on the box"))
            local = _pull(f"{exports}/gap_check.json", out_dir / "gap_check.json")
            shutil.copyfile(local, pol_dir / "gap_check.json")
            gap = json.loads(local.read_text())
            gate = gap.get("gate") or {}
            nom = (gap.get("conditions") or {}).get("nominal") or {}
            ankle = nom.get("ankle_pitch") or {}
            summary = (f"nominal survival {nom.get('success_rate')}, ankle mean "
                       f"{ankle.get('mean_abs')} / RMS {ankle.get('rms_abs')} Nm")
            st.meta["gap"] = {"pass": bool(gate.get("pass")), "summary": summary}
            if not gate.get("pass"):
                raise RuntimeError(
                    f"sim-gap gate FAILED ({summary}) — see verify/gap_check.json; "
                    "per-section stats there drive the next move: a targeted "
                    "choreography edit (tools/edit_choreography.py) or a recipe "
                    "delta, then a FRESH job with a dance.yaml (this job's "
                    "training is already complete)")
            job.log(f"verify: sim-gap gate PASS ({summary})")
            phase = st.meta["phase"] = "exams"

        if phase == "exams":
            seeds = list(params["heldout_seeds"])
            done: list = st.meta.setdefault("exams_done", [])
            for k, seed in enumerate(seeds, 1):
                if k in done:
                    continue
                script = EXAM_SCRIPT.format(
                    nb=NB, py=PY_MJLAB, task=params["eval_task"], ckpt=ckpt,
                    npz=box_npz, num_envs=params["heldout_num_envs"], seed=seed,
                    out=f"{exports}/heldout_eval_s{k}.json")
                _drive_box_job(
                    job, st, f"exam-{slug}-{suffix}-s{k}", script, report=report,
                    desc=f"held-out exam {k}/{len(seeds)} (seed {seed}, "
                         f"{params['heldout_num_envs']} envs)",
                    running=lambda k=k: (0.3 + 0.15 * k,
                                         f"held-out exam {k}/{len(seeds)} running"))
                _pull(f"{exports}/heldout_eval_s{k}.json",
                      out_dir / f"heldout_eval_s{k}.json")
                done.append(k)
                job.log(f"verify: held-out exam s{k} (seed {seed}) complete")
            phase = st.meta["phase"] = "sign"

        if phase == "sign":
            report(0.85, "signing sim-exam verdicts")
            from ..mjlab_verify import build_verdict
            failures = []
            for k in range(1, len(params["heldout_seeds"]) + 1):
                eval_json = json.loads((out_dir / f"heldout_eval_s{k}.json").read_text())
                v = build_verdict(eval_json,
                                  policy_path=pol_dir / "policy.onnx",
                                  motion_path=pol_dir / f"{slug}_deploy.csv",
                                  eval_motion_path=pol_dir / f"{slug}_deploy.npz")
                text = json.dumps(v, indent=2)
                (out_dir / f"heldout_verdict_s{k}.json").write_text(text)
                (pol_dir / f"heldout_verdict_s{k}.json").write_text(text)
                if v["verdict"] != "pass":
                    failures.append(
                        f"s{k}: nominal {v['nominal']['n_success']}/"
                        f"{v['nominal']['num_episodes']}, push "
                        f"{v['push']['n_success']}/{v['push']['num_episodes']}")
                job.log(f"verify: verdict s{k} = {v['verdict']}")
            if failures:
                raise RuntimeError(
                    "held-out exam below the ≥99% show bar — " + "; ".join(failures)
                    + " (policy stays draft; inspect verify/heldout_eval_s*.json)")
            phase = st.meta["phase"] = "register"

        if phase == "register":
            report(0.95, "registering dance + recording exam runs")
            dance = _register_dance(job, slug, pol_dir)
            st.meta["dance_id"] = dance.id
            st.meta["phase"] = "done"
            st.meta.pop("poll_after", None)
            job.log(f"verify: dance '{dance.name}' ({dance.id}) is {dance.status} "
                    f"with {dance.repeatability['consecutive_clean']} clean exam runs")
            report(1.0, f"dance sim-verified — promotion to show-ready stays a "
                        "human action (Shows page)")


def _rel(p: Path) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _register_dance(job: Job, slug: str, pol_dir: Path):
    """Register-or-update the dance record, bind it to the DEPLOYABLE csv (audit
    motion-sha seam), attach the policy, and credit the signed exam verdicts."""
    from .. import shows
    dance = shows.find_dance(job.name) or shows.new_dance(job.name)
    rmeta = job.stages["retarget"].meta
    dance.motion_csv = _rel(pol_dir / f"{slug}_deploy.csv")   # DEPLOYABLE binding
    window = rmeta.get("window") or {}
    if window.get("seconds"):
        # danced span (window), which is what the audio 1.5 s lead-in aligns to
        dance.duration_s = float(window["seconds"])
    if rmeta.get("preview"):
        dance.preview = rmeta["preview"]
    vet_file = job.dir / "retarget" / "vet.json"
    if vet_file.exists():
        dance.vet = json.loads(vet_file.read_text())
    dance.source_job = job.id
    dance.save()
    # attach_policy resets verification state (by design) — do it BEFORE verdicts
    dance = shows.attach_policy(
        dance.id, _rel(pol_dir / "policy.onnx"),
        notes=f"trained by app job {job.id} "
              f"({job.stages['train'].meta.get('run_name')})")
    out_dir = job.dir / "verify"
    for vf in sorted(out_dir.glob("heldout_verdict_s*.json")):
        verdict = json.loads(vf.read_text())
        dance = shows.record_sim_run_from_verdict(dance.id, verdict)
    return dance


# ---- export: deploy-contract audit + music attach ----------------------------------

class ExportStage:
    name = "export"

    def run(self, job: Job, report: Reporter) -> None:
        from .. import shows
        v = job.stages["verify"].meta
        if v.get("phase") != "done" or not v.get("dance_id"):
            raise StageBlocked("needs a sim-verified dance first")
        slug = _slug(job.name)
        pol_dir = Path(job.stages["train"].meta["policy_dir"])
        out_dir = job.stage_dir(self.name)
        dance = shows.load_dance(v["dance_id"])

        # Deploy consumption contract (what pipeline/deploy_runtime.py + the
        # bundle builder read): all four artifacts, and the dance record bound
        # to the deployable CSV.
        report(0.2, "auditing deploy consumption contract")
        required = ["policy.onnx", "policy_meta.json",
                    f"{slug}_deploy.csv", f"{slug}_deploy.npz"]
        missing = [n for n in required if not (pol_dir / n).is_file()]
        if missing:
            raise RuntimeError(f"deploy contract incomplete — missing {missing} "
                               f"in {pol_dir}")
        bound = shows._abs(dance.motion_csv) if dance.motion_csv else None
        if bound is None or Path(bound).resolve() != (pol_dir / f"{slug}_deploy.csv").resolve():
            raise RuntimeError("dance motion_csv is not bound to the deployable "
                               f"CSV ({pol_dir / (slug + '_deploy.csv')}) — audit seam")

        # Music: window the source video's captured soundtrack to the danced span
        # and write it to data/audio/<slug>/music.wav (no-op for CSV inputs / silent
        # videos / when a real music file is already there), then attach whatever
        # data/audio/<slug>/music.* is present. attach_audio_for_dance adds the
        # prep lead-in on top of the already-windowed track.
        report(0.6, "attaching music (if provided)")
        _prepare_windowed_music(job, slug)
        track = _find_music(slug)
        if track and not dance.audio:
            from .. import audio as audio_mod
            try:
                record = audio_mod.attach_audio_for_dance(dance, source_audio=track)
                dance = shows.set_audio(dance.id, record)
                job.log(f"export: music attached from {track} "
                        "(1.5 s lead-in on the show timeline; deploy adds the "
                        "2.5 s activation ramp before that)")
            except Exception as e:  # noqa: BLE001 — audio is presentation-only
                job.log(f"export: audio attach failed (non-fatal): {e}")
        elif not track:
            job.log(f"export: no music found at data/audio/{slug}/music.* — "
                    "dance stays silent (attach later from the Shows page)")

        summary = {
            "dance_id": dance.id,
            "dance_status": dance.status,
            "policy_dir": str(pol_dir),
            "deployable_csv": dance.motion_csv,
            "audio": dance.audio and dance.audio.get("track"),
            "next_human_steps": [
                "review previews + exam verdicts, then promote to show-ready "
                "in the Shows page (guarded: 3 clean signed exams required)",
                "robot day: gantry/tether-first per docs/ROBOT_DAY_RUNBOOK.md — "
                "the app never contacts the robot",
            ],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        report(1.0, "show-ready CANDIDATE complete — promotion + robot day are "
                    "human actions")


def _find_music(slug: str) -> Path | None:
    from ..audio import AUDIO_EXTS
    d = AUDIO_DIR / slug
    if not d.is_dir():
        return None
    for ext in sorted(AUDIO_EXTS):
        p = d / f"music{ext}"
        if p.is_file():
            return p
    return None


def _trim_audio(src: Path, start_s: float, duration_s: float, out: Path) -> Path:
    """Trim `src` audio to [start_s, start_s+duration_s] and write `out` as WAV.

    Reuses audio.py's atrim approach (see mux_audio_onto_video) and matches
    extract_audio's WAV format (pcm_s16le / 44.1 kHz / stereo) so the windowed
    music lines up with the danced span; the export attach then adds the lead-in.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    afilter = f"atrim=start={start_s}:duration={duration_s},asetpts=PTS-STARTPTS"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-af", afilter,
         "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", str(out)],
        check=True, capture_output=True, text=True, timeout=600)
    return out


def _prepare_windowed_music(job: Job, slug: str) -> Path | None:
    """Trim the source video's captured soundtrack to the danced window and write
    it to data/audio/<slug>/music.wav (the path ExportStage's _find_music reads).

    The extract stage captured the video's own audio to the job dir; the retarget
    stage recorded which frame window was actually danced (a tutorial's talking
    intro is trimmed away — a real, load-bearing offset). Here we cut the audio to
    that window so what reaches attach_audio_for_dance is ALREADY the danced span,
    and export only adds the prep lead-in.

    No-op — the dance stays silent — for CSV inputs and silent videos (no captured
    source audio). Never fabricates a placeholder click track: only real captured
    audio becomes music.wav, and a real music file already dropped in by the
    operator is left untouched. Returns the music path, or None when nothing was
    produced.
    """
    src = job.stages["extract"].meta.get("source_audio")
    if not src or not Path(src).exists():
        return None                      # CSV input or silent source: stay silent
    existing = _find_music(slug)
    if existing is not None:
        return existing                  # a real music file is already in place
    window = job.stages["retarget"].meta.get("window") or {}
    start_frame = window.get("start_frame")
    if start_frame is None:
        return None
    from .. import audio as audio_mod
    from ..find_window import CSV_FPS
    window_start_s = float(start_frame) / CSV_FPS
    # danced-span length: prefer the recorded window seconds (what dance.duration_s
    # and the attach/mux use) so music.wav length matches the span exactly.
    dance_s = float(window.get("seconds") or 0.0)
    if dance_s <= 0:
        end_frame = window.get("end_frame")
        if end_frame is None:
            return None
        dance_s = (int(end_frame) - int(start_frame) + 1) / CSV_FPS
    align = audio_mod.compute_alignment(dance_s, window_start_s=window_start_s)
    out = AUDIO_DIR / slug / "music.wav"
    try:
        _trim_audio(Path(src), align.trim_start_s, align.trim_duration_s, out)
    except Exception as e:  # noqa: BLE001 — audio is presentation-only
        job.log(f"export: windowing source audio failed (non-fatal): {e}")
        return None
    job.log(f"export: windowed source audio -> data/audio/{slug}/music.wav "
            f"(source {align.trim_start_s:.2f}s +{align.trim_duration_s:.2f}s "
            "= the danced span; export adds the 1.5 s lead-in)")
    return out
