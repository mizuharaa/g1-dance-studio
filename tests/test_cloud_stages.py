"""Cloud-backed pipeline stages (pipeline/stages/cloud_motion.py).

Everything is mocked at the box-helper seam — NO real box/SSH/GPU calls, same
policy as the rest of the suite. The FakeBox simulates the run_job.sh contract
(start -> status.json running/done/failed -> log) plus scp push/pull.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline import cloud
from pipeline.config import STAGE_ORDER
from pipeline.runner import Runner
from pipeline.stages import cloud_motion as cm
from pipeline.stages.base import StageBlocked


# ---- fake box --------------------------------------------------------------------

class FakeBox:
    """Simulates the box at the thin-helper seam the stages call through."""

    def __init__(self, monkeypatch):
        self.jobs: dict[str, dict] = {}      # name -> {"state", "script"}
        self.logs: dict[str, str] = {}       # name -> log text
        self.pushed: dict[str, bytes] = {}   # remote path -> bytes
        self.files: dict[str, bytes] = {}    # remote path -> bytes served on pull
        self.globs: list[tuple[str, str]] = []  # (substring, result) for ls -dt
        monkeypatch.setattr(cm, "_require_cloud", lambda: None)
        monkeypatch.setattr(cm, "_start_script_job", self.start)
        monkeypatch.setattr(cm, "_job_status", self.status)
        monkeypatch.setattr(cm, "_log_tail", lambda name, n=80: self.logs.get(name, ""))
        monkeypatch.setattr(cm, "_push", self.push)
        monkeypatch.setattr(cm, "_pull", self.pull)
        monkeypatch.setattr(cm, "_remote_first", self.remote_first)

    def start(self, name, script):
        self.jobs[name] = {"state": "running", "script": script}

    def status(self, name):
        j = self.jobs.get(name)
        return {"name": name, "state": j["state"], "rc": 0} if j else None

    def finish(self, name, state="done"):
        self.jobs[name]["state"] = state

    def push(self, local, remote, timeout=0):
        self.pushed[remote] = Path(local).read_bytes()

    def pull(self, remote, local, timeout=0):
        if remote not in self.files:
            raise RuntimeError(f"scp failed (rc=1): no such remote file {remote}")
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(self.files[remote])
        return local

    def remote_first(self, expr):
        for sub, result in self.globs:
            if sub in expr:
                return result
        return None


@pytest.fixture
def box(monkeypatch):
    return FakeBox(monkeypatch)


@pytest.fixture
def policies_env(tmp_path, monkeypatch):
    pol = tmp_path / "policies"
    monkeypatch.setattr(cm, "POLICIES_DIR", pol)
    monkeypatch.setattr(cm, "AUDIO_DIR", tmp_path / "audio")
    return pol


def _mk_deploy_csv(tmp_path) -> Path:
    p = tmp_path / "input_deploy.csv"
    p.write_text("0.0," * 35 + "1.0\n")
    return p


def _train_ready_job(store, tmp_path, name="groove", approved=True):
    """A job whose retarget stage already produced the deployable motion."""
    job = store.new_job(name, input={"type": "video", "source": "clip.mp4"})
    deploy = _mk_deploy_csv(tmp_path)
    job.stages["extract"].state = "done"
    job.stages["retarget"].state = "done"
    job.stages["retarget"].meta = {
        "deploy_csv": str(deploy),
        "window": {"start_frame": 0, "end_frame": 899, "seconds": 30.0,
                   "input_seconds": 32.0},
        "preview": f"/previews/job-{job.id}.mp4",
    }
    if approved:
        job.stages["train"].meta["approved"] = {"at": 1.0, "by": "test"}
    job.save()
    return job


def _drive(stage, job, expect_blocked=True):
    """One stage invocation; returns the StageBlocked (or None when it ran through)."""
    try:
        stage.run(job, lambda p, m: None)
    except StageBlocked as e:
        assert expect_blocked, f"unexpected block: {e}"
        return e
    assert not expect_blocked, "expected StageBlocked but the stage completed"
    return None


# ---- per-dance params --------------------------------------------------------------

def test_params_defaults_are_promoted_recipe(jobs_env):
    store, _ = jobs_env
    job = store.new_job("d")
    p = cm.load_params(job)
    assert p["task"] == "Mjlab-Tracking-Flat-Unitree-G1-Sim2Real"
    assert p["iterations"] == 5000 and p["num_envs"] == 4096
    assert p["heldout_seeds"] == [90001, 90011, 90021]
    # the s2r-b winning delta rides along by default
    assert "--env.rewards.motion_global_root_pos.weight" in p["extra_train_args"]


def test_params_dance_yaml_overrides(jobs_env):
    store, _ = jobs_env
    job = store.new_job("d")
    (job.dir / "dance.yaml").write_text(
        "iterations: 3000\nwindow_start_s: 2.0\nwindow_end_s: 30.0\n")
    p = cm.load_params(job)
    assert p["iterations"] == 3000
    assert p["window_start_s"] == 2.0
    assert p["num_envs"] == 4096          # untouched default


def test_params_unknown_key_is_a_hard_error(jobs_env):
    store, _ = jobs_env
    job = store.new_job("d")
    (job.dir / "dance.yaml").write_text("iteration: 3000\n")  # typo
    with pytest.raises(RuntimeError, match="unknown dance param"):
        cm.load_params(job)


def test_slug_normalizes_names():
    assert cm._slug("Thriller Dance Final") == "thriller_dance_final"
    assert cm._slug("  Grüv!! ") == "gr_v"


# ---- train stage -------------------------------------------------------------------

def test_train_blocks_without_human_approval(jobs_env, tmp_path, box, policies_env):
    store, _ = jobs_env
    job = _train_ready_job(store, tmp_path, approved=False)
    e = _drive(cm.TrainStage(), job)
    assert "approval" in str(e).lower()
    assert not box.jobs                      # nothing launched without the human gate
    assert not getattr(e, "retry_after_s", None)   # polling can't skip this gate


def test_train_full_walkthrough(jobs_env, tmp_path, box, policies_env):
    store, _ = jobs_env
    job = _train_ready_job(store, tmp_path)
    slug, suffix = "groove", cm._suffix(job)
    stage = cm.TrainStage()
    st = job.stages["train"]

    # 1: pushes the deployable CSV, starts csv_to_npz, blocks honestly
    _drive(stage, job)
    assert f"{cm.NB}/motions/groove_deploy.csv" in box.pushed
    assert st.meta["phase"] == "convert"
    convert = f"convert-{slug}-{suffix}"
    assert box.jobs[convert]["state"] == "running"
    assert "csv_to_npz" in box.jobs[convert]["script"]

    # 2: still converting -> still blocked (no fake progress)
    _drive(stage, job)

    # 3: conversion done -> training job launched with the promoted recipe
    box.finish(convert)
    _drive(stage, job)
    run_name = f"train-{slug}-{suffix}"
    script = box.jobs[run_name]["script"]
    assert "cloud/train_sim2real.py Mjlab-Tracking-Flat-Unitree-G1-Sim2Real" in script
    assert "--agent.max-iterations 5000" in script
    assert f"--agent.run-name {run_name}" in script
    assert "--env.rewards.motion_global_root_pos.weight 1.0" in script

    # 4: mid-training the blocked message carries real progress from the log
    box.logs[run_name] = ("Learning iteration 800/5000\n"
                          "Mean reward: 21.30\nMean episode length: 19.2\n")
    e = _drive(stage, job)
    assert "800/5000" in str(e)
    assert getattr(e, "retry_after_s", None)     # the poll loop will re-check

    # 5: training done -> checkpoint located -> ONNX export job
    box.finish(run_name)
    box.globs = [("model_", f"{cm.NB}/logs/rsl_rl/g1_tracking/x_{run_name}/model_4999.pt"),
                 ("g1_tracking", f"{cm.NB}/logs/rsl_rl/g1_tracking/x_{run_name}")]
    _drive(stage, job)
    assert st.meta["checkpoint"].endswith("model_4999.pt")
    export = f"export-{slug}-{suffix}"
    assert "export_policy.py" in box.jobs[export]["script"]

    # 6: export done -> artifacts pulled into data/policies/<slug>/ + meta sidecar
    box.finish(export)
    box.files[f"{cm.NB}/exports/app_{slug}_{suffix}/policy.onnx"] = b"ONNX"
    box.files[f"{cm.NB}/motions/groove_deploy.npz"] = b"NPZ"
    _drive(stage, job, expect_blocked=False)
    pol = policies_env / slug
    assert (pol / "policy.onnx").read_bytes() == b"ONNX"
    assert (pol / "groove_deploy.npz").read_bytes() == b"NPZ"
    assert (pol / "groove_deploy.csv").exists()
    meta = json.loads((pol / "policy_meta.json").read_text())
    assert meta["task"] == "Mjlab-Tracking-Flat-Unitree-G1-Sim2Real"
    assert meta["exported_from_checkpoint"].endswith("model_4999.pt")
    # the deploy contract fields ride along from the canonical interface
    assert len(meta["kp_stiffness"]) == 29 and len(meta["joint_order_29dof"]) == 29
    assert st.meta["phase"] == "done"
    assert "poll_after" not in st.meta


def test_train_box_failure_is_honest_and_retryable(jobs_env, tmp_path, box,
                                                   policies_env):
    store, _ = jobs_env
    job = _train_ready_job(store, tmp_path)
    stage = cm.TrainStage()
    _drive(stage, job)                                    # starts convert
    convert = [n for n in box.jobs if n.startswith("convert-")][0]
    box.finish(convert, state="failed")
    box.logs[convert] = "Traceback: cuda out of memory"
    with pytest.raises(RuntimeError, match="out of memory"):
        stage.run(job, lambda p, m: None)
    # the started marker was cleared, so a Retry relaunches the box job
    assert not job.stages["train"].meta.get(f"started:{convert}")


def test_train_needs_prepped_motion(jobs_env, box, policies_env):
    store, _ = jobs_env
    job = store.new_job("nomotion", input={"type": "video", "source": "x.mp4"})
    job.stages["train"].meta["approved"] = {"at": 1.0}
    e = _drive(cm.TrainStage(), job)
    assert "deployable motion" in str(e)


# ---- verify stage ------------------------------------------------------------------

def _heldout_eval(n=256, n_success=256, seed=90001):
    def cond(name, s):
        return {"condition": name, "num_episodes": n, "n_success": n_success,
                "success_rate": n_success / n, "mpkpe_m": 0.17,
                "ee_pos_error_m": 0.10, "seed": s, "push_enabled": name == "push"}
    return {"task": "Mjlab-Tracking-Flat-Unitree-G1", "checkpoint": "ckpt",
            "motion_file": "m.npz",
            "conditions": {"nominal": cond("nominal", seed),
                           "push": cond("push", seed + 1)}}


def _verified_ready_job(store, tmp_path, policies_env, name="groove"):
    """A job whose train stage completed, with real policy artifacts on disk."""
    job = _train_ready_job(store, tmp_path, name=name)
    slug, suffix = cm._slug(name), cm._suffix(job)
    pol = policies_env / slug
    pol.mkdir(parents=True)
    (pol / "policy.onnx").write_bytes(b"ONNX-BYTES")
    (pol / f"{slug}_deploy.csv").write_bytes(b"CSV-BYTES")
    (pol / f"{slug}_deploy.npz").write_bytes(b"NPZ-BYTES")
    (pol / "policy_meta.json").write_text("{}")
    job.stages["train"].state = "done"
    job.stages["train"].meta.update({
        "phase": "done", "policy_dir": str(pol),
        "checkpoint": f"{cm.NB}/logs/x/model_4999.pt",
        "box_npz": f"{cm.NB}/motions/{slug}_deploy.npz",
        "exports": f"{cm.NB}/exports/app_{slug}_{suffix}",
        "run_name": f"train-{slug}-{suffix}",
    })
    # vet report for the dance record
    (job.dir / "retarget").mkdir(exist_ok=True)
    (job.dir / "retarget" / "vet.json").write_text(json.dumps({"pass": True}))
    job.save()
    return job, pol


def test_verify_full_walkthrough_leaves_dance_sim_verified(
        jobs_env, dances_env, tmp_path, box, policies_env):
    store, _ = jobs_env
    shows, _dances_dir = dances_env
    job, pol = _verified_ready_job(store, tmp_path, policies_env)
    slug, suffix = "groove", cm._suffix(job)
    exports = job.stages["train"].meta["exports"]
    stage = cm.VerifyStage()
    st = job.stages["verify"]

    # gap gate: launch -> blocked -> pass -> pulled to job dir + policy dir
    _drive(stage, job)
    gap_job = f"gap-{slug}-{suffix}"
    assert "sim_gap_check.py" in box.jobs[gap_job]["script"]
    box.finish(gap_job)
    box.files[f"{exports}/gap_check.json"] = json.dumps({
        "gate": {"pass": True},
        "conditions": {"nominal": {"success_rate": 1.0,
                                   "ankle_pitch": {"mean_abs": 6.0, "rms_abs": 8.2}}},
    }).encode()
    _drive(stage, job)          # gap done -> exam s1 started -> blocked
    assert st.meta["gap"]["pass"] is True
    assert (pol / "gap_check.json").exists()

    # three held-out exams, sequential, disjoint seeds
    for k, seed in enumerate([90001, 90011, 90021], 1):
        exam = f"exam-{slug}-{suffix}-s{k}"
        assert f"--seed {seed}" in box.jobs[exam]["script"]
        assert "heldout_eval.py Mjlab-Tracking-Flat-Unitree-G1 " in box.jobs[exam]["script"]
        box.finish(exam)
        box.files[f"{exports}/heldout_eval_s{k}.json"] = json.dumps(
            _heldout_eval(seed=seed)).encode()
        blocked = k < 3
        _drive(stage, job, expect_blocked=blocked)

    # ran through: verdicts signed + bound, dance registered and SIM-VERIFIED
    assert st.meta["phase"] == "done"
    from pipeline.exam_verdict import full_sha256, signature_valid
    for k in (1, 2, 3):
        v = json.loads((job.dir / "verify" / f"heldout_verdict_s{k}.json").read_text())
        assert signature_valid(v)
        assert v["verdict"] == "pass"
        assert v["policy_sha256"] == full_sha256(pol / "policy.onnx")
        assert v["motion_sha256"] == full_sha256(pol / "groove_deploy.csv")
        assert (pol / f"heldout_verdict_s{k}.json").exists()

    dance = shows.load_dance(st.meta["dance_id"])
    assert dance.status == "sim-verified"
    assert dance.repeatability["consecutive_clean"] == 3
    assert dance.policy_sha256 == full_sha256(pol / "policy.onnx")
    assert dance.motion_csv.endswith("groove_deploy.csv")   # DEPLOYABLE binding
    assert dance.duration_s == 30.0                          # danced-span seconds
    assert dance.source_job == job.id

    # promotion remains a HUMAN action — but the guarded machinery now allows it
    promoted = shows.promote(dance, "show-ready")
    assert promoted.status == "show-ready"


def test_verify_gap_gate_failure_stops_the_job(jobs_env, dances_env, tmp_path,
                                               box, policies_env):
    store, _ = jobs_env
    job, _pol = _verified_ready_job(store, tmp_path, policies_env)
    slug, suffix = "groove", cm._suffix(job)
    exports = job.stages["train"].meta["exports"]
    stage = cm.VerifyStage()
    _drive(stage, job)
    box.finish(f"gap-{slug}-{suffix}")
    box.files[f"{exports}/gap_check.json"] = json.dumps({
        "gate": {"pass": False},
        "conditions": {"nominal": {"success_rate": 0.82,
                                   "ankle_pitch": {"mean_abs": 11.0, "rms_abs": 14.0}}},
    }).encode()
    with pytest.raises(RuntimeError, match="sim-gap gate FAILED"):
        stage.run(job, lambda p, m: None)
    # no dance was registered off a failed gate
    assert not job.stages["verify"].meta.get("dance_id")


def test_verify_exam_below_bar_fails_honestly(jobs_env, dances_env, tmp_path,
                                              box, policies_env):
    store, _ = jobs_env
    job, _pol = _verified_ready_job(store, tmp_path, policies_env)
    slug, suffix = "groove", cm._suffix(job)
    exports = job.stages["train"].meta["exports"]
    stage = cm.VerifyStage()
    _drive(stage, job)
    box.finish(f"gap-{slug}-{suffix}")
    box.files[f"{exports}/gap_check.json"] = json.dumps(
        {"gate": {"pass": True}, "conditions": {"nominal": {
            "success_rate": 1.0, "ankle_pitch": {"mean_abs": 6, "rms_abs": 8}}}}).encode()
    for k in (1, 2, 3):
        _drive(stage, job)
        exam = f"exam-{slug}-{suffix}-s{k}"
        box.finish(exam)
        # exam 2 comes back at 98% — the real a1 story; must NOT verify
        box.files[f"{exports}/heldout_eval_s{k}.json"] = json.dumps(
            _heldout_eval(n_success=252 if k == 2 else 256)).encode()
    with pytest.raises(RuntimeError, match="99%"):
        stage.run(job, lambda p, m: None)
    assert not job.stages["verify"].meta.get("dance_id")


def test_verify_needs_trained_policy_first(jobs_env, box, policies_env):
    store, _ = jobs_env
    job = store.new_job("x", input={"type": "csv", "source": "x.csv"})
    e = _drive(cm.VerifyStage(), job)
    assert "trained policy" in str(e)


# ---- export stage ------------------------------------------------------------------

def _exported_ready(store, tmp_path, policies_env, dances_shows):
    job, pol = _verified_ready_job(store, tmp_path, policies_env)
    dance = dances_shows.new_dance("groove", duration_s=30.0,
                                   motion_csv=str(pol / "groove_deploy.csv"),
                                   policy_path=str(pol / "policy.onnx"))
    job.stages["verify"].state = "done"
    job.stages["verify"].meta = {"phase": "done", "dance_id": dance.id}
    job.save()
    return job, pol, dance


def test_export_contract_audit_and_audio(jobs_env, dances_env, tmp_path, box,
                                         policies_env, monkeypatch):
    store, _ = jobs_env
    shows, _d = dances_env
    job, pol, dance = _exported_ready(store, tmp_path, policies_env, shows)
    # provide music at data/audio/<slug>/music.wav (AUDIO_DIR is tmp-patched)
    music = cm.AUDIO_DIR / "groove" / "music.wav"
    music.parent.mkdir(parents=True)
    music.write_bytes(b"RIFF")
    import pipeline.audio as audio_mod
    monkeypatch.setattr(audio_mod, "attach_audio_for_dance",
                        lambda d, source_audio=None, **kw: {
                            "track": str(source_audio), "source": "attached_file",
                            "align": {"audio_delay_s": 1.5}, "muxed_preview": None,
                            "attached_at": None})
    _drive(cm.ExportStage(), job, expect_blocked=False)
    dance = shows.load_dance(dance.id)
    assert dance.audio and dance.audio["track"].endswith("music.wav")
    assert dance.audio["align"]["audio_delay_s"] == 1.5     # the 1.5 s lead-in rule
    summary = json.loads((job.dir / "export" / "summary.json").read_text())
    assert summary["dance_id"] == dance.id
    assert any("promote" in s for s in summary["next_human_steps"])


def test_export_fails_if_motion_binding_broken(jobs_env, dances_env, tmp_path,
                                               box, policies_env):
    store, _ = jobs_env
    shows, _d = dances_env
    job, pol, dance = _exported_ready(store, tmp_path, policies_env, shows)
    dance.motion_csv = str(tmp_path / "somewhere_else.csv")   # pre-ramp / wrong file
    dance.save()
    with pytest.raises(RuntimeError, match="deployable"):
        cm.ExportStage().run(job, lambda p, m: None)


def test_export_fails_on_missing_artifacts(jobs_env, dances_env, tmp_path, box,
                                           policies_env):
    store, _ = jobs_env
    shows, _d = dances_env
    job, pol, dance = _exported_ready(store, tmp_path, policies_env, shows)
    (pol / "groove_deploy.npz").unlink()
    with pytest.raises(RuntimeError, match="deploy contract incomplete"):
        cm.ExportStage().run(job, lambda p, m: None)


# ---- extract stage (cloud path) ----------------------------------------------------

def test_extract_video_walkthrough(jobs_env, tmp_path, box, policies_env,
                                   monkeypatch):
    store, _ = jobs_env
    job = store.new_job("groove", input={"type": "video", "source": "clip.mp4"})
    (job.dir / "input.mp4").write_bytes(b"FAKEVIDEO")
    import pipeline.video_probe as vp
    monkeypatch.setattr(vp, "validate", lambda p: {
        "duration_s": 44.3, "width": 1280, "height": 720, "fps": 30.0,
        "advisories": []})
    monkeypatch.setattr(cm, "_reencode_30fps",
                        lambda src, dst: dst.write_bytes(b"CFR30"))
    stage = cm.ExtractStage()
    st = job.stages["extract"]

    _drive(stage, job)                       # validated, re-encoded, pushed, launched
    stem = f"groove_{cm._suffix(job)}"
    assert box.pushed[f"{cm.NB}/videos_in/{stem}.mp4"] == b"CFR30"
    gvhmr = f"gvhmr-{stem}"
    assert "tools/demo/demo.py" in box.jobs[gvhmr]["script"]
    assert box.jobs[gvhmr]["script"].rstrip().splitlines()[2].endswith("-s")

    _drive(stage, job)                       # still running
    box.finish(gvhmr)
    box.files[f"{cm.NB}/artifacts/gvhmr/{stem}/hmr4d_results.pt"] = b"SMPLPRED"
    _drive(stage, job, expect_blocked=False)
    assert (job.dir / "extract" / "hmr4d_results.pt").read_bytes() == b"SMPLPRED"
    assert st.meta["pred"].endswith("hmr4d_results.pt")


def test_extract_skips_csv_jobs(jobs_env, box, policies_env):
    store, _ = jobs_env
    job = store.new_job("csvin", input={"type": "csv", "source": "m.csv"})
    (job.dir / "input.csv").write_text("0,0,0\n")
    from pipeline.stages.base import SkipStage
    with pytest.raises(SkipStage):
        cm.ExtractStage().run(job, lambda p, m: None)


def test_extract_skips_when_retarget_already_done(jobs_env, box, policies_env):
    """A legacy job (retargeted by hand, extract left 'blocked') must NOT burn
    box GPU re-extracting when startup reconciliation re-queues it."""
    store, _ = jobs_env
    job = store.new_job("legacy", input={"type": "video", "source": "x.mp4"})
    (job.dir / "input.mp4").write_bytes(b"VID")
    job.stages["retarget"].state = "done"
    job.save()
    from pipeline.stages.base import SkipStage
    with pytest.raises(SkipStage, match="retarget already complete"):
        cm.ExtractStage().run(job, lambda p, m: None)
    assert not box.jobs and not box.pushed      # nothing touched the box


# ---- retarget prep tail: show csv + deployable csv ---------------------------------

from .conftest import HAVE_MODEL, make_motion  # noqa: E402


@pytest.mark.model
@pytest.mark.skipif(not HAVE_MODEL, reason="needs the MuJoCo G1 model")
def test_retarget_csv_path_produces_deployable_motion(jobs_env, tmp_path,
                                                      monkeypatch):
    """CSV job end to end through the REAL retarget stage: window -> vet ->
    preview -> prep_motion -> activation ramp, ending at <slug>_deploy.csv
    (what TrainStage pushes)."""
    from pipeline.deploy_ramp import RAMP_FRAMES, default_joint_pos
    from pipeline.stages import local_motion as lm
    monkeypatch.setattr(lm, "PREVIEWS_DIR", tmp_path / "previews")
    store, _ = jobs_env
    job = store.new_job("stand still", input={"type": "csv", "source": "m.csv"})
    np.savetxt(job.dir / "input.csv", make_motion(frames=60), delimiter=",")

    lm.RetargetStage().run(job, lambda p, m: None)

    st = job.stages["retarget"]
    assert st.meta["vet_pass"] is True
    show = Path(st.meta["show_csv"])
    deploy = Path(st.meta["deploy_csv"])
    assert show.name == "stand_still_show.csv" and show.exists()
    assert deploy.name == "stand_still_deploy.csv" and deploy.exists()
    dep = np.loadtxt(deploy, delimiter=",")
    shw = np.loadtxt(show, delimiter=",")
    assert dep.shape[0] == shw.shape[0] + RAMP_FRAMES
    # zero activation lurch: deploy frame 0 == the policy's standby pose
    np.testing.assert_allclose(dep[0, 7:], default_joint_pos(), atol=1e-9)
    assert st.meta["ramp"]["frame0_max_delta_rad"] == 0.0


# ---- blocked-with-poll plumbing (runner + server poll loop) ------------------------

def _full_stages(train_stage):
    class Dummy:
        def __init__(self, name):
            self.name = name

        def run(self, job, report):
            raise StageBlocked("dummy")
    return {name: (train_stage if name == "train" else Dummy(name))
            for name in STAGE_ORDER}


def test_runner_stamps_poll_after_for_retryable_blocks(jobs_env, tmp_path, box,
                                                       policies_env):
    store, _ = jobs_env
    job = _train_ready_job(store, tmp_path)
    job.stages["extract"].state = "skipped"
    job.save()
    Runner(_full_stages(cm.TrainStage())).run_job(job)
    st = store.load_job(job.id).stages["train"]
    assert st.state == "blocked"
    assert st.meta["poll_after"] > 0        # the poll loop will re-queue this job


def test_server_poll_loop_requeues_due_jobs(client, tmp_path):
    c, server = client
    from pipeline import store
    job = store.new_job("pollme", input={"type": "video", "source": "x"})
    job.stages["extract"].state = "blocked"
    job.stages["extract"].meta["poll_after"] = 1.0   # long past due
    job.save()
    # drain anything queued by startup reconciliation first
    while not server._job_queue.empty():
        server._job_queue.get_nowait()
    requeued = server._poll_cloud_jobs_once(now=100.0)
    assert job.id in requeued
    # not due yet -> untouched
    job2 = store.new_job("notyet", input={"type": "video", "source": "x"})
    job2.stages["extract"].state = "blocked"
    job2.stages["extract"].meta["poll_after"] = 1e12
    job2.save()
    assert job2.id not in server._poll_cloud_jobs_once(now=100.0)


def test_approve_train_endpoint(client, tmp_path):
    c, server = client
    from pipeline import store
    job = store.new_job("appr", input={"type": "video", "source": "x"})
    # not ready: retarget incomplete
    r = c.post(f"/api/jobs/{job.id}/approve-train")
    assert r.status_code == 400
    job.stages["retarget"].state = "done"
    job.save()
    r = c.post(f"/api/jobs/{job.id}/approve-train")
    assert r.status_code == 200
    assert store.load_job(job.id).stages["train"].meta["approved"]
    assert c.post("/api/jobs/nope/approve-train").status_code == 404
