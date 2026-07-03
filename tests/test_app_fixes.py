"""Regression tests for the app quality-audit fixes (docs/app_fixes.md).

Each test pins one confirmed finding closed. All headless — no robot/cloud/GPU.
The MuJoCo-dependent grounding tests self-skip when the model is absent.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from .conftest import HAVE_MODEL, make_motion

from pipeline import library, monitor, motion_io, shows, store, video_probe


# ---- monitor: parse real ANSI-coloured training logs (finding #7) ----------------

def test_monitor_parses_ansi_training_log():
    ansi = ("\x1b[1m        Learning iteration 1382/30000        \x1b[0m\n"
            "            Mean reward: 13.25\n"
            "            Mean episode length: 382.66\n"
            "W&B: https://wandb.ai/luong-alois-vng-group/mjlab/runs/40g4byo3")
    info = monitor.parse_job_log("train-dance1-seg", ansi)
    assert info["iteration"] == 1382 and info["max_iteration"] == 30000
    assert info["mean_reward"] == 13.25
    assert info["mean_episode_length"] == 382.66
    assert info["progress"] == round(1382 / 30000, 4)
    assert info["wandb_url"].endswith("40g4byo3")


# ---- motion_io: shape validation (no more cryptic tracebacks) --------------------

def test_motion_io_accepts_valid_36col(motion_csv):
    p = motion_csv(make_motion(frames=10))
    m = motion_io.load_motion_csv(p)
    assert m.shape == (10, 36)


def test_motion_io_rejects_wrong_column_count(tmp_path):
    bad = tmp_path / "bad.csv"
    np.savetxt(bad, np.zeros((5, 10)), delimiter=",")
    with pytest.raises(RuntimeError, match="columns"):
        motion_io.load_motion_csv(bad)


def test_motion_io_rejects_header_row(tmp_path):
    bad = tmp_path / "hdr.csv"
    bad.write_text("x,y,z\n1,2,3\n")
    with pytest.raises(RuntimeError):
        motion_io.load_motion_csv(bad)


def test_motion_io_rejects_nan(tmp_path):
    m = make_motion(frames=4)
    m[2, 5] = np.nan
    bad = tmp_path / "nan.csv"
    np.savetxt(bad, m, delimiter=",")
    with pytest.raises(RuntimeError, match="non-finite"):
        motion_io.load_motion_csv(bad)


def test_motion_io_single_row_ok(tmp_path):
    p = tmp_path / "one.csv"
    np.savetxt(p, make_motion(frames=1), delimiter=",")
    assert motion_io.load_motion_csv(p).shape == (1, 36)


# ---- grounding: absolute-z gate is meaningful only after grounding ---------------

@pytest.mark.skipif(not HAVE_MODEL, reason="needs mujoco G1 model")
def test_grounding_is_idempotent_and_zeros_contact():
    from pipeline import grounding
    m = make_motion(frames=8)
    g1, shift1 = grounding.ground_motion(m)
    # lowest contact of a grounded motion sits on the floor
    assert abs(grounding.min_contact_height(g1)) < 1e-6
    # re-grounding shifts by ~0 (idempotent)
    _, shift2 = grounding.ground_motion(g1)
    assert abs(shift2) < 1e-6


@pytest.mark.skipif(not HAVE_MODEL, reason="needs mujoco G1 model")
def test_ungrounded_translation_does_not_change_vet_verdict(tmp_path, motion_csv):
    """A standing motion translated far below the floor must NOT read as floorwork
    once grounded — the whole audit HIGH: absolute-z tests need grounding first."""
    from .conftest import run_vet
    base = make_motion(frames=20)
    low = base.copy()
    low[:, 2] -= 5.0                      # shove the whole pose 5 m underground
    rc_base, rep_base = run_vet(motion_csv(base, name="base.csv"))
    rc_low, rep_low = run_vet(motion_csv(low, name="low.csv"))
    # grounding makes the two identical: same pelvis verdict, both PASS HARD-3
    assert rep_base["hard"]["pelvis_height"]["pass"]
    assert rep_low["hard"]["pelvis_height"]["pass"]
    assert rc_base == rc_low == 0


# ---- store: one corrupt job.json can't brick startup / the job list --------------

def test_list_jobs_skips_corrupt(jobs_env):
    store_mod, jobs_dir = jobs_env
    good = store_mod.new_job("good", input={"type": "csv"})
    bad_dir = jobs_dir / "20260101-000000-badbad"
    bad_dir.mkdir()
    (bad_dir / "job.json").write_text("{ this is not json")
    jobs = store_mod.list_jobs()
    ids = [j.id for j in jobs]
    assert good.id in ids
    assert not any("badbad" in i for i in ids)  # corrupt one skipped, no crash


def test_load_job_raises_corrupt(jobs_env):
    store_mod, jobs_dir = jobs_env
    d = jobs_dir / "20260101-000000-xyzxyz"
    d.mkdir()
    (d / "job.json").write_text("")  # empty/truncated
    with pytest.raises(store_mod.CorruptJobError):
        store_mod.load_job("20260101-000000-xyzxyz")


def test_job_save_roundtrip_durable(jobs_env):
    store_mod, _ = jobs_env
    job = store_mod.new_job("dur", input={"type": "csv"})
    job.stages["retarget"].state = "done"
    job.save()
    reloaded = store_mod.load_job(job.id)
    assert reloaded.stages["retarget"].state == "done"


# ---- video_probe: geometry hard-reject + either-dim advisory ---------------------

def _validate(monkeypatch, tmp_path, **meta):
    from tests.test_video_probe import fake_meta
    target = tmp_path / "v.mp4"
    target.write_bytes(b"x" * 100)
    monkeypatch.setattr(video_probe, "probe", lambda path: fake_meta(**meta))
    return video_probe.validate(target)


def test_extreme_aspect_rejected(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="aspect"):
        _validate(monkeypatch, tmp_path, width=4000, height=100)


def test_zero_dimension_rejected(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="dimensions"):
        _validate(monkeypatch, tmp_path, width=0, height=0)


def test_single_small_dimension_advises(monkeypatch, tmp_path):
    # 960x540 is a sane 16:9 aspect but below 720p → advisory (old AND-logic missed
    # the width<1280 case when height happened to be >=720; here height<720 fires it)
    got = _validate(monkeypatch, tmp_path, width=960, height=540)
    assert any("below 720p" in a for a in got["advisories"])


def test_zero_duration_clear_message(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="duration"):
        _validate(monkeypatch, tmp_path, duration=0)


# ---- shows.attach_policy: fills the register-first workflow gap -------------------

def test_attach_policy_sets_and_resets_verification(dances_env, tmp_path):
    d = shows.new_dance("thriller", motion_csv="data/x.csv")
    # pretend it had a prior (now-stale) verification
    d.status = "sim-verified"
    d.policy_sha256 = "deadbeef"
    d.sim_exam = {"verdict": "pass"}
    d.repeatability["consecutive_clean"] = 3
    d.save()
    pol = tmp_path / "policy.onnx"
    pol.write_bytes(b"onnx")
    out = shows.attach_policy(d.id, str(pol))
    assert out.policy_path == str(pol)
    assert out.status == "draft"          # re-exam required
    assert out.policy_sha256 is None
    assert out.sim_exam is None
    assert out.repeatability["consecutive_clean"] == 0


def test_attach_policy_missing_file_rejected(dances_env):
    d = shows.new_dance("nope")
    with pytest.raises(ValueError, match="not found"):
        shows.attach_policy(d.id, "/no/such/policy.onnx")


# ---- library export/import round-trip (disaster recovery) ------------------------

def test_library_export_import_roundtrip(dances_env, tmp_path):
    _shows, dances_dir = dances_env
    motion = tmp_path / "m.csv"
    np.savetxt(motion, make_motion(frames=3), delimiter=",")
    d = shows.new_dance("backup-me", motion_csv=str(motion), duration_s=1.0)
    archive = library.export_library()
    assert archive.is_file()
    # wipe the library, then restore from the archive
    import shutil as _sh
    _sh.rmtree(d.dir)
    assert not shows.list_dances()
    ids = library.import_library(archive)
    assert d.id in ids
    restored = shows.load_dance(d.id)
    assert restored.name == "backup-me"
    # the motion file came back and is bundled under the dance dir
    assert restored.motion_csv and shows._abs(restored.motion_csv).is_file()
