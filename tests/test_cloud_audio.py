"""Audio-through-the-pipeline + stand-end tests for the cloud stages.

Covers the LANE 1 goal wiring in pipeline/stages/cloud_motion.py:

  * ExtractStage captures the SOURCE video's own soundtrack to the job dir
    (silent sources / CSV inputs stay silent, no crash);
  * ExportStage windows that captured audio to the danced span, writing
    data/audio/<slug>/music.wav that _find_music + attach_audio_for_dance use;
  * TrainStage rebuilds the deployable WITH the return-to-standing tail so the
    dance ENDS STANDING (final frame at the standby pose).

Everything is mocked at the same seams the rest of the suite uses: the box thin
helpers (no SSH/GPU) and the ffmpeg-shelling audio helpers (ffmpeg need not be
installed). No real data/audio/<real dance> path is written — AUDIO_DIR is
tmp-patched. make_deploy_csv is exercised for real (pure numpy) so the
stand-end assertion is a genuine end-state check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline import audio as audio_mod
from pipeline.stages import cloud_motion as cm
from pipeline.stages.base import StageBlocked


# ---- fake box (box thin-helper seam) ----------------------------------------------

class FakeBox:
    """Minimal stand-in for the GPU box at the helpers cloud_motion calls."""

    def __init__(self, monkeypatch):
        self.jobs: dict[str, dict] = {}
        self.pushed: dict[str, bytes] = {}
        self.files: dict[str, bytes] = {}
        monkeypatch.setattr(cm, "_require_cloud", lambda: None)
        monkeypatch.setattr(cm, "_start_script_job", self.start)
        monkeypatch.setattr(cm, "_job_status", self.status)
        monkeypatch.setattr(cm, "_log_tail", lambda name, n=80: "")
        monkeypatch.setattr(cm, "_push", self.push)
        monkeypatch.setattr(cm, "_pull", self.pull)
        monkeypatch.setattr(cm, "_remote_first", lambda expr: None)

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
            raise RuntimeError(f"scp failed: no such remote file {remote}")
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(self.files[remote])
        return local


@pytest.fixture
def box(monkeypatch):
    return FakeBox(monkeypatch)


@pytest.fixture
def audio_env(tmp_path, monkeypatch):
    """Isolate AUDIO_DIR / POLICIES_DIR so tests never touch real data/audio."""
    monkeypatch.setattr(cm, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(cm, "POLICIES_DIR", tmp_path / "policies")
    return tmp_path


def _drive(stage, job, expect_blocked=True):
    try:
        stage.run(job, lambda p, m: None)
    except StageBlocked as e:
        assert expect_blocked, f"unexpected block: {e}"
        return e
    assert not expect_blocked, "expected StageBlocked but the stage completed"
    return None


# ---- ExtractStage: capture the source soundtrack -----------------------------------

def _video_job(store, name="groove"):
    job = store.new_job(name, input={"type": "video", "source": "clip.mp4"})
    (job.dir / "input.mp4").write_bytes(b"FAKEVIDEO")
    return job


def _mock_extract_ok(monkeypatch, has_audio=True):
    """Mock the intake + ffmpeg seams ExtractStage uses (no real ffmpeg/GVHMR)."""
    import pipeline.video_probe as vp
    monkeypatch.setattr(vp, "validate", lambda p: {
        "duration_s": 44.3, "width": 1280, "height": 720, "fps": 30.0,
        "advisories": []})
    monkeypatch.setattr(cm, "_reencode_30fps",
                        lambda src, dst: dst.write_bytes(b"CFR30"))
    monkeypatch.setattr(audio_mod, "has_audio", lambda v: has_audio)

    def fake_extract(video, out_wav):
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        out_wav.write_bytes(b"RIFFsourceaudio")
        return out_wav

    monkeypatch.setattr(audio_mod, "extract_audio", fake_extract)


def test_extract_captures_source_audio_when_present(jobs_env, tmp_path, box,
                                                    audio_env, monkeypatch):
    store, _ = jobs_env
    job = _video_job(store)
    _mock_extract_ok(monkeypatch, has_audio=True)

    _drive(cm.ExtractStage(), job)          # runs through capture, then blocks on GVHMR

    src = job.stages["extract"].meta["source_audio"]
    assert src is not None and src.endswith("source_audio.wav")
    assert Path(src).exists()
    assert (job.dir / "source_audio.wav").read_bytes() == b"RIFFsourceaudio"
    # the 30 fps GVHMR clip is still the silent re-encode (audio kept separate)
    stem = f"groove_{cm._suffix(job)}"
    assert box.pushed[f"{cm.NB}/videos_in/{stem}.mp4"] == b"CFR30"


def test_extract_silent_video_stays_silent_no_crash(jobs_env, tmp_path, box,
                                                    audio_env, monkeypatch):
    store, _ = jobs_env
    job = _video_job(store, name="silent")
    _mock_extract_ok(monkeypatch, has_audio=False)

    _drive(cm.ExtractStage(), job)          # must not raise on a silent source

    assert job.stages["extract"].meta["source_audio"] is None
    assert not (job.dir / "source_audio.wav").exists()


def test_extract_audio_failure_is_non_fatal(jobs_env, tmp_path, box, audio_env,
                                            monkeypatch):
    """A missing/failing ffprobe (audio helper raises) must not fail extraction —
    the dance just stays silent (this is what happens with no ffmpeg installed)."""
    store, _ = jobs_env
    job = _video_job(store, name="noffmpeg")
    _mock_extract_ok(monkeypatch, has_audio=True)

    def boom(v):
        raise FileNotFoundError("ffprobe: command not found")

    monkeypatch.setattr(audio_mod, "has_audio", boom)

    _drive(cm.ExtractStage(), job)          # extraction still proceeds to GVHMR

    assert job.stages["extract"].meta["source_audio"] is None
    stem = f"noffmpeg_{cm._suffix(job)}"
    assert box.pushed[f"{cm.NB}/videos_in/{stem}.mp4"] == b"CFR30"   # GVHMR clip pushed


# ---- windowed music: trim the captured audio to the danced span --------------------

def _seed_windowed(job, tmp_path, *, start_frame, seconds):
    """Put a captured source audio + a danced window on a job (as extract/retarget
    would). Returns the fake source-audio path."""
    src = tmp_path / "source_audio.wav"
    src.write_bytes(b"RIFFsourceaudio" * 100)
    job.stages["extract"].meta["source_audio"] = str(src)
    job.stages["retarget"].meta["window"] = {
        "start_frame": start_frame,
        "end_frame": start_frame + int(round(seconds * 30)) - 1,
        "seconds": seconds, "input_seconds": seconds + 20.0}
    job.save()
    return src


def _capture_trim(monkeypatch):
    """Replace the ffmpeg trim with a recorder that writes a stub wav."""
    calls: list[dict] = []

    def fake_trim(src, start_s, duration_s, out):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"TRIMMED")
        calls.append({"src": str(src), "start": start_s,
                      "duration": duration_s, "out": str(out)})
        return out

    monkeypatch.setattr(cm, "_trim_audio", fake_trim)
    return calls


def test_windowed_music_trims_to_danced_span(jobs_env, tmp_path, audio_env,
                                             monkeypatch):
    store, _ = jobs_env
    job = store.new_job("groove", input={"type": "video", "source": "clip.mp4"})
    # danced window starts 10 s (frame 300) into the source, lasts 20 s
    _seed_windowed(job, tmp_path, start_frame=300, seconds=20.0)
    calls = _capture_trim(monkeypatch)

    out = cm._prepare_windowed_music(job, "groove")

    assert out is not None
    music = cm.AUDIO_DIR / "groove" / "music.wav"
    assert music.exists() and out == music
    # the offset is real: the talking-intro frames are trimmed off
    assert len(calls) == 1
    assert calls[0]["start"] == pytest.approx(10.0, abs=1e-6)   # 300 / 30 fps
    assert calls[0]["duration"] == pytest.approx(20.0, abs=1e-6)   # danced span
    # _find_music now resolves it (the path/ext ExportStage reads)
    assert cm._find_music("groove") == music


def test_windowed_music_noop_without_source_audio(jobs_env, tmp_path, audio_env,
                                                  monkeypatch):
    """CSV input / silent video: no captured audio -> no music, no crash."""
    store, _ = jobs_env
    job = store.new_job("csvin", input={"type": "csv", "source": "m.csv"})
    job.stages["retarget"].meta["window"] = {
        "start_frame": 0, "end_frame": 599, "seconds": 20.0}
    calls = _capture_trim(monkeypatch)

    assert cm._prepare_windowed_music(job, "csvin") is None
    assert not calls                                   # ffmpeg never invoked
    assert cm._find_music("csvin") is None             # no music produced


def test_windowed_music_does_not_clobber_real_music(jobs_env, tmp_path, audio_env,
                                                    monkeypatch):
    """An operator-supplied real music file already in place is left untouched."""
    store, _ = jobs_env
    job = store.new_job("groove", input={"type": "video", "source": "clip.mp4"})
    _seed_windowed(job, tmp_path, start_frame=0, seconds=20.0)
    real = cm.AUDIO_DIR / "groove" / "music.mp3"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"REAL-MP3")
    calls = _capture_trim(monkeypatch)

    out = cm._prepare_windowed_music(job, "groove")

    assert out == real                                 # returns the existing file
    assert not calls                                   # did NOT re-trim/overwrite
    assert not (cm.AUDIO_DIR / "groove" / "music.wav").exists()


def test_windowed_music_seconds_missing_falls_back_to_frames(jobs_env, tmp_path,
                                                             audio_env, monkeypatch):
    store, _ = jobs_env
    job = store.new_job("groove", input={"type": "video", "source": "clip.mp4"})
    src = tmp_path / "source_audio.wav"
    src.write_bytes(b"RIFF")
    job.stages["extract"].meta["source_audio"] = str(src)
    job.stages["retarget"].meta["window"] = {"start_frame": 60, "end_frame": 659}
    calls = _capture_trim(monkeypatch)

    cm._prepare_windowed_music(job, "groove")

    assert calls[0]["start"] == pytest.approx(2.0, abs=1e-6)     # 60 / 30
    assert calls[0]["duration"] == pytest.approx(20.0, abs=1e-6)  # (659-60+1)/30


# ---- ExportStage end-to-end: source audio -> windowed music.wav -> attach ----------

def _verified_export_job(store, shows, tmp_path, audio_env, *, name="groove",
                         start_frame=300, seconds=20.0):
    """A job at the export gate (verify done, dance bound) with a captured source
    soundtrack + danced window ready to window into music.wav."""
    slug = cm._slug(name)
    pol = cm.POLICIES_DIR / slug
    pol.mkdir(parents=True)
    for n in ("policy.onnx", f"{slug}_deploy.csv", f"{slug}_deploy.npz"):
        (pol / n).write_bytes(b"X")
    (pol / "policy_meta.json").write_text("{}")

    job = store.new_job(name, input={"type": "video", "source": "clip.mp4"})
    _seed_windowed(job, tmp_path, start_frame=start_frame, seconds=seconds)
    job.stages["train"].meta = {"phase": "done", "policy_dir": str(pol)}

    dance = shows.new_dance(name, duration_s=seconds,
                            motion_csv=str(pol / f"{slug}_deploy.csv"),
                            policy_path=str(pol / "policy.onnx"))
    job.stages["verify"].meta = {"phase": "done", "dance_id": dance.id}
    job.save()
    return job, pol, dance


def test_export_windows_source_audio_and_attaches(jobs_env, dances_env, tmp_path,
                                                  audio_env, monkeypatch):
    store, _ = jobs_env
    shows, _d = dances_env
    job, pol, dance = _verified_export_job(store, shows, tmp_path, audio_env)
    calls = _capture_trim(monkeypatch)
    # attach is exercised at its own seam elsewhere; here assert it receives the
    # windowed music.wav we produced (attach itself shells ffmpeg, so mock it).
    monkeypatch.setattr(audio_mod, "attach_audio_for_dance",
                        lambda d, source_audio=None, **kw: {
                            "track": str(source_audio), "source": "attached_file",
                            "align": {"audio_delay_s": 1.5}, "muxed_preview": None,
                            "attached_at": None})

    _drive(cm.ExportStage(), job, expect_blocked=False)

    music = cm.AUDIO_DIR / "groove" / "music.wav"
    assert music.exists()                               # windowed music materialized
    assert len(calls) == 1 and calls[0]["duration"] == pytest.approx(20.0)
    dance = shows.load_dance(dance.id)
    assert dance.audio and dance.audio["track"].endswith("music.wav")
    assert dance.audio["align"]["audio_delay_s"] == 1.5   # export adds the lead-in


def test_export_silent_source_leaves_dance_silent(jobs_env, dances_env, tmp_path,
                                                  audio_env, monkeypatch):
    store, _ = jobs_env
    shows, _d = dances_env
    job, pol, dance = _verified_export_job(store, shows, tmp_path, audio_env)
    # no captured source audio (silent video / CSV): drop the meta
    job.stages["extract"].meta["source_audio"] = None
    job.save()
    calls = _capture_trim(monkeypatch)

    _drive(cm.ExportStage(), job, expect_blocked=False)

    assert not calls
    assert not (cm.AUDIO_DIR / "groove" / "music.wav").exists()
    dance = shows.load_dance(dance.id)
    assert not dance.audio                              # stays silent, no crash


# ---- stand-end: the deployable ends at the standby pose ----------------------------

def _show_csv(path: Path, frames: int = 30) -> Path:
    """A 36-col show motion whose joints are clearly OFF the standby pose, so the
    return-to-standing tail has real work to do (its final frame must land at dj)."""
    m = np.zeros((frames, 36))
    m[:, 2] = 0.75              # standing root height
    m[:, 6] = 1.0               # quat w (xyzw)
    m[:, 7:] = 0.4              # every joint 0.4 rad away from a zero-ish standby
    np.savetxt(path, m, delimiter=",", fmt="%.6f")
    return path


def _train_ready_standend(store, tmp_path, name="groove"):
    job = store.new_job(name, input={"type": "video", "source": "clip.mp4"})
    show = _show_csv(tmp_path / "groove_show.csv")
    # a plausible retarget deploy csv (guard only checks it exists)
    deploy = tmp_path / "groove_deploy.csv"
    deploy.write_text("0.0," * 35 + "1.0\n")
    job.stages["extract"].state = "done"
    job.stages["retarget"].state = "done"
    job.stages["retarget"].meta = {
        "deploy_csv": str(deploy), "show_csv": str(show),
        "window": {"start_frame": 0, "end_frame": 599, "seconds": 20.0,
                   "input_seconds": 22.0}}
    job.stages["train"].meta["approved"] = {"at": 1.0, "by": "test"}
    job.save()
    return job


def test_train_deploy_csv_ends_standing(jobs_env, tmp_path, box, audio_env):
    from pipeline.deploy_ramp import default_joint_pos
    store, _ = jobs_env
    job = _train_ready_standend(store, tmp_path)
    st = job.stages["train"]

    _drive(cm.TrainStage(), job)            # rebuilds stand-end, pushes, blocks on convert

    assert st.meta.get("stand_end") is True
    assert st.meta["deploy_ramp"]["stand_end"] is True
    # the rebuilt deployable that gets pushed/trained ends AT the standby pose
    standend = job.stage_dir("train") / "groove_deploy.csv"
    assert standend.exists()
    dep = np.loadtxt(standend, delimiter=",")
    dj = default_joint_pos()
    final_delta = np.abs(dep[-1, 7:] - dj).max()
    assert final_delta < 0.15                          # ends within 0.15 rad of default
    # and it is genuinely LONGER than the raw show (activation + landing tails added)
    shw = np.loadtxt(_show_csv(tmp_path / "chk_show.csv"), delimiter=",")
    assert dep.shape[0] > shw.shape[0]
    # the version pushed to the box is the stand-end one
    box_csv = f"{cm.NB}/motions/groove_deploy.csv"
    assert box_csv in box.pushed
    assert box.pushed[box_csv] == standend.read_bytes()


def test_train_without_show_csv_leaves_deploy_unchanged(jobs_env, tmp_path, box,
                                                        audio_env):
    """Back-compat: a job with no show CSV on record (e.g. hand-retargeted legacy)
    trains the retarget deploy CSV as-is (no stand-end rebuild, no crash)."""
    store, _ = jobs_env
    job = store.new_job("legacy", input={"type": "video", "source": "clip.mp4"})
    deploy = tmp_path / "legacy_deploy.csv"
    deploy.write_text("0.0," * 35 + "1.0\n")
    job.stages["retarget"].meta = {
        "deploy_csv": str(deploy),
        "window": {"start_frame": 0, "end_frame": 599, "seconds": 20.0}}
    job.stages["train"].meta["approved"] = {"at": 1.0}
    job.save()

    _drive(cm.TrainStage(), job)

    assert job.stages["train"].meta.get("stand_end") is None   # no rebuild
    box_csv = f"{cm.NB}/motions/legacy_deploy.csv"
    assert box.pushed[box_csv] == deploy.read_bytes()          # original pushed
