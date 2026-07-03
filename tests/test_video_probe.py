"""Video intake validation: decision logic on synthetic metadata, plus one
real ffprobe round-trip per outcome class."""
import shutil
import subprocess

import pytest

from pipeline import video_probe

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def fake_meta(duration=30.0, width=1920, height=1080,
              r_fps="30/1", avg_fps="30/1", with_video=True):
    streams = []
    if with_video:
        streams.append({"codec_type": "video", "codec_name": "h264",
                        "width": width, "height": height,
                        "r_frame_rate": r_fps, "avg_frame_rate": avg_fps})
    return {"format": {"duration": str(duration)}, "streams": streams}


@pytest.fixture
def probed(monkeypatch, tmp_path):
    """validate() against synthetic ffprobe output; returns a runner."""
    target = tmp_path / "v.mp4"
    target.write_bytes(b"x" * 1000)

    def _run(**meta_kwargs):
        monkeypatch.setattr(video_probe, "probe",
                            lambda path: fake_meta(**meta_kwargs))
        return video_probe.validate(target)
    return _run


def test_happy_path_metadata(probed):
    got = probed(duration=44.28, width=1498, height=1392,
                 r_fps="30/1", avg_fps="30/1")
    assert got["duration_s"] == 44.28
    assert (got["width"], got["height"]) == (1498, 1392)
    assert got["fps"] == 30.0
    assert got["codec"] == "h264"
    assert got["advisories"] == []


def test_no_video_stream_rejected(probed):
    with pytest.raises(RuntimeError, match="no video stream"):
        probed(with_video=False)


def test_too_short_rejected(probed):
    with pytest.raises(RuntimeError, match="too short"):
        probed(duration=video_probe.MIN_SECONDS - 0.5)


def test_too_long_rejected(probed):
    with pytest.raises(RuntimeError, match="longer than"):
        probed(duration=video_probe.MAX_SECONDS + 1)


def test_boundary_durations_accepted(probed):
    assert probed(duration=video_probe.MIN_SECONDS)["advisories"] == []
    assert probed(duration=video_probe.MAX_SECONDS)["advisories"] == []


def test_low_resolution_advisory(probed):
    got = probed(width=640, height=480)
    assert any("below 720p" in a for a in got["advisories"])


def test_720p_no_resolution_advisory(probed):
    got = probed(width=1280, height=720)
    assert not any("720p" in a for a in got["advisories"])


def test_vfr_advisory(probed):
    # declared 30 fps vs measured ~35.4 fps (the Thriller .mov pattern)
    got = probed(r_fps="30/1", avg_fps="47160/1331")
    assert any("variable frame rate" in a for a in got["advisories"])


def test_cfr_no_vfr_advisory(probed):
    got = probed(r_fps="30000/1001", avg_fps="30000/1001")
    assert not any("variable frame rate" in a for a in got["advisories"])


# ---- real ffprobe round-trips ----------------------------------------------------

pytestmark_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="no ffmpeg")


@pytest.mark.ffmpeg
@pytest.mark.skipif(not HAVE_FFMPEG, reason="no ffmpeg")
def test_real_clip_probes_and_validates(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=duration=16:size=320x240:rate=30",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True, capture_output=True, timeout=120)
    got = video_probe.validate(clip)
    assert got["duration_s"] == pytest.approx(16.0, abs=0.2)
    assert got["fps"] == pytest.approx(30.0, abs=0.1)
    assert any("below 720p" in a for a in got["advisories"])


@pytest.mark.ffmpeg
@pytest.mark.skipif(not HAVE_FFMPEG, reason="no ffmpeg")
def test_garbage_file_rejected(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video at all" * 100)
    with pytest.raises(RuntimeError):
        video_probe.validate(junk)
