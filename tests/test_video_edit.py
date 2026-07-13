"""Tests for the upload trimmer's video ops (uses the repo's sample clip; ffmpeg via imageio)."""
import pytest
from pathlib import Path
from pipeline import video_edit

VID = Path("data/videos/thriller_30fps.mp4")
needs_vid = pytest.mark.skipif(not VID.exists(), reason="sample video absent")


def test_constants():
    assert video_edit.MAX_UNTRIMMED_S == 240 and video_edit.MIN_SEGMENT_S >= 1


@needs_vid
def test_probe_duration():
    assert 35 < video_edit.probe_duration(VID) < 60


@needs_vid
def test_extract_frame(tmp_path):
    out = tmp_path / "f.jpg"
    assert video_edit.extract_frame(VID, 20.0, out) and out.stat().st_size > 1500


@needs_vid
def test_trim_produces_segment(tmp_path):
    out = tmp_path / "t.mp4"
    video_edit.trim(VID, out, 10.0, 15.0)
    assert out.exists() and 12 < video_edit.probe_duration(out) < 20   # ~15s, keyframe-aligned
