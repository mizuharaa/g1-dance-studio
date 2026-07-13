"""Tests for on-demand mp4->webm transcode (the desktop QtWebEngine H.264 workaround).
The ffmpeg transcode itself is monkeypatched — no real encoding in the suite."""
import time
from pathlib import Path
import pytest
from pipeline import video_web


def test_webm_sibling():
    assert video_web.webm_sibling(Path("/a/b/c.mp4")) == Path("/a/b/c.webm")


def test_missing_source(tmp_path):
    assert video_web.ensure_webm(tmp_path / "nope.mp4") == {"ready": False, "status": "source-missing"}


def test_fresh_webm_is_ready(tmp_path):
    mp4 = tmp_path / "v.mp4"; mp4.write_bytes(b"x")
    webm = tmp_path / "v.webm"; webm.write_bytes(b"y")   # newer than mp4
    assert video_web.ensure_webm(mp4) == {"ready": True}


def test_transcode_kicked_off_then_ready(tmp_path, monkeypatch):
    mp4 = tmp_path / "v.mp4"; mp4.write_bytes(b"x")
    # fake transcode: just create the webm (no ffmpeg)
    monkeypatch.setattr(video_web, "_transcode", lambda m, w: Path(w).write_bytes(b"webm"))
    first = video_web.ensure_webm(mp4)
    assert first["ready"] is False and first["status"] == "running"
    for _ in range(40):
        r = video_web.ensure_webm(mp4)
        if r.get("ready"):
            break
        time.sleep(0.05)
    assert r == {"ready": True} and (tmp_path / "v.webm").exists()
