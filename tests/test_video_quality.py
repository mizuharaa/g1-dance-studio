"""Tests for the upload-time video quality rubric (pure scoring — no video needed)."""
from pipeline import video_quality as vq


def test_framerate_scoring():
    assert vq.score_framerate(30)["score"] == 10.0
    assert vq.score_framerate(15)["score"] <= 2.0
    odd = vq.score_framerate(35.4)          # great fps but VFR-risk -> a FLAG, not a low score
    assert odd["score"] == 10.0 and odd["flag"]


def test_resolution_scoring():
    assert vq.score_resolution(1920, 1080)["score"] == 10.0
    low = vq.score_resolution(320, 240)
    assert low["score"] < 4 and low["flag"]


def test_summarize_flag_does_not_demote_verdict():
    # a strong clip whose only issue is an advisory FLAG must still read 'good'
    dims = {
        "framerate": {"score": 10, "note": "ok", "flag": "odd rate"},
        "resolution": {"score": 10, "note": "ok"},
        "lighting": {"score": 8, "note": "ok"},
        "sharpness_snappy": {"score": 9, "note": "ok"},
        "movement_feasibility": {"score": 7, "note": "ok"},
    }
    s = vq.summarize(dims, {"score": 5})
    assert s["verdict"] == "good" and not s["blockers"] and s["flags"]


def test_summarize_low_score_blocks():
    dims = {k: {"score": 2, "note": "bad"} for k in
            ["framerate", "resolution", "lighting", "sharpness_snappy", "movement_feasibility"]}
    s = vq.summarize(dims, {"score": 5})
    assert s["verdict"] == "poor" and s["blockers"]


def test_difficulty_band():
    d = vq.score_difficulty(move_pct=1, burstiness=0.1, move_range=1)   # slow, simple
    assert d["score"] < 4 and d["value"] == "gentle"
