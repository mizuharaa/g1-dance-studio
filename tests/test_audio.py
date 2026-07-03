"""Tests for pipeline/audio.py alignment math (the load-bearing, pure part).

The mux/extract paths shell out to ffmpeg and are exercised by the thriller-demo
integration run, not here; these tests pin the timing arithmetic that keeps the
music locked to the motion through prep."""
import pytest

from pipeline import audio


def test_default_thriller_alignment():
    # 44.3 s danced span, default prep constants -> music delayed by pad+blend_in.
    al = audio.compute_alignment(44.3)
    assert al.audio_delay_s == 1.5           # PAD_IN 1.0 + BLEND_IN 0.5
    assert al.trim_start_s == 0.0
    assert al.trim_duration_s == 44.3
    assert al.music_end_s == 45.8            # 1.5 + 44.3
    assert al.performance_s == 49.3          # 1.5 + 44.3 + 1.0 + 2.5


def test_windowed_dance_trims_source():
    # A motion windowed to start 10 s into the source must trim the audio there.
    al = audio.compute_alignment(20.0, window_start_s=10.0)
    assert al.trim_start_s == 10.0
    assert al.trim_duration_s == 20.0
    assert al.audio_delay_s == 1.5           # delay is prep-driven, not window-driven
    assert al.music_end_s == 21.5


def test_alignment_from_prep_info_matches_frame_count():
    # prep info reports the danced span as in_frames at 30 fps.
    info = {"in_frames": 1329, "out_frames": 1479}   # the real Thriller numbers
    al = audio.alignment_from_prep_info(info)
    assert al.trim_duration_s == pytest.approx(44.3, abs=1e-6)
    assert al.performance_s == pytest.approx(49.3, abs=1e-6)


def test_custom_prep_constants_change_delay():
    al = audio.compute_alignment(30.0, pad_in_s=2.0, blend_in_s=1.0,
                                 blend_out_s=0.0, hold_out_s=0.0)
    assert al.audio_delay_s == 3.0
    assert al.music_end_s == 33.0
    assert al.performance_s == 33.0


def test_music_never_starts_before_motion():
    # Invariant: for any non-negative prep, the dance (and music) begins at or
    # after the standing intro ends — never earlier.
    for pad, blend in [(0.0, 0.0), (1.0, 0.5), (3.0, 2.0)]:
        al = audio.compute_alignment(10.0, pad_in_s=pad, blend_in_s=blend)
        assert al.audio_delay_s == pad + blend
        assert al.music_end_s == al.audio_delay_s + 10.0
        assert al.performance_s >= al.music_end_s


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        audio.compute_alignment(0.0)
    with pytest.raises(ValueError):
        audio.compute_alignment(-5.0)
    with pytest.raises(ValueError):
        audio.compute_alignment(10.0, window_start_s=-1.0)
