"""Show-production tests: music alignment, set-lists, rehearsal-vs-live."""
import shutil

import pytest

from pipeline import audio, setlist, shows


# ---- audio alignment (pure math) -------------------------------------------------

def test_alignment_default_prep():
    al = audio.compute_alignment(44.3)
    # PAD_IN 1.0 + BLEND_IN 0.5 => music delayed 1.5s; blend/hold tail after.
    assert al.audio_delay_s == 1.5
    assert al.music_end_s == pytest.approx(45.8)
    assert al.performance_s == pytest.approx(49.3)
    assert al.trim_start_s == 0.0
    assert al.trim_duration_s == 44.3


def test_alignment_windowed_source():
    al = audio.compute_alignment(20.0, window_start_s=12.5)
    assert al.trim_start_s == 12.5          # take source audio from the windowed start
    assert al.trim_duration_s == 20.0
    assert al.audio_delay_s == 1.5          # delay is independent of the window


def test_alignment_rejects_bad_input():
    with pytest.raises(ValueError):
        audio.compute_alignment(0)
    with pytest.raises(ValueError):
        audio.compute_alignment(10, pad_in_s=-1)


# ---- set-list model + resolver ---------------------------------------------------

@pytest.fixture
def setlists_env(tmp_path, monkeypatch):
    d = tmp_path / "setlists"
    d.mkdir()
    monkeypatch.setattr(setlist, "SETLISTS_DIR", d)
    return d


class FakeDance:
    def __init__(self, name, status, dur, audio=None):
        self.name, self.status, self.duration_s, self.audio = name, status, dur, audio


def test_setlist_create_and_items(setlists_env):
    sl = setlist.new_setlist("Friday Set")
    assert sl.items == []
    sl = setlist.set_items(sl.id, [
        {"dance_id": "a", "gap_after_s": 10, "note": "opener"},
        {"dance_id": "b"},  # gap defaults
    ])
    assert [it["dance_id"] for it in sl.items] == ["a", "b"]
    assert sl.items[1]["gap_after_s"] == setlist.DEFAULT_GAP_S


def test_setlist_item_validation(setlists_env):
    sl = setlist.new_setlist("x")
    with pytest.raises(ValueError):
        setlist.set_items(sl.id, [{"note": "no dance id"}])
    with pytest.raises(ValueError):
        setlist.set_items(sl.id, [{"dance_id": "a", "gap_after_s": -3}])


def test_setlist_resolve_runtime_and_blockers(setlists_env):
    sl = setlist.new_setlist("Show")
    sl = setlist.set_items(sl.id, [
        {"dance_id": "a", "gap_after_s": 8},
        {"dance_id": "b", "gap_after_s": 8},
        {"dance_id": "missing", "gap_after_s": 8},
    ])
    lib = {"a": FakeDance("A", "show-ready", 30, audio={"x": 1}),
           "b": FakeDance("B", "draft", 40)}
    r = setlist.resolve(sl, lambda i: lib.get(i))
    # runtime = 30 + 8gap + 40 + 8gap + 0(missing); last item's gap not counted
    assert r["total_runtime_s"] == pytest.approx(30 + 8 + 40 + 8 + 0)
    assert r["show_ready"] is False
    reasons = {b["reason"] for b in r["blockers"]}
    assert "status is draft" in reasons and "missing" in reasons
    assert r["items"][0]["has_audio"] is True


def test_setlist_all_ready_is_show_ready(setlists_env):
    sl = setlist.new_setlist("Ready")
    sl = setlist.set_items(sl.id, [{"dance_id": "a"}, {"dance_id": "b"}])
    lib = {"a": FakeDance("A", "show-ready", 30), "b": FakeDance("B", "show-ready", 20)}
    r = setlist.resolve(sl, lambda i: lib.get(i))
    assert r["show_ready"] is True and not r["blockers"]


def test_empty_setlist_not_show_ready(setlists_env):
    sl = setlist.new_setlist("Empty")
    r = setlist.resolve(sl, lambda i: None)
    assert r["show_ready"] is False and r["count"] == 0


# ---- rehearsal vs live: only a LIVE incident demotes a dance ---------------------

def _show_ready_dance(shows_mod):
    d = shows_mod.new_dance("D", duration_s=30, status="show-ready")
    d.repeatability["consecutive_clean"] = 5
    d.save()
    return d


def test_live_incident_demotes_dance(dances_env):
    shows_mod, _ = dances_env
    d = _show_ready_dance(shows_mod)
    show = shows_mod.new_show(d, "Op", mode="live")
    shows_mod.record_outcome(show, "incident")
    after = shows_mod.load_dance(d.id)
    assert after.status == "sim-verified"                 # demoted
    assert after.repeatability["consecutive_clean"] == 0  # streak reset
    assert after.incident is not None


def test_rehearsal_incident_does_not_demote(dances_env):
    shows_mod, _ = dances_env
    d = _show_ready_dance(shows_mod)
    show = shows_mod.new_show(d, "Op", mode="rehearsal")
    shows_mod.record_outcome(show, "incident")
    after = shows_mod.load_dance(d.id)
    assert after.status == "show-ready"                    # untouched
    assert after.repeatability["consecutive_clean"] == 5   # streak preserved
    assert after.incident is None


def test_new_show_rejects_bad_mode(dances_env):
    shows_mod, _ = dances_env
    d = _show_ready_dance(shows_mod)
    with pytest.raises(ValueError):
        shows_mod.new_show(d, "Op", mode="bogus")


# ---- audio attach (needs ffmpeg for the placeholder track) -----------------------

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_attach_placeholder_audio(dances_env, monkeypatch, tmp_path):
    shows_mod, _ = dances_env
    monkeypatch.setattr(audio, "PROJECT_ROOT", tmp_path)
    d = shows_mod.new_dance("D", duration_s=10)
    rec = audio.attach_audio_for_dance(d, placeholder_bpm=120)
    assert rec["source"] == "placeholder_click_track"
    assert rec["align"]["audio_delay_s"] == 1.5
    assert rec["muxed_preview"] is None       # no preview to mux onto
    saved = shows_mod.set_audio(d.id, rec)
    assert saved.audio and saved.audio["attached_at"]


def test_attach_audio_needs_duration(dances_env):
    shows_mod, _ = dances_env
    d = shows_mod.new_dance("NoDur")            # duration_s is None
    with pytest.raises(ValueError):
        audio.attach_audio_for_dance(d, placeholder_bpm=120)
