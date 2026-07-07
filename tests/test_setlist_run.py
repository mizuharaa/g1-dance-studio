"""Lane-4 tests: set-list show-run plan (per-item audio cues) + run state machine.

Everything runs in-memory/tmp with fabricated dances (a plain dance_lookup callable):
no robot, no DDS, no real audio playback. The plan only COMPUTES the show-time music
offset from pipeline/show_audio.py's tick0 + RAMP_S + audio_delay_s contract.
"""
import pytest

from pipeline import setlist, show_audio


@pytest.fixture
def setlists_env(tmp_path, monkeypatch):
    d = tmp_path / "setlists"
    d.mkdir()
    monkeypatch.setattr(setlist, "SETLISTS_DIR", d)
    return d


class FakeDance:
    """Just enough of pipeline.shows.Dance for the plan/resolver to read."""

    def __init__(self, name, status, dur, audio=None):
        self.name = name
        self.status = status
        self.duration_s = dur
        self.audio = audio


def _lookup(lib):
    return lambda dance_id: lib.get(dance_id)


# ---- audio cue timing (reuses show_audio's 4.0 s contract) -----------------------

def test_run_plan_audio_offset_default_and_none(setlists_env):
    sl = setlist.new_setlist("Set")
    sl = setlist.set_items(sl.id, [{"dance_id": "loud"}, {"dance_id": "silent"}])
    lib = {
        "loud": FakeDance("Loud", "show-ready", 30, audio={"track": "data/audio/x.wav"}),
        "silent": FakeDance("Quiet", "show-ready", 20, audio=None),
    }
    plan = setlist.setlist_run_plan(sl, _lookup(lib))
    # a dance WITH audio -> offset 4.0 (2.5 s ramp + 1.5 s lead-in); WITHOUT -> None
    assert plan[0]["audio"] == {"track": "data/audio/x.wav", "offset_s": 4.0}
    assert plan[0]["audio"]["offset_s"] == pytest.approx(show_audio.DEFAULT_OFFSET_S)
    assert plan[1]["audio"] is None
    assert plan[0]["has_audio"] is True and plan[1]["has_audio"] is False


def test_run_plan_audio_offset_honours_record_delay(setlists_env):
    sl = setlist.new_setlist("Set")
    sl = setlist.set_items(sl.id, [{"dance_id": "a"}])
    lib = {"a": FakeDance("A", "show-ready", 30,
                          audio={"track": "t.wav", "align": {"audio_delay_s": 2.0}})}
    plan = setlist.setlist_run_plan(sl, _lookup(lib))
    # RAMP_S (2.5) + the record's audio_delay_s (2.0) = 4.5, straight from show_audio
    assert plan[0]["audio"]["offset_s"] == pytest.approx(show_audio.RAMP_S + 2.0)


# ---- whole-set-list gating: all-or-nothing, surface the blockers -----------------

def test_not_show_ready_item_blocks_the_whole_set(setlists_env):
    sl = setlist.new_setlist("Set")
    sl = setlist.set_items(sl.id, [
        {"dance_id": "ready"}, {"dance_id": "draft"}, {"dance_id": "gone"}])
    lib = {"ready": FakeDance("R", "show-ready", 30, audio={"track": "t.wav"}),
           "draft": FakeDance("D", "draft", 40)}
    plan = setlist.setlist_run_plan(sl, _lookup(lib))
    assert plan[0]["blockers"] == []
    assert plan[1]["blockers"] == ["status is draft"]
    assert plan[2]["blockers"] == ["missing"]
    # one blocked number holds the whole set-list
    assert setlist.plan_runnable(plan) is False
    blocked = {b["dance_id"] for b in setlist.plan_blockers(plan)}
    assert blocked == {"draft", "gone"}


def test_all_show_ready_is_runnable(setlists_env):
    sl = setlist.new_setlist("Set")
    sl = setlist.set_items(sl.id, [{"dance_id": "a"}, {"dance_id": "b"}])
    lib = {"a": FakeDance("A", "show-ready", 30, audio={"track": "a.wav"}),
           "b": FakeDance("B", "show-ready", 20)}
    plan = setlist.setlist_run_plan(sl, _lookup(lib))
    assert setlist.plan_runnable(plan) is True
    assert setlist.plan_blockers(plan) == []
    assert plan[0]["audio"]["offset_s"] == 4.0    # a has music
    assert plan[1]["audio"] is None               # b is silent (not a blocker)


def test_empty_setlist_not_runnable(setlists_env):
    sl = setlist.new_setlist("Empty")
    plan = setlist.setlist_run_plan(sl, _lookup({}))
    assert plan == []
    assert setlist.plan_runnable(plan) is False


# ---- per-item state machine + resume ---------------------------------------------

def test_new_run_all_pending(setlists_env):
    sl = setlist.new_setlist("Set")
    sl = setlist.set_items(sl.id, [{"dance_id": "a"}, {"dance_id": "b"}])
    run = setlist.new_run(sl)
    assert run.states == ["pending", "pending"]
    assert setlist.next_index(run) == 0
    assert setlist.remaining_indices(run) == [0, 1]


def test_resume_skips_done_items(setlists_env):
    sl = setlist.new_setlist("Set")
    sl = setlist.set_items(sl.id, [
        {"dance_id": "a"}, {"dance_id": "b"}, {"dance_id": "c"}])
    setlist.new_run(sl)
    setlist.set_item_state(sl.id, 0, "running")
    setlist.set_item_state(sl.id, 0, "done")
    run = setlist.load_run(sl.id)                 # reloaded fresh: state is durable
    assert run.states[0] == "done"
    assert setlist.next_index(run) == 1           # resume skips the done item
    assert setlist.remaining_indices(run) == [1, 2]


def test_all_done_run_is_complete(setlists_env):
    sl = setlist.new_setlist("Set")
    sl = setlist.set_items(sl.id, [{"dance_id": "a"}])
    setlist.new_run(sl)
    setlist.set_item_state(sl.id, 0, "done")
    run = setlist.load_run(sl.id)
    assert setlist.next_index(run) is None
    assert setlist.remaining_indices(run) == []


def test_illegal_and_bad_transitions_rejected(setlists_env):
    sl = setlist.new_setlist("Set")
    sl = setlist.set_items(sl.id, [{"dance_id": "a"}])
    setlist.new_run(sl)
    setlist.set_item_state(sl.id, 0, "running")
    setlist.set_item_state(sl.id, 0, "done")
    with pytest.raises(ValueError):
        setlist.set_item_state(sl.id, 0, "running")   # done is terminal
    with pytest.raises(ValueError):
        setlist.set_item_state(sl.id, 0, "bogus")     # unknown state
    with pytest.raises(IndexError):
        setlist.set_item_state(sl.id, 9, "running")   # out of range


def test_aborted_item_can_be_retried(setlists_env):
    sl = setlist.new_setlist("Set")
    sl = setlist.set_items(sl.id, [{"dance_id": "a"}])
    setlist.new_run(sl)
    setlist.set_item_state(sl.id, 0, "running")
    setlist.set_item_state(sl.id, 0, "aborted")
    run = setlist.set_item_state(sl.id, 0, "running")   # retry an aborted number
    assert run.states[0] == "running"
    assert setlist.next_index(run) == 0                 # not done -> still up next


def test_get_or_create_reconciles_after_edit(setlists_env):
    sl = setlist.new_setlist("Set")
    sl = setlist.set_items(sl.id, [{"dance_id": "a"}, {"dance_id": "b"}])
    setlist.new_run(sl)
    setlist.set_item_state(sl.id, 0, "done")
    # operator appends a third number after the run already started
    sl = setlist.set_items(sl.id, [
        {"dance_id": "a"}, {"dance_id": "b"}, {"dance_id": "c"}])
    run = setlist.get_or_create_run(sl)
    assert len(run.states) == 3
    assert run.states[0] == "done"        # existing state preserved by index
    assert run.states[2] == "pending"     # new tail item padded
    assert setlist.load_run(sl.id).states == ["done", "pending", "pending"]


def test_get_or_create_when_none_exists(setlists_env):
    sl = setlist.new_setlist("Set")
    sl = setlist.set_items(sl.id, [{"dance_id": "a"}])
    assert setlist.load_run(sl.id) is None
    run = setlist.get_or_create_run(sl)
    assert run.states == ["pending"]


def test_run_plan_reflects_run_state(setlists_env):
    sl = setlist.new_setlist("Set")
    sl = setlist.set_items(sl.id, [{"dance_id": "a"}, {"dance_id": "b"}])
    lib = {"a": FakeDance("A", "show-ready", 30, audio={"track": "a.wav"}),
           "b": FakeDance("B", "show-ready", 20)}
    setlist.new_run(sl)
    setlist.set_item_state(sl.id, 0, "running")
    run = setlist.load_run(sl.id)
    plan = setlist.setlist_run_plan(sl, _lookup(lib), run=run)
    assert plan[0]["state"] == "running"
    assert plan[1]["state"] == "pending"
    # without a run, every item defaults to pending
    plan0 = setlist.setlist_run_plan(sl, _lookup(lib))
    assert [p["state"] for p in plan0] == ["pending", "pending"]
