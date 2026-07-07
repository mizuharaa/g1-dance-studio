"""Tests for the pre-show checklist + show-phase ownership model (pipeline/preshow.py).

No robot, no network: robot reachability is injected (bool / callable / None), the
selected venue is a passed-in value, and the operator's CONFIRM ticks are an `acks` set.
Dances are fabricated lightweight objects, plus one real shows.new_dance path in a temp
dir to exercise the on-disk policy re-hash.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from pipeline import exam_verdict as ev
from pipeline import preshow, shows

ALL_ACKS = set(preshow.CONFIRM_KEYS)


@dataclass
class FakeDance:
    """A dance-like object exposing exactly the fields the checklist reads."""
    status: str = "show-ready"
    policy_path: str | None = None
    policy_sha256: str | None = None
    audio: dict | None = None


@dataclass
class FakeVenue:
    name: str = "Home (2 m)"


def _item(report: dict, key: str) -> dict:
    return next(i for i in report["items"] if i["key"] == key)


@pytest.fixture
def good_policy(tmp_path):
    """A real policy file on disk + its pinned sha, so the policy item can pass."""
    p = tmp_path / "policy.onnx"
    p.write_bytes(b"trained-policy-bytes")
    return str(p), ev.full_sha256(p)


def _all_good_dance(good_policy) -> FakeDance:
    path, sha = good_policy
    return FakeDance(status="show-ready", policy_path=path, policy_sha256=sha,
                     audio={"source": "song.wav"})


# ---- not show-ready -> ready False with the right blocker ------------------------

def test_not_show_ready_blocks(good_policy):
    d = _all_good_dance(good_policy)
    d.status = "sim-verified"
    r = preshow.evaluate_checklist(d, robot_ping=True, venue_active=FakeVenue(),
                                   acks=ALL_ACKS)
    assert r["ready"] is False
    sr = _item(r, "show_ready")
    assert sr["ok"] is False and sr["severity"] == "blocker" and sr["kind"] == "auto"
    assert "sim-verified" in sr["detail"]


# ---- all-good + acks -> ready True ------------------------------------------------

def test_all_good_ready_true(good_policy):
    d = _all_good_dance(good_policy)
    r = preshow.evaluate_checklist(d, robot_ping=True, venue_active=FakeVenue(),
                                   acks=ALL_ACKS)
    assert r["ready"] is True
    # every blocker item is ok
    assert all(i["ok"] for i in r["items"] if i["severity"] == "blocker")
    assert _item(r, "venue_selected")["detail"].endswith("Home (2 m)")
    assert _item(r, "audio_attached")["ok"] is True


def test_all_good_but_missing_one_ack_not_ready(good_policy):
    d = _all_good_dance(good_policy)
    r = preshow.evaluate_checklist(d, robot_ping=True, venue_active=FakeVenue(),
                                   acks={"damping_remote", "tether_area"})  # feet missing
    assert r["ready"] is False
    assert _item(r, "feet_placement")["ok"] is False
    assert _item(r, "damping_remote")["ok"] is True


# ---- sha-mismatch policy -> blocker ----------------------------------------------

def test_policy_sha_mismatch_blocks(good_policy):
    d = _all_good_dance(good_policy)
    d.policy_sha256 = "0" * 64  # pinned sha no longer matches the file on disk
    r = preshow.evaluate_checklist(d, robot_ping=True, venue_active=FakeVenue(),
                                   acks=ALL_ACKS)
    assert r["ready"] is False
    pp = _item(r, "policy_pinned")
    assert pp["ok"] is False and pp["severity"] == "blocker"
    assert "sha mismatch" in pp["detail"]


def test_policy_missing_file_blocks(tmp_path):
    d = FakeDance(status="show-ready", policy_path=str(tmp_path / "gone.onnx"),
                  policy_sha256="a" * 64, audio={"source": "s"})
    r = preshow.evaluate_checklist(d, robot_ping=True, venue_active=FakeVenue(),
                                   acks=ALL_ACKS)
    assert r["ready"] is False
    assert "missing" in _item(r, "policy_pinned")["detail"]


def test_no_policy_attached_blocks(good_policy):
    d = FakeDance(status="show-ready", audio={"source": "s"})  # no policy at all
    r = preshow.evaluate_checklist(d, robot_ping=True, venue_active=FakeVenue(),
                                   acks=ALL_ACKS)
    assert _item(r, "policy_pinned")["ok"] is False and r["ready"] is False


# ---- robot reachability injection -------------------------------------------------

def test_robot_ping_none_is_unknown_blocker(good_policy):
    d = _all_good_dance(good_policy)
    r = preshow.evaluate_checklist(d, robot_ping=None, venue_active=FakeVenue(),
                                   acks=ALL_ACKS)
    rr = _item(r, "robot_reachable")
    assert rr["ok"] is False and rr["severity"] == "blocker"
    assert "unknown" in rr["detail"].lower()
    assert r["ready"] is False


def test_robot_ping_callable_true_and_false(good_policy):
    d = _all_good_dance(good_policy)
    r_ok = preshow.evaluate_checklist(d, robot_ping=lambda: True,
                                      venue_active=FakeVenue(), acks=ALL_ACKS)
    assert _item(r_ok, "robot_reachable")["ok"] is True and r_ok["ready"] is True
    r_no = preshow.evaluate_checklist(d, robot_ping=lambda: False,
                                      venue_active=FakeVenue(), acks=ALL_ACKS)
    assert _item(r_no, "robot_reachable")["ok"] is False and r_no["ready"] is False


def test_robot_ping_callable_raises_is_no_go(good_policy):
    d = _all_good_dance(good_policy)

    def boom():
        raise RuntimeError("LAN down")

    r = preshow.evaluate_checklist(d, robot_ping=boom, venue_active=FakeVenue(),
                                   acks=ALL_ACKS)
    rr = _item(r, "robot_reachable")
    assert rr["ok"] is False and "LAN down" in rr["detail"]


# ---- venue selection --------------------------------------------------------------

def test_missing_venue_blocks(good_policy):
    d = _all_good_dance(good_policy)
    r = preshow.evaluate_checklist(d, robot_ping=True, venue_active=None,
                                   acks=ALL_ACKS)
    vs = _item(r, "venue_selected")
    assert vs["ok"] is False and vs["severity"] == "blocker" and r["ready"] is False


def test_venue_accepts_dict_value(good_policy):
    d = _all_good_dance(good_policy)
    r = preshow.evaluate_checklist(d, robot_ping=True,
                                   venue_active={"name": "Studio A"}, acks=ALL_ACKS)
    assert _item(r, "venue_selected")["ok"] is True
    assert "Studio A" in _item(r, "venue_selected")["detail"]


# ---- audio is advisory (warn), never a blocker -----------------------------------

def test_missing_audio_warns_but_does_not_block(good_policy):
    d = _all_good_dance(good_policy)
    d.audio = None  # silent dance is valid
    r = preshow.evaluate_checklist(d, robot_ping=True, venue_active=FakeVenue(),
                                   acks=ALL_ACKS)
    aa = _item(r, "audio_attached")
    assert aa["ok"] is False and aa["severity"] == "warn"
    assert r["ready"] is True  # a warn never blocks the deploy gate


# ---- confirm items require the operator's acks -----------------------------------

def test_confirm_items_need_acks(good_policy):
    d = _all_good_dance(good_policy)
    r_none = preshow.evaluate_checklist(d, robot_ping=True, venue_active=FakeVenue(),
                                        acks=None)
    for key in preshow.CONFIRM_KEYS:
        assert _item(r_none, key)["ok"] is False
        assert _item(r_none, key)["kind"] == "confirm"
    assert r_none["ready"] is False
    r_all = preshow.evaluate_checklist(d, robot_ping=True, venue_active=FakeVenue(),
                                       acks=ALL_ACKS)
    for key in preshow.CONFIRM_KEYS:
        assert _item(r_all, key)["ok"] is True


# ---- report shape + static spec ---------------------------------------------------

def test_report_item_shape():
    r = preshow.evaluate_checklist(FakeDance(status="draft"))
    for it in r["items"]:
        assert set(it) == {"key", "label", "ok", "detail", "kind", "severity"}
        assert it["kind"] in ("auto", "confirm")
        assert it["severity"] in ("blocker", "warn")
    keys = [it["key"] for it in r["items"]]
    assert keys == [s["key"] for s in preshow.checklist_items()]  # order matches spec


# ---- real Dance record via shows.new_dance in a temp dir -------------------------

@pytest.fixture
def dance_env(tmp_path, monkeypatch):
    monkeypatch.setattr(shows, "DATA_DIR", tmp_path)
    monkeypatch.setattr(shows, "DANCES_DIR", tmp_path / "dances")
    monkeypatch.setattr(shows, "SHOWS_DIR", tmp_path / "shows")
    monkeypatch.setattr(shows, "PROJECT_ROOT", tmp_path)
    (tmp_path / "dances").mkdir(parents=True)
    (tmp_path / "shows").mkdir(parents=True)
    (tmp_path / "policy.onnx").write_bytes(b"fake-policy-bytes")
    (tmp_path / "motion.csv").write_text("0,0,0.79\n")
    return tmp_path


def test_checklist_on_real_show_ready_dance(dance_env):
    """A dance driven to show-ready through the real gate passes every blocker."""
    d = shows.new_dance("D", duration_s=30.0, policy_path="policy.onnx",
                        motion_csv="motion.csv")
    sha = ev.full_sha256(shows.PROJECT_ROOT / "policy.onnx")
    for _ in range(3):
        shows.record_sim_run(shows.load_dance(d.id), True, policy_sha256=sha)
    d = shows.promote(shows.load_dance(d.id), "show-ready")
    d = shows.set_audio(d.id, {"source": "song.wav"})

    r = preshow.evaluate_checklist(d, robot_ping=True, venue_active=FakeVenue(),
                                   acks=ALL_ACKS)
    assert r["ready"] is True
    assert _item(r, "policy_pinned")["ok"] is True  # relative path resolved + re-hashed


# ---- show-phase ownership model ---------------------------------------------------

def test_phases_ordered_with_documented_owners():
    phases = preshow.make_show_phases()
    assert [p["phase"] for p in phases] == [
        "WALK_ON", "ARM", "DANCE", "STAND", "WALK_OFF"]
    owners = {p["phase"]: p["owner"] for p in phases}
    assert owners == {
        "WALK_ON": "remote/onboard",
        "ARM": "operator",
        "DANCE": "policy",
        "STAND": "policy->onboard",
        "WALK_OFF": "remote/onboard",
    }
    for p in phases:
        assert set(p) == {"phase", "owner", "note"}
        assert p["note"].strip()


def test_phase_docstring_ties_handoffs():
    doc = preshow.make_show_phases.__doc__
    assert "'ai'" in doc                      # onboard locomotion for WALK_ON/WALK_OFF
    assert "entry" in doc.lower() and "exit" in doc.lower()  # ARM/STAND handoffs
