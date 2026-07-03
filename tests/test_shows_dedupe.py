"""Dance-library seeding idempotency + de-duplication (BUG: Thriller/thriller dupes)."""
import importlib

import pytest


@pytest.fixture
def shows(tmp_path, monkeypatch):
    from pipeline import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    import pipeline.shows as shows_mod
    importlib.reload(shows_mod)
    monkeypatch.setattr(shows_mod, "DANCES_DIR", tmp_path / "dances")
    monkeypatch.setattr(shows_mod, "SHOWS_DIR", tmp_path / "shows")
    (tmp_path / "dances").mkdir(parents=True, exist_ok=True)
    yield shows_mod
    importlib.reload(shows_mod)


def test_find_dance_is_case_insensitive(shows):
    shows.new_dance("Thriller", duration_s=49.3)
    assert shows.find_dance("thriller") is not None
    assert shows.find_dance("THRILLER  ") is not None


def test_dedupe_keeps_policy_bearing_record(shows):
    # an empty seeded draft + a richer policy-bearing entry, same logical name
    shows.new_dance("thriller", duration_s=44.3)
    shows.new_dance("Thriller", duration_s=49.3, policy_path="data/policies/thriller/policy.onnx",
                    policy_sha256="abc123", status="sim-verified")
    removed = shows.dedupe_dances()
    assert removed == 1
    survivors = shows.list_dances()
    assert len(survivors) == 1
    keep = survivors[0]
    assert keep.policy_path == "data/policies/thriller/policy.onnx"  # policy preserved
    assert keep.status == "sim-verified"


def test_dedupe_backfills_missing_fields(shows):
    # richest record wins but is missing a field the loser has -> back-filled
    shows.new_dance("Thriller", duration_s=49.3, policy_path="p.onnx", status="sim-verified")
    shows.new_dance("thriller", duration_s=44.3, motion_csv="m.csv", preview="/previews/x.mp4")
    shows.dedupe_dances()
    keep = shows.list_dances()[0]
    assert keep.policy_path == "p.onnx"
    assert keep.motion_csv == "m.csv" and keep.preview == "/previews/x.mp4"


def test_dedupe_is_idempotent_and_no_op_when_clean(shows):
    shows.new_dance("thriller", duration_s=49.3)
    shows.new_dance("test-segment", duration_s=28.8)
    assert shows.dedupe_dances() == 0
    assert len(shows.list_dances()) == 2
    assert shows.dedupe_dances() == 0  # second pass still no-op


def test_seeding_runs_dedupe_and_stays_idempotent(shows):
    # Simulate the pre-existing duplicates, then confirm the startup path collapses
    # them and a second startup makes no new ones. (No artifacts present, so seeding
    # itself registers nothing new — the dedupe is what must fire.)
    shows.new_dance("thriller", duration_s=44.3)
    shows.new_dance("Thriller", duration_s=49.3, policy_path="p.onnx")
    shows.seed_initial_dances()
    names = [shows._norm_name(d.name) for d in shows.list_dances()]
    assert names.count("thriller") == 1
    shows.seed_initial_dances()
    assert [shows._norm_name(d.name) for d in shows.list_dances()].count("thriller") == 1
