"""Venue REGISTRY: a persisted list of named venues with exactly one ACTIVE,
seeded from the default 'Home (2 m)'. Storage is a single data/venues/venues.json;
every test redirects VENUES_DIR (and therefore that file) to an isolated tmp dir
so nothing touches the real data/.  See pipeline/venue.py."""
import json

import pytest

import pipeline.venue as venue


@pytest.fixture
def reg_dir(tmp_path, monkeypatch):
    # Monkeypatch the directory the registry lives in; _registry_path() reads
    # VENUES_DIR at call time, so this relocates data/venues/venues.json to tmp.
    monkeypatch.setattr(venue, "VENUES_DIR", tmp_path)
    return tmp_path


def _on_disk(reg_dir):
    return json.loads((reg_dir / "venues.json").read_text())


# ---- seeding the default ---- #
def test_seeds_default_when_file_absent(reg_dir):
    assert not (reg_dir / "venues.json").exists()
    active = venue.get_active_venue()
    assert active.name == "Home (2 m)"
    assert active.max_excursion_m == pytest.approx(1.5)   # 2 m radius - 0.5 margin
    # seeding persisted an atomic file with the default active + present
    assert (reg_dir / "venues.json").exists()
    reg = _on_disk(reg_dir)
    assert reg["active"] == active.id
    assert [v["name"] for v in reg["venues"]] == ["Home (2 m)"]


def test_active_max_excursion_reflects_default(reg_dir):
    assert venue.active_max_excursion_m() == pytest.approx(1.5)


def test_list_after_seed_has_one(reg_dir):
    vs = venue.list_venues()
    assert len(vs) == 1 and vs[0].name == "Home (2 m)"


# ---- adding a venue ---- #
def test_add_venue_appears_in_list_without_stealing_active(reg_dir):
    v = venue.add_or_update_venue("Studio", radius_m=3.0, margin_m=0.5,
                                  notes="dance studio")
    assert v.max_excursion_m == pytest.approx(2.5)
    names = {x.name for x in venue.list_venues()}
    assert {"Home (2 m)", "Studio"} <= names
    # adding alone does not change the active selection
    assert venue.get_active_venue().name == "Home (2 m)"
    assert venue.get_venue("Studio").notes == "dance studio"


def test_add_or_update_is_upsert_by_name(reg_dir):
    a = venue.add_or_update_venue("Loft", radius_m=2.0, margin_m=0.5)
    b = venue.add_or_update_venue("Loft", radius_m=4.0, margin_m=0.5, notes="bigger")
    assert a.id == b.id                                  # same entry, not duplicated
    lofts = [x for x in venue.list_venues() if x.name == "Loft"]
    assert len(lofts) == 1
    assert lofts[0].radius_m == pytest.approx(4.0)
    assert lofts[0].notes == "bigger"


def test_make_active_flag_switches_on_add(reg_dir):
    venue.add_or_update_venue("Hall", radius_m=6.0, margin_m=0.5, make_active=True)
    assert venue.get_active_venue().name == "Hall"


# ---- switching the active venue ---- #
def test_switch_active_changes_excursion(reg_dir):
    venue.add_or_update_venue("Gym", radius_m=5.0, margin_m=1.0)     # excursion 4.0
    assert venue.active_max_excursion_m() == pytest.approx(1.5)      # still default
    returned = venue.set_active_venue("Gym")
    assert returned.name == "Gym"
    assert venue.get_active_venue().name == "Gym"
    assert venue.active_max_excursion_m() == pytest.approx(4.0)


def test_set_active_accepts_id_and_name(reg_dir):
    v = venue.add_or_update_venue("Rooftop", radius_m=3.5, margin_m=0.5)
    venue.set_active_venue(v.id)                          # by id
    assert venue.get_active_venue().id == v.id
    venue.set_active_venue("Home (2 m)")                  # by name
    assert venue.get_active_venue().name == "Home (2 m)"


def test_set_active_unknown_raises(reg_dir):
    with pytest.raises(ValueError):
        venue.set_active_venue("Nonexistent")


# ---- validation ---- #
def test_validation_rejects_margin_ge_radius_and_negative(reg_dir):
    with pytest.raises(ValueError):
        venue.add_or_update_venue("Zero", radius_m=1.0, margin_m=1.0)   # excursion 0
    with pytest.raises(ValueError):
        venue.add_or_update_venue("Inv", radius_m=1.0, margin_m=2.0)    # margin>radius
    with pytest.raises(ValueError):
        venue.add_or_update_venue("Neg", radius_m=1.0, margin_m=-0.1)   # margin<0
    with pytest.raises(ValueError):
        venue.add_or_update_venue("   ", radius_m=2.0, margin_m=0.5)    # empty name
    # nothing invalid was persisted
    names = {x.name for x in venue.list_venues()}
    assert not ({"Zero", "Inv", "Neg"} & names)


# ---- persistence ---- #
def test_round_trip_persistence(reg_dir):
    venue.add_or_update_venue("Ballroom", radius_m=5.0, margin_m=0.5,
                              notes="parquet")
    venue.set_active_venue("Ballroom")
    # venue.py holds no in-memory cache, so these calls re-read straight off disk,
    # exactly as a fresh process would.
    assert venue.get_active_venue().name == "Ballroom"
    v = venue.get_venue("Ballroom")
    assert v is not None and v.notes == "parquet"
    assert v.max_excursion_m == pytest.approx(4.5)
    reg = _on_disk(reg_dir)
    assert reg["active"] == v.id
    assert any(e["name"] == "Ballroom" for e in reg["venues"])


def test_dangling_active_pointer_is_repaired(reg_dir):
    # Hand-write a registry whose active points at a venue that does not exist.
    (reg_dir / "venues.json").write_text(json.dumps({
        "active": "ghost",
        "venues": [{"id": "studio", "name": "Studio", "shape": "circle",
                    "radius_m": 3.0, "margin_m": 0.5}],
    }))
    assert venue.get_active_venue().name == "Studio"     # falls back to real venue
    assert venue.active_max_excursion_m() == pytest.approx(2.5)


def test_corrupt_file_reseeds_default(reg_dir):
    (reg_dir / "venues.json").write_text("{ not json ]")
    assert venue.get_active_venue().name == "Home (2 m)"
    assert [v.name for v in venue.list_venues()] == ["Home (2 m)"]
