"""Regression tests for the production-audit consolidation (app-lane findings).

Each locks a specific finding so it can't silently reopen:
- MEC recenter: deployed motion excursion == certified footprint, not ~2x (safety).
- Promote: operator reaches show-ready only with enough clean runs (workflow).
- Library import: never trusts an archive's verification state (security).
- Dedupe: a back-filled file survives the loser dir's deletion (data-integrity).
- Incident outcome demotes a show-ready dance; rehearsal never does (deploy-safety).
"""
from __future__ import annotations

import importlib
import shutil

import numpy as np
import pytest

from pipeline import exam_verdict as ev
from pipeline import shows


@pytest.fixture
def env(tmp_path, monkeypatch):
    """shows.py pointed at a temp data dir, PROJECT_ROOT rebased, with real files."""
    monkeypatch.setattr(shows, "DATA_DIR", tmp_path)
    monkeypatch.setattr(shows, "DANCES_DIR", tmp_path / "dances")
    monkeypatch.setattr(shows, "SHOWS_DIR", tmp_path / "shows")
    monkeypatch.setattr(shows, "PROJECT_ROOT", tmp_path)
    (tmp_path / "dances").mkdir(parents=True)
    (tmp_path / "shows").mkdir(parents=True)
    (tmp_path / "policy.onnx").write_bytes(b"fake-policy-bytes")
    (tmp_path / "motion.csv").write_text("0,0,0.79\n")
    return tmp_path


def _make_show_ready(name="D"):
    """A dance driven to show-ready through the real gate (3 clean signed runs)."""
    from pipeline.config import PROJECT_ROOT  # patched by env
    d = shows.new_dance(name, duration_s=30.0, policy_path="policy.onnx",
                        motion_csv="motion.csv")
    sha = ev.full_sha256(shows.PROJECT_ROOT / "policy.onnx")
    for _ in range(3):
        shows.record_sim_run(shows.load_dance(d.id), True, policy_sha256=sha)
    return shows.promote(shows.load_dance(d.id), "show-ready")


# ---- HIGH: MEC recenter bounds deployed excursion to the certified footprint ----

def test_mec_recenter_bounds_excursion_to_radius():
    from pipeline.find_window import window_center, _mec
    t = np.linspace(0, 2 * np.pi, 60)
    xy = np.stack([3.0 + 0.5 * np.cos(t), 3.0 + 0.5 * np.sin(t)], axis=1)  # circle r=0.5 @ (3,3)
    m = np.zeros((60, 8)); m[:, 0:2] = xy
    cx, cy = window_center(m, 0, len(m) - 1)
    _, radius = _mec(xy)
    rc = xy - np.array([cx, cy])
    assert np.max(np.linalg.norm(rc, axis=1)) == pytest.approx(radius, abs=1e-6)
    f0 = xy - xy[0]  # the old frame-0 recenter drifts far past the certified radius
    assert np.max(np.linalg.norm(f0, axis=1)) > radius + 0.3


def test_local_motion_uses_window_center():
    import inspect
    from pipeline.stages import local_motion
    src = inspect.getsource(local_motion)
    assert "window_center(m, s, e)" in src
    assert "seg[:, 0:2] -= seg[0, 0:2]" not in src


# ---- HIGH: promote to show-ready is gated on the clean-run count ----

def test_promote_gated_on_clean_runs(env):
    d = shows.new_dance("Boogie", duration_s=30.0, policy_path="policy.onnx",
                        motion_csv="motion.csv")
    sha = ev.full_sha256(env / "policy.onnx")
    shows.record_sim_run(shows.load_dance(d.id), True, policy_sha256=sha)  # 1 clean
    assert shows.load_dance(d.id).status == "sim-verified"
    with pytest.raises(ValueError):                                        # <3 -> refused
        shows.promote(shows.load_dance(d.id), "show-ready")
    shows.record_sim_run(shows.load_dance(d.id), True, policy_sha256=sha)
    shows.record_sim_run(shows.load_dance(d.id), True, policy_sha256=sha)
    out = shows.promote(shows.load_dance(d.id), "show-ready")
    assert out.status == "show-ready"


# ---- HIGH: library import never trusts an archive's verification state ----

def test_library_import_resets_verification(env, monkeypatch):
    import pipeline.library as lib
    # library uses shows.DANCES_DIR at runtime (already patched by env); it has its own
    # DATA_DIR/PROJECT_ROOT imported from config — patch those to the temp root.
    monkeypatch.setattr(lib, "DATA_DIR", shows.DATA_DIR)
    monkeypatch.setattr(lib, "PROJECT_ROOT", shows.PROJECT_ROOT)
    d = _make_show_ready("Ready")
    assert d.status == "show-ready" and d.policy_sha256
    archive = lib.export_library(env / "backup.tar.gz")
    shutil.rmtree(shows.DANCES_DIR); shows.DANCES_DIR.mkdir(parents=True)
    ids = lib.import_library(archive)
    assert d.id in ids
    imported = shows.load_dance(d.id)
    assert imported.status == "draft"          # forced back to draft on import
    assert imported.policy_sha256 is None
    assert imported.sim_exam is None
    assert (imported.repeatability or {}).get("consecutive_clean", 0) == 0


# ---- data-integrity: dedupe rescues a back-filled file before deleting the loser ----

def test_dedupe_preserves_backfilled_motion_file(env):
    keeper = shows.new_dance("Thriller", duration_s=49.3, policy_path="policy.onnx")
    loser = shows.new_dance("thriller", duration_s=44.3)
    mf = loser.dir / "motion.csv"
    mf.write_text("0,0,0\n")
    loser.motion_csv = str(mf.relative_to(shows.PROJECT_ROOT))
    loser.save()
    shows.dedupe_dances()
    kept = shows.load_dance(keeper.id)
    assert kept.motion_csv
    assert (shows.PROJECT_ROOT / kept.motion_csv).is_file()   # NOT dangling
    assert not loser.dir.exists()


# ---- deploy-safety: an incident outcome demotes; rehearsal never does ----

def test_incident_outcome_demotes_show_ready(env):
    d = _make_show_ready("ShowDance")
    show = shows.new_show(d, "Alois", mode="live")
    shows.record_outcome(show, "incident")
    after = shows.load_dance(d.id)
    assert after.status == "sim-verified"
    assert (after.repeatability or {}).get("consecutive_clean", 0) == 0


def test_rehearsal_incident_does_not_demote(env):
    d = _make_show_ready("RehDance")
    show = shows.new_show(d, "Alois", mode="rehearsal")
    shows.record_outcome(show, "incident")
    assert shows.load_dance(d.id).status == "show-ready"
