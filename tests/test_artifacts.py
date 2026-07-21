"""Contract tests for pipeline/artifacts.py — the frozen shared manifest module
both audit-fix lanes depend on (tasks/audit_fixes_20260721/CONVENTIONS.md)."""
from __future__ import annotations

import json

from pipeline import artifacts


def test_roundtrip_and_verify(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("1,2,3\n")
    man = artifacts.write_manifest(tmp_path / "bundle.json", {
        "motion": {"final_csv": artifacts.file_entry(f, rel_to=tmp_path)},
    })
    assert man["schema"] == artifacts.SCHEMA
    assert man["bundle_id"] == artifacts.compute_bundle_id(man)
    assert artifacts.verify_manifest(tmp_path / "bundle.json") == []


def test_tamper_detection(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("1,2,3\n")
    artifacts.write_manifest(tmp_path / "bundle.json", {
        "motion": {"final_csv": artifacts.file_entry(f, rel_to=tmp_path)},
    })
    f.write_text("9,9,9\n")                       # mutate a member
    errs = artifacts.verify_manifest(tmp_path / "bundle.json")
    assert any("sha mismatch" in e for e in errs)


def test_missing_member_and_edited_manifest(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("x\n")
    mp = tmp_path / "bundle.json"
    artifacts.write_manifest(mp, {
        "motion": {"final_csv": artifacts.file_entry(f, rel_to=tmp_path)}})
    f.unlink()
    errs = artifacts.verify_manifest(mp)
    assert any("missing file" in e for e in errs)
    m = json.loads(mp.read_text())
    m["motion"]["extra"] = "sneaky"               # edit without re-stamping
    mp.write_text(json.dumps(m))
    errs = artifacts.verify_manifest(mp)
    assert any("bundle_id" in e for e in errs)


def test_bundle_id_changes_with_content(tmp_path):
    a = artifacts.write_manifest(tmp_path / "a.json", {"x": 1, "created_at": "t"})
    b = artifacts.write_manifest(tmp_path / "b.json", {"x": 2, "created_at": "t"})
    assert a["bundle_id"] != b["bundle_id"]
