"""Content-addressed policy version store + rollback (pipeline/policy_store.py).

Every test runs against an isolated tmp store (monkeypatched POLICY_STORE_DIR) —
pure filesystem, no network, no clock (timestamps are passed in via at_epoch).
"""
from __future__ import annotations

import hashlib

import pytest

from pipeline import policy_store as ps


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Isolate the version store under a tmp dir. Returns the module."""
    monkeypatch.setattr(ps, "POLICY_STORE_DIR", tmp_path / "policy_store")
    return ps


def make_policy_dir(root, slug="thriller", *, onnx=b"ONNX-BYTES-v1",
                    meta=True, csv=True, npz=True):
    """Build a fake deploy policy dir with the artifacts policy_store snapshots."""
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "policy.onnx").write_bytes(onnx)
    if meta:
        (d / "policy_meta.json").write_text('{"control_hz": 50}')
    if csv:
        (d / f"{slug}_deploy.csv").write_text("0,0,0.79\n0,0,0.79\n")
    if npz:
        (d / f"{slug}_deploy.npz").write_bytes(b"NPZ-BUNDLE")
    return d


def test_snapshot_then_list_and_get(store, tmp_path):
    pdir = make_policy_dir(tmp_path / "policies")
    vid = store.snapshot_policy("dance-A", pdir, note="baseline", at_epoch=100.0)

    # version_id is the 12-hex prefix of policy.onnx's sha256
    full = hashlib.sha256((pdir / "policy.onnx").read_bytes()).hexdigest()
    assert vid == full[:12]

    versions = store.list_versions("dance-A")
    assert [m["version_id"] for m in versions] == [vid]

    m = store.get_version("dance-A", vid)
    assert m["dance_id"] == "dance-A"
    assert m["slug"] == "thriller"
    assert m["note"] == "baseline"
    assert m["at_epoch"] == 100.0
    assert m["policy_sha256"] == full
    # all four deploy artifacts were captured, and they exist in the store
    assert set(m["files"]) == {"policy.onnx", "policy_meta.json",
                               "thriller_deploy.csv", "thriller_deploy.npz"}
    vdir = store.POLICY_STORE_DIR / "dance-A" / vid
    for name in m["files"]:
        assert (vdir / name).is_file()


def test_no_onnx_raises(store, tmp_path):
    d = tmp_path / "policies" / "empty"
    d.mkdir(parents=True)
    (d / "policy_meta.json").write_text("{}")
    with pytest.raises(ps.PolicyStoreError):
        store.snapshot_policy("dance-A", d)


def test_identical_policy_dedupes_to_same_version_id(store, tmp_path):
    pdir = make_policy_dir(tmp_path / "policies")
    v1 = store.snapshot_policy("dance-A", pdir, note="first", at_epoch=1.0)
    v2 = store.snapshot_policy("dance-A", pdir, note="second", at_epoch=2.0)
    assert v1 == v2
    # only one physical version, and first-write-wins on metadata
    versions = store.list_versions("dance-A")
    assert len(versions) == 1
    assert versions[0]["note"] == "first"
    assert versions[0]["at_epoch"] == 1.0


def test_different_policy_bytes_new_version(store, tmp_path):
    p1 = make_policy_dir(tmp_path / "p1", slug="thriller", onnx=b"weights-A")
    p2 = make_policy_dir(tmp_path / "p2", slug="thriller", onnx=b"weights-B")
    v1 = store.snapshot_policy("dance-A", p1, at_epoch=1.0)
    v2 = store.snapshot_policy("dance-A", p2, at_epoch=2.0)
    assert v1 != v2
    assert {m["version_id"] for m in store.list_versions("dance-A")} == {v1, v2}


def test_manifest_sha_matches_file_bytes(store, tmp_path):
    pdir = make_policy_dir(tmp_path / "policies", onnx=b"some-real-weights")
    vid = store.snapshot_policy("dance-A", pdir, at_epoch=5.0)
    m = store.get_version("dance-A", vid)
    vdir = store.POLICY_STORE_DIR / "dance-A" / vid

    # policy sha in manifest == sha of the STORED policy.onnx bytes
    stored_policy_sha = hashlib.sha256((vdir / "policy.onnx").read_bytes()).hexdigest()
    assert m["policy_sha256"] == stored_policy_sha
    # and it dedupes off exactly this hash
    assert m["version_id"] == stored_policy_sha[:12]

    # motion sha in manifest == sha of the stored deploy csv bytes
    stored_motion_sha = hashlib.sha256(
        (vdir / "thriller_deploy.csv").read_bytes()).hexdigest()
    assert m["motion_sha256"] == stored_motion_sha


def test_motion_sha_none_when_no_deploy_csv(store, tmp_path):
    pdir = make_policy_dir(tmp_path / "policies", csv=False, npz=False)
    vid = store.snapshot_policy("dance-A", pdir, at_epoch=1.0)
    m = store.get_version("dance-A", vid)
    assert m["motion_sha256"] is None
    assert "thriller_deploy.csv" not in m["files"]
    assert set(m["files"]) == {"policy.onnx", "policy_meta.json"}


def test_rollback_files_returns_existing_paths(store, tmp_path):
    pdir = make_policy_dir(tmp_path / "policies")
    vid = store.snapshot_policy("dance-A", pdir, at_epoch=1.0)
    files = store.rollback_files("dance-A", vid)
    assert set(files) == {"policy.onnx", "policy_meta.json",
                          "thriller_deploy.csv", "thriller_deploy.npz"}
    for name, abspath in files.items():
        from pathlib import Path
        p = Path(abspath)
        assert p.is_absolute() and p.is_file()
        assert p.name == name
    # the caller can copy these back byte-identical to the source
    assert (Path(files["policy.onnx"]).read_bytes()
            == (pdir / "policy.onnx").read_bytes())


def test_rollback_unknown_version_raises(store, tmp_path):
    make_policy_dir(tmp_path / "policies")
    with pytest.raises(ps.VersionNotFound):
        store.rollback_files("dance-A", "deadbeef0000")


def test_get_version_rejects_unsafe_id(store, tmp_path):
    # a traversal attempt must not escape the store; it reads as "not found"
    with pytest.raises(ps.VersionNotFound):
        store.get_version("dance-A", "../../etc/passwd")


def test_latest_and_ordering(store, tmp_path):
    assert store.latest_version("dance-A") is None
    p1 = make_policy_dir(tmp_path / "p1", onnx=b"w1")
    p2 = make_policy_dir(tmp_path / "p2", onnx=b"w2")
    p3 = make_policy_dir(tmp_path / "p3", onnx=b"w3")
    v1 = store.snapshot_policy("dance-A", p1, at_epoch=1.0)
    v2 = store.snapshot_policy("dance-A", p2, at_epoch=2.0)
    v3 = store.snapshot_policy("dance-A", p3, at_epoch=3.0)
    # newest first by insertion order (not lexical version_id order)
    assert [m["version_id"] for m in store.list_versions("dance-A")] == [v3, v2, v1]
    assert store.latest_version("dance-A")["version_id"] == v3


def test_prune_keeps_n_newest(store, tmp_path):
    vids = []
    for i in range(4):
        p = make_policy_dir(tmp_path / f"p{i}", onnx=f"weights-{i}".encode())
        vids.append(store.snapshot_policy("dance-A", p, at_epoch=float(i)))
    removed = store.prune("dance-A", keep=2)
    assert removed == 2
    survivors = [m["version_id"] for m in store.list_versions("dance-A")]
    # the two newest (last inserted) survive
    assert survivors == [vids[3], vids[2]]
    # pruned version dirs are gone from disk
    for vid in (vids[0], vids[1]):
        assert not (store.POLICY_STORE_DIR / "dance-A" / vid).exists()
    # idempotent: pruning again to the same count removes nothing
    assert store.prune("dance-A", keep=2) == 0


def test_prune_after_add_keeps_monotonic_order(store, tmp_path):
    # seq must stay strictly increasing across a prune so a later add is not
    # mis-ordered as "older" than a survivor.
    for i in range(3):
        p = make_policy_dir(tmp_path / f"p{i}", onnx=f"w{i}".encode())
        store.snapshot_policy("dance-A", p, at_epoch=float(i))
    store.prune("dance-A", keep=1)
    p_new = make_policy_dir(tmp_path / "pnew", onnx=b"w-new")
    v_new = store.snapshot_policy("dance-A", p_new, at_epoch=99.0)
    assert store.latest_version("dance-A")["version_id"] == v_new


def test_versions_are_isolated_per_dance(store, tmp_path):
    pa = make_policy_dir(tmp_path / "pa", onnx=b"wa")
    pb = make_policy_dir(tmp_path / "pb", onnx=b"wb")
    store.snapshot_policy("dance-A", pa, at_epoch=1.0)
    store.snapshot_policy("dance-B", pb, at_epoch=1.0)
    assert len(store.list_versions("dance-A")) == 1
    assert len(store.list_versions("dance-B")) == 1
    assert store.list_versions("nope") == []
