"""Content-addressed policy version store + rollback.

Problem (operator safety): a retrain overwrites ``data/policies/<slug>/`` in place
and there is NO history — a bad retrain strands the operator with no way back to a
known-good policy. This module snapshots a policy's deploy artifacts into a
content-addressed store so any prior good version can be recovered.

On-disk layout (pure filesystem, no network), following the reboot-safe plain-JSON
+ atomic-write pattern used by pipeline/store.py and pipeline/shows.py::

    data/policy_store/<dance_id>/<version_id>/
        policy.onnx                the deployed policy (required)
        policy_meta.json           obs layout / action scale (if present)
        <slug>_deploy.csv          reference motion the policy tracks (if present)
        <slug>_deploy.npz          packed deploy bundle (if present)
        manifest.json              snapshot metadata; written LAST as the
                                   all-or-nothing completion marker

``version_id`` is a short (12-hex) prefix of the SHA-256 of ``policy.onnx``'s bytes,
so an identical policy always dedupes to the same version_id (re-snapshotting is a
no-op). The manifest carries the FULL 64-hex ``policy_sha256`` — that, not the short
handle, is the identity used for any integrity check (project lesson: short prefixes
are too weak to authenticate identity; they are fine as a human-readable key).

Rollback is deliberately split: this module never edits the dance library. It copies
the artifacts INTO the store and, on rollback, hands the caller the absolute paths of
a version's files (plus the manifest, via get_version) so the main thread can copy
them back into ``data/policies/<slug>/`` and re-bind with shows.attach_policy /
shows.promote. Everything here is pure filesystem + hashlib.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .config import DATA_DIR

SCHEMA = "policy_store/v1"

# Root of the version store. Referenced through the module global so tests can
# monkeypatch it at an isolated tmp dir (see tests/test_policy_store.py).
POLICY_STORE_DIR = DATA_DIR / "policy_store"

# The deploy artifacts we snapshot, in copy order. policy.onnx is REQUIRED (it is
# what we hash for the version_id); the rest are copied only if present. The two
# ``{slug}`` names are formatted per policy dir at snapshot time.
_META_FILES = ("policy.onnx", "policy_meta.json")
_SLUG_FILES = ("{slug}_deploy.csv", "{slug}_deploy.npz")

# Length of the short content hash used as the version_id / directory name.
_VERSION_ID_LEN = 12


class PolicyStoreError(Exception):
    """Base error for the policy version store."""


class VersionNotFound(PolicyStoreError):
    """No snapshot with the requested (dance_id, version_id) exists."""


# ---- hashing / io helpers --------------------------------------------------------

def _sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    """Full 64-hex SHA-256 of a file, read in chunks (policies + npz are large)."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically + durably (fsync then os.replace), matching the
    pipeline/shows.py::_atomic_write pattern so a power loss can't leave a 0-byte
    manifest."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(payload, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _is_safe_vid(version_id: str) -> bool:
    """A stored version_id is a lowercase-hex sha prefix. Reject anything else so a
    caller-supplied vid can never traverse out of the store (``..`` / separators)."""
    return (isinstance(version_id, str) and 0 < len(version_id) <= 64
            and all(c in "0123456789abcdef" for c in version_id))


def _dance_root(dance_id: str) -> Path:
    return POLICY_STORE_DIR / dance_id


def _version_dir(dance_id: str, version_id: str) -> Path:
    return _dance_root(dance_id) / version_id


def _iter_manifests(dance_id: str):
    """Yield the manifest dict of every complete snapshot for a dance (unreadable /
    partial dirs — no manifest.json yet — are skipped)."""
    root = _dance_root(dance_id)
    if not root.is_dir():
        return
    for d in sorted(root.iterdir()):
        mpath = d / "manifest.json"
        if not mpath.is_file():
            continue
        try:
            yield json.loads(mpath.read_text())
        except (json.JSONDecodeError, OSError):
            continue


def _next_seq(dance_id: str) -> int:
    """Monotonic insertion counter = max existing seq + 1. Stays strictly increasing
    even across prunes (unlike a plain count), so list ordering never collides."""
    mx = -1
    for m in _iter_manifests(dance_id):
        s = m.get("seq")
        if isinstance(s, int) and s > mx:
            mx = s
    return mx + 1


# ---- public API ------------------------------------------------------------------

def snapshot_policy(dance_id: str, policy_dir, *, verdicts=None, note: str | None = None,
                    at_epoch: float | None = None) -> str:
    """Snapshot a policy's deploy artifacts into the content-addressed store.

    Copies policy.onnx (required), policy_meta.json, and <slug>_deploy.csv/.npz (if
    present, where slug = the policy dir's name) into
    ``data/policy_store/<dance_id>/<version_id>/`` and writes manifest.json.

    version_id = the 12-hex prefix of policy.onnx's SHA-256, so an identical policy
    dedupes to the same version_id: a repeat snapshot is a no-op and returns the
    existing id (first-write-wins on note/verdicts — the version IS the content).

    ``at_epoch`` is stored verbatim as the snapshot time — this module never calls the
    clock itself (measurement/provenance discipline); the caller passes the timestamp.

    Returns the version_id.
    """
    policy_dir = Path(policy_dir)
    slug = policy_dir.name
    onnx = policy_dir / "policy.onnx"
    if not onnx.is_file():
        raise PolicyStoreError(f"no policy.onnx in {policy_dir} — nothing to snapshot")

    policy_sha = _sha256_file(onnx)
    version_id = policy_sha[:_VERSION_ID_LEN]
    vdir = _version_dir(dance_id, version_id)
    manifest_path = vdir / "manifest.json"
    if manifest_path.is_file():
        return version_id  # identical policy already captured — dedupe

    candidates = list(_META_FILES) + [n.format(slug=slug) for n in _SLUG_FILES]
    vdir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in candidates:
        src = policy_dir / name
        if src.is_file():
            shutil.copyfile(src, vdir / name)
            copied.append(name)

    deploy_csv = vdir / f"{slug}_deploy.csv"
    motion_sha = _sha256_file(deploy_csv) if deploy_csv.is_file() else None

    manifest = {
        "schema": SCHEMA,
        "version_id": version_id,
        "dance_id": dance_id,
        "slug": slug,
        "policy_sha256": policy_sha,
        "motion_sha256": motion_sha,
        "files": copied,
        "note": note,
        "at_epoch": at_epoch,
        "verdicts": verdicts,
        "seq": _next_seq(dance_id),
    }
    # manifest.json is written LAST: its presence marks the snapshot complete, so a
    # crash mid-copy leaves a dir without a manifest that _iter_manifests ignores and
    # a re-snapshot overwrites.
    _atomic_write_json(manifest_path, manifest)
    return version_id


def list_versions(dance_id: str) -> list[dict]:
    """Every snapshot manifest for a dance, newest first (by insertion seq)."""
    manifests = list(_iter_manifests(dance_id))
    manifests.sort(
        key=lambda m: (m.get("seq") if isinstance(m.get("seq"), int) else -1,
                       m.get("version_id", "")),
        reverse=True)
    return manifests


def get_version(dance_id: str, version_id: str) -> dict:
    """The manifest for one snapshot. Raises VersionNotFound if it does not exist."""
    if not _is_safe_vid(version_id):
        raise VersionNotFound(f"{dance_id}/{version_id}")
    mpath = _version_dir(dance_id, version_id) / "manifest.json"
    try:
        return json.loads(mpath.read_text())
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError) as e:
        raise VersionNotFound(f"{dance_id}/{version_id}") from e


def latest_version(dance_id: str) -> dict | None:
    """The newest snapshot manifest for a dance, or None if it has no snapshots."""
    versions = list_versions(dance_id)
    return versions[0] if versions else None


def rollback_files(dance_id: str, version_id: str) -> dict[str, str]:
    """Absolute paths of a version's deploy artifacts: ``{filename: abs_path}``.

    The caller copies each file back into ``data/policies/<slug>/`` (slug and the
    motion/policy shas come from get_version's manifest) and re-binds via
    shows.attach_policy / shows.promote. This module never touches the dance library.
    Raises VersionNotFound if the version is unknown; only files still present in the
    store are returned.
    """
    manifest = get_version(dance_id, version_id)
    vdir = _version_dir(dance_id, version_id)
    out: dict[str, str] = {}
    for name in manifest.get("files", []):
        p = vdir / name
        if p.is_file():
            out[name] = str(p.resolve())
    return out


def prune(dance_id: str, keep: int) -> int:
    """Delete all but the ``keep`` newest snapshots (by insertion seq). Returns the
    number removed. keep is clamped to >= 0; a store with <= keep versions is a no-op."""
    keep = max(0, int(keep))
    victims = list_versions(dance_id)[keep:]
    removed = 0
    for m in victims:
        vid = m.get("version_id", "")
        if not _is_safe_vid(vid):
            continue
        shutil.rmtree(_version_dir(dance_id, vid), ignore_errors=True)
        removed += 1
    return removed
