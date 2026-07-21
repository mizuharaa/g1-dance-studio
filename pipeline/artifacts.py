"""Shared artifact-identity contract for the 2026-07-21 audit-fix wave.

FROZEN COMMON GROUND for both work lanes (see tasks/audit_fixes_20260721/
CONVENTIONS.md). Lane A (motion/eval) and Lane B (policy/show) both import this;
NEITHER lane edits it after the pack commit without a coordination note in the
task-pack README. It exists so the two halves produce/consume the SAME manifest
shape instead of inventing two.

Manifest = "g1.bundle/1": one JSON file binding every member of a dance bundle
by full SHA-256. Producers fill only the sections they own (motion by Lane A's
rebuild, policy by Lane B's exporter); consumers verify whatever sections are
present. bundle_id = sha256 of the canonical JSON with the bundle_id field
removed, so the ID is content-addressed and any member edit changes it.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

SCHEMA = "g1.bundle/1"


def sha256_file(path: str | Path) -> str:
    """Full lowercase hex SHA-256 of a file, streamed (works for GB files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(d: dict) -> bytes:
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode()


def compute_bundle_id(manifest: dict) -> str:
    body = {k: v for k, v in manifest.items() if k != "bundle_id"}
    return hashlib.sha256(_canonical(body)).hexdigest()


def write_manifest(path: str | Path, manifest: dict) -> dict:
    """Stamp schema/created_at/bundle_id and write canonical-ish JSON. Returns
    the stamped manifest. Caller supplies the content sections (see CONVENTIONS
    for the section shapes: motion / model / policy / eval)."""
    m = dict(manifest)
    m["schema"] = SCHEMA
    m.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    m["bundle_id"] = compute_bundle_id(m)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2, sort_keys=True))
    return m


def verify_manifest(path: str | Path, base_dir: str | Path | None = None) -> list[str]:
    """Verify a manifest: schema, self-consistent bundle_id, and — for every
    entry anywhere in the tree shaped {"path": ..., "sha256": ...} — that the
    file exists and its hash matches. Returns a list of human-readable errors
    (empty == verified). Never raises on content problems; raises only if the
    manifest file itself is unreadable."""
    p = Path(path)
    m = json.loads(p.read_text())
    base = Path(base_dir) if base_dir is not None else p.parent
    errs: list[str] = []
    if m.get("schema") != SCHEMA:
        errs.append(f"schema {m.get('schema')!r} != {SCHEMA!r}")
    if m.get("bundle_id") != compute_bundle_id(m):
        errs.append("bundle_id does not match content (manifest edited?)")

    def walk(node, crumb):
        if isinstance(node, dict):
            if "path" in node and "sha256" in node:
                f = Path(node["path"])
                if not f.is_absolute():
                    f = base / f
                if not f.exists():
                    errs.append(f"{crumb}: missing file {node['path']}")
                elif sha256_file(f) != node["sha256"]:
                    errs.append(f"{crumb}: sha mismatch for {node['path']}")
            for k, v in node.items():
                walk(v, f"{crumb}.{k}" if crumb else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{crumb}[{i}]")

    walk(m, "")
    return errs


def file_entry(path: str | Path, rel_to: str | Path | None = None) -> dict:
    """Build a {"path","sha256"} entry; path stored relative to rel_to if given."""
    p = Path(path)
    stored = str(p.relative_to(rel_to)) if rel_to is not None else str(p)
    return {"path": stored, "sha256": sha256_file(p)}
