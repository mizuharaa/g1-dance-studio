"""Export / import the dance library as a single portable archive.

Audit HIGH (disaster recovery): every trained dance lives only under data/ on one
no-GPU laptop, with no way to back it up or move it. A disk failure loses weeks of
paid training. This bundles each dance's metadata + its referenced motion/policy/
preview files into one relocatable .tar.gz, and restores it on another machine with
paths rewritten to that machine's layout.

Archive layout:
    manifest.json                 {schema, exported_at, dances:[id...]}
    dances/<id>/dance.json        the record, with file paths rewritten to be
    dances/<id>/motion.csv        archive-relative (./motion.csv etc.)
    dances/<id>/policy<ext>
    dances/<id>/preview.mp4
"""
from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

from . import shows
from .config import DATA_DIR, PROJECT_ROOT

SCHEMA = "dance_library/v1"
EXPORTS_DIR = DATA_DIR / "exports"

# Dance fields that reference an on-disk file we must bundle and rewrite.
_FILE_FIELDS = {"motion_csv": "motion.csv", "policy_path": None, "preview": "preview.mp4"}


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def export_library(dest: Path | None = None) -> Path:
    """Bundle every registered dance into a .tar.gz. Returns the archive path."""
    return _export(shows.list_dances(), dest)


def export_dance(dance_id: str, dest: Path | None = None) -> Path:
    return _export([shows.load_dance(dance_id)], dest)


def _export(dances: list, dest: Path | None) -> Path:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if dest is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dest = EXPORTS_DIR / f"dance-library-{stamp}.tar.gz"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ids = []
        for d in dances:
            ddir = root / "dances" / d.id
            ddir.mkdir(parents=True, exist_ok=True)
            record = json.loads(json.dumps(_dance_asdict(d)))  # deep copy
            for field, default_name in _FILE_FIELDS.items():
                val = getattr(d, field, None)
                if not val:
                    continue
                src = _resolve(val.lstrip("/") if field == "preview" and
                               val.startswith("/previews/") else val)
                # preview URLs (/previews/x.mp4) map to data/previews/x.mp4
                if field == "preview" and val.startswith("/previews/"):
                    src = DATA_DIR / "previews" / Path(val).name
                if not src.is_file():
                    record[field] = None  # dangling reference — drop it
                    continue
                arc_name = default_name or ("policy" + src.suffix)
                shutil.copyfile(src, ddir / arc_name)
                record[field] = f"./{arc_name}"
            (ddir / "dance.json").write_text(json.dumps(record, indent=2))
            ids.append(d.id)
        (root / "manifest.json").write_text(json.dumps(
            {"schema": SCHEMA, "exported_at": time.time(), "dances": ids}, indent=2))
        with tarfile.open(dest, "w:gz") as tar:
            tar.add(root, arcname=".")
    return dest


def _dance_asdict(d) -> dict:
    from dataclasses import asdict
    return asdict(d)


def import_library(archive: Path, *, overwrite: bool = False) -> list[str]:
    """Restore dances from an archive into data/dances. Returns imported ids.

    Files are copied into each dance's data/dances/<id>/ dir and the record's
    paths rewritten to point there, so the library is self-contained on this
    machine. A colliding id is skipped unless overwrite=True."""
    archive = Path(archive)
    if not archive.is_file():
        raise ValueError(f"archive not found: {archive}")
    imported: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with tarfile.open(archive, "r:gz") as tar:
            _safe_extract(tar, root)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"unrecognized archive schema: {manifest.get('schema')}")
        for did in manifest.get("dances", []):
            src_dir = root / "dances" / did
            rec_path = src_dir / "dance.json"
            if not rec_path.is_file():
                continue
            record = json.loads(rec_path.read_text())
            # Trust boundary: never import a verification state. A backup archive is
            # unauthenticated, so a crafted dance.json could inject a fake "show-ready"
            # dance and bypass the signed-verdict gate. Reset to draft and force a fresh
            # local signed sim-exam before it can be promoted (production audit, HIGH).
            record["status"] = "draft"
            record["sim_exam"] = None
            record["policy_sha256"] = None
            rep = record.get("repeatability")
            if isinstance(rep, dict):
                rep["consecutive_clean"] = 0
            dest_dir = shows.DANCES_DIR / did
            if dest_dir.exists() and not overwrite:
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            for field, _ in _FILE_FIELDS.items():
                val = record.get(field)
                if not val or not str(val).startswith("./"):
                    continue
                fsrc = src_dir / Path(val).name
                if not fsrc.is_file():
                    record[field] = None
                    continue
                shutil.copyfile(fsrc, dest_dir / fsrc.name)
                # rewrite to a project-relative path the app understands
                rel = (dest_dir / fsrc.name).relative_to(PROJECT_ROOT)
                record[field] = f"/previews/{fsrc.name}" if field == "preview" \
                    else str(rel)
                if field == "preview":
                    (DATA_DIR / "previews").mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(fsrc, DATA_DIR / "previews" / fsrc.name)
            (dest_dir / "dance.json").write_text(json.dumps(record, indent=2))
            imported.append(did)
    return imported


def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    """Extract guarding against path traversal (../ or absolute members)."""
    base = path.resolve()
    for member in tar.getmembers():
        target = (base / member.name).resolve()
        if not str(target).startswith(str(base)):
            raise ValueError(f"unsafe path in archive: {member.name}")
    tar.extractall(path)
