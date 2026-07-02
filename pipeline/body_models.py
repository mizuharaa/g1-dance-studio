"""SMPL / SMPL-X body model installer.

The models are license-gated: the user downloads two zips manually and drops
them (unmodified, any filename) into data/body_models/. This module detects
them by their *contents*, validates, and arranges the canonical layout:

    data/body_models/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl        (GVHMR wants pkl)
    data/body_models/smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.{npz,pkl}
    data/body_models/manifest.json

plus a symlink third_party/GMR/assets/body_models/smplx -> our smplx dir
(GMR loads SMPLX_*.pkl from there). The cloud sync ships data/body_models/
to the GreenNode notebook for GVHMR.

Expected zips (from the user's registrations):
  * SMPL   v1.1.0 "for Python 2.7"  — members like
    SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
    (v1.0.0 lacks the neutral model — flagged as a wrong-download error)
  * SMPL-X v1.1 models zip — members like models/smplx/SMPLX_NEUTRAL.npz

CLI:  python -m pipeline.body_models [--install]
"""
from __future__ import annotations

import json
import re
import sys
import time
import zipfile
from pathlib import Path

from .config import DATA_DIR, THIRD_PARTY

BM_DIR = DATA_DIR / "body_models"
MANIFEST = BM_DIR / "manifest.json"
GMR_LINK = THIRD_PARTY / "GMR" / "assets" / "body_models" / "smplx"

# canonical filename -> regex matched against zip member basenames
SMPL_WANT = {
    "SMPL_NEUTRAL.pkl": re.compile(r"basic[mM]odel_neutral_lbs_10_207_0_v1\.[01]\.[0-9]\.pkl$"),
    "SMPL_MALE.pkl":    re.compile(r"basic[mM]odel_m_lbs_10_207_0_v1\.[01]\.[0-9]\.pkl$"),
    "SMPL_FEMALE.pkl":  re.compile(r"basic[mM]odel_f_lbs_10_207_0_v1\.[01]\.[0-9]\.pkl$"),
}
SMPLX_WANT = {
    f"SMPLX_{g}.{ext}": re.compile(rf"SMPLX_{g}\.{ext}$")
    for g in ("NEUTRAL", "MALE", "FEMALE") for ext in ("npz", "pkl")
}
# sanity floor: real model files are tens of MB; tiny matches = corrupt download
MIN_BYTES = 1_000_000


def _classify_zip(zp: Path) -> tuple[str | None, dict[str, str]]:
    """Return (kind, {canonical_name: member}) for a candidate zip."""
    try:
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
    except zipfile.BadZipFile:
        return None, {}
    found_smpl = {want: m for want, rx in SMPL_WANT.items()
                  for m in names if rx.search(m)}
    found_smplx = {want: m for want, rx in SMPLX_WANT.items()
                   for m in names if rx.search(m)}
    if len(found_smplx) >= 2:
        return "smplx", found_smplx
    if found_smpl:
        return "smpl", found_smpl
    return None, {}


def status() -> dict:
    """What's installed, what zips are waiting, what's missing."""
    BM_DIR.mkdir(parents=True, exist_ok=True)
    installed = {
        "smpl": sorted(p.name for p in (BM_DIR / "smpl").glob("SMPL_*.pkl")),
        "smplx": sorted(p.name for p in (BM_DIR / "smplx").glob("SMPLX_*")),
    }
    zips = []
    for zp in sorted(BM_DIR.glob("*.zip")):
        kind, found = _classify_zip(zp)
        zips.append({"file": zp.name, "detected": kind or "unrecognized",
                     "models_inside": len(found)})
    ok = {"smpl": "SMPL_NEUTRAL.pkl" in installed["smpl"],
          "smplx": any(n.startswith("SMPLX_NEUTRAL") for n in installed["smplx"])}
    return {
        "installed": installed,
        "ready": ok["smpl"] and ok["smplx"],
        "smpl_ok": ok["smpl"],
        "smplx_ok": ok["smplx"],
        "zips": zips,
        "gmr_linked": GMR_LINK.is_dir(),
        "hint": None if (ok["smpl"] and ok["smplx"]) else (
            "Drop the SMPL v1.1.0 zip and the SMPL-X v1.1 models zip into "
            f"{BM_DIR} — see PROJECT_STATE.md 2026-07-02 notes."),
    }


def install() -> dict:
    """Unpack every recognized zip in data/body_models/ into the canonical
    layout. Idempotent; existing installed files are kept unless the zip
    provides them again. Raises RuntimeError with a human-readable reason
    on wrong/partial downloads."""
    BM_DIR.mkdir(parents=True, exist_ok=True)
    zips = sorted(BM_DIR.glob("*.zip"))
    if not zips:
        raise RuntimeError(f"no zip files found in {BM_DIR}")

    report: dict = {"installed": [], "problems": [], "at": time.time()}
    for zp in zips:
        kind, found = _classify_zip(zp)
        if kind is None:
            report["problems"].append(
                f"{zp.name}: no SMPL/SMPL-X model files inside — is this the "
                "right download? (SMPL: 'version 1.1.0 for Python 2.7'; "
                "SMPL-X: 'SMPL-X v1.1' models zip)")
            continue
        if kind == "smpl" and "SMPL_NEUTRAL.pkl" not in found:
            report["problems"].append(
                f"{zp.name}: contains SMPL male/female but NO NEUTRAL model — "
                "this is the v1.0.0 package. Download 'version 1.1.0 for "
                "Python 2.7' from smpl.is.tue.mpg.de instead.")
            continue
        too_small = [m for m in found.values()
                     if zipfile.ZipFile(zp).getinfo(m).file_size < MIN_BYTES]
        if too_small:
            report["problems"].append(
                f"{zp.name}: {too_small[0]} is suspiciously small — corrupt/"
                "partial download, re-download the zip")
            continue
        dest = BM_DIR / kind
        dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            for want, member in found.items():
                with z.open(member) as src, open(dest / want, "wb") as out:
                    while chunk := src.read(1 << 20):
                        out.write(chunk)
                report["installed"].append(f"{kind}/{want}")

    if not report["installed"]:
        raise RuntimeError("nothing installed — " +
                           " | ".join(report["problems"]))

    # GMR expects assets/body_models/smplx/SMPLX_*.pkl
    if (BM_DIR / "smplx").is_dir() and not GMR_LINK.exists():
        GMR_LINK.parent.mkdir(parents=True, exist_ok=True)
        GMR_LINK.symlink_to(BM_DIR / "smplx", target_is_directory=True)
        report["gmr_symlink"] = str(GMR_LINK)

    MANIFEST.write_text(json.dumps(report, indent=2))
    return {**report, "status": status()}


if __name__ == "__main__":
    if "--install" in sys.argv:
        try:
            print(json.dumps(install(), indent=2))
        except RuntimeError as e:
            print(json.dumps({"error": str(e)}, indent=2))
            sys.exit(1)
    else:
        print(json.dumps(status(), indent=2))
