"""Set-lists: an ordered sequence of dances that make up one show.

A single trained dance is one number; a real paid show is a *set* — several
dances in order, with gaps/transitions between them and a total runtime. This
module owns the set-list model and its persistence, following the same
reboot-safe plain-JSON pattern as pipeline/store.py and pipeline/shows.py:

    data/setlists/<setlist_id>/setlist.json

A set-list is only "show-ready" when EVERY dance in it is show-ready — the
resolver surfaces exactly which items block it so the operator can fix them.
Running a set-list still drives the SAME per-dance pre-show checklist + typed
DEPLOY record-only gate (see pipeline/shows.py) for each item in turn; nothing
here contacts the robot, and the single-dance show flow is untouched.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import DATA_DIR

SETLISTS_DIR = DATA_DIR / "setlists"

# Default gap (seconds) between numbers — reposition, breathe, cue the next track.
DEFAULT_GAP_S = 8.0


def _atomic_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:  # fsync: power loss can't leave a 0-byte record
        f.write(json.dumps(payload, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextlib.contextmanager
def _record_lock(record_dir: Path):
    record_dir.mkdir(parents=True, exist_ok=True)
    lock_path = record_dir / ".lock"
    with open(lock_path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


@dataclass
class SetList:
    id: str
    name: str
    created_at: float
    updated_at: float
    # ordered items: [{"dance_id": str, "gap_after_s": float, "note": str}]
    items: list = field(default_factory=list)
    notes: str = ""

    @property
    def dir(self) -> Path:
        return SETLISTS_DIR / self.id

    def save(self) -> None:
        self.updated_at = time.time()
        self.dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.dir / "setlist.json", asdict(self))


def new_setlist(name: str, **kw) -> SetList:
    now = time.time()
    sl = SetList(id=time.strftime("%Y%m%d-") + uuid.uuid4().hex[:8],
                 name=name, created_at=now, updated_at=now, **kw)
    sl.save()
    return sl


def load_setlist(setlist_id: str) -> SetList:
    payload = json.loads((SETLISTS_DIR / setlist_id / "setlist.json").read_text())
    return SetList(**payload)


def list_setlists() -> list[SetList]:
    out = []
    if SETLISTS_DIR.is_dir():
        for d in sorted(SETLISTS_DIR.iterdir()):
            if (d / "setlist.json").exists():
                out.append(load_setlist(d.name))
    out.sort(key=lambda s: s.created_at, reverse=True)
    return out


def delete_setlist(setlist_id: str) -> bool:
    import shutil
    d = SETLISTS_DIR / setlist_id
    if not d.is_dir():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True


def _clean_items(items: list) -> list:
    """Validate/normalize an items list from an API payload."""
    out = []
    for it in items or []:
        if not isinstance(it, dict) or not it.get("dance_id"):
            raise ValueError("each set-list item needs a dance_id")
        gap = it.get("gap_after_s", DEFAULT_GAP_S)
        try:
            gap = float(gap)
        except (TypeError, ValueError):
            raise ValueError("gap_after_s must be a number")
        if gap < 0:
            raise ValueError("gap_after_s must be >= 0")
        out.append({"dance_id": str(it["dance_id"]),
                    "gap_after_s": round(gap, 3),
                    "note": str(it.get("note") or "")})
    return out


def set_items(setlist_id: str, items: list) -> SetList:
    """Replace the ordered items (drag-to-reorder / add / remove all go through here)."""
    cleaned = _clean_items(items)
    with _record_lock(SETLISTS_DIR / setlist_id):
        sl = load_setlist(setlist_id)
        sl.items = cleaned
        sl.save()
        return sl


def rename(setlist_id: str, name: str, notes: str | None = None) -> SetList:
    name = (name or "").strip()
    if not name:
        raise ValueError("set-list needs a name")
    with _record_lock(SETLISTS_DIR / setlist_id):
        sl = load_setlist(setlist_id)
        sl.name = name
        if notes is not None:
            sl.notes = notes
        sl.save()
        return sl


def resolve(sl: SetList, dance_lookup) -> dict:
    """Join a set-list against the current dance library into a runnable view.

    `dance_lookup(dance_id)` returns a pipeline.shows.Dance or None. Produces the
    per-item status (present? show-ready? duration, has music), the total runtime
    (dance durations + gaps), and the blocking items — the set-list is show-ready
    only if every item resolves to a show-ready dance.
    """
    resolved, total, blockers = [], 0.0, []
    for i, it in enumerate(sl.items):
        d = None
        try:
            d = dance_lookup(it["dance_id"])
        except Exception:
            d = None
        present = d is not None
        ready = present and d.status == "show-ready"
        dur = (d.duration_s or 0.0) if present else 0.0
        gap = it.get("gap_after_s", DEFAULT_GAP_S)
        total += dur + (gap if i < len(sl.items) - 1 else 0.0)
        if not ready:
            blockers.append({"index": i, "dance_id": it["dance_id"],
                             "reason": "missing" if not present else f"status is {d.status}",
                             "name": (d.name if present else it["dance_id"])})
        resolved.append({
            "index": i,
            "dance_id": it["dance_id"],
            "name": d.name if present else "(missing dance)",
            "present": present,
            "status": d.status if present else None,
            "show_ready": ready,
            "duration_s": dur,
            "has_audio": bool(present and d.audio),
            "gap_after_s": gap,
            "note": it.get("note", ""),
        })
    return {
        "id": sl.id, "name": sl.name, "notes": sl.notes,
        "created_at": sl.created_at, "updated_at": sl.updated_at,
        "items": resolved,
        "count": len(sl.items),
        "total_runtime_s": round(total, 3),
        "show_ready": len(sl.items) > 0 and not blockers,
        "blockers": blockers,
    }


SETLISTS_DIR.mkdir(parents=True, exist_ok=True)
