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

from . import show_audio  # cue-offset math + timeline constants (no robot at import)
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


# ---- show-time run: per-item state machine + audio-cue plan -----------------------
#
# Presentation / orchestration ONLY. This layer owns the ORDER of numbers, the
# per-item run state, and WHEN each number's music should start. It never gates
# show-readiness (that lives in pipeline/shows.py) and never contacts the robot or
# plays a note — it only COMPUTES a plan. The audio timing reuses
# pipeline/show_audio.py's tick0 + RAMP_S + audio_delay_s contract (2.5 s activation
# ramp + 1.5 s standing lead-in = 4.0 s for the default prep) so the operator/UI and
# tools/show_run.sh share one source of truth for the cue offset.

# Per-item run states. A set-list run walks each number pending -> running -> done;
# a number the operator kills mid-show goes running -> aborted (and may be retried).
RUN_STATES = ("pending", "running", "done", "aborted")

# Legal state transitions (setting a state to itself is always an idempotent no-op).
# `done` is terminal so a completed number can't be silently resurrected mid-show.
_LEGAL_TRANSITIONS = {
    "pending": {"pending", "running", "aborted", "done"},
    "running": {"running", "done", "aborted"},
    "aborted": {"aborted", "running", "pending"},
    "done": {"done"},
}


@dataclass
class SetListRun:
    """Durable per-item run state for one set-list (resume-safe across reboots).

    Stored next to the set-list at data/setlists/<id>/run.json. ``states[i]`` is the
    RUN_STATES value for ``SetList.items[i]``; resume replays from the first item that
    is not yet ``done``, so a mid-show crash re-enters exactly where it left off.
    """
    id: str                      # == the set-list id
    states: list                 # aligned 1:1 with SetList.items, each in RUN_STATES
    created_at: float
    updated_at: float
    note: str = ""

    @property
    def dir(self) -> Path:
        return SETLISTS_DIR / self.id

    def save(self) -> None:
        self.updated_at = time.time()
        self.dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.dir / "run.json", asdict(self))


def _fresh_run(sl: SetList) -> SetListRun:
    now = time.time()
    return SetListRun(id=sl.id, states=["pending"] * len(sl.items),
                      created_at=now, updated_at=now)


def new_run(sl: SetList) -> SetListRun:
    """Start (or restart) a run: every item back to pending. Persists immediately."""
    run = _fresh_run(sl)
    run.save()
    return run


def load_run(setlist_id: str) -> SetListRun | None:
    """Load the persisted run for a set-list, or None if none has started."""
    p = SETLISTS_DIR / setlist_id / "run.json"
    if not p.exists():
        return None
    return SetListRun(**json.loads(p.read_text()))


def _reconcile_len(run: SetListRun, n: int) -> bool:
    """Pad/truncate states to n items (set-list edited since the run started).
    Preserves existing states by index; new tail items start pending. Returns
    True iff the states list changed."""
    if len(run.states) == n:
        return False
    run.states = (run.states + ["pending"] * n)[:n]
    return True


def get_or_create_run(sl: SetList) -> SetListRun:
    """Load the run for `sl`, creating a fresh all-pending run if none exists.
    Reconciles the state-list length if the set-list was edited since it started."""
    run = load_run(sl.id)
    if run is None:
        return new_run(sl)
    if _reconcile_len(run, len(sl.items)):
        run.save()
    return run


def set_item_state(setlist_id: str, index: int, to_state: str) -> SetListRun:
    """Transition one item's run state (locked load->mutate->save, finding #28).

    Raises ValueError on an illegal transition (e.g. resurrecting a ``done`` number)
    or unknown state, and IndexError for an out-of-range item. Auto-creates an
    all-pending run if the operator advances an item before new_run()."""
    if to_state not in RUN_STATES:
        raise ValueError(f"unknown run state {to_state!r} (want one of {RUN_STATES})")
    with _record_lock(SETLISTS_DIR / setlist_id):
        run = load_run(setlist_id)
        if run is None:
            run = _fresh_run(load_setlist(setlist_id))
        if not 0 <= index < len(run.states):
            raise IndexError(
                f"item index {index} out of range (0..{len(run.states) - 1})")
        cur = run.states[index]
        if to_state not in _LEGAL_TRANSITIONS[cur]:
            raise ValueError(
                f"illegal transition {cur!r} -> {to_state!r} for item {index}")
        run.states[index] = to_state
        run.save()
        return run


def next_index(run: SetListRun) -> int | None:
    """The next item to run: the first that is not yet ``done`` (resume skips
    already-done numbers). None once every item is done."""
    for i, s in enumerate(run.states):
        if s != "done":
            return i
    return None


def remaining_indices(run: SetListRun) -> list:
    """Indices of every item not yet ``done``, in order (the resume worklist)."""
    return [i for i, s in enumerate(run.states) if s != "done"]


def _default_lookup(dance_id: str):
    """Fallback dance resolver when a caller passes no lookup: the live library."""
    from . import shows  # deferred: avoid an import cycle / hard dep at module load
    try:
        return shows.load_dance(dance_id)
    except (FileNotFoundError, ValueError):
        return None


def _audio_plan(dance) -> dict | None:
    """Show-time music cue for one dance, or None for a silent number. Reuses
    pipeline/show_audio.py's tick0->music offset (RAMP_S + the record's
    audio_delay_s); a bare audio record with no alignment falls back to the 4.0 s
    default prep."""
    audio = getattr(dance, "audio", None)
    if not audio:
        return None
    align = audio.get("align") if isinstance(audio, dict) else None
    try:
        offset = show_audio.cue_offset_for_align(align)
    except (KeyError, TypeError, ValueError):
        offset = show_audio.DEFAULT_OFFSET_S
    return {"track": (audio.get("track") if isinstance(audio, dict) else None),
            "offset_s": round(float(offset), 3)}


def setlist_run_plan(sl: SetList, dance_lookup=None,
                     run: SetListRun | None = None) -> list:
    """The consumable show-run plan: one entry per set-list item, in order, with its
    live-library status, run state, per-item blockers, and (if it has attached music)
    the show-time audio cue. Consumed by the operator/UI and tools/show_run.sh.

    `dance_lookup(dance_id) -> Dance|None` defaults to the live library
    (pipeline.shows.load_dance). `run` (a SetListRun) supplies per-item states;
    without it every item reads ``pending``. Blockers are the show-readiness reasons
    an item cannot run (missing / not show-ready) — audio is optional, so a silent
    number is never a blocker.
    """
    lookup = dance_lookup or _default_lookup
    states = run.states if run is not None else None
    plan = []
    for i, it in enumerate(sl.items):
        try:
            d = lookup(it["dance_id"])
        except Exception:
            d = None
        present = d is not None
        status = d.status if present else None
        show_ready = bool(present and status == "show-ready")
        blockers = []
        if not present:
            blockers.append("missing")
        elif not show_ready:
            blockers.append(f"status is {status}")
        state = states[i] if states is not None and i < len(states) else "pending"
        plan.append({
            "index": i,
            "dance_id": it["dance_id"],
            "name": d.name if present else "(missing dance)",
            "status": status,
            "present": present,
            "show_ready": show_ready,
            "state": state,
            "duration_s": (d.duration_s or 0.0) if present else 0.0,
            "gap_after_s": it.get("gap_after_s", DEFAULT_GAP_S),
            "has_audio": bool(present and getattr(d, "audio", None)),
            "audio": _audio_plan(d) if present else None,
            "note": it.get("note", ""),
            "blockers": blockers,
        })
    return plan


def plan_runnable(plan: list) -> bool:
    """Whole-set-list gate: runnable only if it is non-empty and EVERY item is
    show-ready (no per-item blockers). Mirrors resolve()'s all-or-nothing rule —
    a single blocked number holds the whole set."""
    return len(plan) > 0 and all(not it["blockers"] for it in plan)


def plan_blockers(plan: list) -> list:
    """Flatten the items that block the run into a 'fix these first' list."""
    return [{"index": it["index"], "dance_id": it["dance_id"],
             "name": it["name"], "blockers": it["blockers"]}
            for it in plan if it["blockers"]]


SETLISTS_DIR.mkdir(parents=True, exist_ok=True)
