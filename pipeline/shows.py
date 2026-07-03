"""Show-mode persistence: dance library, performance records, repeatability.

Two on-disk collections, following the same reboot-safe plain-JSON pattern as
pipeline/store.py (atomic writes, one directory per record):

    data/dances/<dance_id>/dance.json    registered dance (library entry)
    data/shows/<show_id>/show.json       one performance: checklist -> deploy -> outcome
    data/shows/<show_id>/show-log.txt    append-only human-readable event log

A *dance* is a choreography that has been (or is being) turned into a trained
policy. Its `status` walks draft -> sim-verified -> show-ready:

    draft         registered, policy missing or unverified
    sim-verified  latest sim exam passed
    show-ready    sim-verified AND >= REPEATABILITY_TARGET consecutive clean
                  sim runs AND a human explicitly promoted it

A *show* is one live performance attempt: the operator completes the pre-show
checklist step by step, the deploy gate unlocks only when every step is done,
and the deploy itself remains RECORD-ONLY (nothing is ever sent to the robot
from here — hardware deployment is a separate, human-driven phase).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import DATA_DIR

DANCES_DIR = DATA_DIR / "dances"
SHOWS_DIR = DATA_DIR / "shows"

DANCE_STATUSES = ("draft", "sim-verified", "show-ready")

# Consecutive clean sim-exam runs required before a dance may be promoted to
# show-ready ("works every time", not "worked once").
REPEATABILITY_TARGET = 3

# Pre-show checklist, in mandatory order. kind: "confirm" = operator must
# explicitly confirm; "number" = operator enters a value (e.g. battery %).
CHECKLIST_STEPS: list[dict] = [
    {"key": "robot_health", "kind": "confirm",
     "title": "Robot health",
     "detail": "Power the robot, wait for it to settle, confirm no error LEDs. "
               "(Automatic health ping arrives with the deploy phase — for now "
               "this is a manual check.)"},
    {"key": "space_clear", "kind": "confirm",
     "title": "Performance area clear",
     "detail": "Hard flat floor, 2 m radius fully clear of people and objects."},
    {"key": "battery", "kind": "number",
     "title": "Battery level (%)",
     "detail": "Read the battery percentage from the robot and enter it. "
               "Below 30% do not start a show."},
    {"key": "estop", "kind": "confirm",
     "title": "E-stop in hand",
     "detail": "Hold the e-stop remote, test that it responds, keep it in hand "
               "for the entire performance."},
    {"key": "venue_ack", "kind": "confirm",
     "title": "Venue limits acknowledged",
     "detail": "This dance was safety-checked for a ≤2 m radius on hard flat "
               "ground. Confirm the venue matches."},
]
CHECKLIST_KEYS = [s["key"] for s in CHECKLIST_STEPS]


def _atomic_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


# ---- dance library ---------------------------------------------------------------

@dataclass
class Dance:
    id: str
    name: str
    created_at: float
    updated_at: float
    status: str = "draft"
    duration_s: float | None = None
    motion_csv: str | None = None      # project-relative path to the G1 motion
    policy_path: str | None = None     # project-relative path to the trained policy
    preview: str | None = None         # /previews/... URL or project-relative path
    vet: dict | None = None            # vet_motion.py JSON report (embedded)
    sim_exam: dict | None = None       # latest sim-exam verdict summary
    source_job: str | None = None      # pipeline job id this dance came from
    notes: str = ""
    # Repeatability: updated by the sim-exam tool via POST /api/dances/{id}/sim-runs
    # (JSON contract in docs/show_mode_contracts.md).
    repeatability: dict = field(default_factory=lambda: {
        "consecutive_clean": 0, "total_runs": 0, "last_run_at": None,
        "history": [],  # newest first, capped
    })

    @property
    def dir(self) -> Path:
        return DANCES_DIR / self.id

    def save(self) -> None:
        self.updated_at = time.time()
        self.dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.dir / "dance.json", asdict(self))


def new_dance(name: str, **kw) -> Dance:
    now = time.time()
    dance = Dance(id=time.strftime("%Y%m%d-") + uuid.uuid4().hex[:8],
                  name=name, created_at=now, updated_at=now, **kw)
    dance.save()
    return dance


def load_dance(dance_id: str) -> Dance:
    payload = json.loads((DANCES_DIR / dance_id / "dance.json").read_text())
    return Dance(**payload)


def list_dances() -> list[Dance]:
    out = []
    if DANCES_DIR.is_dir():
        for d in sorted(DANCES_DIR.iterdir()):
            if (d / "dance.json").exists():
                out.append(load_dance(d.name))
    out.sort(key=lambda x: x.created_at, reverse=True)
    return out


def find_dance(name: str) -> Dance | None:
    for d in list_dances():
        if d.name == name:
            return d
    return None


def record_sim_run(dance: Dance, passed: bool, metrics: dict | None = None,
                   exam_id: str | None = None, video: str | None = None) -> Dance:
    """Record one sim-exam run (the repeatability contract's server side)."""
    rep = dance.repeatability
    rep["total_runs"] += 1
    rep["consecutive_clean"] = rep["consecutive_clean"] + 1 if passed else 0
    rep["last_run_at"] = time.time()
    rep["history"].insert(0, {"passed": passed, "at": rep["last_run_at"],
                              "exam_id": exam_id, "metrics": metrics or {},
                              "video": video})
    del rep["history"][20:]
    if passed:
        dance.sim_exam = {"verdict": "pass", "at": rep["last_run_at"],
                          "exam_id": exam_id, "metrics": metrics or {}}
        if dance.status == "draft":
            dance.status = "sim-verified"
    else:
        dance.sim_exam = {"verdict": "fail", "at": rep["last_run_at"],
                          "exam_id": exam_id, "metrics": metrics or {}}
        if dance.status == "show-ready":
            # A failed exam demotes: "works every time" no longer holds.
            dance.status = "sim-verified"
    dance.save()
    return dance


def promote(dance: Dance, to_status: str) -> Dance:
    """Human-driven status promotion, with the guard rails that make
    'show-ready' mean something."""
    if to_status not in DANCE_STATUSES:
        raise ValueError(f"unknown status: {to_status}")
    if to_status == "show-ready":
        if dance.sim_exam is None or dance.sim_exam.get("verdict") != "pass":
            raise ValueError("cannot promote: latest sim exam has not passed")
        clean = dance.repeatability["consecutive_clean"]
        if clean < REPEATABILITY_TARGET:
            raise ValueError(
                f"cannot promote: {clean}/{REPEATABILITY_TARGET} consecutive "
                "clean sim runs")
    dance.status = to_status
    dance.save()
    return dance


# ---- shows (performances) --------------------------------------------------------

@dataclass
class Show:
    id: str
    dance_id: str
    dance_name: str
    operator: str
    created_at: float
    # step key -> {"at": ts, "value": ..., "confirmed": True}
    steps: dict = field(default_factory=dict)
    deploy: dict | None = None      # {"requested_at": ts, "note": ...} record-only
    outcome: dict | None = None     # {"result": "clean"|"aborted"|"incident", "notes", "at"}
    closed: bool = False

    @property
    def dir(self) -> Path:
        return SHOWS_DIR / self.id

    def log(self, msg: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.dir / "show-log.txt", "a") as f:
            f.write(f"[{stamp}] {msg}\n")

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.dir / "show.json", asdict(self))

    def next_step(self) -> str | None:
        for key in CHECKLIST_KEYS:
            if key not in self.steps:
                return key
        return None

    def checklist_complete(self) -> bool:
        return self.next_step() is None


def new_show(dance: Dance, operator: str) -> Show:
    show = Show(id=time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
                dance_id=dance.id, dance_name=dance.name,
                operator=operator, created_at=time.time())
    show.save()
    show.log(f"show created for dance '{dance.name}' by operator '{operator}'")
    return show


def load_show(show_id: str) -> Show:
    payload = json.loads((SHOWS_DIR / show_id / "show.json").read_text())
    return Show(**payload)


def list_shows() -> list[Show]:
    out = []
    if SHOWS_DIR.is_dir():
        for d in sorted(SHOWS_DIR.iterdir(), reverse=True):
            if (d / "show.json").exists():
                out.append(load_show(d.name))
    return out


def complete_step(show: Show, step: str, value=None) -> Show:
    """Record one checklist step. Steps must be completed in order."""
    if show.closed:
        raise ValueError("show is closed")
    if step not in CHECKLIST_KEYS:
        raise ValueError(f"unknown checklist step: {step}")
    expected = show.next_step()
    if step != expected:
        raise ValueError(f"steps must be completed in order — next is '{expected}'")
    spec = next(s for s in CHECKLIST_STEPS if s["key"] == step)
    record: dict = {"at": time.time()}
    if spec["kind"] == "number":
        # bool is an int subclass: float(True) == 1.0 — reject it explicitly.
        if isinstance(value, bool):
            raise ValueError(f"step '{step}' needs a numeric value")
        try:
            record["value"] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"step '{step}' needs a numeric value")
        if not 0 <= record["value"] <= 100:
            raise ValueError(f"step '{step}' value must be between 0 and 100")
    else:
        if value is not True:
            raise ValueError(f"step '{step}' needs an explicit confirmation")
        record["confirmed"] = True
    show.steps[step] = record
    show.save()
    show.log(f"checklist step '{step}' completed: {record}")
    return show


# ---- seeding ----------------------------------------------------------------------

def seed_initial_dances() -> None:
    """Idempotently register the two known dances if artifacts are present.

    Runs at server startup. Looks for the canonical test segment CSV and the
    Thriller pipeline job (or its committed motion CSV) and registers whatever
    exists that isn't registered yet. Policies are pending for both — status
    stays 'draft' until sim exams report in.
    """
    from . import store  # deferred: avoid import cycle at module load
    from .config import PROJECT_ROOT

    if find_dance("test-segment") is None:
        csv = DATA_DIR / "dance1_subject2_seg.csv"
        if csv.exists():
            new_dance("test-segment", duration_s=28.8,
                      motion_csv=str(csv.relative_to(PROJECT_ROOT)),
                      notes="LAFAN1 dance1_subject2 deployable window — canonical "
                            "benchmark motion. Policy: training in progress.")

    if find_dance("thriller") is None:
        motion_csv = duration = vet = preview = source_job = None
        try:  # preferred source: the completed pipeline job
            for job in store.list_jobs():
                if job.name == "thriller" and job.stages["retarget"].state == "done":
                    source_job = job.id
                    vet_file = job.dir / "retarget" / "vet.json"
                    vet = json.loads(vet_file.read_text()) if vet_file.exists() else None
                    if (job.dir / "retarget" / "preview.mp4").exists():
                        preview = f"/previews/job-{job.id}.mp4"
                    for c in (job.dir / "retarget").glob("*.csv"):
                        motion_csv = str(c)
                        break
                    if vet:
                        duration = vet.get("seconds")
                    break
        except Exception:
            pass
        if motion_csv is None:  # fallback: committed motion CSV
            hits = sorted((DATA_DIR / "motions").glob("thriller*/**/*.csv")) \
                if (DATA_DIR / "motions").is_dir() else []
            if hits:
                motion_csv = str(hits[0])
        if motion_csv or source_job:
            new_dance("thriller", duration_s=duration or 44.3,
                      motion_csv=motion_csv, vet=vet, preview=preview,
                      source_job=source_job,
                      notes="User's Thriller reference video (44.3 s, vet PASS "
                            "full length). Policy: pending first training.")


DANCES_DIR.mkdir(parents=True, exist_ok=True)
SHOWS_DIR.mkdir(parents=True, exist_ok=True)
