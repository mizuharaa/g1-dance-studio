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

import contextlib
import fcntl
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import exam_verdict
from .config import DATA_DIR, PROJECT_ROOT

DANCES_DIR = DATA_DIR / "dances"
SHOWS_DIR = DATA_DIR / "shows"

DANCE_STATUSES = ("draft", "sim-verified", "show-ready")

# Consecutive clean sim-exam runs required before a dance may be promoted to
# show-ready ("works every time", not "worked once").
REPEATABILITY_TARGET = 3

# Battery floor (%) enforced at the pre-show checklist (findings #13/#29): below this
# there is no safe actuation margin for a full performance — do not start.
BATTERY_FLOOR_PCT = 30

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
    with open(tmp, "w") as f:  # fsync so power loss can't leave a 0-byte record
        f.write(json.dumps(payload, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@contextlib.contextmanager
def _record_lock(record_dir: Path):
    """Serialize load->mutate->save on one record dir (finding #28).

    Without this, two concurrent sim-run POSTs both read the same counter and the
    later save wins — a failing run can be masked by a passing one. flock on a sidecar
    file serializes across threads AND processes (the sim_exam CLI vs the web worker).
    """
    record_dir.mkdir(parents=True, exist_ok=True)
    lock_path = record_dir / ".lock"
    with open(lock_path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _abs(project_rel: str) -> Path:
    """Resolve a stored (possibly project-relative) path to absolute."""
    p = Path(project_rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


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
    policy_sha256: str | None = None    # full sha of the exam-passed policy (finding #24/#27)
    incident: dict | None = None        # last live incident/abort that demoted it (finding #9)
    audio: dict | None = None           # music track + alignment (pipeline/audio.py); see
                                        # docs/show_production.md. None = silent dance.
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


def _norm_name(name: str) -> str:
    """Normalize a dance name for identity: case-insensitive, whitespace-collapsed.
    So "Thriller" and "thriller" are the SAME logical dance (they were duplicating)."""
    return " ".join((name or "").strip().lower().split())


def find_dance(name: str) -> Dance | None:
    target = _norm_name(name)
    for d in list_dances():
        if _norm_name(d.name) == target:
            return d
    return None


def _richness(d: Dance) -> float:
    """How complete a dance record is — used to pick the survivor when de-duping.
    A policy-bearing, sim-verified, show-cut record beats an empty seeded draft."""
    score = 0.0
    if d.policy_path:
        score += 100
    if d.policy_sha256:
        score += 50
    if d.sim_exam:
        score += 40
    score += {"show-ready": 30, "sim-verified": 20, "draft": 0}.get(d.status, 0)
    if d.vet:
        score += 10
    if d.preview:
        score += 5
    if d.motion_csv:
        score += 3
    score += (d.duration_s or 0) * 0.01  # tiebreak: longer / show-cut
    return score


def dedupe_dances() -> int:
    """Merge library entries that are the same logical dance (normalized name).

    Keeps the richest record, back-fills its missing fields from the duplicates
    (so an attached policy is never lost), and removes the rest. Returns the count
    removed. Safe/idempotent: a library with no duplicates is left untouched.
    """
    import shutil
    groups: dict[str, list[Dance]] = {}
    for d in list_dances():
        groups.setdefault(_norm_name(d.name), []).append(d)
    removed = 0
    merge_fields = ("policy_path", "policy_sha256", "sim_exam", "motion_csv",
                    "vet", "preview", "duration_s", "source_job", "incident")
    for dupes in groups.values():
        if len(dupes) < 2:
            continue
        dupes.sort(key=_richness, reverse=True)
        keeper, losers = dupes[0], dupes[1:]
        changed = False
        for loser in losers:
            for fld in merge_fields:
                if not getattr(keeper, fld) and getattr(loser, fld):
                    setattr(keeper, fld, getattr(loser, fld))
                    changed = True
            # never let a de-dupe downgrade status
            if DANCE_STATUSES.index(loser.status) > DANCE_STATUSES.index(keeper.status):
                keeper.status = loser.status
                changed = True
        # Before deleting a loser's dir, rescue any keeper file-field that points
        # inside it (a back-filled path would otherwise dangle). Copy the file into
        # the keeper's own dir and rewrite the path (production audit, data-integrity).
        # (preview lives in data/previews/, never under a dance dir, so it can't dangle)
        file_fields = ("policy_path", "motion_csv")
        for loser in losers:
            try:
                loser_abs = loser.dir.resolve()
            except OSError:
                loser_abs = loser.dir
            for fld in file_fields:
                val = getattr(keeper, fld)
                if not val:
                    continue
                src = _abs(str(val))
                try:
                    inside = loser_abs in src.resolve().parents
                except OSError:
                    inside = False
                if not inside or not src.is_file():
                    continue
                keeper.dir.mkdir(parents=True, exist_ok=True)
                dst = keeper.dir / src.name
                shutil.copyfile(src, dst)
                setattr(keeper, fld, str(dst.relative_to(PROJECT_ROOT)))
                changed = True
        if changed:
            keeper.save()
        for loser in losers:
            shutil.rmtree(loser.dir, ignore_errors=True)
            removed += 1
    return removed


class VerdictError(ValueError):
    """A submitted sim-exam verdict is unauthentic or about a different artifact."""


def record_sim_run_from_verdict(dance_id: str, verdict: dict) -> Dance:
    """Record one sim-exam run from an AUTHENTICATED verdict (findings #23/#24/#26/#27).

    The old endpoint trusted a bare ``passed`` bool from the caller — anyone could
    POST ``{"passed": true}`` and march a dance to show-ready. Now the caller must
    submit a signed ``sim_exam/v1`` verdict; we (a) verify the HMAC, (b) require it to
    bind to THIS dance's exact policy+motion bytes, and (c) DERIVE pass from phase
    contents (all phases ran+passed, push force floor, clean==runs). Only then is the
    clean streak credited, and the exam-passed policy sha is pinned onto the dance.
    """
    with _record_lock(DANCES_DIR / dance_id):
        dance = load_dance(dance_id)  # fresh read under lock (finding #28)
        if not exam_verdict.signature_valid(verdict):
            raise VerdictError("verdict signature invalid or missing — not authentic")
        if not dance.policy_path or not dance.motion_csv:
            raise VerdictError("dance has no registered policy/motion to bind the verdict to")
        policy_sha = exam_verdict.full_sha256(_abs(dance.policy_path))
        motion_sha = exam_verdict.full_sha256(_abs(dance.motion_csv))
        if verdict.get("policy_sha256") != policy_sha:
            raise VerdictError("verdict policy_sha256 does not match this dance's policy file")
        if verdict.get("motion_sha256") != motion_sha:
            raise VerdictError("verdict motion_sha256 does not match this dance's motion file")
        passed = exam_verdict.derive_pass(verdict)
        summary = {
            "nominal": verdict.get("nominal"), "push": verdict.get("push"),
            "repeatability": (verdict.get("repeatability") or {}).get("clean"),
        }
        return _apply_sim_run(dance, passed, policy_sha=policy_sha,
                              metrics=summary, exam_id=verdict.get("at"),
                              video=verdict.get("video"))


def _apply_sim_run(dance: Dance, passed: bool, *, policy_sha: str | None = None,
                   metrics: dict | None = None, exam_id: str | None = None,
                   video: str | None = None) -> Dance:
    """Mutate + persist one run's effect on the dance. Caller holds the record lock."""
    rep = dance.repeatability
    rep["total_runs"] += 1
    rep["consecutive_clean"] = rep["consecutive_clean"] + 1 if passed else 0
    rep["last_run_at"] = time.time()
    rep["history"].insert(0, {"passed": passed, "at": rep["last_run_at"],
                              "exam_id": exam_id, "metrics": metrics or {},
                              "video": video, "policy_sha256": policy_sha})
    del rep["history"][20:]
    verdict_str = "pass" if passed else "fail"
    dance.sim_exam = {"verdict": verdict_str, "at": rep["last_run_at"],
                      "exam_id": exam_id, "metrics": metrics or {},
                      "policy_sha256": policy_sha}
    if passed:
        # pin the exam-passed policy identity so a later swap is detectable (finding #27)
        dance.policy_sha256 = policy_sha
        if dance.status == "draft":
            dance.status = "sim-verified"
    elif dance.status == "show-ready":
        # a failed exam demotes: "works every time" no longer holds (finding #9)
        dance.status = "sim-verified"
    dance.save()
    return dance


def record_sim_run(dance: Dance, passed: bool, metrics: dict | None = None,
                   exam_id: str | None = None, video: str | None = None,
                   policy_sha256: str | None = None) -> Dance:
    """Locking wrapper around _apply_sim_run (reloads fresh under lock, finding #28).

    NOTE: this trusts the caller's ``passed`` — kept only for internal/test callers.
    The web endpoint MUST use record_sim_run_from_verdict, which authenticates."""
    with _record_lock(DANCES_DIR / dance.id):
        fresh = load_dance(dance.id)
        return _apply_sim_run(fresh, passed, policy_sha=policy_sha256,
                              metrics=metrics, exam_id=exam_id, video=video)


def promote(dance: Dance, to_status: str) -> Dance:
    """Human-driven status promotion, with the guard rails that make
    'show-ready' mean something."""
    if to_status not in DANCE_STATUSES:
        raise ValueError(f"unknown status: {to_status}")
    with _record_lock(DANCES_DIR / dance.id):
        dance = load_dance(dance.id)  # fresh read under lock (finding #28)
        if to_status == "show-ready":
            if dance.sim_exam is None or dance.sim_exam.get("verdict") != "pass":
                raise ValueError("cannot promote: latest sim exam has not passed")
            clean = dance.repeatability["consecutive_clean"]
            if clean < REPEATABILITY_TARGET:
                raise ValueError(
                    f"cannot promote: {clean}/{REPEATABILITY_TARGET} consecutive "
                    "clean sim runs")
            # the policy on disk NOW must be the exact one the exam passed (findings
            # #24/#25/#27): re-hash and refuse if it was swapped after the exam.
            if not dance.policy_sha256:
                raise ValueError("cannot promote: no exam-pinned policy sha on record")
            if not dance.policy_path or not _abs(dance.policy_path).exists():
                raise ValueError("cannot promote: policy file is missing")
            current = exam_verdict.full_sha256(_abs(dance.policy_path))
            if current != dance.policy_sha256:
                raise ValueError(
                    "cannot promote: policy file changed since the passing exam "
                    "(sha mismatch) — re-run the sim exam on the current policy")
        dance.status = to_status
        dance.save()
        return dance


def attach_policy(dance_id: str, policy_path: str, *, notes: str | None = None) -> Dance:
    """Attach a trained policy to an already-registered dance (audit HIGH workflow gap).

    The register-first / seeded-dance flow had NO way to set policy_path after
    creation, stranding a trained policy before it could ever be exam-verified.
    Attaching a (new) policy invalidates any prior verification: the sim exam ran
    against a different policy, so we reset the streak, clear the pinned sha and
    verdict, and demote to draft. The operator must re-run the sim exam."""
    with _record_lock(DANCES_DIR / dance_id):
        dance = load_dance(dance_id)
        if not _abs(policy_path).is_file():
            raise ValueError(f"policy file not found: {policy_path}")
        dance.policy_path = policy_path
        if notes:
            dance.notes = notes
        # a different policy ⇒ old exam no longer applies (findings #24/#27)
        dance.policy_sha256 = None
        dance.sim_exam = None
        dance.status = "draft"
        dance.repeatability["consecutive_clean"] = 0
        dance.save()
        return dance


def set_audio(dance_id: str, audio: dict | None) -> Dance:
    """Attach (or clear, with None) a dance's music record. Locked load->save.

    Audio is presentation only — it does NOT touch the policy, verification, or
    show-ready status (music has no bearing on whether the robot stays upright)."""
    with _record_lock(DANCES_DIR / dance_id):
        dance = load_dance(dance_id)
        if audio is not None:
            audio = {**audio, "attached_at": time.time()}
        dance.audio = audio
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
    # "live" = a real paid performance (an incident/abort demotes the dance).
    # "rehearsal" = a dry-run: same flow, logged separately, NEVER demotes the dance.
    mode: str = "live"
    setlist_id: str | None = None   # set if this show is one item of a set-list run

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


def new_show(dance: Dance, operator: str, *, mode: str = "live",
             setlist_id: str | None = None) -> Show:
    if mode not in ("live", "rehearsal"):
        raise ValueError("mode must be 'live' or 'rehearsal'")
    show = Show(id=time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
                dance_id=dance.id, dance_name=dance.name,
                operator=operator, created_at=time.time(), mode=mode,
                setlist_id=setlist_id)
    show.save()
    show.log(f"{mode} show created for dance '{dance.name}' by operator '{operator}'")
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
    if step not in CHECKLIST_KEYS:
        raise ValueError(f"unknown checklist step: {step}")
    with _record_lock(SHOWS_DIR / show.id):
        show = load_show(show.id)  # fresh read under lock (finding #28)
        return _complete_step_locked(show, step, value)


def _complete_step_locked(show: Show, step: str, value=None) -> Show:
    if show.closed:
        raise ValueError("show is closed")
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
        # findings #13/#29: the documented do-not-start battery floor must be ENFORCED,
        # not merely advisory, or a show can be started below actuation margin.
        if step == "battery" and record["value"] < BATTERY_FLOOR_PCT:
            raise ValueError(
                f"battery {record['value']:.0f}% is below the {BATTERY_FLOOR_PCT}% floor — do not start"
            )
    else:
        if value is not True:
            raise ValueError(f"step '{step}' needs an explicit confirmation")
        record["confirmed"] = True
    show.steps[step] = record
    show.save()
    show.log(f"checklist step '{step}' completed: {record}")
    return show


def record_outcome(show: Show, result: str, notes: str = "") -> Show:
    """Close a show with an outcome; an incident/abort demotes the dance (finding #9).

    Both the show close and the dance demotion happen under their record locks
    (finding #28) so a concurrent sim-run cannot race the streak reset."""
    if result not in ("clean", "aborted", "incident"):
        raise ValueError("result must be clean|aborted|incident")
    with _record_lock(SHOWS_DIR / show.id):
        show = load_show(show.id)
        if show.closed:
            raise ValueError("show is already closed")
        show.outcome = {"result": result, "notes": notes, "at": time.time()}
        show.closed = True
        show.save()
        show.log(f"outcome recorded: {result} ({show.mode}) — show closed")
        dance_id = show.dance_id
        is_live = show.mode == "live"
    # Only a LIVE incident/abort demotes the dance. A rehearsal is a dry-run: it is
    # logged for the record but must never knock a show-ready dance out of the library.
    if result in ("incident", "aborted") and is_live:
        with _record_lock(DANCES_DIR / dance_id):
            try:
                dance = load_dance(dance_id)
            except (FileNotFoundError, ValueError):
                dance = None
            if dance is not None:
                dance.repeatability["consecutive_clean"] = 0
                dance.incident = {"show_id": show.id, "result": result, "at": time.time()}
                if dance.status == "show-ready":
                    dance.status = "sim-verified"
                dance.save()
                show.log(f"dance '{dance.name}' demoted + streak reset after {result}")
    return load_show(show.id)


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

    dedupe_dances()  # one-time cleanup of any duplicates from prior restarts/merges

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
