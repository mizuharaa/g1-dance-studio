"""Reboot-safe job persistence.

A *job* is one dance video flowing through the pipeline. Everything about a job
lives under data/jobs/<job_id>/:

    job.json          status of every stage (this file is the resume point)
    input.mp4         uploaded reference video
    smpl/             stage outputs: extracted human motion
    motion/           retargeted G1 reference motion
    policy/           trained policy + training metadata (cloud job ids)
    verify/           sim2sim report
    export/           final deployable bundle
    log.txt           append-only human-readable event log

State is plain JSON on disk, written atomically, so a laptop reboot mid-stage
loses nothing: the runner re-enters the first stage not marked "done".
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import JOBS_DIR, STAGE_ORDER

STAGE_STATES = ("pending", "running", "done", "failed", "skipped", "blocked")


class CorruptJobError(Exception):
    """A job.json is unreadable/unparseable. Raised by load_job; list_jobs skips
    these so one bad file can't brick startup or the whole job list (audit HIGH)."""


def _durable_write(path: Path, text: str) -> None:
    """Write + fsync so a power loss can't leave a zero-length state file
    (audit MEDIUM: atomic-but-not-durable). The caller os.replace()s afterward."""
    with open(path, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


@dataclass
class StageStatus:
    state: str = "pending"
    progress: float = 0.0          # 0..1 within the stage
    message: str = ""              # last human-readable progress message
    started_at: float | None = None
    finished_at: float | None = None
    # Arbitrary stage bookkeeping that must survive reboots —
    # e.g. {"cloud_job_id": ..., "provider": ...} for the train stage.
    meta: dict = field(default_factory=dict)


@dataclass
class Job:
    id: str
    name: str
    created_at: float
    stages: dict[str, StageStatus]
    # {"type": "video"|"csv", "source": original path} — decides which stages apply.
    input: dict = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    def stage_dir(self, stage: str) -> Path:
        d = self.dir / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def current_stage(self) -> str | None:
        """First stage that still needs work, or None if the job is complete."""
        for s in STAGE_ORDER:
            if self.stages[s].state not in ("done", "skipped"):
                return s
        return None

    def log(self, msg: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.dir / "log.txt", "a") as f:
            f.write(f"[{stamp}] {msg}\n")

    def save(self) -> None:
        payload = {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "input": self.input,
            "stages": {k: asdict(v) for k, v in self.stages.items()},
        }
        tmp = self.dir / "job.json.tmp"
        _durable_write(tmp, json.dumps(payload, indent=2))
        os.replace(tmp, self.dir / "job.json")


def new_job(name: str, input: dict | None = None) -> Job:
    job = Job(
        id=time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
        name=name,
        created_at=time.time(),
        stages={s: StageStatus() for s in STAGE_ORDER},
        input=input or {},
    )
    job.dir.mkdir(parents=True, exist_ok=True)
    job.save()
    job.log(f"job created: {name}")
    return job


def load_job(job_id: str) -> Job:
    path = JOBS_DIR / job_id / "job.json"
    try:
        payload = json.loads(path.read_text())
        return Job(
            id=payload["id"],
            name=payload["name"],
            created_at=payload["created_at"],
            stages={k: StageStatus(**v) for k, v in payload["stages"].items()},
            input=payload.get("input", {}),
        )
    except FileNotFoundError:
        raise
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise CorruptJobError(f"{job_id}: {e}") from e


def list_jobs() -> list[Job]:
    """All jobs, newest first. A single corrupt/truncated job.json is skipped
    (logged to stderr) instead of aborting the whole list — otherwise one bad
    file bricks app startup and /api/jobs (audit HIGH)."""
    jobs = []
    for d in sorted(JOBS_DIR.iterdir(), reverse=True):
        if not (d / "job.json").exists():
            continue
        try:
            jobs.append(load_job(d.name))
        except CorruptJobError as e:
            print(f"store: skipping corrupt job {e}", file=sys.stderr)
    return jobs
