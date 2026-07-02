"""Stage abstraction.

Each pipeline stage is a class with a `run(job, report)` method. Stages must be:
  * resumable — if interrupted (reboot), re-running must either redo cleanly or
    pick up via state stashed in job.stages[name].meta (e.g. a cloud job id);
  * side-effect-scoped — all outputs go in job.stage_dir(name).
"""
from __future__ import annotations

from typing import Callable, Protocol

from ..store import Job

# report(progress 0..1, message) — stages call this to stream progress to the UI.
Reporter = Callable[[float, str], None]


class SkipStage(Exception):
    """Raised by a stage that does not apply to this job (e.g. extract on a
    CSV motion input). The runner marks it 'skipped' and moves on."""


class StageBlocked(Exception):
    """Raised by a stage that cannot run yet for an external reason (e.g. the
    cloud GPU is not provisioned). The runner marks the stage 'blocked' and
    stops the job without failing it; a later re-queue retries."""


class Stage(Protocol):
    name: str

    def run(self, job: Job, report: Reporter) -> None:
        """Execute the stage to completion. Raise on failure."""
        ...
