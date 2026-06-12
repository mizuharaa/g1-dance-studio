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


class Stage(Protocol):
    name: str

    def run(self, job: Job, report: Reporter) -> None:
        """Execute the stage to completion. Raise on failure."""
        ...
