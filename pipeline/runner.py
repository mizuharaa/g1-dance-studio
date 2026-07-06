"""Executes a job's stages in order, persisting state for reboot recovery."""
from __future__ import annotations

import time
import traceback

from .config import STAGE_ORDER
from .store import Job
from .stages.base import SkipStage, Stage, StageBlocked


class Runner:
    def __init__(self, stages: dict[str, Stage]):
        missing = [s for s in STAGE_ORDER if s not in stages]
        if missing:
            raise ValueError(f"no implementation registered for stages: {missing}")
        self.stages = stages

    def run_job(self, job: Job, until: str | None = None) -> None:
        """Run all incomplete stages in order; `until` stops after that stage
        (the export→robot boundary is always an explicit human action)."""
        while (name := job.current_stage()) is not None:
            self._run_stage(job, name)
            if job.stages[name].state in ("failed", "blocked") or name == until:
                return

    def _run_stage(self, job: Job, name: str) -> None:
        st = job.stages[name]
        st.state = "running"
        st.started_at = st.started_at or time.time()
        job.save()
        job.log(f"stage {name}: started")

        def report(progress: float, message: str) -> None:
            st.progress = max(0.0, min(1.0, progress))
            st.message = message
            job.save()

        try:
            self.stages[name].run(job, report)
            st.state = "done"
            st.progress = 1.0
            st.finished_at = time.time()
            job.log(f"stage {name}: done")
        except SkipStage as e:
            st.state = "skipped"
            st.message = str(e)
            st.finished_at = time.time()
            job.log(f"stage {name}: skipped — {e}")
        except StageBlocked as e:
            st.state = "blocked"
            st.message = str(e)
            st.started_at = None      # it never really ran
            # A stage waiting on a cloud job can ask to be re-checked: the server
            # poll loop re-queues the job once meta["poll_after"] has passed.
            retry_s = getattr(e, "retry_after_s", None)
            if retry_s:
                st.meta["poll_after"] = time.time() + float(retry_s)
            job.log(f"stage {name}: blocked — {e}")
        except Exception as e:
            st.state = "failed"
            st.message = f"{type(e).__name__}: {e}"
            st.finished_at = time.time()
            job.log(f"stage {name}: FAILED\n{traceback.format_exc()}")
        finally:
            job.save()
