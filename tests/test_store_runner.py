"""Job persistence round-trips and runner state transitions."""
import json

import pytest

from pipeline.config import STAGE_ORDER
from pipeline.runner import Runner
from pipeline.stages.base import SkipStage, StageBlocked


def test_new_job_persists_and_loads_back(jobs_env):
    store, jobs_dir = jobs_env
    job = store.new_job("demo", input={"type": "csv", "source": "/tmp/x.csv"})
    assert (jobs_dir / job.id / "job.json").exists()

    loaded = store.load_job(job.id)
    assert loaded.name == "demo"
    assert loaded.input == {"type": "csv", "source": "/tmp/x.csv"}
    assert set(loaded.stages) == set(STAGE_ORDER)
    assert all(s.state == "pending" for s in loaded.stages.values())
    assert loaded.current_stage() == STAGE_ORDER[0]


def test_save_is_atomic_no_tmp_left_behind(jobs_env):
    store, jobs_dir = jobs_env
    job = store.new_job("atomic")
    job.stages["extract"].state = "done"
    job.save()
    assert not (job.dir / "job.json.tmp").exists()
    assert store.load_job(job.id).stages["extract"].state == "done"


def test_current_stage_walks_past_done_and_skipped(jobs_env):
    store, _ = jobs_env
    job = store.new_job("walk")
    job.stages["extract"].state = "skipped"
    job.stages["retarget"].state = "done"
    assert job.current_stage() == "train"
    for s in STAGE_ORDER:
        job.stages[s].state = "done"
    assert job.current_stage() is None


def test_stage_meta_survives_reload(jobs_env):
    store, _ = jobs_env
    job = store.new_job("meta")
    job.stages["train"].meta["cloud_job_id"] = "train-thriller-a1"
    job.save()
    assert (store.load_job(job.id).stages["train"].meta["cloud_job_id"]
            == "train-thriller-a1")


def test_list_jobs_newest_first(jobs_env):
    store, _ = jobs_env
    # ids start with a timestamp, so forge two with known ordering
    a = store.new_job("older")
    b = store.new_job("newer")
    b_dir_name = "99999999-999999-zzzzzz"
    (a.dir.parent / b_dir_name).mkdir()
    (a.dir.parent / b_dir_name / "job.json").write_text(
        (b.dir / "job.json").read_text().replace(b.id, b_dir_name))
    names = [j.id for j in store.list_jobs()]
    assert names[0] == b_dir_name


# ---- runner ---------------------------------------------------------------------

class _Stage:
    def __init__(self, effect=None):
        self.effect = effect
        self.calls = 0

    def run(self, job, report):
        self.calls += 1
        if self.effect:
            raise self.effect
        report(0.5, "halfway")


def _stages(**effects):
    return {name: _Stage(effects.get(name)) for name in STAGE_ORDER}


def test_runner_requires_all_stages():
    with pytest.raises(ValueError, match="extract"):
        Runner({})


def test_full_run_marks_everything_done(jobs_env):
    store, _ = jobs_env
    job = store.new_job("ok")
    Runner(_stages()).run_job(job)
    reloaded = store.load_job(job.id)
    assert [reloaded.stages[s].state for s in STAGE_ORDER] == ["done"] * 5
    assert all(reloaded.stages[s].progress == 1.0 for s in STAGE_ORDER)
    assert reloaded.current_stage() is None


def test_skip_marks_skipped_and_continues(jobs_env):
    store, _ = jobs_env
    job = store.new_job("skip")
    Runner(_stages(extract=SkipStage("csv input"))).run_job(job)
    reloaded = store.load_job(job.id)
    assert reloaded.stages["extract"].state == "skipped"
    assert reloaded.stages["extract"].message == "csv input"
    assert reloaded.stages["export"].state == "done"


def test_blocked_stops_without_failing(jobs_env):
    store, _ = jobs_env
    job = store.new_job("blocked")
    stages = _stages(train=StageBlocked("cloud not provisioned"))
    Runner(stages).run_job(job)
    reloaded = store.load_job(job.id)
    assert reloaded.stages["retarget"].state == "done"
    st = reloaded.stages["train"]
    assert st.state == "blocked"
    assert "cloud" in st.message
    assert st.started_at is None          # blocked = never really ran
    assert reloaded.stages["verify"].state == "pending"   # did not continue
    assert stages["verify"].calls == 0


def test_failure_records_type_and_stops(jobs_env):
    store, _ = jobs_env
    job = store.new_job("boom")
    stages = _stages(retarget=ValueError("bad csv"))
    Runner(stages).run_job(job)
    reloaded = store.load_job(job.id)
    st = reloaded.stages["retarget"]
    assert st.state == "failed"
    assert st.message == "ValueError: bad csv"
    assert st.finished_at is not None
    assert reloaded.stages["train"].state == "pending"
    log = (reloaded.dir / "log.txt").read_text()
    assert "FAILED" in log and "Traceback" in log


def test_until_stops_after_named_stage(jobs_env):
    store, _ = jobs_env
    job = store.new_job("until")
    stages = _stages()
    Runner(stages).run_job(job, until="retarget")
    reloaded = store.load_job(job.id)
    assert reloaded.stages["retarget"].state == "done"
    assert reloaded.stages["train"].state == "pending"
    assert stages["train"].calls == 0


def test_rerun_resumes_from_first_incomplete(jobs_env):
    store, _ = jobs_env
    job = store.new_job("resume")
    stages = _stages(train=StageBlocked("later"))
    runner = Runner(stages)
    runner.run_job(job)
    assert stages["extract"].calls == 1

    # unblock and re-run: earlier stages must not execute again
    stages["train"].effect = None
    runner.run_job(store.load_job(job.id))
    assert stages["extract"].calls == 1
    assert stages["train"].calls == 2      # blocked attempt + real run
    assert store.load_job(job.id).current_stage() is None


def test_progress_report_clamped(jobs_env):
    store, _ = jobs_env
    job = store.new_job("clamp")

    class Wild:
        def run(self, job_, report):
            report(7.3, "over")
            assert job_.stages["extract"].progress == 1.0
            report(-2, "under")
            assert job_.stages["extract"].progress == 0.0

    stages = _stages()
    stages["extract"] = Wild()
    Runner(stages).run_job(job, until="extract")
