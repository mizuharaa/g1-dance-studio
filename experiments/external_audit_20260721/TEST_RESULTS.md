# CPU test result

Command, from repository root:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate g1dance
python -m pytest -q --tb=short
```

Result on baseline `d60ac3c3609a1bd3868ff206709e0c48ddb684d5`:
**604 passed, 7 failed (611 collected)**.

Failures:

```text
tests/test_cloud_audio.py::test_extract_captures_source_audio_when_present
tests/test_cloud_audio.py::test_extract_audio_failure_is_non_fatal
tests/test_cloud_stages.py::test_extract_video_walkthrough
tests/test_free_show.py::test_free_run_builds_free_env_and_standtail_args
tests/test_free_show.py::test_non_free_run_keeps_proven_default
tests/test_show_run.py::test_exit_stand_ignored_in_live
tests/test_video_probe.py::test_too_long_rejected
```

The first three tests stop at the newly added remote `mkdir` call because their
fake box patches `_push` but not `_box_run`; the next three assert superseded
show-video/stand-exit behavior; the final test expects `MAX_SECONDS + 1` to fail
while the implementation deliberately allows a five-second keyframe-copy
tolerance. These are baseline test-contract/fixture drift, not failures caused by
the audit artifacts. They still mean the repository has no green full-suite
baseline.
