# A5 — Lane A's share of the red baseline: 3 cloud-fixture tests

**Finding:** REPORT §5 "Restore a green test baseline" + TEST_RESULTS.md. The
7 red tests train humans to ignore CI. Lane A owns the 3 cloud ones:

- `tests/test_cloud_audio.py::test_extract_captures_source_audio_when_present`
- `tests/test_cloud_audio.py::test_extract_audio_failure_is_non_fatal`
- `tests/test_cloud_stages.py::test_extract_video_walkthrough`

**Cause (per audit):** the extract stage gained a remote `mkdir` via `_box_run`;
the fake-box fixtures patch `_push` but not `_box_run`, so the walkthrough stops
at the new call (KeyError on the pushed-video key downstream).

**Owned files:** `tests/test_cloud_audio.py`, `tests/test_cloud_stages.py`
(fixtures only — if the PRODUCT code is actually wrong, STOP and log in the
coordination board instead of "fixing" the test to match a bug).

## Spec
1. Read `pipeline/stages/cloud_motion.py`'s current box interaction to confirm
   the `_box_run`/mkdir call is intended behavior (it is, per audit — fixture
   drift). Extend the fake box in both test files to record `_box_run` commands
   and return success, so the walkthrough proceeds to the push assertions.
2. Assert the NEW behavior too: the mkdir command hits the expected remote dir
   (cheap regression lock on the box layout).
3. Do NOT weaken any assertion that still reflects intended behavior.

## Acceptance
```bash
python -m pytest tests/test_cloud_audio.py tests/test_cloud_stages.py -q  # all green
python -m pytest -q  # full suite: only Lane B's 4 remaining reds (until B6 lands)
```
