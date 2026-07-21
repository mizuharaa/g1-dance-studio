# B6 — Lane B's share of the red baseline: 3 show tests + 1 video-probe test

**Finding:** TEST_RESULTS.md. Lane B owns:
- `tests/test_free_show.py::test_free_run_builds_free_env_and_standtail_args`
- `tests/test_free_show.py::test_non_free_run_keeps_proven_default`
- `tests/test_show_run.py::test_exit_stand_ignored_in_live`
- `tests/test_video_probe.py::test_too_long_rejected`

**Per the audit:** the three show tests assert SUPERSEDED behavior; the video
test expects `MAX_SECONDS+1` to fail while the implementation deliberately
allows a 5 s keyframe-copy tolerance.

## Spec
1. **Order matters: do this AFTER B3/B4** — those tasks intentionally change
   show behavior again (free-in-live becomes 409; default fallback deleted).
   Rewrite the three show tests to assert the NEW fail-closed contract (B3's
   spec is the source of truth), not the pre-audit behavior and not the
   superseded one.
2. `test_video_probe.py`: decide from `pipeline/video_probe.py`'s docstring
   whether the 5 s tolerance is intended (audit says yes, deliberate). If so,
   test the REAL boundary: `MAX_SECONDS + tolerance` passes,
   `MAX_SECONDS + tolerance + 1` rejects. If the docstring does NOT document
   the tolerance, add the documentation in the same commit.
3. No assertion weakened where current behavior is genuinely wrong — if you
   find that, log it on the coordination board instead of green-washing.

## Acceptance
```bash
python -m pytest tests/test_free_show.py tests/test_show_run.py tests/test_video_probe.py -q
python -m pytest -q   # FULL suite green (with Lane A's A5 landed): the wave's exit gate
```
Update the README status board; the full-suite green line is the whole wave's
definition of done.
