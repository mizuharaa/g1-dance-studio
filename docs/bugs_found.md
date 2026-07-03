# Bugs found by the test-hardening pass (2026-07-03)

Filed by the tests worktree; main triages/fixes (I only wrote tests + this file).

## BUG-1 (HIGH, security): upload filename path traversal — arbitrary file write

`ui/server.py::create_job_upload` builds the save path by interpolating the
client-supplied filename directly:

```python
tmp = DATA_DIR / "videos" / f"upload-{int(time.time())}-{video.filename}"
```

A filename containing `../` escapes `data/videos/`. Verified empirically: uploading
with filename `../../pwned.sh` created a file *outside* `data/videos/` (at
`data/pwned.sh` in the worktree). `_create_job` then also copies by
`src.suffix`/`src.stem`, so a crafted name controls the extension too.

**Impact:** any client of the local API can write/overwrite arbitrary files the
user can write, with attacker-chosen path and extension. For a product that will
ship to operators (and may later expose the API beyond localhost), this is a
real remote-ish file-write.

**Repro:** `tests/test_server_api.py` — add a slash-filename upload (I left this
OUT of the committed suite as a passing test on purpose, since it currently
"passes" by exploiting the bug; see the empirical check in the task notes).

**Suggested fix:** sanitise to the basename and strip separators before use:
```python
from pathlib import PurePosixPath
safe = PurePosixPath(video.filename or "upload").name  # drops any dir components
safe = safe.replace("/", "_").replace("\\", "_") or "upload"
tmp = DATA_DIR / "videos" / f"upload-{int(time.time())}-{safe}"
```
Apply the same basename treatment anywhere a client string becomes a path
(`create_job` input_path is developer-supplied so lower risk, but the upload
path is client-facing).

## BUG-2 (LOW, hygiene): `@app.on_event("startup")` deprecated

`ui/server.py` uses the deprecated `on_event` hook (FastAPI warns). Works today;
migrate to a `lifespan` handler before a FastAPI upgrade removes it.

## Non-bugs verified (documented so nobody "fixes" them)

- Velocity spikes and foot-skate are **advisory** — the gate PASSES motions that
  breach the motor velocity limit (by design: the RL reward smooths infeasible
  references; Unitree's own LAFAN1 retargets breach it on ~30% of frames).
- The vet gate's excursion check is measured **relative to the first frame**, not
  the world origin — a dance far from origin is judged by its own travel. Correct.
- Notebook host-key churn: `pipeline/cloud.py` deliberately uses
  `StrictHostKeyChecking=no` + `UserKnownHostsFile=/dev/null` because GreenNode
  notebooks regenerate host keys on restart. Intentional, documented in-code.
