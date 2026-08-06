# In-app observability ("System" panel) — build summary

Built in worktree `agent-a00839ca27793b8bd`. Merge-ready.

## What it does
A third app mode (**Studio / Show / System**) that answers the questions the user kept
asking Claude directly — "is the GPU running / how's training / how much has it cost" —
by reading the GreenNode box over the existing SSH transport, read-only.

Shows, refreshing every ~20 s:
1. **Cloud GPU box**: reachable dot (green busy / green idle / red stale), GPU load %,
   VRAM, power, temp — pulled live via `nvidia-smi`. Explicit UI note that the GreenNode
   console shows instance *state*, not live load, so this panel differs from the website.
2. **Training jobs**: per active job, iteration/max progress bar, latest mean reward and
   mean episode length, a W&B link when present. Parsed from the box's job logs + status
   JSON. Install/finished non-training jobs are filtered out.
3. **Accrued GPU cost**: box-hours × rate (16,080,632 VND/mo ÷ 730 × 0.75 internal
   discount × 1.10 VAT ≈ 18,170 VND/h) with a bar against the 1.5M VND cap, in VND + USD.
   Reminds the user billing runs until *deletion*.

## Files
- **`pipeline/monitor.py`** (new): pure parsers `parse_gpu` / `parse_job_log` /
  `compute_cost` / `parse_gather`, plus `snapshot()` which does one SSH round-trip
  (combined `_GATHER_CMD`), caches the last good result, and never raises — an
  unreachable box degrades to a stale snapshot, never hangs the UI.
- **`ui/server.py`**: `GET /api/system` (returns the cached snapshot) + a background
  refresher thread started in the startup hook. Import line extended to include `monitor`.
- **`ui/static/index.html`**: System nav button + `#system-main` section.
- **`ui/static/app.js`**: `setMode` generalized to 3 modes; `refreshSystem`/`renderSystem`;
  20 s poll while the System tab is visible.
- **`ui/static/style.css`**: stat grid, job rows, cap bar.
- **`tests/test_monitor.py`** (new, 11 tests): GPU parse (busy/idle/garbage), log parse
  (latest-value semantics, empty, negative reward), cost math (basic, over-cap, bad date),
  full `parse_gather` split. All green; full suite 101 passed / 8 skipped.

## Verification
- Endpoint smoke via TestClient: `/api/system` → 200, correct shape, graceful
  "not configured" when no cloud.json (as in a bare worktree).
- **Real box**: ran `_GATHER_CMD` against the live GreenNode 4090 and fed output through
  `parse_gather` — correctly read GPU 99%/191W/71°C and both training jobs
  (benchmark iter 1726 reward 21.05, thriller iter 1334 reward 25.46).

## Merge / conflict notes
- Touches `ui/server.py` (import line + one startup-hook line + a new endpoint block) and
  `ui/static/*`. No other in-flight worktree was told to touch these, but if the app-audit
  remediation edits `ui/server.py`, expect a trivial import/line merge — both are additive.
- Does NOT touch `pipeline/sim_exam.py`, `pipeline/shows.py`, or `PROJECT_STATE.md`.
- Cost params (`created_at`, `rate_vnd_per_hour`, `cap_vnd`, `usd_per_vnd`) read from a
  `"billing"` block in `.secrets/cloud.json` with the documented defaults — no magic numbers
  hardcoded in the UI. Update `created_at` if the box is recreated.

## Known limitation
The W&B URL only appears if it's in the log tail (`grep | tail -10`); a run that printed
its URL only at startup won't show a link until re-logged. Cosmetic; the numbers are live.
