# AGENT C — Frontend Dashboard Revamp (shadcn + Playwright MCP)

**Status: COMPLETE (2026-07-10).** React source and production build are in
`ui/frontend/`; FastAPI/pywebview serves it. Playwright evidence is in
`docs/ui_revamp/`; backend-only data gaps are in `tasks/API_GAPS.md`.

**Owner: USER'S MANUAL AGENT** — run this with the shadcn MCP server and the Playwright MCP
server connected (and Mobbin MCP for design reference if available). Do not run this lane from
the orchestrator; it depends on those MCP servers.

## Goal

Replace the vanilla-JS UI (`ui/static/app.js`, 811 lines + `style.css`) with a modern,
responsive **operator dashboard**: React + Vite + Tailwind + **shadcn/ui**, dark
mission-control aesthetic inspired by **Maestro agentic_os** (clean panels, strong hierarchy,
dense-but-legible telemetry) but re-themed **blue** (e.g. slate/zinc base, blue-500/600 accent,
blue status glows). This is the operator console for a paid robot show — clarity of state
beats decoration.

## Hard constraints

1. **Backend API is the contract — do not change endpoint semantics.** Read `ui/server.py`
   (953 lines) for the full surface: jobs (`/api/jobs`, `/api/jobs/upload`), dances
   (`/api/dances`, promote/policy/outcome), shows (`/api/shows`, runs, STOP), setlists,
   `/api/system` (GPU/cost/training monitor). Allowed server change: serving the built
   frontend (static dir / SPA fallback) only.
2. **Keep the pywebview desktop wrapper working** (`ui/desktop.py` loads `http://127.0.0.1:8735`).
   Build output must be served by FastAPI — no separate Node server in production; `npm run dev`
   proxying to :8735 is fine during development.
3. **Never weaken safety UX.** The RUN SHOW typed-confirmation phrase, the STOP button
   (`POST /api/shows/runs/current/stop`), and outcome capture (Clean/Aborted/Incident) must
   remain — make them MORE prominent. STOP: always visible during a run, oversized, red.
4. New code lives in `ui/frontend/`; keep the old `ui/static/` until parity, then delete it
   in the final commit.

## Required screens/features

- **Dashboard**: live dance-run state machine front and center
  (idle → preflight → deploying → dancing → stand-handback → done | incident), progress along
  the dance timeline, current show + policy + venue, system panel (GPU %, cloud cost, training
  status from `/api/system`).
- **Audit log**: filterable timeline per dance and global — show outcomes, falls/incidents,
  demotions, sim-exam verdicts (signed pass/fail + survival %), promotions, deploys. If a
  dance failed, the operator must see *when, at which second, and what the verdict said*.
- **Stats**: per-dance held-out survival %, tracking error (mpkpe), training cost/iterations,
  latency-gate results (40/60/80 ms survival from gap_check artifacts), run history.
- **Pipeline studio**: drag-drop video upload (keep `/api/jobs/upload` flow), 5-stage progress
  (extract → retarget → train → verify → export) with per-stage logs and failure reasons.
- **Shows/setlists**: builder + perform mode; show-ready blockers clearly explained.
- **Responsive**: usable on a laptop half-screen and on a tablet at the venue; test at 1440,
  1024, 768 wide minimum.

## Method

1. Explore the running app first: `python ui/server.py --host 127.0.0.1 --port 8735`, then
   drive it with **Playwright MCP** to inventory every existing screen/action before writing code.
2. Pull reference layouts (Maestro agentic_os style) via Mobbin MCP or screenshots; pick the
   blue theme tokens once, put them in Tailwind config/CSS vars — light not required, dark-first.
3. Build with **shadcn MCP** components (Card, Table, Badge, Dialog, Tabs, Chart via Recharts).
4. **Verify with Playwright MCP against the real server** at the three breakpoints: upload flow,
   run-state rendering, STOP visibility, audit-log filters. Screenshot evidence into
   `docs/ui_revamp/`.
5. Commit per feature; suite `pytest` must stay green (server tests); update `PROJECT_STATE.md`.

## Out of scope
`pipeline/` (except nothing), `cloud/`, `deploy/`, robot anything. If an endpoint is missing
data you need (e.g. richer audit events), write the gap to `tasks/API_GAPS.md` for the
orchestrator instead of patching the backend yourself.
