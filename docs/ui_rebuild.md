# UI rebuild — dark "Creative Studio" (Direction B)

Replaced the amateur debug-panel `ui/static/*` with a professional frontend built to
the user-chosen mockup (design/mockup_b). Plain HTML/CSS/JS, no build step, served by
ui/server.py StaticFiles. No server endpoints changed (additive-safe merge).

## Files
- `ui/static/index.html` — app shell: sidebar nav (Dashboard/Library/Create/Show
  Mode/System/Settings), topbar, 6 screen containers, toast + modal roots, hidden file input.
- `ui/static/style.css` — design system lifted verbatim from mockup_b (dark near-black
  canvas, cyan→violet accent, cards/badges/stepper/gauge/sidebar) + production state
  styles (banners, toasts, modals, empty/loading, stage-log, job rows).
- `ui/static/app.js` — the engine: engine-down-aware fetch + banner, toasts, modals,
  real-data renderers per screen, actions, 20s/30s polling.

## What's wired to real endpoints
- Dashboard: /api/system (GPU, jobs, cost), /api/dances counts, next-steps.
- Library: /api/dances cards + status badges + filter/search; dance detail modal with
  /previews video (onerror fallback) + attach-policy (POST /api/dances/{id}/policy).
- Create/Studio: /api/jobs list, /api/jobs/upload (drag-drop + picker), real stage
  stepper from /api/jobs/{id}, log tail, vet table, preview, /api/jobs/{id}/retry.
- Show: /api/shows history + show-ready dances; create show + checklist wizard
  (/api/shows/{id}/steps/{key}) + typed-DEPLOY record-only gate (/api/shows/{id}/deploy).
- System: /api/system GPU/jobs/cost + W&B links + stale/unreachable states.
- Settings: /api/cloud (+ config/test), /api/bodymodels, /api/library/export.

## Bug fixed
The old nav only toggled a CSS class without swapping views (mode buttons looked dead).
The new `go()` toggles `.screen.active` AND re-renders — verified: clicking each of the
6 nav items switches the active screen (design/rebuild_shots/*.png).

## Audit behaviors carried over
Engine-down banner on network failure; buttons disabled during in-flight requests
(upload, retry, cloud save/test, checklist steps); deploy dialog scoped per-show (no
target leak); preview `<video>` onerror → graceful "unavailable"; cloud connection
fields are inputs the user edits explicitly (no auto-refresh wipe — Settings only
re-renders on explicit Save/Test).

## Verified
Server imports OK; assets serve 200; all 6 screens render real data + switch via nav
(pilot screenshots in design/rebuild_shots/); graceful empty/unreachable states; full
test suite 123 passed / 11 skipped. Merge-ready — touches only ui/static/ + docs +
screenshots, no conflict with pipeline/sim_exam.py or PROJECT_STATE work.
