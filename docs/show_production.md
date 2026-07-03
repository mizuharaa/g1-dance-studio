# Show production workflow

Turns the app from "train one dance" into "produce and run a real paid show":
music sync, set-lists (ordered multi-dance shows), a rehearsal mode distinct from
live, and a show timeline. Pure software — nothing here contacts the robot; the
per-dance deploy gate stays record-only + typed-DEPLOY as before.

## 1. Music sync (per dance)

A dance can carry a music track. `pipeline/audio.py`:
- **Ingest**: an on-disk audio file (`source_path`), extraction from a video
  (`extract_from_video`, fails clearly if the video is silent — the Thriller source
  IS silent), or a generated placeholder click track (to verify sync before the real
  licensed song exists).
- **Align**: the music is delayed by `pad_in + blend_in` (**1.5 s** for the default
  prep) so it stays on the beat despite the standing intro `prep_motion` prepends.
  Alignment is computed in seconds and is preserved end-to-end (30→50 fps→50 Hz).
- **Mux**: if the dance has a silent preview, the aligned music is muxed onto a copy
  so the preview *plays with sound*.

Stored on the `Dance` record as `audio = {track, source, align{...}, muxed_preview}`
(presentation only — audio never affects show-ready status).

API: `POST /api/dances/{id}/audio` (attach), `DELETE …/audio` (clear),
`GET …/audio-file` (serve the track for the player). UI: attach/replace/remove in the
dance detail, a ♪ badge in the library + on show-ready cards, and the muxed preview is
used automatically when present.

## 2. Set-lists (ordered multi-dance shows)

`pipeline/setlist.py` — a `SetList` is an ordered list of items
`{dance_id, gap_after_s, note}`, persisted at `data/setlists/<id>/setlist.json`.
`resolve(sl, dance_lookup)` joins it against the live library into a runnable view:
per-item status/duration/has-audio, **total runtime** (durations + gaps), and
**blockers** — a set-list is show-ready only if *every* item is a show-ready dance.

API: `GET/POST /api/setlists`, `GET/POST/DELETE /api/setlists/{id}` (POST sets
name/notes/items — reorder/add/remove all go through `items`). UI: **Show → Set Lists**
tab — create, drag-order (↑/↓), edit gaps, remove, live runtime + blocker readout, and
a **Run set / Rehearse set** button (disabled until show-ready).

The **runner** walks the set in order: each number opens its own pre-show checklist and
the same typed-DEPLOY record-only gate, then an outcome, then advances. Nothing auto-fires.

## 3. Rehearsal mode

A show carries `mode: "live" | "rehearsal"` (default live). A global **Live/Rehearsal
toggle** in Show mode makes the state visually unmistakable (amber banner + tagged
modal + tagged history). The only behavioural difference: **a rehearsal incident/abort
never demotes the dance** — a dry run can't knock a show-ready dance out of the library
(`shows.record_outcome` guards the demotion on `mode == "live"`). Rehearsals log to
history separately, badged as such.

## 4. Show timeline

**Show → Timeline** tab: the selected set-list drawn as proportional blocks (dance =
gradient, gap = hatched) with durations, ♪ markers, total runtime, and a blocker note —
the whole performance at a glance.

## Tests

`tests/test_show_production.py` (13): alignment math (default/windowed/invalid),
set-list create/validation/resolve/runtime/blockers/show-ready, rehearsal-vs-live
demotion, audio attach (placeholder + no-duration guard). Full suite green (180 passed).

## Merge / conflict notes

- Backend: **added** `pipeline/setlist.py`; **extended** `pipeline/shows.py` (Dance.audio,
  Show.mode/setlist_id, new_show mode arg, set_audio, rehearsal-guarded record_outcome),
  `pipeline/audio.py` (attach_audio_for_dance), `ui/server.py` (audio + set-list endpoints,
  create_show mode). No changes to sim_exam/exam_verdict/mjlab_verify/vet/find_window/
  venue/monitor/deploy/PROJECT_STATE.
- Frontend: `ui/static/app.js` + `style.css` — the Show screen gained sub-tabs; the
  dashboard/library/studio/system/settings renderers are unchanged, so this merges
  cleanly against the responsive/monitor UI work (both touch disjoint regions).
- Verification: all endpoints curl-verified live (setlist CRUD, audio placeholder attach,
  rehearsal show); `node --check` clean on app.js; the pilot browser was contended by the
  parent so no fresh screenshots this run — the parent can screenshot Show → Set Lists /
  Timeline after merge.
