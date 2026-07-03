# UI responsive / fullscreen fix (2026-07-04)

**Root cause:** `.content` had `max-width:1180px` and was left-aligned (no auto-margin),
so any window wider than ~1420px (1180 content + 236 sidebar) left dead space on the
right. Compounding it, the card grids (`.g-2/.g-3/.g-4`, `.dance-grid`) used fixed
column counts (`repeat(N,1fr)`) that never added columns on wide screens.

**Fix (ui/static/style.css only — no markup/JS/data changes):**
- `.content` -> `width:100%; max-width:1680px; margin:0 auto` with fluid
  `padding:26px clamp(24px,3vw,44px) 60px`. Fills up to 1680px then centers with
  balanced margins (readable cap; intentional on ultrawide).
- Grids -> responsive auto-fit/auto-fill minmax so cards reflow to the width:
  stat cards `.g-4` 4->6 cols on wide; `.g-2` two-up->one-up under 980px; dance
  `.dance-grid` scales column count with width.
- Breakpoints: <=980px drops dashboard two-up rows to one column; <=760px collapses
  the sidebar to an icon rail; <=520px single-column + hides search.

**Verified** (server on :8791, pilot browser): at 1280px content fills 1044/1044px
(no dead space); at simulated ~1940px content = 1680px centered, stat grid = 6 cols;
nav switches all screens; no horizontal overflow. Suite green (150 passed, 11 skipped).
Screenshots: design/responsive_shots/
