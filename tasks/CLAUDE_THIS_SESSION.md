# CLAUDE (this session) — what I do myself vs. what I delegate

**Round:** 2026-07-13. This is the "what do you do as well" file: everything I can finish on the
**laptop with no GPU box, no robot, and no sudo/creds** I do myself; the rest becomes Agents A/B/C.

## ✅ DONE by me this session (on branches / main)
- **E-STOP / E-kill switch.** `show_runner.emergency_kill()` + `POST /api/safety/estop`
  (SIGTERM the tracked show AND stray `deploy_runtime`/`show_run.sh`; never SIGKILL, so it always
  damps). Always-available E-STOP in the top bar + a big one on a new **Safety** screen. Honest
  scope messaging (the remote B-damp stays the primary hard stop). Backend tests pass. *(committed)*
- **Safety UI + robot-state viz.** New Safety screen: coarse robot-state figure by phase (red thrash
  on fall), pre-arm reminders headlined by the **feet-flat-on-ground** fix for the thrash-when-
  suspended failure, honest CAN/CAN'T scope, live runtime log. Ground-contact warning also in the
  run gate. *(committed)*
- **Dark mode + contrast.** Toggle (persisted, system-default), `.dark` remaps for the hardcoded
  utilities, light-mode text darkened. *(committed)*
- **Video PREVIEW fix (in-app).** Root cause: desktop app is PySide6 QtWebEngine (no H.264 codec).
  Added a pywebview `open_external` bridge + auto-fallback "Open in browser". *(committed)*
- **Show-DISPLAY fix (Lane 1).** `tools/show_display.py`: demote VLC to last resort (`mpv > ffplay
  > vlc`), `SHOW_PLAYER` override, defensive VLC flags for the "recursion" bug, loud fallback warning.
  Branch **`fix/show-display-mpv`**. *(One thing left that needs YOU: `sudo apt install -y mpv`.)*
- **Twitch source re-prep (Lane 2).** Re-prepped the raw Thriller through the now-wired de-glitch
  filter: **jerk peak 101,701 → 4,806 rad/s³ (÷21), spike frames 67 → 4**, fidelity kept (RMS Δ
  0.033 rad). Clean motion + metrics committed. Branch **`fix/twitch-source-reprep`**. This is the
  INPUT for Agent A's retrain — it does not fix the robot by itself.
- **Docs:** `docs/RETRAIN_RUNBOOK.md` (step-by-step retrain) + this task board.

## ❌ What I CANNOT do here → delegated
- **Create/run a GPU box** — GreenNode is console-only (no API) and `.secrets/` isn't on this
  checkout. → **Agent A** (retrain).
- **Touch the real robot** — needs a human + damping remote (CLAUDE.md hard rule). → **Agent C**.
- **`sudo apt install mpv`** — no sudo in my shell. → you, one command (closes Lane 1).
- **Trust the airborne guard** — it needs gantry validation (no foot-force sensor to lean on). →
  **Agent B** designs/validates; I gave the full spec.
- **Push to GitHub** — neither `origin` (no PAT) nor `handoff` (no SSH key) is reachable from my
  shell. → push the branches yourself (commands in the chat / README).

## The problem status I verified (so nothing is claimed "fixed" that isn't)
| Problem | Resolved? | Where it's handled |
|---|---|---|
| Twitchy / limb-snapping | Filter DONE + proven (÷21 jerk), but deployed policy predates it | Lane 2 (done) → **Agent A** retrain |
| Latency (~45 s buckle / drift) | **No** — lat80 failed; v5 curriculum staged, not trained | **Agent A** → **Agent C** |
| Show video = colourful static | Code DONE; needs `apt install mpv` | Lane 1 (me) + you |
| Stand-exit handback | **Unvalidated** on hardware | **Agent C** |
| Thrash-when-suspended | Root-caused; UI reminder + E-STOP shipped; code guard designed | me (UI) + **Agent B** |
| E-kill switch | **DONE** | me |
