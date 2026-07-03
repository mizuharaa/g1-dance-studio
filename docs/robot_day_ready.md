# Robot-day dress rehearsal — summary (for main to integrate)

Software-side rehearsal of the full deploy chain. Robot never touched; all dry-run.

## What the rehearsal PROVED
- **Gate refuses correctly:** `gen_config.py` aborts for the real Thriller policy — no
  passing sim_exam/v1 verdict exists (Thriller is 98.4% held-out, below the 99% bar). Good.
- **Interlocks all fire:** `01/02/10` refuse without `CONFIRMED_BY_HUMAN`; gantry refuses
  without `--stage`, `--estop-confirmed`, `--gantry-confirmed`, and rejects an injection
  dance name; dry-runs print the exact PC2 commands without executing.

## What it EXPOSED and FIXED (real bugs that would have broken robot day)
- **`02_push_bundle.sh` would crash on every real bundle** — it asserted
  `man["exam"]["verdict"]=="pass"`, but `gen_config.py` writes `exam.authorized` (no
  `verdict` key) → KeyError. Also compared **truncated 16-char** shas against the manifest's
  **full 64-hex** values → always-mismatch. Fixed: check `exam.authorized`, verify every
  hash-pinned file with full sha256, and refuse rehearsal bundles. Verified passing on a
  simulated authorized bundle (previously an immediate crash).
- **`10_gantry_test.sh` still logged "robot falls back to damping"** — the exact unverified
  claim the safety review removed from `kill_now.sh`. Corrected to point at the remote e-stop
  and step 3a.
- Added `gen_config.py --rehearsal`: assembles a bundle without the exam gate for packaging
  validation, stamps it `REHEARSAL_ONLY`; `02_push_bundle.sh` refuses to push it.

## New materials
- `docs/ROBOT_DAY_RUNBOOK.md` — full ordered procedure + FMEA table + abort ladder.
  Adds **Step 3a (kill/command-loss test, feet off ground)** as the day's most important
  measurement, and states the no-torque-cut-e-stop reality plainly.
- `docs/ROBOT_DAY_CHECKLIST.md` — one-page printable card.
- `scripts/preflight_robot_day` — laptop GO/NO-GO validator (policy + show-ready verdict +
  scripts + shellcheck + bundle-assembles + kill-path + LAN). Dry-run/read-only.

## BLOCKER before a real robot day (honest)
- **Thriller is not show-ready** (98.4% < 99%). Preflight correctly returns NO-GO. Robot day
  waits on Thriller attempt 2 clearing the held-out gate.
- **Step 3a is unproven:** whether command-loss → damping on this robot is unknown until
  measured on the gantry. Until then the remote e-stop is the only trusted stop, and ground
  use is not authorized. An on-Jetson command-loss deadman is the recommended follow-up.
- **base_lin_vel sim2real gap:** the deployed controller needs a state estimator for it.

Tests: 167 passed / 11 skipped. shellcheck clean. Files: `deploy/gen_config.py`,
`deploy/02_push_bundle.sh`, `deploy/10_gantry_test.sh`, `scripts/preflight_robot_day`,
`docs/ROBOT_DAY_RUNBOOK.md`, `docs/ROBOT_DAY_CHECKLIST.md`.
