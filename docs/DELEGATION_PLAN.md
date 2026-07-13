# DELEGATION PLAN — parallel lanes as branches (2026-07-13)

The remaining work splits into independent lanes. Each is a **branch** you can hand to a
separate agent/human, then resync to `main`. Lanes are ordered by what unblocks what.
`origin` = HashBrake/g1-dance, `handoff` = mizuharaa/unitree_g1_dance (this machine can't
push — no key loaded; push from a machine that has `.secrets/`).

Legend — **where it runs**: 💻 laptop · ☁️ GPU box · 🤖 robot(+human) · 📄 docs-only.

| Branch | Lane | Where | Status | Blocks |
|---|---|---|---|---|
| `fix/show-display-mpv` | Show-video player off VLC | 💻 | **CODE DONE** (needs `apt install mpv`) | — |
| `fix/twitch-source-reprep` | Clean (de-glitched) Thriller motion | 💻 | **DONE** (motion + metrics) | retrain input |
| `train/latency-curriculum-v5` | v5 retrain (latency + fidelity + clean motion) | ☁️ | READY, needs a box | hw validation |
| `feat/airborne-contact-guard` | Refuse/damp if the robot isn't on the ground | 💻 code, 🤖 validate | SPEC + starter | safer arming |
| `hw/validate-v5-and-exit-fix` | Hardware: latency, drift, ~45 s buckle, stand-exit | 🤖 | SPEC (needs robot) | show-ready |

---

## Lane 1 — `fix/show-display-mpv`  (💻 DONE in this branch)
**Problem:** the show display renders colourful static ("VLC: Too high level of recursion").
Content is correct; the *player* is broken.
**Done here:** `tools/show_display.py` now (a) demotes VLC to last-resort (`mpv > ffplay > vlc`),
(b) honours `SHOW_PLAYER=mpv|ffplay|vlc` to force a player, (c) adds defensive VLC flags
(`--no-one-instance --no-osd --no-spu --no-sub-autodetect-file`) to dodge the recursion bug when
VLC is the only option, (d) warns loudly when it falls back to VLC. Tests updated + green.
**To actually close it (one command, needs sudo — do it yourself):**
```
sudo apt install -y mpv     # then the player auto-selects mpv; nothing else to change
```
Verify: `SHOW_VIDEO=data/previews/thriller_side_by_side_csv.mp4 python tools/show_display.py`.

## Lane 2 — `fix/twitch-source-reprep`  (💻 DONE in this branch)
**Problem:** twitchy / limb-snapping. Root cause = per-frame jitter from the GVHMR landmark
detection, baked into the deployed policy (trained on **unfiltered** motion).
**Done here:** re-prepped the raw retarget through the now-wired filter →
`data/motions/thriller/thriller_g1_clean.csv` + `data/telemetry/twitch_reprep_20260713/`.
Measured: **jerk peak 101,701 → 4,806 rad/s³ (÷21), spike frames 67 → 4**, fidelity kept
(DOF RMS Δ 0.033 rad). **This does NOT fix the robot by itself** — it is the clean INPUT for
Lane 3 (retrain). Hand this branch's `.csv`/`.npz` to Lane 3.

## Lane 3 — `train/latency-curriculum-v5`  (☁️ needs a GPU box)
**Do exactly `docs/RETRAIN_RUNBOOK.md`.** Recipe already committed (`cloud/sim2real_task_v5.py`,
`cloud/train_v5_curriculum.sh`). Train on Lane 2's clean `.npz`. Gates: 40 ms+push survival AND
nominal drift < 1 m AND held-out ≥ 99%. **Delete the box when artifacts are pulled.**
Deliverable back to `main`: `data/policies/thriller_v5fid/` + signed `heldout_verdict.json`.

## Lane 4 — `feat/airborne-contact-guard`  (💻 code · 🤖 validate)
**Problem (the origin of this whole thread):** a suspended / barely-grounded robot thrashes when
switched into a policy, because `LegOdometry` assumes a planted foot and fabricates base velocity
(see the diagnosis in chat + `pipeline/leg_odometry.py:8`). Neither the start guard nor the fall
detector checks ground contact (both are orientation-only).
**Constraint found:** this G1's `unitree_hg` LowState has **no foot-force sensor** (foot_force is
only on `unitree_go`) — so the guard must be a **kinematic/IMU heuristic**, and MUST be
hardware-validated before it's trusted.
**Design (implement in `pipeline/deploy_runtime.py`, unit-test in `tests/`):**
1. **Start guard `_check_feet_planted(q, dq, imu)` at t0**, BEFORE releasing onboard (like
   `_check_start_upright`). At the ready pose the robot is nearly still on the ground; command a
   tiny, bounded ankle dither (or use the first 0.2 s of settling) and check that the two feet's
   *implied* base velocities (`LegOdometry` per-foot `per_foot_v`) **agree** and are near-zero.
   Suspended → the two disagree / are non-zero even at rest. Refuse to arm on failure (safe: onboard
   still holds).
2. **In-loop trip (debounced, env-gated `AIRBORNE_TRIP=1`)**: if the two per-foot base-velocity
   estimates diverge beyond a threshold for N consecutive ticks (contact-confidence collapse, cf.
   `FusedBaseEstimator`'s `contact` term), damp + hand back — same path as `_check_fall`.
   Conservative threshold + debounce so a legitimate step (one foot lifted) never trips it.
**Validation (🤖, gantry):** confirm it (a) REFUSES when the robot is hoisted feet-off, (b) does
NOT refuse/trip during a normal grounded dance (no false positives across a full run). Only then
raise it from advisory to enforcing by default.

## Lane 5 — `hw/validate-v5-and-exit-fix`  (🤖 needs robot + damping remote)
The 3 items that are purely hardware-gated (cannot be automated):
1. **Validate the v5 policy** (Lane 3) on hardware — gantry → slack-tether → free. Bring tether
   slack (drift); **watch for the ~45 s buckle**.
2. **Stand-exit handback** — never got a clean end-of-dance test. This run finally exercises it.
3. **Drift** — confirm root-pos-weight-1.0 + curriculum actually holds station on hardware.
Record outcomes in the app (Clean/Aborted/Incident) so a bad policy auto-demotes.

---

## Resync to `main`
Each lane is small and mostly touches disjoint files (deploy_runtime is the only shared file —
Lane 4 owns it; sequence Lane 4 before anything else that edits it). Suggested flow:
```bash
# push a lane for delegation (from a machine with creds):
git push handoff <branch>          # or: git push origin <branch>
# when a lane is reviewed + (for 🤖 lanes) hardware-validated:
git checkout main && git merge --no-ff <branch> -m "merge <branch>: <what/gate passed>"
```
Merge order: `fix/show-display-mpv`, `fix/twitch-source-reprep` (safe now) →
`feat/airborne-contact-guard` (after gantry validation) → `train/latency-curriculum-v5`
(after the ≥99% gate) → close `hw/validate-v5-and-exit-fix` (record-only; its artifact is the
signed outcome + a PROJECT_STATE entry).
```
