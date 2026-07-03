# ROBOT DAY — one-page card (print me)

**Golden rule:** this G1 has **NO torque-cutting e-stop**. The remote's **B (damping)** button
and the **power switch** are the only real stops. Keep the remote in your hand whenever motors
are powered. If unsure — hit **B**, then power.

## Before you touch the robot
- [ ] On the laptop: `./scripts/preflight_robot_day --dance thriller` → must print **GO** (or GO-with-caution you understand).
- [ ] Dance is **show-ready** (held-out ≥99% + sim-verified). If it isn't, STOP — no hardware.
- [ ] Gantry/harness rigged, straps rated & locked. Robot can hang with **feet ~5 cm off ground**.
- [ ] 2 m clear radius, hard flat floor. Nobody within arm's reach once powered.
- [ ] Remote e-stop tested **today** (press B, confirm damping) and in your hand.
- [ ] Fire extinguisher / cut-power plan known. Battery ≥ 50%.

## Order of operations (never skip ahead)
1. **Power on, DON'T deploy.** Health check: all 29 motors report, no faults, firmware noted. (~30 min)
2. **Laptop → robot LAN.** `./deploy/01_pc2_install.sh --yes-install` then `./deploy/02_push_bundle.sh --dance thriller --yes-push` (needs `CONFIRMED_BY_HUMAN=alois`).
3. **Verify launch line & damping** on PC2, then `touch LAUNCH_LINE_VERIFIED` in the bundle (runbook §3).
4. **Gantry, feet OFF ground:** `./deploy/10_gantry_test.sh --dance thriller --stage gantry --gantry-confirmed --estop-confirmed --arm` → controller starts in **DAMPING HOLD** (no motion yet).
5. **§3a KILL TEST (do this before ANY motion):** with feet off ground, run `deploy/kill_now.sh` and **watch** — does the robot go limp (damp) or hold/lurch? Record it. This is the single most important measurement of the day.
6. **Arm playback from the remote** (operator sequence) — watch joints track the dance in the air. Abort criteria below.
7. **Ground, harnessed, line taut:** `--stage ground ... --arm`, stand-and-hold, then slow first motion, then full dance.
8. **Push test** only after a clean ground run.

## STOP IMMEDIATELY (hit B, then power) if:
- Any joint jerks, buzzes, oscillates, or moves to a limit.
- Robot leans/sags past the reference, or tracking visibly diverges.
- Any smoke, burning smell, unusual heat, or motor fault light.
- Controller log shows NaN / "limit" / missed control ticks.
- Battery sag, comms drop, or **anything** you didn't expect.

**Abort ladder:** 1) remote **B** (in hand, beats everything) → 2) `deploy/kill_now.sh` → 3) **power switch**.
Never approach the robot until you've **visually** confirmed it's still.
