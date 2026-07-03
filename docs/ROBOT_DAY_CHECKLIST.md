# ROBOT DAY — one-page card (print me)

**Golden rule:** this G1 has **NO torque-cutting e-stop**. The remote's **B (damping)** button
and the **power switch** are the only real stops. Keep the remote in your hand whenever motors
are powered. If unsure — hit **B**, then power.

## Before you touch the robot
- [ ] `export CONFIRMED_BY_HUMAN=alois`; `./scripts/preflight_robot_day --stage gantry` → **GO** (or GO-with-caution you understand).
- [ ] Gantry rigged, straps rated & locked. Robot hangs with **feet ~5 cm off ground**. (Gantry can lower to a taut line for the ground stages.)
- [ ] Policy bar: **gantry accepts the current policy (98.4%)**. **Ground-free needs ≥99%** OR a conscious `--informed-override` — decide in the moment.
- [ ] 2 m clear radius, hard flat floor. Nobody within arm's reach once powered.
- [ ] Remote e-stop tested **today** (press B, confirm damping) and in your hand.
- [ ] Cut-power plan known. Battery ≥ 50%.

## Order of operations (never skip ahead)
1. **Network up.** `cat /sys/class/net/enp0s31f6/carrier`=1; `ping -c2 192.168.123.164`; `ssh unitree@192.168.123.164` (prompt: **1**).
2. **Controller onto PC2 (first — the likely time-sink).** `./deploy/01_pc2_install.sh --yes-install`. If `docker pull` fails (no internet on robot LAN): `docker save qiayuanl/unitree:jazzy | ssh unitree@192.168.123.164 'docker load'`.
3. **Power on, DON'T deploy.** Health check: all 29 motors report, no faults, firmware noted. (~30 min)
4. **Joint-calibration check (DO NOT SKIP — fall risk):** standby, feet off ground, SDK env: `python deploy/check_joint_calibration.py --meta data/policies/thriller/policy_meta.json` → must be **GO** (standby pose matches sim). NO-GO = do not run the policy.
5. **Push the bundle.** `./deploy/02_push_bundle.sh --dance thriller --yes-push`. Verify launch line & that the controller loads the **bundle's SIM gains (policy_meta.json)**, then `touch SIM_GAINS_LOADED LAUNCH_LINE_VERIFIED` on PC2.
6. **Gantry, feet OFF ground:** `./deploy/10_gantry_test.sh --dance thriller --stage gantry --gantry-confirmed --estop-confirmed --arm` → controller starts in **DAMPING HOLD** (no motion yet).
7. **§3a KILL TEST (before ANY motion):** feet off ground, run `deploy/kill_now.sh` and **watch** — limp (damp) or hold/lurch? Record it. The single most important measurement of the day.
8. **Arm playback from the remote** — watch joints track the dance in the air. Check the estimator reads sane (~0 velocity) on the gantry.
9. **Ground stages (gantry lowered to taut line → slack line):** `--stage ground-tethered` then `--stage ground-free` — each with its typed phrase + prior-stage gate. Ground-free needs the kill→damping + estimator confirmations (and ≥99% or `--informed-override`).
10. **Push test** only after a clean, repeated ground-free run.

## STOP IMMEDIATELY (hit B, then power) if:
- Any joint jerks, buzzes, oscillates, or moves to a limit.
- Robot leans/sags past the reference, or tracking visibly diverges.
- Any smoke, burning smell, unusual heat, or motor fault light.
- Controller log shows NaN / "limit" / missed control ticks.
- Battery sag, comms drop, or **anything** you didn't expect.

**Abort ladder:** 1) remote **B** (in hand, beats everything) → 2) `deploy/kill_now.sh` → 3) **power switch**.
Never approach the robot until you've **visually** confirmed it's still.
