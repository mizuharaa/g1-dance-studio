# G1 Dance — Handover / Resume (2026-07-05 ~00:20 ICT)

**Read this first, then `PROJECT_STATE.md` for the full day-by-day log.** This file is the
fast path to resume; PROJECT_STATE is the source of truth.

---

## ONE-LINE STATUS
Full-body Thriller runs on the real G1 tethered and balances up to ~10–30 s, but the same
policy that is **0 Nm ankle in sim uses ~15 Nm on hardware** — a sim2real gap that causes the
thermal wall, the sag/gain-boost, and the stepping brace. **Next: a targeted sim2real RETRAIN.
All deploy-side patching is done (it can't close the gap).**

## THE DECISIVE FINDING (why we retrain — don't re-litigate)
- Sim policy dances with **ankle torque mean 0.0 Nm** (keeps CoM over feet). Real robot: **~15 Nm**.
- That 0→15 Nm gap is the SHARED root of all three walls:
  - **Thermal**: ankle overworks only on real HW → 22.5 °C/min at 1.5× gains → fault < 2 min.
  - **Sag / needing 2× gain boost**: real tracking can't match sim.
  - **Stepping brace @14–16 s**: proven CHOREOGRAPHY-hard even with a *perfect* estimate
    (disambiguation test), i.e. the policy relies on tracking the real robot can't deliver.
- Prime suspect: **control/comms latency + actuator response** the sim doesn't model.
- Deploy-side patches tried and RULED OUT as the fix: leg-odom estimator, velocity smoothing,
  IMU fusion, leg-gain boost, gravity feedforward. All treat symptoms; none closes 15 Nm.

## THE PLAN (uncompromised — keeps the full-body ground dance)
Targeted **sim2real retrain** of the tracking policy with real conditions modeled:
1. **Latency randomization** (action/obs delay ~10–40 ms) — prime suspect.
2. **Actuator-response DR** (torque scale, friction, bandwidth).
3. **Torque/energy reward penalty** → learns low ankle torque → cool by design.
4. **Obs noise matching leg-odometry** (so our estimator IS the deploy obs; leg-odom is
   97.8 % within the policy's ±0.5 m/s trained band — already good enough).
5. **Mass / gain / push DR** → robust balance through stepping.
Reuses everything already built (leg-odom, deploy runtime, safety spine, thermal monitor).

## IMMEDIATE NEXT ACTION (all sim/code, NO robot)
1. Author the retrain config: extend the mjlab tracking task
   (`Mjlab-Tracking-Flat-Unitree-G1`) on the GPU box with the 5 items above.
2. **Verify in sim BEFORE spending GPU hours**: confirm the retrained policy keeps sim ankle
   torque low AND survives injected latency + pushes (the exact things that broke it on HW).
   Reuse the offline check pattern in `cloud/sim_ankle.py` (on the box) + the disambiguation
   method in scratchpad.
3. Train (GPU box), export ONNX, run held-out gate, then ONE clean tethered HW test.

## KEY FACTS / INFRA
- **GPU box** (alive): `root@103.245.250.152:46936`, key `~/g1-dance/.secrets/greennode_ssh_key`,
  work dir on box `/workspace/notebook-data` (envs/mjlab, repos/mjlab, cloud/ scripts,
  motions/thriller_deploy.npz, run_job.sh for detached tmux jobs).
- **Training gains == deploy gains** (verified: ankle kp 29, knee 99, hip 40) — NOT a gain bug.
- Robot model + gains config on box: `repos/mjlab/.../unitree_g1/g1_constants.py`.
- **Proven gantry policy**: `data/policies/thriller/` (policy.onnx, policy_meta.json,
  thriller_deploy.npz) — 100 % in sim, full 160-dim obs.
- **Deploy runtime**: `pipeline/deploy_runtime.py`. Modes: `read` (safe, default),
  `move-to-default`, `run`, `stand-hold`, `ground-run`, `ground-run-odom`, `ground-run-legodom`.
- **Leg odometry + fused estimator + gravity_comp**: `pipeline/leg_odometry.py` (all offline-
  validated; leg-odom is the deploy estimator that works, fusion/FF shelved as not-the-fix).
- Env `tv` = robot runtime (unitree_sdk2py, onnxruntime, mujoco). Env `g1dance` = pipeline/tests.

## ROBOT SAFETY (non-negotiable — a 35 kg robot, no torque-cut e-stop)
- NEVER command motion without: human present, tether rigged to catch, **damping remote in hand**.
- All motion modes need `--i-will-watch-the-robot` AND env `CONFIRMED_BY_HUMAN=alois`.
- Robot iface `enp0s31f6`; robot IP `192.168.123.164`.
- **Motion-service gotcha**: releasing it for low-level control freezes `rt/odommodestate` AND
  can strand the remote — the runtime now auto-restores `SelectMode("ai")` on exit. If the remote
  won't pair, run `SelectMode("ai")` from the laptop or reboot the robot.
- **Signal the PYTHON pid, not the bash wrapper**, to stop a run (else the child orphans and holds
  the robot energized — happened twice). Use `pgrep -f "python.*deploy_runtime"`.
- **Thermal**: read `motor_state[i].temperature`; warn ~80 °C, fault ~90 °C. Monitor drains DDS to
  the LATEST msg (a stale-backlog bug once let a motor hit 80 °C blind — fixed).

## HOW TO RESUME IN A FRESH SESSION
Start the new session in `~/g1-dance` and paste:

> Resuming the G1 dance project. Read `HANDOVER.md` then `PROJECT_STATE.md`. We concluded the
> thermal/balance/stepping failures are one sim2real gap (sim ankle 0 Nm vs real 15 Nm) and the
> fix is a targeted sim2real retrain (latency + actuator DR + torque penalty + obs noise + mass/
> push DR), not more deploy patching. Start by authoring the retrain config on the GPU box and
> verifying in sim (ankle torque stays low + survives injected latency/pushes) BEFORE training.
> Do not run the robot until I'm rigged with the damping remote.

That's enough for a fresh Claude to pick up exactly here.
