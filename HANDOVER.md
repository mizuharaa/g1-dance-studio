# G1 Dance — Handover / Resume (2026-07-05 ~00:20 ICT)

**Read this first, then `PROJECT_STATE.md` for the full day-by-day log.** This file is the
fast path to resume; PROJECT_STATE is the source of truth.

---

## ONE-LINE STATUS
Full-body Thriller runs on the real G1 tethered and balances up to ~10–30 s. The old
"sim ankle 0 Nm vs real 15 Nm = one latency-shaped sim2real gap" story was AUDITED AND
PARTIALLY OVERTURNED (2026-07-05, docs/first_principles_audit.md — read it): the sim number
was a measurement artifact; the corrected picture is an ankle-hungry policy (~6–8 Nm mean in
clean sim) + static PD sag + REAL DEPLOY OBS BUGS (now fixed). A corrected retrain
(train-thriller-s2r) is running with an auto-gate.

## THE CORRECTED FINDINGS (verdict: CRITICAL-MISTAKE-FOUND, caught before GPU spend)
- **Sim "0 Nm ankle" was false**: sim_ankle.py indexed actuator-ordered `actuator_force`
  with joint-tree indices — it measured the LEFT WRIST. Correct (qfrc_actuator): sim ankle
  ~6–8 Nm mean / 15–20 p95, transients saturating the 50 Nm clamp. Honest sim2real excess ≈ 2×.
- **The real ~15 Nm has a STATIC signature** (kp·sag = 28.5 × 0.506 rad ≈ 14.4 Nm): latency
  cannot produce steady sag; the trained ankle PD stiffness (57 Nm/rad) is below the gravity
  destabilizing stiffness (~202 Nm/rad) — the POLICY is the balance controller, pure PD topples.
- **Deploy obs bugs (FIXED, free wins)**: (a) reference world yaw (t=0: 90.3°) was never
  aligned to the IMU frame — measured action corruption ≈ the whole action signal at 90°;
  fixed with yaw re-anchor at policy start. (b) pelvis IMU used where training anchors on
  torso_link — fixed via waist FK (validated vs MuJoCo). (c) gravity-FF was never actually
  falsified (the test sent ~zero ankle FF). (d) per-run telemetry now records tau_est/temps.
- **Latency still matters dynamically** (measured: 20 ms delay halves survival, 40 ms kills)
  but as robustness hygiene (train 0–20 ms), not the headline.
- The 14–16 s brace is one of MANY high-lean/step segments (quasi-static scan: 16 segments
  >30 Nm, worst 43–47 s) — per-section eval in sim_gap_check pinpoints what remains infeasible.

## THE PLAN (recipe v2 — running as train-thriller-s2r)
1. **Torque penalty headline**: joint_torques_l2 -2e-5 + ankle_torque_l2 -4e-4 (qfrc-based).
2. **System-ID-informed mass/CoM**: +hands payload, torso ≥ model mass, CoM x ±5 cm,
   ankle zero-offset ±0.08 rad.
3. **Actuator DR (modest)**: gains ±15 %, effort 0.8–1.0, friction 0–0.4, armature 0.9–1.4.
4. **Obs dynamics matching leg-odom**: lag 30–80 ms + slew + stance-break bias episodes
   (custom obs term), obs delay 0–20 ms.
5. **Latency DR 0–20 ms** (40 ms is EVAL-ONLY), 20 s episodes, action_rate_l2 -0.2.
Gate: cloud/sim_gap_check.py — FULL-motion, 7 conditions; survival ≥99 % nominal / ≥95 % worst,
ankle mean ≤6/8 Nm, p95 ≤15/20, RMS ≤12 Nm (thermal), mpkpe ≤0.31, per-section stats.
(NOTE: the old heldout_eval "100 %" only certified the FIRST 10 s — episode_length_s was
never overridden. sim_gap_check supersedes it.)

## IMMEDIATE NEXT ACTION
1. Read `exports/thriller_s2r/RESULT.txt` on the box (job s2r-autopilot waits for
   train-thriller-s2r, exports, runs the gate, writes the verdict).
2. GATE_PASS → pull policy.onnx, stage data/policies/thriller_s2r/ (reuse policy_meta.json —
   same gains/scales/obs), render rollout video for visual sign-off, then ONE tethered HW test
   (human + damping remote; robot-facing gates unchanged).
3. GATE_FAIL → per-section stats say which segments fail → targeted choreography edit
   (music-sync-preserving) or reward re-weighting for attempt 2.
4. STRATEGY (user decision pending, audit §5): recommend HYBRID — arm-dance-over-onboard
   as the bookable show baseline (P≈0.85), full-body retrain as the premium act.

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
