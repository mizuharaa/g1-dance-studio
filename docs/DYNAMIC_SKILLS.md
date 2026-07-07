# DYNAMIC SKILLS (backflip) — hardware-risk memo & decision gate

> **This document gates any PHYSICAL attempt of an acro skill.** The training
> lane (`cloud/dynamic_skills_task.py`, task `Mjlab-Tracking-Flat-Unitree-G1-Acro`)
> is SIM-ONLY by design. A hardware attempt is a separate, explicit human
> decision made against the evidence checklist at the bottom — and the current
> recommendation is **DO NOT attempt on hardware** (§5).

Status 2026-07-06: reference verified, `train-acro-1` training on the box
(10k iters); sim results section below is filled by `cloud/acro_eval.py` /
`exports/acro1/RESULT.txt` when the autopilot finishes.

## 1. What is being trained

A single standing backflip, 5.57 s reference (167 frames @ 30 fps), tracked by
the same BeyondMimic/mjlab machinery as the dances but under a dedicated task
profile (rationale documented in `cloud/dynamic_skills_task.py`):

- flip-aware terminations: stock deviation checks suppressed inside a
  precomputed flight-grace window (46/277 frames at 50 Hz = 16.6%) and relaxed
  outside it — mid-air phase lag must not kill the episode where the learning
  signal lives;
- **full effort limits, no torque penalties** — flips need peak torque; the
  s2r thermal-realism recipe is deliberately absent;
- **no push events** — the resulting policy is NOT push-robust (honest,
  documented consequence);
- RSI/adaptive-sampling tuned for a ~6 s skill.

## 2. Reference provenance & verification (both committed)

- Source: HuggingFace `LuluCao/KungfuAthleteBot` (Apache-2.0), clips 278/280 —
  martial-arts video → GVHMR → GMR retarget to G1 29-DoF, screened upstream.
  Full chain in `data/acro_refs/kungfuathletebot_backflip/PROVENANCE.txt`.
- Verified by `tools/check_acro_reference.py` (the acro replacement for the
  show vet, which a backflip legitimately violates): **PASS** —
  flip rotation 6.36 rad horizontal (1.01 rev, mocap yaw 4.08 rad reported
  separately, does not mask the flip); airborne 0.50 s (frames 70–84, max foot
  rise 1.60 m); joint limits clean; ends recoverable (root z 1.01 m, torso
  upright, gravity-z −0.995); quats normalized.
- Physics advisory from the same check: **peak joint velocity 40.6 rad/s with
  6 joints over 20 rad/s** — at or beyond true G1 motor ratings (20–37 rad/s).
  The RL policy does not have to reproduce reference velocities exactly, but
  this says the maneuver lives at the hardware's envelope, not inside it.

## 3. Why a backflip is categorically riskier than anything we have deployed

Every risk we have managed so far assumed an upright, grounded robot. A flip
breaks all of those assumptions at once:

1. **Full inversion + flight.** The failure mode is not "falls over" (dance
   worst case, tether-catchable) but "lands inverted or mid-rotation": head/
   torso-first impact of a 35 kg machine from ~1.1 m apex. No damping mode
   helps mid-air — damping a flying robot just guarantees a crumple landing.
2. **The tether becomes a hazard, not a safety.** A line taut at the wrong
   moment mid-rotation converts a maybe-landing into a certain crash and loads
   the anchor dynamically (snatch load ≫ static weight). Gantry-catch
   procedures from robot day DO NOT TRANSFER.
3. **The remote's B-damping is the only stop, and it is useless in flight**
   (no torque-cut e-stop on this G1 — established robot-day fact). The
   operator has no meaningful abort between launch and touchdown (~0.5 s).
4. **Deploy-stack assumptions break.** The proven estimator (leg odometry)
   assumes stance contact; in flight it is garbage, and `base_lin_vel` feeds
   the policy obs. The activation-ramp/standby machinery, telemetry
   thresholds, and the sim exam DR envelope were all validated on grounded
   motion only.
5. **Landing loads.** Launch vz ~1.0 m/s, apex 1.13 m, landing compresses
   through ankle/knee at full effort limits — the training profile explicitly
   removed the torque penalties that kept the dance policy inside the ankles'
   comfortable envelope (dance ankle RMS 8.9 Nm; flip landings will spike far
   beyond; exact numbers from the sim eval below). Repeated attempts risk
   actuator/structure damage even on "successful" landings.
6. **No push-robustness by design** (§1). Ground irregularity, a breeze of
   contact, or estimator noise at launch has no trained recovery.
7. **Sim2real for aerials is research-grade.** Our validated pipeline evidence
   (Thriller: sim gate numbers reproduced on hardware) is for grounded
   choreography. Vendor backflip demos run on internal controllers tuned per
   maneuver; nothing in our chain has demonstrated aerial transfer.

## 4. Containment rules (in force now)

- Acro references/policies **never** enter `data/dances/`, the show library,
  set-lists, or the deploy bundle machinery (`cloud/dynamic_skills_task.py`
  documents the vet bypass; `tools/check_acro_reference.py` is the only intake
  gate). The show promotion path cannot see them.
- `exports/acro1/RESULT.txt` carries the SIM-ONLY banner; `cloud/acro_eval.py`
  audits landing success + peak torque/velocity, and `headless_render_acro.py`
  produces the review video.

## 5. Recommendation and decision gate

**Recommendation: treat the backflip as a SIM SHOWPIECE (video artifact) for
the foreseeable future.** It is a compelling R&D demo and a useful stress test
of the training recipe; it is not a paid-show skill, and the risk/benefit for
this robot (the company's performance asset, no spare) is not close.

If a hardware attempt is ever seriously considered, ALL of the following must
exist first — and then the user still makes the call in person:

- [ ] Sim: landing success ≥ 99% under DR + obs noise (acro_eval, held-out
      seeds) — *pending, fills from `exports/acro1/`*
- [ ] Sim: peak joint torque/velocity audit vs motor ratings with margins
      stated — *pending, same source*
- [ ] A push/perturbation robustness pass (currently absent BY DESIGN)
- [ ] A flight-phase-valid state estimator (leg odometry is stance-only)
- [ ] An impact-load engineering review vs G1 structural/actuator specs
      (vendor guidance; warranty implications)
- [ ] A physical containment plan that does not rely on the tether mid-flip
      (crash mats, exclusion zone, robot writes-off-able)
- [ ] Sim video reviewed by the user + this memo re-read on the day

Anything less and the answer stays no.

## 6. Sim results

**Attempt 1 (train-acro-1, 10k iters, 2026-07-06): FAILED by reward hacking —
landed 0/64 with rotation 0.000 rev while 64/64 "survived upright".** The
policy learned to skip the flip: with all deviation checks suppressed inside
the flight-grace window and the reference upright again after touchdown, never
leaving the ground was termination-free and paid better than attempting the
flip. Evidence: `data/reports/acro/attempt1/` (RESULT.txt + acro_eval.json),
render `data/previews/rollout_acro1.mp4`. Peak torques even without flipping:
knee 114–123/139 Nm, ankle saturated 50/50 Nm — supports §3's landing-load
concern.

**Attempt 2 (train-acro-2, 10k iters, 2026-07-07): FAILED — and the failure
mode changed exactly as the fix predicted, which is what makes the finding
conclusive.** The in-grace flip-skip detector (`IN_GRACE_ORI_THRESHOLD = 1.7`)
removed the skip optimum: nobody "survives upright" anymore (0/64 survived).
The policy now genuinely attempts the launch — knee torque saturates at its
exact rating (139/139 Nm, p95 134), ankle 50/50, waist 50/50, impact 199 m/s²
— but achieves only **0.165 rad mean rotation of the 7.34 rad required (~2%)**
before dying at the apex check. Evidence: `data/reports/acro/attempt2/`,
render `data/previews/rollout_acro2.mp4`.

**LANE VERDICT (2 attempts, cross-checked): this reference is beyond the G1's
actuator envelope at TRUE effort limits.** Independent corroboration from
intake (§2): the reference's own peak joint velocities (40.6 rad/s, 6 joints
over the 20–37 rad/s ratings) already said the maneuver lives outside the
hardware envelope. A1 (skip-friendly terminations) and A2 (skip-punishing)
bracket the recipe space: with tracking-RL and honest limits, the G1 cannot
produce the launch impulse + angular momentum this human flip demands.
Publicly demonstrated G1 flips use maneuver-specific trajectories with much
lower amplitude, not human-mocap tracking.

**Path forward (USER decision, not auto-continued):** a backflip requires a
G1-FEASIBLE reference — authored or sourced at lower amplitude (smaller foot
rise, longer wind-up, per-joint torque-aware retiming) — which is a
choreography/R&D investment, not an attempt-3 knob. Until then the backflip
is closed as: sim evidence says no, hardware question never opens (§5's
recommendation stands, now with data).
