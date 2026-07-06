# Fluidity forensics — why the legs look clunky (2026-07-06, evening)

**Question.** The user watched the full Thriller on hardware (promoted s2r-b,
4 full-dance runs) and reports: legs don't replicate the reference, movements
"clunky", "very unstable, not natural/fluid" — worse than the sim rollout video.
This report turns that into numbers and attributes causes so the fix is chosen
on evidence.

**Data.** 4 full-dance telemetry runs (`data/telemetry/20260706-{114445,115004,
115456,133905}_ground-run-legodom.npz`, 2589 ticks @50 Hz each, tick-aligned to
`data/policies/thriller/thriller_deploy.npz`), the promoted policy ONNX run
offline through the real deploy obs code (perfect-tracking replays, clean-obs
and leg-odom-obs), the box's sim tracking eval of the same checkpoint
(`arm_tracking_s2rb_baseline.json`, nominal, 64 envs, joint-space |q−ref|), the
gap check `s2rb_gap_check.json`, and the retarget-fidelity artifacts.
Tool: `tools/fluidity_forensics.py` → `data/reports/fluidity_forensics.json`.

---

## 1. Headline verdict (one paragraph)

The legs' failure to replicate the reference is **almost entirely already in
sim**: hardware leg tracking is 13.4° RMS vs **12.96° RMS for the same policy
in nominal sim** — the policy *chooses* to under-perform the leg choreography
(executed leg amplitude ≈ 0.35–0.44 of reference; knees 20°+ RMS in sim too).
What hardware **adds** is the *instability look*: a near-constant **0.16 rad/s
(≈9°/s) of 2–10 Hz roll/pitch body wobble, 91 % of it incoherent with the
choreography**, plus 3.3× the reference's >10 Hz leg-velocity buzz. The wobble's
energy source is **policy-intrinsic** (the policy outputs 3.1× the 2–10 Hz leg
action energy the dance needs, even on perfectly clean observations); hardware
converts it into visible sway through the **80–105 ms leg plant lag**. The
leg-odometry/obs-noise pathway is measured and **negligible** (≤2 % effect).
The reference itself lost little for legs (82 % of 2–10 Hz leg velocity energy
kept; the two kicks blunted 13.7→8.4 rad/s) — the clamp damage was an arm
problem. **No v3 variant addresses the leg wobble mechanism; v3a is the only
one aimed at leg under-tracking, and it simultaneously relaxes the action-rate
penalty, which risks making the wobble worse.**

---

## 2. Leg tracking vs reference (deliverable 1)

RMS of q−ref in degrees, 4-run aggregate (p95 = 95th pct of |q−ref|, full run).
Sim = the same checkpoint, same metric, nominal sim (box eval).

| joint | 0–13 s | 13–17.5 s | 25–36 s | 40–49.5 s | full | p95 | SIM full | SIM p95 |
|---|---|---|---|---|---|---|---|---|
| left_hip_pitch | 6.3 | 13.6 | 13.6 | 12.7 | 11.7 | 25.1 | 12.2 | 25.8 |
| left_hip_roll | 9.6 | 6.3 | 9.5 | 6.5 | 8.6 | 16.0 | 9.7 | 17.1 |
| left_hip_yaw | 6.5 | 6.6 | 9.0 | 9.1 | 7.7 | 14.9 | 6.6 | 12.2 |
| **left_knee** | 16.1 | **34.3** | 19.4 | 21.4 | **21.2** | **38.8** | 19.8 | 37.2 |
| **left_ankle_pitch** | 17.4 | **27.4** | 16.4 | 17.0 | **18.1** | 34.1 | 17.7 | 33.0 |
| left_ankle_roll | 8.5 | 8.4 | 8.3 | 7.7 | 8.1 | 14.6 | 8.8 | 14.9 |
| right_hip_pitch | 8.3 | 19.4 | 13.2 | 12.4 | 12.6 | 25.2 | 12.4 | 24.4 |
| right_hip_roll | 6.8 | 7.7 | 11.6 | 11.1 | 9.7 | 19.4 | 9.0 | 16.6 |
| right_hip_yaw | 6.9 | 5.7 | 8.7 | 8.2 | 7.3 | 14.0 | 6.5 | 12.2 |
| **right_knee** | 15.4 | **27.1** | 23.1 | 24.6 | **22.1** | **39.9** | 20.7 | 36.9 |
| right_ankle_pitch | 9.7 | 15.9 | 17.6 | 18.1 | 14.5 | 29.2 | 14.4 | 28.7 |
| right_ankle_roll | 5.0 | 5.3 | 6.2 | 6.8 | 6.1 | 10.9 | 5.8 | 10.0 |

Group rollup (track = q−ref; plant = target−q, what the motors fail to execute;
policy = target−ref, what the policy chose not to command):

| group | 0–13 s | 13–17.5 s | 25–36 s | 40–49.5 s | full | plant | policy | SIM full |
|---|---|---|---|---|---|---|---|---|
| legs | 10.5 | **17.7** | 13.9 | 14.2 | **13.4** | 10.9 | 17.2 | **12.96** |
| waist | 6.4 | 9.7 | 5.6 | 7.3 | 7.3 | 14.9 | 14.8 | 5.85 |
| arms | 12.7 | 19.8 | 13.5 | 13.1 | 14.3 | 9.9 | 17.3 | 13.81 |

Readings:

- **Worst leg joints: knees (21–22° RMS, p95 ≈ 39°) and ankle_pitch (14–18°),
  worst in 13–17.5 s** (side-step window: left knee 34.3° RMS). This is the
  window the user perceives as most broken — and it is, by 1.6× vs the rest.
- **Hardware ≈ sim on tracking.** Legs 13.4 vs 12.96, knees +5–7 %, ankles +1 %.
  Per-run spread 12.7–14.1 (tight). The "legs don't replicate the reference"
  deficit survives an ideal robot: it's in the policy.
- Legs are not worse than arms in RMS — but they are in **executed amplitude**:
  regression of q on ref gives legs 0.35–0.44 (ankle_pitch 0.12–0.18!) vs arms
  mean 0.82. The audience sees legs doing <½ the choreography, arms doing ~80 %.
- The waist row shows the mechanism: plant error (14.9°) is *twice* the tracking
  error (7.3°) — the policy swings targets as **torque levers** (PD setpoint
  excursions), the plant low-passes them onto the reference. Leg "policy error"
  17.2° > track 13.4° for the same reason. Target excursions ≠ pose intent; the
  policy is a balance controller that happens to dance.
- Timing is NOT the story: visible lag of q behind ref is 0–40 ms on legs
  (hips/knees 0–25 ms) vs arms mean 33 ms — legs are on the beat, just small.

## 3. Jitter / smoothness (deliverable 2)

Action-rate mean |Δa| per tick (policy units), legs; four traces:

| trace | legs | waist | arms |
|---|---|---|---|
| reference demand (what a perfect tracker needs) | 0.0278 | 0.0151 | 0.0648 |
| **clean-obs ONNX replay (policy intrinsic)** | **0.0672** | 0.0690 | 0.0737 |
| leg-odom-obs ONNX replay (adds estimator artifacts) | 0.0679 | 0.0691 | 0.0739 |
| hardware (4-run mean) | 0.0571 | 0.0681 | 0.0671 |

**The policy emits 2.4× the leg action-rate the choreography requires, on
perfectly clean observations.** The deploy estimator adds +1 %. Hardware sits
*below* the clean replay — obs noise is not amplifying action jitter.

Band-split action RMS (0–2 / 2–10 / >10 Hz), legs:

| trace | motion 0–2 Hz | wobble 2–10 Hz | buzz 10–25 Hz |
|---|---|---|---|
| reference demand | 0.535 | 0.065 | 0.003 |
| clean replay | 0.735 | **0.199** | 0.016 |
| leg-odom replay | 0.738 | 0.203 | 0.015 |
| hardware | 0.580 | 0.179 | 0.017 |

The 2–10 Hz leg action content is **3.1× demand and fully present with clean
obs** — it is the policy's learned balance-correction chatter, not a hardware
artifact.

Measured joint velocity + jerk vs what the reference demands (HW/ref ratio):

| band | legs vel | legs jerk | arms vel | arms jerk |
|---|---|---|---|---|
| motion 0–2 Hz | **0.64** | 1.16 | 0.84 | 0.91 |
| wobble 2–10 Hz | 1.09 | **1.63** | 0.82 | 0.85 |
| buzz 10–25 Hz | **3.31** | **2.68** | 1.27 | 1.07 |

The leg signature of "clunky" in one row each: legs do **64 % of the
low-frequency motion** (arms 84 %), yet **1.6× the 2–10 Hz jerk and 3.3× the
>10 Hz velocity buzz**. Less dance, more shake — arms show neither excess.

## 4. Stability wobble (deliverable 3)

Roll/pitch gyro band RMS (rad/s), phases classified from reference foot heights
(swing = a foot >6 cm up, 87 % of this step-heavy dance; quasi-stance = both
feet planted, 8 %; ramp = 0–2.5 s activation):

| phase | HW 2–10 Hz | ref demand | HW >10 Hz | HW 0–2 Hz |
|---|---|---|---|---|
| ramp (should be still) | **0.175** (runs: 0.27/0.19/0.14/0.10) | 0.000 | 0.047 | 0.171 |
| quasi-stance | 0.043 | 0.013 | 0.011 | 0.079 |
| swing | **0.163** (runs: 0.158–0.167) | 0.291 | 0.048 | 0.191 |

By section (2–10 Hz, HW vs ref): 0–13 s 0.126/0.116 · 13–17.5 s 0.189/0.286 ·
25–36 s 0.157/0.227 · 40–49.5 s 0.182/0.394.

Coherence split (full run, Welch, of HW gyro power vs the reference pelvis
angular velocity): **2–10 Hz band is 91 % incoherent** (0.221 of 0.231 rad/s
RMS), 0–2 Hz 85 %, >10 Hz 93 %.

Readings:

- The "unstable look" is a **near-constant ~0.16 rad/s (9°/s) 2–10 Hz body
  wobble whenever the legs are working** — flat across sections (0.13–0.19)
  while the demand varies 0.12–0.39, and repeatable to ±3 % across all four
  runs. It is a deterministic property of the closed loop, not noise.
- It is **not the choreography**: 91 % incoherent with the reference. Worse,
  the demanded 2–10 Hz body rotation (kicks' body language, 0.29 rad/s in
  swing) is largely *not performed* — the robot swaps expressive body dynamics
  for its own sway. Double hit to "natural/fluid".
- The activation settle transient (0.10–0.27 rad/s wobble in the first 2.5 s,
  raw gyro up to 0.63 early) sets the "unstable" first impression.
- Quasi-stance wobble is small (0.04) — this is a *moving-leg* phenomenon,
  i.e. balance corrections during single support, not a standing instability.

## 5. Cause attribution (deliverable 4)

**(a) Policy-intrinsic (dominant for both symptoms).**
Evidence: sim tracking 12.96° ≈ HW 13.4° (the replication deficit needs no
hardware to exist); executed leg amplitude ~0.4 chosen by the policy (policy
error 17.2° > track error 13.4°); 2–10 Hz leg action energy 3.1× demand on
clean obs; action-rate 2.4× demand on clean obs. The s2r-b recipe bought its
wins (torque −35 %, drift 0.64 m, zero falls) with smoothness- and
torque-penalties plus heavy DR — the policy learned "small, safe, busy legs".

**(b) Obs-noise / estimator (negligible).**
Evidence: leg-odom replay vs clean replay: +1 % action-rate, +2 % wobble-band;
HW action-rate bursts correlate with stepping windows *less* than the
reference demand itself does (corr −0.08…+0.36 vs +0.51 for demand; step/stance
ratio 0.8–3.9 vs 6.8 for demand) — no stepping-phase obs-noise signature.
Bound: ≤5 % of either symptom.

**(c) PD softness / plant lag (the hardware-only additions: wobble + buzz).**
Evidence: leg plant lag vs commanded target 80–105 ms (hips 95–105, knees 80,
ankle_pitch 90; arms for comparison 80–140); ankle_pitch executes only
0.14–0.22 of its target excursions at kp 28.5 (gravity stiffness ≈ 202 Nm/rad,
audit); measured leg buzz (3.31×) far exceeds what the action buzz explains
(1.07× of replay) → plant dither; and the same policy renders a stable sim
rollout — the 0.16 rad/s incoherent wobble appears only where the intrinsic
2–10 Hz corrections meet an ~90 ms-lagged plant. Balance corrections arrive
half a wobble-period late → sustained sway. Note the trained cmd-delay DR
covered 0–20 ms; the *effective* leg plant behaves like 80–105 ms.

**(d) Reference style loss (small for legs, large for arms).**
Evidence: legs kept **82 %** of 2–10 Hz velocity energy through the blanket
clamp (arms 33 %, waist 43 %); leg peak velocity 13.7→8.4 rad/s (the two
kicks, −39 %); arms 56.4→8.5. The front-end did not make the legs clunky.

**Ranking (share of the perceived "clunky/unstable legs" gap):**

| rank | cause | share | fixes it |
|---|---|---|---|
| 1 | (a) policy style: under-performed leg amplitude + intrinsic 2–10 Hz action chatter | **~45 %** | train-side reward shaping (partly v3a) |
| 2 | (c) leg plant lag/softness converting chatter into visible 0.16 rad/s body wobble + 3.3× buzz | **~40 %** | actuator-lag DR in train; leg-gain treatment at deploy |
| 3 | (d) reference clamp: legs' 2 kicks blunted, 18 % HF loss | ~10 % | v3d sharp reference |
| 4 | (b) obs noise / leg-odom estimator | ~5 % | (already fine) |

Split by symptom: "legs don't replicate" ≈ 85 % (a) / 10 % (c) / 5 % (d).
"unstable, not fluid" ≈ 45 % (c) / 45 % (a) / 10 % (b+d). The (a)×(c)
interaction is multiplicative: either end can kill the wobble.

## 6. v3 decision guidance (deliverable 5)

| variant | what it touches | what this analysis says |
|---|---|---|
| v3a precision (action_rate −0.2→−0.1, body pos/ori ×1.5) | (a)-tracking | Right target, **wrong sign on the wobble**: relaxing action_rate will likely raise the intrinsic 2–10 Hz chatter that hardware turns into sway. Gate its rollout on the **2–10 Hz leg action band** (bar: ≤0.20, the s2r-b clean-replay level) — not only arm RMS. |
| v3b arm-plant (arm friction DR + arm gains ×2.5) | arms only | Does **nothing for legs** — leg actuator cfgs untouched. Judge it on arms. |
| v3c long (v3a ×10k iters) | (a) mildly | Same caveat as v3a. |
| v3d sharp ref | (d) | Restores arm accents (33→96 %) and the two leg kicks; **won't move leg amplitude (a) or wobble (c)**. |

**None of the four addresses:** (i) the leg plant-lag sim2real gap (cmd-delay
DR 0–20 ms trained vs 80–105 ms effective leg lag — the wobble mechanism),
(ii) leg-specific tracking pressure (knees at 20° RMS in sim are apparently
cheap under the current body-pos reward), (iii) any wobble-band penalty.

**If v3 disappoints — the single best v4 lever:** a **"calm-legs precision"
retrain** on the sharp reference = v3a's tracking weights **+ per-group
action-rate split (legs −0.3…−0.4, arms −0.1)** + a base angular-velocity
tracking/penalty term (penalize pelvis roll/pitch ang-vel deviation from the
reference — directly prices the 0.16 rad/s wobble) + first-order actuator-lag
DR on the legs stretched toward the measured 60–100 ms. That attacks (a) and
(c) at the source while keeping the torque/drift wins.

**Deploy-side experiments (no retrain, one tethered session):**
1. `ground_leg_kp_scale` already exists in the runtime (all runs so far = 1.0).
   A/B at **1.25 then 1.5** with the thermal monitor: lag ∝ 1/√kp, so 1.5×
   cuts leg plant lag ~18 % and stiffens the ankle against its 0.14–0.22 amp
   ratio. Watch ankle |τ| RMS (s2r-b now runs 8.9 Nm; the old 20 Nm heat was
   the pre-retrain policy at 2× — bounded risk, but gate on the 12 Nm RMS bar).
   Re-run `tools/fluidity_forensics.py` on the new telemetry; success = swing
   wobble-band < 0.10 rad/s with tracking no worse.
2. Deploy-side action low-pass on legs (~6 Hz) would cut the wobble source but
   adds ~25 ms in-band lag to balance corrections — **last resort, tethered
   only**.

**Follow-up measurement (box, 5-min job after the v3 trainings finish):** dump
one sim rollout trace (actions + pelvis ang-vel) of the promoted checkpoint so
the "sim doesn't wobble" claim gets a number (today it rests on the clean-obs
replay + the rollout video + zero-fall gap checks); same dump for each v3
candidate feeds the 2–10 Hz action gate above.

---

## Methods / caveats

- All tracking errors compare same-tick q vs reference (50 Hz, 2589 ticks,
  tick-for-tick; runs verified aligned, plant lag left IN — it's what the
  audience sees). Sections use the deploy-npz timeline (2.5 s ramp included).
- Replays run the real `policy.onnx` through the real `build_obs_odom` deploy
  code on perfect-tracking obs (npz truth), once with ideal disp/vel, once with
  the actual `LegOdometry` in the loop; they are open-loop in the plant sense
  (obs say "on reference"), so they measure the policy's *output* jitter, not
  closed-loop sim behavior. HW action-rate < replay is expected (the real loop
  operates slightly off-reference where the policy is smoother).
- Coherence (Welch, nperseg 256) is delay-insensitive; incoherent fraction is
  a fair "not linearly explained by choreography" measure.
- Leg lag-vs-target numbers carry the torque-lever caveat (low corr); the
  vs-reference lag is the visible-timing number.
- The band-filtered ramp-phase numbers were cross-checked against raw RMS
  (filtfilt edge effects are not the source; raw early-window gyro is larger).
- Foot-lift classification from reference foot z (bodies 6/12): base+3 cm
  planted, base+6 cm swing, ±0.3 s dilation; Thriller is 87 % swing-phase —
  the leg-odom stance assumption is violated most of the dance, yet (b) still
  measures negligible.

Numbers: `data/reports/fluidity_forensics.json`. Tool:
`tools/fluidity_forensics.py` (read-only on all inputs).
