All evidence is in hand: the confirmed/disputed/refuted findings, the repo's current plan-of-record state (PROJECT_STATE 2026-07-05 correction, authored `cloud/sim2real_task.py`, `cloud/sim_gap_check.py` gates), and I additionally verified in code that no yaw re-anchoring exists in any of the three obs builders (`deploy_runtime.py:253/292/346` all pass raw `ref_aquat` vs `imu_quat`) while the reference's t=0 torso yaw in the npz world frame is 90.3 degrees — a previously unverified deploy-lens claim I can now partially confirm. Composing the synthesis memo.

# First-Principles Audit — Synthesis Verdict

**Project:** G1 Thriller full-body dance — sim2real retrain decision
**Question asked:** "Are we on the right path, or did we make a critical mistake?" (asked before committing GPU-hours to the sim2real retrain)
**Basis:** 8 investigator lenses, adversarially verified (18 confirmed findings, 1 disputed, 1 refuted), plus synthesis-level code checks. Robot untouched; GPU box read-only.
**Date:** 2026-07-05

---

## 1. VERDICT: CRITICAL-MISTAKE-FOUND

**The decisive evidence behind the retrain plan was wrong: the "sim ankle 0 Nm" number never measured the ankle — `cloud/sim_ankle.py` indexed mjlab's actuator-ordered `actuator_force` array with joint-tree indices and actually read the left wrist-roll (its "knee" read the left elbow), so the "0 → 15 Nm ONE sim2real gap, prime suspect latency" conclusion was built on a measurement artifact (confirmed by three independent verifiers, by physics, and by the same-instant qfrc cross-check).** The strategic destination survives — a retrain is still the right vehicle — but for materially different reasons than the plan of record states: correctly measured, the policy is ankle-hungry even in perfect sim (~6–8 Nm mean, transients saturating the 50 Nm clamp), the real ~15 Nm has a pure static PD signature that latency randomization cannot produce or fix, and the plan's #1 item (latency DR 10–40 ms) and its original verification gate (which reused the broken read and would have passed vacuously) were both wrong. The mistake was caught before GPU spend — partially by the main session itself (PROJECT_STATE 2026-07-05 correction) — and this audit additionally found one probable deploy-side obs bug (no yaw re-anchoring of the reference frame) and one unresolved hardware anomaly (apparent knee torque-delivery deficit) that must be closed cheaply before training.

What survives vs. what broke, in one table:

| Original claim (00:15 entry / HANDOVER) | Audit outcome |
|---|---|
| Sim ankle torque ~0 Nm during the dance | **FALSE — artifact.** Read left_wrist_roll. Correct (qfrc_actuator): mean ~6–8 Nm, p95 15–20 Nm, transients to the 50 Nm clamp |
| Real ankle ~15 Nm mean at trained gains + FF | **Roughly real but dirty:** window contaminated by the 2× approach ramp (decontaminated bound 9–31 Nm); tether biases it down; tau_est semantics uncalibrated; the 60–65 Nm "max" is not sim-comparable |
| "0 → 15 Nm is ONE sim2real gap" | **FALSE.** Honest gap ≈ 2× (sim ~6–8 vs real ~15), and most of the real number is required physics + sag statics |
| "Prime suspect: latency + actuator response" | **Wrong mechanism for the walls.** Static signature; stand-hold (constant command, delay-invariant) shows the same sag and ~20 Nm heat. Latency matters only dynamically |
| "No deploy-side patch can close it" | **Overstated.** Posture/CoM/calibration is a real 2–4× thermal lever (bounded below by the policy's own ~6–8 Nm sim floor); plus at least one deploy obs bug is fixable for free |
| "Fix = retrain (5-item DR recipe)" | **Direction survives, recipe re-ranked:** torque penalty + posture is the headline; latency DR demoted to 0–20 ms robustness hygiene |

---

## 2. THE GAP ITSELF

**Is "sim 0 Nm vs real 15 Nm" a real, comparable, replicated gap? No, on all three counts, as constructed.**

- **Not real (sim side):** `sim_ankle.py`'s hasattr fallback fired silently (`applied_torque` does not exist in mjlab), and `actuator_force` is ctrl-ordered (arm 5020 group first, ankles at indices 25–28, `sort_actuators=False`) while the script used joint-tree indices 4/3. Its "ankle 0.0/0.3 Nm" is the left wrist-roll (gravity moment ~0 by axis geometry); its "knee 0.4 Nm" is the left elbow. Independent physics makes the reported numbers impossible: the trained ready pose puts the whole-body CoM +3.21 cm ahead of the ankle-pitch axis (33.34 kg model), requiring ~5.25 Nm **per ankle just to stand**, and ~10.6 Nm per knee; "max 0.3 Nm over a dance" would need the CoP pinned within ~2 mm of the ankle axis for 52 s.
- **Not comparable (real side):** the 60–65 Nm max exceeds the 50 Nm limit that clips every sim torque (and the actuator's rated peak); the ~18-sample window (`sleep 5.2` blind sync) provably overlapped the 2×-gain approach ramp and missed the tail of the policy segment; tau_est semantics for the parallel two-motor ankle (PR mode, motor-space vs joint-space) were never calibrated; the tether biases the mean low. The **mean** is nonetheless roughly trustworthy: kp·err = 28.5 × 0.506 rad = 14.4 Nm matches it, and every decontamination scenario still leaves the policy-segment mean at 9–31 Nm.
- **Not replicated:** every load-bearing quantity was a single run (sim measurement 1×/12 s window/never-deployed checkpoint; real 15 Nm 1× 5 s; thermal 1× 34 s; "onboard ~0 Nm" 0×, assumed).

**The honest gap:** sim ~6–8 Nm mean / p95 15–20 (full motion, deployed checkpoint, `reports/sim_gap_check_a2_1500_full.json`: L 7.9 / R 7.4 Nm, 99.2% survival) vs real ~15 Nm mean — **a ~2× hardware excess on top of an inherent policy cost**, not an infinite anomaly.

**Most probable mechanism, ranked:**

1. **Static posture/CoM excess — the sag (highest confidence).** The trained gains cannot passively hold the pose anywhere: total ankle PD stiffness 57 Nm/rad (171 at 3×) vs gravitational destabilizing stiffness mgh ≈ 202 Nm/rad; a firmware-identical PD sim replica sags through the exact HW signature (ankle −21° → −46/−50°, ~14 Nm) and topples at 1×/2×/3× (even 8× fails). On HW the ankle parked exactly on its −50.0° mechanical stop; stand-hold — a constant command stream on which latency has strictly zero effect — held ~20 Nm continuous, gain-independent, equal to the CoP-at-toe static ceiling (W/2 × 0.12 m ≈ 20 Nm). In sim the policy is the balance controller; on HW it sagged anyway (see #3/#5 for why it may have balanced worse than sim).
2. **Inherent policy/choreography torque floor.** Even perfect sim uses ~6–8 Nm/ankle mean; the pose's static floor is 5.25 Nm/ankle; the correctly computed choreography floor is ~5–7 Nm/ankle mean (the earlier 10–12 Nm/leg floor claim was REFUTED — its p90 79/max 126 Nm frames violate the 41 Nm CoP-at-toe cap and came from planted-foot misclassification on a floating reference). Only a torque penalty and/or posture change moves this component.
3. **Deploy obs-frame corruption (new, probable-major, partially verified at synthesis).** All three obs builders (`deploy_runtime.py:253/292/346`) compute `motion_anchor_ori_b` as raw `R_imu^T · R_ref` with **no yaw alignment** between the npz world frame and the boot-relative IMU frame; I verified the reference's t=0 torso yaw is **90.3°** in the npz frame. Training (RSI spawn at reference, 0.8 rad anchor-ori termination) never saw more than ~46° of this term, so unless the robot happened to boot facing the npz heading, the policy ran far OOD on this input during every ground run (the deploy lens's unverified ONNX experiment measured waist/ankle target corruption comparable to the whole action signal). This cannot explain stand-hold statics (no policy in the loop) but plausibly explains part of the dynamic 2× excess and degraded HW balance — and **no DR item fixes a structural obs bug**. `motion_anchor_pos_b` in the odom mode mixes the same two frames.
4. **Mass/CoM model error.** Real ~35 kg vs 33.34 kg model (+5%, Inspire hands/battery/covers): real, cheap to fix, worth ~1–2 Nm — not the explanation.
5. **Actuator delivery shortfall (knee/hip) — unresolved.** The one weight-bearing settle implies delivered knee torque ~0.2–0.4× of commanded kp·err (58/96 Nm commanded vs 15–30 Nm static demand; no static configuration with full delivery is consistent, tether included), while the ankle demonstrably delivers ~1.0×. Single, provably contaminated datapoint (the pose was not freely balanceable — the tether was load-bearing). If real (per-joint scale or a ~15–25 Nm sustained cap), it changes the actuator-DR ranges; must be measured, not assumed.
6. **Latency — real but small and dynamic-only.** Delayed-target torque at the ankle: 0.37/0.74 Nm |mean|, zero-mean, at 20/40 ms (computed from the actual reference); deploy adds ~5–10 ms (unmeasured) on top of the 20 ms ZOH training already contains. Injected 40 ms in sim: ankle mean 6 → 9.9 Nm, p95 33.6, falls — so it matters for dynamic robustness and the stepping section, and contributes nothing to the thermal/sag walls.

---

## 3. RETRAIN PLAN CORRECTIONS

The authored config (`cloud/sim2real_task.py`) already self-corrected partway (torque penalties in, order-safe qfrc gate). Amendments against the audit, item by item:

**Re-ranked recipe:**

1. **Torque/energy penalty + posture — HEADLINE (was item 3).** Keep `joint_torques_l2 -2e-5` + `ankle_torque_l2 -4e-4` (qfrc-based). Success must be gated on **posture actually shifting**, not just the torque scalar: add a gate on mean sagittal CoM-to-ankle lever (or CoP-to-ankle offset) in stance — target ≤1.5–2 cm from the current +3.2 cm. If the penalty alone doesn't move the CoP rearward, add an explicit stance-CoP shaping term rather than cranking the penalty weight (tracking-quality cost is the known failure mode).
2. **System-ID before DR (new item).** (a) Weigh the robot as-deployed; set the **nominal** model mass to measured (~35 kg: hands +1.1–1.3 kg at wrists, battery), DR around it (the authored base_com ±5 cm, torso mass 0.95–1.15, hand payload 0–0.6 kg/wrist are good). (b) Add **ankle-specific joint zero-offset DR ±0.05–0.1 rad** (BeyondMimic's published table; our ±0.02 encoder bias is 3–5× too narrow for the parallel ankle). (c) Resolve the knee-delivery question (Section 4 test) **before** setting torque-scale ranges — keep the authored effort 0.80–1.00 / pd_gains 0.85–1.15 unless the test says otherwise; do not blanket-widen to 0.3×.
3. **Actuator-response DR — keep, modest.** Authored frictionloss 0–0.4, armature 0.9–1.4 are reasonable (the 4-bar ankle armature is a documented guess). Drop "bandwidth" as a named suspect — sim and firmware run the same PD law (confirmed; the commit-48fb86d "position actuator implicitly holds gravity" narrative is wrong physics and should be corrected in the docs).
4. **Obs DYNAMICS, not wider noise.** Authored obs delay 0–1 steps is not the deploy estimator: model leg-odom's actual filter — first-order lag 30–80 ms + slew limit (0.30 m/s per tick) + episodic stance-break bias ±0.15 m/s on base_lin_vel (or run `LegOdometry` itself in the training obs path). Keep the trained white bands (leg-odom is already 97.8–99% inside ±0.5 m/s).
5. **Latency DR — demoted to hygiene, range corrected.** Train at **0–20 ms** (0–4 physics steps), not 0–40 ms: the static walls are latency-free, 10–40 ms exceeds published practice (BeyondMimic explicitly rejects latency DR and ships a <10 ms loop; established recipes use 0–20 ms or 1-step-on-50%), and over-wide delay DR risks a conservative policy. Keep 40 ms as an **eval** condition only. In parallel, measure the real loop latency once and consider just minimizing it.

**Additions that are not DR:**

6. **Choreography item (required on every path).** The 14–16 s stepping brace is proven choreography-hard with a perfect state estimate — no retrain fixes it. Edit the segment (simplify the step to a weight shift, widen stance, or slow it 10–20%) and trim the big-lean transients that saturate the 50 Nm clamp even in sim. Re-render the reference; train/eval on `thriller_deploy.npz`, full motion.
7. **Deploy-side fixes bundled now (free, no retrain):** (a) **yaw re-anchor at policy start** — capture Δyaw = yaw(imu₀) − yaw(ref₀) and rotate all reference world quantities (aquat, disp) into the IMU frame (also fixes the mixed-frame anchor_pos in odom mode); (b) pelvis-vs-torso anchor: FK from pelvis IMU + waist joints to the torso frame (up to ~30° anchor-ori error during ±25° waist moves is claimed, unverified — quantify offline first); (c) `GROUND_MAX_ACTION` default 6.0 → measured need (10), and re-measure for the retrained policy; (d) drain-to-latest in `read_state` (one line, matches the monitor fix); (e) archive ffmon.py + raw output into `logs/` for provenance.
8. **Training-episode structure:** 10 s episodes with pushes every 1–3 s mean the policy rarely holds long unperturbed stance — add stance-weighted sampling or a fraction of long (30–50 s) episodes so the thermal-relevant standing behavior is actually trained.

**Eval gates (amend `cloud/sim_gap_check.py`):** current gate "ankle mean ≤5 Nm worst-condition" sits at-or-below the true floor (~5–7 Nm/ankle) and may be unpassable without tracking sacrifice. Recommend: survival ≥99% nominal / ≥95% worst; ankle mean ≤6 Nm nominal, ≤8 Nm worst-condition; p95 ≤15 nominal / ≤20 worst; mpkpe ≤0.25 nominal; **plus** the posture gate (CoM lever ≤2 cm), **plus** a thermal projection gate using RMS not mean: predicted rate 22.5·(τ_RMS/20)² ≤ 8 °C/min ⇒ τ_RMS ≤ ~12 Nm; report per-section stats (0–10 s, 13–17 s, worst 5 s window).

---

## 4. CHEAPEST DECISIVE PRE-GPU EXPERIMENTS (ordered)

1. **Lock the corrected sim baseline (in flight, minutes).** Full-motion qfrc_actuator eval on the deployed model_1500 across the 7 conditions. Discriminates: inherent floor vs artifact; sets the gate numbers. (Already running — just consume it.)
2. **Offline obs-frame sensitivity test (local, hours).** Feed the ONNX policy sim-perfect obs vs deploy-built obs with the anchor-ori yaw offset swept 0–180° (and the pelvis-vs-torso approximation toggled); measure action divergence and rollout survival in the local MuJoCo harness. Discriminates: whether the missing yaw re-anchor / torso-anchor approximation materially corrupted every HW run — i.e., how much of the "2× hardware excess" is a free deploy fix rather than a retrain problem.
3. **Statics system-ID on existing telemetry (local, hours).** Fit tau_est vs kp·err per joint across the recorded runs (ankle already fits ~1.0×; bound the knee ratio with tether scenarios). Discriminates: actuator delivery shortfall vs contamination; directly sets (or removes) the torque-scale DR extension.
4. **DDS obs-staleness measurement (robot powered, read-only, no commands).** Read rt/lowstate at 50 Hz for 60 s; compare msg tick vs wall clock. Discriminates: whether obs latency exists at all beyond ~2 ms; centers the latency DR range on data.
5. **Stage-0 onboard-stand capture (robot in normal 'ai' standby, read-only).** Log ankle/knee/hip tau_est + temperatures for 2–3 min while the onboard controller stands. Discriminates: tau_est semantics (H4) against a known stance; the assumed "onboard ~0 Nm" baseline (prediction if healthy: 3–6 Nm, <2 °C/min); the reference posture the retrain should aim for.
6. **One instrumented tethered rerun (single run, class already performed).** Per-tick npz logging (tau_est, q, stage-id, wall clock), slack-vs-taut tether A/B, and a ±3° commanded ankle-bias sweep during stand. Discriminates: decontaminates the 15 Nm; maps posture → torque → °C/min directly; tests whether a deploy-side posture trim alone reaches the ≤8 Nm/leg sustainable band.

Items 1–4 need no robot motion at all; 1–3 need no robot. Do all six before training: total ~1–2 days, and items 2–3 can still change the retrain config.

---

## 5. STRATEGY: retrain vs arm-over-onboard vs hybrid

**Recommendation: HYBRID — lock arm-over-onboard as the bookable show baseline now; run the corrected retrain as the premium full-body act.**

Honest probabilities (subjective, stated so they can be argued):

- **Arm-over-onboard pivot** (arms/torso choreography over the vendor balance controller): P(show-ready within 1–2 robot sessions) ≈ **0.85**. It rides a proven balance stack; thermal presumably benign (confirm with the Stage-0 capture). Lower artistic ceiling — no leg choreography. This path silently vanished from the plan of record and should be reinstated as the revenue floor.
- **Corrected retrain, full-body:** P(passes the corrected sim gates) ≈ 0.75. P(first retrained policy stands + dances on HW at trained gains within the thermal budget | gates passed AND deploy fixes + pre-GPU checks done) ≈ **0.55**; without the pre-GPU program ≈ 0.35 (unresolved knee delivery, tau semantics, obs-frame bug would still be live). P(14–16 s section survives without a choreography edit) ≈ 0.15 — edit it regardless (with edit: ≈ 0.8). Net full-body show-bar within ~2 training iterations: ≈ **0.5–0.65** with the pre-GPU program.
- **Retrain as sole path (no hybrid):** expected calendar risk to the paid show is dominated by single-sample hardware unknowns; not recommended.

Cost logic: the retrain is cheap in money (~$1–2/run) but expensive in robot-session time and show risk; the pivot is cheap in both and de-risks the booking. They share almost all infrastructure, so the hybrid costs little extra.

---

## 6. WEAKEST EVIDENCE STILL STANDING (re-measure on next robot contact)

1. **tau_est semantics at the parallel ankle (H4).** The mean fits joint-space statics, but 60–65 Nm "peaks" exceed the joint's rated/modeled 50 Nm; motor-space vs joint-space never confirmed. → Stage-0 capture + kp·err identity check.
2. **The 15 Nm figure itself.** Single 5-s run, window provably contaminated by the 2× approach ramp (policy-segment mean only bounded 9–31 Nm), tether biasing down. → instrumented rerun with stage tags.
3. **Knee delivery deficit (~0.2–0.4× commanded).** Single contaminated datapoint that no static scenario fully explains away; if real it reshapes actuator DR; if not, drop it. → hang/bench torque-step test.
4. **"Onboard stands at ~0 Nm ankle."** Assumed, never measured; geometry permits ~0 but ~5 Nm is equally consistent with "never overheats" (≤8 Nm/leg sustains >10 min). → Stage-0 capture.
5. **"~20 Nm ankle continuous."** maxTau was the max over all 12 leg motors; the argmax was never printed — ankle attribution inferred from temperature. → per-joint logging in the same capture.
6. **Deploy obs-frame claims (yaw re-anchor, pelvis-as-torso).** Code-verified missing at synthesis (and ref yaw = 90.3° at t=0), but the realized error during past runs depends on never-logged boot headings, and the "corrupts actions as much as the whole action signal" magnitude comes from an unverified single ONNX experiment. → experiment 2, then fix unconditionally.
7. **Thermal model.** One 34-s datapoint at 1.5×; τ² scaling assumed; no dissipation term; the source's own 22.5 °C/min understates its endpoints (26.5). Use RMS torque, validate the slope on the next run.

**Process note (why the reasoning broke):** four "root cause found" declarations in one day, capped by a "don't re-litigate" HANDOVER written ~1 h after an unverified measurement; the sim number was accepted without a cross-check because it matched the narrative, while the disconfirming physics (a 33 kg robot cannot dance at 0.3 Nm max ankle torque) was one static computation away. The qfrc cross-check that eventually caught it takes minutes. Adopt the rule: **no DECISIVE label without an independent cross-check or a replication**, and commit raw measurement scripts + outputs (ffmon.py et al.) into `logs/` so load-bearing numbers have durable provenance.

---

### Appendix: corrected load-bearing numbers (carry these forward)

| Quantity | Value | Source/status |
|---|---|---|
| Trained gains (ankle/knee/hip kp) | 28.5 / 99.1 / 40.2 | verified = deployed |
| Ankle PD stiffness vs gravity stiffness | 57 vs ~202 Nm/rad | pure PD statically unstable at any tested gain; policy is the balance controller |
| Static floor at trained ready pose | 5.25 Nm/ankle (CoM +3.21 cm) | reproduced 5× |
| Choreography floor (correct) | ~5–7 Nm/ankle mean; quasi-static cap ~41 Nm total | 10–12 Nm/leg claim refuted |
| Corrected sim ankle (deployed policy, nominal) | mean ~6–8 Nm, p95 15–20, transients to 50 Nm clamp | qfrc_actuator, full motion |
| Real ankle (trained gains + FF run) | L 14.2 / R 16.3 Nm mean; segment-decontaminated bound 9–31 Nm; max 60–65 not sim-comparable | single run, dirty window |
| Honest sim2real excess | ~2× mean | not 0 → 15 |
| Thermal | 22.5–26.5 °C/min @ ~20 Nm; τ²-scaled: ~12.7 @15 Nm (≈3.4 min to fault), ~3.6 @8 Nm (≈12 min), ~1.4 @5.3 Nm | single datapoint; gate on RMS ≤ ~12 Nm |
| Latency contribution to statics | ~0 (stand-hold is delay-invariant; 0.4–0.8 Nm zero-mean on the action stream) | latency DR = dynamic robustness only (40 ms injected: 6 → 9.9 Nm + falls) |
| Reference t=0 yaw in npz frame | 90.3°; no yaw re-anchor in any obs builder | fix deploy-side before any HW test |