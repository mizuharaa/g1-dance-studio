# Long Dance Plan — full 2–3 min Thriller (with jumps) through the APP pipeline

**Owner:** Lane D planning doc. **Status:** honest readiness assessment, 2026-07-07.
**Scope:** running a FULL 2–3 minute Thriller — including advanced moves like jumps —
through the *desktop app* pipeline (upload video → extract → retarget → train → exam →
show-ready candidate), NOT the hand-driven CLI flow. This is a plan + readiness memo, not
a build. It states plainly what EXISTS today versus what is BLOCKED, and gives a phased
path with the gating dependency for each phase.

> **One-line honest summary.** The *grounded, in-place* long dance is de-risked on the
> training side and the app pipeline is wired for it end-to-end — but it is BLOCKED on two
> things the user must supply (a 2–3 min in-place source video, and a recreated GPU box),
> and 2–3 min *hardware* endurance is unproven. **Jumps are a separate R&D track, not a
> pipeline feature** — the closest thing we attempted (a backflip) FAILED as infeasible at
> the G1's true actuator limits, and any aerial is a hardware-risk decision, not a knob.

---

## 1. Where we actually are today (the honest baseline)

- **Today's hardware-validated show is a ~50 s cut, not 2–3 min.** The untethered
  performance proven on hardware is the `thriller_standtail_candidate` motion: v3e sharp
  dance + a 2.5 s return-to-standing tail + 1.5 s hold = **2709 frames / 54.2 s** (danced
  span ~50 s). It runs fully untethered, on-beat with music, ending standing
  (PROJECT_STATE 2026-07-07 milestones). A 2–3 min dance is therefore a **NEW, longer
  training run**, not a re-cut of what exists.

- **The app pipeline is wired video → sim-verified, and it already accepts long input.**
  `pipeline/stages/local_motion.py` (retarget/vet/preview/prep, laptop) +
  `pipeline/stages/cloud_motion.py` (extract/train/verify/export, GPU box) reproduce the
  exact recipe that made Thriller show-ready. The video intake gate
  (`pipeline/video_probe.py`) already allows **15 s … 4 min** (`MAX_SECONDS = 240`), so a
  2–3 min clip is accepted with no code change. The per-dance knob file (`dance.yaml` in
  the job dir, parsed by `cloud_motion.load_params`) is the intended place to raise
  `iterations` and pass recipe deltas for a longer horizon.

- **BUT the GPU box is DELETED.** `PROJECT_STATE 2026-07-07`: "GREENNODE BOX DELETED (user
  console click; verified: SSH refused)." The app's `train` and `verify` stages call
  `_require_cloud()` and raise `StageBlocked` when no box is configured. So today the app
  runs a long video only as far as **retarget → vet → preview** (all laptop-local); the
  moment it needs the GPU it honestly blocks. Recreating the box is a **user console
  action** (no API — `docs/BOX_RECREATE_RUNBOOK.md`, ~10 min fast path if the
  `g1dance-data` Network Volume survived, ~45–60 min full re-provision otherwise).

- **The 2 m dance area is the choreography constraint, and it caps clip length for
  *traveling* mocap.** `pipeline/find_window.py` `longest_window()` keeps only the longest
  contiguous window whose root-XY minimal-enclosing-circle radius ≤ 1.5 m. Stock traveling
  mocap hits that wall at **~62 s** (PROJECT_STATE 2026-07-04/05: longest clean in-area
  window 62 s, excursion 1.47 m of the 1.5 m limit). Important nuance: **the vet gate does
  not cap *length* — it caps *excursion*.** An in-place 2–3 min piece stays inside the
  circle for its whole duration, so `longest_window` keeps the whole thing. **In-place
  choreography is precisely what unlocks 2–3 min.**

---

## 2. EXISTS vs BLOCKED (the ground truth)

| Capability | Status | Evidence / where |
|---|---|---|
| App accepts a 2–3 min video | **EXISTS** | `video_probe.py` `MAX_SECONDS=240` |
| Video → SMPL → G1 retarget → vet → preview (laptop) | **EXISTS** | `local_motion.RetargetStage`, `retarget_gvhmr.py`, `vet_motion.py` |
| Vet gate keeps a *full* in-place window (no length cap) | **EXISTS** | `find_window.longest_window` (excursion-bounded, not time-bounded) |
| Per-dance knobs (iterations, recipe deltas, explicit window) | **EXISTS** | `cloud_motion.load_params` + `dance.yaml`; `docs/NEW_DANCE_PLAYBOOK.md §4` |
| Music capture + window-aligned attach for a long clip | **EXISTS** | `ExtractStage` audio capture, `ExportStage._prepare_windowed_music` |
| Stand-to-stand ending for a trained dance | **EXISTS** | `TrainStage` rebuilds deploy CSV with `deploy_ramp stand_end=True` |
| Longer-horizon *training* recipe works | **EXISTS (sim-validated to 67 s)** | PROJECT_STATE 2026-07-05: dance2-long 67 s, 100% clean, joint err 0.099 rad (tighter than 49 s Thriller's 0.117) |
| GPU box for extract/train/exam | **BLOCKED — deleted** | PROJECT_STATE 2026-07-07; recreate via `docs/BOX_RECREATE_RUNBOOK.md` (user console) |
| A 2–3 min **in-place** source video | **BLOCKED — not filmed** | product finding: 2–3 min pieces must be choreographed to stay roughly in place |
| 2–3 min training run actually done | **BLOCKED — never trained** | max trained = 67 s; 2–3 min is extrapolation, not evidence |
| 2–3 min **hardware** endurance (thermal/battery) | **BLOCKED — unproven** | see §4 |
| Signed show-ready standing-end Thriller | **BLOCKED — needs box re-exam** | standtail motion never through the 3× held-out exam (box deleted) |
| **Jumps / aerial skills** | **BLOCKED — R&D, infeasible as attempted** | see §5; `docs/DYNAMIC_SKILLS.md` |

---

## 3. The gating dependencies (what must be true, and who owns each)

1. **A 2–3 min in-place source video** — *user*. Single continuous shot, tripod, one
   person, full body in frame, choreographed to stay roughly in place (small footprint
   inside the 2 m circle). Traveling choreography will be silently trimmed by the vet
   window to ~62 s; a 2–3 min result *requires* in-place motion. VFR/odd fps is fine (the
   app re-encodes to 30 fps). This is a filming/choreography task, not a software task.

2. **The GPU box, recreated** — *user console + us*. `docs/BOX_RECREATE_RUNBOOK.md`. No
   API exists (GreenNode notebook lifecycle is console-only), so the user must click
   create; we then re-point `.secrets/cloud.json`, re-run the idempotent provisioners, and
   smoke-test. Everything the app needs (extract, csv_to_npz, train, exams) runs there.

3. **Per-dance `dance.yaml`** — *us, at job time*. A 2–3 min motion is ~2–3× the frames of
   the 54 s standtail; the promoted defaults (`iterations: 5000`, the Sim2Real task + s2r-b
   delta) are tuned for ~50 s. The long-dance validation that de-risked longer horizons
   used a **larger adaptive sampling kernel and a higher iteration cap** (the "single-clip +
   adaptive-kernel" recipe, PROJECT_STATE 2026-07-05). Concretely, expect to raise
   `iterations` and pass the adaptive-kernel / recipe deltas through `extra_train_args`,
   plus optionally pin `window_start_s`/`window_end_s` if the vet window needs a manual cut.
   **No specific numbers are asserted here** — the 67 s validation is the only data point;
   2–3 min needs its own tuning pass, treated as 1–3 attempts like every prior dance.

---

## 4. Endurance at 2–3 min — de-risked in sim, UNPROVEN on hardware

- **Training side: de-risked.** The 67 s long-dance converged and evaluated 100% clean in
  sim, with *tighter* joint tracking (0.099 rad) than the 49 s Thriller (0.117 rad) — longer
  clips did **not** degrade tracking with the long-horizon recipe (PROJECT_STATE
  2026-07-05). This is genuine evidence that the *learning* scales past 60 s. It is **not**
  evidence that it scales to 180 s — 67 s is the longest we have actually trained/evaluated.
  Treat 2–3 min as "supported by trend, not yet demonstrated."

- **Hardware side: unproven, and specifically these three envelopes must be re-measured at
  the true 2–3 min length before any untethered long run:**
  - **Ankle thermal.** Ankle thermal was the *old* wall; at ~50 s the ankles run cool
    (telemetry temps 55–57 °C across today's free runs). That clearance was measured at
    ~50 s, not 2–3 min. Motor/driver temperature is cumulative — **re-check at full length.**
  - **Battery / endurance.** No 2–3 min continuous-dance battery draw has been measured.
  - **Balance-envelope drift over time.** Today's free runs hold ~14 ° peak torso tilt at
    the sharp arm accents (56 ° margin to the 70 ° fall trigger) for ~50 s; whether that
    envelope holds for 3–4× as long, with thermal derating creeping in, is untested.
  - Mitigation already in the deploy spine: fall detector @0.35/3-tick, start-upright
    guard, exit-stand handoff, `GROUND_LEG_KP_SCALE=1.5` sagittal boost — all validated at
    ~50 s. They carry over, but their behavior over a 2–3 min run is unmeasured.

---

## 5. JUMPS — the blunt version: R&D, not a pipeline feature

**A jump is an AERIAL skill. The only aerial we have attempted is a backflip, and it
FAILED as infeasible at the G1's true actuator limits.** This is not a pessimistic guess —
it is two cross-checked sim results (`docs/DYNAMIC_SKILLS.md §6`):

- **Attempt 1** reward-hacked (skipped the flip; landed 0/64, rotation 0.000 rev).
- **Attempt 2**, with the skip loophole closed, *genuinely attempted* the launch: **knee
  torque saturated at its exact rating (139/139 Nm), ankle 50/50, waist 50/50** — and
  achieved only **~2% of the required rotation (0.165 rad of 7.34 rad)** before dying at
  the apex. Independent corroboration from intake: the reference's own peak joint
  velocities (40.6 rad/s, 6 joints over the 20–37 rad/s ratings) already said the maneuver
  lives *outside* the hardware envelope. Verdict: with tracking-RL and honest limits, the
  G1 cannot produce this launch impulse.

**Why a jump is not simply "an easier flip you can drop into the show":**

- **A jump is less extreme than a flip but is the SAME CLASS of risk.** Both have a flight
  phase, and in flight every one of our proven safety assumptions breaks:
  - The proven state estimator is **leg odometry — stance-only.** In flight `base_lin_vel`
    (a live policy input) is garbage. Our whole grounded pipeline assumes stance contact.
  - **No torque-cut hardware e-stop** on this G1 — the remote's B-damping is the only stop,
    and it is useless mid-air (damping a flying robot guarantees a crumple landing). This is
    an established robot-day fact.
  - **Landing loads** compress through ankle/knee at full effort at touchdown; the acro
    training profile deliberately *removes* the torque penalties that keep the dance policy
    inside the ankles' comfortable envelope (`cloud/dynamic_skills_task.py §2`).
  - **Sim2real for aerials is unproven** in our chain. Every hardware transfer we have
    validated (Thriller sim gate reproduced on hardware) is *grounded* choreography.

- **Jumps need their own machinery, which exists only in sim and is walled off from the
  show:**
  - The **dynamic-skills task profile** (`cloud/dynamic_skills_task.py`, task
    `Mjlab-Tracking-Flat-Unitree-G1-Acro`) — flight-grace terminations that suppress the
    stock deviation checks during the airborne window, full effort limits, no push events.
    It is **SIM-ONLY by design** and its own docstring documents that acro motions **never**
    enter `data/dances/`, the show library, set-lists, or the deploy bundle machinery.
  - Its own intake/exit gates: `tools/check_acro_reference.py` (FK feasibility: rotation
    present, feet trajectories, joint limits, velocity-vs-motor-ratings) *before* training,
    and `cloud/acro_eval.py` (landing-success + peak torque/velocity + landing-impact audit)
    *after*. The show `vet_motion.py` is deliberately bypassed (a jump legitimately violates
    its upright/pelvis-height assumptions), which is exactly why a jump can't ride the show
    promotion path.

- **A jump therefore requires, before any hardware talk (this is the `DYNAMIC_SKILLS.md §5`
  decision gate, applied to a jump):**
  1. A **G1-FEASIBLE jump reference** — authored or sourced at an amplitude the actuators
     can actually produce (the backflip lesson: human-mocap amplitude saturates the motors).
     This is a choreography/R&D investment, not an attempt-N knob.
  2. A **feasibility analysis** — is the jump's launch impulse + landing load inside
     motor/torque limits with margin? Run `check_acro_reference.py` on the reference, then
     `acro_eval.py` on a trained candidate.
  3. **Likely a separate policy or a hybrid** — the tracking policy is trained for *grounded
     balance* with push robustness; the acro profile trains for flight with no push
     robustness. These are different objectives; a jump inside a dance likely means a
     dedicated aerial policy (or a phase-switched hybrid), not the grounded show policy.
  4. Its **own validation staircase** — a **sim landing-success gate ≥99%** under DR + obs
     noise (`acro_eval.py`, held-out seeds), a torque/velocity margin audit, a
     push-robustness pass (absent by design), a flight-valid state estimator, an impact-load
     engineering review, and a containment plan that does **not** rely on the tether
     mid-flight — *then* the user makes the hardware call in person.

**Bottom line for jumps:** they are **not drop-in into the standtail show.** They are a
separate R&D program and a hardware-risk decision of the same character as the backflip —
which the current sim evidence closes as "no" until a G1-feasible reference exists.

---

## 6. Concrete phased path (each phase names its gating dependency)

### Phase A — Choreograph + film the in-place 2–3 min piece  · gate: USER (video)
Choreograph Thriller (or the chosen routine) to stay roughly in place inside the 2 m
circle for the full 2–3 min. Film one continuous tripod shot, one dancer, full body,
15 s–4 min. Deliverable: a source video. **Nothing downstream can start without this** —
and if the choreography travels, the vet window silently trims it back toward ~62 s.

### Phase B — Recreate the GPU box  · gate: USER console + us
Follow `docs/BOX_RECREATE_RUNBOOK.md`: create the notebook (console), re-point
`.secrets/cloud.json`, re-run `00_bootstrap.sh` / `10_gvhmr.sh` / `20_training.sh mjlab`,
smoke-test. Fast path ~10 min if the `g1dance-data` volume survived. This unblocks every
GPU stage of the app.

### Phase C — Run the long dance through the app (grounded, no jumps)  · gate: A + B + dance.yaml
Upload the video in Studio. The app runs extract → retarget → vet → preview automatically.
**Human gate 1: watch the preview, confirm the vet table, click Approve training.** Drop a
`dance.yaml` first to raise `iterations` and pass the long-horizon recipe deltas
(adaptive-kernel etc.) via `extra_train_args`; optionally pin the window. Then the app runs
csv_to_npz → `train_sim2real` → export → sim-gap gate v3 → 3× held-out exams → signed
verdicts → **sim-verified candidate**. Expect 1–3 training attempts (the normal budget) to
tune the longer horizon. **Human gate 2: promote to show-ready** (guarded by 3 clean signed
exams). Music attaches automatically from the video's soundtrack, window-aligned. This is
the phase the pipeline is actually built for; its only new risk vs today is *length*, which
§4 says is trend-supported but not yet demonstrated at 2–3 min.

### Phase D — Endurance validation at true length  · gate: a converged Phase-C policy + robot day
First in sim: confirm the full 2–3 min evaluates clean (the app's gap gate + exams already
score the full motion). Then on hardware, staged per the tether progression: re-measure
**ankle thermal, battery draw, and balance-envelope drift** across the *full* 2–3 min run
(§4). Only after these hold does an untethered 2–3 min run become evidence-supported. Robot
day is a human-present, damping-remote-in-hand gate — the app never contacts the robot.

### Phase E (SEPARATE TRACK, parallelizable but independent) — Jumps R&D  · gate: G1-feasible reference + full acro staircase
Do **not** couple this to Phases C/D. Per §5: source/author a G1-feasible jump reference →
`check_acro_reference.py` feasibility → train on the acro task → `acro_eval.py`
landing-success ≥99% + torque/velocity margin audit → push-robustness pass → flight-valid
estimator → impact-load review → tethered staircase that never relies on the tether
mid-flight → **user's in-person hardware-risk decision.** If any gate says the reference is
beyond the actuator envelope (as the backflip did), the jump stays a sim artifact. A jump
only ever enters a show as a *separate* validated skill, never by riding the grounded show
promotion path (which is walled off from acro by design).

---

## 7. What this plan does NOT claim

- It does **not** claim a 2–3 min dance has been trained or run — max trained is 67 s (sim).
- It does **not** claim 2–3 min hardware endurance — thermal/battery/balance-over-time at
  full length are unmeasured.
- It does **not** claim jumps are feasible on this G1 — the one aerial we tried failed at
  the actuator limits, and a jump is the same class of risk, gated separately.
- It invents **no** numbers: every figure here traces to PROJECT_STATE, `logs/jobs.md`,
  `docs/DYNAMIC_SKILLS.md`, or the pipeline source cited inline.

**Net:** the grounded, in-place 2–3 min dance is a realistic *near-term* deliverable once
the user supplies the in-place video and recreates the box — the training side is
de-risked and the app is wired for it, with endurance re-validation owed on hardware.
**Jumps are a genuine, separate, hardware-risk R&D problem, not a longer-dance feature.**
