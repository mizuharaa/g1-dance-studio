# AGENT B — Motion Quality: fix the twitch/glitch on fast moves

**Owner: Claude-orchestrated agent.** Runs entirely on this Windows repo clone (CPU only).

## Symptom (user report, 2026-07-10)

In the 3D preview the robot **twitches / body parts glitch for a sudden moment** during fast
hand–body coordination moves. The preview plays the *actual retargeted motion data*, so these
are real spikes in the CSV/npz the policy is trained to track — on hardware they become jerky
commands and a balance risk. This is a data-quality bug in the video→motion front-end, not a
renderer bug.

## Likely root cause (verified by code audit 2026-07-10 — confirm with measurements)

The pipeline has **no temporal filtering**:
- GVHMR estimates SMPL pose **per-frame** — fast motion + motion blur ⇒ frame-to-frame jitter
  and occasional outlier frames (limb flips).
- `pipeline/retarget_gvhmr.py` passes straight to GMR (optional velocity limit only).
- `pipeline/prep_motion.py` `_clamp_joint_velocities` caps per-frame deltas *after the fact* —
  an outlier frame gets clamped into a multi-frame drag then snaps back: itself a glitch source.

## Tasks (measure → fix → verify, per CLAUDE.md discipline)

1. **Quantify first.** New `tools/motion_quality.py`: per-joint velocity/accel/jerk profiles
   over any motion CSV (`data/*.csv`, `data/dances/*/`); report spike frames
   (e.g. |accel| > N·MAD), worst joints, spike timestamps. Run on the Thriller deploy CSV and
   at least 2 others; **commit script + raw outputs** to `data/telemetry/motion_quality_20260710/`.
   Confirm spikes cluster at the fast-move timestamps the user saw glitch.
2. **Fix in the pipeline, pre-retarget and post-retarget:**
   - Outlier-frame rejection + interpolation on the joint-angle track (median/hampel filter).
   - Temporal smoothing: One-Euro or Savitzky–Golay (scipy is available) on joint angles —
     tune so sharp choreography stays sharp (report before/after jerk AND a tracking-fidelity
     delta vs the raw motion; don't blur the dance).
   - Root/quaternion track: slerp-aware smoothing (no naive per-component filtering on quats).
   - Wire it into `prep_motion.py` as a stage BEFORE the velocity clamp; velocity clamp becomes
     the last-resort guard, and its `vel_clamped_frames` count should drop to ~0 on clean input.
3. **Add a motion-quality gate metric** to `pipeline/vet_motion.py` (max jerk, spike count) so
   glitchy motions are flagged at vet time, with thresholds derived from your measurements.
4. **Verify visually.** Render before/after previews. `tools/render_deploy_sim.py` uses EGL
   (Linux) — on Windows try mujoco's default GL first; if rendering fails, fall back to
   per-joint trajectory plots (matplotlib) at the spike frames. Commit the evidence.
5. **Tests.** Small `test_motion_quality.py`: synthetic motion + injected spike ⇒ detected and
   removed; clean sharp motion ⇒ untouched within tolerance.

## Out of scope
- Retraining (cloud/GPU — original laptop). Note in `PROJECT_STATE.md` that existing trained
  policies used UNFILTERED motion; the filter applies to future extractions.
- The drag-drop UI itself (Lane C). Your work is the data path behind it.
- `deploy_runtime.py`, `cloud/`, `ui/`.

## Acceptance
- Committed before/after metrics proving spike reduction without dulling choreography.
- Filter wired into the extract→retarget→prep flow + vet gate metric + tests green.
- `PROJECT_STATE.md` decision-log entry with the numbers.

---

## Phase 2 — FEASIBILITY (new, 2026-07-10) — the robot skips moves it physically can't do
De-glitch (Phase 1) is merged. But the tester saw the robot do only ~60–70 % and SKIP moves:
the reference asks for joint speeds past the G1 motor-class limit (vet advisory already shows
~30 % of frames over ~3π rad/s), so the policy washes them out. Make the reference something
the robot CAN do:
1. **Feasibility retime** — where peak joint velocity exceeds the motor limit, time-warp just
   that segment (slow only the impossible bits) so the beat is kept where possible but no joint
   is commanded past its limit. Report per-segment slowdown + which joints drove it.
2. **Unitree High-Motion warm-tips as authoring constraints** (from the G1 SDK dev guide):
   bring the **knee toward straight** where possible, **reduce stride frequency**, keep **feet
   closer together**, and **avoid dead-still** frames — apply as soft constraints in the retarget
   / a post-retarget pass so motions read natural AND stay in the robot's envelope.
3. Feed the feasible motion to Lane E (retrain) and Lane D (sandbox) — validate the robot's
   achieved-vs-reference fraction goes UP.

Acceptance: a `feasibility` report per motion (frames retimed, joints clamped, achievable
fraction) + the vet gate flags an infeasible motion + tests. Coordinate with Lane D's sandbox
(that's where "achievable fraction" is measured honestly).

### Phase-2 STATUS (2026-07-10): feasibility ANALYSIS delivered; retime deprioritised
tools/motion_feasibility.py ships (per-joint vel vs motor limit, headroom, feasible/comfortable
flags; feeds the vet gate + future RAW dances, which ARE infeasible pre-clamp). But the retime
does NOT fix the Thriller 60-70% — proven the motion is feasible AND slowing it doesn't help
(data/telemetry/feasibility_20260710/). The warm-tips (knee/feet/stride) remain a future authoring
option, but the primary fix for fidelity is Lane E, not this lane. Lane E is unblocked.
