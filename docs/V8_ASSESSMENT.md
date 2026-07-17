# v8 revamp — post-training assessment (2026-07-17)

> Written after attempt 5 (train-thriller_v8s2r-0716) finished training. Companion to
> `docs/PROJECT_STATE.md` (decision log) and `experiments/REGISTRY.md` (numbers).
> Gate numbers in §2 come from the re-verify run (`cloud/verify_v8_rerun.sh`).

## 1. What the v8 revamp changed (and why it should matter)

The revamp attacked the two root causes that survived four attempts (v5–v7), plus the
trust problems around them. Everything below is committed and in the trained recipe:

| Change | Root cause it fixes | Evidence |
|---|---|---|
| **1.8× slowed, per-frame-grounded motion** (`thriller_grounded_repaired_1p8x`) | The reference demanded **173.6 Nm peak ankle torque** vs ~40 Nm real capability — physically impossible by 3–4×; no reward weight can fix an impossible target (why v5/v6/v7 all plateaued at 86–92% survival with falls at the same two beats) | Agent B inverse-dynamics pass; sweep in `experiments/motion_feasibility/` |
| **Foot-grounding fix** (per-frame contact grounding) | Reference floated the support foot ~0.10 m in **78% of frames** (retarget vertical drift ~163 mm was "corrected" by a single global z-offset) | float 78.63%→0.00%, no penetration, root height preserved |
| **Estimator-free actor obs**: dropped `base_lin_vel` + `motion_anchor_pos_b` from the actor (critic keeps them — asymmetric teacher-student), 154-dim/frame × **5-frame history** = 770-dim | We fed **critic-only privileged terms into the actor**, then built a leg-odometry estimator to fake them on hardware — the "state-estimation hole" AND a structural reason the gate read optimistic (sim gets those terms free & perfect) | Agent 0 upstream audit: Unitree's first-party task does exactly this (`No-State-Estimation` variant) |
| **Hip-strategy actuation deltas** (candidate A): ankle soft-barrier >35 Nm replaces global L2, per-channel ankle action-rate penalty, waist-track slack ×0.5 at the two hard beats, ankle effort clamp 50→**40 Nm** (velocity-derated envelope) | Ankles saturated at the 50 Nm sim clamp exactly at the fall beats; real ankle has *less* than 50 Nm at speed (Isaac's torque-speed curve vs mjlab's flat clamp) — sim was optimistic exactly where we fell | Agent D design memo `experiments/actuation_design_v8.md` |
| **Faithful laptop preview** (official G1 MJCF + mjlab armatures) | Menagerie-model sandbox under-represented policies (v7 read 7% amplitude on menagerie vs 96–118% + fall at 20.7 s on the faithful model) — the preview now AGREES with the cloud gate | `experiments/g1_model_reconciliation.md` |
| **UI truthing**: per-policy sim preview (sha of actual policy.onnx), same-scene reference-ghost overlay, landmark overlay on source video, auto-publish on pull | "Video not displaying after retrain" + no honest way to SEE the gap | Agent C, committed + Playwright-verified |
| **5 deploy-side safety guards** (foot-contact-before-run, action clamps + rate limits, estimator-validity, independent damping watchdog, never-run-suspended) | The two tethered fall incidents | 30 tests, staged for next robot-day |

**Training health (new information, from the stage-3 tfevents):** unlike v7 — whose
final checkpoint had collapsed to 3% survival — v8 shows **no late-training divergence**:
reward and tracking error are flat from iter 8k through 12k (error_body_pos ≈ 0.10 m).
The +5k-iteration stage 3 that actively hurt v7 did not hurt v8. In-training episode
lengths look short (~5 s) but that is an artifact of the tight 0.4 m drift-termination
band plus mjlab's adaptive sampler parking ~90% of episodes on the hardest motion bin —
not a gate verdict.

## 2. Gate results (re-verify run)

<!-- FILLED AFTER verify_v8_rerun.sh COMPLETES -->
_pending — see `exports/train-thriller_v8s2r-0716/gap.json` + `calibration_anchor_gap.json`_

## 3. What was BROKEN when "training finished" (found + fixed 2026-07-17)

Training completed, but **nothing downstream of it worked**:

1. **The verify chain gated the wrong task.** `train_v8_curriculum.sh` never passed
   `--task`, so checkpoint screening, gap_check, and heldout all instantiated the stock
   160-dim task against the 770-dim v8 checkpoint. Every screen errored; the picker
   fell back to the **last** checkpoint blind (the exact v7 failure mode it was built to
   prevent); no gap.json existed. Fixed: `--task`/`--task-module` plumbed through all
   three eval scripts (commit 085ab43).
2. **The exported policy.onnx never matched the picked winner.** mjlab exports ONNX
   once, at training end; `export_policy.py` just copies it. Any time the picker chose
   a non-final checkpoint, the shipped ONNX was silently a different network than the
   gated one — **v7's staged policy.onnx was actually the collapsed iter-11997 net,
   not the gated iter-10000 winner** (never deployed, so no harm, but it would have
   been). Fixed: `cloud/export_ckpt_onnx.py` re-exports the picked checkpoint through
   the runner's own exporter.
3. **Gate calibration was impossible** — the ~70%-IRL anchor policy's training
   checkpoint no longer exists (only its deploy ONNX). Fixed: `sim_gap_check.py --onnx`
   runs a deploy-contract onnxruntime rollout, so old promoted policies can be scored
   by the current gate. The Agent A calibration is part of the re-verify run.

## 4. Needs attention (ordered)

1. **Robot is DOWN** — burnt DC-DC converter on the power board (2026-07-15). Deploy is
   blocked on Unitree support regardless of how good v8 is. This is the critical path
   to any hardware validation.
2. **Deploy wave for the new obs contract** (only after v8 signs): `deploy_runtime.py`
   must maintain a rolling 5-frame (770-dim) obs history buffer per the DEPLOY CONTRACT
   block in `cloud/sim2real_task_v8.py`; then DELETE the dead odometry path
   (`pipeline/leg_odometry.py`, `build_obs_odom`, obs-delay DR on those terms). Do not
   touch deploy code before the policy signs.
3. **Gate trust is now measurable but the mapping is one point.** The calibration run
   ties gate% ↔ the single real datapoint (~70% IRL). One anchor is a bar, not a curve —
   treat "calibrated" estimates as rough until a second hardware datapoint exists
   (first post-repair tethered run should be logged for exactly this).
4. **1.8× slowdown is a bet.** Feasible ankle-only bound was 2.5×; 1.8× works only if
   the policy actually learned hip-strategy. If the gate shows the two beats still
   failing, fallbacks are one env var: `G1_SLOWDOWN=2.0` then `2.5`.
5. **Box lifecycle discipline.** The box idled ~14.5 h after the verify crash
   (~$10) because the failure wasn't noticed. The chain now hard-fails loudly, but the
   standing rule stays: pull → sign → **DELETE the box the same session**.
6. **2–3 min dances are still unvalidated.** Thriller (~89 s at 1.8×) is the longest
   motion trained. The product target needs a full-length dance through the whole
   pipeline once the recipe signs.

## 5. Isaac Lab: migrate or not?

**Recommendation: do NOT migrate training to Isaac Lab. Keep mjlab as the training
stack; adopt Isaac Lab (or the official Unitree deploy sim) later, only as an
independent *verification* environment — and only if the calibration says our gate
still can't be trusted.**

Reasoning:

- **The original reason to consider Isaac Lab is gone.** It was pinned as primary in
  June because BeyondMimic targeted it; mjlab was the "bounded fallback". The project
  inverted that months of work ago: the entire recipe chain (v5→v8), curriculum
  scripts, gate, picker, exporter, and the faithful laptop preview are mjlab-native,
  with a known-good pinned env (mjlab 1.5.0 + mujoco-warp 3.10.0.1 + warp 1.14.0).
  Migration = re-porting the v8 task + revalidating physics + a new round of
  env-debugging on GreenNode's fixed-image notebooks (Isaac Sim there was already
  flagged high-risk), for zero demonstrated benefit.
- **Upstream parity does not require it.** Unitree ships BOTH `unitree_rl_lab`
  (Isaac) and `unitree_rl_mjlab`; the v8 obs contract was ported from their mjlab
  variant. We are aligned with first-party practice on the stack we already run.
- **The one real Isaac advantage — the torque-speed-curve actuator model — was
  neutralized cheaply**: v8 clamps ankle effort to the 40 Nm velocity-derated envelope
  inside mjlab, which is *more conservative* than mjlab's old flat 50 Nm and captures
  the part of the curve that was killing us.
- **What Isaac Lab is still good for: independence.** Our gate scores in the same
  simulator the policy trained in. The calibration anchor mitigates this, but a
  second, physics-independent simulator is the strongest cross-check available
  without hardware. That's a *verification* use (run the exported ONNX in an Isaac
  or unitree_mujoco scene, compare survival/fall-timing), not a training migration —
  a bounded 1–2 day experiment, best done on the new PC's GPU when it arrives, and
  only worth doing if (a) the calibration mapping looks untrustworthy or (b) a policy
  passes the gate but fails on the repaired robot.

## 6. Bottom line

- The revamp fixed real, quantified root causes (impossible torque target, floating
  feet, privileged-obs leak, optimistic ankle model) rather than re-rolling reward
  weights — and training is, for the first time, *stable* at full length.
- "Training finished" did **not** mean "everything works": the verify/export chain was
  silently broken in three places; all three are now fixed and re-running.
- Whether v8 is deployable is decided by `gap.json` vs the **calibrated** bar (§2),
  then by the robot coming back from repair. Isaac Lab migration is not needed for
  either.
