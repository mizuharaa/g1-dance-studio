# Ground retrain — decision tree for the next attempt (v3+)

Staged 2026-07-04 while v2 trains. **Sim only.** Pick the branch by what the v2
held-out eval (`data/policies/thriller_ground_v2/RESULT.txt` + `heldout_*.json`) shows.
Run one attempt at a time — the box has ONE GPU; launch v3 only after v2's job ends.

## Context: where the wall actually is

- The failed original run died instantly on `ee_body_pos` (wrist/ankle **height** > 0.25 m)
  — an exploration cliff. v2 loosened that bound 0.25→0.6 to get past it, and it did.
- But mid-v2 the binding wall **moved to the anchor terminations**: `anchor_pos`
  (torso **height** error > 0.25 m) and `anchor_ori` (torso tilt > 0.8 rad), with
  `error_anchor_rot` ~1.0 rad and `time_out` pinned at 0. Read: past the ee cliff the
  policy **can't hold the torso upright/at height** — the balance problem that the dropped
  `base_lin_vel` (torso velocity) obs used to feed. This is the fundamental estimator-free
  cost, not a threshold artifact.
- Mid-training numbers are **pessimistic** (adaptive sampler concentrates on the hardest
  motion bins). The eval from a clean frame-0 start is the true test — trust RESULT.txt.

## Launch template (matches v2 infra)

```
ssh <box> 'cd /workspace/notebook-data/cloud && bash run_job.sh start <name> -- \
  bash job_train.sh Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation \
    /workspace/notebook-data/motions/thriller_deploy.npz <name> <TYRO OVERRIDES>'
```
Then the autopilot pattern (`autopilot_v2.sh`, re-point to the new job/out) exports +
gates at strict 0.25 AND matched bounds, with the ankle/wrist safety breakdown.

## Branches (choose by v2's RESULT)

### A. v2 nominal@matched ≥ 0.95 AND ankle_height_err ≤ 0.15 m  →  SHIP v2
Promote `thriller_ground_v2/` → `thriller_ground/`; it's tethered-ground SIM_READY.
Loose wrists are fine. No v3 needed.

### B. v2 survives but ANKLE tracking is loose (ankle_height_err > 0.15 m)  →  v3 = split ee
Keep ankles tight for fall-safety, stop terminating on wrists (they don't affect balance):
```
--env.terminations.ee-body-pos.params.body-names "(left_ankle_roll_link, right_ankle_roll_link)" \
--env.terminations.ee-body-pos.params.threshold 0.25 \
--env.rewards.action-rate-l2.weight -0.2 --agent.max-iterations 3000
```
Rationale: this is a *cleaner* split than v2 — tight ankles (safety), arms free (were the
bootstrap killer). Plausibly the best config even if v2 barely passes.

### C. v2 dies on the ANCHOR wall (low survival even @0.6; error_anchor_rot high)  →  balance problem
The real blocker. Options, in order of preference:
1. **Longer training + curriculum**: 6000 iters; if mjlab exposes a termination
   curriculum, start `anchor_ori` loose (1.2 rad) and tighten. Estimator-free balance is
   simply slower to learn — more samples may cross it.
2. **Modest anchor loosening** (accept looser but standing): `anchor-ori.params.threshold
   1.1`, `anchor-pos.params.threshold 0.35`. WARN: looser torso bounds = closer to a real
   tip; only acceptable because deployment is tethered-first, and the eval must still show
   the robot RECOVERS (not just survives by a lenient bound). Report achieved error_anchor_rot.
3. **Reconsider the premise**: if estimator-free Thriller won't balance, the honest path is
   a torso-pose estimate on the robot (even a rough IMU-integrated base_lin_vel, or the
   DLIO/LiDAR estimator) so the FULL-obs gantry policy — already 100% — can deploy on the
   ground. This trades sim R&D for robot infra. Flag to Alois; it's a strategy call, not a
   knob.

### D. v2 never learned (survival ~0 everywhere)  →  back to exploration
Unlikely given v2 bootstrapped, but if so: the ee loosening wasn't enough alone — combine
branch B's split with a bigger initial ee bound (0.8) and re-check episode-length growth in
the first 150 iters before committing GPU-hours.

## Standing rule
No policy reaches the robot without RESULT.txt = SIM_READY=YES **and** a human-supervised,
tethered-first bring-up. A lenient-threshold "pass" that only survives by not being
terminated (vs. actually recovering) is NOT show-ready — say so in the report.
