# Ground policy — obs-restricted (estimator-free) Thriller

**Why:** the gantry policy's obs includes 3 estimator-dependent terms the real robot
can't measure without a torso-position estimator: `base_lin_vel` (3) and
`motion_anchor_pos_b` (3). On the gantry we fed approximations (~0) because the robot
doesn't translate; **on the ground those become wrong → the policy flies blind → fall.**
Fix (BeyondMimic arXiv 2508.08241, §"omit the linear components"): retrain an
**obs-restricted** policy that drops those terms entirely, so it needs NO estimator.

## The task
mjlab already ships it: **`Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation`**
(`env_cfgs.py: has_state_estimation=False` filters `["motion_anchor_pos_b","base_lin_vel"]`
from the actor obs). Training launched on `thriller_deploy.npz` (the ramped deploy motion),
`--env.rewards.action_rate_l2.weight=-0.2` (attempt-2's winning delta), 3000 iters.
Autopilot `tools/overnight_ground.sh` exports + held-out-evals (on the **same
No-State-Estimation task**) + signs → `data/policies/thriller_ground/RESULT.txt`.

## New actor obs layout — 154-dim (was 160)
Dropped `motion_anchor_pos_b` (3) + `base_lin_vel` (3). Remaining, IN ORDER:

| term | width | source on real robot (all MEASURABLE, no estimator) |
|---|---|---|
| command | 58 | reference joint_pos+vel (known motion) |
| motion_anchor_ori_b | 6 | IMU torso orientation vs reference orientation (measurable) |
| base_ang_vel | 3 | IMU gyro |
| joint_pos | 29 | encoders (q − default) |
| joint_vel | 29 | encoders |
| actions | 29 | last commanded action |

**Every term is measurable from IMU + encoders + the known reference — no position
estimate needed.** This is what makes it honestly ground-deployable.

## Required change to `pipeline/deploy_runtime.py` (do NOT do autonomously — human-supervised ground session)
`build_obs()` + the module `OBS_LAYOUT` must switch to the 154-dim layout for the ground
policy:
- **REMOVE** the `motion_anchor_pos_b` term (currently `R_rob.T @ (ref_apos - ref_apos0)` —
  a gantry approximation) and the `base_lin_vel` term (currently `np.zeros(3)`).
- Keep `command`, `motion_anchor_ori_b` (IMU-vs-ref, already measurable), `base_ang_vel`
  (gyro), `joint_pos` (q−default), `joint_vel`, `actions`.
- New `OBS_LAYOUT = [("command",58),("motion_anchor_ori_b",6),("base_ang_vel",3),
  ("joint_pos",29),("joint_vel",29),("actions",29)]` → 154.
- Point `--policy`/`--meta`/`--motion-npz` at `data/policies/thriller_ground/`.
- Recommend a `--ground` flag (or auto-detect obs width from policy_meta
  `actor_obs_terms_in_order`) so the runtime picks the right builder; guard: assert the
  built obs width == the ONNX input dim before commanding anything.

## Honest caveats
- **Position-drift-tolerant, not position-accurate.** Without position feedback the robot
  can drift in xy over the dance (research: ±0.25–0.5 m). Fine for an **in-place** show;
  choreography must not require precise stage travel. Mark the floor + re-center between runs.
- **Push robustness is lower by design.** The No-State-Estimation config disables the
  push-randomization event in training (can't learn velocity-based recovery without velocity
  obs). Expect `push_survival` < `nominal_survival` in the held-out gate — that's the tradeoff.
- **SIM_READY ≠ show-ready on the ground.** A passing sim verdict here only clears the
  policy for a **human-supervised, tethered-first** ground bring-up (per ROBOT_DAY_PLAN.md
  ground stages). NEVER run this on the ground autonomously — the ground fall-risk is real.

## Status
Training running (train-thriller-ground, ~30 min). Result lands at
`data/policies/thriller_ground/RESULT.txt` via the autopilot. Verify the exported
`policy_meta.json` `actor_obs_terms_in_order` matches the 154-dim layout above before use.
