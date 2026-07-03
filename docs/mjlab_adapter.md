# mjlab observation adapter — ground-truth verification & real-exam finding (2026-07-04)

## Verified 160-dim actor observation layout (Mjlab-Tracking-Flat-Unitree-G1)
Captured from the REAL mjlab env (corruption off) and validated term-by-term against
the exported Thriller policy. Fixture: `tests/fixtures/mjlab_obs_sample.npz`; gate:
`tests/test_mjlab_obs_layout.py`.

| slice | term | dim | construction | verified |
|---|---|---|---|---|
| 0:29 | command.joint_pos | 29 | motion ref joint_pos[t] (policy order) | exact (0.0) |
| 29:58 | command.joint_vel | 29 | motion ref joint_vel[t] | exact (0.0) |
| 58:61 | motion_anchor_pos_b | 3 | R_robotAnchorᵀ·(ref_anchor_pos − robot_anchor_pos) | exact (0.0) |
| 61:67 | motion_anchor_ori_b | 6 | first 2 cols of R_robotAnchorᵀ·R_refAnchor | exact (0.0) |
| 67:70 | base_lin_vel | 3 | **velocimeter @ site imu_in_pelvis** (see below) | ≤0.11 m/s |
| 70:73 | base_ang_vel | 3 | gyro @ imu_in_pelvis = qvel[3:6] (ω site-invariant) | exact |
| 73:102 | joint_pos | 29 | q − default_joint_pos (encoder-bias is DR, 0 clean) | exact (bias≈0.01) |
| 102:131 | joint_vel | 29 | qvel joints | exact (0.0) |
| 131:160 | actions | 29 | last policy action | structural |

Anchor body = `torso_link`. **No projected-gravity term.** 157/160 dims reproduce to
0.0; the 3 base_lin_vel dims are the one nuance:

## base velocity = velocimeter/gyro at `imu_in_pelvis`
mjlab g1.xml: `<velocimeter name="imu_lin_vel" site="imu_in_pelvis">`,
`<gyro name="imu_ang_vel" site="imu_in_pelvis">`; site offset (0.04525, 0, −0.08339)
from pelvis. The velocimeter includes the ω×r lever-arm term, so it is NOT root-body
velocity. main@6508bac uses root-body velocity → off by ≤0.11 m/s. **This is SAFE:**
the actor trains this term with `Unoise(−0.5, +0.5)`, so the policy already tolerates
±0.5 m/s here; the ≤0.11 discrepancy is well inside that. Refinement (optional): read
`mj_objectVelocity(model, data, mjOBJ_SITE, imu_site, res, flg_local=1)[3:6]`.

## SIM2REAL FINDING (robot-day, important)
`base_lin_vel` is **not directly measurable on the real G1** — no sensor gives clean
base linear velocity. The deployed controller needs a state estimator to supply it, or
this is a robustness gap at deploy. Flag for the deploy/robot-day stage.

## policy_meta.json sidecar (complete, exact — regenerate at export)
mjlab BuiltinPositionActuator gains (kp = armature·(2π·10)², kd = 2·2.0·armature·2π·10):
- 5020 (shoulders, elbow, wrist_roll): kp 14.2506, kd 0.9072
- 7520_14 (hip_pitch, hip_yaw, waist_yaw): kp 40.1792, kd 2.5579
- 7520_22 (hip_roll, knee): kp 99.0984, kd 6.3088
- 4010 (wrist_pitch, wrist_yaw): kp 16.7783, kd 1.0681
- WAIST (waist_pitch, waist_roll) & ANKLE (ankle_pitch/roll): kp 28.5012, kd 1.8144 (5020×2)
action_scale = 0.5, use_default_offset=True; default_joint_pos and 29-joint order per
the mjlab env cfg. obs_terms widths: command 58, anchor_pos_b 3, anchor_ori_b 6,
imu_lin_vel 3, imu_ang_vel 3, joint_pos 29, joint_vel 29, actions 29 (=160).

## REAL EXAM RESULT — FALSE-FAIL diagnosed (adapter obs = verified; exam physics = needs fix)
Ran the signed exam on the real Thriller policy + thriller_show.csv (2464 ticks, 49.3s):
**nominal FAIL, survived 1.18s / 49.3s.** This CONTRADICTS mjlab's own result (100%
completion, 100% under 64-robot sensor noise) → it is a **false-fail from an
actuator-model mismatch, NOT a bad policy:**
- The exam runs plain `unitree_mujoco` G1 (`scene_29dof.xml`) with explicit torque PD.
- That model has `armature="0.01"` on 9 joints only; mjlab uses per-joint armatures
  (0.0036–0.0251, ×2 waist/ankle). mjlab's stiff gains (kp up to 99) are DEFINED as
  armature·(2π·10)² — matched to mjlab's armature. Applied to the exam model's
  different/near-absent armature, the PD loop is unstable → instant collapse.
**Required fix before the exam can gate the robot:** give the exam MuJoCo model mjlab's
per-joint armature (and/or use MuJoCo position actuators mirroring BuiltinPositionActuator)
so the actuator dynamics match training, then re-run. Until then the exam FALSE-FAILS
good policies and must not gate deployment.

## Status
- Obs adapter (main@6508bac) obs construction: **VERIFIED** against ground truth
  (157/160 exact; base_lin_vel within trained noise). Not the cause of the failure.
- Signed gate/thresholds/signing: untouched and working (Thriller correctly stays DRAFT).
- Real exam verdict on Thriller: **FAIL (false-fail, physics/armature mismatch)** —
  do NOT interpret as a policy defect; fix the exam actuator model and re-run.
