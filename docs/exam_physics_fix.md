# Sim-exam physics reconciliation — findings (2026-07-04)

**Task:** make the independent signed sim2sim exam (`pipeline/sim_exam.py`) score the
real mjlab-trained Thriller policy, and run the real exam honestly.

## Outcome: two faithful fixes landed + a decisive fidelity finding. Thriller stays DRAFT.

The exam still cannot reproduce mjlab's result, and I did **not** tune it until it
passed. The honest verdict is now **`invalid`** (a harness fidelity problem), not a
false `fail` that would be misread as "the policy is bad."

### Fixes made (genuine, faithful — kept)

1. **Per-joint armature reconciliation.** The plain `unitree_mujoco` G1 ships a flat
   `dof_armature = 0.01` on 9 joints; mjlab uses per-joint rotor inertia and matches its
   stiff PD gains to it (`kp = armature·(2π·10)²`). The exam now recovers each joint's
   armature from the exported gains (`armature = kp/(2π·10)²`) and writes it onto the
   model — the recovered values snap exactly to the real G1 motor armatures
   (5020=0.0036, 7520_14=0.0102, 7520_22=0.0251, 4010=0.0043, ankles/waist=2×5020),
   asserted in code so a mis-scaled meta can't slip through.
2. **IMU velocimeter lever-arm.** The obs velocity terms (`base_/imu_lin_vel`,
   `base_/imu_ang_vel`) were the raw root-body velocity. mjlab's are a velocimeter/gyro
   at site `imu_in_pelvis` (offset (0.04525, 0, −0.08339)), which includes the ω×r
   lever-arm term (large during dynamic motion). The exam now moves the model's IMU site
   to that offset and reads the true site velocity via `mj_objectVelocity`.

### The decisive finding: the exam MODEL is not dynamically equivalent to mjlab's

After both fixes, the real Thriller policy still falls at ~1.2 s (torso sinks + tips),
despite mjlab's own engine scoring **100% completion** (clean and under 64-env sensor
noise). Root-caused by elimination:

- Not the obs (verified 157/160 exact vs a real mjlab sample; velocity now lever-arm-correct).
- Not the armature (recovered exact motor values).
- Not the integrator: **identical** failure under Euler, implicitfast, and implicit.
- Not the policy/action-scale: **a pure static pose-hold (no policy, rigid PD or MuJoCo
  position servo, feet settled on the floor) still collapses at 1.38 s.**

A rigid hold of a valid standing pose collapsing means the **`unitree_mujoco` G1 model
used by the exam is not dynamically equivalent to mjlab's training model** (foot-contact
geometry, mass distribution, or ground/friction). A policy — and even a static stance —
that balances in mjlab does not balance here. Parameter-matching cannot close this;
the exam is built on the wrong physical model.

### Guard added (safety)

`static_pose_hold_ok()` runs before scoring. If the model can't hold the pose, the exam
prints a loud warning and emits `verdict = "invalid"` with `model_faithful: false` —
never `pass` (safe) and never `fail` (which a human would misread as a bad policy).
This prevents anyone from concluding Thriller is bad because of a broken harness.

## Recommended path to a trustworthy exam (pick one)

1. **Run the independent check inside mjlab itself** — different seeds, held-out DR,
   fresh env instances — a true independent pass within faithful physics. Cheapest,
   highest-fidelity. (mjlab already reports 100%; formalize it as the signed gate.)
2. **Rebuild the exam on mjlab's actual G1 model + `BuiltinPositionActuator`** (the
   MuJoCo Menagerie G1 mjlab trains on), not `unitree_mujoco`. Then this explicit-MuJoCo
   exam becomes a genuine second engine. Gate it behind the `static_pose_hold_ok` check.
3. Only after (1) or (2): the signed `/api/dances/sim-runs` path marks Thriller
   show-ready. Until then it correctly stays DRAFT.

## Also confirmed (route to deploy/robot-day)

- **`base_lin_vel` is not measurable on the real G1** — the deployed controller needs a
  state estimator for it, or it is a sim2real robustness gap.
- **Action scale is per-joint** (~0.35–0.55 legs/arms, **0.074 wrists**,
  `use_default_offset=true`), not the scalar `0.5` in some meta — the exporter's
  `policy_meta.json` should carry the per-joint vector; a scalar over-drives the wrists.
