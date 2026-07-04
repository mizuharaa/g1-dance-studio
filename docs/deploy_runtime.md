# Laptop-side deploy runtime (`pipeline/deploy_runtime.py`)

Runs the trained mjlab ONNX policy on the **real Unitree G1 over Ethernet**, driven from
the laptop exactly like `~/robot`'s teleop (unitree_sdk2py + CycloneDDS on `enp0s31f6`).
No Docker / onboard controller — the robot's Docker env belongs to a colleague and the
BeyondMimic image isn't present.

## Env
`conda activate tv` — it has unitree_sdk2py + CycloneDDS + numpy + onnxruntime (onnxruntime
installed 2026-07-05). Robot artifacts under `data/policies/thriller/`.

## Modes
| mode | safety | what it does |
|---|---|---|
| `read` (default) | **sends NOTHING** | reads LowState, builds the 160-D obs, runs the policy, prints obs sanity + actions + target-vs-now. Use to sanity-check before any motion. |
| `move-to-default` | GATED | cosine-interpolate from the current (limp) pose to the ready pose `default_joint_pos`, conservative gains (30% kp). The policy assumes the robot STARTS here. |
| `run` | GATED | 50 Hz policy loop on `thriller_deploy` (has the 2.5 s activation ramp), policy_meta gains, full clamps + NaN→damp + cycle watchdog. |

Motion modes refuse unless **BOTH** `--i-will-watch-the-robot` **AND** env
`CONFIRMED_BY_HUMAN=alois`. Any NaN/inf/out-of-range/overrun → immediate damping. Targets
clamped to joint limits, gains are the SIM gains from policy_meta (never stock).

```
conda activate tv
python -m pipeline.deploy_runtime --mode read                       # safe, first
CONFIRMED_BY_HUMAN=alois python -m pipeline.deploy_runtime --mode move-to-default --i-will-watch-the-robot
CONFIRMED_BY_HUMAN=alois python -m pipeline.deploy_runtime --mode run --i-will-watch-the-robot
```

## READ-ONLY result on the real robot (2026-07-05, limp on gantry)
- obs: 160-D, **0 non-finite**, range [-1.13, 0.99]. Exact terms (command/joint_pos/
  joint_vel/base_ang_vel/actions) all sane; robot near-still (gyro≈0, joint_vel≈0).
- policy output: **finite, bounded actions [-1.42, 2.10]** — healthy. Target-vs-now shows
  big moves (up to 93°) because the robot is limp, not at the ready pose — expected, and
  exactly why `move-to-default` must run before `run`.

## OBS FIDELITY (must read before GROUND use)
148/160 dims are built EXACTLY from LowState + the reference motion. The other 12
(`motion_anchor_pos_b` 3, `motion_anchor_ori_b` 6, `base_lin_vel` 3) need the torso's
world pose/velocity, which the robot can't measure without a state estimator:
- On the **gantry** the base barely moves → `base_lin_vel≈0` (inside training noise) and
  `motion_anchor_pos_b` is approximated as the reference's displacement-from-start in the
  IMU frame (≈0 at t=0). `motion_anchor_ori_b` uses the IMU quaternion (pelvis≈torso).
  Good enough for gantry sanity + tracking.
- For **ground-free**, wire a real torso-pose estimator (DLIO/LIO, per docs/sim2real_derisk.md)
  into `build_obs` for these terms, OR retrain an obs-restricted policy that drops them.

## Known assumptions to verify on hardware before `run`
- LowState motor index order == `joint_order_29dof` (matches `check_joint_calibration.py`).
- `mode_machine` is read from LowState; LowCmd uses the HG `rt/lowcmd` topic + CRC.
- The IMU quaternion frame vs the mjlab torso frame — validate `motion_anchor_ori_b`
  against a known orientation before trusting ground playback.
