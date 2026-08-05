# HoloSoma (amazon-far) evaluation — 2026-08-05

Repo: https://github.com/amazon-far/holosoma (Apache-2.0, ~1.6k stars, active).
Cloned to `third_party/holosoma` (gitignored). Evaluated against our v12 symptoms:
robot looks perpetually near loss-of-balance, horizontal movement limited, legs
under-reach (matches measured style gap: sway 3× stiff, leg reach 60–79%,
`experiments/style_gap_20260722/`).

## What it is

Amazon FAR's humanoid framework: RL locomotion + whole-body tracking (WBT) for
**Unitree G1 29-DoF** (+ Booster T1). Trains on IsaacGym / IsaacSim / **MJWarp**
(same mujoco-warp family as our mjlab stack), PPO + **FastSAC**. Own retargeting
module (SMPL-H / mocap / LAFAN world-joint-positions in). Own real-G1 inference
stack (50 Hz ONNX, ethernet, sim2sim over loopback) — and it SHIPS two pretrained
**G1 dancing policies**: `fastsac_g1_29dof_dancing.onnx` + `ppo_g1_29dof_dancing.onnx`
under `src/holosoma_inference/holosoma_inference/models/wbt/`.

**Lineage: their WBT recipe is explicitly BeyondMimic-derived** (their code comments
say "reproducing beyondmimic") — the SAME lineage as our mjlab task. Core rewards
match ours incl. the global body lin/ang velocity tracking terms (σ=1.0 / 3.14,
w=1.0) — so "we're missing velocity rewards" is NOT the diff.

## The diffs that matter (mapped to our symptoms)

1. **Full-body keypoint TERMINATION** (`BadTrackingZOnly`): episode ends if
   ankle/wrist keypoints deviate >0.25 m from reference (plus ref pos 0.5 / ori
   0.8 over 14 tracked bodies). Ours terminates on anchor-z + XY-drift only —
   NOTHING terminates leg under-reach, so tracking reward can trade the legs away.
   Their design makes leg reach mandatory. ← prime candidate for the reach symptom.
2. **Radically leaner shaping.** Their entire regularizer set: action_rate −0.1,
   soft dof-pos limit −10, undesired-contacts −0.1. NO ankle torque terms, no
   stance shaping, no saturation penalties. Robustness comes from DR instead:
   friction 0.3–1.6, base-COM + actuator randomization, **pushes every 1–3 s up to
   0.5 m/s during training**. Our audit already found much of our ankle/stance
   shaping inert or inverted; their real-robot dancing result is evidence that the
   heavy shaping isn't needed. ← candidate for the timid/stiff style.
3. **FastSAC** (off-policy): their best dancing policy is FastSAC-trained. A
   genuinely new tool vs our PPO-only stack; may escape conservative optima.
4. **Deploy contract simplicity**: actor obs = motion command + ref-ori + base
   ang-vel + dof pos/vel + actions, **history_length=1**, estimator-free. A
   working real-G1 dancing policy with NO history stack questions whether our
   770-dim 5-frame contract is buying anything.
5. **Caveat kept from OUR evidence**: their ctrl-delay DR is 0–1 steps and
   disabled by default — weaker than our measured 40–80 ms hardware latency.
   Any recipe borrowing keeps our latency DR.

## What it does NOT replace

Our video→SMPL front end (GVHMR; they need mocap-grade world joint positions),
motion vetting/feasibility gates, the show app, and our safety-hardened deploy
runtime + runbooks. Full framework migration: not recommended — costs exceed
benefits while the delta is recipe-level.

## Recommended actions (cheap → expensive)

1. **Laptop, free**: run their shipped dancing ONNX sim2sim (their stack,
   loopback + MuJoCo, no GPU) — qualitative benchmark: does THEIR G1 look
   balanced/committed? Optionally score it through our style metrics for numbers.
2. **v14 A/B on the next box (mjlab, our stack)**: v12 recipe MINUS most custom
   shaping, PLUS their keypoint termination (feet/wrists 0.25 m). One knob at a
   time per sweep discipline.
3. **Optional third arm**: FastSAC via their MJWarp path on the same motion, as
   an independent-optimizer probe.
4. **Later (gantry)**: their inference stack as an INDEPENDENT deploy cross-check
   of our obs/gains handling — the independent verification the trust chain
   (2026-07-16) always wanted.
