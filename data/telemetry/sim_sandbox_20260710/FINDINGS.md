# AGENT D — Policy-in-the-loop Sim Sandbox: first results (2026-07-10)

`tools/sim_sandbox.py` runs the ACTUAL `policy.onnx` in a dynamic MuJoCo sim using the
EXACT deploy contract (obs builder, inference, action→target, PD all IMPORTED from
`pipeline/deploy_runtime.py` — the twin can't drift from the robot). This is the "honest
preview": what the robot really does, vs the reference the 3D preview plays.

## It reproduces BOTH tester findings — locally, before hardware
| condition | survives full dance? | achieved fidelity | note |
|---|---|---|---|
| ideal (0 ms latency), free | falls ~7.2 s | 72.7 % | free sim (no tether) drifts+topples early |
| ideal, **tethered** (kp 150) | **yes** | **79.7 %** | the operator's condition; full dance plays |
| **60 ms latency**, tethered | **falls ~20 s** | 79.7 % (pre-fall) | latency alone destabilises it |

- **Fidelity ~72–80 %** — matches the tester's "robot does 60–70 %". The policy washes out
  ~20–28 % of the reference range. **Worst-tracked DoFs: L_wrist_pitch, L/R ankle_pitch,
  R_knee, R_shoulder_roll, R_ankle_roll** — the subtle distal joints, exactly the "subtle
  moves it skips" the tester saw.
- **Latency is destabilising:** ideal survives; 60 ms → fall at ~20 s even tethered. This
  reproduces the 2026-07-09 drift-then-fall on the laptop, no robot needed — and validates
  the Lane-E retrain thesis (right latency DR) with a local, visual, testable harness.

## Rendered honest preview
`thriller_policy_rollout_ideal.mp4` (full dance, tethered, ideal) + `tracking_report_ideal.json`
(per-DoF achieved fraction). `--latency-ms` and `--tether-kp` flags let you A/B ideal vs
hardware-like.

## Caveats / trust-gate work (next)
- SIM-TO-SIM gap: uses the menagerie G1 model, NOT the mjlab model the policy trained on —
  contact/mass differ, so the FALL TIMING is not quantitative (fidelity fractions are the
  trustworthy signal). Close by matching mjlab contact params OR the trust gate below.
- TRUST GATE (per AGENT_D): cross-validate the sandbox obs against a real `deploy_runtime
  --mode read` log (byte-compare, tol 1e-5) so the twin is provably faithful. Needs the robot.
- Tether model is a soft base spring (approx), not the real anchor geometry.
