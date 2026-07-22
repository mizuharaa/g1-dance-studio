# Style gap diagnosis (2026-07-22) — v12-final policy, uncertainty scene rollout

User report: "still twitching, legs weird and short, drifting still there".
All three CONFIRMED quantitatively (script = the sandbox probe recorded below);
the harness-v2 gate passes because it gates survival/drift/torque — NOT style.

| complaint | measured | baseline/target |
|---|---|---|
| twitch | leg TARGET HF(>5Hz) 3.97%, 1.77 deg/tick | real robot targets 1.2-2.3% |
| short legs | reach 60-79% (arm-swing window, sharper v12 ref) | v11 got 57-91% vs a BLUNTED ref |
| stiff sway | detrended base sway 0.116 m | reference 0.33 m (3x under) |
| drift | 0.53 m final / 0.68 max (clean rollout) | passes 1.5 bar; visible on camera |

Root causes: legs have tracking rewards but NO per-channel action-rate penalty
(ankles do); leg tracking stds (0.30/0.40) too loose for the sharper reference;
arms clean (0.29% HF) => not an actuation/scene artifact.

REVAMP (attempt 10, recipe v13 = v11 +):
1. per-channel hip/knee action-rate penalty (mirror v8's ankle one) -> kill chatter
2. leg stds tightened 0.30->0.26 pos / 0.40->0.34 ori (env vars, no code)
3. style metrics recorded per gate run going forward (this probe)
Drift 0.53m: at the practical floor for a position-blind policy; revisit only
after robot calibration (audit's bounded-observer plan).
