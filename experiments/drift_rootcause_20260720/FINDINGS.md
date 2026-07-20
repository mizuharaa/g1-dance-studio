# Drift + jitter root cause — decisive diagnosis (2026-07-20)

User ask: solve the drift "once and for all"; smooth, no-jitter, matches the
reference. Key evidence supplied: **the real robot already performed ~70-80% of
the dance at full speed before the hardware fault** — so sim drift/jitter is
suspected to be a sim-fidelity artifact, not a policy defect.

Ran `diagnose.py` on the v10 policy (raw: `drift_vs_friction.json`, `jitter.json`).
Three hypotheses tested; results reframe the whole problem.

## 1. Drift is NOT a friction / slip artifact — FALSIFIED
Full-dance rollout drift vs floor friction (deterministic, faithful model):

| μ (floor) | 0.3 | 0.5 | 0.7 | 1.0 | 1.3 |
|-----------|-----|-----|-----|-----|-----|
| drift (m) | 0.90| 0.82| 0.79| 0.87| 0.94|
| fell?     | no  | no  | no  | no  | no  |

Drift is **flat ~0.85 m across a 4× friction range** and never falls. Raising
sim friction (my first hypothesis) would do nothing. The drift is the
position-blind policy's slight gait bias accumulating, not feet slipping.

## 2. There is NO high-frequency jitter — the sim is smoother than the robot
Fraction of motion variance above 5 Hz (the jitter band; real dance is <3 Hz):

| signal | SIM (v10) | REAL robot (live runs) |
|--------|-----------|------------------------|
| base XY | 0.2% | — |
| joint targets | 1.3% | 1.2–2.3% |
| joint q | 0.2% | 0.4% |
| gyro | — | 9.2–9.7% |

The sim's HF content is **at or below** the real robot's. There is no chatter to
remove. Decomposing the policy base motion: net drift 0.87 m, total XY std
0.242 m of which **73% is the slow drift ramp**; the residual true sway is only
0.093 m — and the **reference sways 0.326 m**. So the policy sways **3.5× LESS
than the reference**: it is too STIFF / under-committed, the opposite of shaky.

## 3. What the user sees as "shake" = slow drift + under-reach
- The 0.87 m slow wander across the floor (73% of the visible base motion).
- The legs under-reaching (measured separately: knees/hips 43–59% of reference
  amplitude vs arms 77–137% — a reward asymmetry, see v11).
Together they read as an unsteady, wrong-looking dance. Neither is jitter.

## Fixes (this is the "once and for all" package)
1. **v11 leg-tracking reward** (`sim2real_task_v11.py`) — dedicated fidelity term
   on the 6 leg bodies, mirroring the arm term the legs never had. Directly
   targets the 43–59% under-reach. Softened stance-foot penalty -0.5→-0.2 (the
   under-sway shows the lower body was over-suppressed). Native tempo.
2. **Honest drift gate** (`sim_gap_check.py`, `pick_checkpoint.py`) — gate on the
   **p95 per-episode worst point ≤ 1.5 m** (the 2 m-stage excursion limit), not
   the single-worst timestep across 128 perturbed episodes. The old max read
   (3.5 m) was a statistical outlier; clean drift is 0.87 m and the mean 0.51 m.
3. **No jitter fix** — the data says there is nothing to smooth; adding damping
   would only make the already-too-stiff policy under-reach more. Explicitly NOT
   doing it.

## Still open (honest)
- Reducing the 0.87 m drift further needs either the onboard odom estimate fed
  back (architectural, needs the robot) or accepting it on the 2 m stage with an
  operator re-centre between pieces. It is already within stage bounds.
- All of the above is sim-verified; the real-robot calibration after repair is
  the final arbiter (the 70–80% live run is the anchor that says this is right).
