# ACTUATION / CONTROL DESIGN — v8 (Agent D)

**Date:** 2026-07-16 · **Author:** Agent D (actuator-commanding / control) · **No GPU**
**Deliverable per PROMPT D.** Inputs used: Agent 0 upstream audit (T–N-curve finding,
kp/kd confirmed), Agent B motion-feasibility (`pipeline/motion_dynamics.py`,
`pipeline/g1_limits.py`, the 2.5× ankle-only repair). All numbers below are reproduced
on CPU by scripts under `tools/` + raw JSON under `experiments/motion_feasibility/`
(measurement-discipline rule). **Commits nothing** — orchestrator commits; Agent F trains.

---

## 0. TL;DR

- **Mildest safe slowdown with hips helping ≈ 1.7× (optimistic) – 1.9× (conservative);
  design target 1.8×**, vs Agent B's **2.5× ankle-only**. This confirms and tightens
  Agent B's 1.7–2.0× guess. The show goes from ~2 min (2.5×) to ~89 s (1.8×).
- **No mild slowdown is feasible ankle-only** — even 2.0× peaks at 62 Nm on the ankle.
  Anything milder than 2.5× *requires* the policy to use hip/torso angular momentum.
  So 1.8× is deliberately a reference that is feasible **only if** the RL policy learns
  hip strategy; the reward/actuation deltas below are what induce that.
- **kp/kd are NOT touched** (Agent 0 confirmed they already match upstream; raising kp
  increases peak ankle torque = the failure). The levers are: a **feasible (repaired)
  reference**, a **soft-barrier ankle penalty at ~40 Nm** (replaces the global L2), a
  **velocity-honest ankle effort clamp** (kill the sim's optimistic flat 50 Nm), and
  **freeing the waist/torso to counter-rotate** at the hard beats.
- **Top candidate (A):** train on the 1.8× repaired reference + ankle soft-barrier +
  ankle-channel action-rate + waist-tracking slack at the two beats + ankle effort clamp
  lowered to the velocity-derated envelope. All four are **training-only; zero deploy
  change** (same 50 Hz PD, same gains, same obs contract). Agent F trains it.

---

## 1. HIP-STRATEGY REDISTRIBUTION ANALYSIS

### 1.1 The physics (reuses Agent B's centroidal machinery verbatim)

`motion_dynamics.py` computes the support-ankle balance moment as
`τ_ankle(t) = F_z·‖ZMP − CoM_xy‖`. The ZMP already contains the reference's **own**
rate-of-change of centroidal angular momentum `Ḣ` (the ZMP formula subtracts
`Ḣ_y/F_z` in x and adds `Ḣ_x/F_z` in y). Split by axis:

| plane | ankle demand | corrected by extra `Ḣ` from |
|---|---|---|
| sagittal (x) | `τ_x = F_z·|ZMP_x−CoM_x|` → **ankle_pitch** | hip_pitch (88 Nm ×2) + waist_pitch (50) → ceiling ~226 Nm |
| lateral (y) | `τ_y = F_z·|ZMP_y−CoM_y|` → **ankle_roll** | hip_roll (139 Nm ×2) + waist_roll (50) → ceiling ~328 Nm |

**Hip strategy = inject extra `Ḣ` beyond the reference so the required ZMP moves back
toward the foot, unloading the ankle.** Extra `Ḣ` has torque units, so the substitution
is a clean subtraction in torque space:
`τ_ankle_hip = √( max(0,τ_x−ΔḢ_sag)² + max(0,τ_y−ΔḢ_lat)² )`.

**Why hips can't fully replace slowdown (the honest bound):** the trunk is a *bounded
flywheel* — it cannot counter-rotate forever, so hip strategy cancels only the
**transient** part of the ankle demand, never the **sustained quasi-static lean**. The
model (`tools/actuation_hip_strategy.py`) therefore lets the hip remove up to `C_HIP`
Nm of the fast component (moving-average window `W_HIP`=0.4 s) but never pull an axis
below its own sustained lean:
`τ_ankle_hip(t) = max( sustained(t), τ_ref(t) − C_HIP )`, split per plane by the effort
ceilings. `C_HIP` is the trunk's realizable angular-momentum-**rate** authority; because
its exact value needs GPU confirmation we sweep a band: **40 (conservative, single-axis
excursion-limited) / 70 (moderate) / 100 (aggressive, both hips)** Nm. The limb
torque-headroom ceiling is ~90 Nm sagittal / ~130 Nm lateral, so 100 is the physical cap.

### 1.2 Result — the mildest-safe-slowdown answer

Raw: `experiments/motion_feasibility/thriller_hip_strategy.json`
(`tools/actuation_hip_strategy.py data/motions/thriller/thriller_g1_clean.csv`).
Format `max/p95/%>40` Nm. **C=0 reproduces Agent B's ankle-only curve** (max 173.7 vs
her 173.58 at 1.0×; 40.0 vs 39.4 at 2.5× ✓ — cross-check passes).

```
factor  dur(s)  C=0 ankle-only    C=40 conservative  C=70 moderate     C=100 aggressive
1.00    49.3   173.7/106.0/63.1   153.9/90.0/53.3    142.1/82.7/51.7   130.4/79.6/51.7
1.30    64.1   122.1/ 70.9/32.8   101.3/57.3/19.4     89.5/54.3/17.7    78.0/53.3/17.5
1.50    73.9    98.0/ 57.2/18.6    77.8/45.1/ 7.8     66.1/42.9/ 7.2    59.2/42.5/ 7.0
1.70    83.8    79.9/ 46.1/ 8.3    60.6/35.7/ 2.7     49.8/35.1/ 1.9    49.8/34.9/ 1.8
1.90    93.7    65.7/ 37.8/ 4.0    47.6/29.4/ 0.3     41.4/28.7/ 0.1    41.4/28.7/ 0.1
2.00    98.6    62.0/ 34.6/ 2.5    43.9/26.4/ 0.1     38.2/26.3/ 0.0    38.2/26.2/ 0.0
2.50   123.3    40.0/ 23.0/ 0.0    25.4/18.0/ 0.0     25.4/18.0/ 0.0    25.4/18.0/ 0.0
```

Two feasibility bars:
- **STRICT** (motor-protective, *every* frame ≤ 40 Nm usable): dominated by the brief
  sustained-lean floor → **2.0× with hips (C≥70)** vs 2.5× ankle-only.
- **PRACTICAL** (thermal/show — sustained p95 ≤ 40, brief peaks ≤ 50 Nm hard clamp,
  ≤ 3% of frames in the 40–50 transient band): **1.7× (C≥70) / 1.9× (C=40)** vs 2.5×
  ankle-only. This is the bar that matters: the 50 Nm clamp tolerates <0.1 s touches;
  what kills motors (and the robot) is *sustained* saturation, which p95 captures.

| trunk authority | STRICT (max≤40) | PRACTICAL (p95≤40, max≤50) |
|---|---|---|
| ankle-only (Agent B) | 2.5× | 2.5× |
| C_HIP = 40 Nm | 2.5× | **1.9×** |
| C_HIP = 70 Nm | 2.0× | **1.7×** |
| C_HIP = 100 Nm | 2.0× | **1.7×** |

**Sensitivity:** with a shorter trunk counter-rotate window `W_HIP`=0.3 s (less transient
cancellation), the practical answer moves 1.7×→1.9× across the whole C band — so **1.7×
is the optimistic end and 1.9× the conservative end. Recommend 1.8× as the design
target** (robust across the W_HIP sensitivity, splits the band).

### 1.3 The 1.8× reference is feasible *only with hips* (why that's the right target)

`experiments/motion_feasibility/thriller_g1_repaired_1p8x_scorecard.json`
(sha256 `03883369cea9…`, style-similarity **1.000** — pure uniform time-stretch, music
stays synced): **ankle-only** p95 **40.3**, max **73.8**, **5.3%** of frames over 40.
That is still the v7-failure regime under ankle-only balance. Under hip strategy
(C=70) the same reference lands at ~p95 32 / max ~46 / ~1% over — feasible.

**Interpretation:** 1.8× is mild *precisely because* it banks on the policy using hips.
This is a bet, and it is the bet the user asked us to make. The fallback if the first
GPU run shows the policy under-using hips (ankle p95 still >20 at the beats) is **2.0×**
(ankle-only max 62 but p95 34.6 — much closer to self-feasible), then 2.5× (Agent B's
guaranteed ankle-only floor). Ranked configs below are written for 1.8×; the slowdown
factor is a one-line env var so Agent F can walk it 1.8→2.0→2.5 without a recipe rewrite.

---

## 2. RANKED CANDIDATE CONFIGURATIONS (deltas vs `cloud/sim2real_task_v7.py`)

All three keep **kp/kd/effort-map exactly as `policy_meta.json`** (Agent 0: they match
upstream and are the deploy gains; re-deriving them from the armature/impedance model
`kp=armature·(2π·10)²`, `kd=2ζ·armature·2π·10`, ζ=2 gives the same numbers — do not
touch). All three train on the **repaired (slowed) reference**, which is the root-cause
fix: v7 failed because the *reference itself* demanded 173 Nm ankle — no reward can add
torque the choreography lacks headroom for.

### Shared prerequisite (applies to A/B/C) — the actuation deltas

1. **Feasible reference.** Train on `thriller_g1_repaired_1p8x.csv` (or regenerate via
   `python tools/motion_repair.py <clean.csv> --apply-factor 1.8 --out …`). Env var
   `G1_SLOWDOWN=1.8` selects it; 2.0/2.5 are the fallbacks. **Deploy:** just a different
   motion file + time-stretch the music to match; **PD loop unchanged.**
2. **Velocity-honest ankle effort clamp (implements Agent 0's T–N finding).** mjlab's
   flat `effort_limit_sim`=50 Nm ankle is *optimistic* — the real ankle derates with
   speed (`g1_limits.effective_torque_limit()`), and the two fall beats are fast +
   high-torque. **Lower the ankle `effort_limit_sim` to the usable envelope (40 Nm)**
   AND widen the effort DR downward (`dr_effort_limits` 0.80–1.00 → **0.65–0.95** on the
   4 ankle channels only). This trains the policy against *less* ankle authority than
   the flat clamp, so it never learns to rely on 40–50 Nm ankle torque the hardware
   won't deliver at speed. This is the **safe** direction (the forbidden direction is
   *raising* the sim ankle limit). **Deploy:** none — deploy already clamps at the true
   motor limit; this only constrains what the policy learns to command.

### CANDIDATE A — *Feasible reference + hip-strategy shaping* ★ RECOMMENDED (Agent F trains this)

Directly targets the user's #1 priority: make the policy move balance load onto hips/torso.

**Reward deltas vs v7** (v7 = ankle_torque_l2 −1e-3, action_rate_l2 −0.25, arm terms 1.0):
- **Replace the global `ankle_torque_l2` with a soft-barrier** `ankle_torque_barrier`:
  `penalty = Σ_ankles relu(|τ| − τ_soft)²`, `τ_soft = 35 Nm`, weight **−5e-3**. Rationale:
  the global L2 (−1e-3) penalizes *all* ankle torque, which also suppresses the small
  legitimate ankle use in gentle passages and over-smooths gestures; a barrier is ≈0
  below 35 Nm and rises steeply toward 40, so it bites **only near saturation** — exactly
  where hip strategy should take over. (Implementation: a class term mirroring the
  existing `ankle_torque_l2` class in `sim2real_task.py`, reading `qfrc_actuator` on the
  4 ankle ids, applying the relu-squared hinge.)
- **Per-channel ankle action-rate limit** `ankle_action_rate_l2`: L2 on the *first
  difference* of the 4 ankle action channels, weight **−0.05** (on top of the global
  action_rate −0.25). Rationale: the v7 falls are oscillatory ankle saturation; damping
  the ankle command rate removes the buzz that spikes `qfrc_actuator` without needing a
  kd change (which would be deploy-coupled).
- **Free the waist to counter-rotate at the hard beats.** Add a time-gated slack on the
  waist tracking terms: multiply the waist components of `motion_*` tracking rewards by
  **0.5** inside 13–18 s and 25–36 s (scaled to the slowdown clock), full weight
  elsewhere. Rationale: the only way to unload the ankle while still tracking the CoM is
  to add trunk angular momentum (hip strategy); if the tracking penalty pins the waist to
  the reference, the policy is *denied its flywheel*. Slackening waist tracking at the two
  beats gives the policy room to inject the ΔḢ the analysis credits. **Arms/shoulders
  stay at 1.0** (gesture fidelity — the waist is the less-visible, higher-inertia
  flywheel, so we spend it first).

**Optional (ablation, +1 GPU run):** explicit `torso_angmom_use` reward = +small ×
`‖upper-body centroidal angular momentum‖` gated by ankle-near-saturation, if the slack
alone doesn't move hip usage. Prefer the slack first (fewer moving parts).

**Analytical evidence:** §1.2 — at 1.8× the reference clears the practical bar iff
C_HIP≳50 Nm; the soft-barrier + waist slack are the terms that make the policy realize
that C_HIP. **Deploy-side changes: NONE** (all reward/effort, training-only; obs, action
space, gains, 50 Hz PD all unchanged). Composes cleanly with Agent 0's 155-dim
no-state-estimation actor (these deltas don't touch obs).

### CANDIDATE B — *A + residual-action parameterization* (more robust, needs deploy change)

Everything in A, plus reparameterize the policy output as a **residual around the
(now-feasible) repaired reference**: commanded joint target = reference_pose(t) +
action·action_scale, instead of default_pose + action·action_scale. Rationale: with a
feasible feedforward, the policy only has to command *corrections*, so action magnitudes
(and thus peak PD torque, ∝ kp·error) shrink — less chance of driving the ankle to
saturation from a large tracking error. Pairs naturally with hip strategy because the
corrections it learns are the balance deviations, which the soft-barrier steers to the hips.

**Analytical evidence:** peak PD torque `≈ kp·(target−q)`; a feasible feedforward cuts
the tracking-error term that dominates the spikes. **Deploy-side change: YES** —
`pipeline/deploy_runtime.py` must add `reference_pose(t)` to the policy output before the
PD step (a phase-indexed feedforward table shipped alongside the policy). Non-trivial but
well-scoped. Ranked below A only because of that deploy change + the extra plumbing/risk;
if A's first GPU run is marginal on survival, B is the next lever.

### CANDIDATE C — *A + modest ankle impedance softening* (gain reshaping — ablation only)

Everything in A, plus a **small ankle kd increase** (ζ 2.0→2.5 on the 4 ankle joints
only: `kd = 2·ζ·armature·2π·10` → ankle kd 1.814→2.268), leaving kp fixed. Rationale:
more ankle damping reduces the oscillatory torque demand that spikes `qfrc_actuator`.

**Why this is ranked LAST and proposed only as an ablation:** (1) kd is **deploy-coupled**
— the new value must ship in `policy_meta.json` and run identically on hardware, and be
re-validated end-to-end. (2) It **can backfire**: damping torque is `kd·q̇`, and the fall
beats are exactly where `q̇` is largest, so a higher ankle kd can *add* torque precisely
when the ankle is already saturating. (3) It changes the impedance model's ζ away from
the BeyondMimic value, which Agent 0 verified matches upstream. Do **not** ship this
without a dedicated GPU sweep showing it lowers, not raises, ankle p95 at the beats.
We include it so the option is documented and bounded, not so it is adopted casually.

---

## 3. WHAT NOT TO DO (rejected options + why)

- **Raise ankle `effort_limit_sim` in sim** (the tempting "give it more torque"): trains a
  policy that commands torque the hardware cannot execute at speed → guaranteed sim2real
  gap and a fall on the robot. The T–N curve (Agent 0) means the real ankle has *less*
  than 50 Nm at the fast beats; we go the other way (clamp to 40). **Hard no.**
- **"Just raise kp"** to track the hard motion: peak PD torque ∝ kp·error, so higher kp
  *increases* ankle saturation — the exact v7 failure. **Hard no** (PROMPT D constraint).
- **Pure torque control / direct torque commands:** deploy is PD at 50 Hz (`policy_meta`
  gains); there is no torque-command path on the robot. Any torque-shaping must be
  expressed through target angles + the fixed PD, not commanded torque. **Out of scope.**
- **Global ankle L2 heavier than −1e-3:** v7 already sits there; pushing it further
  over-smooths the gestures (the arms/torso go limp) without fixing the root cause (an
  infeasible reference). Replaced by the soft-barrier, which is 0 in the gentle regions.
- **Keep 1.0× and lower the survival bar:** the reference demands 173 Nm ankle — no bar
  and no policy makes that motor-safe. Feasibility is upstream of the acceptance bar.
- **Change kp/kd globally to chase smoothness:** they are the deploy gains and match
  upstream; global changes ripple into every joint's torque and the obs/action scaling.
  Only the bounded, ablation-gated ankle-kd move (Candidate C) is even considered, and
  only with its own validation.

---

## 4. WHAT NEEDS GPU SIM-VALIDATION (training wave)

The CPU analysis proves the **reference** is feasible under hip strategy; it **cannot**
prove the **policy** will realize it or hit the survival gate. GPU-only items:

| item | candidate | what to confirm |
|---|---|---|
| Policy actually learns hip counter-rotation (realized `C_HIP`) | A (core) | ankle p95 at 13–18 s / 25–36 s drops to <20 Nm **and** waist/torso angular-momentum usage rises at those beats (log centroidal `Ḣ`) |
| 1.8× vs 2.0× vs 2.5× is the right point | A | survival ≥ target at 1.8×; if <, walk `G1_SLOWDOWN` up (one env var) |
| Soft-barrier τ_soft=35 / weight −5e-3 doesn't over- or under-bite | A | ankle p95 ≤ 15 (gate) without gesture collapse (arm mpkpe stays good) |
| Lowered ankle effort clamp (40 Nm + 0.65–0.95 DR) doesn't hurt survival elsewhere | A | no new falls outside the two beats |
| Waist-tracking slack doesn't visibly degrade the dance | A | rr_mpkpe still ✓; visual check on the beats |
| Residual-action stability + the deploy feedforward table | B | training stability + a sim2sim replay of the feedforward path |
| Ankle-kd↑ lowers (not raises) ankle p95 at the fast beats | C | dedicated sweep; only ship if p95 drops |

Recommended first GPU run: **Candidate A at 1.8×**, on top of Agent 0's 155-dim
no-state-estimation actor + Agent A's calibrated gate. If survival is marginal (90–95%),
try 2.0× before adding Candidate B's machinery.

---

*Provenance: `tools/actuation_hip_strategy.py` (+ `--json` raw output),
`experiments/motion_feasibility/thriller_hip_strategy.json`,
`experiments/motion_feasibility/thriller_g1_repaired_1p8x_scorecard.json`. Reuses
`pipeline/motion_dynamics.py`, `pipeline/g1_limits.py`, `tools/motion_repair.py`
unmodified. kp/kd/effort per `data/policies/thriller_csv_ankle_penalty/policy_meta.json`.*
