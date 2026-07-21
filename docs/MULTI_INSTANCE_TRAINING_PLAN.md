# Multi-instance training — plans & options (2026-07-21)

## The bottleneck is SEARCH BREADTH, not single-run speed

10+ sessions on one dance were slow for two reasons, and neither is "one run is
too slow":
1. **Runs were serial** — one policy at a time, ~6 h wall each (15 min provision
   + 4-5 h train + ~40 min verify), so exploring N recipes took N × 6 h.
2. **Runs were inconclusive** — the 2026-07-21 audit found the reference was
   blunted, the preview was a broken witness, and the ankle recipe was inert. So
   several of those sessions couldn't have told us anything.

Multi-instance training attacks #1. The audit fixed #2. Together they compound.
**The goal is more decisive attempts per wall-clock, not a faster single run** —
single-GPU RL wall-clock is roughly fixed, but we can run many configs at once.

Two enablers just landed that make parallel search actually work:
- The recipe is **env-var driven** (24 knobs: `G1_ANKLE_BARRIER_TAU/_W`,
  `G1_STANCE_*`, `G1_SAT_DURATION_W`, `G1_LEG_POS_W`, `G1_DRIFT_TERM_M`,
  `G1_GATE_*`, curriculum iters, seed …). A sweep is a **list of env-var sets** —
  zero code divergence per variant.
- Fix J records the **actual numeric gate bars inside each `gap.json`**, so
  cross-box results are finally comparable on ONE fixed bar (before, PASS/FAIL
  embedded a per-run bar and "v11 beat v8" was apples-to-oranges).

---

## Plans (tiered; recommend running 1 → 2, add 3 as volume grows)

### Plan 1 — Parallel recipe sweep  ★ do first
N notebooks, each a **full run with a distinct env-var config**, all on the v12
motion + audit fixes; pull every `gap.json`, rank on the fixed bars.
- Turns "4 serial attempts over a week" into "4 parallel attempts in one 6 h wave."
- Cost: N× per wave, but **same total compute as N serial runs at 1/N wall-clock**.
- Best for knobs that affect the **whole** curriculum (ankle effort DR envelope
  F, barrier G) — those need a consistent config across all 4 stages, so they
  can't share a trunk.

### Plan 2 — Shared-trunk fork  ★ efficiency multiplier
The curriculum's first 3 stages (0.60→0.90× tempo, ~7250 iters, ~3.5 h) are
**identical across variants of the final-stage-sensitive knobs**. Train the trunk
ONCE, snapshot `model_7250.pt`, then fork N cheap notebooks that resume ONLY the
native-tempo final stage (~2250 iters, ~1 h each) with different configs.
- Saves ~**60 % of compute per variant** vs Plan 1.
- Applies to knobs that bite at native tempo / late: `G1_DRIFT_TERM_M`, stance
  weights, curriculum stop-point, leg-tracking weights.
- Does NOT apply to the ankle DR envelope/barrier (they want consistency from
  stage 1). Use Plan 1 for those, Plan 2 for the rest.
- Requires getting the stage-3 checkpoint to each fork box (laptop relay or a
  shared bucket — see Orchestration).

### Plan 3 — Successive halving / early screen  ★ compute-saver at scale
Launch 6-8 configs only to the 0.90× stage (~7250 iters), cheap-screen each with
`sim_gap_check --quick`, **kill the bottom half**, promote survivors to the
native final + full verify.
- Fewest wasted GPU-hours when the search is wide and most configs are duds.
- Leans on the existing `pick_checkpoint`/`--quick` screen being early-predictive
  (it roughly is). Add a hard "log what was killed" so we never silently drop a
  late-blooming config.

### Plan 4 — Population-based training (PBT)  — later
N workers train concurrently; periodically losers copy the winner's weights and
perturb hyperparams. Best once we know **which** knobs matter (from 1-3). Higher
orchestration cost (weight-sharing across boxes). Defer.

### Alternative — one multi-GPU box vs many single-GPU notebooks
`rsl_rl` does data-parallel via `torchrun`/NCCL, but that needs **one box with
multiple GPUs** (shared NCCL), not multiple separate notebooks (multi-node needs
inter-node networking the notebooks likely lack). A 2×/4× 4090 box cuts ONE run's
wall-clock ~1.6-3×, but that speeds one policy, not the search. **Recommendation:
for our bottleneck, N single-GPU notebooks on different configs beat one
multi-GPU box on one config.** Only go multi-GPU when a single 2-3-min-dance
final's wall-clock itself becomes the constraint. → OPEN QUESTION for you: does
GreenNode offer multi-GPU instances and/or persistent volumes/custom images?

---

## Orchestration (laptop = controller; extends run_attempt9 + monitor.py)

A `cloud/sweep.sh` + `configs.yaml` driver:
1. You provision the boxes (console + reCAPTCHA) and paste their `IP:port`s.
2. Driver pushes `cloud/` + the v12 motion to each, launches each with its
   env-var config (`RUN_NAME` distinct), all logging to **one W&B project**.
3. One dashboard polls all boxes (reuse `monitor.snapshot` per box).
4. On completion, pulls every `gap.json` + heldout + policy.onnx, emits a
   **comparison table**: survival, ankle p95, drift p95 (fix K), leg-reach %,
   rr_mpkpe — all on the fixed bars (fix J). Winner promotes; renders previews
   for the top 2 only (save render time).

No shared filesystem across notebooks → aggregate via **laptop pull** or **W&B**
(already configured). For Plan 2's trunk checkpoint, relay `model_7250.pt` through
the laptop (or a cheap object bucket if GreenNode has one).

---

## Logistics, cost & friction (honest)

- **Provisioning tax:** each notebook = one reCAPTCHA + ~15 min mjlab install
  from the frozen lock. N boxes = N reCAPTCHAs. Mitigations: (a) a GreenNode
  **persistent volume / custom image** with mjlab pre-installed → provision drops
  to ~1 min; (b) provision a wave once and run **several sweeps on it** before
  deleting; (c) keep the env on the persistent volume between waves.
- **Billing** = creation→deletion per box (~$25/box/run historically). A 4-box
  6 h wave ≈ ~$100 for 4 decisive results vs ~4 weeks serial. Plan 2/3 cut this.
- **Seed discipline:** fix the seed per config so a result is attributable to the
  config, not seed luck (v10/v11 tail-episode variance was real). OR sweep seed
  as an explicit dimension.
- **Verify pipelining:** the ~40 min verify is GPU-bound and currently blocks the
  training box. Offload it to a second cheap instance (or run it while the box
  starts the next config) to keep the expensive GPU on training.

---

## Proposed FIRST WAVE — validate the audit fixes + tune the new knobs

F/G/H/I are untested (the audit gave starting points, said "needs GPU"). First
wave = confirm no regression + find the sensitive knobs. 4 boxes, v12 motion,
all audit fixes, W&B project `thriller-sweep-1`:

| Box | Purpose | Key env deltas vs audit-default |
|-----|---------|--------------------------------|
| A (control) | audit-fixed reference | defaults (tau_soft 16, stance_flat residual, sat table, DR 26-38, drift 0.6/0.8) |
| B | aggressive ankle unload | `G1_ANKLE_BARRIER_TAU=12` + retuned `_W` |
| C | favor reach/stability | `G1_ANKLE_BARRIER_TAU=18`, `G1_DRIFT_TERM_M` looser (0.8/1.0) |
| D | seed control | = A, different seed (isolate seed vs config) |

Full run each (ankle knobs affect all stages → Plan 1, not fork). Compare on the
fixed bars; the winner + what D reveals about seed variance sets wave 2. If you
want to also tune the late-stage knobs cheaply, add a Plan-2 trunk box and fork
drift/stance variants off its `model_7250.pt` in parallel with A-D.

Also run **`cloud/verify_obs_layout.py` once** on any box this wave — it's the
box-side gate that fix E needs before any 770-dim build is trusted for deploy.

---

## Risks / caveats
- Parallel ≠ free: N boxes cost N× while running; the win is wall-clock + fewer
  wasted *conclusive* runs, not fewer dollars per result (Plan 2/3 recover that).
- Comparisons are only fair because of fix J/K/L — do NOT compare against pre-fix
  gap.json (v8-v11) numbers.
- More boxes = more concurrent SSH; keep the orchestrator's polling sparse
  (already a lesson) and stagger launches to avoid provider rate limits.
- The real quality ceiling is still the sim2real gap + the down robot; a sweep
  finds the best SIM policy faster, but the post-repair calibration run remains
  the final arbiter.
