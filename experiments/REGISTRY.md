# Experiment / Policy Registry

Single source of truth for every trained policy + gate run. **Never report a sim %
alone again** — every row carries both the raw gate number and (once Agent A lands a
calibration mapping) the calibrated real-world estimate. Append a row per run; never
overwrite. Seeded 2026-07-16 at git `18fc762`.

Columns: run_id | date | motion (file + sha256) | recipe (file + git hash) | gate
config hash | best checkpoint | gate raw-output path | calibrated real-world estimate
| notes.

## Seed rows (existing models — established before the revamp)

| run_id | date | motion | recipe | best ckpt | gate raw output | raw gate result | calibrated real est. | role / notes |
|---|---|---|---|---|---|---|---|---|
| `thriller_csv_ankle_penalty` | 2026-07-08 | thriller_g1_clean (csv→npz) | dance.yaml (ankle_torque_l2 -1e-3, action_rate_l2 -0.25) | model (96da66) | `data/policies/thriller_csv_ankle_penalty/gap_check.json` | survival 100%, mpkpe 0.154, ankle p95 10.7 | **~70% mimicry IRL (ONLY ground truth)** | ⭐ **CALIBRATION ANCHOR** — sha `444864f9…`. Agent A must run THIS through the current gate to tie gate%↔real%. |
| `thriller_v7ank` (iter 10000) | 2026-07-15 | thriller_clean.npz | `cloud/sim2real_task_v7.py` | model_10000.pt | `exports/train-thriller_v7ank-0715/gap.json` | nominal surv 85.9%, push 87.5%, ankle p95 16.5, drift 0.81, rr_mpkpe 0.09 | (pending Agent A) | **BASELINE TO BEAT** — sha `fec81199…`. Best-checkpoint-selected; last ckpt had collapsed to 3%. |
| `thriller_v6sk` | 2026-07-14 | thriller_clean.npz | `cloud/sim2real_task_v6.py` | model_9997.pt | `exports/train-thriller_v6sk-0714/gap.json` | nominal surv 92.2%, drift 1.67 (FAIL), ankle p95 17.7 | (pending Agent A) | Failure-signature ref — sha `6bb9598c…`. Drift unsolved; same two-beat collapse (13–18s, 25–36s). |
| `thriller_v5fid` | 2026-07-13 | thriller_clean.npz | `cloud/sim2real_task_v5.py` | — | (v5 exports) | drift 4.56 (FAIL), survival 92.2%, ankle p95 16.4/21.5 | (pending Agent A) | Failure-signature ref — drift wildly unsolved; motivated the v6 XY-drift termination. |

**Common failure signature (v5/v6/v7):** survival plateaus 86–92%; falls cluster at
**13–18s and 25–36s** where **ankle motors saturate (50 Nm hard limit)**. This is the
target of the whole revamp (Agents B + D + F).

**Note:** trained motion `.npz` files are gitignored / live on the (now-deleted) box;
Agent B/F must regenerate or pull them and record sha256 on use.

## Repaired reference motions (Agent B — motion feasibility)

Every repaired motion gets a new sha256 and a row here; the source is NEVER
overwritten. Feasibility = the ankle-strategy dynamic pass (`pipeline/motion_dynamics.py`,
mj_inverse on CPU; ankle demand = F_z·‖ZMP−CoM‖, speed-derated limit, ankle capped
at 40 Nm). Scorecards + raw JSON/NPZ under `experiments/motion_feasibility/`.

| motion (file + sha256) | derived from | repair applied | ankle max (Nm) | ankle p95 | % frames > 40 Nm | style sim | scorecard |
|---|---|---|---|---|---|---|---|
| `thriller_g1_clean.csv` (`d9e4fc2dc39fbdbc`) | source retarget | — (SOURCE) | **173.6** | 102.3 | 47.5% | 1.000 | `experiments/motion_feasibility/thriller_g1_clean_scorecard.json` |
| `thriller_g1_repaired.csv` (`0d3ffc28492b5e50`) | thriller_g1_clean | **global slowdown 2.5×** (pure; music stays synced under uniform time-stretch) | **39.4** ✅ | 22.3 | **0.0%** ✅ | **0.999** | `experiments/motion_feasibility/thriller_g1_repaired_scorecard.json` |

**Global-slowdown sweep (torque ∝ 1/T², confirmed):** factor 1.0→p95 102 / 47.5% over;
1.3→69 / 25%; 1.5→55 / 15%; 1.7→45 / 7%; 2.0→34 / 2.3%; **2.5→22 / 0%**; 3.0→16 / 0%.
Mildest factor clearing the *majority* ≈ 1.7×; clearing *everything* (≤40 Nm) ≈ 2.5×.
The 2.5× number is a CONSERVATIVE upper bound (pure ankle-strategy; hip-strategy
substitution would let a milder factor suffice). Raw: `tools/motion_repair.py --sweep`.

**Dynamic-pass validation (the key check):** the pass lights up exactly at the
predicted fall beats — sharp spike at **15–18 s** (peak 233 Nm on the ankle-penalty
deploy motion) and sustained high demand across **24–45 s** (the 25–36 s beat +
tail), with quiet standing reading ~0.2 Nm (correct baseline). Raw:
`experiments/motion_feasibility/*_dyn.json`.

**Distinct un-fixed defect (feeds the retarget/grounding work, NOT torque):** the
Thriller reference FLOATS the lower foot ~0.10 m in **78% of frames**. Global
slowdown fixes torque but not grounding; `motion_triage.py` correctly still tags the
repaired motion as a source error on the floaty-feet axis. Fix belongs in
grounding/retarget (per-contact grounding + `retarget_gvhmr.dof_aware_postprocess`).

## New runs (appended by agents)

_(none yet — Agent F appends v8 here)_
