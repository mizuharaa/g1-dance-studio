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

## New runs (appended by agents)

_(none yet — Agent F appends v8 here)_
