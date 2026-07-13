# AGENT A — Latency-curriculum retrain (v5) on a fresh GPU box

**Round:** 2026-07-13. **Owner:** manual agent **+ human** (box creation is console-only).
**Where it runs:** ☁️ GreenNode GPU box (RTX 4090). **Branch:** `train/latency-curriculum-v5`.
**Needs:** `.secrets/` (RSA SSH key, GreenNode login, W&B key), the clean motion from
Agent-work Lane 2 (`data/motions/thriller/thriller_g1_clean.csv` — already on `main`).

## Goal
Produce a Thriller policy that is robust to the real **40–80 ms** latency AND trained on the
**de-glitched** motion — fixing both the ~45 s drift/buckle and the twitchy limb-snapping.

## Why this exists (do not skip the reading)
- The last retrain (`lat80`) **FAILED**: survival 0.000, drift 2–7 m
  (`data/telemetry/latency_retrain_20260710/RESULT.md`). Cause: 0–80 ms delay from step 0 destroyed
  station-keeping.
- The v5 recipe fixes it: **staged latency curriculum** (0–20 → 0–50 → 0–60 ms via resume) +
  arm-fidelity terms + root-pos weight 1.0. Recipe is committed: `cloud/sim2real_task_v5.py`,
  `cloud/train_v5_curriculum.sh`.

## Steps
**Follow `docs/RETRAIN_RUNBOOK.md` exactly.** Summary:
1. Convert the clean CSV → training `.npz` (§1 of the runbook).
2. Create + provision the box (`docs/BOX_RECREATE_RUNBOOK.md`; **RSA key, TCP 22**). scp the npz.
3. `MOTION=$NB/motions/thriller_clean.npz bash cloud/train_v5_curriculum.sh` in tmux (~10k iters, ~5 h).
   - ⚠️ **Verify each curriculum stage actually RESUMED** (reward continues, doesn't reset) — the
     mjlab resume flag names are unverified (`train_v5_curriculum.sh` header warns about this).
4. Verify chain: export → `sim_gap_check` → `heldout_eval` ×3.
5. Pull artifacts, md5-verify, sign with `pipeline/mjlab_verify.py`, attach + promote in the app.
6. **DELETE THE BOX** (billing; owner emphatic — stop ≠ delete).

## GATE (all must pass — else do NOT deploy, iterate the recipe)
- `gap.json`: survival passes at **40 ms + push**; inspect 60/80 ms lines.
- **nominal root drift < 1 m** (the lat80 failure mode).
- held-out survival **≥ 99 %** across seeds 90001 / 90011 / 90021.

## Deliverable back to `main`
`data/policies/thriller_v5fid/` (policy.onnx + policy_meta.json + gap.json) + signed
`heldout_verdict.json`. Update `PROJECT_STATE.md`. Then hand off to **Agent C** (hardware).

## Budget
~5 h box ≈ **~90k VND**. Owner cap for the window: 1.5M VND. Pause + report if it would exceed.

## File boundaries
`cloud/` (recipe already there), `data/motions/thriller/`, `data/policies/thriller_v5fid/`,
`PROJECT_STATE.md`, `logs/jobs.md`. **Do NOT** touch `pipeline/deploy_runtime.py` (that's Agent B).
