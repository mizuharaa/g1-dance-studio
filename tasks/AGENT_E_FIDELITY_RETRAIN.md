# AGENT E — Fidelity retrain: track the subtle moves + match the robot's real signals

**Owner: Windows-side Claude (reassigned by user 2026-07-10) + human for box creation.**
GPU: recreate the GreenNode box (`docs/BOX_RECREATE_RUNBOOK.md`; RSA key only, add TCP 22).
Budget cap: 1.5M VND (user-authorized 2026-07-10).

**STATUS 2026-07-10: recipe IMPLEMENTED, launch blocked on two inputs:**
- code: `cloud/sim2real_task_v5.py` (arm-scoped tracking terms + root-pos 1.0 + env-var
  delay caps) + `cloud/train_sim2real_v5.py` + `cloud/train_v5_curriculum.sh`
  (3-stage staged-resume curriculum: 0-20 ms/4k → 0-50 ms/+3k → 0-60 ms/+3k iters).
- blocked on: (1) `.secrets/` copied to the Windows machine (user doing), (2) Lane B
  Phase-2 feasibility motion from the Ubuntu agent (user decision: wait, train once).
- on the box, before stage 2: verify the rsl_rl resume flag names (`--help | grep -i resume`)
  — marked in the script.

## Why
Two failures compound into the "60–70 %": (1) the current policy **washes out subtle / arm
moves** (tracking reward trades them against balance); (2) the last latency retrain FAILED —
0–80 ms DR from step 0 was too blunt → **drift 2–7 m, survival 0.000**
(`data/telemetry/latency_retrain_20260710/RESULT.md`). Retrain so the policy tracks the subtle
moves AND survives real latency WITHOUT losing station-keeping.

## Recipe (fix both at once)
1. **Reward:** up-weight the motion-tracking term on ARMS / HANDS + the subtle DoFs the tester
   saw skipped (shoulders, elbows, wrists). Keep `motion_global_root_pos.weight = 1.0` so
   station-keeping doesn't collapse like lat80 did.
2. **Latency DR = CURRICULUM,** not a blunt cap: ramp delay 0 → ~50–60 ms over training (the
   measured *pure added* band; 80 ms over-states it because sim PD already models mechanical
   lag — DIAGNOSIS.md). ~10 k iters (5 k was too few for the harder task).
3. **Train on the Lane-B de-glitched + Lane-B-feasible motion** (not the raw glitchy CSV).
4. **Match received signals:** model the leg-odometry estimation noise + lag the robot actually
   feeds the policy, so sim obs ≈ robot obs (this is what "train the baseline to match what the
   robot receives" means).
5. **Verify:** `gap_check` gated at 40 ms+push (already hardened, `cloud/sim_gap_check.py`)
   AND drift < 1 m nominal; then validate in the **Lane-D sandbox** before any hardware run.

## Acceptance
- `gap_check` PASS at 40 ms+push AND nominal drift < 1 m (the lat80 failure mode gone).
- Lane-D sandbox tracking report shows subtle-move fidelity UP vs the current policy
  (the 60–70 % rises).
- **DELETE the GPU box when done.** `PROJECT_STATE.md` decision-log entry per phase.
