# Training prep — attempt 11 (v14), for the 2026-08-06 box session

> Naming: the user calls this "v13 training" — **v14 supersedes v13** (it contains
> v13's leg anti-chatter term; v13's own artifacts died with box 40711). The run
> is `run_attempt11.sh`, RUN_NAME `train-thriller_v14style-MMDD`.

## What this run must beat (hardware-measured 2026-08-05, same robot/floor/day)

| Metric | v12 (hardware) | anchor (hardware) | v14 sim bar |
|---|---|---|---|
| Leg amplitude achieved/commanded | 58% | 64% | **≥ 0.80 (gated)** |
| Survival at 60 ms cmd delay | 73% (sim) | n/a | **≥ 95% (gated)** |
| Clean survival / drift / ankle | v12 passed all | — | unchanged bars (22/25 ankle, drift p95 1.5) |

## The audit findings this run carries (2026-08-05)

1. **Latency DR was a myth at the top**: delay lag counts 5 ms physics steps; the
   old final stage's MAX_LAG=12 = 60 ms — *zero margin* at the robot's real
   40–60 ms. Ladder now 4/1 → 8/2 → 12/3 → **16/4 (80 ms cmd + 80 ms obs)**.
2. **The gate never measured leg amplitude** — the one number that matched
   hardware. `sim_gap_check` now records `amp_ratios` (achieved/reference p95
   amplitude, leg + arm groups) per condition and gates `leg_amp >= G1_GATE_LEG_AMP_MIN`
   (0 = report-only for old-run comparability; launcher sets 0.80).
3. New gate row: `survival >= 0.95 [cmd_delay60ms]` (env `G1_GATE_DELAY60_SURVIVAL_MIN`).
4. v14 recipe (`sim2real_task_v14.py`): lean shaping (drops ankle_torque_barrier,
   stance terms, saturation — HoloSoma evidence + our audit) + feet/wrist keypoint
   termination 0.25 m + keeps anti-chatter, leg tracking, 40 Nm ankle clamp DR.

## Box session runbook (fresh box — all old boxes deleted)

1. Provision per `docs/BOX_RECREATE_RUNBOOK.md` (RTX 4090 notebook, Network
   Volume g1dance-data if it still exists; else fresh volume).
2. Push (laptop):
   `scp -P <PORT> -i .secrets/greennode_rsa cloud/*.py cloud/*.sh root@<IP>:/workspace/notebook-data/cloud/`
   `scp -P <PORT> -i .secrets/greennode_rsa -r data/motions/thriller/v12_bundle root@<IP>:/workspace/notebook-data/motions/`
3. On the box: `cd /workspace/notebook-data && nohup bash cloud/run_attempt11.sh > attempt11.out 2>&1 &`
   (selfcheck + 64-env smoke run BEFORE the spend; bundle manifest verified first.)
4. Optional second box / second run: `G1_V14_LEAN=0` = keypoint-termination-only arm.
5. Pull + sign + **DELETE the box** (billing runs creation→deletion).

## After the run — deploy readiness gaps to close BEFORE hardware

- v14 exports the same 770-dim estimator-free contract as v12 → `ground-run`
  mode + HistoryStacker already validated on hardware (2026-08-05 runs).
- Contact guard is upright-gated now; entry gate passed at 26 Nm standing.
- Robot-day: killswitch terminal (L2+B software layer), camera repositioned
  (full body in frame), anchor available as A/B on the same floor.
