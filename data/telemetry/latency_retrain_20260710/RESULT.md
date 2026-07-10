# Latency-robust retrain — RESULT: FAILED verification (2026-07-10)

run-name `train-thriller_lat80-2607` (box nb-9c7ba766). Task Sim2Real, 4096 envs, 5000 iters,
latency DR **0-80 ms** (was 0-20 ms) + root-pos weight 1.0. See PROJECT_STATE 2026-07-10.

## Verdict: WORSE than the ankle policy. Do NOT deploy.
gap_check (gap_check.json): survival **0.000 in ALL 11 conditions**, incl. nominal (0 ms).
drift_max **2.2-7.1 m** everywhere (ankle policy was 0.46 m nominal). BUT root-relative
tracking rr_mpkpe **0.079** (crisp) — it dances well, it just can't hold station.

## Root cause (cross-checked — gap_check AND training curve agree)
The 0-80 ms latency DR was too aggressive for 5000 iters. Training never converged on
station-keeping: `motion_global_root_pos` reward stalled at **0.05**, mean episode length
**~4.6 s** of a 56 s dance (early `anchor_pos`/`ee_body_pos` terminations). The policy kept
the dance and dropped drift control. Widening delay bluntly traded away station-keeping.
The NEW 40 ms gap gate correctly REFUSED it (the old 20 ms gate would have passed it).

## Recommended next recipe (not yet run)
Curriculum: ramp delay 0 -> ~60 ms over training (learn dance+station first, then adapt);
OR moderate range 0-50 ms (covers the real added-latency band; 80 ms over-states it since
sim PD already models mechanical lag) + ~10k iters + a stronger root-position penalty.
