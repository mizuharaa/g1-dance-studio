# Twitch / limb-snapping — source motion re-prep (2026-07-13)

**Symptom:** the robot looks twitchy, limbs snapping at a fast/unnatural pace (a
"malfunction" look), most visible in the preview.

**Root cause (confirmed here):** per-frame jitter from the GVHMR landmark detection
survives retargeting as accel/jerk spikes in the deploy motion. The de-glitch filter
(`tools/motion_quality.clean_motion`) is wired into `pipeline/prep_motion` as of commit
`16f6aa7`, **but the currently deployed Thriller policy was trained on the UNFILTERED
motion** — so it still carries these spikes.

**What this dir shows:** re-prepping the raw retarget `data/motions/thriller/thriller_g1.csv`
through the now-wired filter → `data/motions/thriller/thriller_g1_clean.csv`.

| metric | before (deployed policy's motion) | after (clean) | change |
|---|---|---|---|
| jerk peak (rad/s³) | 101,701 | 4,806 | **÷21.2** |
| jerk p99 (rad/s³) | 5,276 | 1,174 | ÷4.5 |
| accel-spike frames | 67 | 4 | −94 % |
| outlier frames replaced | — | 133 | — |
| DOF RMS Δ vs raw (rad) | — | 0.0327 (≈1.9°) | sharpness kept |
| DOF p99 Δ vs raw (rad) | — | 0.0602 (≈3.4°) | sharpness kept |

Raw numbers: `reprep_metrics.json` (the exact `prep_motion.prep()` return).

**Regenerate (deterministic, laptop CPU):**
```bash
python -m pipeline.prep_motion --in data/motions/thriller/thriller_g1.csv \
                               --out data/motions/thriller/thriller_g1_clean.csv
```

**This does NOT fix the robot by itself.** The clean CSV is the INPUT for the retrain
(`tasks/AGENT_A_LATENCY_RETRAIN.md` / `docs/RETRAIN_RUNBOOK.md`): the policy must be
re-trained on it for the twitch to go away on hardware. The v5 recipe trains on this motion.
