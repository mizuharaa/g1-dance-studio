# Training recipe research — PARTIAL (2026-07-03 night)

Status: ultracode sweep was cut off by the session limit (1 of 5 researchers finished).
Full relaunch queued (resume run wf_f06cf88b-697 — completed sweep results are cached).
What survived is load-bearing for tonight's training regardless:

## Verified facts from the reference implementation (whole_body_tracking@cd65172)

1. **Long clips need no special machinery** — the repo/paper treat 10 s episodes +
   adaptive-sampling RSI over 1 s bins as sufficient for ARBITRARY clip length.
   2–3 min dances should train with the same recipe (episode samples start points
   across the whole clip). This de-risks the product target substantially.
2. **No wall-clock numbers exist in the repo** — our benchmark IS the calibration.
   (Paper: arxiv 2508.08241 — fetch during full sweep.)
3. `adaptive_kernel_size` smoothing defaults to 1 (identity) in all shipped configs —
   not active in baseline; don't cargo-cult it.
4. **No action filtering / torque-limit reward / energy penalty / PD-gain
   randomization in training** — sim2real rests on the analytic armature actuator
   model + 4 DR terms. Any action smoothing lives deploy-side
   (motion_tracking_controller), NOT in training configs.
5. **train.py has NO local-file path for motions** — hard-requires the W&B registry
   artifact (--registry_name, downloads motion.npz). For mjlab or offline use:
   patch or mirror through the registry (ours: wandb-registry-motions, exists).

## Still owed by the full sweep
Reward weights baseline + deltas for dance; mjlab config surface + G1 gotchas;
4090 wall-clock expectations; sim2real checklist; ranked long-dance strategies.
