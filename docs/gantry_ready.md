# Robot-day / gantry readiness — deploy path fixed + staged (2026-07-04)

## Deploy-path blockers fixed (production audit)
- gen_config.py consumed only `exam_*.json` and read `nominal.duration_s` (KeyError on
  mjlab verdicts) → now finds `*verdict*.json` (heldout_verdict.json) in --exam-dir AND
  the policy dir (or explicit --verdict), and derives duration from the motion CSV.
- Verdict motion-sha seam: mjlab_verify now signs `motion_sha256` = the DEPLOYABLE csv
  and records `motion_npz_sha256` = the evaluated npz (provenance). Consumers bind to the
  csv, so a real verdict can authorize. Thriller's verdict regenerated bound to
  thriller_deploy.csv.

## Staged full-day authorization (gates mandatory between stages)
- gen_config `--gantry`: builds a scope=gantry-only bundle from a SIGNED, bound, sub-99%
  verdict (feet-off-ground only). Strictly more than --rehearsal, strictly less than the
  ground/show gate (which still requires a >=99% pass — unchanged, not weakened).
- 10_gantry_test.sh stages: gantry -> ground-tethered -> ground-free -> push-test, each a
  distinct typed phrase + entry gate. ground-free HARD-gates on gantry+tethered passed,
  --kill-damping-confirmed, --estimator-verified, and a full (>=99%) bundle OR a loud
  --informed-override. ground/push refuse a gantry-only bundle.
- 02_push_bundle.sh accepts full OR gantry-only (loud label), still refuses rehearsal.

## Safety build details wired
- Bundle built from **thriller_deploy.csv** (2.5s standby ramp — avoids the ~39deg
  activation lurch). Never the raw show clip.
- Bundle carries **policy_meta.json** (SIM PD gains, ζ=2 overdamped), hash-pinned; the
  start script REFUSES to leave damping unless USE_SIM_GAINS=1 + SIM_GAINS_LOADED marker
  (robot must load THESE gains, not stock Unitree — fall risk).
- Telemetry: 10_gantry_test mounts a telemetry dir; deploy/pull_telemetry.sh pulls it.

## Staged: Thriller attempt-1 (98.4%) gantry bundle
- Preflight `--stage gantry` => GO; `--stage ground-free` => NO-GO (needs >=99%).
- Attempt-2 (>=99%) exports overnight → rebuild without --gantry to unlock ground-free.

## Docs
- docs/ROBOT_DAY_PLAN.md (full staged day), docs/ROBOT_DAY_CHECKLIST.md (one-pager).
