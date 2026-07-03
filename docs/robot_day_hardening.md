# Robot-day hardening — first-contact gotchas (2026-07-04 night)

Hardened docs/ROBOT_DAY_PLAN.md + docs/ROBOT_DAY_CHECKLIST.md and added a
joint-calibration helper, against the real first-contact risks (robot's first-ever
learned-policy run; controller never installed on PC2).

Added:
1. **MORNING START HERE** block at the top of the plan — ordered, skip-proof.
2. **Stage 0a — Network + PC2 controller install** (was entirely missing): carrier/ping/ssh
   checks, `01_pc2_install.sh` flow, and the no-internet-on-robot-LAN fallback
   (`docker save | ssh 'docker load'`). Flagged as the likely time-sink; gated.
3. **Joint-calibration check** (`deploy/check_joint_calibration.py`, new): reads LowState via
   unitree_sdk2py, compares standby joints to policy_meta `default_joint_pos`, flags any joint
   off > threshold (default 8°). A real fall risk if the real zeros differ from sim. NO-GO
   blocks running the policy. Added to Stage 0 gate + checklist. (Runs on the day, with the
   robot; degrades to a clear message if the SDK env/LAN isn't set up.)
4. **Initial-pose match** note — the 2.5 s ramp assumes the robot is at the standby default.
5. **DLIO state-estimator sanity** as a gantry step — base_lin_vel≈0 is expected/OK on the
   gantry; a diverging estimator reading is a ground-free blocker.
6. **First 30 minutes troubleshooting table** — top ~9 failure modes → immediate action.
7. Corrected the checklist's stale "must be show-ready or STOP" (gantry accepts sub-99%;
   only ground-free needs ≥99% or a conscious override) and fixed the command flow to the
   staged model.

Scripted: joint-calibration + PC2-install (both scripted; install has a manual fallback).
Manual (needs the robot): the actual damping test, estimator read, and stage attestations.
Verified: py_compile OK, test suite green (192 passed), shellcheck shows only pre-existing
SC1091 info on unchanged scripts.
