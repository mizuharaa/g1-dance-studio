# Safety review remediation — summary

Remediates docs/safety_review_findings.md (33 confirmed). This pass closed the two
CRITICALs and the load-bearing chain-of-trust + human-factors findings in software,
and reclassified the hardware-only findings into mandatory robot-day gates.

## Central fix — authenticated, content-derived authorization (new pipeline/exam_verdict.py)
The whole exam→readiness→deploy chain no longer trusts the self-declared `"verdict"`
string. `authorize()` requires BOTH a valid HMAC signature (key in `.secrets/`, written
by sim_exam, not by the web process) AND a pass re-derived from phase CONTENTS
(nominal.pass ∧ push.pass ∧ repeat.pass ∧ clean==runs≥3 ∧ push force ≥ floor), rejecting
empty/partial phases. Closes/relies-on: **#0 (critical)**, #7, #19, #21, #23, #26, #32.

## Code-fixed (with regression tests in tests/test_safety_remediation.py — 12 tests, all green; full suite 83 passed)
| # | Sev | Fix |
|---|-----|-----|
| 0 | crit | gen_config re-derives verdict + verifies signature; hand-edit/fabrication rejected |
| 1 | crit (partial) | kill_now.sh SIGTERM→SIGKILL grace window; false "falls to damping" claim removed; empirical check reclassified to runbook 3a |
| 2,10 | high | kill_now.sh no longer requires CONFIRMED_BY_HUMAN — fires from any shell |
| 4 | high | sim_exam absolute fall gate (world tilt >0.7rad / torso z <0.45m) independent of reference |
| 5,18 | high | sim_exam nominal PASS now requires tracking-quality floor (mean joint & anchor err), not just upright |
| 9 | high | incident/aborted outcome force-demotes dance + resets clean streak (server.py) |
| 13,29 | high/med | BATTERY_FLOOR_PCT=30 enforced in complete_step |
| 21 | med | honest verdict: "incomplete" (never "pass") + exit code 2 when phases skipped |
| 22 | med | push force floor (MIN_PUSH_FORCE_N=150) enforced in exam + authorize |
| 3,8,20 | high/med | start script asserts START_MODE=damping from hashed controller.env; gantry script verifies it on PC2 |
| 8,19 | high/med | every bundle file hash-pinned in manifest (files_sha256) |
| 16 | med | gantry script requires TTY for typed phrase (no pipe/heredoc) |
| 17,30 | med | gantry vs ground split via --stage with distinct typed phrases + preconditions |
| 31 | low | --dance allowlist ^[A-Za-z0-9_-]{1,64}$ in gen_config + gantry script |
| 32 | low | full 64-hex sha256 everywhere (sim_exam, gen_config) |
| 24,25,27 | med | policy_sha256 field on Dance + authorize() binds verdict to policy/motion sha |

## Reclassified to mandatory robot-day gates (docs/ROBOT_DAY_RUNBOOK.md new Step 3a + abort ladder; docs/OPERATOR_MANUAL.md)
- **#1** SIGKILL→damping must be measured on the gantry before ground use.
- **#11,#12** on-Jetson comms-loss deadman + NaN/overrun→damping must be verified (kill_now rides the SSH link it backstops); unmitigated → gantry-only / short runs.
- **#14** corrected the false "hardware e-stop cuts motor torque" claim in the manual and runbook — this tether-free G1's only stop is the remote's B-damping (not a power cut); power switch is the sole guaranteed torque removal.

## Not yet done (documented residual — recommend a follow-up pass)
- **#6** full domain-randomization + de-correlated seeds + obs-noise + latency in the exam
  (the exam is still optimistic vs real sensors; seeds are only jittered). Medium.
- **#23/#24 server-side ingestion:** the `/api/dances/{id}/sim-runs` endpoint still accepts
  a bare `passed` bool; it should ingest the signed verdict and verify sha binding. The
  `authorize()` primitive and `policy_sha256` field are in place for this — wiring the
  endpoint is the remaining step. Medium.
- **#28** per-dance file lock around load-mutate-save (concurrency). Low (single-operator).
- **#3a heartbeat supervisor** is a hardware/controller task, not laptop software.

## Merge-readiness
Worktree touches pipeline/exam_verdict.py (new), pipeline/sim_exam.py, pipeline/shows.py,
ui/server.py, deploy/*, docs/*, tests/test_safety_remediation.py. Full suite green,
shellcheck clean. No PROJECT_STATE.md edits (main owns it).
