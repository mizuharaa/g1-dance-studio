# deploy/ — robot-day kit

Everything Stage-5/6 needs, pre-staged so robot day is execution, not invention.
Read `docs/ROBOT_DAY_RUNBOOK.md` first; scripts enforce its interlocks.

| file | role | can touch the robot? |
|---|---|---|
| `lib.sh` | shared gates (env var, dry-run, ssh helper) | no |
| `gen_config.py` | bundle builder; HARD-GATED on a passing sim exam | no (laptop only) |
| `01_pc2_install.sh` | install docker image + controller repo on PC2 | software only, gated |
| `02_push_bundle.sh` | scp bundle to PC2 (integrity re-check) | files only, gated |
| `10_gantry_test.sh` | start controller in DAMPING HOLD | yes — most gated |
| `kill_now.sh` | instant software abort | yes (safety-positive) |

Interlock model, outermost to innermost:
1. `CONFIRMED_BY_HUMAN=alois` env var — set by a person, once per session.
2. Per-script explicit flags (`--yes-install`, `--yes-push`, `--arm`, ...).
3. Dry-run default everywhere; `10_gantry_test.sh` additionally demands a typed
   phrase and starts the controller in damping hold only.
4. In-bundle `LAUNCH_LINE_VERIFIED` marker — the container entrypoint refuses to
   run until the controller launch line is verified against its README on PC2.
5. The app's deploy button stays record-only; it never calls these scripts.

The pipeline gate order is: vet PASS → training converged → sim exam PASS
(`pipeline/sim_exam.py`, contract `docs/show_mode_contracts.md`) → `gen_config.py`
accepts → runbook steps on robot day.
