# Cloud provisioning scripts (GreenNode notebook)

These run **on the GreenNode notebook instance**, not the laptop. The laptop app
ships them over the cloud transport (SSH or the Jupyter tunnel) and executes them;
they can also be run by hand in a notebook terminal.

Order:

| script | what | re-run after every instance Stop? |
|---|---|---|
| `00_bootstrap.sh` | layout under the persistent mount, tmux, env.sh | **yes** (block storage is wiped) |
| `10_gvhmr.sh` | GVHMR video→SMPL-X stack + checkpoints | yes (fast no-op if cached) |
| `20_training.sh` | Isaac Lab 2.1.0 + BeyondMimic; `bash 20_training.sh mjlab` for the fallback | yes (fast no-op) |
| `run_job.sh` | tmux job wrapper: `start/status/tail/list/stop` | n/a (tool) |

Invariants:
- everything valuable lives under `$NB_DATA` (default `/workspace/notebook-data`,
  the Network Volume mount) — the only thing that survives a Stop;
- every script is idempotent and logs preflight facts (GPU, disk, python);
- `20_training.sh` writes `$NB_DATA/reports/training_stack.json` with
  `isaac_ready` / `isaac_failed` / `mjlab_ready` so the laptop knows which
  trainer to drive;
- body models are license-gated: synced from the laptop (`data/body_models/`),
  never downloaded here.
