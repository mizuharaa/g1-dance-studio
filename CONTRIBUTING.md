# Contributing

Internal project. These conventions exist because breaking them has already cost
real days and real money — each one traces to a documented incident.

## Ground rules

1. **Read `docs/PROJECT_STATE.md` before changing anything.** It is the single
   source of truth (mission, decision log, next actions) and must be updated after
   every meaningful step. Sessions and people are stateless; that file is not.
2. **Measurement discipline.** Never label a finding decisive without an
   independent cross-check. Every measurement script is committed **together with
   its raw output** (`experiments/` or `data/telemetry/`) so load-bearing numbers
   have durable provenance. (A single mis-indexed sim readout once drove a full
   day of wrong conclusions.)
3. **Never modify `~/robot/`.** That is the working teleop setup — read-only
   reference.
4. **Robot safety is non-negotiable.** No low-level commands to the robot unless
   the motion passed sim verification AND a human confirmed the robot is secured
   (gantry/clear space, damping remote in hand). Deploys require a typed
   confirmation. The G1 has **no hardware e-stop**.
5. **GPU cost hygiene.** GreenNode bills from box creation to **deletion** —
   stopping does not pause billing. Pull artifacts, then delete. Preflight
   (`--selfcheck` + the 64-env smoke test) before every long run.

## Environment

- Laptop: conda env `g1dance` (Python 3.10) — UI, motion tools, tests.
- Training: isolated venv from `cloud/env_lock/requirements.lock.txt`. The pins
  are load-bearing: `mjlab==1.5.0`, `mujoco-warp==3.10.0.1`, `warp-lang==1.14.0`,
  torch cu128. Unpinned installs have CUDA-crashed at env reset twice.
- `MUJOCO_GL=egl` only for rendering/verification — it breaks Warp training.
- No system ffmpeg on the laptop: use `imageio_ffmpeg.get_ffmpeg_exe()`.

## Code conventions

- Python: match the surrounding style; keep modules SDK-free where they are today
  (e.g. `pipeline/deploy_guards.py` is deliberately pure).
- Frontend: `cd ui/frontend && npm run build` must pass; the checked-in `dist/`
  is what the desktop app serves — rebuild it in the same commit as `src/` changes.
- Tests: `./scripts/run_tests` (pytest). New guards/pipeline logic ships with tests.
- Checkpoints sort **numerically** (`model_500` < `model_3999`); export the **best**
  checkpoint via `cloud/pick_checkpoint.py`, never blindly the last.
- Motions are immutable: repairs write a new file with a new SHA-256 and a row in
  `experiments/REGISTRY.md`; never overwrite a source motion.

## Commits & experiments

- Commit early and often; messages describe pipeline progress, not file diffs.
- Every training run appends to `experiments/REGISTRY.md`: motion SHA, recipe git
  hash, gate config, best checkpoint, raw gate output path, calibrated estimate.
- Never report a raw sim percentage alone — pair it with the calibrated
  real-world estimate once the gate calibration lands.

## Reporting issues

Use the templates in `.github/ISSUE_TEMPLATE/`. For robot incidents, include the
run log path (`data/shows/<run>/run.log`), telemetry npz, and whether the damping
remote was used.
