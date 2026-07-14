# GreenNode training box — HARD constraints & migration checklist

**Read this before touching a fresh box.** Every item below cost real debugging time
(mostly the 2026-07-14 migration). The recipes are fine; the *environment* is the trap.

## The box
GreenNode Notebook, 1× RTX 4090 (24 GB), **compute-only** image (PyTorch 2.5.1 / CUDA 12.4).
Provision with `cloud/00_bootstrap.sh` then `cloud/20_training.sh mjlab`. Config to create
one: `docs/GREENNODE_SETUP.md` / `docs/BOX_RECREATE_RUNBOOK.md`.

## Hard constraints (each one broke us; do not relearn)

1. **Deps MUST be pinned — `pip install mjlab` is a trap.** It leaves deps unpinned and
   re-resolves to newest on every box. Newest = **mujoco-warp 3.10.0.2 / warp-lang 1.15.0**,
   which **device-side-assert (CUDA 710) at the first env reset**. Known-good is
   **mujoco-warp 3.10.0.1 + warp-lang 1.14.0 + torch 2.11.0+cu128**. `20_training.sh` now
   installs from `cloud/env_lock/requirements.lock.txt` (exact 143 pkgs) and verifies the
   versions. If you ever reinstall by hand, use the lock.

2. **torch must be the cu128 (CUDA 12.8) build.** The default index gives `+cu130` (CUDA 13),
   which Warp 1.14 can't interop with → **illegal memory access (CUDA 700)**. Install torch
   with `--index-url https://download.pytorch.org/whl/cu128`.

3. **Training must run WITHOUT `MUJOCO_GL`.** An EGL GL context collides with Warp's CUDA
   context → illegal memory access at the 4096-env reset. Set `MUJOCO_GL=egl` **only** for
   the verify chain (gap_check/heldout, which render). The curriculum scripts already do this
   split; when launching by hand, `unset MUJOCO_GL` for training.

4. **This image cannot render headless.** EGL has no `PLATFORM_DEVICE`; osmesa's PyOpenGL is
   broken. So the box can **train but not make a video**. For an honest render:
   `cloud/dump_v6_traj.py` rolls the policy in real mjlab physics and dumps a qpos CSV (no GL),
   then `pipeline/playback_csv.py` renders it **on the laptop** (which has working GL).
   The laptop *menagerie* sandbox (`tools/sim_studio`) uses different dynamics and **lies**
   (shows the policy falling) — never trust it for pass/fail; trust the mjlab `gap.json`.

5. **Block storage is ephemeral; only the Network Volume persists.** `/workspace` (the mounted
   Network Volume) survives Stop/Delete of the instance; everything else is wiped. **KEEP the
   Network Volume `g1dance-data` on teardown** → next box is a 10-min restore, not a rebuild.

6. **Image ships no tmux / ffmpeg** → `00_bootstrap.sh` installs static builds into `$NB/bin`.

7. **tyro bool flags need a value:** `--video True`, `--no-terminations True` (bare `--video`
   errors "expected 1 value").

8. **Billing runs creation→deletion; idle still bills** (~18k VND/h ≈ $0.72/h). **Delete the
   INSTANCE** in the console (Stop does not stop the meter). Keep the Network Volume.

## Known-good versions (from a working box, 2026-07-14)
| package | version | why it matters |
|---|---|---|
| mujoco | 3.10.0 | |
| **mujoco-warp** | **3.10.0.1** | 3.10.0.2 device-asserts at reset |
| **warp-lang** | **1.14.0** | 1.15.0 device-asserts at reset |
| **torch** | **2.11.0+cu128** | cu130 → illegal memory access |
| rsl-rl-lib | 5.4.0 | |
| numpy | 2.4.6 | |

Full lock: `cloud/env_lock/requirements.lock.txt`.

## Migration checklist (bring a box up reliably)
1. **Reuse** Network Volume `g1dance-data` (don't recreate). Create the notebook per
   `BOX_RECREATE_RUNBOOK.md`; SSH key = `.secrets/greennode_rsa`.
2. Update `.secrets/cloud.json` host/port; `ssh-keygen -R "[host]:port"`.
3. Push `cloud/*` + the motion npz/csv + `.secrets/wandb.key` → `$NB/.wandb_key`.
4. `bash 00_bootstrap.sh` then `bash 20_training.sh mjlab`
   (installs from the lock, then **smoke-tests the GPU stack**).
5. Confirm `reports/training_stack.json` shows `mjlab_ready` (NOT `smoke_failed`).
6. Launch `run_attempt*.sh` — its preflight re-runs `--selfcheck` **and** the smoke gate
   before spending the ~3 h. If the smoke test fails, reinstall from the lock; do not run.

## The one rule that saves the most time
**Freeze, don't rebuild.** Trust anything measured in **mjlab** (`gap.json`, heldout);
distrust anything simulated/rendered on the **laptop**. Change the environment only on
purpose, and re-capture the lock (`pip freeze`) after any deliberate change.
