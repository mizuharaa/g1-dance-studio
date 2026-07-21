"""BOX-ONLY smoke test for AUDIT FIX E: prove the deploy/preview HistoryStacker
layout matches what the trained mjlab actor actually consumes.

Why this is a hard gate: the actor is history-stacked (154 dims/frame x N). A
TRANSPOSED flatten (frame-major instead of term-major, or newest->oldest) is
SILENT — no crash, the ONNX still runs — but it feeds the policy garbage and
would drive a fall on the real robot. pipeline.deploy_runtime.HistoryStacker and
tools.sim_sandbox both assume mjlab's flatten_history_dim=True layout: per TERM
(in obs order) that term's N frames oldest->newest, then terms concatenated.
This script confirms that against the live env before any 770-dim deploy build.

Run on the box (mjlab installed):
  $NB/envs/mjlab/bin/python $NB/cloud/verify_obs_layout.py \
      --task Mjlab-Tracking-Flat-Unitree-G1-S2R-V11 --task-module sim2real_task_v11 \
      --motion-file $NB/motions/thriller_v12_100.npz

PASS => the deploy contract is validated; sign the build. FAIL => do NOT deploy
the 770-dim policy until the layout is reconciled.
"""
from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--task-module", default="")
    ap.add_argument("--motion-file", required=True)
    ap.add_argument("--num-envs", type=int, default=2)
    a = ap.parse_args()

    if a.task_module:
        importlib.import_module(a.task_module)   # registers the custom task
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(a.task)
    cfg.scene.num_envs = a.num_envs
    cfg.commands.motion.motion_file = a.motion_file
    env = ManagerBasedRlEnv(cfg)
    obs, _ = env.reset()

    actor = obs["actor"] if isinstance(obs, dict) else obs
    flat = actor[0].detach().cpu().numpy().reshape(-1)
    om = env.observation_manager
    group = "actor"
    terms = om.active_terms[group]
    dims = om.group_obs_term_dim[group]
    hist = int(getattr(cfg.observations.actor, "history_length", 0) or 0) or 1
    per_frame = sum(int(np.prod(d)) for d in dims)

    print(f"task={a.task}  n_hist={hist}  per_frame={per_frame}  flat={flat.shape[0]}")
    print(f"terms (in order): {list(terms)}")

    # Rebuild the term-major oldest->newest layout our HistoryStacker produces from
    # the manager's own per-term history buffer, and byte-compare to the env's flat.
    # The manager stores each term's history; we read it the same way and stack.
    try:
        buf = om._group_obs_term_history_buffer[group]   # term -> [envs, hist, dim]
    except Exception:
        print("!! cannot access history buffer on this mjlab build — inspect om internals",
              file=sys.stderr)
        return 3

    parts = []
    for name in terms:
        h = buf[name][0].detach().cpu().numpy()          # [hist, dim]
        for f in range(h.shape[0]):                      # oldest -> newest
            parts.append(h[f].reshape(-1))
    expect = np.concatenate(parts)

    ok = expect.shape == flat.shape and np.allclose(expect, flat, atol=1e-5)
    if ok:
        print(f"PASS: term-major oldest->newest layout matches the live actor obs "
              f"({flat.shape[0]} dims). HistoryStacker is deploy-correct.")
        return 0
    # diagnose the most common transposition to make the failure actionable
    frame_major = np.concatenate([
        np.concatenate([buf[n][0].detach().cpu().numpy()[f].reshape(-1) for n in terms])
        for f in range(hist)])
    hint = "FRAME-MAJOR" if np.allclose(frame_major, flat, atol=1e-5) else "UNKNOWN"
    print(f"!! FAIL: our term-major layout does NOT match the live obs; live layout "
          f"looks {hint}. Do NOT deploy — reconcile HistoryStacker before signing.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
