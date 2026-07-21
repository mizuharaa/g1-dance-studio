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


def rebuild_term_major(term_history: dict[str, np.ndarray], term_order) -> np.ndarray:
    """Flatten ``{term: [history, width]}`` in deploy's exact actor order."""
    parts = []
    for name in term_order:
        if name not in term_history:
            raise KeyError(f"missing history for actor term {name!r}")
        history = np.asarray(term_history[name])
        if history.ndim < 2:
            raise ValueError(
                f"actor term {name!r} history must be [history, dim...], "
                f"got shape {history.shape}"
            )
        parts.append(history.reshape(-1))  # oldest -> newest within each term
    if not parts:
        raise ValueError("actor term order is empty")
    return np.concatenate(parts)


def _rebuild_frame_major(term_history: dict[str, np.ndarray], term_order) -> np.ndarray:
    histories = [np.asarray(term_history[name]) for name in term_order]
    lengths = {history.shape[0] for history in histories}
    if len(lengths) != 1:
        raise ValueError(f"actor terms have inconsistent history lengths: {sorted(lengths)}")
    return np.concatenate([
        np.concatenate([history[frame].reshape(-1) for history in histories])
        for frame in range(histories[0].shape[0])
    ])


def _api_unavailable(path: str, exc: BaseException) -> int:
    print(
        f"API-UNAVAILABLE: {path}: {type(exc).__name__}: {exc}",
        file=sys.stderr,
    )
    return 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--task-module", default="")
    ap.add_argument("--motion-file", required=True)
    ap.add_argument("--num-envs", type=int, default=2)
    a = ap.parse_args()

    if a.task_module:
        try:
            importlib.import_module(a.task_module)   # registers the custom task
        except Exception as exc:  # noqa: BLE001 — box API gate must degrade loudly
            return _api_unavailable(
                f"importlib.import_module({a.task_module!r})", exc
            )
    try:
        from mjlab.envs import ManagerBasedRlEnv
    except Exception as exc:  # noqa: BLE001
        return _api_unavailable("mjlab.envs.ManagerBasedRlEnv", exc)
    try:
        from mjlab.tasks.registry import load_env_cfg
    except Exception as exc:  # noqa: BLE001
        return _api_unavailable("mjlab.tasks.registry.load_env_cfg", exc)

    try:
        cfg = load_env_cfg(a.task)
    except Exception as exc:  # noqa: BLE001
        return _api_unavailable(f"load_env_cfg({a.task!r})", exc)
    try:
        cfg.scene.num_envs = a.num_envs
    except Exception as exc:  # noqa: BLE001
        return _api_unavailable("cfg.scene.num_envs", exc)
    try:
        cfg.commands["motion"].motion_file = a.motion_file
    except Exception as exc:  # noqa: BLE001
        return _api_unavailable('cfg.commands["motion"].motion_file', exc)
    try:
        actor_cfg = cfg.observations["actor"]
    except Exception as exc:  # noqa: BLE001
        return _api_unavailable('cfg.observations["actor"]', exc)
    try:
        hist = int(actor_cfg.history_length or 0) or 1
    except Exception as exc:  # noqa: BLE001
        return _api_unavailable('cfg.observations["actor"].history_length', exc)
    try:
        flatten_history = bool(actor_cfg.flatten_history_dim)
    except Exception as exc:  # noqa: BLE001
        return _api_unavailable('cfg.observations["actor"].flatten_history_dim', exc)
    if hist > 1 and not flatten_history:
        print(
            '!! FAIL: cfg.observations["actor"].flatten_history_dim is false; '
            "the actor does not have the deploy flat-history contract.",
            file=sys.stderr,
        )
        return 1
    try:
        env = ManagerBasedRlEnv(cfg)
    except Exception as exc:  # noqa: BLE001
        return _api_unavailable("ManagerBasedRlEnv(cfg)", exc)
    try:
        obs, _ = env.reset()
    except Exception as exc:  # noqa: BLE001
        return _api_unavailable("ManagerBasedRlEnv.reset()", exc)

    try:
        actor = obs["actor"] if isinstance(obs, dict) else obs
        flat = actor[0].detach().cpu().numpy().reshape(-1)
    except Exception as exc:  # noqa: BLE001
        return _api_unavailable('env.reset()[0]["actor"][0]', exc)
    try:
        om = env.observation_manager
    except Exception as exc:  # noqa: BLE001
        return _api_unavailable("env.observation_manager", exc)
    group = "actor"
    try:
        terms = list(om.active_terms[group])
    except Exception as exc:  # noqa: BLE001
        return _api_unavailable('env.observation_manager.active_terms["actor"]', exc)

    # Rebuild the term-major oldest->newest layout our HistoryStacker produces from
    # the manager's own per-term history buffer, and byte-compare to the env's flat.
    try:
        buffers = om._group_obs_term_history_buffer[group]
    except Exception as exc:  # noqa: BLE001
        return _api_unavailable(
            'env.observation_manager._group_obs_term_history_buffer["actor"]', exc
        )
    term_history: dict[str, np.ndarray] = {}
    for name in terms:
        path = (
            'env.observation_manager._group_obs_term_history_buffer'
            f'["actor"][{name!r}].buffer[0]'
        )
        try:
            # CircularBuffer.buffer is [batch, history, dim...] in chronological
            # order. Select env 0; indexing the CircularBuffer itself is not valid.
            term_history[name] = buffers[name].buffer[0].detach().cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            return _api_unavailable(path, exc)
    try:
        expect = rebuild_term_major(term_history, terms)
    except Exception as exc:  # noqa: BLE001
        return _api_unavailable("rebuild_term_major(actor history)", exc)

    lengths = {history.shape[0] for history in term_history.values()}
    if lengths != {hist}:
        print(
            f"!! FAIL: live history lengths {sorted(lengths)} != cfg history_length "
            f"{hist}. Do NOT deploy.",
            file=sys.stderr,
        )
        return 1
    per_frame = sum(int(np.prod(history.shape[1:])) for history in term_history.values())
    print(f"task={a.task}  n_hist={hist}  per_frame={per_frame}  flat={flat.shape[0]}")
    print(f"terms (in order): {terms}")

    ok = expect.shape == flat.shape and np.array_equal(expect, flat)
    if ok:
        print(f"PASS: term-major oldest->newest layout matches the live actor obs "
              f"({flat.shape[0]} dims). HistoryStacker is deploy-correct.")
        return 0
    # diagnose the most common transposition to make the failure actionable
    try:
        frame_major = _rebuild_frame_major(term_history, terms)
        hint = "FRAME-MAJOR" if np.array_equal(frame_major, flat) else "UNKNOWN"
    except (KeyError, ValueError):
        hint = "UNKNOWN"
    print(f"!! FAIL: our term-major layout does NOT match the live obs; live layout "
          f"looks {hint}. Do NOT deploy — reconcile HistoryStacker before signing.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
