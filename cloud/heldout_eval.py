"""Held-out robustness eval of a trained mjlab tracking policy.

Adapted from mjlab's tasks/tracking/scripts/evaluate.py, but:
  * loads a LOCAL checkpoint + LOCAL motion (no W&B),
  * uses a HELD-OUT seed disjoint from training,
  * runs two conditions — sensor noise only, and sensor noise + external
    shoves — so we measure both clean generalization and shove-recovery,
  * writes a JSON the laptop turns into a signed verdict.

Harness v2 (2026-07-21, audit F4): conditions are built with the SAME
make_condition_cfg builder as cloud/sim_gap_check.py (no drift between the two
evaluators): explicit construction from the task's PLAY cfg — no startup DR,
RSI, or cmd/obs delay leaks from the train cfg — with the training push event
deep-copied in only for the push row. Both rows share ONE seed (paired
comparisons); each row records a CONVENTIONS §3.4 `realized` block from the
FINAL cfg. The episode horizon is EXACT (frames/fps, no +0.2 padding — past
the last frame pinned mjlab teleports survivors to frame zero and rescores
the start). Success = reached T; entry/exit handoff is out of scope here.

Output keys keep the legacy names "nominal"/"push" because
pipeline/mjlab_verify.py reads them; each row carries `honest_name`
("noise"/"noise_push") and the top level carries harness_version: 2 —
comparisons across harness versions must check it.

This is same-ENGINE held-out verification (mjlab), NOT a different-simulator
sim2sim check — the plain-MuJoCo model isn't dynamically faithful. It catches a
policy that overfits training seeds or can't take a shove; it does NOT catch
mjlab-specific physics exploitation. Robot-day (gantry-first) is the real gate.

Run on the box:
  envs/mjlab/bin/python cloud/heldout_eval.py Mjlab-Tracking-Flat-Unitree-G1 \
    --checkpoint <model.pt> --motion-file <motion.npz> --num-envs 256 \
    --seed 90001 --output-file <out.json>
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Shared harness-v2 pieces — SAME builder as the gap check (audit F4: the two
# evaluators must not drift). Pure-python; importable without mjlab.
from sim_gap_check import (  # noqa: E402
    HARNESS_VERSION,
    _motion_frames_fps,
    exact_horizon,
    extract_realized,
    make_condition_cfg,
)

# mjlab/tyro exist only in the box env; guarded so tests can import this module
# on the laptop (see tests/test_eval_harness.py). main() refuses without them.
try:
    import tyro
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.tasks.tracking.mdp import MotionCommandCfg
    from mjlab.tasks.tracking.mdp.commands import MotionCommand
    from mjlab.tasks.tracking.mdp.metrics import (
        compute_ee_position_error,
        compute_mpkpe,
    )
    from mjlab.utils.torch import configure_torch_backends

    _EVAL_IMPORT_ERROR: Exception | None = None
except ImportError as _e:  # laptop / CI: import surface only
    tyro = None
    ManagerBasedRlEnv = MjlabOnPolicyRunner = RslRlVecEnvWrapper = None
    load_env_cfg = load_rl_cfg = load_runner_cls = None
    MotionCommandCfg = MotionCommand = None
    compute_ee_position_error = compute_mpkpe = None
    configure_torch_backends = None
    _EVAL_IMPORT_ERROR = _e


@dataclass(frozen=True)
class Cfg:
    checkpoint: str
    motion_file: str
    task: str = "Mjlab-Tracking-Flat-Unitree-G1"
    task_module: str = ""  # module to import first (registers custom tasks, e.g. sim2real_task_v8)
    num_envs: int = 256
    seed: int = 90001
    device: str | None = None
    output_file: str = "heldout_eval.json"


def _run_condition(
    cfg: Cfg,
    device: str,
    push: bool,
    base_play_cfg,
    donor_train_cfg,
    agent_cfg,
    episode_length_s: float,
    max_steps: int,
) -> dict:
    # Explicit condition construction (harness v2, shared builder): PLAY cfg +
    # sensor noise (+ the training push event for the push row). The SAME seed
    # is used for both rows — paired comparison, recorded in `realized`.
    env_cfg = make_condition_cfg(
        base_play_cfg,
        seed=cfg.seed,
        noise=True,
        push=push,
        donor_train_cfg=donor_train_cfg,
    )

    motion_cmd = env_cfg.commands.get("motion")
    if not isinstance(motion_cmd, MotionCommandCfg):
        raise ValueError(f"{cfg.task} is not a tracking task")
    motion_cmd.motion_file = cfg.motion_file
    env_cfg.episode_length_s = episode_length_s
    env_cfg.scene.num_envs = cfg.num_envs

    # §3.4 realized block from the FINAL cfg (what actually runs).
    realized = extract_realized(env_cfg)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, _as_dict(agent_cfg), device=device)
    runner.load(cfg.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)

    command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
    ee_body_names = env_cfg.terminations["ee_body_pos"].params["body_names"]

    n = cfg.num_envs
    done_envs = torch.zeros(n, dtype=torch.bool, device=device)
    success = torch.zeros(n, dtype=torch.bool, device=device)
    mpkpe_acc, active_acc, ee_pos_acc = [], [], []

    obs = env.get_observations()
    step = 0
    while not done_envs.all() and step < max_steps:
        ref = SimpleNamespace(
            num_envs=command.num_envs, device=command.device, cfg=command.cfg,
            body_pos_w=command.body_pos_w.clone(),
            body_pos_relative_w=command.body_pos_relative_w.clone(),
            body_quat_relative_w=command.body_quat_relative_w.clone(),
            joint_vel=command.joint_vel.clone(),
        )
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        ref.robot_body_pos_w = command.robot_body_pos_w
        ref.robot_body_quat_w = command.robot_body_quat_w
        ref.robot_joint_vel = command.robot_joint_vel
        rc = cast(MotionCommand, ref)

        active = ~done_envs
        active_acc.append(active.float())
        mpkpe_acc.append(torch.where(active, compute_mpkpe(rc), 0.0))
        ee_pos_acc.append(torch.where(active, compute_ee_position_error(rc, ee_body_names), 0.0))

        terminated = env.unwrapped.termination_manager.terminated
        truncated = env.unwrapped.termination_manager.time_outs
        newly = dones.bool() & ~done_envs
        if newly.any():
            success = success | (newly & truncated & ~terminated)
            done_envs = done_envs | newly
        step += 1

    # EXACT horizon: never step past T (audit F4 — past the last reference
    # frame pinned mjlab teleports survivors to frame zero).
    assert step <= max_steps, f"steps_run {step} > T {max_steps}"
    # success = reached T (envs alive at T survived the whole motion).
    success = success | ~done_envs

    active_steps = torch.stack(active_acc, 0).sum(0).clamp(min=1)
    mpkpe = (torch.stack(mpkpe_acc, 0).sum(0) / active_steps).mean().item()
    ee_pos = (torch.stack(ee_pos_acc, 0).sum(0) / active_steps).mean().item()
    out = {
        # legacy key kept for pipeline/mjlab_verify.py; honest_name is the truth
        "condition": "push" if push else "nominal",
        "honest_name": "noise_push" if push else "noise",
        "realized": realized,  # CONVENTIONS §3.4 — from the FINAL cfg
        "num_episodes": n,
        "success_rate": success.float().mean().item(),
        "n_success": int(success.sum().item()),
        "mpkpe_m": mpkpe,
        "ee_pos_error_m": ee_pos,
        "steps_run": step,
        "seed": env_cfg.seed,
        "push_enabled": push,
    }
    env.close()
    return out


def _as_dict(agent_cfg) -> dict:
    from dataclasses import asdict, is_dataclass
    return asdict(agent_cfg) if is_dataclass(agent_cfg) else dict(agent_cfg)


def main() -> None:
    if _EVAL_IMPORT_ERROR is not None:
        raise SystemExit(f"heldout_eval needs the mjlab box env: {_EVAL_IMPORT_ERROR}")
    import mjlab.tasks  # noqa: F401
    default_task = "Mjlab-Tracking-Flat-Unitree-G1"
    raw = sys.argv[1:]
    argv = [a for i, a in enumerate(raw)
            if a != default_task or (i > 0 and raw[i - 1] == "--task")]
    cfg = tyro.cli(Cfg, args=argv)
    if cfg.task_module:
        import importlib

        importlib.import_module(cfg.task_module)
    task_id = cfg.task
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    # Certify the FULL dance at the EXACT horizon: the train cfg caps
    # episode_length_s at 20 s (audit L), and the old +0.2 s padding wrapped
    # survivors to frame zero (audit F4). T = frames/fps, nothing more.
    frames, fps = _motion_frames_fps(cfg.motion_file)
    episode_length_s, max_steps = exact_horizon(frames, fps)

    base_play_cfg = load_env_cfg(task_id, play=True)
    donor_train_cfg = load_env_cfg(task_id, play=False)
    agent_cfg = load_rl_cfg(task_id)

    print(f"[INFO] heldout_eval harness v{HARNESS_VERSION}: episode "
          f"{episode_length_s:.2f}s (T={max_steps} steps, exact), seed={cfg.seed} "
          f"(paired across both rows)")

    results = {}
    for push in (False, True):
        cond = _run_condition(cfg, device, push, base_play_cfg, donor_train_cfg,
                              agent_cfg, episode_length_s, max_steps)
        results[cond["condition"]] = cond
        print(f"[{cond['condition']}] success={cond['success_rate']:.3f} "
              f"({cond['n_success']}/{cond['num_episodes']}) mpkpe={cond['mpkpe_m']:.4f}m")

    Path(cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_file, "w") as f:
        json.dump({"harness_version": HARNESS_VERSION,
                   "task": task_id, "checkpoint": cfg.checkpoint,
                   "motion_file": cfg.motion_file,
                   "episode_length_s": episode_length_s,
                   "horizon_steps": max_steps,
                   "conditions": results}, f, indent=2)
    print(f"[INFO] wrote {cfg.output_file}")


if __name__ == "__main__":
    main()
