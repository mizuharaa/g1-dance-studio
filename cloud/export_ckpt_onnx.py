#!/usr/bin/env python3
"""Export an ARBITRARY mjlab tracking checkpoint to a deploy ONNX.

Why this exists: mjlab exports one ONNX per run, at training end — i.e. always the
LAST checkpoint. cloud/export_policy.py just copies that file, so whenever
pick_checkpoint selects a non-final winner (the whole point of the picker), the
staged policy.onnx silently disagrees with the gated checkpoint. This script loads
the winner .pt into the task's runner and re-runs the runner's own ONNX exporter
(motion baked in + metadata attached), so gap.json and policy.onnx describe the
SAME network.

Usage (box):
  $PY cloud/export_ckpt_onnx.py --checkpoint <model_N.pt> --motion-file <motion.npz> \
      --out-dir <exports> --task Mjlab-Tracking-Flat-Unitree-G1-S2R-V8 \
      --task-module sim2real_task_v8
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from pipeline import artifacts


@dataclass(frozen=True)
class Cfg:
  checkpoint: str
  motion_file: str
  out_dir: str
  filename: str = "policy.onnx"
  task: str = "Mjlab-Tracking-Flat-Unitree-G1"
  task_module: str = ""  # module registering custom tasks (e.g. sim2real_task_v8)
  num_envs: int = 2  # runner needs a live env; keep it tiny
  device: str | None = None
  # None derives the contract from the live actor: estimator-free policies require
  # contact; estimator-dependent gantry policies do not. The CLI can override it.
  requires_ground_contact: bool | None = None


def _plain_list(value: Any) -> list:
  """Turn tensors/arrays/scalars into a JSON-safe flat list."""
  if hasattr(value, "detach"):
    value = value.detach().cpu()
  if hasattr(value, "tolist"):
    value = value.tolist()
  if isinstance(value, (tuple, list)):
    if len(value) == 1 and isinstance(value[0], (tuple, list)):
      value = value[0]
    return list(value)
  return [value]


def _value_info_shape(value_info) -> list[int | str | None]:
  out: list[int | str | None] = []
  for dim in value_info.type.tensor_type.shape.dim:
    if dim.HasField("dim_value"):
      out.append(int(dim.dim_value))
    elif dim.HasField("dim_param") and dim.dim_param:
      out.append(str(dim.dim_param))
    else:
      out.append(None)
  return out


def onnx_contract(path: str | Path) -> tuple[dict[str, list], list[str]]:
  """Read the graph's actual public I/O contract, without executing it."""
  import onnx

  model = onnx.load(str(path), load_external_data=False)
  initializers = {value.name for value in model.graph.initializer}
  inputs = {
    value.name: _value_info_shape(value)
    for value in model.graph.input
    if value.name not in initializers
  }
  return inputs, [value.name for value in model.graph.output]


def _natural_ctrl_ids(robot) -> list[int]:
  by_joint = {
    actuator.target.split("/")[-1]: int(actuator.id)
    for actuator in robot.spec.actuators
  }
  missing = [name for name in robot.joint_names if name not in by_joint]
  if missing:
    raise ValueError(f"live robot has joints without actuators: {missing}")
  return [by_joint[name] for name in robot.joint_names]


def _actor_contract(env) -> tuple[list[str], list[int], int, int]:
  manager = env.observation_manager
  names = list(manager.active_terms["actor"])
  flat_dims = list(manager.group_obs_term_dim["actor"])
  if len(names) != len(flat_dims):
    raise ValueError("live actor term names/dimensions have different lengths")

  histories: list[int] = []
  widths: list[int] = []
  for name, dims in zip(names, flat_dims, strict=True):
    term_cfg = manager.get_term_cfg("actor", name)
    history = max(1, int(getattr(term_cfg, "history_length", 0) or 0))
    flat_width = math.prod(int(v) for v in dims)
    if flat_width % history:
      raise ValueError(
        f"live actor term {name!r} width {flat_width} is not divisible by "
        f"history_length {history}"
      )
    if history > 1 and not bool(getattr(term_cfg, "flatten_history_dim", False)):
      raise ValueError(f"live actor term {name!r} history is not flattened")
    histories.append(history)
    widths.append(flat_width // history)

  if not histories or len(set(histories)) != 1:
    raise ValueError(f"live actor terms do not share one history length: {histories}")
  return names, widths, sum(widths), histories[0]


def build_policy_meta(
  env_cfg,
  env,
  onnx_path: str | Path,
  *,
  task_id: str,
  task_module: str,
  checkpoint: str,
  requires_ground_contact: bool,
) -> dict:
  """Build policy_meta v2 exclusively from the exported graph and live env."""
  inputs, outputs = onnx_contract(onnx_path)
  names, widths, obs_per_frame, history_length = _actor_contract(env)
  actual_obs_width = inputs.get("obs", [None])[-1]
  expected_obs_width = obs_per_frame * history_length
  if actual_obs_width != expected_obs_width:
    raise ValueError(
      f"exported ONNX obs width {actual_obs_width!r} != live actor width "
      f"{obs_per_frame} x history {history_length} = {expected_obs_width}"
    )

  robot = env.scene["robot"]
  joint_names = list(robot.joint_names)
  ctrl_ids = _natural_ctrl_ids(robot)
  action = env.action_manager.get_term("joint_pos")
  action_names = list(getattr(action, "target_names", joint_names))
  if action_names != joint_names:
    raise ValueError(
      "live joint-position action order differs from robot joint order: "
      f"actions={action_names}, robot={joint_names}"
    )

  model = env.sim.mj_model
  default = _plain_list(robot.data.default_joint_pos[0])
  kp = _plain_list(model.actuator_gainprm[ctrl_ids, 0])
  kd = _plain_list(-model.actuator_biasprm[ctrl_ids, 2])
  force_ranges = model.actuator_forcerange[ctrl_ids]
  effort = [max(abs(float(lo)), abs(float(hi))) for lo, hi in force_ranges]
  raw_scale = getattr(action, "_scale", getattr(action, "scale", None))
  if isinstance(raw_scale, dict):
    # cfg-style pattern dict ({regex: value}) — resolve per joint, later keys win
    import re as _re
    resolved = [None] * len(joint_names)
    for pat, val in raw_scale.items():
      for j, jn in enumerate(joint_names):
        if pat == jn or _re.fullmatch(pat, jn):
          resolved[j] = float(val)
    missing = [jn for jn, v in zip(joint_names, resolved) if v is None]
    if missing:
      raise ValueError(f"action scale dict does not cover joints: {missing}")
    scale = resolved
  else:
    scale = _plain_list(raw_scale)
    # live term scale is [num_envs, n_actions]; _plain_list unwraps a single
    # nesting only, so a multi-env export yields one ROW per env — the scale is
    # env-invariant, take row 0 (verified identical shape to the joint count).
    if scale and isinstance(scale[0], (list, tuple)):
      scale = list(scale[0])
  if len(scale) == 1:
    scale *= len(joint_names)
  for field, values in {
    "default_joint_pos": default,
    "joint_stiffness": kp,
    "joint_damping": kd,
    "effort_limit": effort,
    "action_scale": scale,
  }.items():
    if len(values) != len(joint_names):
      raise ValueError(
        f"live {field} has {len(values)} entries for {len(joint_names)} joints"
      )

  # pinned mjlab 1.5.0 nests the timestep at sim.mujoco.timestep (verified on
  # the box: SimulationCfg has no 'dt'); keep fallbacks for other layouts.
  sim = env_cfg.sim
  sim_dt = None
  for getter in (lambda: sim.mujoco.timestep, lambda: sim.dt,
                 lambda: sim.timestep):
    try:
      sim_dt = float(getter())
      break
    except AttributeError:
      continue
  if sim_dt is None:
    raise ValueError(
      "cannot resolve physics timestep from SimulationCfg "
      f"(fields: {[getattr(f, 'name', f) for f in getattr(sim, '__dataclass_fields__', {}).values()] or dir(sim)})")
  decimation = int(getattr(env_cfg, "decimation"))
  obs_terms = [[name, width, history_length] for name, width in zip(names, widths, strict=True)]
  return {
    "schema": "g1.policy_meta/2",
    "policy_meta_version": 2,
    "framework": "mjlab",
    "task": task_id,
    "task_module": task_module,
    "exported_from_checkpoint": checkpoint,
    "control_hz": 1.0 / (sim_dt * decimation),
    "sim_dt": sim_dt,
    "decimation": decimation,
    "actor_obs_terms_in_order": names,
    "actor_obs_term_widths": [[name, width] for name, width in zip(names, widths, strict=True)],
    "obs_terms": obs_terms,
    "obs_concatenated": True,
    "obs_per_frame": obs_per_frame,
    "history_length": history_length,
    "flatten_layout": "term-major-oldest-first",
    "onnx_inputs": inputs,
    "onnx_outputs": outputs,
    "requires_ground_contact": bool(requires_ground_contact),
    "anchor_body_name": "torso_link",
    "joint_names": joint_names,
    "joint_order_29dof": joint_names,
    "default_joint_pos": default,
    "default_joint_pos_rad": default,
    "joint_stiffness": kp,
    "kp": kp,
    "kp_stiffness": kp,
    "joint_damping": kd,
    "kd": kd,
    "kd_damping": kd,
    "effort_limit_nm": effort,
    "action_scale": scale,
    "action_scale_per_joint": scale,
    "action_use_default_offset": bool(
      getattr(getattr(action, "cfg", None), "use_default_offset", True)
    ),
  }


def write_policy_artifacts(
  out_dir: str | Path,
  onnx_path: str | Path,
  meta: dict,
  *,
  task_id: str,
  task_module: str,
  checkpoint: str,
) -> dict:
  """Write the sidecar and merge Lane B's sections into bundle.json."""
  out_dir = Path(out_dir)
  onnx_path = Path(onnx_path)
  meta_path = out_dir / "policy_meta.json"
  meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

  manifest_path = out_dir / "bundle.json"
  try:
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
  except (OSError, json.JSONDecodeError) as exc:
    raise ValueError(f"cannot preserve existing bundle manifest: {exc}") from exc
  model = dict(manifest.get("model") or {})
  model.update({"task_id": task_id, "task_module": task_module})
  manifest["model"] = model
  manifest["policy"] = {
    "onnx": artifacts.file_entry(onnx_path, out_dir),
    "meta": artifacts.file_entry(meta_path, out_dir),
    "checkpoint": checkpoint,
    "obs_per_frame": int(meta["obs_per_frame"]),
    "history_length": int(meta["history_length"]),
    "flatten_layout": meta["flatten_layout"],
    "requires_ground_contact": bool(meta["requires_ground_contact"]),
  }
  return artifacts.write_manifest(manifest_path, manifest)


def main() -> None:
  import torch
  import tyro
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.tasks.tracking.mdp import MotionCommandCfg
  from mjlab.utils.torch import configure_torch_backends

  cfg = tyro.cli(Cfg)
  if cfg.task_module:
    importlib.import_module(cfg.task_module)
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task, play=False)
  agent_cfg = load_rl_cfg(cfg.task)
  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(f"{cfg.task} is not a tracking task")
  motion_cmd.motion_file = cfg.motion_file
  env_cfg.scene.num_envs = cfg.num_envs

  live_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(live_env, clip_actions=agent_cfg.clip_actions)
  agent_dict = asdict(agent_cfg) if is_dataclass(agent_cfg) else dict(agent_cfg)
  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, agent_dict, device=device)
  runner.load(cfg.checkpoint, map_location=device)
  runner.export_policy_to_onnx(cfg.out_dir, filename=cfg.filename)
  onnx_path = Path(cfg.out_dir) / cfg.filename
  names = list(live_env.observation_manager.active_terms["actor"])
  estimator_terms = {"base_lin_vel", "motion_anchor_pos_b"}
  requires_contact = (
    not bool(estimator_terms.intersection(names))
    if cfg.requires_ground_contact is None
    else cfg.requires_ground_contact
  )
  meta = build_policy_meta(
    env_cfg,
    live_env,
    onnx_path,
    task_id=cfg.task,
    task_module=cfg.task_module,
    checkpoint=cfg.checkpoint,
    requires_ground_contact=requires_contact,
  )
  manifest = write_policy_artifacts(
    cfg.out_dir,
    onnx_path,
    meta,
    task_id=cfg.task,
    task_module=cfg.task_module,
    checkpoint=cfg.checkpoint,
  )
  print(
    f"EXPORTED {cfg.checkpoint} -> {onnx_path} with policy_meta v2 "
    f"and bundle {manifest['bundle_id']}"
  )


if __name__ == "__main__":
  main()
