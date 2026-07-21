# Pinned mjlab 1.5.0 source evidence

Source: `mjlab-1.5.0-py3-none-any.whl`, downloaded without dependencies and
inspected as a zip. Full wheel SHA-256:
`93aa539d1c7d8e984a34b8855967d304dd14a01e60c95aa03d7ac71c228f070c`.
`crosscheck.py` independently checks these semantics and records a SHA-256 for
every source member in `crosscheck.json`.

## Effort-limit selection and overwrite semantics

Wheel member `mjlab/envs/mdp/dr/actuator.py`, upstream lines 192–256:

```python
@requires_model_fields("actuator_forcerange", "jnt_actfrcrange", "tendon_actfrcrange")
def effort_limits(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  effort_limit_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform"] = "uniform",
  operation: Operation | str = "scale",
) -> None:
  # ...
  asset: Entity = env.scene[asset_cfg.name]
  # ...
  if isinstance(asset_cfg.actuator_ids, list):
    actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
  else:
    actuators = asset.actuators[asset_cfg.actuator_ids]
  if not isinstance(actuators, list):
    actuators = [actuators]
  for actuator in actuators:
    ctrl_ids = actuator.global_ctrl_ids
    # ...
    if isinstance(actuator, (BuiltinPositionActuator, BuiltinDcMotorActuator)) or (
      isinstance(actuator, XmlActuator) and actuator.command_field == "position"
    ):
      if op.name == "scale":
        default_forcerange = env.sim.get_default_field("actuator_forcerange")
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 0] = (
          default_forcerange[ctrl_ids, 0] * effort_samples
        )
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids, 1] = (
          default_forcerange[ctrl_ids, 1] * effort_samples
        )
```

The function never reads `joint_ids`. It selects `asset.actuators` through
`actuator_ids`, and every scale writes from the pristine default rather than
composing with an earlier event.

Wheel member `mjlab/managers/scene_entity_config.py`, upstream lines 20–27 and
66–94, resolves joint and actuator selectors independently:

```python
_FIELD_CONFIGS = [
  _FieldConfig("joint_names", "joint_ids", "find_joints", "num_joints", "joint"),
  # ...
  _FieldConfig(
    "actuator_names", "actuator_ids", "find_actuators", "num_actuators", "actuator"
  ),
]

joint_names: str | tuple[str, ...] | None = None
joint_ids: list[int] | slice = field(default_factory=lambda: slice(None))
# ...
actuator_names: str | list[str] | None = None
actuator_ids: list[int] | slice = field(default_factory=lambda: slice(None))
```

Therefore `SceneEntityCfg("robot", joint_names=...)` leaves
`actuator_ids=slice(None)` and `effort_limits` applies to all high-level actuator
groups.

Wheel member `mjlab/managers/event_manager.py`, upstream lines 341–357 and
260–264, preserves configuration insertion order and executes in that order:

```python
for term_name, term_cfg in self.cfg.items():
  # ...
  self._mode_term_cfgs[term_cfg.mode].append(term_cfg)

for index, term_cfg in enumerate(self._mode_term_cfgs[mode]):
  # execute term_cfg.func ...
```

The composed event inserted last by v8 therefore overwrites every earlier
effort-limit event on all actuator groups.

## Configuration dictionaries and observation history

Wheel member `mjlab/envs/manager_based_rl_env.py`, upstream line 121:

```python
commands: dict[str, CommandTermCfg] = field(default_factory=dict)
```

The repository's G1 config likewise accesses `cfg.commands["motion"]` and
`cfg.observations["actor"]`; they are not attribute namespaces.

Wheel member `mjlab/managers/observation_manager.py`, upstream lines 348–359:

```python
if term_cfg.delay_max_lag > 0:
  delay_buffer = self._group_obs_term_delay_buffer[group_name][term_name]
  delay_buffer.append(obs)
  obs = delay_buffer.compute()
if term_cfg.history_length > 0:
  circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
  if update_history or not circular_buffer.is_initialized:
    circular_buffer.append(obs)
  if term_cfg.flatten_history_dim:
    group_obs[term_name] = circular_buffer.buffer.reshape(self._env.num_envs, -1)
```

Each term's history is flattened before terms are concatenated: term-major.
Wheel member `mjlab/utils/buffers/circular_buffer.py`, upstream lines 162–175
and 207–215, returns chronological history and backfills the first frame:

```python
@property
def buffer(self) -> torch.Tensor:
  """History in chronological order (oldest to newest)."""
  start = (self._pointer + 1) % self._max_len
  idx = (torch.arange(self._max_len, device=self._device) + start) % self._max_len
  buf = self._buffer.index_select(0, idx)
  return buf.transpose(0, 1)

self._pointer = (self._pointer + 1) % self._max_len
self._buffer[self._pointer] = data
# Backfill entire history with first frame for newly initialized batches.
is_first_push = self._num_pushes == 0
condition = is_first_push.view(1, self._batch_size, *([1] * (data.ndim - 1)))
torch.where(condition, data.unsqueeze(0), self._buffer, out=self._buffer)
```

This confirms the shared `HistoryStacker`'s static layout and warmup behavior.

## Motion-end resampling teleports simulation state

Wheel member `mjlab/tasks/tracking/mdp/commands.py`, upstream lines 319–375 and
407–417:

```python
def _resample_command(self, env_ids: torch.Tensor):
  if self.cfg.sampling_mode == "start":
    self.time_steps[env_ids] = 0
  # ... construct reference root and joint state ...
  self._write_reference_state_to_sim(
    env_ids,
    root_pos,
    root_ori,
    root_lin_vel,
    root_ang_vel,
    joint_pos,
    joint_vel,
  )

def _update_command(self):
  self.time_steps += 1
  env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
  if env_ids.numel() > 0:
    self._resample_command(env_ids)
    self._env.sim.forward()
  self.update_relative_body_poses()
```

With `sampling_mode="start"`, stepping past the last frame resets the clock and
writes frame-zero reference root/joint state into the simulated robot.

## The built-in push is not a force

Wheel member `mjlab/envs/mdp/events.py`, upstream lines 316–337:

```python
def push_by_setting_velocity(...):
  """Push an entity by overwriting its root velocity with a sampled offset.

  This is an *instantaneous, mass-independent* kick: it adds a uniformly sampled
  delta directly to the root velocity, ignoring inertia and contact dynamics.
  """
  asset: Entity = env.scene[asset_cfg.name]
  vel_w = asset.data.root_link_vel_w[env_ids]
  vel_w += _sample_se3_range(velocity_range, vel_w.shape, env.device)
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)
```

It cannot be recast as a uniquely defined Newton force without choosing an
arbitrary application interval.

## Pinned G1 collision setup

Wheel member `mjlab/asset_zoo/robots/unitree_g1/g1_constants.py`, upstream
lines 217–225 and 251–275:

```python
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)

return EntityCfg(
  init_state=KNEES_BENT_KEYFRAME,
  collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
  articulation=G1_ARTICULATION,
)
```

The raw pinned XML and the repository preview-model summaries are recomputed in
`crosscheck.json`; the training config then randomizes foot friction at startup.
