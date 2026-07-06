"""DYNAMIC-SKILLS (acro) tracking task: Mjlab-Tracking-Flat-Unitree-G1-Acro.

SIM-ONLY profile for short, highly-dynamic skills (backflip / aerial moves).
A physical attempt is a SEPARATE human decision gated by docs/DYNAMIC_SKILLS.md
(hardware-risk memo). Nothing here touches the robot or the show pipeline.

WHY A SEPARATE PROFILE (vs the show/s2r recipes):

  1. TERMINATIONS — the stock BeyondMimic deviation terminations are hostile to
     flip learning even though they are reference-relative:
       * anchor_pos (z-only, 0.25 m) and ee_body_pos (z-only, 0.25 m): during
         launch/flight the reference moves vertically at 2.5-3.5 m/s, so a
         ~100 ms phase lag — an otherwise correct flip — shows up as a 0.25-0.35 m
         instantaneous z "error" and kills the episode mid-air, exactly where
         the learning signal lives.
       * anchor_ori (|projected-gravity-z(ref) - projected-gravity-z(robot)|
         > 0.8; the metric's full range is 0..2): while the reference sweeps
         through horizontal, a small rotation-phase lag produces a large
         instantaneous gravity-z mismatch -> mid-flip termination.
     Replacement: the same stock checks, but (a) SUPPRESSED inside a per-frame
     "flight grace" window precomputed from the REFERENCE (both ankle_roll
     links more than FEET_AERIAL_RISE above their own grounded baseline,
     dilated +-GRACE_DILATE_STEPS control steps to cover launch and
     touchdown), and (b) RELAXED outside it (z 0.25->0.45 m,
     ori 0.8->1.4, ee 0.25->0.45 m) to tolerate crouch-depth/landing-compression
     timing while still firing on a genuine fall (fallen torso z error vs a
     standing reference is >=0.6 m; its gravity-z mismatch ~2). A robot that
     crashes mid-flight terminates within ~0.2 s of the grace window closing,
     so the adaptive sampler still gets a clean failure signal near the right
     time bin.

  2. FULL EFFORT LIMITS, NO TORQUE PENALTIES — flips need peak torque. The
     s2r recipe's joint_torques_l2 / ankle_torque_l2 penalties and the
     0.80-1.00 effort-limit derating are deliberately ABSENT. Thermal realism
     is a later stage, added back only if the skill ever heads to hardware.

  3. RSI FOR SHORT SKILLS — sampling stays "adaptive" (BeyondMimic bins), with
     adaptive_kernel_size 1->4 (the non-causal kernel spreads failure credit to
     the bins leading INTO the failure = the approach/launch phase, which is
     where a missed landing is actually caused) and adaptive_uniform_ratio
     0.1->0.2 (a ~5-10 s skill has only ~5-10 bins; keep coverage so sampling
     never collapses onto the landing bin). Reset randomization ranges stay
     stock — mid-air resets inherit the reference's linear/angular velocity,
     which is what makes flip learning tractable at all.

  4. NO PUSH EVENTS during skill acquisition — the stock push sets base
     velocity by +-0.5 m/s every 1-3 s; mid-flight that guarantees a missed
     landing, and with adaptive sampling focused on the flight bins it poisons
     the exact data the policy needs. Honest consequence: the resulting policy
     is NOT push-robust; a robustness pass (small pushes outside the flight
     window) is future work and is required before any hardware talk.

  5. Everything else stock: mild DR (base_com, encoder_bias, foot friction),
     self-collision penalty kept at -10 (a tuck that clips through the torso
     would be a fake flip), action_rate_l2 -0.1, joint_limit -10, 10 s episodes.

VET-GATE NOTE (documented contract): acro references BYPASS pipeline/
vet_motion.py — its hard checks encode show-dance assumptions (pelvis >=0.35 m
would fail deep crouches/rolls, the excursion/foot-skate math assumes upright
locomotion) and its advisory velocity check would flag every flip. Acro motions
instead get: tools/check_acro_reference.py (FK sanity: flip rotation actually
present, feet trajectories, joint limits, velocity vs motor ratings) BEFORE
training, and cloud/acro_eval.py (landing success, peak torque/velocity audit)
AFTER training. They must never be staged into data/dances/ or the show deploy
machinery.

Launch (on the box):
  ./envs/mjlab/bin/python cloud/train_dynamic_skills.py \
      Mjlab-Tracking-Flat-Unitree-G1-Acro \
      --env.commands.motion.motion-file motions/<acro>.npz \
      --env.scene.num-envs 4096 --agent.max-iterations 10000 \
      --agent.run-name train-acro-1 --video False
"""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1-Acro"

# --- flight-grace parameters --------------------------------------------------
# Airborne is judged RELATIVE to each foot's grounded baseline (5th percentile of
# its z over the whole reference): the ankle_roll link FRAME rides ~0.03-0.13 m
# even while the sole is planted (heel-lift/toe poses), so absolute thresholds
# misread dance stance as flight. Calibrated on thriller_deploy.npz (a ground
# dance = zero true flight): rise 0.10 -> 38.7% false-aerial, rise 0.20 -> 0.0%.
# A real flip carries the feet 0.5-1.0 m above baseline, far beyond 0.20.
FEET_AERIAL_RISE = 0.20    # m above the foot's own grounded baseline = airborne
GRACE_DILATE_STEPS = 10    # +-0.2 s at 50 Hz: covers launch and touchdown edges

# --- relaxed out-of-grace thresholds (stock values in parentheses) -------------
ANCHOR_POS_Z_THRESHOLD = 0.45   # (0.25) crouch/landing-compression timing slack
ANCHOR_ORI_THRESHOLD = 1.4      # (0.8)  metric range 0..2; ~2 when flat-on-floor
EE_BODY_POS_Z_THRESHOLD = 0.45  # (0.25) feet/hand swing timing slack


class _FlightGraceMixin:
  """Shared lazily-built per-frame grace mask over the reference motion.

  Built on first call (the command manager may not exist when the termination
  manager is constructed): a boolean mask over motion frames where BOTH
  reference feet are airborne, dilated by GRACE_DILATE_STEPS on each side.
  Indexed by the per-env motion time step, so grace follows reference phase.
  """

  def __init__(self, cfg, env):
    self._mask: torch.Tensor | None = None

  def reset(self, env_ids=None):  # termination manager calls this on class terms
    pass

  def _grace(self, env, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    if self._mask is None:
      names = list(command.cfg.body_names)
      feet = [i for i, n in enumerate(names) if n.endswith("_ankle_roll_link")]
      if len(feet) != 2:
        raise RuntimeError(f"expected 2 ankle_roll links in tracked bodies, got {feet}")
      feet_z = command.motion.body_pos_w[:, feet, 2]          # (T, 2)
      baseline = feet_z.quantile(0.05, dim=0)                 # per-foot grounded z
      aerial = (feet_z > baseline + FEET_AERIAL_RISE).all(dim=1)  # (T,)
      k = GRACE_DILATE_STEPS
      dilated = torch.nn.functional.max_pool1d(
        aerial.float().view(1, 1, -1), kernel_size=2 * k + 1, stride=1, padding=k
      ).view(-1)
      self._mask = dilated > 0.5
      n_frames = int(self._mask.sum().item())
      print(f"[acro] flight-grace mask: {n_frames}/{len(self._mask)} frames "
            f"({n_frames / max(len(self._mask), 1) * 100:.1f}%) suppressed", flush=True)
    return self._mask[command.time_steps]


class anchor_pos_z_flip_aware(_FlightGraceMixin):
  def __call__(self, env, command_name: str, threshold: float) -> torch.Tensor:
    raw = tracking_mdp.bad_anchor_pos_z_only(env, command_name, threshold)
    return raw & ~self._grace(env, command_name)


class anchor_ori_flip_aware(_FlightGraceMixin):
  def __call__(self, env, asset_cfg, command_name: str, threshold: float) -> torch.Tensor:
    raw = tracking_mdp.bad_anchor_ori(env, asset_cfg, command_name, threshold)
    return raw & ~self._grace(env, command_name)


class ee_body_pos_z_flip_aware(_FlightGraceMixin):
  def __call__(self, env, command_name: str, threshold: float, body_names) -> torch.Tensor:
    raw = tracking_mdp.bad_motion_body_pos_z_only(env, command_name, threshold, body_names)
    return raw & ~self._grace(env, command_name)


def _apply_acro(cfg, train: bool):
  """Mutate a freshly built STOCK G1 tracking cfg with the acro deltas."""
  # -- 1. flip-aware terminations (both modes: play/eval judge on the same law)
  cfg.terminations["anchor_pos"] = TerminationTermCfg(
    func=anchor_pos_z_flip_aware,
    params={"command_name": "motion", "threshold": ANCHOR_POS_Z_THRESHOLD},
  )
  cfg.terminations["anchor_ori"] = TerminationTermCfg(
    func=anchor_ori_flip_aware,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "command_name": "motion",
      "threshold": ANCHOR_ORI_THRESHOLD,
    },
  )
  cfg.terminations["ee_body_pos"] = TerminationTermCfg(
    func=ee_body_pos_z_flip_aware,
    params={
      "command_name": "motion",
      "threshold": EE_BODY_POS_Z_THRESHOLD,
      "body_names": (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
      ),
    },
  )

  if not train:
    return cfg

  # -- 3. RSI tuned for short skills (train only; play uses sampling_mode start)
  motion_cmd = cfg.commands["motion"]
  motion_cmd.adaptive_kernel_size = 4
  motion_cmd.adaptive_uniform_ratio = 0.2

  # -- 4. no pushes during skill acquisition (see module docstring, honesty note)
  cfg.events.pop("push_robot", None)

  return cfg


def _make(train: bool, play: bool):
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  return _apply_acro(cfg, train=train)


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=_make(train=True, play=False),
  play_env_cfg=_make(train=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
