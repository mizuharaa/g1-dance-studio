#!/usr/bin/env bash
# Runs INSIDE qiayuanl/unitree:jazzy on PC2. Contract: load policy, hold
# damping; motion playback is armed only by the operator's remote sequence.
set -euo pipefail
source /bundle/controller.env
if [ "${START_MODE:-}" != "damping" ]; then
  echo 'REFUSING: controller.env START_MODE is not damping.'; exit 79
fi
# FALL-RISK GATE: the policy trained on SIM PD gains (policy_meta.json, low
# overdamped ζ=2). Stock Unitree gains destabilize it. Require the gains spec and
# an explicit attestation the controller loaded THESE gains (set on robot day).
if [ "${USE_SIM_GAINS:-}" != "1" ] || [ ! -f /bundle/policy_meta.json ]; then
  echo 'REFUSING: SIM gains (policy_meta.json) not present/selected.'; exit 77
fi
if [ ! -f /bundle/SIM_GAINS_LOADED ]; then
  echo 'REFUSING: controller has not been confirmed to load the SIM PD gains from'
  echo 'policy_meta.json (kp/kd/effort/default_pos). Stock gains = fall risk.'
  echo 'See ROBOT_DAY_PLAN step 3 — verify gains, then touch SIM_GAINS_LOADED.'
  exit 76
fi
# ACTIVATION HAZARD: motion.csv (thriller_deploy) begins with a 2.5s ramp from
# default_joint_pos so frame-0 == standby (delta ~0). Do NOT substitute the raw
# show clip: activation would lurch (up to ~39deg elbow/knee step).
if [ ! -f /bundle/LAUNCH_LINE_VERIFIED ]; then
  echo 'REFUSING: controller launch line not verified on robot day yet.'
  echo 'See docs/ROBOT_DAY_PLAN.md step 3 — then touch LAUNCH_LINE_VERIFIED'
  echo 'in the bundle and re-push.'
  exit 78
fi
# ROBOT-DAY: replace with the verified launch line, e.g.:
# ros2 launch motion_tracking_controller tracking.launch.py \
#   policy:=/bundle/policy.onnx start_mode:=damping
exit 78
