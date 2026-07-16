"""Single source of truth for the Unitree G1 (29-DoF) actuation envelope.

WHY THIS FILE EXISTS
--------------------
The motion pipeline used to read joint ranges off whatever MJCF happened to be
handy (``vet_motion.py`` loads the *menagerie* G1, which is NOT the training
model), and torque limits were an assumed "flat 50 Nm ankle" folklore number.
Feasibility must be measured against the robot's ACTUAL envelope, so this module
extracts it once, from the model the policy is actually trained in, and every
feasibility check imports from here.

PROVENANCE (every number is traceable — measurement-discipline rule)
--------------------------------------------------------------------
* Effort limits, velocity limits, armatures, kp/kd: BeyondMimic / mjlab G1 config
  ``third_party/whole_body_tracking/.../robots/g1.py`` (the actuator groups),
  cross-checked line-for-line against the exported
  ``data/policies/thriller_csv_ankle_penalty/policy_meta.json`` (``effort_limit_nm``,
  ``kp_stiffness``, ``kd_damping``). They agree exactly. Agent 0's upstream audit
  (``experiments/upstream_alignment_report.md`` §5) confirms these are the SIM
  gains == the DEPLOY gains (BeyondMimic impedance model,
  ``kp = armature*(2*pi*10)^2``, ``kd = 2*zeta*armature*(2*pi*10)``, zeta=2).
* Joint POSITION ranges + full rigid-body inertias: the official Unitree MJCF
  ``third_party/unitree_mujoco/unitree_robots/g1/g1_29dof.xml`` (29 hinges in
  exact LAFAN1 order, verified against policy_meta ``joint_order_29dof``).

THE TORQUE-SPEED CAVEAT (Agent 0, report §5 — load-bearing for dynamics)
-----------------------------------------------------------------------
Our mjlab sim uses a FLAT ``effort_limit_sim`` clamp (ankle = 50 Nm at ANY
speed). Isaac's ``UnitreeActuator`` models a real torque-speed (T-N) curve:
available torque FALLS as joint speed rises. At the two failure beats
(13-18 s, 25-36 s) the ankles are simultaneously FAST and high-torque, so the
REAL ankle has LESS than 50 Nm available exactly where we fall — our flat-clamp
sim is OPTIMISTIC. ``effective_torque_limit()`` below models a conservative
speed-derated limit so the dynamic feasibility pass does not repeat the sim's
optimism. See that function's docstring for the exact assumption.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# The official Unitree G1 MJCF: correct inertias + LAFAN1 joint order. This is
# the model the dynamic pass runs mj_inverse against (armatures patched to the
# mjlab values below so the joint-space inertia matches the training model).
MODEL_XML = ROOT / "third_party/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"

# 29-DoF joint order (LAFAN1 / mjlab csv_to_npz). Motion CSV cols 7..36 are these.
JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
N_JOINTS = 29
IDX = {n: i for i, n in enumerate(JOINT_ORDER)}

# --- Per-joint EFFORT limits [Nm] (BeyondMimic effort_limit_sim == policy_meta) ---
EFFORT_LIMIT_NM = np.array([
    88.0, 139.0, 88.0, 139.0, 50.0, 50.0,      # left leg  (hipP,hipR,hipY,knee,ankP,ankR)
    88.0, 139.0, 88.0, 139.0, 50.0, 50.0,      # right leg
    88.0, 50.0, 50.0,                          # waist yaw, roll, pitch
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,    # left arm (shP,shR,shY,elbow,wristR,wristP,wristY)
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,    # right arm
])

# --- Per-joint VELOCITY limits [rad/s] (BeyondMimic velocity_limit_sim) ---
# These double as the DC-motor no-load speed in effective_torque_limit().
VELOCITY_LIMIT = np.array([
    32.0, 20.0, 32.0, 20.0, 37.0, 37.0,        # left leg
    32.0, 20.0, 32.0, 20.0, 37.0, 37.0,        # right leg
    32.0, 37.0, 37.0,                          # waist
    37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0,  # left arm
    37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0,  # right arm
])

# --- Per-joint ARMATURE (rotor inertia reflected to joint) [kg m^2] ---
_A5020, _A7520_14, _A7520_22, _A4010 = 0.003609725, 0.010177520, 0.025101925, 0.00425
ARMATURE = np.array([
    _A7520_14, _A7520_22, _A7520_14, _A7520_22, 2 * _A5020, 2 * _A5020,   # left leg
    _A7520_14, _A7520_22, _A7520_14, _A7520_22, 2 * _A5020, 2 * _A5020,   # right leg
    _A7520_14, 2 * _A5020, 2 * _A5020,                                    # waist yaw,roll,pitch
    _A5020, _A5020, _A5020, _A5020, _A5020, _A4010, _A4010,               # left arm
    _A5020, _A5020, _A5020, _A5020, _A5020, _A4010, _A4010,               # right arm
])

# --- PD gains (deploy == sim; from policy_meta.json) ---
KP = np.array([
    40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
    40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
    40.179, 28.501, 28.501,
    14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,
    14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,
])
KD = np.array([
    2.5579, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144,
    2.5579, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144,
    2.5579, 1.8144, 1.8144,
    0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,
    0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,
])

DEFAULT_JOINT_POS = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
])

# Joint-group index helpers
ANKLE_IDX = np.array([IDX[n] for n in JOINT_ORDER if "ankle" in n])           # 4,5,10,11
KNEE_IDX = np.array([IDX[n] for n in JOINT_ORDER if "knee" in n])
HIP_IDX = np.array([IDX[n] for n in JOINT_ORDER if "hip" in n])
WAIST_IDX = np.array([IDX[n] for n in JOINT_ORDER if "waist" in n])
LEG_IDX = np.array(sorted(set(HIP_IDX) | set(KNEE_IDX) | set(ANKLE_IDX)))
ARM_IDX = np.array([IDX[n] for n in JOINT_ORDER
                    if any(k in n for k in ("shoulder", "elbow", "wrist"))])

# --- Torque-speed derating parameters (the safety-critical assumption) ---------
# Conservative headroom below the hard clamp we require the REPAIRED motion to sit
# under, per the ankle. Spec: "target required ankle torque <= 35-40 Nm".
ANKLE_HEADROOM_NM = 40.0
# Global headroom factor applied to every joint's (speed-derated) limit before
# flagging. 0.90 => keep 10% margin off the actuator envelope everywhere.
GLOBAL_HEADROOM = 0.90


def effective_torque_limit(joint_vel: np.ndarray) -> np.ndarray:
    """Speed-dependent effective torque limit [Nm], broadcast over ``joint_vel``.

    MODEL (documented assumption — we could NOT obtain Isaac's exact Y1/Y2/X1/X2
    knee-points on this box, so we use a conservative DC-motor linear derate):

        tau_eff(w) = effort_limit * (1 - |w| / w_free) ,  clamped to [0, effort_limit]

    with ``w_free = VELOCITY_LIMIT`` (the sim no-load speed). Stall torque is set
    EQUAL to the effort limit, so torque derates from the peak the instant the
    joint moves — intentionally pessimistic (real stall torque is higher, so the
    real usable-at-low-speed torque is >= this). This guarantees we never call a
    frame feasible that the flat-clamp sim would, and it bites hardest exactly at
    the fast beats, which is the point. It is STILL optimistic vs. a true T-N knee
    curve at very high speed; callers additionally cap the ankle at
    ANKLE_HEADROOM_NM. ``joint_vel`` may be (29,) or (N,29)."""
    jv = np.abs(np.asarray(joint_vel, dtype=float))
    derate = np.clip(1.0 - jv / VELOCITY_LIMIT, 0.0, 1.0)
    return derate * EFFORT_LIMIT_NM


def flag_limit(joint_vel: np.ndarray) -> np.ndarray:
    """The binding torque limit used to FLAG a frame: the speed-derated effective
    limit with GLOBAL_HEADROOM applied, and ankles additionally capped at
    ANKLE_HEADROOM_NM. Shape follows ``joint_vel`` ((29,) or (N,29))."""
    lim = GLOBAL_HEADROOM * effective_torque_limit(joint_vel)
    lim = np.atleast_2d(lim).copy()
    lim[:, ANKLE_IDX] = np.minimum(lim[:, ANKLE_IDX], ANKLE_HEADROOM_NM)
    return lim.reshape(np.shape(effective_torque_limit(joint_vel)))


# --- Position ranges: loaded from the official MJCF at import (hardware truth) --
def _load_position_ranges():
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        lo = np.zeros(N_JOINTS)
        hi = np.zeros(N_JOINTS)
        for i, name in enumerate(JOINT_ORDER):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            lo[i], hi[i] = m.jnt_range[jid]
        return lo, hi
    except Exception:
        return None, None


POS_LO, POS_HI = _load_position_ranges()


def build_model():
    """Return an ``mujoco.MjModel`` of the G1 with per-joint armature patched to
    the mjlab/BeyondMimic values (so the joint-space inertia matches the training
    model) and CONTACT DISABLED (the dynamic pass supplies the ground reaction as
    the free-base residual wrench, not through contact solving). CPU only."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    for i, name in enumerate(JOINT_ORDER):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        dofadr = m.jnt_dofadr[jid]
        m.dof_armature[dofadr] = ARMATURE[i]
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return m


# --- Cross-check vs Unitree published G1 EDU specs -----------------------------
# Unitree's public G1 material lists joint peak torques by motor class. Our
# effort table maps onto them and there are NO material discrepancies at the
# joints that matter for balance (legs/ankles):
#   knee / hip-roll  = 139 Nm  (7520-22 class, the big leg motors)
#   hip-pitch / -yaw / waist-yaw = 88 Nm (7520-14 class)
#   ankle P/R, waist R/P = 50 Nm (dual 5020)
#   shoulders/elbow/wrist-roll = 25 Nm (5020) ; wrist P/Y = 5 Nm (4010)
# DISCREPANCY TO NOTE: these are effort_limit_sim CLAMPS, not the real T-N curve.
# Unitree's marketing "peak torque" is a stall/low-speed number; sustained and
# high-speed torque is lower. That is precisely why effective_torque_limit()
# derates and why the ankle is additionally capped at 40 Nm for repair.
CROSSCHECK_NOTE = (
    "effort limits match BeyondMimic g1.py and policy_meta.json exactly; they are "
    "flat clamps, not the real velocity-derated T-N curve (see effective_torque_limit)."
)


def summary() -> dict:
    return {
        "n_joints": N_JOINTS,
        "model_xml": str(MODEL_XML),
        "effort_limit_nm": EFFORT_LIMIT_NM.tolist(),
        "velocity_limit_rad_s": VELOCITY_LIMIT.tolist(),
        "ankle_idx": ANKLE_IDX.tolist(),
        "ankle_headroom_nm": ANKLE_HEADROOM_NM,
        "global_headroom": GLOBAL_HEADROOM,
        "pos_lo": None if POS_LO is None else POS_LO.tolist(),
        "pos_hi": None if POS_HI is None else POS_HI.tolist(),
        "torque_speed_model": "DCMotor linear derate, w_free=velocity_limit, stall=effort_limit",
        "crosscheck": CROSSCHECK_NOTE,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
