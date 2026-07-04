"""Kinematic (leg) odometry for the G1 — a SERVICE-INDEPENDENT base-state estimate.

The onboard estimate (rt/odommodestate) FREEZES the moment we release the motion service
to take low-level control (confirmed on hardware 2026-07-04) — so it cannot feed the
policy during a run. This estimates the two terms the ground policy needs (base linear
velocity + base height) from ONLY LowState (joint q/dq) + the IMU, which we always have.

Principle: with a foot planted (stationary in the world), the pelvis velocity is fixed by
the leg kinematics. For stance foot f, the world foot velocity is zero:
    0 = v_pelvis_body + omega × r_f + J_f · dq_leg          (all in the pelvis/body frame)
  => v_pelvis_body = -(J_f · dq_leg + omega × r_f)
where r_f = foot position in the pelvis frame (FK) and J_f = foot linear Jacobian wrt the
joints. Base height above the (flat) ground = -(R_base · r_stance)_z. The two feet are
blended by how "planted" each is (lower world-z gets more weight), so it stays smooth
across contact switches. base_lin_vel trained with ±0.5 m/s noise, so this need only be
roughly right — validated offline against the reference (tools/validate_leg_odom.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
G1_XML = ROOT / "third_party/mujoco_menagerie/unitree_g1/g1.xml"
PELVIS_BODY = "pelvis"
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
MAX_BASE_SPEED = 2.5   # m/s clip; kills flight-phase spikes (true max ~1.3)


def _skew_cross(w, r):
    return np.cross(w, r)


class LegOdometry:
    """FK/Jacobian-based base velocity + height from joint state, using the G1 MuJoCo model.

    joint_order: the 29-name order the caller's q/dq use (policy_meta joint_order_29dof).
    We remap it to the model's actuated-joint order once, so q[i] always lands on the
    right model DOF regardless of ordering.
    """

    def __init__(self, joint_order: list[str], xml: Path = G1_XML):
        import mujoco
        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, PELVIS_BODY)
        self.foot_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                         for b in FOOT_BODIES]
        assert self.pelvis_id >= 0 and all(f >= 0 for f in self.foot_ids)

        # Map caller joint_order (29) -> model qpos/qvel addresses for those joints.
        self.nu = len(joint_order)
        self.qpos_adr = np.zeros(self.nu, int)
        self.qvel_adr = np.zeros(self.nu, int)
        for i, name in enumerate(joint_order):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"joint '{name}' not in {xml.name}")
            self.qpos_adr[i] = self.model.jnt_qposadr[jid]
            self.qvel_adr[i] = self.model.jnt_dofadr[jid]
        # free-base addresses (first joint is the floating base)
        self._has_free = self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE

    def _fk(self, q):
        """Place the base at the origin (identity) and the 29 joints at q, forward-kinematics.
        With the base at origin/identity, world frame == pelvis frame."""
        d = self.data
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        if self._has_free:
            d.qpos[3] = 1.0  # base quat w=1 (wxyz), pos=0
        d.qpos[self.qpos_adr] = q
        self._mj.mj_kinematics(self.model, d)
        self._mj.mj_comPos(self.model, d)  # needed for jacobians

    def estimate(self, q, dq, R_base, gyro_body):
        """Return (base_lin_vel_body[3], base_height[m], info).

        q,dq: 29 joint pos/vel (caller order). R_base: 3x3 body->world (from IMU quat).
        gyro_body: base angular velocity in the body frame (IMU gyro).
        """
        q = np.asarray(q, float); dq = np.asarray(dq, float)
        self._fk(q)
        d = self.data
        w = np.asarray(gyro_body, float)
        per_foot_v, weights, foot_world_z = [], [], []
        jacp = np.zeros((3, self.model.nv))
        for fid in self.foot_ids:
            r_f = d.xpos[fid].copy()                     # foot pos in pelvis frame
            self._mj.mj_jacBody(self.model, d, jacp, None, fid)
            J_leg = jacp[:, self.qvel_adr]               # 3x29 foot lin-vel Jacobian (joints)
            foot_vel_from_joints = J_leg @ dq            # foot vel in pelvis frame from joints
            v_pelvis = -(foot_vel_from_joints + _skew_cross(w, r_f))  # pelvis vel, body frame
            per_foot_v.append(v_pelvis)
            fz_world = float((R_base @ r_f)[2])          # foot height rel. pelvis, in world
            foot_world_z.append(fz_world)
        # contact weight: the LOWER foot (more negative world z) is more planted.
        fz = np.array(foot_world_z)
        # softmax on -fz (temperature 0.03 m): sharp but smooth switch
        s = np.exp(-(fz - fz.min()) / 0.03)
        weights = s / s.sum()
        v_body = weights[0] * per_foot_v[0] + weights[1] * per_foot_v[1]
        # Clip to a physical bound: during flight/fast-swing (no clean stance) the estimate
        # can spike (~3 m/s vs a true ~1.2). The policy tolerates ±0.5 zero-mean noise but a
        # one-tick garbage spike is worth killing. True base speed maxes ~1.3 m/s -> ±2.5 is
        # ample headroom while capping the ~2% swing-phase outliers.
        v_body = np.clip(v_body, -MAX_BASE_SPEED, MAX_BASE_SPEED)
        # base height above flat ground = -(world-z of the stance foot rel. pelvis)
        base_height = -float(np.dot(weights, fz))
        info = {"weights": weights, "foot_world_z_rel": fz,
                "per_foot_v": per_foot_v}
        return v_body, base_height, info
