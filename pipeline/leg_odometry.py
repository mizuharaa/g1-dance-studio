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
MAX_BASE_SPEED = 2.5   # m/s hard clip; last-resort bound (true max ~1.3)
# Temporal smoothing: the TRUE base velocity is smooth (max ~1.2 m/s), so any fast jump in
# the raw kinematic estimate is a swing-phase spike, not real motion. A single such spike fed
# to the policy caused a sudden lateral "acrobatic" move on hardware (2026-07-04, 30s run).
# EMA + per-tick rate limit rejects those spikes while tracking the real (slow) velocity.
VEL_EMA_ALPHA = 0.35        # EMA weight on the new raw sample (lower = smoother)
VEL_MAX_STEP = 0.30         # m/s max change per 50 Hz tick (~15 m/s^2 — above real, kills spikes)


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
        self._v_smooth = None  # EMA state for spike-rejected base velocity

    def reset_filter(self):
        """Clear the velocity smoother — call at the start of each run so a stale value
        from a prior segment can't leak in."""
        self._v_smooth = None

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

    def gravity_comp(self, q_target, R_base=None):
        """Feedforward gravity-compensation torque (29-vec, caller joint order) to HOLD the
        pose q_target. Computed via MuJoCo inverse dynamics with the base pinned at the given
        orientation (from the IMU, so torso tilt is accounted for), feet supporting the load.
        This is the torque the sim's position actuator provides implicitly; sending it as the
        real robot's tau feedforward lets the legs hold the pose at the TRAINED gains (no boost),
        so the ankle only carries its true ~0.2 Nm load instead of 20 Nm of PD-fighting-sag heat.
        """
        d = self.data
        d.qpos[:] = 0.0
        if self._has_free:
            if R_base is not None:
                # base quat (wxyz) from the rotation matrix, so gravity is in the torso frame
                q_wxyz = np.empty(4)
                self._mj.mju_mat2Quat(q_wxyz, np.asarray(R_base, float).reshape(-1))
                d.qpos[3:7] = q_wxyz
            else:
                d.qpos[3] = 1.0
        d.qpos[self.qpos_adr] = np.asarray(q_target, float)
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        self._mj.mj_inverse(self.model, d)
        return np.array([d.qfrc_inverse[a] for a in self.qvel_adr], float)

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
        # Spike rejection + EMA: the true velocity is smooth, so rate-limit the change per
        # tick (kills single-frame swing spikes) then low-pass. First sample seeds the state.
        if self._v_smooth is None:
            self._v_smooth = v_body.copy()
        else:
            step = np.clip(v_body - self._v_smooth, -VEL_MAX_STEP, VEL_MAX_STEP)
            v_rate_limited = self._v_smooth + step
            self._v_smooth = (VEL_EMA_ALPHA * v_rate_limited
                              + (1.0 - VEL_EMA_ALPHA) * self._v_smooth)
        v_body = self._v_smooth.copy()
        # base height above flat ground = -(world-z of the stance foot rel. pelvis)
        base_height = -float(np.dot(weights, fz))
        # per-foot kinematic base height (each foot's own read) — the fused estimator picks
        # the height of whichever foot it judges planted, not the blended value.
        per_foot_h = [-float(z) for z in fz]
        info = {"weights": weights, "foot_world_z_rel": fz,
                "per_foot_v": per_foot_v,          # base vel (body frame) implied by each foot
                "per_foot_h": per_foot_h}          # base height implied by each foot
        return v_body, base_height, info


# Fusion constants. IMU carries the estimate through flight; kinematics anchors it during
# stance. Contact is detected from each foot's IMPLIED world velocity: a planted foot is
# (near) stationary in the world, so |v_world + R·(foot-vel-from-joints)| ~ 0.
GRAVITY = 9.81
V_CONTACT = 0.30           # m/s scale for "planted" (foot world speed below this => planted)
K_V_CORRECT = 0.25         # per-tick pull of velocity toward the planted-foot kinematic
K_H_CORRECT = 0.15         # per-tick pull of height toward the planted-foot kinematic
ACCEL_CLAMP = 40.0         # m/s^2 sanity clamp on IMU-derived world accel


class FusedBaseEstimator:
    """Complementary filter for base velocity + height that survives STEPPING.

    Pure leg (kinematic) odometry degrades when a foot lifts — the planted-foot assumption
    breaks, so velocity AND height go bad for the ~0.4 s of a step, which made the policy
    throw a whole-body brace on hardware (2026-07-04). This fuses:
      * PREDICT with the IMU: integrate gravity-removed world acceleration -> carries the
        estimate through flight, when kinematics is blind.
      * CORRECT with kinematics, weighted by CONTACT CONFIDENCE: when a foot is solidly
        planted (its implied world velocity ~ 0), pull the estimate toward that foot's
        kinematic read; when both feet are moving (mid-step), trust the IMU integration.
    Drift from integration is bounded because every foot-plant re-anchors it. Wraps a
    LegOdometry instance; call reset() at the start of each run.
    """

    def __init__(self, leg_odom: "LegOdometry"):
        self.legodom = leg_odom
        self._v_world = None       # fused base velocity, WORLD frame
        self._h = None             # fused base height
        self._prev_planted = 1.0

    def reset(self):
        self._v_world = None
        self._h = None
        self.legodom.reset_filter()

    def estimate(self, q, dq, R_base, gyro_body, accel_body, dt):
        """Return (base_lin_vel_body[3], base_height[m], info). accel_body = IMU accelerometer
        (specific force, body frame, incl. gravity reaction). dt = tick period (s)."""
        # Raw per-foot kinematic reads (bypass the EMA — the filter does its own smoothing).
        _, _, li = self.legodom.estimate(q, dq, R_base, gyro_body)
        per_foot_v = li["per_foot_v"]           # base vel (body) implied by each foot
        per_foot_h = li["per_foot_h"]           # base height implied by each foot
        R = np.asarray(R_base, float)

        # World acceleration from the IMU: a_world = R·f_body - g  (g points down).
        a_world = R @ np.asarray(accel_body, float) - np.array([0.0, 0.0, GRAVITY])
        a_world = np.clip(a_world, -ACCEL_CLAMP, ACCEL_CLAMP)

        # Seed on first call from the (blended) kinematic read.
        if self._v_world is None:
            v0, h0, _ = self.legodom.estimate(q, dq, R_base, gyro_body)
            self._v_world = R @ v0
            self._h = h0
            return R.T @ self._v_world, self._h, {"contact": 1.0, "a_world": a_world}

        # PREDICT (IMU integration).
        self._v_world = self._v_world + a_world * dt
        self._h = self._h + self._v_world[2] * dt

        # CONTACT DETECTION: each foot's implied WORLD velocity given the current fused base.
        # foot_world_vel_i = v_world - R·(per_foot_v_i)  [since per_foot_v_i = -(J·dq+w×r)].
        best_i, best_conf = 0, 0.0
        for i, vf in enumerate(per_foot_v):
            foot_world_vel = self._v_world - R @ np.asarray(vf, float)
            conf_i = float(np.exp(-np.linalg.norm(foot_world_vel) / V_CONTACT))
            if conf_i > best_conf:
                best_conf, best_i = conf_i, i

        # CORRECT toward the most-planted foot, scaled by its confidence.
        v_kin_world = R @ np.asarray(per_foot_v[best_i], float)
        self._v_world = self._v_world + K_V_CORRECT * best_conf * (v_kin_world - self._v_world)
        self._h = self._h + K_H_CORRECT * best_conf * (per_foot_h[best_i] - self._h)
        self._prev_planted = best_conf

        return R.T @ self._v_world, self._h, {"contact": best_conf, "a_world": a_world,
                                              "planted_foot": best_i}
