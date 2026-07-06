"""Tests for the 2026-07-05 audit deploy fixes (offline — no robot, no SDK):

  * yaw re-anchoring of the reference world frame (Reference.align_yaw) — the npz
    frame's t=0 yaw is 90.3 deg, so unaligned anchor_ori was far out of the training
    distribution unless the robot happened to boot facing that heading;
  * torso-anchor orientation (_anchor_quat = pelvis IMU quat composed with waist FK),
    validated against MuJoCo forward kinematics on the menagerie G1;
  * run telemetry recorder (auditable hardware numbers).
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

dr = pytest.importorskip("pipeline.deploy_runtime")

Q0 = np.array([1.0, 0, 0, 0])


def _fixt():
    return dr.Meta(dr.DEFAULT_META), dr.Reference(dr.DEFAULT_MOTION)


def _quat_yaw(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _ori_error_angle(q_ref, q_rob):
    m = dr.quat_wxyz_to_mat(q_rob).T @ dr.quat_wxyz_to_mat(q_ref)
    return float(np.arccos(np.clip((np.trace(m) - 1) / 2, -1, 1)))


# ---- yaw re-anchoring ----------------------------------------------------------

def test_reference_frame0_yaw_is_far_from_identity():
    """Documents the bug: the npz world frame's t=0 torso yaw is ~90 deg, so a robot
    booted at yaw 0 saw a huge anchor_ori error under the old (unaligned) code."""
    _, ref = _fixt()
    yaw0 = dr.yaw_of_quat_wxyz(ref.aquat[0])
    assert abs(np.degrees(yaw0)) > 45  # 90.3 deg measured in the audit


def test_align_yaw_zeroes_initial_yaw_error():
    meta, ref = _fixt()
    robot_q = _quat_yaw(np.deg2rad(137.0))
    before = _ori_error_angle(ref.aquat[0], robot_q)
    dyaw = ref.align_yaw(robot_q)
    after = _ori_error_angle(ref.aquat[0], robot_q)
    assert before > np.deg2rad(40)          # the OOD error the old code shipped
    assert after < np.deg2rad(8)            # only the ref's own t=0 pitch/roll remains
    assert np.isclose(dr.yaw_of_quat_wxyz(ref.aquat[0]), dr.yaw_of_quat_wxyz(robot_q),
                      atol=1e-6)
    assert ref.yaw_offset == pytest.approx(dyaw)


def test_align_yaw_rotates_displacements_preserves_heights_and_origin():
    _, ref = _fixt()
    apos_before = ref.apos.copy()
    dyaw = ref.align_yaw(_quat_yaw(np.deg2rad(90.0)) )
    c, s = np.cos(dyaw), np.sin(dyaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    disp_before = apos_before - apos_before[0]
    disp_after = ref.apos - ref.apos[0]
    assert np.allclose(disp_after, disp_before @ Rz.T, atol=1e-9)
    assert np.allclose(ref.apos[0], apos_before[0])          # origin unchanged
    assert np.allclose(ref.apos[:, 2], apos_before[:, 2])    # heights unchanged
    # displacement magnitudes preserved (pure yaw rotation)
    assert np.allclose(np.linalg.norm(disp_after, axis=1),
                       np.linalg.norm(disp_before, axis=1), atol=1e-9)


def test_anchor_ori_term_invariant_to_boot_heading_after_align():
    """The whole point: two robots facing different directions must see the SAME obs
    after alignment (the dance is heading-relative, as in training)."""
    meta, _ = _fixt()
    terms = []
    for yaw_deg in (0.0, 90.3, -134.0):
        ref = dr.Reference(dr.DEFAULT_MOTION)
        robot_q = _quat_yaw(np.deg2rad(yaw_deg))
        ref.align_yaw(robot_q)
        tick = 300
        t = dr.mat_first_two_cols_b(ref.aquat[tick], robot_q)
        terms.append(t)
    assert np.allclose(terms[0], terms[1], atol=1e-8)
    assert np.allclose(terms[0], terms[2], atol=1e-8)


# ---- torso anchor (waist FK) ----------------------------------------------------

def test_anchor_quat_reduces_to_pelvis_at_zero_waist():
    meta, _ = _fixt()
    q = meta.default.copy()
    for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
        i = meta.waist_idx[name]
        if i is not None:
            q[i] = 0.0
    robot_q = _quat_yaw(0.7)
    out = dr._anchor_quat(meta, q, robot_q)
    assert np.allclose(out, robot_q, atol=1e-12)


def test_waist_fk_matches_mujoco():
    """_anchor_quat's quaternion chain (yaw z, roll x, pitch y) must equal MuJoCo FK
    torso orientation on the menagerie G1 for random waist angles + base quats."""
    mujoco = pytest.importorskip("mujoco")
    meta, _ = _fixt()
    xml = dr.ROOT / "third_party/mujoco_menagerie/unitree_g1/g1.xml"
    if not xml.exists():
        pytest.skip("menagerie G1 model not present")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    rng = np.random.default_rng(7)
    for _ in range(10):
        base_q = rng.normal(size=4)
        base_q /= np.linalg.norm(base_q)
        if base_q[0] < 0:
            base_q = -base_q
        wy, wr, wp = rng.uniform(-0.4, 0.4, 3)
        data.qpos[:] = 0
        data.qpos[3:7] = base_q                     # freejoint quat (wxyz)
        q29 = np.zeros(29)
        for name, val in (("waist_yaw_joint", wy), ("waist_roll_joint", wr),
                          ("waist_pitch_joint", wp)):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = val
            q29[meta.waist_idx[name]] = val
        mujoco.mj_kinematics(model, data)
        expected = data.xquat[torso_id]             # wxyz
        got = dr._anchor_quat(meta, q29, base_q)
        # quats are sign-ambiguous — compare |dot| ~ 1
        assert abs(float(np.dot(expected, got))) > 1 - 1e-9


def test_build_obs_uses_torso_frame_when_waist_bent(monkeypatch):
    """Bending the waist must change the anchor terms (torso frame), and must NOT
    change base_ang_vel (pelvis gyro passthrough)."""
    meta, ref = _fixt()
    q = meta.default.copy()
    _, t_straight = dr.build_obs(meta, ref, q, np.zeros(29), Q0, np.ones(3) * 0.1,
                                 np.zeros(29), tick=100)
    q_bent = q.copy()
    q_bent[meta.waist_idx["waist_pitch_joint"]] = 0.4
    _, t_bent = dr.build_obs(meta, ref, q_bent, np.zeros(29), Q0, np.ones(3) * 0.1,
                             np.zeros(29), tick=100)
    assert not np.allclose(t_straight["motion_anchor_ori_b"], t_bent["motion_anchor_ori_b"])
    assert np.allclose(t_straight["base_ang_vel"], t_bent["base_ang_vel"])
    # pelvis fallback restores the old behaviour
    monkeypatch.setattr(dr, "TORSO_ANCHOR", False)
    _, t_fallback = dr.build_obs(meta, ref, q_bent, np.zeros(29), Q0, np.ones(3) * 0.1,
                                 np.zeros(29), tick=100)
    _, t_fallback0 = dr.build_obs(meta, ref, q, np.zeros(29), Q0, np.ones(3) * 0.1,
                                  np.zeros(29), tick=100)
    assert np.allclose(t_fallback["motion_anchor_ori_b"], t_fallback0["motion_anchor_ori_b"])


# ---- telemetry -------------------------------------------------------------------

def _fake_msg():
    ms = [SimpleNamespace(tau_est=float(i), temperature=[40 + i, 41 + i])
          for i in range(29)]
    return SimpleNamespace(motor_state=ms)


def test_telemetry_records_and_saves(tmp_path, monkeypatch):
    meta, _ = _fixt()
    monkeypatch.setattr(dr, "TELEMETRY", True)
    monkeypatch.setattr(dr, "TELEMETRY_DIR", tmp_path)
    telem = dr.Telemetry("test-mode", meta, extra={"note": "unit"})
    telem.path = tmp_path / "t.npz"
    for tick in range(5):
        telem.add(tick, meta.default, np.zeros(29), _fake_msg(), Q0, np.zeros(3),
                  np.full(29, 0.5), meta.default)
    telem.save(quiet=True)
    d = np.load(tmp_path / "t.npz")
    assert d["tau_est"].shape == (5, 29) and d["tau_est"][0, 3] == 3.0
    assert d["temp"].shape == (5, 29) and d["temp"][0, 0] == 40.0
    assert d["q"].shape == (5, 29) and d["action"].shape == (5, 29)
    rm = json.loads(str(d["run_meta_json"]))
    assert rm["mode"] == "test-mode" and rm["note"] == "unit"


def test_telemetry_add_never_raises(monkeypatch):
    meta, _ = _fixt()
    telem = dr.Telemetry("test-mode", meta)
    telem.add(0, None, None, object(), None, None, None, None)  # garbage in
    assert telem.rows["tick"] == []  # dropped, not raised


def test_ground_max_action_default_is_measured_need():
    import os
    if "GROUND_MAX_ACTION" not in os.environ:
        assert dr.GROUND_MAX_ACTION == 10.0


def test_action_cap_vector_per_joint():
    """Arm joints get scaled headroom (wrist choreography rides 10-12 units by
    design); legs/waist keep the tight runaway tripwire (HW measured legs <=3.4)."""
    meta = dr.Meta(dr.DEFAULT_META)
    cap = dr.action_cap_vector(meta, 10.0)
    for i, name in enumerate(meta.joint_order):
        if any(t in name for t in ("shoulder", "elbow", "wrist")):
            assert cap[i] == pytest.approx(10.0 * dr.ARM_ACTION_CAP_SCALE)
        else:
            assert cap[i] == pytest.approx(10.0)
    # the observed benign wrist action (12.12) passes; a leg at 12 trips
    a = np.zeros(29)
    a[meta.joint_order.index("right_wrist_yaw_joint")] = 12.12
    assert not np.any(np.abs(a) > cap)
    a2 = np.zeros(29)
    a2[meta.joint_order.index("left_knee_joint")] = 12.0
    assert np.any(np.abs(a2) > cap)
