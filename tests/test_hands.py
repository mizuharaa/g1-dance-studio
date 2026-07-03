"""Tests for the hand expressiveness + SAFETY collision gate (pipeline/hands.py).

The collision-gate tests need the menagerie G1+hands model; they skip cleanly if
third_party clones aren't present (e.g. bare CI).
"""
import numpy as np
import pytest

from pipeline import hands


# ---- authored pose library + track (no model needed) ------------------------
def test_inspire_poses_are_within_range():
    for name, pose in hands.INSPIRE_POSES.items():
        assert len(pose) == hands.INSPIRE_DOF, name
        assert all(hands.INSPIRE_MIN <= v <= hands.INSPIRE_MAX for v in pose), name


def test_clamp_inspire():
    assert hands.clamp_inspire([-50, 2000, 500, 0, 1000, 1001]) == [0, 1000, 500, 0, 1000, 1000]


def test_hand_track_interpolates_and_holds():
    kf = [
        hands.HandKeyframe(0.0, [0] * 6, [0] * 6),
        hands.HandKeyframe(1.0, [1000] * 6, [1000] * 6),
    ]
    track = hands.build_hand_track(kf, n_frames=61, fps=30.0)
    assert track.shape == (61, 12)
    assert np.allclose(track[0], 0)             # held at start
    assert np.allclose(track[15], 500, atol=20)  # ~halfway at t=0.5s
    assert np.allclose(track[30], 1000)          # second keyframe at t=1.0s
    assert np.allclose(track[-1], 1000)          # held at end


def test_hand_track_lead_in_shifts_keyframes():
    kf = [hands.HandKeyframe(0.0, [0] * 6, [0] * 6),
          hands.HandKeyframe(1.0, [1000] * 6, [1000] * 6)]
    # with a 1.0s lead-in, the first pose is held through the standing intro
    track = hands.build_hand_track(kf, n_frames=61, fps=30.0, lead_in_s=1.0)
    assert np.allclose(track[0], 0)
    assert np.allclose(track[30], 0)   # t=1.0s still the first pose (shifted)


# ---- the collision gate -----------------------------------------------------
def _has_model() -> bool:
    return hands.DEX3_MJCF.exists()


requires_model = pytest.mark.skipif(not _has_model(), reason="menagerie G1+hands model absent")


@requires_model
def test_gate_passes_open_hands():
    import mujoco
    m = hands.HandModel().load()
    ndof = sum(1 for j in range(m.njnt)
               if "hand" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""))
    track = np.zeros((3, ndof))  # neutral / open
    r = hands.check_hand_collisions(track, hands.HandModel())
    assert r["passed"] is True
    assert r["self_collisions"] == 0 and r["body_collisions"] == 0


@requires_model
def test_gate_flags_curled_self_collision():
    import mujoco
    m = hands.HandModel().load()
    ranges = [m.jnt_range[j] for j in range(m.njnt)
              if "hand" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "")]
    mx = np.array([(hi if abs(hi) >= abs(lo) else lo) for lo, hi in ranges])
    track = np.tile(mx, (3, 1))  # fingers driven fully into a fist
    r = hands.check_hand_collisions(track, hands.HandModel())
    assert r["passed"] is False
    assert r["self_collisions"] > 0
    assert r["first_self_frame"] == 0


@requires_model
def test_gate_flags_joint_limit_violation():
    import mujoco
    m = hands.HandModel().load()
    ranges = [m.jnt_range[j] for j in range(m.njnt)
              if "hand" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "")]
    mx = np.array([(hi if abs(hi) >= abs(lo) else lo) for lo, hi in ranges])
    track = np.tile(mx * 5, (2, 1))  # way past limits
    r = hands.check_hand_collisions(track, hands.HandModel())
    assert r["limit_violations"] > 0


@requires_model
def test_gate_reports_substrate_is_not_inspire():
    r = hands.check_hand_collisions(np.zeros((1, 14)), hands.HandModel())
    assert r["certifies_real_inspire"] is False
    assert "Inspire" in r["note"]
