"""A1/F1 regression: per-control effort-scope resolution (cloud/effort_scope.py).
The mjlab dr.effort_limits joint_names trap (external audit F1) is prevented by
resolving exact control ids; these tests pin that pure logic without mjlab."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cloud"))
import effort_scope  # noqa: E402

ANKLES = ("left_ankle_pitch_joint", "right_ankle_pitch_joint",
          "left_ankle_roll_joint", "right_ankle_roll_joint")


class FakeAct:
    def __init__(self, names, ids):
        self.target_names = list(names)
        self.global_ctrl_ids = list(ids)


def g1_like_groups():
    """Mimic the pinned wheel's 6 high-level groups (one owns many controls)."""
    return [
        FakeAct(["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                 "right_wrist_roll_joint"], [15, 16, 26]),        # 5020 (multi)
        FakeAct(["left_hip_pitch_joint", "right_hip_yaw_joint"], [0, 8]),
        FakeAct(["left_knee_joint", "right_hip_roll_joint"], [3, 7]),
        FakeAct(["left_wrist_pitch_joint", "right_wrist_yaw_joint"], [20, 28]),
        FakeAct(["waist_yaw_joint", "waist_roll_joint"], [12, 13]),
        FakeAct(list(ANKLES), [4, 10, 5, 11]),                    # ankle group
    ]


def test_ankle_ctrls_resolve_exactly():
    s = effort_scope.resolve_effort_scopes(g1_like_groups(), ANKLES)
    assert s["ankle_ctrl_ids"] == [4, 5, 10, 11]
    assert set(s["ankle_ctrl_ids"]).isdisjoint(s["non_ankle_ctrl_ids"])
    assert len(s["non_ankle_ctrl_ids"]) == 11  # every listed non-ankle ctrl


def test_mixed_group_splits_per_position():
    # one group owning ankle AND non-ankle controls must split by position
    acts = [FakeAct(["left_ankle_pitch_joint", "left_knee_joint",
                     "right_ankle_pitch_joint", "left_ankle_roll_joint",
                     "right_ankle_roll_joint"], [4, 3, 10, 5, 11])]
    s = effort_scope.resolve_effort_scopes(acts, ANKLES)
    assert s["ankle_ctrl_ids"] == [4, 5, 10, 11]
    assert s["non_ankle_ctrl_ids"] == [3]


def test_missing_ankle_group_raises():
    with pytest.raises(RuntimeError, match="not found"):
        effort_scope.resolve_effort_scopes(
            [FakeAct(["left_knee_joint"], [3])], ANKLES)


def test_renamed_attr_and_no_names_raises():
    class Bare:
        global_ctrl_ids = [1]
    with pytest.raises(RuntimeError, match="exposes none"):
        effort_scope.resolve_effort_scopes([Bare()], ANKLES)


def test_mixed_group_with_length_mismatch_refuses():
    a = FakeAct(["left_ankle_pitch_joint", "left_knee_joint"], [4])
    with pytest.raises(RuntimeError, match="cannot split safely"):
        effort_scope.resolve_effort_scopes([a], ANKLES)


def test_duplicate_ankle_across_groups_refuses():
    acts = [FakeAct(list(ANKLES), [4, 10, 5, 11]),
            FakeAct(["left_ankle_pitch_joint"], [4])]
    with pytest.raises(RuntimeError, match="!= once|overlap"):
        effort_scope.resolve_effort_scopes(acts, ANKLES)
