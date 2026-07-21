"""Pure actuator-scope resolution for the F1 effort-randomization fix (task A1,
tasks/audit_fixes_20260721/A1_effort_scope_F1.md).

WHY THIS EXISTS: mjlab's dr.effort_limits selects asset_cfg.actuator_ids and
NEVER reads joint_names (pinned-wheel evidence in
experiments/external_audit_20260721/PINNED_MJLAB_EVIDENCE.md) — the old
joint_names-scoped events silently applied to every actuator group. The fixed
event (cloud/sim2real_task_v8.scoped_effort_limits) needs exact per-CONTROL ids
for the ankle vs non-ankle split. That resolution is pure logic over the
asset's actuator groups, so it lives here mjlab-free and is CPU-unit-tested in
tests/test_effort_scope.py.

Lane A addition (logged in the task-pack README coordination log). No mjlab
imports allowed in this file.
"""
from __future__ import annotations

import re

# The 29 G1 joints in LAFAN1 order (inlined: box-side cloud/ scripts cannot
# import pipeline/g1_limits). Used by the v8 selfcheck's intended-band table.
G1_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# Attributes an mjlab actuator group may expose its joint-target names under
# (defensive across minor versions; resolution FAILS LOUD if none is present).
_NAME_ATTRS = ("target_names", "joint_names", "names")


def _names_of(actuator) -> list[str]:
    for attr in _NAME_ATTRS:
        v = getattr(actuator, attr, None)
        if v:
            return [str(x) for x in v]
    raise RuntimeError(
        "effort_scope: actuator group %r exposes none of %s — cannot resolve "
        "per-control scoping; refuse rather than randomize blindly (F1)."
        % (type(actuator).__name__, _NAME_ATTRS))


def _ctrl_ids_of(actuator) -> list[int]:
    ids = getattr(actuator, "global_ctrl_ids", None)
    if ids is None:
        raise RuntimeError(
            "effort_scope: actuator group %r has no global_ctrl_ids" %
            type(actuator).__name__)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(i) for i in ids]


def _is_ankle(name: str, ankle_patterns) -> bool:
    return any(p == name or re.fullmatch(p, name) for p in ankle_patterns)


def resolve_effort_scopes(actuators, ankle_joint_names) -> dict:
    """Split every control id across all actuator groups into ankle vs
    non-ankle, by each group's joint-target names.

    Handles a group owning multiple controls: when the group's name list and
    ctrl-id list have equal length, controls are classified per-position; a
    length mismatch is tolerated ONLY for an unmixed group (all ankle or all
    non-ankle) and is otherwise a hard error.

    Asserts (hard, fail-loud — this feeds torque randomization):
      * every ankle joint name matched exactly once across all groups;
      * ankle control count == len(ankle_joint_names);
      * no control id appears in both scopes / twice.
    Returns {"ankle_ctrl_ids": [...], "non_ankle_ctrl_ids": [...],
             "groups": [{"type": ..., "n_ctrl": ..., "ankle": bool|"mixed"}]}.
    """
    ankle_ids: list[int] = []
    other_ids: list[int] = []
    matched: dict[str, int] = {}
    groups = []
    for act in actuators:
        names = _names_of(act)
        ids = _ctrl_ids_of(act)
        flags = [_is_ankle(n, ankle_joint_names) for n in names]
        if len(names) == len(ids):
            for n, i, f in zip(names, ids, flags):
                (ankle_ids if f else other_ids).append(i)
                if f:
                    matched[n] = matched.get(n, 0) + 1
            kind = "mixed" if (any(flags) and not all(flags)) else all(flags)
        elif not any(flags) or all(flags):
            # unmixed group: classify wholesale despite the length mismatch
            (ankle_ids if all(flags) else other_ids).extend(ids)
            for n, f in zip(names, flags):
                if f:
                    matched[n] = matched.get(n, 0) + 1
            kind = all(flags)
        else:
            raise RuntimeError(
                "effort_scope: group %r mixes ankle and non-ankle joints but "
                "names (%d) != ctrl ids (%d) — cannot split safely" %
                (type(act).__name__, len(names), len(ids)))
        groups.append({"type": type(act).__name__, "n_ctrl": len(ids),
                       "ankle": kind})

    want = [p for p in ankle_joint_names]
    missing = [p for p in want
               if not any(p == n or re.fullmatch(p, n) for n in matched)]
    if missing:
        raise RuntimeError(f"effort_scope: ankle joints not found in any "
                           f"actuator group: {missing}")
    dupes = [n for n, c in matched.items() if c != 1]
    if dupes:
        raise RuntimeError(f"effort_scope: ankle joints matched != once: {dupes}")
    if len(ankle_ids) != len(want):
        raise RuntimeError(
            f"effort_scope: resolved {len(ankle_ids)} ankle controls for "
            f"{len(want)} ankle joints — refusing")
    overlap = set(ankle_ids) & set(other_ids)
    if overlap or len(set(ankle_ids)) != len(ankle_ids) \
            or len(set(other_ids)) != len(other_ids):
        raise RuntimeError(f"effort_scope: duplicate/overlapping ctrl ids "
                           f"(overlap={sorted(overlap)})")
    return {"ankle_ctrl_ids": sorted(ankle_ids),
            "non_ankle_ctrl_ids": sorted(other_ids),
            "groups": groups}
