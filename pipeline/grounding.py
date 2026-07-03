"""Ground-reference a G1 motion so absolute-z safety tests are meaningful.

The audit found a HIGH safety-gate bug: vet_motion's no-floorwork (HARD-3) and
foot-skate checks, and find_window's window selection, all compare against an
absolute floor (z=0) — but nothing grounded the motion first. GMR retargeting
runs with ``offset_to_ground=False`` (so root z carries GVHMR's global
translation), meaning a genuine deep-squat could pass HARD-3 (or a downward
offset could empty the deployable window) purely because the floor wasn't at 0.

The only grounding code in the tree was ``prep_motion._min_height_fk`` — an
orphan never wired into the automated pipeline. This module promotes it to a
shared helper used at retarget intake (and defensively inside vet), so the
gate always sees a floor-referenced motion. Grounding is idempotent: grounding
an already-grounded motion shifts it by ~0.

CSV convention: 36 cols, 0:3 root xyz, 3:7 root quat (xyzw), 7:36 joints.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT

MODEL_XML = PROJECT_ROOT / "third_party/mujoco_menagerie/unitree_g1/scene.xml"

# If the un-grounded lowest contact point is further than this from the floor, the
# input almost certainly wasn't ground-referenced (e.g. raw GMR output). Callers
# surface it as an advisory so a silently un-grounded motion can't slip through.
UNGROUNDED_FLAG_M = 0.05


@lru_cache(maxsize=1)
def _model():
    import mujoco
    return mujoco.MjModel.from_xml_path(str(MODEL_XML))


def min_contact_height(motion: np.ndarray, model=None) -> float:
    """Lowest z of any ROBOT geom over the trajectory (world/floor geoms excluded).

    This is the FK-based true floor contact, not the root height."""
    import mujoco
    model = model or _model()
    data = mujoco.MjData(model)
    robot_geoms = np.flatnonzero(model.geom_bodyid != 0)
    zmin = np.inf
    for row in motion:
        data.qpos[:3] = row[:3]
        data.qpos[3:7] = row[[6, 3, 4, 5]]  # xyzw -> wxyz
        data.qpos[7:] = row[7:]
        mujoco.mj_kinematics(model, data)
        zmin = min(zmin, float(data.geom_xpos[robot_geoms, 2].min()))
    return zmin


def ground_motion(motion: np.ndarray, model=None) -> tuple[np.ndarray, float]:
    """Return (grounded_copy, shift_m): the motion with root z shifted so the
    lowest robot geom sits on z=0. shift_m is the amount subtracted (the
    un-grounded contact height); |shift_m| large ⇒ the input wasn't grounded.

    Idempotent: re-grounding a grounded motion returns shift≈0."""
    zmin = min_contact_height(motion, model)
    out = motion.copy()
    out[:, 2] -= zmin
    return out, zmin


def have_model() -> bool:
    return MODEL_XML.exists()
