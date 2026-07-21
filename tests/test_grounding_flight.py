"""Audit F8 regression tests: flight-aware, support-gated per-frame grounding.

The old ground_motion_per_frame low-passed the SAME contact signal it
classified, so a sustained jump plateau read as a new floor and was subtracted
to ~0 (experiments/external_audit_20260721/REPORT.md §F8). These synthetic
cases pin the fix:

  (a) standing + linear 0.15 m camera drift  -> drift removed, 0 flight
  (b) 2 s, 0.20 m jump plateau               -> height preserved, flagged flight
  (c) short 200 ms hop                       -> preserved + flagged
  (d) alternating single-support walk + drift-> drift removed, steps kept
  (e) crouch (feet grounded, root low)       -> NOT flagged as flight

All motions are tiny synthetic (N, 36) trajectories run through the REAL G1
model (pipeline.g1_limits.build_model — contact disabled, FK only). No
thriller/data dependency.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.g1_limits import MODEL_XML

pytestmark = pytest.mark.skipif(not MODEL_XML.exists(),
                                reason="needs the G1 mujoco model")

FPS = 30.0

# CSV convention: 36 cols = 0:3 root xyz | 3:7 root quat (xyzw) | 7:36 joints
L_HIP_PITCH, L_KNEE, L_ANKLE_PITCH = 7 + 0, 7 + 3, 7 + 4
R_HIP_PITCH, R_KNEE, R_ANKLE_PITCH = 7 + 6, 7 + 9, 7 + 10


@pytest.fixture(scope="module")
def model():
    from pipeline.g1_limits import build_model
    return build_model()


def _row(model, joints: dict[int, float] | None = None) -> np.ndarray:
    """One 36-col frame with identity orientation and the SOLES exactly on z=0
    (root z solved by FK so the lower foot-sole surface sits at 0)."""
    from pipeline.grounding import _fk_heights
    row = np.zeros(36)
    row[6] = 1.0                       # quat xyzw identity
    for col, val in (joints or {}).items():
        row[col] = val
    row[2] = 1.0
    sole = float(_fk_heights(row[None, :], model)[1].min())
    row[2] = 1.0 - sole
    return row


def _soles(model, motion: np.ndarray) -> np.ndarray:
    from pipeline.grounding import _fk_heights
    return _fk_heights(motion, model)[1]


# ---- sanity: sole-surface measurement, not geom centers --------------------------

def test_foot_geoms_resolved_and_sole_is_surface(model):
    from pipeline import grounding
    left, right = grounding._foot_geom_ids(model)
    assert left and right and not set(left) & set(right)
    # standing row built so soles sit at z=0: the surface-aware contact height
    # is ~0, and it sits one sole-sphere radius BELOW the old center-based min
    row = _row(model)[None, :]
    surface = float(grounding.per_contact_height(row, model)[0])
    assert abs(surface) < 1e-6
    import mujoco
    data = mujoco.MjData(model)
    data.qpos[:3] = row[0, :3]
    data.qpos[3:7] = row[0, [6, 3, 4, 5]]
    data.qpos[7:] = row[0, 7:]
    mujoco.mj_kinematics(model, data)
    center_min = float(data.geom_xpos[np.flatnonzero(model.geom_bodyid != 0), 2].min())
    assert center_min - surface >= 0.004   # sole spheres are r=5 mm

# ---- (a) camera drift ------------------------------------------------------------

def test_linear_drift_removed_no_flight(model):
    from pipeline.grounding import ground_motion_per_frame
    n = 120                                       # 4 s
    m = np.tile(_row(model), (n, 1))
    m[:, 2] += np.linspace(0.0, 0.15, n)          # slow vertical camera drift
    out, info = ground_motion_per_frame(m, model, fps=FPS)
    assert info["flight_frames"] == 0
    assert info["flight_windows_s"] == []
    assert float(np.ptp(out[:, 2])) <= 0.03       # 0.15 m of drift collapsed
    assert abs(info["floor_drift_m"] - 0.15) <= 0.03
    assert info["support_pct"] > 90.0
    assert info["grounded_start"]


# ---- (b) sustained 2 s jump plateau ----------------------------------------------

def test_two_second_jump_plateau_preserved_and_flagged(model):
    from pipeline.grounding import ground_motion_per_frame
    base = _row(model)
    stand_in, ramp, plateau, stand_out = 45, 3, 60, 45   # plateau = 2 s
    n = stand_in + ramp + plateau + ramp + stand_out
    m = np.tile(base, (n, 1))
    dz = np.zeros(n)
    a = stand_in
    dz[a:a + ramp] = np.linspace(0.0, 0.20, ramp + 2)[1:-1]
    dz[a + ramp:a + ramp + plateau] = 0.20
    dz[a + ramp + plateau:a + 2 * ramp + plateau] = np.linspace(0.20, 0.0, ramp + 2)[1:-1]
    m[:, 2] += dz
    out, info = ground_motion_per_frame(m, model, fps=FPS)
    z_stand = float(np.median(out[:stand_in, 2]))
    z_plateau = float(np.median(out[a + ramp:a + ramp + plateau, 2]))
    # the audit's failure mode: plateau subtracted to ~0. Now: kept within 2 cm.
    assert abs((z_plateau - z_stand) - 0.20) <= 0.02
    assert info["flight_frames"] >= plateau - 2
    assert len(info["flight_windows_s"]) == 1
    lo, hi = info["flight_windows_s"][0]
    assert lo <= (a + ramp) / FPS + 0.1 and hi >= (a + ramp + plateau) / FPS - 0.1
    # the floor never chased the plateau
    assert info["drift_removed_mm"] <= 10.0


# ---- (c) short 200 ms hop --------------------------------------------------------

def test_short_hop_preserved_and_flagged(model):
    from pipeline.grounding import ground_motion_per_frame
    base = _row(model)
    stand, hop = 30, 6                            # 6 frames = 200 ms
    n = stand + hop + stand
    m = np.tile(base, (n, 1))
    m[stand:stand + hop, 2] += 0.12
    out, info = ground_motion_per_frame(m, model, fps=FPS)
    z_stand = float(np.median(out[:stand, 2]))
    z_hop = float(np.median(out[stand:stand + hop, 2]))
    assert abs((z_hop - z_stand) - 0.12) <= 0.02
    assert info["flight_frames"] >= hop - 1
    assert len(info["flight_windows_s"]) == 1


# ---- (d) alternating single-support walk with drift ------------------------------

def test_alternating_support_walk_drift_removed_steps_kept(model):
    from pipeline.grounding import ground_motion_per_frame
    # pick the hip-pitch sign that actually LIFTS the foot on this model
    lift = {}
    for side, (hip, knee) in (("L", (L_HIP_PITCH, L_KNEE)),
                              ("R", (R_HIP_PITCH, R_KNEE))):
        for sign in (-1.0, 1.0):
            j = {hip: sign * 0.7, knee: 0.9}
            r = np.zeros(36)
            r[6] = 1.0
            r[2] = 1.0
            for c, v in j.items():
                r[c] = v
            s = _soles(model, r[None, :])[0]
            idx = 0 if side == "L" else 1
            if (s[idx] - s[1 - idx]) > 0.05:      # that foot is clearly higher
                lift[side] = j
                break
        assert side in lift, f"could not construct a {side} foot lift"
    base = _row(model)
    phase = 10                                    # frames per stance phase
    seq = ["both", "both", "L", "both", "R", "both", "L", "both", "R", "both"]
    n = phase * len(seq)
    m = np.tile(base, (n, 1))
    for k, ph in enumerate(seq):
        if ph in lift:
            for c, v in lift[ph].items():
                m[k * phase:(k + 1) * phase, c] = v
    m[:, 2] += np.linspace(0.0, 0.10, n)          # drift on top of the walk
    swing_before = _soles(model, m)               # designed step heights
    out, info = ground_motion_per_frame(m, model, fps=FPS)
    assert info["flight_frames"] == 0             # one foot always supports
    assert info["support_pct"] > 90.0
    assert float(np.ptp(out[:, 2]) - np.ptp(m[:, 2])) < 0.0  # drift shrank
    soles_after = _soles(model, out)
    for k, ph in enumerate(seq):
        sl = slice(k * phase + 2, (k + 1) * phase - 2)  # interior of the phase
        if ph in ("L", "R"):
            idx = 0 if ph == "L" else 1
            step_h = float(np.median(soles_after[sl, idx] - soles_after[sl, 1 - idx]))
            designed = float(np.median(swing_before[sl, idx] - swing_before[sl, 1 - idx]))
            assert abs(step_h - designed) < 1e-9   # pure z-shift: steps untouched
            assert step_h > 0.05                   # ... and NOT flattened
            assert abs(float(np.median(soles_after[sl, 1 - idx]))) <= 0.02  # support on floor
    assert float(np.ptp(out[:, 2])) <= 0.04       # 0.10 m drift removed


# ---- (e) crouch is not flight ----------------------------------------------------

def test_crouch_grounded_not_flight(model):
    from pipeline.grounding import ground_motion_per_frame
    stand_j: dict[int, float] = {}
    crouch_j = {L_HIP_PITCH: -0.5, L_KNEE: 1.0, L_ANKLE_PITCH: -0.5,
                R_HIP_PITCH: -0.5, R_KNEE: 1.0, R_ANKLE_PITCH: -0.5}
    stand = _row(model, stand_j)
    crouch = _row(model, crouch_j)
    assert stand[2] - crouch[2] >= 0.03           # the crouch really lowers the root
    ramp = 15
    rows = []
    rows += [stand] * 30
    for t in np.linspace(0.0, 1.0, ramp):         # feet stay planted: blend the
        j = {c: t * v for c, v in crouch_j.items()}   # joints, re-solve root z
        rows.append(_row(model, j))
    rows += [crouch] * 30
    for t in np.linspace(1.0, 0.0, ramp):
        j = {c: t * v for c, v in crouch_j.items()}
        rows.append(_row(model, j))
    rows += [stand] * 30
    m = np.stack(rows)
    out, info = ground_motion_per_frame(m, model, fps=FPS)
    assert info["flight_frames"] == 0             # feet never left the floor
    assert info["flight_windows_s"] == []
    assert info["support_pct"] > 95.0
    # crouch depth preserved in the output
    z_stand = float(np.median(out[:30, 2]))
    z_crouch = float(np.median(out[30 + ramp:30 + ramp + 30, 2]))
    assert (z_stand - z_crouch) >= (stand[2] - crouch[2]) - 0.02
    assert info["drift_removed_mm"] <= 10.0
