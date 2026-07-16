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

TWO grounding modes (2026-07-16 floaty-feet fix — REGISTRY 'distinct un-fixed
defect'):
  * ``ground_motion`` — a single GLOBAL z offset (plants the ONE lowest instant).
    Idempotent; kept for the vet gate's absolute-z checks. But it leaves the
    support foot FLOATING wherever the retarget's global translation drifts
    vertically over the clip — the §3.3 defect (Thriller: support foot >0.10 m
    off the floor in ~78 % of frames).
  * ``ground_motion_per_frame`` — removes the slow vertical drift so the support
    foot sits on z≈0 EVERY frame (float ~78 %→0 %). This is what the
    retarget-intake and show-prep steps now use before the motion reaches
    training/preview. Relative heights (root-above-foot) are preserved exactly.

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


def per_contact_height(motion: np.ndarray, model=None) -> np.ndarray:
    """Per-frame lowest z of any ROBOT geom (world/floor geoms excluded), as an
    (N,) array — the true FK floor-contact height at each frame (the height of
    whichever foot geom is lowest that frame, i.e. the support foot)."""
    import mujoco
    model = model or _model()
    data = mujoco.MjData(model)
    robot_geoms = np.flatnonzero(model.geom_bodyid != 0)
    out = np.empty(len(motion))
    for i, row in enumerate(motion):
        data.qpos[:3] = row[:3]
        data.qpos[3:7] = row[[6, 3, 4, 5]]  # xyzw -> wxyz
        data.qpos[7:] = row[7:]
        mujoco.mj_kinematics(model, data)
        out[i] = float(data.geom_xpos[robot_geoms, 2].min())
    return out


def min_contact_height(motion: np.ndarray, model=None) -> float:
    """Lowest z of any ROBOT geom over the WHOLE trajectory (world/floor geoms
    excluded) — a single scalar. This is the trajectory-wide floor contact used
    by the global (idempotent) ``ground_motion``; for per-frame grounding use
    ``per_contact_height`` / ``ground_motion_per_frame``."""
    return float(per_contact_height(motion, model).min())


def ground_motion(motion: np.ndarray, model=None) -> tuple[np.ndarray, float]:
    """Return (grounded_copy, shift_m): the motion with root z shifted by a
    SINGLE global offset so the lowest robot geom over the whole trajectory sits
    on z=0. shift_m is the amount subtracted (the un-grounded contact height);
    |shift_m| large ⇒ the input wasn't grounded.

    Idempotent: re-grounding a grounded motion returns shift≈0.

    NOTE: a single global offset only plants the ONE lowest instant. If the
    retarget's global translation drifts vertically over the clip (GVHMR/GMR
    routinely do — the estimated floor bobs), the support foot still FLOATS in
    most other frames (the §3.3 'floaty feet' defect). For a motion headed to
    training/preview use ``ground_motion_per_frame`` instead, which plants the
    support foot every frame. This global helper is kept for the vet gate's
    absolute-z idempotency check and as a building block."""
    zmin = min_contact_height(motion, model)
    out = motion.copy()
    out[:, 2] -= zmin
    return out, zmin


def _sg_smooth_1d(x: np.ndarray, window: int, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay low-pass along a 1-D signal (numpy-only, no scipy — this
    module stays importable in a bare env). Fits a local polynomial in a sliding
    window and evaluates it at the centre; edges use edge padding."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if window % 2 == 0:
        window += 1
    if n < window:
        return x.copy()
    half = window // 2
    j = np.arange(-half, half + 1)
    A = np.vander(j, poly + 1, increasing=True)
    coef = np.linalg.pinv(A)[0]              # row that evaluates the fit at j=0
    xp = np.pad(x, (half, half), mode="edge")
    out = np.empty(n)
    for i in range(n):
        out[i] = coef @ xp[i:i + window]
    return out


# Per-frame grounding parameters (see ground_motion_per_frame). Tuned + verified
# on the Thriller reference (experiments/grounding_fix/); the SG window is short
# on purpose so the ground line tracks the drift without lagging into penetration.
GROUND_SMOOTH_WIN = 9          # frames (~0.3 s at 30 fps) — the drift low-pass
FLIGHT_BAND_M = 0.08           # contact rising this far above the floor line ...
FLIGHT_MIN_S = 0.12            # ... for at least this long ⇒ a genuine airborne phase


def ground_motion_per_frame(
    motion: np.ndarray,
    model=None,
    fps: float = 30.0,
    smooth_win: int = GROUND_SMOOTH_WIN,
) -> tuple[np.ndarray, dict]:
    """Per-frame foot-contact grounding: remove the slow vertical DRIFT in the
    retarget's global translation so the support (lower) foot sits on z≈0 in
    EVERY frame — not just at the single global minimum the old ``ground_motion``
    plants. This is the fix for the §3.3 'floaty feet' source-motion defect
    (Thriller floated the support foot >0.10 m in ~78 % of frames).

    Method:
      1. c_i = per-frame lowest robot-geom height (the true contact height; the
         min is over both feet, so the PLANTED foot dominates — a lifted swing
         foot doesn't raise it).
      2. A slowly-varying GROUND LINE g_i is fit to c_i by a short Savitzky-Golay
         low-pass. It tracks the drift/bob of the estimated floor but NOT the
         few-mm frame noise, so subtracting it removes the float WITHOUT adding
         root-z jitter (measured jerk is not increased).
      3. FLIGHT GUARD: during a genuine airborne phase (contact rises FLIGHT_BAND_M
         above the floor line for ≥ FLIGHT_MIN_S) the feet SHOULD leave the ground,
         so the ground line is held flat (linearly interpolated) across the span
         instead of chasing the feet down — which would delete the jump. Drift-only
         motion (Thriller) never trips this: it is a no-op there.
      4. root z_i -= g_i, then a tiny residual GLOBAL lift so the lowest contact
         over the whole clip is ≥ 0 (guarantees NO new penetration).

    Grounding only translates the whole body vertically per frame, so every
    RELATIVE height (root-above-foot, pelvis-above-foot — what the tracking policy
    actually targets) is preserved EXACTLY; only the spurious vertical drift is
    removed, which makes the root-height target more correct, not less.

    Returns (grounded_copy, info)."""
    c = per_contact_height(motion, model)
    g = _sg_smooth_1d(c, smooth_win)

    # flight guard: hold the floor line flat across sustained airborne spans
    air = (c - g) > FLIGHT_BAND_M
    flight_frames = 0
    if air.any():
        n = len(c)
        min_fl = max(1, int(round(FLIGHT_MIN_S * fps)))
        i = 0
        while i < n:
            if air[i]:
                j = i
                while j < n and air[j]:
                    j += 1
                if (j - i) >= min_fl:
                    a, b = max(i - 1, 0), min(j, n - 1)
                    g[i:j] = np.linspace(g[a], g[b], j - i)
                    flight_frames += (j - i)
                i = j
            else:
                i += 1

    out = motion.copy()
    out[:, 2] -= g

    # no-penetration guarantee: lift so the lowest contact over the clip sits ≥ 0
    resid = float(per_contact_height(out, model).min())
    if resid < 0:
        out[:, 2] -= resid                    # add |resid| uniformly

    info = {
        "mode": "per_frame",
        "drift_removed_mm": round(float(g.max() - g.min()) * 1000, 1),
        "mean_shift_m": round(float(g.mean()), 4),
        "resid_lift_mm": round(-min(resid, 0.0) * 1000, 1),
        "flight_frames": int(flight_frames),
    }
    return out, info


def have_model() -> bool:
    return MODEL_XML.exists()
