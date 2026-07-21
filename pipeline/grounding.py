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

import math
import warnings
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


def _foot_geom_ids(model) -> tuple[list[int], list[int]]:
    """Resolve the FOOT collision geoms, (left_ids, right_ids).

    Preference order (audit F8 — grounding must measure the SOLE, not whatever
    geom center happens to be lowest):
      1. explicitly named sole geoms ``{side}_foot[1-7]_collision`` if the model
         names them (mjlab-style XMLs);
      2. fallback: the collision-active geoms on the ``{side}_ankle_roll_link``
         body (both the menagerie scene and the unitree g1_29dof XML model the
         sole as 4 unnamed r=5 mm spheres there; the visual mesh is contype 0
         and excluded);
      3. last resort: ALL geoms on that body (a model with contact fully
         stripped from the geoms).
    """
    import mujoco
    out = []
    for side in ("left", "right"):
        ids = []
        for k in range(1, 8):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                    f"{side}_foot{k}_collision")
            if gid >= 0:
                ids.append(int(gid))
        if not ids:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    f"{side}_ankle_roll_link")
            if bid >= 0:
                on_body = np.flatnonzero(model.geom_bodyid == bid)
                ids = [int(g) for g in on_body
                       if model.geom_contype[g] or model.geom_conaffinity[g]]
                if not ids:
                    ids = [int(g) for g in on_body]
        if not ids:
            raise ValueError(
                f"cannot resolve {side} foot geoms: no '{side}_foot*_collision' "
                f"geom and no '{side}_ankle_roll_link' body in the model")
        out.append(ids)
    return out[0], out[1]


def _geom_lowest_z(model, data, gid: int, conservative_fallback: bool) -> float:
    """World z of the geom's LOWEST SURFACE point (audit F8: centers are not
    surfaces — a foot sphere center sits one radius above the sole).

    Exact for sphere/capsule/cylinder/box/ellipsoid using the geom's world
    orientation. For other types (mesh): with ``conservative_fallback`` the
    height is ``center_z - size.max()`` (documented conservative bound — may
    UNDERSTATE the height, never overstate it); otherwise the center z is used
    (meshes on the G1 are never the low envelope — the sole is modelled by
    sphere primitives, which are handled exactly).
    """
    import mujoco
    z = float(data.geom_xpos[gid, 2])
    t = int(model.geom_type[gid])
    s = model.geom_size[gid]
    R = data.geom_xmat[gid]                # row-major 3x3; R[6:9] = world-z row
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        drop = float(s[0])
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        drop = float(s[0] + s[1] * abs(R[8]))
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        drop = float(s[1] * abs(R[8]) + s[0] * math.sqrt(max(0.0, 1.0 - R[8] * R[8])))
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        drop = float(abs(R[6]) * s[0] + abs(R[7]) * s[1] + abs(R[8]) * s[2])
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        drop = float(math.sqrt((R[6] * s[0]) ** 2 + (R[7] * s[1]) ** 2
                               + (R[8] * s[2]) ** 2))
    else:
        drop = float(s.max()) if conservative_fallback else 0.0
    return z - drop


def _fk_heights(motion: np.ndarray, model=None) -> tuple[np.ndarray, np.ndarray]:
    """One FK sweep → ``(lowest (N,), soles (N,2))``.

    ``lowest``: per-frame lowest SURFACE z over all robot geoms (world/floor
    excluded; primitive geoms surface-exact, meshes at center — see
    ``_geom_lowest_z``). ``soles``: per-foot (left, right) sole heights from the
    resolved foot collision geoms (conservative fallback for exotic types).
    """
    import mujoco
    model = model or _model()
    data = mujoco.MjData(model)
    robot_geoms = [int(g) for g in np.flatnonzero(model.geom_bodyid != 0)]
    left_ids, right_ids = _foot_geom_ids(model)
    lowest = np.empty(len(motion))
    soles = np.empty((len(motion), 2))
    for i, row in enumerate(motion):
        data.qpos[:3] = row[:3]
        data.qpos[3:7] = row[[6, 3, 4, 5]]  # xyzw -> wxyz
        data.qpos[7:] = row[7:]
        mujoco.mj_kinematics(model, data)
        lowest[i] = min(_geom_lowest_z(model, data, g, conservative_fallback=False)
                        for g in robot_geoms)
        soles[i, 0] = min(_geom_lowest_z(model, data, g, conservative_fallback=True)
                          for g in left_ids)
        soles[i, 1] = min(_geom_lowest_z(model, data, g, conservative_fallback=True)
                          for g in right_ids)
    return lowest, soles


def per_contact_height(motion: np.ndarray, model=None) -> np.ndarray:
    """Per-frame lowest SURFACE z of any ROBOT geom (world/floor geoms
    excluded), as an (N,) array — the true FK floor-contact height at each
    frame. Audit F8 fix: primitive geoms (the G1's foot-sole spheres included)
    are measured at their lowest surface point, not their center — the old
    center-based figure floated one sole-sphere radius above the real contact.
    Mesh geoms are still measured at center (on the G1 they are never the low
    envelope; the sole is primitive spheres)."""
    return _fk_heights(motion, model)[0]


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


# Per-frame grounding parameters (see ground_motion_per_frame).
GROUND_SMOOTH_WIN = 9          # frames — legacy SG window, kept for signature compat
FLIGHT_BAND_M = 0.08           # BOTH soles this far above the floor line ...
FLIGHT_MIN_S = 0.12            # ... for at least this long ⇒ a genuine airborne phase
                               # (4 frames at 30 fps — the audit F8 minimum)
SUPPORT_BAND_M = 0.03          # sole within this of the floor estimate ⇒ support candidate
SUPPORT_VMAX_MPS = 0.20        # ... AND its vertical speed below this
# EMA time constant of the floor. Deliberately FAST: jumps are protected by the
# support GATE (an airborne sole is outside the gated bands, so the floor holds
# no matter how fast the EMA is); the tau only sets how tightly the floor rides
# gated drift/bob, and the real retarget drift (thriller: 145 mm of float, with
# bobs at up to ~0.26 m/s) needs tight tracking to keep the support foot within
# the audit's ~2 cm grounded-comparator budget.
FLOOR_TAU_S = 0.08
SEED_S = 0.5                   # floor seeded from this initial window (grounded start)
# Drift re-lock (tier 2): a sole BELOW the flight band (floor + FLIGHT_BAND_M)
# moving slower than this also updates the floor. Retarget floor-bob reaches
# ~0.26 m/s (above the 0.2 m/s tier-1 gate) but a genuine ballistic take-off /
# landing crosses the band at ~2 m/s, so this still cannot chase a real jump —
# and a sustained plateau above FLIGHT_BAND_M is never eligible at any speed.
# Consequence (documented contract): only airborne phases clearing
# FLIGHT_BAND_M count as flight; a sub-8 cm slow float is treated as drift.
RELOCK_VMAX_MPS = 0.35


def ground_motion_per_frame(
    motion: np.ndarray,
    model=None,
    fps: float = 30.0,
    smooth_win: int = GROUND_SMOOTH_WIN,
) -> tuple[np.ndarray, dict]:
    """Per-frame foot-contact grounding: remove the slow vertical DRIFT in the
    retarget's global translation so the support foot sits on z≈0 in EVERY
    frame — while PRESERVING genuine airborne phases (audit F8: the old version
    low-passed the same contact signal it classified, so a sustained 2 s jump
    plateau read as a new floor and was subtracted to ~0).

    Method (support-gated floor, audit F8 fix):
      1. Per-FOOT sole-surface heights from the resolved foot collision geoms
         (``_foot_geom_ids`` / ``_geom_lowest_z`` — surfaces, not geom centers).
      2. Floor estimate seeded from the first ``SEED_S`` seconds (median of the
         lower-sole height; the motion is ASSUMED to start grounded — if the
         seed window is not quiet a warning is emitted and
         ``info['grounded_start']`` is False).
      3. Causal support gate: a foot is a SUPPORT candidate at frame i when its
         sole is within ``SUPPORT_BAND_M`` of the current floor estimate (or
         below it — a sole cannot be under the true floor, so below-floor
         always counts, letting the estimate correct downward) AND its vertical
         speed is < ``SUPPORT_VMAX_MPS``; additionally (tier-2 drift re-lock) a
         sole BELOW the flight band (floor + ``FLIGHT_BAND_M``) moving slower
         than ``RELOCK_VMAX_MPS`` qualifies, so sub-flight-band retarget bob is
         tracked instead of accumulating as float. Frames with ≥1 support foot
         update the floor by an EMA (``FLOOR_TAU_S``); frames with NO support
         HOLD the last floor value unchanged — a jump plateau sits above the
         flight band at ballistic speeds, so it can never become the floor, no
         matter how long it lasts.
      4. root z_i -= floor_i, with the recorded floor clamped per frame to the
         lowest robot surface (no new penetration — and no whole-clip lift, so
         one deep stomp cannot float the rest of the motion).
      5. Flight report: spans where BOTH soles are > floor + ``FLIGHT_BAND_M``
         for ≥ ``FLIGHT_MIN_S`` are counted and returned in
         ``info['flight_windows_s']`` — their true height survives because the
         held floor, not the airborne soles, is what gets subtracted.

    Grounding only translates the whole body vertically per frame, so every
    RELATIVE height (root-above-foot — what the tracking policy targets) is
    preserved EXACTLY.

    ``smooth_win`` is retained for signature compatibility but unused: the SG
    low-pass it configured is replaced by the support-gated EMA above.

    Returns (grounded_copy, info)."""
    del smooth_win  # legacy SG parameter — superseded by the support-gated EMA
    n = len(motion)
    lowest, soles = _fk_heights(motion, model)
    c = soles.min(axis=1)                       # lower-sole height per frame
    dt = 1.0 / fps
    v = (np.gradient(soles, dt, axis=0) if n >= 2
         else np.zeros_like(soles))             # per-foot vertical speed

    # seed the floor from the (assumed grounded) start
    n0 = min(n, max(1, int(round(SEED_S * fps))))
    floor = float(np.median(c[:n0]))
    seed_dev = float(np.max(np.abs(c[:n0] - floor))) if n0 else 0.0
    grounded_start = seed_dev <= 2 * SUPPORT_BAND_M
    if not grounded_start:
        warnings.warn(
            f"ground_motion_per_frame: motion does not start grounded (sole "
            f"height deviates {seed_dev:.3f} m from the seed floor within the "
            f"first {SEED_S} s) — floor seed may be unreliable", stacklevel=2)

    # causal support-gated floor: EMA on supported frames, HOLD otherwise.
    # tier 1 = the spec's support candidate (within SUPPORT_BAND_M, slow);
    # tier 2 = drift re-lock (below the flight band, slow-ish — see
    # RELOCK_VMAX_MPS). A sole above floor + FLIGHT_BAND_M NEVER updates the
    # floor, so a sustained jump plateau cannot become the floor.
    alpha = 1.0 - math.exp(-dt / FLOOR_TAU_S)
    g = np.empty(n)
    supported = np.zeros(n, dtype=bool)
    for i in range(n):
        sup = [f for f in (0, 1)
               if (soles[i, f] - floor <= SUPPORT_BAND_M
                   and abs(v[i, f]) < SUPPORT_VMAX_MPS)
               or (soles[i, f] - floor < FLIGHT_BAND_M
                   and abs(v[i, f]) < RELOCK_VMAX_MPS)]
        if sup:
            floor += alpha * (min(soles[i, f] for f in sup) - floor)
            supported[i] = True
        g[i] = floor

    # flight report: both soles well above the (held) floor for long enough
    air = (c - g) > FLIGHT_BAND_M
    min_fl = max(1, int(round(FLIGHT_MIN_S * fps)))
    windows: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if air[i]:
            j = i
            while j < n and air[j]:
                j += 1
            if (j - i) >= min_fl:
                windows.append((i, j))
            i = j
        else:
            i += 1
    flight_frames = sum(j - i for i, j in windows)

    # no-penetration guarantee, PER FRAME: the floor cannot sit above the lowest
    # robot surface (physically impossible), so clamp the recorded floor down on
    # the (transient) frames where a fast stomp dives below the estimate. This
    # replaces the old whole-clip residual lift, which let one deep stomp float
    # the entire motion (thriller_deploy: a single -66 mm frame lifted every
    # other frame 66 mm off the floor). The clamp is on the RECORDED g only —
    # the EMA state is untouched, so a one-frame dive can't unlock the floor.
    g = np.minimum(g, lowest)

    out = motion.copy()
    out[:, 2] -= g

    # belt-and-braces: pure z-translation ⇒ post-shift lowest surface is
    # (lowest - g) ≥ 0 by the clamp above; lift for float-rounding only
    resid = float((lowest - g).min())
    if resid < 0:
        out[:, 2] -= resid                    # add |resid| uniformly

    info = {
        "mode": "per_frame",
        "drift_removed_mm": round(float(g.max() - g.min()) * 1000, 1),
        "mean_shift_m": round(float(g.mean()), 4),
        "resid_lift_mm": round(-min(resid, 0.0) * 1000, 1),
        "flight_frames": int(flight_frames),
        "flight_windows_s": [[round(i / fps, 3), round(j / fps, 3)]
                             for i, j in windows],
        "floor_drift_m": round(float(g[-1] - g[0]), 4),
        "support_pct": round(100.0 * float(supported.mean()), 1) if n else 0.0,
        "grounded_start": bool(grounded_start),
    }
    return out, info


def have_model() -> bool:
    return MODEL_XML.exists()
