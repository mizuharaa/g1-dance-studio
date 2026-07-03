#!/usr/bin/env python
"""Hand expressiveness for the G1 dance pipeline — SAFETY-FIRST prototype.

Framing (per owner concerns): the Inspire hands are expensive and must NEVER be
driven by an unvetted motion. So the load-bearing deliverable here is a
**hand-collision vet gate** that runs entirely in simulation: replay a hand-pose
track over the full dance and reject it if any finger self-collides, hits the
body, or exceeds a joint limit. Nothing in this module ever commands real hands.

Two hard findings this module encodes (both verified against the repo, see
docs/hands_feature.md):

1. HAND POSE IS NOT RECOVERABLE from our current front-end. GVHMR predicts a
   BODY-only SMPL (63-dim body_pose, no fingers); the GMR loader hardcodes
   left/right_hand_pose to zeros. `probe_gvhmr_hands()` proves this on any pred.
   So expressive hand motion must be AUTHORED (pose library + keyframes) until a
   whole-body-with-hands estimator replaces/augments GVHMR.

2. MODEL MISMATCH. The only G1+hands MuJoCo model we have (menagerie
   g1_with_hands.xml) is the Unitree **Dex3** 3-finger hand (7 DoF/hand). The
   real robot has **Inspire RH56DFTP** 5-finger hands (6 DoF/hand). The collision
   gate machinery below is correct and model-agnostic, but until an Inspire MJCF
   is sourced it runs on the Dex3 substrate and CANNOT certify the real Inspire
   hands. `HandModel` is parameterised so re-pointing at an Inspire MJCF is a
   one-line change.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _third_party() -> Path:
    """third_party/ holds gitignored clones absent from git worktrees; fall back
    to the primary checkout so this runs from a worktree too."""
    for base in (ROOT, Path.home() / "g1-dance"):
        tp = base / "third_party"
        if (tp / "mujoco_menagerie").exists():
            return tp
    return ROOT / "third_party"


DEX3_MJCF = _third_party() / "mujoco_menagerie" / "unitree_g1" / "g1_with_hands.xml"

# ---- Inspire RH56DFTP spec (the REAL robot's hands) --------------------------
# 6 actuated DoF/hand, commanded as angle_set[6] over DDS rt/inspire_hand/ctrl/{l,r};
# range 0..1000 (0 = fully curled/closed, 1000 = open) per the robot SDK examples.
INSPIRE_DOF = 6
INSPIRE_MIN, INSPIRE_MAX = 0, 1000
# angle_set index -> finger (from inspire_hand_sdk examples)
INSPIRE_FINGERS = ("pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot")

# A small library of named, hardware-safe Inspire poses (0=curled, 1000=open).
INSPIRE_POSES = {
    "open":  [1000, 1000, 1000, 1000, 1000, 1000],
    "fist":  [0, 0, 0, 0, 0, 400],
    "point": [0, 0, 0, 1000, 0, 400],   # index extended
    "claw":  [350, 350, 350, 350, 500, 600],   # Thriller: half-curled rigid claw
    "flat":  [1000, 1000, 1000, 1000, 1000, 1000],
}


def clamp_inspire(pose) -> list[int]:
    return [int(max(INSPIRE_MIN, min(INSPIRE_MAX, v))) for v in pose]


# ---- Finding #1: hand data is not recoverable from GVHMR --------------------
def probe_gvhmr_hands(pred_file: Path) -> dict:
    """Return whether a GVHMR prediction actually contains finger articulation.

    GVHMR is body-only; this reports the honest answer (almost always: absent).
    """
    import torch
    d = torch.load(str(pred_file), map_location="cpu", weights_only=False)

    def _walk(o, p=""):
        out = {}
        if isinstance(o, dict):
            for k, v in o.items():
                out.update(_walk(v, f"{p}/{k}"))
        else:
            out[p] = tuple(getattr(o, "shape", ()))
        return out

    keys = _walk(d)
    hand_keys = [k for k in keys if any(h in k.lower()
                 for h in ("hand", "finger", "mano", "left_hand", "right_hand"))]
    body_pose = next((keys[k] for k in keys if k.endswith("/body_pose")), None)
    return {
        "has_hand_data": bool(hand_keys),
        "hand_keys": hand_keys,
        "body_pose_shape": body_pose,     # (N, 63) = 21 body joints, no fingers
        "verdict": ("hand pose present" if hand_keys else
                    "NO hand data — GVHMR is body-only; fingers must be authored"),
    }


# ---- Time-synced authored hand track ----------------------------------------
@dataclass
class HandKeyframe:
    t_s: float
    left: list           # Inspire 6-DoF (or model DoF) target
    right: list


def build_hand_track(keyframes: list[HandKeyframe], n_frames: int, fps: float,
                     lead_in_s: float = 0.0) -> np.ndarray:
    """Per-frame [left(6) | right(6)] track, linearly interpolated between
    keyframes and held at the ends, aligned to the body motion timeline.

    lead_in_s shifts keyframe times to match prep_motion's standing pad/blend so
    the hands stay in their first pose during the intro and hit their marks in
    sync with the body dance.
    """
    assert keyframes, "need at least one keyframe"
    kf = sorted(keyframes, key=lambda k: k.t_s)
    times = np.array([k.t_s + lead_in_s for k in kf])
    left = np.array([k.left for k in kf], dtype=float)
    right = np.array([k.right for k in kf], dtype=float)
    dof = left.shape[1]
    track = np.zeros((n_frames, 2 * dof))
    ts = np.arange(n_frames) / fps
    for j in range(dof):
        track[:, j] = np.interp(ts, times, left[:, j])
        track[:, dof + j] = np.interp(ts, times, right[:, j])
    return track


# ---- Finding #2 + the SAFETY GATE: hand collision vet ------------------------
@dataclass
class HandModel:
    """A G1+hands MuJoCo model, parameterised so it can point at Dex3 (now) or an
    Inspire MJCF (once sourced) without changing the gate logic."""
    mjcf: Path = DEX3_MJCF
    is_real_inspire: bool = False   # True only when pointed at a real Inspire MJCF

    def load(self):
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(self.mjcf))
        return m


def _hand_geoms(m):
    """Geom ids belonging to any body whose name contains 'hand'."""
    import mujoco
    ids, finger_of = [], {}
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if "hand" in bn:
            ids.append(g)
            # finger key: e.g. left_hand_thumb_0_link -> left/thumb
            parts = bn.split("_")
            side = "left" if bn.startswith("left") else "right"
            finger = parts[2] if len(parts) > 2 else "palm"
            finger_of[g] = (side, finger)
    return ids, finger_of


def check_hand_collisions(track: np.ndarray, model: HandModel = HandModel(),
                          margin_m: float = 0.003) -> dict:
    """Replay a hand-pose track and flag self-collision / hand-body collision /
    joint-limit violation on any frame. Pure geometry (no dynamics) via
    mj_geomDistance, so it is independent of the model's contype/conaffinity.

    track columns are the model's hand joints (left then right); if the track's
    width differs from the model's hand-DoF (e.g. a 12-wide Inspire track on the
    7-DoF/hand Dex3 substrate), it is min-length matched and a note is returned.
    """
    import mujoco
    m = model.load()
    d = mujoco.MjData(m)

    # hand joint ids/qpos addresses, in model order
    hjoints = []
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if "hand" in n:
            hjoints.append((j, m.jnt_qposadr[j], m.jnt_range[j]))
    hg, finger_of = _hand_geoms(m)
    # Body geoms to test hands AGAINST — exclude the same-arm attachment chain
    # (wrist/elbow/forearm/rubber links the hand is naturally adjacent to), else
    # every neutral pose reads as a "collision". Torso/legs/other-arm stay checked.
    _ARM_END = ("wrist", "elbow", "forearm", "rubber")
    hgset = set(hg)
    body_geoms = []
    for g in range(m.ngeom):
        if g in hgset or not (m.geom_contype[g] | m.geom_conaffinity[g]):
            continue
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        if any(k in bn for k in _ARM_END):
            continue
        body_geoms.append(g)

    n_model_dof = len(hjoints)
    half = track.shape[1] // 2
    used = min(half, n_model_dof // 2) if n_model_dof else 0

    frames_checked = 0
    self_hits, body_hits, limit_hits = [], [], []
    for f in range(track.shape[0]):
        # write hand joints (left block then right block, min-matched)
        left_block = track[f, :used]
        right_block = track[f, half:half + used]
        vals = list(left_block) + list(right_block)
        for (jid, adr, rng), v in zip(hjoints, vals):
            lo, hi = rng
            if lo < hi and (v < lo - 1e-6 or v > hi + 1e-6):
                limit_hits.append((f, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)))
            d.qpos[adr] = float(np.clip(v, rng[0], rng[1]) if rng[0] < rng[1] else v)
        mujoco.mj_forward(m, d)
        frames_checked += 1
        # self-collision: hand geoms on DIFFERENT fingers closer than margin
        for a_i in range(len(hg)):
            ga = hg[a_i]
            for b_i in range(a_i + 1, len(hg)):
                gb = hg[b_i]
                if finger_of.get(ga) == finger_of.get(gb):
                    continue  # same finger link chain — adjacency, skip
                dist = mujoco.mj_geomDistance(m, d, ga, gb, margin_m + 0.02, None)
                if dist < margin_m:
                    self_hits.append((f, dist))
                    break
        # hand vs body
        for ga in hg:
            for gb in body_geoms:
                dist = mujoco.mj_geomDistance(m, d, ga, gb, margin_m + 0.02, None)
                if dist < margin_m:
                    body_hits.append((f, dist))
                    break

    passed = not (self_hits or body_hits or limit_hits)
    return {
        "passed": passed,
        "frames": frames_checked,
        "self_collisions": len(self_hits),
        "body_collisions": len(body_hits),
        "limit_violations": len(limit_hits),
        "first_self_frame": self_hits[0][0] if self_hits else None,
        "first_body_frame": body_hits[0][0] if body_hits else None,
        "model": str(model.mjcf.name),
        "certifies_real_inspire": model.is_real_inspire,
        "note": ("" if model.is_real_inspire else
                 "SUBSTRATE = Unitree Dex3 (3-finger), NOT the robot's Inspire "
                 "RH56DFTP (5-finger). Gate machinery is valid; re-point .mjcf at "
                 "an Inspire model before this can protect the real hands."),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, help="GVHMR pred .pt to probe for hands")
    args = ap.parse_args()
    if args.probe:
        print(probe_gvhmr_hands(args.probe))
