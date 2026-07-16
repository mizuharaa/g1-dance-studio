"""Triage a bad-looking G1 dance clip into the ONE layer that actually owns the fix.

The handoff (§3.3) is emphatic: "floaty / impossible / offset" motion has THREE
distinct sources and the fix differs for each. People waste days fixing the wrong
layer. This script forces the distinction with measurements before anyone touches
code:

  A. SOURCE-MOTION ERROR  (the *reference* CSV is kinematically/dynamically bad:
     feet float, joints jitter, poses out of range, ankle torque impossible)
       -> fix HERE: pipeline/grounding.py, prep_motion.py, motion_repair.py, the
          DOF-aware retarget. This is Agent B's layer.
  B. SANDBOX-MODEL ARTIFACT  (the reference is FINE, but the on-laptop preview is
     rendered in the *menagerie* G1 model that != the mjlab training model, so it
     looks offset / washed-out / collapses)
       -> NOT a motion defect. Agent E (preview fidelity). Do not touch the motion.
  C. TRAINED-POLICY BEHAVIOR  (reference fine, but the trained policy drifts /
     under-reaches / topples)
       -> RL recipe / sim fidelity. Agent F. Needs a policy rollout to see.

Decision logic (evidence-driven):
  * Always measure the REFERENCE (dynamic + kinematic + quality). If it trips the
    source-error thresholds -> verdict A (with the specific failing checks).
  * If a POLICY ROLLOUT csv is supplied, compare it to the reference; large
    tracking divergence on a CLEAN reference -> verdict C.
  * Reference clean, no rollout, but the complaint is about the PREVIEW look ->
    verdict B (menagerie artifact) is the most likely, flagged as such.

Usage:
  python tools/motion_triage.py --reference motion.csv [--rollout policy_deploy.csv]
                                [--preview-model menagerie|mjlab] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import motion_dynamics as MD
from pipeline.motion_io import load_motion_csv


def _jerk_spike_pct(m: np.ndarray, fps: float) -> float:
    """% frames with a robust accel spike (numpy-only; avoids the scipy dependency
    in tools.motion_quality so triage runs in a bare env). Robust z-score of the
    per-joint acceleration, flooring at 150 rad/s^2 to ignore quiet noise."""
    j = m[:, 7:]
    acc = np.abs(np.diff(j, axis=0, n=2)) * fps * fps           # (N-2,29)
    med = np.median(acc, axis=0)
    mad = np.median(np.abs(acc - med), axis=0) + 1e-6
    z = (acc - med) / (1.4826 * mad)
    hit = ((z > 10.0) & (acc > 150.0)).any(axis=1)
    return 100.0 * float(hit.mean())

# source-motion-error thresholds
FLOATY_FEET_PCT = 30.0        # % frames the lower foot floats > 0.10 m
ANKLE_OVER_PCT = 20.0         # % frames ankle demand > headroom
JERK_SPIKE_PCT = 5.0          # % frames with an accel spike
POS_VIOL_RAD = 0.05           # worst joint-limit violation
# policy-tracking divergence threshold (rad, mean joint error)
TRACK_DIVERGE_RAD = 0.30


def _track_error(ref: np.ndarray, roll: np.ndarray) -> float:
    """Mean joint-angle error after resampling both to a common length (time-
    normalized; ignores tempo/offset, catches shape divergence)."""
    S = 300
    tr = np.linspace(0, 1, len(ref))
    tp = np.linspace(0, 1, len(roll))
    td = np.linspace(0, 1, S)
    a = np.stack([np.interp(td, tr, ref[:, 7 + j]) for j in range(29)], 1)
    b = np.stack([np.interp(td, tp, roll[:, 7 + j]) for j in range(29)], 1)
    return float(np.abs(a - b).mean())


def triage(reference: str, rollout: str | None = None,
           preview_model: str = "menagerie") -> dict:
    ref = load_motion_csv(reference)
    dyn = MD.analyze(reference)
    spike_pct = _jerk_spike_pct(ref, MD.CSV_FPS)

    checks = {
        "floaty_feet_pct": dyn["balance"]["floaty_feet_pct"],
        "ankle_over_headroom_pct": dyn["dynamic"]["ankle_frames_over_headroom_pct"],
        "jerk_spike_pct": round(spike_pct, 2),
        "pos_worst_violation_rad": dyn["kinematic"]["pos_worst_violation_rad"],
        "ankle_tau_max_nm": dyn["dynamic"]["ankle_tau_max_nm"],
    }
    source_fail = []
    if checks["floaty_feet_pct"] > FLOATY_FEET_PCT:
        source_fail.append(f"feet float {checks['floaty_feet_pct']}% > {FLOATY_FEET_PCT}%")
    if checks["ankle_over_headroom_pct"] > ANKLE_OVER_PCT:
        source_fail.append(
            f"ankle demand over headroom {checks['ankle_over_headroom_pct']}% "
            f"> {ANKLE_OVER_PCT}% (peak {checks['ankle_tau_max_nm']} Nm)")
    if checks["jerk_spike_pct"] > JERK_SPIKE_PCT:
        source_fail.append(f"jerk spikes {checks['jerk_spike_pct']}% > {JERK_SPIKE_PCT}%")
    if checks["pos_worst_violation_rad"] > POS_VIOL_RAD:
        source_fail.append(f"joint out of range {checks['pos_worst_violation_rad']} rad")

    track = None
    if rollout:
        roll = load_motion_csv(rollout)
        track = round(_track_error(ref, roll), 3)

    # --- verdict ---
    if source_fail:
        verdict = "A_SOURCE_MOTION_ERROR"
        owner = "Agent B (this layer): grounding / prep / motion_repair / DOF-aware retarget"
        why = ("The REFERENCE itself is infeasible before any physics/policy. Repair "
               "it (global slowdown for torque; grounding + retarget for floaty feet) "
               "before spending GPU. Failing checks: " + "; ".join(source_fail))
    elif track is not None and track > TRACK_DIVERGE_RAD:
        verdict = "C_TRAINED_POLICY_BEHAVIOR"
        owner = "Agent F (RL recipe / sim fidelity)"
        why = (f"Reference is feasible but the policy rollout diverges "
               f"({track} rad mean joint error > {TRACK_DIVERGE_RAD}). The bad look "
               f"is the trained policy under-reaching/drifting, not the motion.")
    elif track is not None:
        verdict = "OK_POLICY_TRACKS"
        owner = "none — reference feasible and policy tracks it"
        why = (f"Reference feasible and rollout tracks it ({track} rad). If it still "
               f"looks bad, suspect the preview MODEL (verdict B) or rendering.")
    else:
        verdict = "B_SANDBOX_MODEL_ARTIFACT"
        owner = "Agent E (preview fidelity — menagerie != mjlab)"
        why = ("The reference passes feasibility and no policy rollout was supplied. "
               "A floaty/offset LOOK with a clean reference is most likely the "
               f"preview model artifact (preview_model='{preview_model}'): the laptop "
               "sandbox renders the menagerie G1, not the mjlab training model. "
               "Provide --rollout to rule policy behavior in/out.")

    return {
        "reference": reference,
        "rollout": rollout,
        "verdict": verdict,
        "owns_the_fix": owner,
        "why": why,
        "evidence": checks,
        "policy_track_error_rad": track,
        "thresholds": {
            "floaty_feet_pct": FLOATY_FEET_PCT, "ankle_over_headroom_pct": ANKLE_OVER_PCT,
            "jerk_spike_pct": JERK_SPIKE_PCT, "pos_viol_rad": POS_VIOL_RAD,
            "track_diverge_rad": TRACK_DIVERGE_RAD},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--rollout", default=None,
                    help="a trained-policy deploy/rollout CSV (36-col) to test verdict C")
    ap.add_argument("--preview-model", default="menagerie",
                    choices=["menagerie", "mjlab"])
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    r = triage(args.reference, args.rollout, args.preview_model)
    print(f"VERDICT: {r['verdict']}")
    print(f"OWNS THE FIX: {r['owns_the_fix']}")
    print(f"WHY: {r['why']}")
    print(f"EVIDENCE: {json.dumps(r['evidence'])}")
    if r["policy_track_error_rad"] is not None:
        print(f"POLICY TRACK ERROR: {r['policy_track_error_rad']} rad")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(r, indent=2))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
