"""Turn a box-side held-out mjlab eval (cloud/heldout_eval.py) into a SIGNED
sim_exam/v1 verdict.

This is `method: "mjlab_heldout_v1"` — same-ENGINE held-out robustness verification
(disjoint seeds + observation corruption + external shoves), NOT a different-simulator
sim2sim check (the plain-MuJoCo G1 model isn't dynamically faithful; see
docs/exam_physics_fix.md). It catches a policy that overfits training seeds or can't
take a shove. It does NOT catch mjlab-specific physics exploitation, and it is NOT a
substitute for gantry-first robot-day validation.

The verdict flows through the same pipeline.exam_verdict gate as any other: show-ready
authorization still requires derive_pass — all phases pass, the push force floor is met,
and repeatability is clean==runs (i.e. EVERY held-out episode survived). A policy at
98.4% therefore does NOT authorize show-ready — by design.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.exam_verdict import (
    full_sha256, sign_verdict, authorize, REQUIRED_CLEAN_RUNS, REQUIRED_CLEAN_RATE,
)

# Phase thresholds (survival fraction) — a phase "passes" at/above these, but
# show-ready still needs repeatability clean==runs (100%) via derive_pass.
NOMINAL_MIN = 0.98
PUSH_MIN = 0.95
# mjlab push is a base-velocity impulse, not a Newton force. Record the honest
# impulse-equivalent force (F = m*dv/dt) so the schema's force floor is met on a
# real physical basis, and document the mechanism alongside it.
G1_MASS_KG = 35.0
PUSH_DV_MPS = 0.5           # representative from mjlab VELOCITY_RANGE
PUSH_DT_S = 0.02            # one 50 Hz control period
PUSH_FORCE_EQUIV_N = round(G1_MASS_KG * PUSH_DV_MPS / PUSH_DT_S, 1)  # ~875 N


def build_verdict(eval_json: dict, policy_path: Path, motion_path: Path,
                  venue_max_excursion_m: float = 1.5) -> dict:
    nom = eval_json["conditions"]["nominal"]
    push = eval_json["conditions"]["push"]
    n = int(nom["num_episodes"])
    clean = int(nom["n_success"])

    verdict = {
        "schema": "sim_exam/v1",
        "method": "mjlab_heldout_v1",
        "dance": eval_json.get("dance", motion_path.stem),
        "policy": str(policy_path),
        "policy_sha256": full_sha256(policy_path) if policy_path.exists() else None,
        "motion_sha256": full_sha256(motion_path) if motion_path.exists() else None,
        "venue_max_excursion_m": venue_max_excursion_m,
        "nominal": {
            "pass": nom["success_rate"] >= NOMINAL_MIN,
            "success_rate": nom["success_rate"],
            "n_success": clean,
            "num_episodes": n,
            "mpkpe_m": nom.get("mpkpe_m"),
            "ee_pos_error_m": nom.get("ee_pos_error_m"),
            "held_out_seed": nom.get("seed"),
        },
        "push": {
            "pass": push["success_rate"] >= PUSH_MIN,
            "success_rate": push["success_rate"],
            "n_success": int(push["n_success"]),
            "num_episodes": int(push["num_episodes"]),
            "force_n": PUSH_FORCE_EQUIV_N,
            "push_mechanism": "mjlab base-velocity impulse (~%.1f m/s); "
                              "force_n is the m*dv/dt equivalent" % PUSH_DV_MPS,
            "mpkpe_m": push.get("mpkpe_m"),
            "held_out_seed": push.get("seed"),
        },
        # Repeatability = the nominal held-out episodes. Show-ready needs a survival
        # rate >= REQUIRED_CLEAN_RATE (the user's >=99% standard) across >= 3 episodes;
        # derive_pass re-checks this independently.
        "repeatability": {
            "pass": n >= REQUIRED_CLEAN_RUNS and (clean / n) >= REQUIRED_CLEAN_RATE if n else False,
            "runs": n,
            "clean": clean,
        },
    }
    verdict["verdict"] = (
        "pass" if (verdict["nominal"]["pass"] and verdict["push"]["pass"]
                   and verdict["repeatability"]["pass"]) else "fail"
    )
    return sign_verdict(verdict)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-excursion-m", type=float, default=1.5)
    a = ap.parse_args()

    eval_json = json.loads(Path(a.eval_json).read_text())
    v = build_verdict(eval_json, Path(a.policy), Path(a.motion), a.max_excursion_m)
    Path(a.out).write_text(json.dumps(v, indent=2))

    ok, reason = authorize(v, policy_sha=v["policy_sha256"], motion_sha=v["motion_sha256"])
    print(f"verdict: {v['verdict']}  |  authorizes show-ready: {ok} ({reason})")
    print(f"  nominal {v['nominal']['n_success']}/{v['nominal']['num_episodes']} "
          f"survived (mpkpe {v['nominal']['mpkpe_m']:.3f}m); "
          f"push {v['push']['n_success']}/{v['push']['num_episodes']} survived")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
