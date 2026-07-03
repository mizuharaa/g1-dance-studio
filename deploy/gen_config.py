"""Generate a robot-day deploy bundle from a trained policy + motion pair.

HARD GATE: refuses unless a sim_exam/v1 verdict JSON exists for this exact
policy+motion (matched by sha256) with verdict == "pass". The exam is the
Stage-4 contract — no bundle, no deploy, no exceptions.

Output bundle (deploy/bundles/<dance>/):
    policy.onnx           the trained policy (copied, sha256-pinned)
    motion.csv            the vetted 30 fps reference motion
    exam_verdict.json     the passing exam (provenance)
    bundle.json           deploy_bundle/v1 manifest (docs/show_mode_contracts.md)
    controller.env        target/controller parameters for the push scripts

The bundle is inert data — pushing/running it on PC2 is done by the gated
shell scripts (01_pc2_install.sh, 02_push_bundle.sh, 10_gantry_test.sh).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pipeline.exam_verdict import authorize, full_sha256

KIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = KIT_DIR.parent

PC2_HOST = "192.168.123.164"
PC2_USER = "unitree"
CONTROLLER_IMAGE = "qiayuanl/unitree:jazzy"
DANCE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # finding #31: no shell metachars to PC2


def sha256(p: Path) -> str:
    return full_sha256(p)  # full 64-hex identity (finding #32)


def find_passing_exam(policy_sha: str, motion_sha: str, exam_dir: Path) -> Path | None:
    """Return an exam file that AUTHORIZES this exact policy+motion, or None.

    Findings #0/#7/#19/#21: authorization is derived from phase contents AND an HMAC
    signature — never the self-declared ``verdict`` string, never an empty-dict phase.
    """
    for f in sorted(exam_dir.glob("exam_*.json")):
        try:
            v = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        ok, _reason = authorize(v, policy_sha=policy_sha, motion_sha=motion_sha)
        if ok:
            return f
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dance", required=True)
    ap.add_argument("--policy", required=True, type=Path)
    ap.add_argument("--motion", required=True, type=Path)
    ap.add_argument(
        "--exam-dir", type=Path, default=PROJECT_ROOT / "data" / "exports",
        help="where sim_exam verdicts live",
    )
    ap.add_argument(
        "--rehearsal", action="store_true",
        help="REHEARSAL ONLY: assemble the bundle WITHOUT the exam gate to validate "
             "packaging mechanics. Stamps the bundle non-deployable — 02_push_bundle.sh "
             "refuses to push it. Never produces an authorized bundle.",
    )
    args = ap.parse_args()

    if not DANCE_RE.match(args.dance):
        raise SystemExit(f"ABORT: --dance {args.dance!r} must match {DANCE_RE.pattern} (finding #31)")
    for f in (args.policy, args.motion):
        if not f.exists():
            raise SystemExit(f"ABORT: {f} does not exist")

    p_sha, m_sha = sha256(args.policy), sha256(args.motion)
    out = KIT_DIR / "bundles" / args.dance
    out.mkdir(parents=True, exist_ok=True)

    if args.rehearsal:
        # No exam gate — bundle is marked REHEARSAL_ONLY and can never be pushed.
        n_rows = sum(1 for _ in args.motion.open()) - 1  # minus header
        verdict = {"schema": "sim_exam/v1", "verdict": "rehearsal", "REHEARSAL_ONLY": True,
                   "nominal": {"duration_s": round(max(n_rows, 0) / 30.0, 2)}}
        (out / "exam_verdict.json").write_text(json.dumps(verdict, indent=1))
        (out / "REHEARSAL_ONLY").write_text(
            "Built with --rehearsal (no exam authorization). MUST NOT be pushed or run "
            "on a robot. Rebuild without --rehearsal once the dance is show-ready.\n")
        print("REHEARSAL bundle (NOT deployable): validating packaging only")
    else:
        exam = find_passing_exam(p_sha, m_sha, args.exam_dir)
        if exam is None:
            raise SystemExit(
                "ABORT: no PASSING sim_exam/v1 verdict (with push + repeatability phases) "
                f"matches policy {p_sha} + motion {m_sha} in {args.exam_dir}.\n"
                "Run: python -m pipeline.sim_exam --policy <onnx> --motion <csv> first."
            )
        verdict = json.loads(exam.read_text())
        shutil.copy2(exam, out / "exam_verdict.json")

    shutil.copy2(args.policy, out / "policy.onnx")
    shutil.copy2(args.motion, out / "motion.csv")

    # controller.env: launch parameters as REVIEWED DATA (findings #8/#20). start_mode
    # is fixed to damping here and asserted by the start script — not hand-editable.
    (out / "controller.env").write_text(
        f"DANCE={args.dance}\nPOLICY=policy.onnx\nMOTION=motion.csv\n"
        f"CONTROLLER_IMAGE={CONTROLLER_IMAGE}\nCONTROL_HZ=50\nSTART_MODE=damping\n"
    )
    # In-container entrypoint used by 10_gantry_test.sh. Asserts start_mode=damping from
    # controller.env; the launch line is filled in and LAUNCH_LINE_VERIFIED touched on
    # robot day (runbook step 3). Until then it refuses — itself an interlock.
    start = out / "start_controller_damping_hold.sh"
    start.write_text(
        "#!/usr/bin/env bash\n"
        "# Runs INSIDE qiayuanl/unitree:jazzy on PC2. Contract: load policy, hold\n"
        "# damping; motion playback is armed only by the operator's remote sequence.\n"
        "set -euo pipefail\n"
        "source /bundle/controller.env\n"
        "if [ \"${START_MODE:-}\" != \"damping\" ]; then\n"
        "  echo 'REFUSING: controller.env START_MODE is not damping.'; exit 79\n"
        "fi\n"
        "if [ ! -f /bundle/LAUNCH_LINE_VERIFIED ]; then\n"
        "  echo 'REFUSING: controller launch line not verified on robot day yet.'\n"
        "  echo 'See docs/ROBOT_DAY_RUNBOOK.md step 3 — then touch LAUNCH_LINE_VERIFIED'\n"
        "  echo 'in the bundle and re-push.'\n"
        "  exit 78\n"
        "fi\n"
        "# ROBOT-DAY: replace with the verified launch line, e.g.:\n"
        "# ros2 launch motion_tracking_controller tracking.launch.py \\\n"
        "#   policy:=/bundle/policy.onnx start_mode:=damping\n"
        "exit 78\n"
    )
    start.chmod(0o755)

    # Hash-pin EVERY bundle file (findings #8/#19): policy, motion, exam, controller.env,
    # start script — so 02_push_bundle.sh can detect any post-generation tampering.
    files = {
        name: full_sha256(out / name)
        for name in ("policy.onnx", "motion.csv", "exam_verdict.json",
                     "controller.env", "start_controller_damping_hold.sh")
    }
    manifest = {
        "schema": "deploy_bundle/v1",
        "dance": args.dance,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": {"file": "policy.onnx", "sha256": p_sha},
        "motion": {
            "file": "motion.csv",
            "sha256": m_sha,
            "duration_s": verdict["nominal"]["duration_s"],
        },
        # NOT the self-declared string: authorization was re-derived above via authorize().
        # rehearsal bundles are explicitly NOT authorized and carry the marker file.
        "exam": {"file": "exam_verdict.json", "authorized": not args.rehearsal},
        "rehearsal": bool(args.rehearsal),
        "files_sha256": files,
        "controller": {"image": CONTROLLER_IMAGE, "notes": "see deploy/README.md"},
        "target": {"pc2": PC2_HOST, "user": PC2_USER},
    }
    (out / "bundle.json").write_text(json.dumps(manifest, indent=1))
    tag = "REHEARSAL (not deployable)" if args.rehearsal else "authorized"
    print(f"bundle ready: {out}  [{tag}]")
    print(f"  policy {p_sha[:16]}…  motion {m_sha[:16]}…")
    print("next: deploy/02_push_bundle.sh --dance", args.dance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
