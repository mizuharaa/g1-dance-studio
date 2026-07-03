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
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

KIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = KIT_DIR.parent

PC2_HOST = "192.168.123.164"
PC2_USER = "unitree"
CONTROLLER_IMAGE = "qiayuanl/unitree:jazzy"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def find_passing_exam(policy_sha: str, motion_sha: str, exam_dir: Path) -> Path | None:
    for f in sorted(exam_dir.glob("exam_*.json")):
        try:
            v = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if (
            v.get("schema") == "sim_exam/v1"
            and v.get("verdict") == "pass"
            and v.get("policy_sha256") == policy_sha
            and v.get("motion_sha256") == motion_sha
            and v.get("push") is not None
            and v.get("repeatability") is not None
        ):
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
    args = ap.parse_args()

    for f in (args.policy, args.motion):
        if not f.exists():
            raise SystemExit(f"ABORT: {f} does not exist")

    p_sha, m_sha = sha256(args.policy), sha256(args.motion)
    exam = find_passing_exam(p_sha, m_sha, args.exam_dir)
    if exam is None:
        raise SystemExit(
            "ABORT: no PASSING sim_exam/v1 verdict (with push + repeatability phases) "
            f"matches policy {p_sha} + motion {m_sha} in {args.exam_dir}.\n"
            "Run: python -m pipeline.sim_exam --policy <onnx> --motion <csv> first."
        )

    verdict = json.loads(exam.read_text())
    out = KIT_DIR / "bundles" / args.dance
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.policy, out / "policy.onnx")
    shutil.copy2(args.motion, out / "motion.csv")
    shutil.copy2(exam, out / "exam_verdict.json")

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
        "exam": {"file": "exam_verdict.json", "verdict": verdict["verdict"]},
        "controller": {"image": CONTROLLER_IMAGE, "notes": "see deploy/README.md"},
        "target": {"pc2": PC2_HOST, "user": PC2_USER},
    }
    (out / "bundle.json").write_text(json.dumps(manifest, indent=1))
    (out / "controller.env").write_text(
        f"DANCE={args.dance}\nPOLICY=policy.onnx\nMOTION=motion.csv\n"
        f"CONTROLLER_IMAGE={CONTROLLER_IMAGE}\nCONTROL_HZ=50\n"
    )
    # In-container entrypoint used by 10_gantry_test.sh. The exact controller
    # launch line is pinned on robot day against the controller README (runbook
    # step 3) — until then this refuses to run, which is itself an interlock.
    start = out / "start_controller_damping_hold.sh"
    start.write_text(
        "#!/usr/bin/env bash\n"
        "# Runs INSIDE qiayuanl/unitree:jazzy on PC2. Contract: load policy, hold\n"
        "# damping; motion playback is armed only by the operator's remote sequence.\n"
        "set -euo pipefail\n"
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
    print(f"bundle ready: {out}")
    print(f"  policy {p_sha}  motion {m_sha}  exam {exam.name} (pass)")
    print("next: deploy/02_push_bundle.sh --dance", args.dance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
