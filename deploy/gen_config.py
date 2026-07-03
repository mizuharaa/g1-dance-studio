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

from pipeline.exam_verdict import authorize, full_sha256, signature_valid

KIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = KIT_DIR.parent

PC2_HOST = "192.168.123.164"
PC2_USER = "unitree"
CONTROLLER_IMAGE = "qiayuanl/unitree:jazzy"
DANCE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # finding #31: no shell metachars to PC2


def sha256(p: Path) -> str:
    return full_sha256(p)  # full 64-hex identity (finding #32)


def motion_duration_s(motion_csv: Path, fps: float = 30.0) -> float:
    """Duration from the deployable CSV itself (rows minus header) / fps.

    Production-audit fix: the mjlab_heldout verdict has no nominal.duration_s (only
    sim_exam.py emitted that), so read the motion instead of the verdict — works for
    every verdict producer and the rehearsal path alike.
    """
    n_rows = max(sum(1 for _ in motion_csv.open()) - 1, 0)
    return round(n_rows / fps, 2)


def _verdict_files(explicit: Path | None, *dirs: Path):
    """Candidate verdict JSONs: an explicit --verdict wins; else exam_*.json AND
    *verdict*.json across the given dirs (the real producer, mjlab_verify, writes
    heldout_verdict.json — production-audit glob-mismatch fix)."""
    if explicit is not None:
        yield explicit
        return
    seen = set()
    for d in dirs:
        if not d or not d.is_dir():
            continue
        for f in sorted(list(d.glob("exam_*.json")) + list(d.glob("*verdict*.json"))):
            if f not in seen:
                seen.add(f)
                yield f


def _load(f: Path) -> dict | None:
    try:
        return json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def find_passing_exam(policy_sha: str, motion_sha: str, *dirs: Path,
                      explicit: Path | None = None) -> Path | None:
    """A verdict that AUTHORIZES show-ready for this exact policy+motion, or None.

    Findings #0/#7/#19/#21: authorization is derived from phase contents AND an HMAC
    signature — never the self-declared ``verdict`` string, never an empty-dict phase.
    """
    for f in _verdict_files(explicit, *dirs):
        v = _load(f)
        if v is None:
            continue
        ok, _reason = authorize(v, policy_sha=policy_sha, motion_sha=motion_sha)
        if ok:
            return f
    return None


def find_gantry_verdict(policy_sha: str, motion_sha: str, *dirs: Path,
                        explicit: Path | None = None) -> Path | None:
    """A SIGNED verdict that BINDS to this exact policy+motion — pass OR fail.

    The gantry test (feet off ground) exists to validate a policy that is NOT yet
    show-ready, so requiring derive_pass here would be backwards. But we still demand a
    genuine signed verdict bound to these exact artifacts: you cannot gantry a random,
    never-evaluated policy. This is strictly MORE than --rehearsal (which has no gate)
    and strictly LESS than the ground/show gate (which requires a >=99% pass).
    """
    for f in _verdict_files(explicit, *dirs):
        v = _load(f)
        if v is None or not signature_valid(v):
            continue
        if v.get("policy_sha256") == policy_sha and v.get("motion_sha256") == motion_sha:
            return f
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dance", required=True)
    ap.add_argument("--policy", required=True, type=Path)
    ap.add_argument("--motion", required=True, type=Path,
                    help="the DEPLOYABLE motion CSV (use thriller_deploy.csv — it has the "
                         "2.5s ramp from default_joint_pos so frame-0 matches standby; the "
                         "raw show clip would LURCH on activation)")
    ap.add_argument("--policy-meta", type=Path, default=None,
                    help="policy_meta.json with the SIM PD gains the policy trained against "
                         "(default: <policy dir>/policy_meta.json). The robot MUST load THESE "
                         "gains, not stock Unitree gains, or the low overdamped policy is "
                         "unstable — a fall risk. Carried in the bundle + asserted at start.")
    ap.add_argument(
        "--exam-dir", type=Path, default=PROJECT_ROOT / "data" / "exports",
        help="where sim_exam verdicts live",
    )
    ap.add_argument(
        "--verdict", type=Path, default=None,
        help="explicit verdict JSON (e.g. data/policies/<dance>/heldout_verdict.json). "
             "If omitted, exam_*.json and *verdict*.json are searched in --exam-dir and "
             "the policy's own directory.",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--gantry", action="store_true",
        help="GANTRY-ONLY: build a bundle for FEET-OFF-GROUND gantry testing of a policy "
             "that is NOT yet show-ready. Requires a genuine SIGNED verdict bound to this "
             "exact policy+motion (pass OR fail) — not a random policy. Stamped "
             "scope=gantry-only; 10_gantry_test.sh refuses --stage ground for it.",
    )
    mode.add_argument(
        "--rehearsal", action="store_true",
        help="REHEARSAL ONLY: assemble the bundle WITHOUT any gate to validate packaging "
             "mechanics. Stamps the bundle non-deployable — 02_push_bundle.sh refuses to "
             "push it. Never produces a pushable bundle.",
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
    search_dirs = (args.exam_dir, args.policy.resolve().parent)
    scope = "gantry-only" if args.gantry else "full"

    if args.rehearsal:
        # No gate — bundle is marked REHEARSAL_ONLY and can never be pushed.
        verdict = {"schema": "sim_exam/v1", "verdict": "rehearsal", "REHEARSAL_ONLY": True}
        (out / "exam_verdict.json").write_text(json.dumps(verdict, indent=1))
        (out / "REHEARSAL_ONLY").write_text(
            "Built with --rehearsal (no authorization). MUST NOT be pushed or run on a "
            "robot. Rebuild without --rehearsal once the dance is show-ready.\n")
        print("REHEARSAL bundle (NOT deployable): validating packaging only")
    elif args.gantry:
        gv = find_gantry_verdict(p_sha, m_sha, *search_dirs, explicit=args.verdict)
        if gv is None:
            raise SystemExit(
                "ABORT: no SIGNED verdict binds policy "
                f"{p_sha[:16]}… + motion {m_sha[:16]}… (searched --verdict / *verdict*.json "
                f"in {', '.join(str(d) for d in search_dirs)}).\n"
                "A gantry bundle still needs a genuine mjlab_verify verdict for this exact "
                "policy+motion — run pipeline.mjlab_verify first (pass NOT required).\n"
                "Note: motion_sha256 must match the DEPLOYABLE .csv — regenerate the verdict "
                "with --motion <csv> --eval-motion <npz> if it was signed against the .npz."
            )
        verdict = json.loads(gv.read_text())
        shutil.copy2(gv, out / "exam_verdict.json")
        (out / "GANTRY_ONLY").write_text(
            "scope=gantry-only. Built with --gantry from a signed but NOT-show-ready "
            "verdict (verdict=%r, %s/%s held-out). Valid ONLY for feet-off-ground gantry "
            "testing (10_gantry_test.sh --stage gantry). GROUND/show use is refused until a "
            ">=99%% show-ready verdict exists and a full bundle is rebuilt.\n" % (
                verdict.get("verdict"),
                verdict.get("nominal", {}).get("n_success", "?"),
                verdict.get("nominal", {}).get("num_episodes", "?"),
            ))
        print(f"GANTRY-ONLY bundle: signed verdict={verdict.get('verdict')!r} "
              f"(NOT show-ready) — feet-off-ground testing ONLY")
    else:
        exam = find_passing_exam(p_sha, m_sha, *search_dirs, explicit=args.verdict)
        if exam is None:
            raise SystemExit(
                "ABORT: no PASSING sim_exam/v1 verdict (signed, all phases pass, "
                f">=99% survival) matches policy {p_sha[:16]}… + motion {m_sha[:16]}… "
                f"in {', '.join(str(d) for d in search_dirs)}.\n"
                "For a not-yet-show-ready policy on the GANTRY, use --gantry instead.\n"
                "Run pipeline.mjlab_verify (or sim_exam) to produce a verdict first."
            )
        verdict = json.loads(exam.read_text())
        shutil.copy2(exam, out / "exam_verdict.json")

    shutil.copy2(args.policy, out / "policy.onnx")
    shutil.copy2(args.motion, out / "motion.csv")

    # SIM PD gains: the robot MUST load these (per-joint kp/kd/effort/default_pos, ζ=2
    # overdamped), NOT stock Unitree gains — the low overdamped policy is unstable on
    # stock gains (fall risk). Carried in the bundle, hash-pinned, asserted at start.
    meta_src = args.policy_meta or (args.policy.resolve().parent / "policy_meta.json")
    if not meta_src.exists():
        raise SystemExit(
            f"ABORT: policy_meta.json (SIM gains spec) not found at {meta_src}. "
            "The bundle cannot omit the gains the policy trained against — pass --policy-meta.")
    shutil.copy2(meta_src, out / "policy_meta.json")

    # controller.env: launch parameters as REVIEWED DATA (findings #8/#20). start_mode
    # is fixed to damping here and asserted by the start script — not hand-editable.
    (out / "controller.env").write_text(
        f"DANCE={args.dance}\nPOLICY=policy.onnx\nMOTION=motion.csv\n"
        f"GAINS=policy_meta.json\nUSE_SIM_GAINS=1\n"
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
        "# FALL-RISK GATE: the policy trained on SIM PD gains (policy_meta.json, low\n"
        "# overdamped ζ=2). Stock Unitree gains destabilize it. Require the gains spec and\n"
        "# an explicit attestation the controller loaded THESE gains (set on robot day).\n"
        "if [ \"${USE_SIM_GAINS:-}\" != \"1\" ] || [ ! -f /bundle/policy_meta.json ]; then\n"
        "  echo 'REFUSING: SIM gains (policy_meta.json) not present/selected.'; exit 77\n"
        "fi\n"
        "if [ ! -f /bundle/SIM_GAINS_LOADED ]; then\n"
        "  echo 'REFUSING: controller has not been confirmed to load the SIM PD gains from'\n"
        "  echo 'policy_meta.json (kp/kd/effort/default_pos). Stock gains = fall risk.'\n"
        "  echo 'See ROBOT_DAY_PLAN step 3 — verify gains, then touch SIM_GAINS_LOADED.'\n"
        "  exit 76\n"
        "fi\n"
        "# ACTIVATION HAZARD: motion.csv (thriller_deploy) begins with a 2.5s ramp from\n"
        "# default_joint_pos so frame-0 == standby (delta ~0). Do NOT substitute the raw\n"
        "# show clip: activation would lurch (up to ~39deg elbow/knee step).\n"
        "if [ ! -f /bundle/LAUNCH_LINE_VERIFIED ]; then\n"
        "  echo 'REFUSING: controller launch line not verified on robot day yet.'\n"
        "  echo 'See docs/ROBOT_DAY_PLAN.md step 3 — then touch LAUNCH_LINE_VERIFIED'\n"
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
        for name in ("policy.onnx", "motion.csv", "policy_meta.json", "exam_verdict.json",
                     "controller.env", "start_controller_damping_hold.sh")
    }
    manifest = {
        "schema": "deploy_bundle/v1",
        "dance": args.dance,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # scope gates downstream use: "full" = show-ready, deployable to ground/show;
        # "gantry-only" = feet-off-ground testing ONLY; "rehearsal" = never pushable.
        "scope": "rehearsal" if args.rehearsal else scope,
        "policy": {"file": "policy.onnx", "sha256": p_sha},
        "motion": {
            "file": "motion.csv",
            "sha256": m_sha,
            # Production-audit fix: duration from the motion itself, not verdict.nominal
            # (mjlab verdicts have no duration_s → KeyError before any bundle was written).
            "duration_s": motion_duration_s(args.motion),
        },
        # NOT the self-declared string. "authorized" means SHOW-READY (full ground/show
        # deploy) — re-derived via authorize() above. A gantry-only bundle is bound to a
        # signed verdict but is NOT show-ready, so authorized=False + gantry_authorized=True.
        "exam": {
            "file": "exam_verdict.json",
            "authorized": scope == "full" and not args.rehearsal,
            "gantry_authorized": args.gantry,
        },
        "rehearsal": bool(args.rehearsal),
        "files_sha256": files,
        "controller": {"image": CONTROLLER_IMAGE, "notes": "see deploy/README.md"},
        "target": {"pc2": PC2_HOST, "user": PC2_USER},
    }
    (out / "bundle.json").write_text(json.dumps(manifest, indent=1))
    tag = {"rehearsal": "REHEARSAL (not deployable)",
           "gantry-only": "GANTRY-ONLY (feet-off-ground testing)",
           "full": "authorized (show-ready)"}[manifest["scope"]]
    print(f"bundle ready: {out}  [{tag}]")
    print(f"  policy {p_sha[:16]}…  motion {m_sha[:16]}…  ({motion_duration_s(args.motion)}s)")
    print("next: deploy/02_push_bundle.sh --dance", args.dance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
