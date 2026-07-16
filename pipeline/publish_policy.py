"""Auto-publish a trained/pulled policy to the frontend — no manual steps.

The failure this closes: a training run finishes, its artifacts (policy.onnx, gap.json,
heldout_*.json, ...) get pulled into data/policies/<tag>/ by scripts/retrain_pull.sh —
and then nothing. The policy is on disk but the app never sees it, because a Dance
record is only created through the UI's attach-policy flow. So the Simulation tab (which
lists exactly the dances that have a policy_path) shows no video for that run. That is
why v6 and v7 had no sim preview.

publish() makes a completed+pulled run ALWAYS appear on the frontend:

  1. ensure_preview_assets() — the honest sim (tools/sim_studio via pipeline.sim_preview)
     needs, alongside policy.onnx, a policy-INDEPENDENT policy_meta.json and a *_deploy.npz
     motion in the same dir. A fresh pull usually has only policy.onnx (+ gap/heldout json),
     so we copy those two from data/policies/thriller/ (the shared preview motion) if absent.
  2. register_or_update() — find the Dance by name (create it if new) and attach_policy()
     so policy_path points at this run's policy.onnx. Uses the real store code
     (pipeline.shows) — no hand-written dance.json.
  3. sim_preview.render_sync() — render the honest preview (faithful mjlab model) so the
     video is on the frontend the instant the pull finishes.

ROBUSTNESS CONTRACT: publish() must never crash the pull. Asset/render failures are
logged and swallowed; a render failure still leaves a registered dance (the UI can
re-render on demand). Only an outright-missing policy.onnx makes publish() return None.

CLI (called from the pull/finalize path):
    python -m pipeline.publish_policy data/policies/thriller_v7ank \
        --name "Thriller — v7 (attempt 4)" [--no-render] [--async]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .config import PROJECT_ROOT
from . import shows, sim_preview

# The canonical Thriller policy dir supplies the shared, policy-INDEPENDENT preview
# assets (the deploy motion + meta) when a freshly pulled dir lacks them.
_SHARED = PROJECT_ROOT / "data" / "policies" / "thriller"

_README = """\
# Preview assets for this policy

This directory holds a policy pulled from a cloud training run. To render the
Simulation-tab preview (tools/sim_studio), pipeline/sim_preview needs, next to
`policy.onnx`:

  - `policy_meta.json`  — joint order / gains, IDENTICAL across Thriller policies
                          (policy-independent), copied from data/policies/thriller/.
  - `*_deploy.npz`      — the reference Thriller motion the preview plays as the
                          "intended dance" (left pane). This is the SHARED
                          `thriller_deploy` motion copied from data/policies/thriller/,
                          NOT this policy's own trajectory — it only drives the
                          reference/left side; the right side is this policy rolled out.

Both are added automatically by pipeline/publish_policy.py on pull if missing.
"""


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def ensure_preview_assets(policy_dir: Path, *, log=print) -> bool:
    """Make policy_dir render-ready. Returns True iff a policy.onnx is present.

    Copies policy_meta.json and a *_deploy.npz from data/policies/thriller/ if absent,
    and drops a README noting the shared-motion provenance. Never raises for a missing
    optional asset — logs and continues (the on-demand UI render can still be retried)."""
    policy_dir = Path(policy_dir)
    onnx = policy_dir / "policy.onnx"
    if not onnx.is_file():
        log(f"publish_policy: no policy.onnx in {_rel(policy_dir)} — nothing to publish")
        return False

    meta = policy_dir / "policy_meta.json"
    if not meta.is_file():
        src = _SHARED / "policy_meta.json"
        if src.is_file():
            shutil.copyfile(src, meta)
            log(f"publish_policy: copied shared policy_meta.json -> {_rel(meta)}")
        else:
            log(f"publish_policy: WARN shared policy_meta.json missing at {_rel(src)}")

    if not any(policy_dir.glob("*_deploy.npz")):
        src = next(_SHARED.glob("*_deploy.npz"), None)
        if src is not None:
            dst = policy_dir / src.name
            shutil.copyfile(src, dst)
            log(f"publish_policy: copied shared preview motion {src.name} -> {_rel(dst)}")
        else:
            log(f"publish_policy: WARN no *_deploy.npz in shared dir {_rel(_SHARED)}")

    readme = policy_dir / "README.md"
    if not readme.exists():
        try:
            readme.write_text(_README)
        except OSError as e:  # non-fatal
            log(f"publish_policy: could not write README ({e})")
    return True


def register_or_update(policy_dir: Path, name: str, *, notes: str | None = None,
                       log=print) -> shows.Dance:
    """Register a new Dance for `name` (or reuse the existing one) and attach this
    policy to it via the real store code. Returns the Dance."""
    onnx_rel = _rel(Path(policy_dir) / "policy.onnx")
    existing = shows.find_dance(name)
    if existing is None:
        dance = shows.new_dance(name, notes=notes or "")
        log(f"publish_policy: registered new dance '{name}' -> {dance.id}")
    else:
        dance = existing
        log(f"publish_policy: updating existing dance '{name}' -> {dance.id}")
    # attach_policy() sets policy_path and (correctly) resets verification state to
    # draft — this is a policy the sim exam has not yet passed.
    dance = shows.attach_policy(dance.id, onnx_rel, notes=notes)
    return dance


def publish(policy_dir, name: str, *, notes: str | None = None,
            render: bool = True, wait: bool = True, log=print) -> shows.Dance | None:
    """Full publish: ensure assets -> register/update dance -> render preview.

    render=True triggers the honest sim (faithful model). wait=True renders in the
    FOREGROUND (render_sync) — required in a short-lived CLI/pull process where a daemon
    thread would be killed on exit; wait=False (render_async) suits the long-lived server.
    Returns the Dance, or None only if there is no policy.onnx to publish. A render error
    is logged and swallowed: the dance still exists and the UI can re-render on demand."""
    policy_dir = Path(policy_dir)
    if not ensure_preview_assets(policy_dir, log=log):
        return None
    dance = register_or_update(policy_dir, name, notes=notes, log=log)
    if render:
        try:
            if wait:
                log(f"publish_policy: rendering honest sim preview for {dance.id} "
                    "(foreground, faithful model — a few minutes)…")
                res = sim_preview.render_sync(dance)
            else:
                res = sim_preview.render_async(dance)
            log(f"publish_policy: preview status for {dance.id}: {res.get('status')} "
                f"(sha {res.get('sha')})")
        except Exception as e:  # noqa: BLE001 — a preview failure must NOT fail the pull
            log(f"publish_policy: preview render failed for {dance.id} "
                f"({type(e).__name__}: {e}) — dance is registered; re-render in the UI")
    return dance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("policy_dir", type=Path, help="policy dir holding policy.onnx")
    ap.add_argument("--name", default=None,
                    help="dance name (default: the policy dir's basename)")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--no-render", action="store_true",
                    help="register/update the dance but skip the preview render")
    ap.add_argument("--async", dest="run_async", action="store_true",
                    help="render in a background thread (server context) instead of "
                         "blocking; NOT for a short-lived CLI process")
    args = ap.parse_args(argv)
    name = args.name or Path(args.policy_dir).resolve().name
    dance = publish(args.policy_dir, name, notes=args.notes,
                    render=not args.no_render, wait=not args.run_async)
    if dance is None:
        print("publish_policy: nothing published (no policy.onnx).", file=sys.stderr)
        return 1
    print(f"publish_policy: OK dance={dance.id} name={dance.name!r} "
          f"policy_path={dance.policy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
