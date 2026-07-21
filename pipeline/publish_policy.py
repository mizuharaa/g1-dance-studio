"""Auto-publish a trained/pulled policy to the frontend — no manual steps.

The failure this closes: a training run finishes, its artifacts (policy.onnx, gap.json,
heldout_*.json, ...) get pulled into data/policies/<tag>/ by scripts/retrain_pull.sh —
and then nothing. The policy is on disk but the app never sees it, because a Dance
record is only created through the UI's attach-policy flow. So the Simulation tab (which
lists exactly the dances that have a policy_path) shows no video for that run. That is
why v6 and v7 had no sim preview.

publish() makes a completed+pulled run ALWAYS appear on the frontend:

  1. ensure_preview_assets() — the hardware-uncertainty scene
     needs, alongside policy.onnx, that policy's OWN policy_meta.json and a *_deploy.npz
     motion. Metadata is never borrowed. Motion prefers this policy's own lineage and uses
     the shared thriller_deploy.npz only as a last-resort visual reference.
  2. register_or_update() — find the Dance by name (create it if new) and attach_policy()
     so policy_path points at this run's policy.onnx. Uses the real store code
     (pipeline.shows) — no hand-written dance.json.
  3. sim_preview.render_sync() — render the hardware-uncertainty preview so the
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
import os
import shutil
import sys
from pathlib import Path

from .config import PROJECT_ROOT
from . import shows, sim_preview

# The canonical Thriller policy dir supplies only a last-resort preview motion.
# Metadata is policy-specific and must never be copied between policy graphs.
_SHARED = PROJECT_ROOT / "data" / "policies" / "thriller"

_README = """\
# Preview assets for this policy

This directory holds a policy pulled from a cloud training run. To render the
Simulation-tab preview (tools/sim_studio), pipeline/sim_preview needs, next to
`policy.onnx`:

  - `policy_meta.json`  — this policy's export-time contract. Its `onnx_inputs`
                          must exactly match this directory's ONNX graph.
  - `*_deploy.npz`      — the reference motion the preview plays as the "intended dance"
                          (left pane) AND feeds as the policy's command input (right pane).
                          Preferred source is THIS policy's own lineage: the staged npz
                          pulled from the run, else a conversion of the pulled deploy CSV.
                          Only if neither is available is the SHARED `thriller_deploy`
                          motion copied as a last resort (wrong-lineage — see finding C).

The motion may be added automatically. Metadata is never synthesized or borrowed;
without a matching sidecar this dance is registered without a preview.
"""


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


# csv_to_npz (vendored mjlab) hardcodes its output sink to /tmp/motion.npz and does NOT
# clear it between runs (audit finding B), so we rm it first and copy the fresh file out.
_CSV2NPZ_SINK = Path("/tmp/motion.npz")


def _convert_csv_to_npz(csv_path: Path, dst: Path, *, log=print) -> bool:
    """Convert a pulled deploy CSV to a *_deploy.npz via mjlab's csv_to_npz FK, so the
    preview plays THIS policy's own motion (this is the correct-lineage source, unlike the
    shared thriller_deploy.npz). Mirrors pipeline/stages/cloud_motion.py's CONVERT_SCRIPT
    (30->50 fps). Returns True iff it staged a non-empty npz.

    DEFENSIVE by contract: mjlab is NOT installed on the laptop, so locally this returns
    False and the caller falls back to the shared motion. Never raises."""
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("mjlab") is None:
        log(f"publish_policy: mjlab not available locally — cannot convert "
            f"{_rel(csv_path)} to npz (will fall back to shared motion)")
        return False
    try:
        _CSV2NPZ_SINK.unlink()                       # never trust a stale /tmp sink (finding B)
    except FileNotFoundError:
        pass
    except OSError as e:
        log(f"publish_policy: could not clear {_CSV2NPZ_SINK} ({e})")
        return False
    env = {**os.environ, "MUJOCO_GL": "egl", "WANDB_MODE": "offline"}
    cmd = [sys.executable, "-m", "mjlab.scripts.csv_to_npz",
           "--input-file", str(csv_path), "--output-name", dst.stem,
           "--input-fps", "30", "--output-fps", "50"]
    try:
        r = subprocess.run(cmd, env=env, check=False, timeout=900,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as e:  # noqa: BLE001 — conversion must never crash publish
        log(f"publish_policy: csv_to_npz failed ({type(e).__name__}: {e})")
        return False
    if not (_CSV2NPZ_SINK.is_file() and _CSV2NPZ_SINK.stat().st_size > 0):
        tail = (r.stdout or b"").decode("utf-8", "replace")[-300:]
        log(f"publish_policy: csv_to_npz produced no {_CSV2NPZ_SINK} (rc={r.returncode}); "
            f"falling back to shared motion. tail: {tail!r}")
        return False
    try:
        shutil.copyfile(_CSV2NPZ_SINK, dst)
    except OSError as e:
        log(f"publish_policy: could not stage converted npz ({e})")
        return False
    log(f"publish_policy: converted {_rel(csv_path)} -> {_rel(dst)} "
        "(this policy's OWN motion)")
    return True


def _onnx_inputs(path: Path) -> dict[str, list]:
    """Return graph input names and shapes without running policy code."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    initializers = {value.name for value in model.graph.initializer}
    inputs: dict[str, list] = {}
    for value in model.graph.input:
        if value.name in initializers:
            continue
        shape = []
        for dim in value.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            elif dim.HasField("dim_param") and dim.dim_param:
                shape.append(str(dim.dim_param))
            else:
                shape.append(None)
        inputs[value.name] = shape
    return inputs


def policy_meta_matches_onnx(policy_dir: Path, *, log=print) -> bool:
    """Fail closed unless the policy-specific sidecar matches the graph exactly."""
    policy_dir = Path(policy_dir)
    meta_path = policy_dir / "policy_meta.json"
    if not meta_path.is_file():
        log(
            "publish_policy: REFUSED preview — no policy-specific policy_meta.json "
            f"beside {_rel(policy_dir / 'policy.onnx')}"
        )
        return False
    try:
        import json

        meta = json.loads(meta_path.read_text())
        actual = _onnx_inputs(policy_dir / "policy.onnx")
        declared = meta.get("onnx_inputs")
        if declared != actual:
            log(
                "publish_policy: REFUSED preview — policy_meta onnx_inputs "
                f"{declared!r} != actual ONNX inputs {actual!r}"
            )
            return False
        for field in ("obs_per_frame", "history_length", "flatten_layout",
                      "requires_ground_contact", "actor_obs_terms_in_order"):
            if field not in meta:
                log(f"publish_policy: REFUSED preview — policy_meta missing {field}")
                return False
    except Exception as exc:  # noqa: BLE001 — publish must never fail the pull
        log(
            "publish_policy: REFUSED preview — cannot verify policy_meta against ONNX "
            f"({type(exc).__name__}: {exc})"
        )
        return False
    return True


def ensure_preview_assets(policy_dir: Path, *, log=print) -> bool:
    """Make policy_dir render-ready. Returns True iff a policy.onnx is present.

    Never copies or creates policy metadata. For the *_deploy.npz preview motion,
    PREFERS this policy's own lineage: the pulled staged npz, else a
    conversion of the pulled deploy CSV (mjlab csv_to_npz FK), and only as a LAST resort
    the shared thriller_deploy.npz. Drops a README noting provenance. Never raises for a
    missing optional asset — logs and continues (the on-demand UI render can be retried).

    NEVER-RAISE CONTRACT: publish() returns None (an UNregistered dance) when this returns
    False, which reintroduces the v6/v7 no-preview regression — so this must return True
    whenever policy.onnx exists, no matter what happens to the optional preview assets."""
    policy_dir = Path(policy_dir)
    onnx = policy_dir / "policy.onnx"
    if not onnx.is_file():
        log(f"publish_policy: no policy.onnx in {_rel(policy_dir)} — nothing to publish")
        return False

    if not any(policy_dir.glob("*_deploy.npz")):
        # Prefer THIS policy's own motion. The manual pull (retrain_pull.sh) now pulls the
        # staged *_deploy.npz directly; if that is somehow absent, convert the pulled deploy
        # CSV via mjlab csv_to_npz FK. Only if BOTH are unavailable do we copy the shared
        # thriller_deploy.npz — the wrong-lineage Jul-7 retarget that used to drive both the
        # reference pane AND the policy command input for every pulled policy (finding C).
        dst = policy_dir / f"{policy_dir.name}_deploy.npz"
        csv = (next(policy_dir.glob("*_clean.csv"), None)
               or next(policy_dir.glob("*_deploy.csv"), None))
        if csv is not None and _convert_csv_to_npz(csv, dst, log=log):
            pass  # staged this policy's own converted motion
        else:
            src = next(_SHARED.glob("*_deploy.npz"), None)
            if src is not None:
                shared_dst = policy_dir / src.name
                shutil.copyfile(src, shared_dst)
                log(f"publish_policy: LAST-RESORT copied shared preview motion {src.name} "
                    f"-> {_rel(shared_dst)} (NOT this policy's lineage — no own npz/csv)")
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

    render=True triggers the hardware-uncertainty scene. wait=True renders in the
    FOREGROUND (render_sync) — required in a short-lived CLI/pull process where a daemon
    thread would be killed on exit; wait=False (render_async) suits the long-lived server.
    Returns the Dance, or None only if there is no policy.onnx to publish. A render error
    is logged and swallowed: the dance still exists and the UI can re-render on demand."""
    policy_dir = Path(policy_dir)
    if not ensure_preview_assets(policy_dir, log=log):
        return None
    contract_ok = policy_meta_matches_onnx(policy_dir, log=log)
    dance = register_or_update(policy_dir, name, notes=notes, log=log)
    # attach_policy invalidates the exam, and the old preview is equally stale. Save
    # before attempting a replacement so every failure path remains fail closed.
    dance.preview = None
    dance.save()
    if render and contract_ok:
        try:
            if wait:
                log(f"publish_policy: rendering hardware-uncertainty preview for {dance.id} "
                    "(foreground, hardware-uncertainty scene — a few minutes)…")
                res = sim_preview.render_sync(dance)
            else:
                res = sim_preview.render_async(dance)
            log(f"publish_policy: preview status for {dance.id}: {res.get('status')} "
                f"(sha {res.get('sha')})")
        except Exception as e:  # noqa: BLE001 — a preview failure must NOT fail the pull
            log(f"publish_policy: preview render failed for {dance.id} "
                f"({type(e).__name__}: {e}) — dance is registered; re-render in the UI")
    elif render:
        log(
            f"publish_policy: dance {dance.id} registered WITHOUT preview because its "
            "policy metadata is missing or does not match the ONNX"
        )
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
