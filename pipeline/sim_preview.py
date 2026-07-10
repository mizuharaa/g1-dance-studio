"""On-demand policy-in-the-loop sim previews for the Simulation tab.

Renders the honest sim (tools/sim_studio: REFERENCE vs POLICY) for a dance's current policy
and stores it VERSIONED by policy sha, so a retrain produces a NEW version while the OLD one
is kept — that is what lets the UI show before-vs-after side by side.

Layout:  data/previews/sim/<dance_id>/<sha8>.mp4  (+ <sha8>.json meta)
Served by the existing /previews static mount -> /previews/sim/<dance_id>/<sha8>.mp4

Render is slow (~1-2 min), so render_async() spawns a daemon thread and returns immediately;
the UI polls list_sims() for status. No robot, no GPU — pure MuJoCo + onnxruntime.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from pipeline.config import DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIM_ROOT = DATA_DIR / "previews" / "sim"

_status: dict[tuple[str, str], str] = {}     # (dance_id, sha8) -> rendering|ready|failed:<msg>
_lock = threading.Lock()


def _sha8(dance) -> str:
    return (getattr(dance, "policy_sha256", None) or "nopolicy")[:8]


def _sim_dir(dance_id: str) -> Path:
    return SIM_ROOT / dance_id


def _policy_dir(dance) -> Path:
    """Dir holding policy.onnx + policy_meta.json + *_deploy.npz (sim_studio --dance)."""
    if not getattr(dance, "policy_path", None):
        raise ValueError("dance has no policy_path — train it first")
    return (PROJECT_ROOT / dance.policy_path).parent


def list_sims(dance_id: str) -> list[dict]:
    """All stored sim versions for a dance, newest first, plus any in-flight render."""
    out: list[dict] = []
    d = _sim_dir(dance_id)
    if d.exists():
        for j in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                meta = json.loads(j.read_text())
            except Exception:
                meta = {}
            sha = j.stem
            out.append({
                "sha": sha,
                "url": f"/previews/sim/{dance_id}/{sha}.mp4",
                "achieved": meta.get("right_achieved"),
                "created_at": meta.get("created_at"),
                "policy_sha256": meta.get("policy_sha256"),
                "status": _status.get((dance_id, sha), "ready"),
            })
    seen = {o["sha"] for o in out}
    for (did, sha), st in list(_status.items()):
        if did == dance_id and sha not in seen and st != "ready":
            out.append({"sha": sha, "url": None, "achieved": None,
                        "created_at": None, "status": st})
    return out


def render_async(dance) -> dict:
    """Kick off (or reuse) a render of the dance's CURRENT policy. Returns status now."""
    sha = _sha8(dance)
    key = (dance.id, sha)
    mp4 = _sim_dir(dance.id) / f"{sha}.mp4"
    with _lock:
        if _status.get(key) == "rendering":
            return {"status": "rendering", "sha": sha}
        if mp4.exists():
            _status.pop(key, None)
            return {"status": "ready", "sha": sha,
                    "url": f"/previews/sim/{dance.id}/{sha}.mp4"}
        _status[key] = "rendering"
    threading.Thread(target=_render, args=(dance, sha), daemon=True).start()
    return {"status": "rendering", "sha": sha}


def _render(dance, sha: str) -> None:
    key = (dance.id, sha)
    try:
        d = _sim_dir(dance.id)
        d.mkdir(parents=True, exist_ok=True)
        mp4, meta_p = d / f"{sha}.mp4", d / f"{sha}.report.json"
        subprocess.run(
            [sys.executable, "-m", "tools.sim_studio", "--dance", str(_policy_dir(dance)),
             "--steps", "1600", "--tether-kp", "0",     # 0 = honest amplitude (no base pinning)
             "--out", str(mp4), "--report", str(meta_p)],
            cwd=str(PROJECT_ROOT), check=True, timeout=1800,
            env={**os.environ, "MUJOCO_GL": "egl"})
        report = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        (d / f"{sha}.json").write_text(json.dumps({
            "label": getattr(dance, "name", dance.id),
            "policy_sha256": getattr(dance, "policy_sha256", None),
            "created_at": time.time(),
            "kind": "reference_vs_policy",
            **report,
        }))
        with _lock:
            _status[key] = "ready"
    except Exception as e:  # noqa: BLE001 — surface to the UI, never crash the server
        with _lock:
            _status[key] = f"failed:{str(e)[:200]}"
