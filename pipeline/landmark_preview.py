"""On-demand pose-estimation landmark overlay for the Simulation tab.

Renders the GVHMR skeleton projected onto the ORIGINAL source video (tools/landmark_overlay)
so the operator can visually debug "did pose-estimation track the dancer?" — the earliest,
cheapest place to catch garbage-in. Keyed by the source JOB id (the landmark overlay is a
property of the uploaded video, not of any particular trained policy).

Layout:  data/previews/landmarks/<job_id>.mp4  (+ <job_id>.json meta)
Served by the existing /previews static mount -> /previews/landmarks/<job_id>.mp4

Render is slow (SMPL-X forward + per-frame draw), so render_async() spawns a daemon
thread and returns immediately; the UI polls status(). No robot, no GPU.
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
LM_ROOT = DATA_DIR / "previews" / "landmarks"
JOBS_DIR = DATA_DIR / "jobs"

_status: dict[str, str] = {}     # job_id -> rendering|ready|failed:<msg>|unavailable
_lock = threading.Lock()


def _sources(job_id: str) -> tuple[Path, Path] | None:
    """(pred.pt, source video) for a job, or None if this job has no GVHMR extract
    (e.g. a CSV-sourced dance — no video to overlay)."""
    ex = JOBS_DIR / job_id / "extract"
    pred = ex / "hmr4d_results.pt"
    video = ex / "input_30fps.mp4"
    if not video.exists():
        video = JOBS_DIR / job_id / "input.mp4"
    if pred.exists() and video.exists():
        return pred, video
    return None


def status(job_id: str) -> dict:
    """Current landmark-overlay state for a job: ready(+url) / rendering / unavailable / failed."""
    if not job_id:
        return {"status": "unavailable", "reason": "dance has no source job"}
    mp4 = LM_ROOT / f"{job_id}.mp4"
    if mp4.exists():
        return {"status": "ready", "url": f"/previews/landmarks/{job_id}.mp4"}
    st = _status.get(job_id)
    if st:
        return {"status": st}
    if _sources(job_id) is None:
        return {"status": "unavailable",
                "reason": "no pose-estimation output for this dance (not video-sourced)"}
    return {"status": "idle"}


def render_async(job_id: str) -> dict:
    """Kick off (or reuse) a landmark overlay render for a job. Returns status now."""
    if not job_id:
        return {"status": "unavailable", "reason": "dance has no source job"}
    mp4 = LM_ROOT / f"{job_id}.mp4"
    with _lock:
        if _status.get(job_id) == "rendering":
            return {"status": "rendering"}
        if mp4.exists():
            _status.pop(job_id, None)
            return {"status": "ready", "url": f"/previews/landmarks/{job_id}.mp4"}
        if _sources(job_id) is None:
            return {"status": "unavailable",
                    "reason": "no pose-estimation output for this dance (not video-sourced)"}
        _status[job_id] = "rendering"
    threading.Thread(target=_render, args=(job_id,), daemon=True).start()
    return {"status": "rendering"}


def _render(job_id: str) -> None:
    try:
        src = _sources(job_id)
        if src is None:
            with _lock:
                _status[job_id] = "failed:no source video/pred"
            return
        LM_ROOT.mkdir(parents=True, exist_ok=True)
        mp4 = LM_ROOT / f"{job_id}.mp4"
        env = {**os.environ}
        conda_lib = Path.home() / "miniconda3/envs/g1dance/lib"
        if conda_lib.exists():
            env["LD_LIBRARY_PATH"] = f"{conda_lib}:{env.get('LD_LIBRARY_PATH', '')}"
        subprocess.run(
            [sys.executable, "-m", "tools.landmark_overlay", "--job",
             str(JOBS_DIR / job_id), "--out", str(mp4)],
            cwd=str(PROJECT_ROOT), check=True, timeout=2400, env=env)
        (LM_ROOT / f"{job_id}.json").write_text(json.dumps({
            "job_id": job_id, "created_at": time.time(), "kind": "landmark_overlay"}))
        with _lock:
            _status[job_id] = "ready"
    except Exception as e:  # noqa: BLE001 — surface to UI, never crash the server
        with _lock:
            _status[job_id] = f"failed:{str(e)[:200]}"
