"""Local engine behind the G1 Dance Studio desktop app.

FastAPI app serving the ui/static frontend plus a JSON API over the pipeline's
job store (pipeline/store.py). This process never talks to the robot: the
deploy endpoint only records an explicit human confirmation request on disk.

Run standalone for development:
    python ui/server.py --port 8735
Normally launched inside a pywebview window by ui/desktop.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Body, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from pipeline.config import DATA_DIR, STAGE_ORDER
from pipeline import store

STATIC_DIR = Path(__file__).parent / "static"
PREVIEWS_DIR = DATA_DIR / "previews"
VET_SCRIPT = PROJECT_ROOT / "pipeline" / "vet_motion.py"

app = FastAPI(title="G1 Dance Studio")

# Vet runs load MuJoCo and FK every frame (~seconds); cache per (path, mtime).
_vet_cache: dict[tuple[str, float], dict] = {}


def _job_dict(job: store.Job) -> dict:
    return {
        "id": job.id,
        "name": job.name,
        "created_at": job.created_at,
        "current_stage": job.current_stage(),
        "stages": {k: vars(v) for k, v in job.stages.items()},
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "stage_order": STAGE_ORDER}


@app.get("/api/jobs")
def jobs() -> list[dict]:
    return [_job_dict(j) for j in store.list_jobs()]


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str) -> dict:
    try:
        job = store.load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    d = _job_dict(job)
    log = job.dir / "log.txt"
    d["log_tail"] = log.read_text().splitlines()[-40:] if log.exists() else []
    return d


def _create_job_from_video(name: str, src: Path) -> store.Job:
    job = store.new_job(name)
    shutil.copyfile(src, job.dir / "input.mp4")
    job.log(f"input video: {src} ({src.stat().st_size} bytes)")
    # Stage implementations land in later phases; the job waits at "extract".
    job.log("job queued — pipeline stages not yet implemented in this skeleton")
    return job


@app.post("/api/jobs")
def create_job(payload: dict = Body(...)) -> dict:
    """Create a job from a video already on disk (path from the file picker)."""
    video_path = Path(payload.get("video_path", "")).expanduser()
    if not video_path.is_file():
        raise HTTPException(400, f"video file not found: {video_path}")
    name = payload.get("name") or video_path.stem
    return _job_dict(_create_job_from_video(name, video_path))


@app.post("/api/jobs/upload")
async def create_job_upload(video: UploadFile) -> dict:
    """Create a job from a browser-style file upload."""
    tmp = DATA_DIR / "videos" / f"upload-{int(time.time())}-{video.filename}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        shutil.copyfileobj(video.file, f)
    name = Path(video.filename or "upload").stem
    return _job_dict(_create_job_from_video(name, tmp))


@app.post("/api/jobs/{job_id}/deploy")
def deploy_gate(job_id: str, payload: dict = Body(...)) -> dict:
    """Deploy-gate PLACEHOLDER. Records the request; never contacts the robot.

    Real deployment (a later phase) additionally requires: sim2sim verify PASS,
    robot secured (gantry / clear area), e-stop in hand — always human-confirmed.
    """
    try:
        job = store.load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    if payload.get("confirm_phrase") != "DEPLOY":
        raise HTTPException(400, 'deployment requires typing the phrase "DEPLOY"')

    record = {"requested_at": time.time(), "job": job_id,
              "note": "placeholder — nothing was sent to the robot"}
    req_file = job.dir / "deploy_requests.json"
    requests = json.loads(req_file.read_text()) if req_file.exists() else []
    requests.append(record)
    req_file.write_text(json.dumps(requests, indent=2))
    job.log("deploy requested by user — recorded only (deploy not implemented)")
    return {"recorded": True, "deployed": False,
            "note": "Deployment is not implemented yet. Nothing was sent to the robot."}


@app.get("/api/motions")
def motions() -> list[dict]:
    """Motion CSVs available for vetting (data/*.csv, newest first)."""
    files = sorted(DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    return [{"name": p.name, "path": str(p.relative_to(PROJECT_ROOT))}
            for p in files]


@app.get("/api/vet")
def vet(csv: str) -> dict:
    """Run the vetting gate on a motion CSV and return its JSON report."""
    path = (PROJECT_ROOT / csv).resolve()
    if not path.is_relative_to(PROJECT_ROOT) or path.suffix != ".csv":
        raise HTTPException(400, "csv must be a .csv path inside the project")
    if not path.is_file():
        raise HTTPException(404, f"not found: {csv}")

    key = (str(path), path.stat().st_mtime)
    if key not in _vet_cache:
        proc = subprocess.run(
            [sys.executable, str(VET_SCRIPT), str(path), "--json"],
            capture_output=True, text=True, timeout=300)
        if not proc.stdout:
            raise HTTPException(500, f"vet_motion failed: {proc.stderr[-500:]}")
        _vet_cache[key] = json.loads(proc.stdout)
    return _vet_cache[key]


@app.get("/api/previews")
def previews() -> list[dict]:
    files = sorted(PREVIEWS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    return [{"name": p.name, "url": f"/previews/{p.name}",
             "size": p.stat().st_size} for p in files]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
# StaticFiles handles HTTP Range requests, so the <video> player can seek.
app.mount("/previews", StaticFiles(directory=PREVIEWS_DIR), name="previews")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8735)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
