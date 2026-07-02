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
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Body, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from pipeline.config import DATA_DIR, STAGE_ORDER
from pipeline import store
from pipeline.runner import Runner
from pipeline.stages.local_motion import build_stages

STATIC_DIR = Path(__file__).parent / "static"
PREVIEWS_DIR = DATA_DIR / "previews"
VET_SCRIPT = PROJECT_ROOT / "pipeline" / "vet_motion.py"

app = FastAPI(title="G1 Dance Studio")

# Vet runs load MuJoCo and FK every frame (~seconds); cache per (path, mtime).
_vet_cache: dict[tuple[str, float], dict] = {}

# ---- background job worker ---------------------------------------------------
# One worker thread executes queued jobs via the pipeline runner. Stage state is
# persisted by the runner (pipeline/store.py), so a crash/reboot loses nothing:
# _reconcile_jobs() re-queues whatever was interrupted.

_job_queue: "queue.Queue[str]" = queue.Queue()
_runner = Runner(build_stages())


def _worker_loop() -> None:
    while True:
        job_id = _job_queue.get()
        try:
            job = store.load_job(job_id)
            _runner.run_job(job)
        except Exception:
            try:
                store.load_job(job_id).log(
                    f"worker error:\n{traceback.format_exc()}")
            except Exception:
                pass
        finally:
            _job_queue.task_done()


def _reconcile_jobs() -> None:
    """On startup: re-queue interrupted/pending/blocked jobs; leave failed ones
    for the explicit Retry button."""
    for job in store.list_jobs():
        cur = job.current_stage()
        if cur is None:
            continue
        st = job.stages[cur]
        if st.state == "running":  # process died mid-stage
            st.state = "pending"
            st.message = "interrupted by restart — re-queued"
            st.progress = 0.0
            job.save()
            job.log(f"stage {cur}: was running at shutdown — re-queued")
        if st.state in ("pending", "blocked"):
            _job_queue.put(job.id)


@app.on_event("startup")
def _start_worker() -> None:
    _reconcile_jobs()
    threading.Thread(target=_worker_loop, name="job-worker", daemon=True).start()


def _job_dict(job: store.Job) -> dict:
    d = {
        "id": job.id,
        "name": job.name,
        "created_at": job.created_at,
        "input": job.input,
        "current_stage": job.current_stage(),
        "stages": {k: vars(v) for k, v in job.stages.items()},
    }
    preview = job.dir / "retarget" / "preview.mp4"
    if preview.exists():
        d["preview_url"] = f"/previews/job-{job.id}.mp4"
    vet = job.dir / "retarget" / "vet.json"
    if vet.exists():
        d["vet"] = json.loads(vet.read_text())
    return d


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


def _create_job(name: str, src: Path) -> store.Job:
    """Create a job from an input file: .csv = robot motion, else video."""
    kind = "csv" if src.suffix.lower() == ".csv" else "video"
    job = store.new_job(name, input={"type": kind, "source": str(src)})
    shutil.copyfile(src, job.dir / ("input.csv" if kind == "csv" else "input.mp4"))
    job.log(f"input {kind}: {src} ({src.stat().st_size} bytes)")
    _job_queue.put(job.id)
    return job


@app.post("/api/jobs")
def create_job(payload: dict = Body(...)) -> dict:
    """Create a job from a file already on disk (video, or motion CSV)."""
    src = Path(payload.get("input_path") or payload.get("video_path", "")).expanduser()
    if not src.is_file():
        raise HTTPException(400, f"input file not found: {src}")
    name = payload.get("name") or src.stem
    return _job_dict(_create_job(name, src))


@app.post("/api/jobs/upload")
async def create_job_upload(video: UploadFile) -> dict:
    """Create a job from a browser-style file upload."""
    tmp = DATA_DIR / "videos" / f"upload-{int(time.time())}-{video.filename}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        shutil.copyfileobj(video.file, f)
    name = Path(video.filename or "upload").stem
    return _job_dict(_create_job(name, tmp))


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str) -> dict:
    """Reset a failed/blocked current stage to pending and re-queue the job."""
    try:
        job = store.load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    cur = job.current_stage()
    if cur is None:
        raise HTTPException(400, "job is already complete")
    st = job.stages[cur]
    if st.state not in ("failed", "blocked"):
        raise HTTPException(400, f"stage {cur} is {st.state}; nothing to retry")
    st.state = "pending"
    st.message = ""
    st.progress = 0.0
    st.started_at = st.finished_at = None
    job.save()
    job.log(f"stage {cur}: reset by user — re-queued")
    _job_queue.put(job.id)
    return _job_dict(job)


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
