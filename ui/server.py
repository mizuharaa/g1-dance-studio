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
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Body, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from pipeline.config import DATA_DIR, STAGE_ORDER
from pipeline import body_models, cloud, shows, store
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
    shows.seed_initial_dances()
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
    # BUG-1: the client controls filename — keep only its basename so a name
    # like "../../evil.sh" cannot escape the videos directory.
    safe_name = Path(video.filename or "upload").name
    tmp = DATA_DIR / "videos" / f"upload-{int(time.time())}-{safe_name}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        shutil.copyfileobj(video.file, f)
    name = Path(safe_name).stem
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


# ---- cloud GPU box (GreenNode) ------------------------------------------------
# Config in .secrets/cloud.json; this server only talks to the GPU box, never
# the robot. Last test result is cached so the UI can poll cheaply.

_cloud_last: dict = {}


@app.get("/api/cloud")
def cloud_info() -> dict:
    return {"config": cloud.masked_config(), "last_test": _cloud_last or None}


@app.post("/api/cloud/config")
def cloud_config(payload: dict = Body(...)) -> dict:
    if payload.get("transport") not in ("", "ssh", "jupyter", None):
        raise HTTPException(400, "transport must be 'ssh' or 'jupyter'")
    cloud.update_config(payload)
    return {"config": cloud.masked_config()}


@app.post("/api/cloud/test")
def cloud_test() -> dict:
    global _cloud_last
    _cloud_last = cloud.test_connection()
    return _cloud_last


# ---- body models ----------------------------------------------------------------

@app.get("/api/bodymodels")
def bodymodels_status() -> dict:
    return body_models.status()


@app.post("/api/bodymodels/install")
def bodymodels_install() -> dict:
    try:
        return body_models.install()
    except RuntimeError as e:
        raise HTTPException(400, str(e))


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


# ---- show mode: dance library ---------------------------------------------------
# Persistence in pipeline/shows.py. The deploy step inside a show remains
# RECORD-ONLY, exactly like the per-job deploy gate above.

def _dance_dict(d: shows.Dance) -> dict:
    out = asdict(d)
    out["repeatability_target"] = shows.REPEATABILITY_TARGET
    return out


def _load_dance_or_404(dance_id: str) -> shows.Dance:
    try:
        return shows.load_dance(dance_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such dance: {dance_id}")


@app.get("/api/dances")
def dances() -> list[dict]:
    return [_dance_dict(d) for d in shows.list_dances()]


@app.get("/api/dances/{dance_id}")
def dance_detail(dance_id: str) -> dict:
    return _dance_dict(_load_dance_or_404(dance_id))


@app.post("/api/dances")
def register_dance(payload: dict = Body(...)) -> dict:
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "dance needs a name")
    if shows.find_dance(name):
        raise HTTPException(400, f"dance '{name}' already exists")
    fields = {k: payload[k] for k in
              ("duration_s", "motion_csv", "policy_path", "preview", "notes",
               "source_job") if k in payload}
    return _dance_dict(shows.new_dance(name, **fields))


@app.post("/api/dances/{dance_id}/promote")
def promote_dance(dance_id: str, payload: dict = Body(...)) -> dict:
    dance = _load_dance_or_404(dance_id)
    try:
        return _dance_dict(shows.promote(dance, payload.get("status", "")))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/dances/{dance_id}/sim-runs")
def record_sim_run(dance_id: str, payload: dict = Body(...)) -> dict:
    """Repeatability contract endpoint — see docs/show_mode_contracts.md §1.

    Findings #23/#24: the caller must submit a signed sim_exam/v1 verdict (inline as
    `verdict`, or a path as `verdict_path`), NOT a bare `passed` bool. The verdict's
    HMAC is verified, it must bind to this dance's exact policy+motion bytes, and the
    pass is re-derived from phase contents before any clean-streak credit."""
    _load_dance_or_404(dance_id)  # 404 before touching the verdict
    verdict = payload.get("verdict")
    if verdict is None and payload.get("verdict_path"):
        vp = Path(payload["verdict_path"]).expanduser()
        if not vp.is_file():
            raise HTTPException(400, f"verdict_path not found: {vp}")
        verdict = json.loads(vp.read_text())
    if not isinstance(verdict, dict):
        raise HTTPException(400, "payload needs a signed sim_exam/v1 'verdict' "
                                 "(object) or 'verdict_path'")
    try:
        dance = shows.record_sim_run_from_verdict(dance_id, verdict)
    except shows.VerdictError as e:
        raise HTTPException(400, str(e))
    return _dance_dict(dance)


# ---- show mode: performances (pre-show checklist -> deploy gate -> outcome) -----

def _show_dict(s: shows.Show) -> dict:
    out = asdict(s)
    out["next_step"] = s.next_step()
    out["checklist_complete"] = s.checklist_complete()
    out["checklist_spec"] = shows.CHECKLIST_STEPS
    return out


def _load_show_or_404(show_id: str) -> shows.Show:
    try:
        return shows.load_show(show_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such show: {show_id}")


@app.get("/api/shows")
def show_history() -> list[dict]:
    return [_show_dict(s) for s in shows.list_shows()]


@app.get("/api/shows/{show_id}")
def show_detail(show_id: str) -> dict:
    return _show_dict(_load_show_or_404(show_id))


@app.post("/api/shows")
def create_show(payload: dict = Body(...)) -> dict:
    dance = _load_dance_or_404(payload.get("dance_id", ""))
    operator = (payload.get("operator") or "").strip()
    if not operator:
        raise HTTPException(400, "operator name is required")
    return _show_dict(shows.new_show(dance, operator))


@app.post("/api/shows/{show_id}/steps/{step}")
def complete_show_step(show_id: str, step: str, payload: dict = Body(...)) -> dict:
    show = _load_show_or_404(show_id)
    try:
        return _show_dict(shows.complete_step(show, step, payload.get("value")))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/shows/{show_id}/deploy")
def show_deploy_gate(show_id: str, payload: dict = Body(...)) -> dict:
    """Show-mode deploy gate. RECORD-ONLY, like the per-job gate: unlocks only
    after the full checklist, still requires the typed phrase, never contacts
    the robot."""
    show = _load_show_or_404(show_id)
    if show.closed:
        raise HTTPException(400, "show is closed")
    if not show.checklist_complete():
        raise HTTPException(400,
            f"pre-show checklist incomplete — next step: '{show.next_step()}'")
    if payload.get("confirm_phrase") != "DEPLOY":
        raise HTTPException(400, 'deployment requires typing the phrase "DEPLOY"')
    show.deploy = {"requested_at": time.time(),
                   "note": "placeholder — nothing was sent to the robot"}
    show.save()
    show.log("deploy requested by operator — RECORDED ONLY (no robot contact)")
    return {"recorded": True, "deployed": False, "show": _show_dict(show),
            "note": "Deployment is not implemented yet. Nothing was sent to the robot."}


@app.post("/api/shows/{show_id}/outcome")
def record_outcome(show_id: str, payload: dict = Body(...)) -> dict:
    show = _load_show_or_404(show_id)
    # finding #9/#28: close + (on incident/abort) demote-and-reset happen under record
    # locks inside shows.record_outcome, so a concurrent sim-run can't race the streak.
    try:
        show = shows.record_outcome(show, payload.get("result", ""),
                                    payload.get("notes", ""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _show_dict(show)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
# StaticFiles handles HTTP Range requests, so the <video> player can seek.
# follow_symlink: job previews are symlinks into data/jobs/<id>/retarget/.
app.mount("/previews", StaticFiles(directory=PREVIEWS_DIR, follow_symlink=True),
          name="previews")
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
