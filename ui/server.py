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
from pipeline import audio as audio_mod
from pipeline import (body_models, cloud, monitor, policy_store, preshow, setlist,
                      show_runner, shows, store, venue)
from pipeline.runner import Runner
from pipeline.stages.local_motion import build_stages

STATIC_DIR = Path(__file__).parent / "static"
PREVIEWS_DIR = DATA_DIR / "previews"
VET_SCRIPT = PROJECT_ROOT / "pipeline" / "vet_motion.py"

# Upload guard rails (audit HIGH: unbounded upload + double copy exhausts disk).
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB — a 4-min clip fits comfortably
MIN_FREE_DISK_BYTES = 3 * 1024 * 1024 * 1024  # keep 3 GB headroom for state writes
_UPLOAD_CHUNK = 1024 * 1024

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
            tb = traceback.format_exc()
            print(f"job-worker error on {job_id}:\n{tb}", file=sys.stderr)
            try:
                store.load_job(job_id).log(f"worker error:\n{tb}")
            except Exception:
                pass  # already logged to stderr above — never lose the trace silently
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


# How often the poll loop looks for cloud-blocked jobs due a re-check. Stages set
# meta["poll_after"] (via StageBlocked.retry_after_s) when they are waiting on a
# box job; re-queueing re-runs the stage, which re-reads the box status.json.
CLOUD_POLL_S = 30


def _poll_cloud_jobs_once(now: float | None = None) -> list[str]:
    """Re-queue every job whose current stage is blocked with an expired
    poll_after. Returns the re-queued job ids (for tests)."""
    now = time.time() if now is None else now
    requeued = []
    for job in store.list_jobs():
        cur = job.current_stage()
        if cur is None:
            continue
        st = job.stages[cur]
        if st.state != "blocked":
            continue
        poll_after = st.meta.get("poll_after")
        if not poll_after or now < poll_after:
            continue
        # bump so a slow worker doesn't collect duplicate re-queues every tick
        st.meta["poll_after"] = now + CLOUD_POLL_S
        job.save()
        _job_queue.put(job.id)
        requeued.append(job.id)
    return requeued


def _cloud_poll_loop() -> None:
    while True:
        time.sleep(CLOUD_POLL_S)
        try:
            _poll_cloud_jobs_once()
        except Exception:  # noqa: BLE001 — the poller must never die
            print(f"cloud-poll error:\n{traceback.format_exc()}", file=sys.stderr)


@app.on_event("startup")
def _start_worker() -> None:
    # A bad job.json or a seeding hiccup must NEVER prevent the worker thread from
    # starting (audit HIGH: one corrupt record bricked the whole app). Guard each.
    try:
        _reconcile_jobs()
    except Exception:
        print(f"startup: _reconcile_jobs failed:\n{traceback.format_exc()}",
              file=sys.stderr)
    try:
        shows.seed_initial_dances()
    except Exception:
        print(f"startup: seed_initial_dances failed:\n{traceback.format_exc()}",
              file=sys.stderr)
    threading.Thread(target=_worker_loop, name="job-worker", daemon=True).start()
    threading.Thread(target=_system_refresh_loop, name="system-monitor",
                     daemon=True).start()
    threading.Thread(target=_cloud_poll_loop, name="cloud-poll",
                     daemon=True).start()


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


def _create_job(name: str, src: Path, *, move: bool = False) -> store.Job:
    """Create a job from an input file: .csv = robot motion, else video.

    move=True relocates src into the job dir (rename) instead of copying — used
    for uploads so a big file isn't written to disk twice (audit HIGH)."""
    kind = "csv" if src.suffix.lower() == ".csv" else "video"
    size = src.stat().st_size
    job = store.new_job(name, input={"type": kind, "source": str(src)})
    dest = job.dir / ("input.csv" if kind == "csv" else "input.mp4")
    if move:
        shutil.move(str(src), dest)   # rename within data/ — no second full copy
    else:
        shutil.copyfile(src, dest)
    job.log(f"input {kind}: {src.name} ({size} bytes)")
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
    """Create a job from a browser-style file upload.

    Guards (audit HIGH): the client controls the filename → keep only its
    basename so '../../evil.sh' can't escape the dir; cap the size and check free
    disk while streaming so a multi-GB file can't fill the operator's laptop; and
    hand the temp file to the job by MOVE so it isn't copied to disk twice."""
    safe_name = Path(video.filename or "upload").name
    videos_dir = DATA_DIR / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(videos_dir).free
    if free < MIN_FREE_DISK_BYTES:
        raise HTTPException(507, "not enough free disk space to accept an upload")
    tmp = videos_dir / f"upload-{int(time.time())}-{safe_name}"
    written = 0
    try:
        with open(tmp, "wb") as f:
            while chunk := await video.read(_UPLOAD_CHUNK):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413, f"upload exceeds the {MAX_UPLOAD_BYTES // (1024**3)} GB "
                             "limit — trim the clip to the segment to learn")
                if written > free - MIN_FREE_DISK_BYTES:
                    raise HTTPException(507, "upload would fill the disk — aborted")
                f.write(chunk)
    except BaseException:
        tmp.unlink(missing_ok=True)  # never leave a partial/oversized file behind
        raise
    return _job_dict(_create_job(Path(safe_name).stem, tmp, move=True))


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


@app.post("/api/jobs/{job_id}/approve-train")
def approve_train(job_id: str) -> dict:
    """The human preview gate (architecture §5): training costs 2-3 GPU-hours, so
    it never starts until the operator has watched the MuJoCo preview and clicked
    Approve. Records who/when in the train stage meta and re-queues the job."""
    try:
        job = store.load_job(job_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such job: {job_id}")
    if job.stages["retarget"].state != "done":
        raise HTTPException(400, "nothing to approve yet — the motion preview is "
                                 "not ready (retarget stage incomplete)")
    st = job.stages["train"]
    if st.state == "done":
        raise HTTPException(400, "training already completed")
    if not st.meta.get("approved"):
        st.meta["approved"] = {"at": time.time(), "by": "operator"}
        job.save()
        job.log("train: APPROVED by operator (preview reviewed) — queueing")
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


# ---- system monitor (read-only box observability) --------------------------------
# Surfaces GPU load, training progress, and accrued GreenNode cost in the app so the
# user doesn't have to ask "is the GPU running / how's training / what's it cost".
# A background refresher keeps a cached snapshot so the endpoint returns instantly and
# a slow box never blocks a request.

_system_snapshot: dict = {}


def _system_refresh_loop() -> None:
    global _system_snapshot
    while True:
        try:
            _system_snapshot = monitor.snapshot()
        except Exception:  # snapshot() shouldn't raise, but never kill the thread
            pass
        time.sleep(20)


@app.get("/api/system")
def system_status() -> dict:
    """Latest cached box snapshot (GPU, training jobs, cost). Cheap to poll."""
    if not _system_snapshot:
        return monitor.snapshot()
    return _system_snapshot


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


@app.post("/api/dances/{dance_id}/policy")
def attach_policy(dance_id: str, payload: dict = Body(...)) -> dict:
    """Attach a trained policy to an existing dance (audit HIGH workflow gap).

    The register-first flow had no way to set policy_path after creation, stranding
    a trained policy before it could be exam-verified. Attaching a policy resets the
    dance's verification state (the exam ran on a different policy) — re-run the exam."""
    _load_dance_or_404(dance_id)
    policy = (payload.get("policy_path") or "").strip()
    if not policy:
        raise HTTPException(400, "policy_path is required")
    try:
        return _dance_dict(shows.attach_policy(dance_id, policy,
                                               notes=payload.get("notes")))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/library/export")
def library_export() -> FileResponse:
    """Download the whole dance library as one portable .tar.gz (audit HIGH:
    disaster recovery — a laptop failure otherwise loses weeks of training)."""
    from pipeline import library
    archive = library.export_library()
    return FileResponse(archive, media_type="application/gzip",
                        filename=archive.name)


@app.post("/api/library/import")
def library_import(payload: dict = Body(...)) -> dict:
    """Restore dances from an archive path on disk (round-trips with export)."""
    from pipeline import library
    src = (payload.get("archive_path") or "").strip()
    if not src:
        raise HTTPException(400, "archive_path is required")
    try:
        ids = library.import_library(Path(src).expanduser(),
                                     overwrite=bool(payload.get("overwrite")))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"imported": ids, "count": len(ids)}


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


# ---- music / audio per dance ----------------------------------------------------

@app.post("/api/dances/{dance_id}/audio")
def attach_dance_audio(dance_id: str, payload: dict = Body(...)) -> dict:
    """Give a dance its music track (source: an on-disk audio file, extraction from
    a video, or a generated placeholder). Aligns it to the prepped-motion timeline
    (music delayed past the standing intro) and muxes it onto the preview so the
    preview plays with sound. Presentation only — never touches show-ready status."""
    dance = _load_dance_or_404(dance_id)
    src_audio = payload.get("source_path")
    vid = payload.get("extract_from_video")
    kwargs: dict = {}
    if src_audio:
        kwargs["source_audio"] = Path(src_audio).expanduser()
    elif vid:
        kwargs["extract_from_video"] = Path(vid).expanduser()
    else:
        kwargs["placeholder_bpm"] = float(payload.get("bpm") or 118.0)
    if payload.get("window_start_s") is not None:
        kwargs["window_start_s"] = float(payload["window_start_s"])
    try:
        record = audio_mod.attach_audio_for_dance(dance, **kwargs)
        return _dance_dict(shows.set_audio(dance_id, record))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/dances/{dance_id}/audio")
def clear_dance_audio(dance_id: str) -> dict:
    _load_dance_or_404(dance_id)
    return _dance_dict(shows.set_audio(dance_id, None))


@app.get("/api/dances/{dance_id}/audio-file")
def dance_audio_file(dance_id: str) -> FileResponse:
    """Serve a dance's music track for the preview player."""
    dance = _load_dance_or_404(dance_id)
    if not dance.audio or not dance.audio.get("track"):
        raise HTTPException(404, "this dance has no music attached")
    track = Path(dance.audio["track"])
    if not track.is_absolute():
        track = PROJECT_ROOT / dance.audio["track"]
    if not track.is_file():
        raise HTTPException(404, "music file is missing on disk")
    return FileResponse(track)


# ---- set-lists: an ordered sequence of dances = one show ------------------------

def _setlist_dict(sl: setlist.SetList) -> dict:
    return setlist.resolve(sl, _dance_lookup)


def _dance_lookup(dance_id: str):
    try:
        return shows.load_dance(dance_id)
    except (FileNotFoundError, ValueError):
        return None


@app.get("/api/setlists")
def setlists() -> list[dict]:
    return [_setlist_dict(sl) for sl in setlist.list_setlists()]


@app.get("/api/setlists/{setlist_id}")
def setlist_detail(setlist_id: str) -> dict:
    try:
        return _setlist_dict(setlist.load_setlist(setlist_id))
    except FileNotFoundError:
        raise HTTPException(404, f"no such set-list: {setlist_id}")


@app.post("/api/setlists")
def create_setlist(payload: dict = Body(...)) -> dict:
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "set-list needs a name")
    sl = setlist.new_setlist(name, notes=payload.get("notes", ""))
    if payload.get("items"):
        try:
            sl = setlist.set_items(sl.id, payload["items"])
        except ValueError as e:
            raise HTTPException(400, str(e))
    return _setlist_dict(sl)


@app.post("/api/setlists/{setlist_id}")
def edit_setlist(setlist_id: str, payload: dict = Body(...)) -> dict:
    try:
        if "name" in payload or "notes" in payload:
            setlist.rename(setlist_id, payload.get("name") or setlist.load_setlist(setlist_id).name,
                           notes=payload.get("notes"))
        if "items" in payload:
            setlist.set_items(setlist_id, payload["items"])
        return _setlist_dict(setlist.load_setlist(setlist_id))
    except FileNotFoundError:
        raise HTTPException(404, f"no such set-list: {setlist_id}")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/setlists/{setlist_id}")
def remove_setlist(setlist_id: str) -> dict:
    if not setlist.delete_setlist(setlist_id):
        raise HTTPException(404, f"no such set-list: {setlist_id}")
    return {"deleted": setlist_id}


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
    mode = payload.get("mode", "live")
    try:
        return _show_dict(shows.new_show(dance, operator, mode=mode,
                                         setlist_id=payload.get("setlist_id")))
    except ValueError as e:
        raise HTTPException(400, str(e))


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


# ---- one-button live show: spawn the proven show_run.sh path --------------------
# Unlike the record-only deploy gate above, THIS actually launches the robot show
# (tools/show_run.sh -> deploy_runtime ground-run-legodom + music). All robot safety
# rests on the operator's hand-held damping remote — this G1 has no hardware e-stop.

# The operator must type this EXACT phrase to start a run. Typing it while physically
# holding the damping remote (surfaced in the UI as the big red "REMOTE = ONLY STOP")
# IS the explicit human confirmation CLAUDE.md requires before any deploy.
RUN_CONFIRMATION_PHRASE = "I AM PRESENT WITH THE DAMPING REMOTE"


@app.post("/api/shows/{dance_id}/run")
def run_show(dance_id: str, payload: dict = Body(...)) -> dict:
    """Launch a live show for a show-ready dance. Ordered guards, distinct codes."""
    # 1) dance exists AND is show-ready. A genuinely missing dance is a 404 (house
    #    style, _load_dance_or_404); a wrong-status dance is the 409 operational guard.
    dance = _load_dance_or_404(dance_id)
    if dance.status != "show-ready":
        raise HTTPException(409, f"dance '{dance.name}' is not show-ready "
                                 f"(status: {dance.status}) — it cannot be run")
    # 2) audio attached — a show plays music; refuse a silent dance.
    if not (dance.audio and dance.audio.get("track")):
        raise HTTPException(409, "dance has no music attached — attach a track first")
    # 3) robot reachable: PC2 must answer a single 1 s ping on the control net.
    if not show_runner.robot_reachable():
        raise HTTPException(409, f"robot PC2 ({show_runner.ROBOT_HOST}) is not "
                                 "reachable — check the control-net cable and power")
    # 4) single-run lock / unresolved open show (reuses shows.record_outcome's
    #    show.closed state via show_runner.why_blocked).
    busy = show_runner.why_blocked()
    if busy:
        raise HTTPException(409, busy)
    # 5) exact confirmation phrase — this + the physical remote is the deploy consent.
    if payload.get("confirmation") != RUN_CONFIRMATION_PHRASE:
        raise HTTPException(403, "confirmation phrase does not match — type it "
                                 "EXACTLY, with the damping remote in your hand")
    operator = (payload.get("operator") or "").strip()
    if not operator:
        raise HTTPException(400, "operator name is required")
    mode = payload.get("mode", "rehearsal")
    if mode not in ("rehearsal", "live"):
        raise HTTPException(400, "mode must be 'rehearsal' or 'live'")
    # stand-at-end is experimental + rehearsal-only; begin_run enforces the mode gate.
    exit_stand = bool(payload.get("exit_stand"))
    try:
        show = show_runner.begin_run(
            dance, operator=operator, mode=mode, exit_stand=exit_stand,
            audio_mode=payload.get("audio_mode") or "laptop",
            body=payload.get("body"))
    except show_runner.RunBusy as e:  # lost the check-and-spawn race
        raise HTTPException(409, str(e))
    return {"started": True, "show": _show_dict(show),
            "run": show_runner.current_status()}


@app.get("/api/shows/runs/current")
def current_run() -> dict:
    """Live run status for the app's status panel: running flag, show id, mode,
    coarse phase, and the last ~15 run.log lines. Cheap to poll."""
    return show_runner.current_status()


# ---- venues (dance-area registry; drives the vet excursion limit) ----------------

@app.get("/api/venues")
def venues() -> dict:
    return {"venues": [v.to_public() for v in venue.list_venues()],
            "active": venue.get_active_venue().to_public()}


@app.post("/api/venues")
def upsert_venue(payload: dict = Body(...)) -> dict:
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "venue needs a name")
    try:
        v = venue.add_or_update_venue(
            name, radius_m=float(payload["radius_m"]),
            margin_m=float(payload.get("margin_m", venue.DEFAULT_MARGIN_M)),
            notes=payload.get("notes", ""),
            make_active=bool(payload.get("make_active", False)))
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(400, f"bad venue: {e}")
    return v.to_public()


@app.post("/api/venues/active")
def choose_active_venue(payload: dict = Body(...)) -> dict:
    key = payload.get("key") or payload.get("name")
    if not key:
        raise HTTPException(400, "need a venue key or name")
    try:
        return venue.set_active_venue(key).to_public()
    except (ValueError, KeyError) as e:
        raise HTTPException(404, str(e))


# ---- pre-show checklist + show-phase ownership model -----------------------------

@app.post("/api/dances/{dance_id}/checklist")
def dance_checklist(dance_id: str, payload: dict = Body(default={})) -> dict:
    """Evaluate the pre-show checklist for a dance. Body: {acks: [confirm keys the
    operator ticked]}. Robot reachability is a live ping; the active venue feeds the
    venue-selected item. ready == all blocker items satisfied."""
    dance = _load_dance_or_404(dance_id)
    acks = set(payload.get("acks") or [])
    report = preshow.evaluate_checklist(
        dance, robot_ping=lambda: show_runner.robot_reachable(),
        venue_active=venue.get_active_venue(), acks=acks)
    report["confirm_keys"] = list(preshow.CONFIRM_KEYS)
    return report


@app.get("/api/show-phases")
def show_phases() -> dict:
    """Who controls the robot at each show phase (walk-on -> dance -> walk-off)."""
    return {"phases": preshow.make_show_phases()}


# ---- policy version store + rollback ---------------------------------------------

@app.get("/api/dances/{dance_id}/versions")
def policy_versions(dance_id: str) -> dict:
    _load_dance_or_404(dance_id)
    return {"versions": policy_store.list_versions(dance_id)}


@app.post("/api/dances/{dance_id}/rollback")
def rollback_policy(dance_id: str, payload: dict = Body(...)) -> dict:
    """Restore a stored policy version's files into the dance's policy dir. This RESETS
    verification (attach_policy demotes to draft) — the operator re-runs the sim exam to
    re-promote, so a rollback can never silently reinstate show-ready without the gate."""
    version_id = payload.get("version_id")
    if not version_id:
        raise HTTPException(400, "need version_id")
    dance = _load_dance_or_404(dance_id)
    if not dance.policy_path:
        raise HTTPException(400, "dance has no policy directory to roll back into")
    try:
        files = policy_store.rollback_files(dance_id, version_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(404, f"no such version {version_id}: {e}")
    import shutil
    pol_dir = shows._abs(dance.policy_path).parent
    for fname, src in files.items():
        shutil.copyfile(src, pol_dir / fname)
    dance = shows.attach_policy(dance_id, dance.policy_path,
                                notes=f"rolled back to policy version {version_id[:12]}")
    return {"dance": _dance_dict(dance), "restored": list(files),
            "note": "files restored; status reset to draft — re-run the sim exam to re-promote"}


# ---- set-list run plan (show-time audio cues + all-show-ready gating) -------------

@app.get("/api/setlists/{setlist_id}/run-plan")
def setlist_run_plan(setlist_id: str) -> dict:
    try:
        sl = setlist.load_setlist(setlist_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no such set-list: {setlist_id}")
    run = setlist.get_or_create_run(sl)
    plan = setlist.setlist_run_plan(sl, _dance_lookup, run)
    return {"plan": plan, "runnable": setlist.plan_runnable(plan),
            "blockers": setlist.plan_blockers(plan),
            "next_index": setlist.next_index(run)}


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
