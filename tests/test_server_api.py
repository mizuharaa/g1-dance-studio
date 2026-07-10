"""API surface of ui/server.py via TestClient — runner stubbed, store isolated."""
import json

import numpy as np
import pytest

from .conftest import HAVE_MODEL, WORKTREE, make_motion


def _mk_csv(tmp_path, name="dance.csv"):
    p = tmp_path / name
    np.savetxt(p, make_motion(), delimiter=",", fmt="%.6f")
    return p


def test_health(client):
    c, _ = client
    got = c.get("/api/health").json()
    assert got["ok"] is True
    assert got["stage_order"][0] == "extract"


def test_react_frontend_and_spa_fallback(client):
    c, _ = client
    root = c.get("/")
    assert root.status_code == 200
    assert "G1 Operator Console" in root.text
    assert c.get("/operator/audit").status_code == 200
    # The SPA fallback must not turn a missing API route into an HTML 200.
    assert c.get("/api/not-a-real-endpoint").status_code == 404


def test_jobs_empty_initially(client):
    c, _ = client
    assert c.get("/api/jobs").json() == []


def test_create_job_from_csv_path(client, tmp_path):
    c, _ = client
    src = _mk_csv(tmp_path)
    got = c.post("/api/jobs", json={"input_path": str(src)}).json()
    assert got["name"] == "dance"
    assert got["input"]["type"] == "csv"
    assert got["current_stage"] == "extract"
    # input copied into the job dir with the canonical name
    detail = c.get(f"/api/jobs/{got['id']}").json()
    assert any("input csv" in line for line in detail["log_tail"])


def test_create_job_missing_file_400(client):
    c, _ = client
    r = c.post("/api/jobs", json={"input_path": "/nonexistent/file.mp4"})
    assert r.status_code == 400
    assert "not found" in r.json()["detail"]


def test_job_detail_404(client):
    c, _ = client
    assert c.get("/api/jobs/nope").status_code == 404


def test_upload_creates_video_job(client):
    c, _ = client
    r = c.post("/api/jobs/upload",
               files={"video": ("mydance.mov", b"fake-bytes", "video/quicktime")})
    got = r.json()
    assert got["input"]["type"] == "video"
    assert got["name"] == "mydance"


def test_retry_rejects_non_failed_stage(client, tmp_path):
    c, _ = client
    job = c.post("/api/jobs",
                 json={"input_path": str(_mk_csv(tmp_path))}).json()
    r = c.post(f"/api/jobs/{job['id']}/retry")
    assert r.status_code == 400
    assert "pending" in r.json()["detail"]


def test_retry_requeues_failed_stage(client, tmp_path, jobs_env):
    c, _ = client
    store, _ = jobs_env
    created = c.post("/api/jobs",
                     json={"input_path": str(_mk_csv(tmp_path))}).json()
    j = store.load_job(created["id"])
    j.stages["extract"].state = "failed"
    j.stages["extract"].message = "boom"
    j.save()
    got = c.post(f"/api/jobs/{created['id']}/retry").json()
    assert got["stages"]["extract"]["state"] == "pending"
    assert got["stages"]["extract"]["message"] == ""


# ---- the deploy gate: the load-bearing safety test --------------------------------

def test_deploy_refused_without_typed_phrase(client, tmp_path):
    c, _ = client
    job = c.post("/api/jobs",
                 json={"input_path": str(_mk_csv(tmp_path))}).json()
    for bad in ({}, {"confirm_phrase": ""}, {"confirm_phrase": "deploy"},
                {"confirm_phrase": "DEPLOY "}, {"confirm": "DEPLOY"}):
        r = c.post(f"/api/jobs/{job['id']}/deploy", json=bad)
        assert r.status_code == 400, f"accepted: {bad!r}"


def test_deploy_with_phrase_records_only(client, tmp_path, jobs_env):
    c, _ = client
    store, _ = jobs_env
    job = c.post("/api/jobs",
                 json={"input_path": str(_mk_csv(tmp_path))}).json()
    got = c.post(f"/api/jobs/{job['id']}/deploy",
                 json={"confirm_phrase": "DEPLOY"}).json()
    assert got == {"recorded": True, "deployed": False, "note": got["note"]}
    assert "Nothing was sent to the robot" in got["note"]
    reqs = json.loads(
        (store.load_job(job["id"]).dir / "deploy_requests.json").read_text())
    assert len(reqs) == 1 and reqs[0]["job"] == job["id"]


# ---- restart recovery -------------------------------------------------------------

def test_reconcile_requeues_interrupted_stage(client, tmp_path, jobs_env,
                                              monkeypatch):
    c, server = client
    store, _ = jobs_env
    created = c.post("/api/jobs",
                     json={"input_path": str(_mk_csv(tmp_path))}).json()
    j = store.load_job(created["id"])
    j.stages["extract"].state = "running"      # simulate death mid-stage
    j.save()
    # capture queue puts (the live worker thread would drain the real queue)
    queued: list[str] = []
    monkeypatch.setattr(server._job_queue, "put", queued.append)
    server._reconcile_jobs()
    j2 = store.load_job(created["id"])
    assert j2.stages["extract"].state == "pending"
    assert "re-queued" in j2.stages["extract"].message
    assert queued == [created["id"]]


def test_reconcile_leaves_failed_jobs_alone(client, tmp_path, jobs_env,
                                            monkeypatch):
    c, server = client
    store, _ = jobs_env
    created = c.post("/api/jobs",
                     json={"input_path": str(_mk_csv(tmp_path))}).json()
    j = store.load_job(created["id"])
    j.stages["extract"].state = "failed"
    j.save()
    queued: list[str] = []
    monkeypatch.setattr(server._job_queue, "put", queued.append)
    server._reconcile_jobs()
    assert store.load_job(created["id"]).stages["extract"].state == "failed"
    assert queued == []


# ---- vet endpoint guards ----------------------------------------------------------

def test_vet_rejects_paths_outside_project(client):
    c, _ = client
    assert c.get("/api/vet", params={"csv": "../../../etc/passwd"}
                 ).status_code == 400


def test_vet_rejects_non_csv(client):
    c, _ = client
    assert c.get("/api/vet", params={"csv": "pipeline/vet_motion.py"}
                 ).status_code == 400


def test_vet_404_for_missing_csv(client):
    c, _ = client
    assert c.get("/api/vet", params={"csv": "data/does_not_exist.csv"}
                 ).status_code == 404


@pytest.mark.model
@pytest.mark.skipif(not HAVE_MODEL, reason="G1 model not present")
def test_vet_endpoint_runs_and_caches(client):
    c, server = client
    rel = "data/_pytest_vet.csv"
    p = WORKTREE / rel
    np.savetxt(p, make_motion(), delimiter=",", fmt="%.6f")
    try:
        got = c.get("/api/vet", params={"csv": rel}).json()
        assert got["pass"] is True
        key_hits = [k for k in server._vet_cache if k[0] == str(p.resolve())]
        assert key_hits, "vet result was not cached"
    finally:
        p.unlink(missing_ok=True)


# ---- cloud endpoints (no network: transport left unconfigured) ---------------------

def test_cloud_config_validates_transport(client):
    c, _ = client
    r = c.post("/api/cloud/config", json={"transport": "carrier-pigeon"})
    assert r.status_code == 400


def test_cloud_endpoints_mask_secrets(client, tmp_path, monkeypatch):
    from pipeline import cloud as cloud_mod
    monkeypatch.setattr(cloud_mod, "CONFIG_PATH", tmp_path / "cloud.json")
    c, _ = client
    c.post("/api/cloud/config",
           json={"transport": "jupyter",
                 "jupyter": {"url": "http://x", "token": "supersecret"}})
    got = c.get("/api/cloud").json()
    assert got["config"]["jupyter"]["token"] == "•set•"
    assert "supersecret" not in json.dumps(got)


# ---- previews static mount ---------------------------------------------------------

def test_previews_range_requests(client):
    c, server = client
    f = server.PREVIEWS_DIR / "_pytest_clip.mp4"
    f.write_bytes(bytes(range(256)) * 4)
    try:
        full = c.get("/previews/_pytest_clip.mp4")
        assert full.status_code == 200
        part = c.get("/previews/_pytest_clip.mp4",
                     headers={"Range": "bytes=0-99"})
        assert part.status_code == 206
        assert len(part.content) == 100
        assert part.headers["content-range"].startswith("bytes 0-99/")
    finally:
        f.unlink(missing_ok=True)
