"""Tests for the one-button live-show runner (POST /api/shows/{id}/run + status).

The subprocess spawn is ALWAYS monkeypatched — no test ever launches the real
tools/show_run.sh (which would contact the robot). The run_env fixture installs a
spawn guard that raises if a guard-rejection path ever reaches the spawn.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import exam_verdict as ev

PHRASE = "I AM PRESENT WITH THE DAMPING REMOTE"


class FakeProc:
    """Minimal Popen stand-in: poll() returns None while 'running', else the rc."""
    def __init__(self, rc=None):
        self._rc = rc
        self.pid = 12345

    def poll(self):
        return self._rc


def _install_spawn(show_runner, monkeypatch, lines, rc=None):
    """Replace spawn_show_process with a fake that writes `lines` to the run log and
    returns a FakeProc (rc=None => still running). Returns a list capturing envs."""
    envs: list[dict] = []

    def _spawn(cmd, env, log_path):
        envs.append(env)
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n")
        return FakeProc(rc=rc)

    monkeypatch.setattr(show_runner, "spawn_show_process", _spawn)
    return envs


def _show_ready_with_audio(shows_mod, name="Thriller"):
    """A dance driven to show-ready through the real gate, with music attached."""
    (shows_mod.PROJECT_ROOT / "policy.onnx").write_bytes(b"fake-policy-bytes")
    (shows_mod.PROJECT_ROOT / "motion.csv").write_text("0,0,0.79\n")
    d = shows_mod.new_dance(name, duration_s=30.0, policy_path="policy.onnx",
                            motion_csv="motion.csv")
    sha = ev.full_sha256(shows_mod.PROJECT_ROOT / "policy.onnx")
    for _ in range(3):
        shows_mod.record_sim_run(shows_mod.load_dance(d.id), True, policy_sha256=sha)
    shows_mod.promote(shows_mod.load_dance(d.id), "show-ready")
    return shows_mod.set_audio(d.id, {"track": "data/audio/song.wav"})


@pytest.fixture
def run_env(dances_env, client, monkeypatch):
    """Isolated shows library + TestClient, robot faked reachable, spawn forbidden by
    default (individual tests opt into a fake spawn)."""
    shows_mod, _ = dances_env
    c, server = client
    from pipeline import show_runner
    monkeypatch.setattr(show_runner, "_current", None)
    monkeypatch.setattr(show_runner, "robot_reachable", lambda *a, **k: True)

    def _forbid(*a, **k):
        raise AssertionError("the real show_run.sh must never be spawned in tests")
    monkeypatch.setattr(show_runner, "spawn_show_process", _forbid)
    return c, server, shows_mod, show_runner


PILOT_LINES = [
    "SHOW RUN: dance=X audio=laptop latency_comp=0.0s",
    "GROUND-RUN-LEGODOM: stage-1 firm move-to-default (4s)+hold, then policy",
    "at default — starting leg-odometry policy. Keep tension on the tether;",
]


# ---- guard: dance must be show-ready --------------------------------------------

def test_run_rejects_non_show_ready(run_env):
    c, _, shows_mod, _ = run_env
    d = shows_mod.new_dance("Draftling", duration_s=10.0)  # status draft
    r = c.post(f"/api/shows/{d.id}/run",
               json={"operator": "alois", "mode": "rehearsal", "confirmation": PHRASE})
    assert r.status_code == 409
    assert "show-ready" in r.json()["detail"]


def test_run_missing_dance_404(run_env):
    c, _, _, _ = run_env
    r = c.post("/api/shows/nope/run",
               json={"operator": "alois", "mode": "rehearsal", "confirmation": PHRASE})
    assert r.status_code == 404


# ---- guard: audio must be attached ----------------------------------------------

def test_run_rejects_without_audio(run_env):
    c, _, shows_mod, _ = run_env
    d = _show_ready_with_audio(shows_mod, "Silent")
    shows_mod.set_audio(d.id, None)  # strip the music
    r = c.post(f"/api/shows/{d.id}/run",
               json={"operator": "alois", "mode": "rehearsal", "confirmation": PHRASE})
    assert r.status_code == 409
    assert "music" in r.json()["detail"]


# ---- guard: robot must be reachable ---------------------------------------------

def test_run_rejects_unreachable_robot(run_env, monkeypatch):
    c, _, shows_mod, show_runner = run_env
    d = _show_ready_with_audio(shows_mod, "Unreachable")
    monkeypatch.setattr(show_runner, "robot_reachable", lambda *a, **k: False)
    r = c.post(f"/api/shows/{d.id}/run",
               json={"operator": "alois", "mode": "rehearsal", "confirmation": PHRASE})
    assert r.status_code == 409
    assert "reachable" in r.json()["detail"].lower()


# ---- guard: exact confirmation phrase (403) -------------------------------------

def test_run_rejects_wrong_phrase(run_env):
    c, _, shows_mod, _ = run_env
    d = _show_ready_with_audio(shows_mod, "Phrasey")
    for bad in ({}, {"confirmation": ""}, {"confirmation": PHRASE.lower()},
                {"confirmation": PHRASE + " "}, {"confirmation": " " + PHRASE}):
        body = {"operator": "alois", "mode": "rehearsal", **bad}
        r = c.post(f"/api/shows/{d.id}/run", json=body)
        assert r.status_code == 403, f"accepted a bad phrase: {bad!r}"


# ---- happy path -----------------------------------------------------------------

def test_run_happy_path_creates_show_and_status(run_env, monkeypatch):
    c, server, shows_mod, show_runner = run_env
    d = _show_ready_with_audio(shows_mod, "Thriller")
    envs = _install_spawn(show_runner, monkeypatch, PILOT_LINES, rc=None)

    r = c.post(f"/api/shows/{d.id}/run",
               json={"operator": "alois", "mode": "rehearsal", "confirmation": PHRASE})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["started"] is True
    show_id = body["show"]["id"]

    # a real Show record was created for this run
    show = shows_mod.load_show(show_id)
    assert show.mode == "rehearsal" and show.dance_id == d.id and not show.closed

    # the spawn env carried the fixed show knobs
    assert len(envs) == 1
    env = envs[0]
    assert env["CONFIRMED_BY_HUMAN"] == "alois"
    assert env["AUDIO_MODE"] == "laptop"
    assert env["AUDIO_LATENCY_COMP"] == "0.0"
    assert env["ARM_ACTION_CAP_SCALE"] == "2.2"
    assert env["DANCE_ID"] == d.id
    assert "EXIT_MODE" not in env  # ramp-to-damping default (exit_stand not set)

    # status endpoint reflects the fake log + liveness + derived phase
    st = c.get("/api/shows/runs/current").json()
    assert st["running"] is True
    assert st["show_id"] == show_id
    assert st["mode"] == "rehearsal"
    assert st["phase"] == "performing"
    assert any("starting leg-odometry policy" in ln for ln in st["last_lines"])


def test_status_idle_when_no_run(run_env):
    c, _, _, _ = run_env
    st = c.get("/api/shows/runs/current").json()
    assert st == {"running": False, "show_id": None, "mode": None,
                  "phase": "idle", "last_lines": [], "dance_id": None,
                  "started_at": None}


# ---- stand-at-end is rehearsal-only + experimental ------------------------------

def test_exit_stand_sets_env_only_in_rehearsal(run_env, monkeypatch):
    c, _, shows_mod, show_runner = run_env
    d = _show_ready_with_audio(shows_mod, "Stander")
    envs = _install_spawn(show_runner, monkeypatch, PILOT_LINES, rc=None)
    c.post(f"/api/shows/{d.id}/run",
           json={"operator": "alois", "mode": "rehearsal", "exit_stand": True,
                 "confirmation": PHRASE})
    assert envs[-1]["EXIT_MODE"] == "stand"


def test_exit_stand_ignored_in_live(run_env, monkeypatch):
    c, _, shows_mod, show_runner = run_env
    d = _show_ready_with_audio(shows_mod, "LiveNoStand")
    envs = _install_spawn(show_runner, monkeypatch, PILOT_LINES, rc=None)
    c.post(f"/api/shows/{d.id}/run",
           json={"operator": "alois", "mode": "live", "exit_stand": True,
                 "confirmation": PHRASE})
    assert "EXIT_MODE" not in envs[-1]  # never stand-exit a live show


# ---- single-run lock ------------------------------------------------------------

def test_run_lock_rejects_second_while_running(run_env, monkeypatch):
    c, _, shows_mod, show_runner = run_env
    d = _show_ready_with_audio(shows_mod, "Busy")
    _install_spawn(show_runner, monkeypatch, PILOT_LINES, rc=None)  # stays running
    assert c.post(f"/api/shows/{d.id}/run",
                  json={"operator": "alois", "mode": "rehearsal",
                        "confirmation": PHRASE}).status_code == 200
    r2 = c.post(f"/api/shows/{d.id}/run",
                json={"operator": "alois", "mode": "rehearsal", "confirmation": PHRASE})
    assert r2.status_code == 409
    assert "already running" in r2.json()["detail"]


# ---- an unresolved open show blocks the next run --------------------------------

def test_outcome_required_before_next_run(run_env, monkeypatch):
    c, _, shows_mod, show_runner = run_env
    d = _show_ready_with_audio(shows_mod, "Resolver")
    _install_spawn(show_runner, monkeypatch, PILOT_LINES, rc=0)  # exits immediately

    r1 = c.post(f"/api/shows/{d.id}/run",
                json={"operator": "alois", "mode": "rehearsal", "confirmation": PHRASE})
    assert r1.status_code == 200
    show_id = r1.json()["show"]["id"]

    # process exited but the show has no outcome -> next run blocked
    r2 = c.post(f"/api/shows/{d.id}/run",
                json={"operator": "alois", "mode": "rehearsal", "confirmation": PHRASE})
    assert r2.status_code == 409
    assert "outcome" in r2.json()["detail"]

    # record the outcome via the EXISTING endpoint (shows.record_outcome)
    assert c.post(f"/api/shows/{show_id}/outcome",
                  json={"result": "clean"}).status_code == 200

    # now a fresh run is allowed again
    r3 = c.post(f"/api/shows/{d.id}/run",
                json={"operator": "alois", "mode": "rehearsal", "confirmation": PHRASE})
    assert r3.status_code == 200


# ---- emergency software E-STOP (POST /api/safety/estop) -------------------------
# The pkill of stray processes is ALWAYS monkeypatched — a test must never signal real
# deploy_runtime / show_run.sh processes on the host.

def test_estop_no_run_is_honest(run_env, monkeypatch):
    """With nothing to stop, the E-STOP does not lie: it reports stopped=False and points
    the operator at the remote / power switch (the only stop for remote/onboard motion)."""
    c, _, _, show_runner = run_env
    monkeypatch.setattr(show_runner, "_pkill_deploy", lambda: [])
    r = c.post("/api/safety/estop")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stopped"] is False
    assert body["was_running"] is False
    assert "remote" in body["detail"].lower()


def test_estop_damps_running_show_with_sigterm(run_env, monkeypatch):
    """A running app-launched show is SIGTERMed (so deploy_runtime damps) — never SIGKILLed
    (which would skip damping and leave the motors energised)."""
    import os
    import signal
    c, _, shows_mod, show_runner = run_env
    d = _show_ready_with_audio(shows_mod, "Estoppable")
    _install_spawn(show_runner, monkeypatch, PILOT_LINES, rc=None)  # stays running
    assert c.post(f"/api/shows/{d.id}/run",
                  json={"operator": "alois", "mode": "rehearsal",
                        "confirmation": PHRASE}).status_code == 200
    sent = {}
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: sent.update(pgid=pgid, sig=sig))
    monkeypatch.setattr(show_runner, "_pkill_deploy", lambda: [])
    body = c.post("/api/safety/estop").json()
    assert body["stopped"] is True
    assert body["tracked_stopped"] is True
    assert body["was_running"] is True
    assert sent["sig"] == signal.SIGTERM   # NOT SIGKILL


def test_estop_signals_stray_deploy_when_untracked(run_env, monkeypatch):
    """Even with no tracked run, the E-STOP catches a stray deploy process (one launched
    outside the app, or a leftover after a crash)."""
    c, _, _, show_runner = run_env  # _current is None in the fixture
    monkeypatch.setattr(show_runner, "_pkill_deploy", lambda: ["deploy_runtime"])
    body = c.post("/api/safety/estop").json()
    assert body["stopped"] is True
    assert body["strays_signaled"] == ["deploy_runtime"]
    assert body["was_running"] is False


def test_safety_status_reports_reachability_and_run(run_env):
    """The Safety panel's snapshot endpoint returns robot reachability + live run status."""
    c, _, _, _ = run_env  # robot_reachable faked True by the fixture
    body = c.get("/api/safety/status").json()
    assert body["robot_reachable"] is True
    assert body["run"]["running"] is False
    assert body["run"]["phase"] == "idle"
