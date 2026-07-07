"""Tests for surfacing a tripped fall detector through the show runner + app API.

deploy_runtime's _check_fall raises RuntimeError("FALL DETECTED ...") when torso
uprightness drops below FALL_UPRIGHT_MIN; the mode's abort path prints that as
"STOP: FALL DETECTED ... -> damping" to the run log show_runner streams. These
tests verify current_status() flags fall_detected + a 'fall' phase, and that the
field passes through the /api/shows/runs/current endpoint verbatim.

Mirrors tests/test_show_run.py: the subprocess spawn is ALWAYS monkeypatched, and
current_status() reads a fake run log (no real robot / show_run.sh is ever touched).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import exam_verdict as ev
from pipeline import show_runner

PHRASE = "I AM PRESENT WITH THE DAMPING REMOTE"

# The single line the runtime prints on a fall: the mode's except wraps _check_fall's
# RuntimeError as "STOP: <msg> -> damping", and the msg itself starts "FALL DETECTED".
FALL_LINE = ("STOP: FALL DETECTED at tick 512: torso 73 deg from vertical "
             "(uprightness 0.31 < 0.35) -> damping + onboard handoff -> damping")

FALL_LINES = [
    "SHOW RUN: dance=X audio=laptop latency_comp=0.0s",
    "at default — starting leg-odometry policy. Keep tension on the tether;",
    FALL_LINE,
    "damping engaged; restoring onboard 'ai'",
]

# A clean run that ends on the normal ramp-to-damping exit (no fall).
CLEAN_LINES = [
    "SHOW RUN: dance=X audio=laptop latency_comp=0.0s",
    "at default — starting leg-odometry policy. Keep tension on the tether;",
    "segment done; ramp to damping",
]

# A non-fall abort (e.g. the operator damped) — STOP but no FALL.
PLAIN_STOP_LINES = [
    "SHOW RUN: dance=X audio=laptop latency_comp=0.0s",
    "at default — starting leg-odometry policy. Keep tension on the tether;",
    "STOP: KeyboardInterrupt -> damping",
]


class FakeProc:
    """Minimal Popen stand-in: poll() returns None while 'running', else the rc."""
    def __init__(self, rc=None):
        self._rc = rc
        self.pid = 4242

    def poll(self):
        return self._rc


def _set_current(monkeypatch, tmp_path, lines, rc=None):
    """Point show_runner._current at a fake run whose run.log holds `lines`.

    rc=None => the fake process is still 'running'; an int => it has exited. Returns
    the log path (mimics how the real spawn writes stdout to run.log)."""
    log = tmp_path / "run.log"
    log.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(show_runner, "_current", {
        "show_id": "20260707-fall", "dance_id": "dance-x", "mode": "live",
        "proc": FakeProc(rc=rc), "log_path": str(log), "started_at": 123.0,
    })
    return log


# ---- current_status(): fall log -> fall_detected + 'fall' phase -----------------

def test_fall_log_sets_fall_detected_and_phase(monkeypatch, tmp_path):
    _set_current(monkeypatch, tmp_path, FALL_LINES, rc=1)  # exited after the fall
    st = show_runner.current_status()
    assert st["fall_detected"] is True
    assert st["phase"] == "fall"


def test_fall_detected_while_still_running(monkeypatch, tmp_path):
    # The raise -> damp -> finalize sequence may still be in flight when we poll.
    _set_current(monkeypatch, tmp_path, FALL_LINES, rc=None)
    st = show_runner.current_status()
    assert st["running"] is True
    assert st["fall_detected"] is True
    assert st["phase"] == "fall"


def test_stop_containing_fall_without_literal_marker(monkeypatch, tmp_path):
    # Defensive: a STOP line that mentions FALL but not the exact "FALL DETECTED".
    lines = ["at default — starting leg-odometry policy.",
             "STOP: robot FALL past uprightness floor -> damping"]
    _set_current(monkeypatch, tmp_path, lines, rc=1)
    st = show_runner.current_status()
    assert st["fall_detected"] is True
    assert st["phase"] == "fall"


# ---- current_status(): clean / non-fall logs -> fall_detected False -------------

def test_clean_log_has_no_fall(monkeypatch, tmp_path):
    _set_current(monkeypatch, tmp_path, CLEAN_LINES, rc=0)
    st = show_runner.current_status()
    assert st["fall_detected"] is False
    assert st["phase"] == "ramp-to-damping"


def test_plain_stop_is_not_a_fall(monkeypatch, tmp_path):
    _set_current(monkeypatch, tmp_path, PLAIN_STOP_LINES, rc=1)
    st = show_runner.current_status()
    assert st["fall_detected"] is False
    assert st["phase"] == "stopped"


def test_helper_matches_both_markers():
    assert show_runner._log_shows_fall(FALL_LINE) is True
    assert show_runner._log_shows_fall("STOP: FALL past floor -> damping") is True
    assert show_runner._log_shows_fall("segment done; ramp to damping") is False
    assert show_runner._log_shows_fall("STOP: KeyboardInterrupt -> damping") is False


# ---- API passthrough: /api/shows/runs/current carries fall_detected -------------

def _show_ready_with_audio(shows_mod, name="Faller"):
    """A dance driven to show-ready through the real gate, with music attached
    (verbatim mirror of the tests/test_show_run.py helper)."""
    (shows_mod.PROJECT_ROOT / "policy.onnx").write_bytes(b"fake-policy-bytes")
    (shows_mod.PROJECT_ROOT / "motion.csv").write_text("0,0,0.79\n")
    d = shows_mod.new_dance(name, duration_s=30.0, policy_path="policy.onnx",
                            motion_csv="motion.csv")
    sha = ev.full_sha256(shows_mod.PROJECT_ROOT / "policy.onnx")
    for _ in range(3):
        shows_mod.record_sim_run(shows_mod.load_dance(d.id), True, policy_sha256=sha)
    shows_mod.promote(shows_mod.load_dance(d.id), "show-ready")
    return shows_mod.set_audio(d.id, {"track": "data/audio/song.wav"})


def _install_spawn(monkeypatch, lines, rc=None):
    """Replace spawn_show_process with a fake that writes `lines` to the run log."""
    def _spawn(cmd, env, log_path):
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n")
        return FakeProc(rc=rc)

    monkeypatch.setattr(show_runner, "spawn_show_process", _spawn)


@pytest.fixture
def run_env(dances_env, client, monkeypatch):
    """Isolated shows library + TestClient, robot faked reachable, no run in flight."""
    shows_mod, _ = dances_env
    c, server = client
    monkeypatch.setattr(show_runner, "_current", None)
    monkeypatch.setattr(show_runner, "robot_reachable", lambda *a, **k: True)
    return c, shows_mod


def test_api_current_surfaces_fall(run_env, monkeypatch):
    c, shows_mod = run_env
    d = _show_ready_with_audio(shows_mod, "Toppler")
    _install_spawn(monkeypatch, FALL_LINES, rc=1)  # spawns, writes the fall log, exits
    r = c.post(f"/api/shows/{d.id}/run",
               json={"operator": "alois", "mode": "live", "confirmation": PHRASE})
    assert r.status_code == 200, r.text
    st = c.get("/api/shows/runs/current").json()
    assert st["fall_detected"] is True
    assert st["phase"] == "fall"


def test_api_current_clean_run_no_fall(run_env, monkeypatch):
    c, shows_mod = run_env
    d = _show_ready_with_audio(shows_mod, "Steady")
    _install_spawn(monkeypatch, CLEAN_LINES, rc=None)  # still running, clean
    r = c.post(f"/api/shows/{d.id}/run",
               json={"operator": "alois", "mode": "live", "confirmation": PHRASE})
    assert r.status_code == 200, r.text
    st = c.get("/api/shows/runs/current").json()
    assert st["fall_detected"] is False
    assert st["phase"] != "fall"
