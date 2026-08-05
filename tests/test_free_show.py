"""Tests for the gated unsigned "free" rehearsal config and signed show path.

The unsigned standtail configuration is forbidden in live mode. Its rehearsal-only
escape hatch requires an explicit process opt-in and an internally hash-valid bundle.
Normal runs always pass the exact promotion-recorded policy/meta/NPZ arguments; there
is no default-artifact fallback.

The subprocess spawn is ALWAYS monkeypatched — no test ever launches the real
tools/show_run.sh (which would contact the robot).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import artifacts, exam_verdict as ev

PHRASE = "I AM PRESENT WITH THE DAMPING REMOTE"
# free knobs that must NEVER leak onto the proven default path; deleted from the test
# process env so their absence is a real signal, not a fluke of the host environment.
FREE_KEYS = ("GROUND_LEG_KP_SCALE", "EXIT_MODE", "MAX_SECS", "SHOW_VIDEO", "SHOW_DISPLAY")


class FakeProc:
    """Minimal Popen stand-in: poll() returns None while 'running', else the rc."""
    def __init__(self, rc=None):
        self._rc = rc
        self.pid = 12345

    def poll(self):
        return self._rc


def _install_spawn(show_runner, monkeypatch, lines, rc=None):
    """Replace spawn_show_process with a fake that writes `lines` to the run log and
    returns a FakeProc (rc=None => still running). Returns a list capturing (cmd, env)
    tuples so a test can assert on BOTH the CLI args and the env."""
    calls: list[tuple[list[str], dict]] = []

    def _spawn(cmd, env, log_path):
        calls.append((cmd, env))
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n")
        return FakeProc(rc=rc)

    monkeypatch.setattr(show_runner, "spawn_show_process", _spawn)
    return calls


def _show_ready_with_audio(shows_mod, name="Thriller"):
    """A dance promoted with a complete, hash-pinned legacy bundle and music."""
    (shows_mod.PROJECT_ROOT / "policy.onnx").write_bytes(b"fake-policy-bytes")
    (shows_mod.PROJECT_ROOT / "policy_meta.json").write_text("{}")
    (shows_mod.PROJECT_ROOT / "dance_deploy.npz").write_bytes(b"fake-motion-npz")
    (shows_mod.PROJECT_ROOT / "motion.csv").write_text("0,0,0.79\n")
    d = shows_mod.new_dance(name, duration_s=30.0, policy_path="policy.onnx",
                            motion_csv="motion.csv")
    sha = ev.full_sha256(shows_mod.PROJECT_ROOT / "policy.onnx")
    for _ in range(3):
        shows_mod.record_sim_run(shows_mod.load_dance(d.id), True, policy_sha256=sha)
    shows_mod.promote(shows_mod.load_dance(d.id), "show-ready")
    return shows_mod.set_audio(d.id, {"track": "data/audio/song.wav"})


def _write_free_bundle(shows_mod, show_runner):
    """Create the explicitly opted-in trial bundle under the isolated project root."""
    policy_dir = shows_mod.PROJECT_ROOT / show_runner.FREE_POLICY_DIR
    policy_dir.mkdir(parents=True)
    policy = policy_dir / "policy.onnx"
    meta = policy_dir / "policy_meta.json"
    npz = policy_dir / "thriller_deploy.npz"
    policy.write_bytes(b"free-policy")
    meta.write_text("{}")
    npz.write_bytes(b"free-motion-npz")
    artifacts.write_manifest(policy_dir / "bundle.json", {
        "policy": {
            "onnx": artifacts.file_entry(policy, policy_dir),
            "meta": artifacts.file_entry(meta, policy_dir),
        },
        "motion": {
            "tempo_npz": {"100": artifacts.file_entry(npz, policy_dir)},
        },
    })
    return policy, meta, npz


@pytest.fixture
def run_env(dances_env, client, monkeypatch):
    """Isolated shows library + TestClient, robot faked reachable, spawn forbidden by
    default (individual tests opt into a fake spawn). Free knobs cleared from the host
    env so 'absent on the default path' is meaningful."""
    shows_mod, _ = dances_env
    c, server = client
    from pipeline import show_runner
    monkeypatch.setattr(show_runner, "_current", None)
    monkeypatch.setattr(show_runner, "robot_reachable", lambda *a, **k: True)
    monkeypatch.setattr(server.venue, "get_active_venue",
                        lambda: SimpleNamespace(name="Test venue"))
    monkeypatch.delenv("G1_ALLOW_UNSIGNED_FREE", raising=False)
    for k in FREE_KEYS:
        monkeypatch.delenv(k, raising=False)

    def _forbid(*a, **k):
        raise AssertionError("the real show_run.sh must never be spawned in tests")
    monkeypatch.setattr(show_runner, "spawn_show_process", _forbid)
    return c, server, shows_mod, show_runner


PILOT_LINES = [
    "SHOW RUN: dance=X audio=laptop latency_comp=0.0s",
    "GROUND-RUN-LEGODOM: stage-1 firm move-to-default (4s)+hold, then policy",
    "at default — starting leg-odometry policy. Keep tension on the tether;",
]


# ---- free run: forbidden live; explicitly opted-in and hashed in rehearsal -------

def test_free_live_is_refused_without_spawning(run_env, monkeypatch):
    c, _, shows_mod, show_runner = run_env
    d = _show_ready_with_audio(shows_mod, "Thriller")
    calls = _install_spawn(show_runner, monkeypatch, PILOT_LINES, rc=None)

    r = c.post(f"/api/shows/{d.id}/run",
               json={"operator": "alois", "mode": "live", "free": True,
                     "confirmation": PHRASE})
    assert r.status_code == 409
    assert "forbidden in live mode" in r.json()["detail"]
    assert calls == []


def test_free_run_forces_stand_even_in_rehearsal_without_exit_stand(run_env, monkeypatch):
    """Free forces the stand handoff from the standtail motion regardless of the manual
    exit_stand toggle — it comes from the config, not the checkbox."""
    c, _, shows_mod, show_runner = run_env
    d = _show_ready_with_audio(shows_mod, "FreeRehearse")
    policy, meta, npz = _write_free_bundle(shows_mod, show_runner)
    monkeypatch.setenv("G1_ALLOW_UNSIGNED_FREE", "1")
    calls = _install_spawn(show_runner, monkeypatch, PILOT_LINES, rc=None)
    r = c.post(f"/api/shows/{d.id}/run",
               json={"operator": "alois", "mode": "rehearsal", "free": True,
                     "confirmation": PHRASE})
    assert r.status_code == 200, r.text
    cmd, env = calls[0]
    assert env["EXIT_MODE"] == "stand"
    assert env["GROUND_LEG_KP_SCALE"] == "1.5"
    assert env["MAX_SECS"] == "57"
    assert cmd[cmd.index("--policy") + 1] == str(policy.relative_to(shows_mod.PROJECT_ROOT))
    assert cmd[cmd.index("--meta") + 1] == str(meta.relative_to(shows_mod.PROJECT_ROOT))
    assert cmd[cmd.index("--motion-npz") + 1] == str(npz.relative_to(shows_mod.PROJECT_ROOT))


# ---- non-free run: proven default is untouched -----------------------------------

def test_non_free_run_passes_exact_promoted_bundle(run_env, monkeypatch):
    c, _, shows_mod, show_runner = run_env
    d = _show_ready_with_audio(shows_mod, "ProvenDefault")
    calls = _install_spawn(show_runner, monkeypatch, PILOT_LINES, rc=None)

    r = c.post(f"/api/shows/{d.id}/run",
               json={"operator": "alois", "mode": "live", "confirmation": PHRASE})
    assert r.status_code == 200, r.text
    cmd, env = calls[0]

    # none of the free-only control knobs leak onto the signed path
    for k in ("GROUND_LEG_KP_SCALE", "EXIT_MODE", "MAX_SECS"):
        assert k not in env, f"free knob {k} leaked onto the proven default"
    # the arm cap is a shared default (2.2) — present on both paths
    assert env["ARM_ACTION_CAP_SCALE"] == "2.2"
    # No implicit deploy defaults: the exact promoted bundle is always explicit.
    assert cmd == [
        str(show_runner.SHOW_RUN_SH),
        "--policy", d.policy_path,
        "--meta", d.meta_path,
        "--motion-npz", d.npz_path,
    ]


def test_non_free_rehearsal_exit_stand_unaffected(run_env, monkeypatch):
    """The pre-existing experimental exit_stand path (rehearsal-only) still works and
    does NOT pull in the free policy override."""
    c, _, shows_mod, show_runner = run_env
    d = _show_ready_with_audio(shows_mod, "ExitStand")
    calls = _install_spawn(show_runner, monkeypatch, PILOT_LINES, rc=None)
    r = c.post(f"/api/shows/{d.id}/run",
               json={"operator": "alois", "mode": "rehearsal", "exit_stand": True,
                     "confirmation": PHRASE})
    assert r.status_code == 200, r.text
    cmd, env = calls[0]
    assert env["EXIT_MODE"] == "stand"
    assert "GROUND_LEG_KP_SCALE" not in env  # not the free config
    assert cmd == [
        str(show_runner.SHOW_RUN_SH),
        "--policy", d.policy_path,
        "--meta", d.meta_path,
        "--motion-npz", d.npz_path,
    ]


# ---- the full guard chain still holds for a free request -------------------------

def test_free_does_not_bypass_show_ready_guard(run_env):
    c, _, shows_mod, _ = run_env
    draft = shows_mod.new_dance("FreeDraft", duration_s=10.0)  # status draft
    r = c.post(f"/api/shows/{draft.id}/run",
               json={"operator": "alois", "mode": "live", "free": True,
                     "confirmation": PHRASE})
    assert r.status_code == 409
    assert "show-ready" in r.json()["detail"]


def test_free_silent_dance_not_blocked_on_audio(run_env):
    """2026-08-05: audio is optional for free shows too — a silent free run must
    fail on the FREE-mode guards (live forbidden), never on missing music."""
    c, _, shows_mod, _ = run_env
    d = _show_ready_with_audio(shows_mod, "FreeSilent")
    shows_mod.set_audio(d.id, None)  # strip the music
    r = c.post(f"/api/shows/{d.id}/run",
               json={"operator": "alois", "mode": "live", "free": True,
                     "confirmation": PHRASE})
    assert r.status_code == 409
    assert "music" not in r.json()["detail"]        # audio never the blocker
    assert "free" in r.json()["detail"].lower()     # the real guard: free+live forbidden


def test_free_requires_reachable_robot(run_env, monkeypatch):
    c, _, shows_mod, show_runner = run_env
    d = _show_ready_with_audio(shows_mod, "FreeUnreachable")
    monkeypatch.setattr(show_runner, "robot_reachable", lambda *a, **k: False)
    r = c.post(f"/api/shows/{d.id}/run",
               json={"operator": "alois", "mode": "rehearsal", "free": True,
                     "confirmation": PHRASE})
    assert r.status_code == 409
    assert "reachable" in r.json()["detail"].lower()


def test_free_requires_confirmation_phrase(run_env, monkeypatch):
    c, _, shows_mod, _ = run_env
    from pipeline import show_runner
    d = _show_ready_with_audio(shows_mod, "FreePhrase")
    _write_free_bundle(shows_mod, show_runner)
    monkeypatch.setenv("G1_ALLOW_UNSIGNED_FREE", "1")
    r = c.post(f"/api/shows/{d.id}/run",
               json={"operator": "alois", "mode": "rehearsal", "free": True,
                     "confirmation": PHRASE.lower()})
    assert r.status_code == 403


def test_free_run_honors_single_run_lock(run_env, monkeypatch):
    c, _, shows_mod, show_runner = run_env
    d = _show_ready_with_audio(shows_mod, "FreeBusy")
    _write_free_bundle(shows_mod, show_runner)
    monkeypatch.setenv("G1_ALLOW_UNSIGNED_FREE", "1")
    _install_spawn(show_runner, monkeypatch, PILOT_LINES, rc=None)  # stays running
    assert c.post(f"/api/shows/{d.id}/run",
                  json={"operator": "alois", "mode": "rehearsal", "free": True,
                        "confirmation": PHRASE}).status_code == 200
    r2 = c.post(f"/api/shows/{d.id}/run",
                json={"operator": "alois", "mode": "rehearsal", "free": True,
                      "confirmation": PHRASE})
    assert r2.status_code == 409
    assert "already running" in r2.json()["detail"]
