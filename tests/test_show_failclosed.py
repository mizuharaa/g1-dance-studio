"""Fail-closed show bundle authorization regression tests (F6)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import artifacts, exam_verdict, policy_store

PHRASE = "I AM PRESENT WITH THE DAMPING REMOTE"


class FakeProc:
    pid = 12345

    @staticmethod
    def poll():
        return None


def _promoted_bundle(shows, root: Path, name: str = "Authorized"):
    policy_dir = root / "bundle"
    policy_dir.mkdir(exist_ok=True)
    policy = policy_dir / "policy.onnx"
    meta = policy_dir / "policy_meta.json"
    npz = policy_dir / "dance_deploy.npz"
    motion = root / "motion.csv"
    policy.write_bytes(b"policy-v1")
    meta.write_text(json.dumps({"contract": "v2"}))
    npz.write_bytes(b"npz-v1")
    motion.write_text("0,0,0.79\n")
    artifacts.write_manifest(policy_dir / "bundle.json", {
        "policy": {
            "onnx": artifacts.file_entry(policy, policy_dir),
            "meta": artifacts.file_entry(meta, policy_dir),
        },
        "motion": {
            "tempo_npz": {"100": artifacts.file_entry(npz, policy_dir)},
        },
    })
    dance = shows.new_dance(
        name,
        duration_s=30,
        policy_path=str(policy.relative_to(root)),
        motion_csv=str(motion.relative_to(root)),
    )
    policy_sha = exam_verdict.full_sha256(policy)
    for _ in range(shows.REPEATABILITY_TARGET):
        dance = shows.record_sim_run(
            shows.load_dance(dance.id), True, policy_sha256=policy_sha
        )
    dance = shows.promote(shows.load_dance(dance.id), "show-ready")
    return shows.set_audio(dance.id, {"track": "data/audio/song.wav"})


@pytest.fixture
def run_env(dances_env, client, monkeypatch):
    shows, _ = dances_env
    http, server = client
    from pipeline import show_runner

    monkeypatch.setattr(policy_store, "snapshot_policy", lambda *_a, **_k: None)
    monkeypatch.setattr(show_runner, "_current", None)
    monkeypatch.setattr(show_runner, "robot_reachable", lambda *_a, **_k: True)
    monkeypatch.setattr(
        server.venue,
        "get_active_venue",
        lambda: SimpleNamespace(name="Test venue"),
    )
    calls = []

    def spawn(cmd, env, log_path):
        calls.append((cmd, env, log_path))
        return FakeProc()

    monkeypatch.setattr(show_runner, "spawn_show_process", spawn)
    monkeypatch.delenv("G1_ALLOW_UNSIGNED_FREE", raising=False)
    return http, shows, show_runner, calls


def _run(http, dance, **updates):
    body = {
        "operator": "alois",
        "mode": "rehearsal",
        "confirmation": PHRASE,
        **updates,
    }
    return http.post(f"/api/shows/{dance.id}/run", json=body)


def test_promotion_records_complete_bundle_identity(run_env):
    _http, shows, _runner, _calls = run_env
    dance = _promoted_bundle(shows, shows.PROJECT_ROOT)
    manifest = json.loads((shows.PROJECT_ROOT / "bundle" / "bundle.json").read_text())
    assert dance.status == "show-ready"
    assert dance.meta_path == "bundle/policy_meta.json"
    assert dance.npz_path == "bundle/dance_deploy.npz"
    assert dance.policy_sha256 == artifacts.sha256_file(shows.PROJECT_ROOT / dance.policy_path)
    assert dance.meta_sha256 == artifacts.sha256_file(shows.PROJECT_ROOT / dance.meta_path)
    assert dance.npz_sha256 == artifacts.sha256_file(shows.PROJECT_ROOT / dance.npz_path)
    assert dance.motion_sha256 == artifacts.sha256_file(shows.PROJECT_ROOT / dance.motion_csv)
    assert dance.bundle_id == manifest["bundle_id"]
    assert dance.legacy_bundle is False


def test_promotion_without_manifest_records_complete_legacy_bundle(run_env):
    _http, shows, _runner, _calls = run_env
    root = shows.PROJECT_ROOT
    policy = root / "policy.onnx"
    meta = root / "policy_meta.json"
    npz = root / "only_deploy.npz"
    motion = root / "motion.csv"
    policy.write_bytes(b"legacy-policy")
    meta.write_text("{}")
    npz.write_bytes(b"legacy-npz")
    motion.write_text("0,0,0.79\n")
    dance = shows.new_dance(
        "Complete legacy", policy_path="policy.onnx", motion_csv="motion.csv"
    )
    sha = artifacts.sha256_file(policy)
    for _ in range(shows.REPEATABILITY_TARGET):
        shows.record_sim_run(shows.load_dance(dance.id), True, policy_sha256=sha)
    dance = shows.promote(shows.load_dance(dance.id), "show-ready")
    assert dance.legacy_bundle is True
    assert dance.bundle_id is None
    assert dance.meta_path == "policy_meta.json"
    assert dance.npz_path == "only_deploy.npz"


@pytest.mark.parametrize(
    ("field", "member"),
    [
        ("policy_path", "policy.onnx"),
        ("meta_path", "policy_meta.json"),
        ("npz_path", "motion NPZ"),
        ("motion_csv", "motion CSV"),
    ],
)
def test_mutating_any_promoted_member_makes_spawn_impossible(
    run_env, field, member
):
    http, shows, _runner, calls = run_env
    dance = _promoted_bundle(shows, shows.PROJECT_ROOT, name=field)
    path = Path(getattr(dance, field))
    if not path.is_absolute():
        path = shows.PROJECT_ROOT / path
    path.write_bytes(path.read_bytes() + b"tampered")

    response = _run(http, dance)
    assert response.status_code == 422
    assert member in response.json()["detail"]
    assert calls == []


def test_deleting_recorded_npz_makes_spawn_impossible(run_env):
    http, shows, _runner, calls = run_env
    dance = _promoted_bundle(shows, shows.PROJECT_ROOT, name="Deleted NPZ")
    (shows.PROJECT_ROOT / dance.npz_path).unlink()

    response = _run(http, dance)
    assert response.status_code == 422
    assert "motion NPZ" in response.json()["detail"]
    assert "missing" in response.json()["detail"]
    assert calls == []


def test_mutating_manifest_makes_spawn_impossible(run_env):
    http, shows, _runner, calls = run_env
    dance = _promoted_bundle(shows, shows.PROJECT_ROOT, name="Manifest")
    manifest_path = shows.PROJECT_ROOT / "bundle" / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["unauthorized_edit"] = True
    manifest_path.write_text(json.dumps(manifest))

    response = _run(http, dance)
    assert response.status_code == 422
    assert "bundle.json" in response.json()["detail"]
    assert calls == []


def test_extra_earlier_npz_is_ignored_and_happy_path_uses_recorded_npz(run_env):
    http, shows, runner, calls = run_env
    dance = _promoted_bundle(shows, shows.PROJECT_ROOT, name="Exact NPZ")
    extra = shows.PROJECT_ROOT / "bundle" / "000_deploy.npz"
    extra.write_bytes(b"wrong-but-lexicographically-first")

    response = _run(http, dance)
    assert response.status_code == 200, response.text
    assert len(calls) == 1
    cmd = calls[0][0]
    assert cmd[0] == str(runner.SHOW_RUN_SH)
    assert cmd[cmd.index("--policy") + 1] == dance.policy_path
    assert cmd[cmd.index("--meta") + 1] == dance.meta_path
    assert cmd[cmd.index("--motion-npz") + 1] == dance.npz_path
    assert str(extra.relative_to(shows.PROJECT_ROOT)) not in cmd


def test_free_is_forbidden_in_live_mode(run_env):
    http, shows, _runner, calls = run_env
    dance = _promoted_bundle(shows, shows.PROJECT_ROOT, name="No Free Live")
    response = _run(http, dance, mode="live", free=True)
    assert response.status_code == 409
    assert "forbidden in live mode" in response.json()["detail"]
    assert calls == []


def test_free_rehearsal_requires_explicit_env_opt_in(run_env):
    http, shows, _runner, calls = run_env
    dance = _promoted_bundle(shows, shows.PROJECT_ROOT, name="No Free Trial")
    response = _run(http, dance, free=True)
    assert response.status_code == 409
    assert "G1_ALLOW_UNSIGNED_FREE=1" in response.json()["detail"]
    assert calls == []


def test_opted_in_free_rehearsal_rehashes_and_uses_standtail_bundle(
    run_env, monkeypatch
):
    http, shows, runner, calls = run_env
    dance = _promoted_bundle(shows, shows.PROJECT_ROOT, name="Free Trial")
    free_dir = shows.PROJECT_ROOT / runner.FREE_POLICY_DIR
    free_dir.mkdir(parents=True)
    policy = free_dir / "policy.onnx"
    meta = free_dir / "policy_meta.json"
    npz = free_dir / "thriller_deploy.npz"
    policy.write_bytes(b"free-policy")
    meta.write_text("{}")
    npz.write_bytes(b"free-npz")
    artifacts.write_manifest(free_dir / "bundle.json", {
        "policy": {
            "onnx": artifacts.file_entry(policy, free_dir),
            "meta": artifacts.file_entry(meta, free_dir),
        },
        "motion": {"tempo_npz": {"100": artifacts.file_entry(npz, free_dir)}},
    })
    monkeypatch.setenv("G1_ALLOW_UNSIGNED_FREE", "1")

    response = _run(http, dance, free=True)
    assert response.status_code == 200, response.text
    cmd = calls[0][0]
    assert cmd[cmd.index("--policy") + 1] == str(policy.relative_to(shows.PROJECT_ROOT))
    assert cmd[cmd.index("--meta") + 1] == str(meta.relative_to(shows.PROJECT_ROOT))
    assert cmd[cmd.index("--motion-npz") + 1] == str(npz.relative_to(shows.PROJECT_ROOT))


def test_incomplete_legacy_dance_returns_422_naming_member(run_env):
    http, shows, _runner, calls = run_env
    policy = shows.PROJECT_ROOT / "legacy.onnx"
    motion = shows.PROJECT_ROOT / "legacy.csv"
    policy.write_bytes(b"legacy-policy")
    motion.write_text("0,0,0.79\n")
    dance = shows.new_dance(
        "Incomplete legacy",
        status="show-ready",
        policy_path="legacy.onnx",
        motion_csv="legacy.csv",
        policy_sha256=artifacts.sha256_file(policy),
        motion_sha256=artifacts.sha256_file(motion),
        legacy_bundle=True,
        audio={"track": "data/audio/song.wav"},
    )

    response = _run(http, dance)
    assert response.status_code == 422
    assert "policy_meta.json" in response.json()["detail"]
    assert calls == []


def test_bundle_refusal_precedes_typed_phrase_check(run_env):
    http, shows, _runner, calls = run_env
    dance = _promoted_bundle(shows, shows.PROJECT_ROOT, name="Order")
    (shows.PROJECT_ROOT / dance.meta_path).write_text("tampered")

    response = _run(http, dance, confirmation="wrong")
    assert response.status_code == 422
    assert "policy_meta.json" in response.json()["detail"]
    assert calls == []


def test_validated_bundle_still_requires_exact_typed_phrase(run_env):
    http, shows, _runner, calls = run_env
    dance = _promoted_bundle(shows, shows.PROJECT_ROOT, name="Consent")
    response = _run(http, dance, confirmation="wrong")
    assert response.status_code == 403
    assert "EXACTLY" in response.json()["detail"]
    assert calls == []
