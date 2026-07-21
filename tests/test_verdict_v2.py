"""Honest disturbance and replay-proof verdict-v2 regressions (F7)."""
from __future__ import annotations

import json
import uuid

import pytest

from pipeline import exam_verdict, mjlab_verify, shows

KEY = b"verdict-v2-test-key" * 2


def _eval(seed: int = 1200) -> dict:
    return {
        "dance": "test",
        "conditions": {
            "nominal": {
                "num_episodes": 4,
                "n_success": 4,
                "success_rate": 1.0,
                "mpkpe_m": 0.1,
                "ee_pos_error_m": 0.05,
                "seed": seed,
            },
            "push": {
                "num_episodes": 4,
                "n_success": 4,
                "success_rate": 1.0,
                "mpkpe_m": 0.1,
                "seed": seed + 1,
            },
        },
    }


@pytest.fixture(autouse=True)
def signing_key(monkeypatch):
    monkeypatch.setattr(exam_verdict, "_signing_key", lambda: KEY)


@pytest.fixture
def verdict_env(tmp_path, monkeypatch):
    monkeypatch.setattr(shows, "DATA_DIR", tmp_path)
    monkeypatch.setattr(shows, "DANCES_DIR", tmp_path / "dances")
    monkeypatch.setattr(shows, "SHOWS_DIR", tmp_path / "shows")
    monkeypatch.setattr(shows, "PROJECT_ROOT", tmp_path)
    (tmp_path / "dances").mkdir()
    (tmp_path / "shows").mkdir()
    policy = tmp_path / "policy.onnx"
    motion = tmp_path / "motion.csv"
    policy.write_bytes(b"policy")
    motion.write_text("0,0,0.79\n")
    dance = shows.new_dance(
        "Verdict v2", policy_path="policy.onnx", motion_csv="motion.csv"
    )
    return dance, policy, motion


def _verdict(policy, motion, seed=1200):
    return mjlab_verify.build_verdict(_eval(seed), policy, motion)


def test_v2_round_trips_signature_and_honest_disturbance(verdict_env):
    _dance, policy, motion = verdict_env
    verdict = _verdict(policy, motion)
    assert verdict["schema"] == "sim_exam/v2"
    assert uuid.UUID(verdict["eval_id"]).version == 4
    assert verdict["created_at"].endswith("+00:00")
    assert verdict["disturbance"] == {
        "kind": "delta_velocity",
        "delta_v_mps": 0.5,
        "axes": ["x", "y", "z"],
        "seed": 1201,
    }
    assert "force_n" not in json.dumps(verdict)
    assert exam_verdict.signature_valid(verdict)
    assert exam_verdict.derive_pass(verdict)


def test_tampering_disturbance_breaks_signature(verdict_env):
    _dance, policy, motion = verdict_env
    verdict = _verdict(policy, motion)
    verdict["disturbance"]["delta_v_mps"] = 0.1
    assert not exam_verdict.signature_valid(verdict)


def test_delta_velocity_below_floor_cannot_pass_even_when_signed(verdict_env):
    _dance, policy, motion = verdict_env
    verdict = _verdict(policy, motion)
    verdict["disturbance"]["delta_v_mps"] = 0.49
    verdict = exam_verdict.sign_verdict(verdict)
    assert exam_verdict.signature_valid(verdict)
    assert not exam_verdict.derive_pass(verdict)


def test_same_signed_verdict_replay_gets_one_credit_and_explicit_refusals(verdict_env):
    dance, policy, motion = verdict_env
    verdict = _verdict(policy, motion)
    credits = []
    first = shows.record_sim_run_from_verdict(dance.id, verdict)
    credits.append(first.repeatability["consecutive_clean"])
    for _ in range(2):
        with pytest.raises(shows.VerdictReplayError, match="already recorded"):
            shows.record_sim_run_from_verdict(dance.id, verdict)
    current = shows.load_dance(dance.id)
    assert credits == [1]
    assert current.repeatability["consecutive_clean"] == 1
    assert current.repeatability["total_runs"] == 1
    assert current.exam_ids == [verdict["eval_id"]]


def test_signed_verdict_without_eval_id_is_refused(verdict_env):
    dance, policy, motion = verdict_env
    verdict = _verdict(policy, motion)
    verdict.pop("eval_id")
    verdict = exam_verdict.sign_verdict(verdict)
    with pytest.raises(shows.VerdictError, match="eval_id"):
        shows.record_sim_run_from_verdict(dance.id, verdict)
    assert shows.load_dance(dance.id).repeatability["total_runs"] == 0


def test_three_distinct_eval_ids_produce_three_credits(verdict_env):
    dance, policy, motion = verdict_env
    ids = []
    credits = []
    for seed in (1200, 1300, 1400):
        verdict = _verdict(policy, motion, seed)
        ids.append(verdict["eval_id"])
        recorded = shows.record_sim_run_from_verdict(dance.id, verdict)
        credits.append(recorded.repeatability["consecutive_clean"])
    assert credits == [1, 2, 3]
    assert len(set(ids)) == 3
    assert shows.load_dance(dance.id).exam_ids == ids


def test_duplicate_eval_id_is_http_409(dances_env, client):
    shows_mod, _ = dances_env
    http, _server = client
    policy = shows_mod.PROJECT_ROOT / "policy.onnx"
    motion = shows_mod.PROJECT_ROOT / "motion.csv"
    policy.write_bytes(b"policy")
    motion.write_text("0,0,0.79\n")
    dance = shows_mod.new_dance(
        "HTTP replay", policy_path="policy.onnx", motion_csv="motion.csv"
    )
    verdict = _verdict(policy, motion)
    endpoint = f"/api/dances/{dance.id}/sim-runs"
    assert http.post(endpoint, json={"verdict": verdict}).status_code == 200
    replay = http.post(endpoint, json={"verdict": verdict})
    assert replay.status_code == 409
    assert "already recorded" in replay.json()["detail"]


def test_old_show_ready_record_is_badged_legacy_unverified(verdict_env):
    dance, _policy, _motion = verdict_env
    dance.status = "show-ready"
    dance.repeatability["consecutive_clean"] = 3
    dance.repeatability["history"] = [
        {"passed": True, "exam_id": None} for _ in range(3)
    ]
    dance.save()
    loaded = shows.load_dance(dance.id)
    assert loaded.status == "show-ready"
    assert loaded.exam_ids == []
    assert loaded.repeat_evidence == "legacy-unverified"
