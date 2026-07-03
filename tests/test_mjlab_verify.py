"""Tests for the held-out mjlab verdict producer (pipeline/mjlab_verify.py)."""
from __future__ import annotations

import json

from pipeline import mjlab_verify
from pipeline.exam_verdict import authorize, signature_valid, full_sha256


def _eval(nom_success, push_success, n=128):
    return {
        "dance": "thriller",
        "conditions": {
            "nominal": {"condition": "nominal", "num_episodes": n,
                        "n_success": int(round(nom_success * n)),
                        "success_rate": nom_success, "mpkpe_m": 0.17,
                        "ee_pos_error_m": 0.1, "seed": 90001},
            "push": {"condition": "push", "num_episodes": n,
                     "n_success": int(round(push_success * n)),
                     "success_rate": push_success, "mpkpe_m": 0.17, "seed": 90002},
        },
    }


def _files(tmp_path):
    pol = tmp_path / "policy.onnx"; pol.write_bytes(b"fake-policy")
    mot = tmp_path / "motion.npz"; mot.write_bytes(b"fake-motion")
    return pol, mot


def test_signed_and_method_labeled(tmp_path):
    pol, mot = _files(tmp_path)
    v = mjlab_verify.build_verdict(_eval(1.0, 1.0), pol, mot)
    assert v["schema"] == "sim_exam/v1"
    assert v["method"] == "mjlab_heldout_v1"  # never mislabeled as cross-engine
    assert signature_valid(v)  # tamper-evident


def test_perfect_run_authorizes_show_ready(tmp_path):
    pol, mot = _files(tmp_path)
    v = mjlab_verify.build_verdict(_eval(1.0, 1.0), pol, mot)
    ok, reason = authorize(v, policy_sha=full_sha256(pol), motion_sha=full_sha256(mot))
    assert ok, reason  # 128/128 clean → clean==runs, push force floor met


def test_98pct_does_not_authorize(tmp_path):
    # The real Thriller result: strong but not clean-every-time → must NOT pass.
    pol, mot = _files(tmp_path)
    v = mjlab_verify.build_verdict(_eval(0.984, 0.984), pol, mot)
    ok, _ = authorize(v, policy_sha=full_sha256(pol), motion_sha=full_sha256(mot))
    assert not ok
    assert v["repeatability"]["clean"] < v["repeatability"]["runs"]


def test_push_force_meets_floor(tmp_path):
    pol, mot = _files(tmp_path)
    v = mjlab_verify.build_verdict(_eval(1.0, 1.0), pol, mot)
    assert v["push"]["force_n"] >= 150.0  # MIN_PUSH_FORCE_N, honest m*dv/dt equiv


def test_tamper_breaks_signature(tmp_path):
    pol, mot = _files(tmp_path)
    v = mjlab_verify.build_verdict(_eval(0.984, 0.984), pol, mot)
    v["repeatability"]["clean"] = v["repeatability"]["runs"]  # forge to 100%
    assert not signature_valid(v)  # forging the numbers invalidates the signature
