"""Regression tests for the deploy-path safety review (docs/safety_review_findings.md).

Each test names the finding it closes and fails if the hole reopens.
"""
import json

import pytest

from pipeline import exam_verdict as ev


def _good_verdict(policy_sha="a" * 64, motion_sha="b" * 64):
    return {
        "schema": "sim_exam/v1",
        "policy_sha256": policy_sha,
        "motion_sha256": motion_sha,
        "nominal": {"pass": True, "duration_s": 44.3, "tracked": True},
        "push": {"pass": True, "force_n": 250.0, "recovery_rate": 1.0},
        "repeatability": {"pass": True, "runs": 3, "clean": 3},
        "verdict": "pass",
    }


KEY = b"\x01" * 32


# ---- finding #0: verdict-string trust + fabrication ------------------------------
def test_hand_edited_fail_to_pass_is_rejected():
    """A genuine FAIL whose 'verdict' string is flipped to 'pass' must NOT authorize."""
    v = _good_verdict()
    v["nominal"]["pass"] = False          # it actually failed
    v["verdict"] = "pass"                  # attacker flips the string
    signed = ev.sign_verdict(v, KEY)       # even if (somehow) re-signed...
    ok, reason = ev.authorize(signed, key=KEY)
    assert not ok and "content-derived" in reason


def test_empty_phase_dicts_do_not_satisfy_presence():
    """The old `is not None` check passed {} — content derivation must reject it."""
    v = _good_verdict()
    v["push"] = {}
    v["repeatability"] = {}
    signed = ev.sign_verdict(v, KEY)
    ok, _ = ev.authorize(signed, key=KEY)
    assert not ok


def test_unsigned_verdict_never_authorizes():
    """A fabricated verdict with no valid signature is inert (finding #0)."""
    v = _good_verdict()  # no signature at all
    ok, reason = ev.authorize(v, key=KEY)
    assert not ok and "signature" in reason


def test_wrong_key_signature_rejected():
    signed = ev.sign_verdict(_good_verdict(), b"\x02" * 32)
    ok, _ = ev.authorize(signed, key=KEY)
    assert not ok


def test_genuine_signed_pass_authorizes():
    signed = ev.sign_verdict(_good_verdict(), KEY)
    ok, reason = ev.authorize(signed, policy_sha="a" * 64, motion_sha="b" * 64, key=KEY)
    assert ok, reason


def test_sha_binding_mismatch_rejected():
    """A valid verdict for policy A cannot authorize policy B (findings #7/#25/#27)."""
    signed = ev.sign_verdict(_good_verdict(), KEY)
    ok, reason = ev.authorize(signed, policy_sha="c" * 64, key=KEY)
    assert not ok and "policy sha" in reason


# ---- finding #21: honest verdict / phase completeness ----------------------------
def test_incomplete_repeatability_below_floor_rejected():
    v = _good_verdict()
    v["repeatability"] = {"pass": True, "runs": 2, "clean": 2}  # below REQUIRED_CLEAN_RUNS
    signed = ev.sign_verdict(v, KEY)
    assert not ev.authorize(signed, key=KEY)[0]


# ---- finding #22: push force floor -----------------------------------------------
def test_below_floor_push_force_rejected():
    v = _good_verdict()
    v["push"] = {"pass": True, "force_n": 5.0, "recovery_rate": 1.0}  # love-tap
    signed = ev.sign_verdict(v, KEY)
    assert not ev.authorize(signed, key=KEY)[0]


# ---- finding #32: full sha256 ----------------------------------------------------
def test_full_sha256_length(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    assert len(ev.full_sha256(f)) == 64


# ---- gen_config uses the authenticated gate (findings #0/#7/#19) ------------------
def test_gen_config_find_passing_exam_requires_signature(tmp_path, monkeypatch):
    from deploy import gen_config
    exam_dir = tmp_path
    v = _good_verdict()  # unsigned
    (exam_dir / "exam_x.json").write_text(json.dumps(v))
    # unsigned verdict must not be found as "passing"
    assert gen_config.find_passing_exam("a" * 64, "b" * 64, exam_dir) is None
    # signed one is found
    (exam_dir / "exam_y.json").write_text(json.dumps(ev.sign_verdict(v)))
    assert gen_config.find_passing_exam("a" * 64, "b" * 64, exam_dir) is not None


# ---- finding #31: dance-name allowlist -------------------------------------------
def test_dance_name_allowlist():
    from deploy.gen_config import DANCE_RE
    assert DANCE_RE.match("thriller_show-1")
    assert not DANCE_RE.match("../../etc/passwd")
    assert not DANCE_RE.match("a; rm -rf /")


# ---- findings #13/#29: battery floor enforced ------------------------------------
def test_battery_below_floor_rejected(tmp_path, monkeypatch):
    from pipeline import shows
    monkeypatch.setattr(shows, "DATA_DIR", tmp_path)
    monkeypatch.setattr(shows, "DANCES_DIR", tmp_path / "dances")
    monkeypatch.setattr(shows, "SHOWS_DIR", tmp_path / "shows")
    (tmp_path / "dances").mkdir(parents=True)
    (tmp_path / "shows").mkdir(parents=True)
    d = shows.new_dance("t", status="show-ready")
    show = shows.new_show(d, "op")
    # advance to the battery step
    shows.complete_step(show, "robot_health", True)
    shows.complete_step(show, "space_clear", True)
    with pytest.raises(ValueError, match="floor"):
        shows.complete_step(show, "battery", 25.0)
    # at/above floor is accepted (mutators now return the fresh record — finding #28)
    show = shows.complete_step(show, "battery", 55.0)
    assert show.steps["battery"]["value"] == 55.0
