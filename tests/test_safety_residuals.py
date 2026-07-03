"""Regression tests for the 3 residual safety-review findings (#23/#24, #28, #6).

Closes docs/safety_review_findings.md residuals: the app now authenticates sim-exam
verdicts before crediting show-readiness, per-record mutations are serialized, and the
exam's repeatability phase applies real domain randomization.
"""
from __future__ import annotations

import threading

import pytest

from pipeline import exam_verdict as ev
from pipeline import shows


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def show_env(tmp_path, monkeypatch):
    """Point shows.py at a temp data dir and give a dance a real policy+motion file."""
    monkeypatch.setattr(shows, "DATA_DIR", tmp_path)
    monkeypatch.setattr(shows, "DANCES_DIR", tmp_path / "dances")
    monkeypatch.setattr(shows, "SHOWS_DIR", tmp_path / "shows")
    monkeypatch.setattr(shows, "PROJECT_ROOT", tmp_path)
    (tmp_path / "dances").mkdir(parents=True)
    (tmp_path / "shows").mkdir(parents=True)
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"fake-policy-bytes")
    motion = tmp_path / "motion.csv"
    motion.write_text("0,0,0.79\n")
    dance = shows.new_dance("t", policy_path="policy.onnx", motion_csv="motion.csv")
    return dance, policy, motion


def _signed_pass_verdict(policy, motion):
    v = {
        "schema": "sim_exam/v1",
        "policy_sha256": ev.full_sha256(policy),
        "motion_sha256": ev.full_sha256(motion),
        "nominal": {"pass": True, "duration_s": 44.3, "tracked": True},
        "push": {"pass": True, "force_n": 250.0},
        "repeatability": {"pass": True, "runs": 3, "clean": 3},
        "verdict": "pass",
    }
    return ev.sign_verdict(v)


# ---- #23/#24: sim-runs must authenticate the verdict -----------------------------
def test_signed_pass_verdict_credits_streak_and_pins_sha(show_env):
    dance, policy, motion = show_env
    v = _signed_pass_verdict(policy, motion)
    out = shows.record_sim_run_from_verdict(dance.id, v)
    assert out.repeatability["consecutive_clean"] == 1
    assert out.status == "sim-verified"
    assert out.policy_sha256 == ev.full_sha256(policy)  # exam-passed policy pinned


def test_unsigned_verdict_is_rejected(show_env):
    """The old hole: a bare pass claim. An unsigned verdict must not credit anything."""
    dance, policy, motion = show_env
    v = _signed_pass_verdict(policy, motion)
    v.pop("signature")
    with pytest.raises(shows.VerdictError, match="signature"):
        shows.record_sim_run_from_verdict(dance.id, v)


def test_verdict_for_a_different_policy_is_rejected(show_env):
    """A validly-signed verdict about some OTHER artifact must not credit this dance."""
    dance, policy, motion = show_env
    v = _signed_pass_verdict(policy, motion)
    v["policy_sha256"] = "d" * 64
    v = ev.sign_verdict(v)  # re-sign so signature is valid but sha is wrong
    with pytest.raises(shows.VerdictError, match="policy_sha256"):
        shows.record_sim_run_from_verdict(dance.id, v)


def test_promote_refuses_after_policy_swapped(show_env):
    """#24/#27: pass the exam, then swap the policy file — promotion must refuse."""
    dance, policy, motion = show_env
    for _ in range(shows.REPEATABILITY_TARGET):
        shows.record_sim_run_from_verdict(dance.id, _signed_pass_verdict(policy, motion))
    # earned it — promotion works on the exact tested bytes
    ok = shows.promote(shows.load_dance(dance.id), "show-ready")
    assert ok.status == "show-ready"
    # now swap the policy on disk and demote back, then try to re-promote
    shows.load_dance(dance.id)
    policy.write_bytes(b"a-DIFFERENT-policy")
    d = shows.load_dance(dance.id)
    d.status = "sim-verified"
    d.save()
    with pytest.raises(ValueError, match="sha mismatch"):
        shows.promote(shows.load_dance(dance.id), "show-ready")


# ---- #28: concurrent mutations must not mask a failing run -----------------------
def test_concurrent_runs_do_not_mask_a_failure(show_env):
    """Interleave passes with one fail across threads: the fail must reset the streak,
    never be lost to a racing read-modify-write."""
    dance, policy, motion = show_env
    # seed a clean streak
    for _ in range(3):
        shows.record_sim_run(shows.load_dance(dance.id), True)
    barrier = threading.Barrier(6)

    def do(passed):
        barrier.wait()
        shows.record_sim_run(shows.load_dance(dance.id), passed)

    threads = [threading.Thread(target=do, args=(p,))
               for p in (True, True, False, True, True, True)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    final = shows.load_dance(dance.id)
    # 3 seed + 6 concurrent = 9 total runs all durably recorded (no lost update)
    assert final.repeatability["total_runs"] == 9
    # exactly one failure exists in history — the streak counter cannot exceed the
    # number of passes that followed it, so the fail was not masked away
    fails = [h for h in final.repeatability["history"] if not h["passed"]]
    assert len(fails) == 1


# ---- #6: exam domain randomization -----------------------------------------------
def test_repeatability_records_dr_and_distinct_seeds(monkeypatch):
    """run_repeatability must apply DR (recorded) and use de-correlated seeds."""
    from pipeline import sim_exam

    calls = []

    def fake_nominal(env, seed=0, jitter=0.0, dr=None, **kw):
        calls.append((seed, dr))
        return {"pass": True, "survived_s": 1.0, "mean_joint_err_rad": 0.01}

    monkeypatch.setattr(sim_exam, "run_nominal", fake_nominal)
    out = sim_exam.run_repeatability(env=None, runs=4)
    seeds = [c[0] for c in calls]
    assert len(set(seeds)) == 4                      # de-correlated (was identical)
    assert all(c[1] for c in calls)                  # DR applied every run
    assert out["dr"] and "friction_scale" in out["dr"]  # ranges recorded for audit


@pytest.mark.model
def test_dr_perturbs_state_and_restores_model():
    """Real DR effect: obs differs run-to-run under noise, and the shared model's
    friction/mass are restored after a DR run (no cross-run accumulation)."""
    import numpy as np
    from pipeline import sim_exam

    if not sim_exam.G1_XML.exists():
        pytest.skip("G1 model not available")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(sim_exam.G1_XML))
    # a short standing motion in the 36-col CSV convention
    import tempfile
    from pathlib import Path
    frames = np.zeros((6, 36))
    frames[:, 2] = 0.79
    frames[:, 6] = 1.0
    tmp = Path(tempfile.mkstemp(suffix=".csv")[1])
    np.savetxt(tmp, frames, delimiter=",", fmt="%.6f")
    motion = sim_exam.load_motion(tmp, model, "torso_link")
    policy = sim_exam.load_policy("stub", model, motion)
    env = sim_exam.ExamEnv(model, policy, motion)
    fric0 = model.geom_friction.copy()

    env.reset(seed=1, dr=sim_exam.DR_RANGES)
    obs_a = env.obs(0)
    env.reset(seed=2, dr=sim_exam.DR_RANGES)
    obs_b = env.obs(0)
    assert not np.allclose(obs_a, obs_b)             # DR + noise actually perturb
    env.reset(seed=0)                                # clean reset restores the model
    assert np.allclose(model.geom_friction, fric0)   # no accumulated perturbation
