"""AGENT D — policy-in-the-loop sandbox smoke tests.

Guards that the sandbox still runs the REAL deploy contract (imported from
pipeline/deploy_runtime.py) and produces a sane tracking report — so a change to the
obs builder / action mapping that would silently desync the twin from the robot fails
here. Needs mujoco + onnxruntime + the thriller policy bundle; skipped otherwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")

from tools.sim_sandbox import run_sandbox, tracking_report  # noqa: E402

DANCE = Path(__file__).resolve().parent.parent / "data/policies/thriller_csv_ankle_penalty"
pytestmark = pytest.mark.skipif(not DANCE.exists(), reason="thriller policy bundle absent")


def test_sandbox_runs_deploy_contract_and_reports():
    # window must include real dance motion (onset ~3.6 s) for the fidelity metric to mean
    # something; before that the reference barely moves and the ratio is degenerate.
    out, model, meta = run_sandbox(DANCE, steps=300, latency_ms=0.0, tether_kp=150.0)
    rep = tracking_report(out)
    assert 0.0 <= rep["achieved_fraction_overall"] <= 1.0   # a valid fraction
    assert rep["rms_err_rad"] > 0.0                         # the policy does NOT track perfectly
    assert len(rep["per_dof_achieved"]) == 29               # full 29-dof contract
    assert out["q"].shape[1] == 29 and out["ref_jp"].shape[1] == 29


def test_latency_injection_is_not_a_noop():
    a, _, _ = run_sandbox(DANCE, steps=60, latency_ms=0.0, tether_kp=150.0)
    b, _, _ = run_sandbox(DANCE, steps=60, latency_ms=80.0, tether_kp=150.0)
    n = min(len(a["q"]), len(b["q"]))
    assert not np.allclose(a["q"][:n], b["q"][:n])          # delay changes what the robot does


def test_studio_kinematic_reference_matches_the_deploy_npz():
    """The 'intended' panel plays the reference joint_pos straight — verify it does."""
    from tools.sim_studio import _kinematic_reference
    rec = _kinematic_reference(DANCE, steps=50)
    assert rec["q"].shape[1] == 29 and len(rec["q"]) == 50
    assert rec["achieved"] == 1.0                           # reference = a perfect tracker
    import pipeline.deploy_runtime as D
    ref = D.Reference(next(DANCE.glob("*_deploy.npz")))
    assert np.allclose(rec["q"][10], ref.jp[10])            # frame 10 IS reference frame 10
