"""pipeline/success_estimate.py — pre-training predicted-survival crosscheck.

Pure band-prediction logic is tested without MuJoCo; the end-to-end CLI test
runs the real checker on the committed adaptive Thriller motion and self-skips
when the menagerie model is absent (conftest convention).
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from .conftest import HAVE_MODEL, WORKTREE

from pipeline import success_estimate as se

CAL_PATH = WORKTREE / "data/calibration/success_calibration.json"
ADAPTIVE_CSV = WORKTREE / "data/motions/thriller/thriller_g1_grounded_adaptive.csv"


# ---- calibration table ----------------------------------------------------------

def test_calibration_file_loads_and_has_provenance():
    cal = se.load_calibration()
    assert cal["primary_metric"] == "ankle_frames_over_headroom_pct"
    assert len(cal["rows"]) >= 4          # 5 mined anchors (rows may grow)
    for row in cal["rows"]:
        assert row["nominal_survival_pct"] is not None
        assert row["metrics"]["ankle_frames_over_headroom_pct"] is not None
        assert row["gap_file"]            # provenance back to exports/
        assert row["motion"]
        assert row["provenance"]
    # the mined survival anchors must be present (regression on the mining)
    surv = sorted(r["nominal_survival_pct"] for r in cal["rows"])
    assert 85.94 in surv and 100.0 in surv


def _synthetic_cal():
    """A clean monotone calibration so the strict-monotonicity contract can be
    asserted independent of the messy real history."""
    rows = [
        {"metrics": {"ankle_frames_over_headroom_pct": 0.0},
         "nominal_survival_pct": 95.0},
        {"metrics": {"ankle_frames_over_headroom_pct": 5.0},
         "nominal_survival_pct": 90.0},
        {"metrics": {"ankle_frames_over_headroom_pct": 20.0},
         "nominal_survival_pct": 70.0},
        {"metrics": {"ankle_frames_over_headroom_pct": 50.0},
         "nominal_survival_pct": 40.0},
    ]
    return {"primary_metric": "ankle_frames_over_headroom_pct", "rows": rows}


def test_band_monotone_worse_metric_never_predicts_better():
    cal = _synthetic_cal()
    xs = [0.0, 2.0, 5.0, 12.0, 20.0, 35.0, 50.0, 70.0, 120.0]
    bands = [se.predict_band(x, cal) for x in xs]
    for (lo_a, hi_a), (lo_b, hi_b) in zip(bands, bands[1:]):
        assert lo_b <= lo_a + 1e-9, "lower bound must not improve as metric worsens"
        assert hi_b <= hi_a + 1e-9, "upper bound must not improve as metric worsens"
    # strictly worse far beyond the calibrated range (extrapolation decay)
    assert bands[-1][0] < bands[0][0]
    assert bands[-1][1] < bands[0][1]


def test_band_monotone_on_the_real_table():
    cal = se.load_calibration()
    xs = [0.0, 0.5, 1.5, 5.0, 20.0, 60.0]
    bands = [se.predict_band(x, cal) for x in xs]
    for (lo_a, hi_a), (lo_b, hi_b) in zip(bands, bands[1:]):
        assert lo_b <= lo_a + 1e-9
        assert hi_b <= hi_a + 1e-9


def test_band_is_honest_wide_and_clamped():
    cal = se.load_calibration()
    for x in (0.0, 1.0, 10.0, 200.0):
        lo, hi = se.predict_band(x, cal)
        assert hi - lo >= se.MIN_BAND_WIDTH - 1e-9, "n=5 cannot support a tight band"
        assert se.BAND_FLOOR <= lo <= hi <= se.BAND_CEIL, "never promise 0 or 100%"


def test_predict_band_rejects_empty_rows():
    with pytest.raises(ValueError):
        se.predict_band(1.0, {"primary_metric": "ankle_frames_over_headroom_pct",
                              "rows": [{"metrics": {}, "nominal_survival_pct": None}]})


# ---- risk-window merging ----------------------------------------------------------

def test_merge_risk_windows_merges_sorts_and_caps():
    wins = [[1.0, 1.5], [1.8, 2.6],          # gap 0.3 -> merged: 1.0-2.6
            [10.0, 10.03],                    # single-frame blip -> dropped
            [14.6, 16.4], [30.0, 30.4], [40.0, 41.0], [44.0, 44.5]]
    out = se.merge_risk_windows(wins, top=3)
    assert len(out) == 3
    assert [w["start_s"] for w in out] == sorted(w["start_s"] for w in out)
    assert {(w["start_s"], w["end_s"]) for w in out} == {(1.0, 2.6), (14.6, 16.4), (40.0, 41.0)}
    assert all(w["label"] for w in out)


def test_merge_risk_windows_tolerates_garbage():
    assert se.merge_risk_windows(None) == []
    assert se.merge_risk_windows([["x", 1], [None], []]) == []


# ---- end-to-end (MuJoCo) -----------------------------------------------------------

@pytest.mark.skipif(not HAVE_MODEL, reason="needs third_party/mujoco_menagerie")
@pytest.mark.skipif(not ADAPTIVE_CSV.exists(), reason="adaptive thriller CSV absent")
def test_cli_runs_on_adaptive_thriller(tmp_path):
    out = tmp_path / "est.json"
    proc = subprocess.run(
        [sys.executable, "-m", "pipeline.success_estimate", str(ADAPTIVE_CSV),
         "--json", str(out)],
        capture_output=True, text=True, timeout=600, cwd=WORKTREE)
    assert proc.returncode == 0, proc.stderr[-800:]
    est = json.loads(out.read_text())
    assert est["predicted_survival_pct_range"].endswith("%")
    assert est["predicted_survival_lo_pct"] <= est["predicted_survival_hi_pct"]
    assert "rough" in est["confidence"]
    assert est["hard_blockers"] == []
    # the honesty notes must survive to the artifact the UI shows
    assert any("not a guarantee" in n for n in est["notes"])
    assert any("OVER-estimated" in n for n in est["notes"])


@pytest.mark.skipif(not HAVE_MODEL, reason="needs third_party/mujoco_menagerie")
@pytest.mark.skipif(not ADAPTIVE_CSV.exists(), reason="adaptive thriller CSV absent")
def test_estimate_picks_up_vet_blockers():
    vet = {"hard": {"excursion": {"pass": False}, "grounding": {"pass": True}}}
    est = se.estimate(ADAPTIVE_CSV, vet_report=vet)
    assert est["hard_blockers"] == ["excursion"]


# ---- API surface --------------------------------------------------------------------

def test_dance_success_estimate_endpoint(client, jobs_env, dances_env):
    c, _ = client
    shows, _ = dances_env
    store, _ = jobs_env
    # a dance with no source job -> honest {available: false}
    d0 = shows.new_dance("no-estimate")
    got = c.get(f"/api/dances/{d0.id}/success-estimate").json()
    assert got == {"available": False, "estimate": None}
    # missing dance -> 404 (never a silent empty payload)
    assert c.get("/api/dances/nope/success-estimate").status_code == 404
    # a dance whose source job carries the estimate written by the stage
    job = store.new_job("est-dance")
    est = {"version": 1, "predicted_survival_pct_range": "86-99%",
           "risk_windows": [], "hard_blockers": []}
    ret = job.stage_dir("retarget")
    (ret / "success_estimate.json").write_text(json.dumps(est))
    d1 = shows.new_dance("with-estimate", source_job=job.id)
    got = c.get(f"/api/dances/{d1.id}/success-estimate").json()
    assert got["available"] is True
    assert got["estimate"]["predicted_survival_pct_range"] == "86-99%"
    # the job payload mirrors the vet pattern
    jd = c.get(f"/api/jobs/{job.id}").json()
    assert jd["success_estimate"]["predicted_survival_pct_range"] == "86-99%"
