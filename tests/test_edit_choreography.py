"""Tests for tools/edit_choreography.py — the music-sync-preserving
choreography difficulty editor.

Covers the four spec'd cases:
  (a) identity round trip: no sections + no cap => output identical to input;
  (b) a real-CSV section edit: frame count identical, arms bit-identical,
      boundary velocity continuity, proxy reduced within each section;
  (c) the lean cap reduces the >cap frame count on the real CSV;
  (d) the validators catch a deliberately broken edit (injected velocity
      spike => tool reports FAILED).

The real-CSV tests self-skip when the deployable Thriller CSV or the MuJoCo
model is absent (same convention as the rest of the suite).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import HAVE_MODEL, WORKTREE, make_motion

TOOL = WORKTREE / "tools/edit_choreography.py"
REAL_CSV = WORKTREE / "data/policies/thriller/thriller_deploy.csv"

_spec = importlib.util.spec_from_file_location("edit_choreography", TOOL)
ec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ec)

needs_model = pytest.mark.skipif(not HAVE_MODEL, reason="mujoco model missing")
needs_real = pytest.mark.skipif(
    not (HAVE_MODEL and REAL_CSV.exists()),
    reason="real thriller_deploy.csv or mujoco model missing")


def _args(tmp_path: Path, csv: Path, **kw) -> argparse.Namespace:
    d = dict(csv=str(csv), out=str(tmp_path / "out.csv"),
             report=str(tmp_path / "report.json"), sections="",
             leg_scale=0.6, lean_cap_nm=0.0, edge_blend_s=0.3)
    d.update(kw)
    return argparse.Namespace(**d)


# ------------------------------------------------------------------ pure helpers

def test_parse_sections():
    assert ec.parse_sections("") == []
    assert ec.parse_sections("13.0-17.5,43.0-47.0") == [(13.0, 17.5),
                                                        (43.0, 47.0)]
    # unsorted input comes back sorted
    assert ec.parse_sections("43.0-47.0, 13.0-17.5") == [(13.0, 17.5),
                                                         (43.0, 47.0)]
    with pytest.raises(ValueError):
        ec.parse_sections("17.5-13.0")
    with pytest.raises(ValueError):
        ec.parse_sections("nonsense")


def test_sections_to_frames_overlap_and_clamp():
    assert ec.sections_to_frames([(1.0, 2.0)], 300) == [(30, 60)]
    # clamped to the last frame
    assert ec.sections_to_frames([(1.0, 99.0)], 90) == [(30, 89)]
    with pytest.raises(ValueError):                     # overlap
        ec.sections_to_frames([(1.0, 2.0), (1.9, 3.0)], 300)
    with pytest.raises(ValueError):                     # fully out of range
        ec.sections_to_frames([(50.0, 60.0)], 90)


def test_edge_weight_is_zero_at_edges_one_inside():
    w = ec._edge_weight(60, 9)
    assert w[0] == 0.0 and w[-1] == 0.0
    assert np.all(w[15:45] == 1.0)
    assert np.all((w >= 0) & (w <= 1))
    # cosine ramp => no step anywhere
    assert np.abs(np.diff(w)).max() < 0.2


def test_smooth_factor_field_full_depth_and_no_steps():
    tf = np.ones(100)
    tf[50] = 0.85
    f = ec._smooth_factor_field(tf, 6)
    assert f[50] == pytest.approx(0.85)                 # full reduction kept
    assert f[43] == 1.0 and f[57] == 1.0                # bounded support
    assert np.abs(np.diff(f)).max() < 0.05              # never steps


def test_section_edit_touches_only_selected_columns():
    m = make_motion(120)
    m[:, 7] = 0.5 * np.sin(np.linspace(0, 6 * np.pi, 120))   # a "leg" col
    m[:, 22] = 0.3 * np.cos(np.linspace(0, 6 * np.pi, 120))  # an "arm" col
    out = ec.apply_section_edits(m, [(30, 90)], 0.5, 9, [7, 2])
    assert out.shape == m.shape
    assert np.array_equal(out[:, 22], m[:, 22])              # arm untouched
    assert np.array_equal(out[:30], m[:30])                  # outside untouched
    assert np.array_equal(out[91:], m[91:])
    assert np.array_equal(out[30], m[30])                    # boundary frames
    assert np.array_equal(out[90], m[90])
    assert not np.array_equal(out[40:80, 7], m[40:80, 7])    # interior edited


# --------------------------------------------------------- (a) identity round trip

@needs_model
def test_identity_roundtrip_via_cli(tmp_path, motion_csv):
    src = motion_csv(make_motion(45), name="in.csv")
    out = tmp_path / "out.csv"
    rep = tmp_path / "report.json"
    rc = ec.main(["--csv", str(src), "--out", str(out), "--report", str(rep),
                  "--sections", "", "--leg-scale", "0.6", "--lean-cap-nm", "0"])
    assert rc == 0
    assert np.array_equal(np.loadtxt(out, delimiter=","),
                          np.loadtxt(src, delimiter=","))
    report = json.loads(rep.read_text())
    assert report["pass"] is True
    assert report["edit_applied"] is False
    assert report["validation"]["frame_count"]["before"] == 45
    assert report["validation"]["frame_count"]["after"] == 45


# ------------------------------------------------- (b) section edit on the real CSV

@needs_real
def test_real_csv_section_edit(tmp_path):
    args = _args(tmp_path, REAL_CSV, sections="13.0-17.5,43.0-47.0",
                 leg_scale=0.6)
    report, ok = ec.run(args)
    assert ok, f"section edit failed validation: {report['validation']}"

    orig = np.loadtxt(REAL_CSV, delimiter=",")
    edited = np.loadtxt(args.out, delimiter=",")

    # frame count identical (music sync)
    assert edited.shape == orig.shape

    # arms bit-identical
    model = ec.load_model()
    names = ec.joint_names(model)
    arm_cols = [7 + j for j, n in enumerate(names)
                if any(k in n for k in ("shoulder", "elbow", "wrist"))]
    assert len(arm_cols) == 14
    assert np.array_equal(edited[:, arm_cols], orig[:, arm_cols])
    # waist + root XY + root quat untouched too (no lean cap in this run)
    waist_cols = [7 + j for j, n in enumerate(names) if "waist" in n]
    assert np.array_equal(edited[:, waist_cols], orig[:, waist_cols])
    assert np.array_equal(edited[:, 0:2], orig[:, 0:2])
    assert np.array_equal(edited[:, 3:7], orig[:, 3:7])

    # boundary continuity: velocity around every section edge <= original max
    v_orig_max = np.abs(np.diff(orig[:, 7:], axis=0) * 30.0).max()
    for s, e in report["sections_frames"]:
        for f in (s, e):
            edge_v = np.abs(np.diff(edited[f - 1:f + 2, 7:], axis=0) * 30.0)
            assert edge_v.max() <= v_orig_max + 1e-6
    # boundary frames themselves are unchanged
    for s, e in report["sections_frames"]:
        assert np.array_equal(edited[s], orig[s])
        assert np.array_equal(edited[e], orig[e])

    # proxy reduced within each section
    for sec in report["validation"]["proxy"]["sections"]:
        assert sec["after"]["mean_abs_nm"] < sec["before"]["mean_abs_nm"], sec

    # tool-level checks all pass
    v = report["validation"]
    assert v["joint_velocity"]["pass"] and v["foot_height"]["pass"]
    assert v["vet"]["pass"] is True


# ------------------------------------------------------- (c) lean cap on real CSV

@needs_real
def test_real_csv_lean_cap_reduces_over_cap_frames():
    from pipeline.motion_io import load_motion_csv
    model = ec.load_model()
    orig = load_motion_csv(REAL_CSV)
    capped, info = ec.apply_lean_cap(orig, model, 35.0,
                                     ec.pitch_cap_columns(model))
    assert info["applied"] is True
    assert info["frames_over_cap_after"] < info["frames_over_cap_before"]
    # guarded: the cap must never make the worst frame worse
    assert info["max_abs_proxy_after_nm"] <= info["max_abs_proxy_before_nm"] + 0.1
    # residual over-cap frames are reported honestly
    assert len(info["residual_over_cap_frames"]) > 0
    # only the pitch-cap columns (and nothing else) may differ
    diff_cols = np.flatnonzero(np.any(capped != orig, axis=0))
    assert set(diff_cols).issubset(set(info["cap_joint_columns"]))
    # no new velocity spikes from the smoothed scale field
    v0 = np.abs(np.diff(orig[:, 7:], axis=0) * 30.0).max()
    v1 = np.abs(np.diff(capped[:, 7:], axis=0) * 30.0).max()
    assert v1 <= v0 + 1e-6


# -------------------------------------------------- (d) broken edit => FAILED

@needs_model
def test_validation_catches_injected_velocity_spike(tmp_path, motion_csv):
    src = motion_csv(make_motion(60), name="in.csv")

    def corrupt(m):
        m = m.copy()
        m[30, 10] += 1.5        # one-frame joint jump = huge velocity spike
        return m

    report, ok = ec.run(_args(tmp_path, src), _corrupt=corrupt)
    assert ok is False
    assert report["pass"] is False
    assert report["validation"]["joint_velocity"]["pass"] is False
    # the report is still written for post-mortem
    written = json.loads(Path(report["cli_args"]["report"]).read_text())
    assert written["pass"] is False


@needs_model
def test_validation_catches_frame_count_change(tmp_path, motion_csv):
    """A timing change (dropped frames) must fail validation — music sync is
    the tool's core invariant."""
    src = motion_csv(make_motion(60), name="in.csv")
    report, ok = ec.run(_args(tmp_path, src), _corrupt=lambda m: m[:-5])
    assert ok is False
    assert report["validation"]["frame_count"]["pass"] is False
