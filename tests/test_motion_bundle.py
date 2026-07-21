"""Audit F2 regression tests: hash-bound, deterministic motion-bundle build.

tools/build_motion_bundle.py exists because the v12 training CSV matched NO
retained scorecard (deleted /tmp source, post-scorecard rewrite, stale
feasibility numbers) and the launcher checked only existence + frame count.
These tests pin the fix on a small synthetic motion (no thriller/data
dependency, real G1 model):

  * one build -> verify_manifest returns [] and the scorecard certifies the
    final BYTES (sha in scorecard == sha of file == sha in manifest)
  * two builds -> final.csv and scorecard.json byte-identical (deterministic;
    bundle.json alone carries created_at and is excluded)
  * tampered final.csv -> the launcher-style (hashlib-only, no pipeline/)
    verification fails, as does pipeline.artifacts.verify_manifest
  * edited manifest -> content-addressed bundle_id mismatch is detected
  * artificially blunted clean (monkeypatched clean_motion) -> fidelity gate
    hard-fails, NO bundle.json is left behind (stale one removed)
  * over-envelope feasibility summary (monkeypatched analyze) -> build fails
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pipeline.artifacts import sha256_file, verify_manifest
from pipeline.g1_limits import MODEL_XML

pytestmark = pytest.mark.skipif(not MODEL_XML.exists(),
                                reason="needs the G1 mujoco model")

FPS = 30.0
N_FRAMES = 120


def _synthetic_motion() -> np.ndarray:
    """Standing motion with a gentle 1 Hz arm swing: moving joints (~0.6 rad
    peak-to-peak, ~1.9 rad/s) that survive guard-clean, soles ON the floor from
    frame 0 (grounded seed), and torque demand far inside the envelope."""
    from pipeline.g1_limits import build_model
    from pipeline.grounding import _fk_heights
    m = np.zeros((N_FRAMES, 36))
    m[:, 6] = 1.0                       # quat xyzw identity
    t = np.arange(N_FRAMES) / FPS
    for j in (15, 16, 22, 23):          # shoulder joints, both arms
        m[:, 7 + j] = 0.3 * np.sin(2 * np.pi * 1.0 * t)
    m[:, 7 + 12] = 0.1 * np.sin(2 * np.pi * 0.5 * t)  # waist yaw sway
    m[:, 2] = 1.0
    sole = float(_fk_heights(m[:1], build_model())[1].min())
    m[:, 2] = 1.0 - sole
    return m


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    import tools.build_motion_bundle as bmb
    d = tmp_path_factory.mktemp("bundle_src")
    src_csv = d / "src.csv"
    np.savetxt(src_csv, _synthetic_motion(), delimiter=",")
    out = d / "out"
    bmb.build_bundle(src_csv, out)
    return src_csv, out


def _launcher_verify(bundle_dir: Path, csv: Path) -> list[str]:
    """Mirror of the self-contained hashlib walker embedded in
    cloud/run_attempt9.sh (boxes have no pipeline/, so the launcher re-hashes
    {path,sha256} members and recomputes the content-addressed bundle_id with
    hashlib/json alone). Semantics must match that heredoc."""
    m = json.loads((bundle_dir / "bundle.json").read_text())
    errs: list[str] = []
    body = {k: v for k, v in m.items() if k != "bundle_id"}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if m.get("bundle_id") != digest:
        errs.append("bundle_id does not match manifest content")

    def walk(node, crumb):
        if isinstance(node, dict):
            if "path" in node and "sha256" in node:
                f = Path(node["path"])
                if not f.is_absolute():
                    f = bundle_dir / f
                if not f.exists():
                    errs.append(f"{crumb}: missing file {node['path']}")
                elif sha256_file(f) != node["sha256"]:
                    errs.append(f"{crumb}: sha mismatch for {node['path']}")
            for k, v in node.items():
                walk(v, f"{crumb}.{k}" if crumb else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{crumb}[{i}]")

    walk(m, "")
    fin = m.get("motion", {}).get("final_csv") or {}
    if not csv.is_file() or sha256_file(csv) != fin.get("sha256"):
        errs.append("training CSV is not the manifest's final_csv")
    return errs


# ---- clean build ----------------------------------------------------------------

def test_manifest_verifies_clean(built):
    _, out = built
    assert verify_manifest(out / "bundle.json") == []
    motion = json.loads((out / "bundle.json").read_text())["motion"]
    for member in ("source_csv", "final_csv", "scorecard"):
        assert set(motion[member]) == {"path", "sha256"}
    assert motion["grounding"]["flight_aware"] is True
    assert "params" in motion["grounding"]
    # launcher-style verification agrees with pipeline.artifacts
    assert _launcher_verify(out, out / "final.csv") == []


def test_scorecard_certifies_final_bytes(built):
    src_csv, out = built
    score = json.loads((out / "scorecard.json").read_text())
    manifest = json.loads((out / "bundle.json").read_text())
    assert score["final"]["sha256"] == sha256_file(out / "final.csv")
    assert score["final"]["sha256"] == manifest["motion"]["final_csv"]["sha256"]
    assert score["source"]["sha256"] == sha256_file(src_csv)
    # the copied source is byte-identical to the immutable input
    assert sha256_file(out / "source.csv") == sha256_file(src_csv)
    assert score["gates"]["failures"] == []
    # retention table is per joint, with per-window worst rows for moving joints
    rows = score["fidelity_retention"]["per_joint"]
    assert len(rows) == 29
    moving = [r for r in rows if r["moving"]]
    assert moving, "synthetic motion must have moving joints"
    for r in moving:
        assert r["amp_retention"] > 0.75
        assert r["worst_window_amp"] is not None
    # grounding outputs recorded (audit F2: honest numbers in the scorecard)
    for key in ("flight_windows_s", "floor_drift_m", "support_pct",
                "grounded_start"):
        assert key in score["grounding"]["info"]


def test_build_is_deterministic(built, tmp_path):
    src_csv, out = built
    import tools.build_motion_bundle as bmb
    out2 = tmp_path / "out2"
    bmb.build_bundle(src_csv, out2)
    assert (out / "final.csv").read_bytes() == (out2 / "final.csv").read_bytes()
    assert (out / "scorecard.json").read_bytes() == \
        (out2 / "scorecard.json").read_bytes()


# ---- tamper detection -----------------------------------------------------------

def test_tampered_final_csv_refused(built, tmp_path):
    _, out = built
    box = tmp_path / "pushed"
    shutil.copytree(out, box)
    with open(box / "final.csv", "ab") as f:
        f.write(b"0")
    errs = _launcher_verify(box, box / "final.csv")
    assert any("final_csv" in e or "final.csv" in e for e in errs)
    assert verify_manifest(box / "bundle.json") != []


def test_edited_manifest_refused(built, tmp_path):
    _, out = built
    box = tmp_path / "pushed"
    shutil.copytree(out, box)
    m = json.loads((box / "bundle.json").read_text())
    m["motion"]["grounding"]["flight_aware"] = False   # any content edit
    (box / "bundle.json").write_text(json.dumps(m, indent=2, sort_keys=True))
    errs = _launcher_verify(box, box / "final.csv")
    assert any("bundle_id" in e for e in errs)


def test_wrong_training_csv_refused(built, tmp_path):
    """G1_MOTION_CSV pointing at bytes other than the manifest's final_csv must
    fail even when the bundle itself is intact."""
    src_csv, out = built
    errs = _launcher_verify(out, src_csv)
    assert any("final_csv" in e for e in errs)


# ---- gates fail closed ----------------------------------------------------------

def test_fidelity_gate_trips_on_blunted_clean(built, monkeypatch, tmp_path):
    src_csv, _ = built
    import tools.build_motion_bundle as bmb

    def blunted_clean(motion, fps=FPS):
        out = motion.copy()
        j = out[:, 7:]
        out[:, 7:] = j.mean(axis=0) + 0.5 * (j - j.mean(axis=0))
        info = {"outlier_frames_replaced": 0, "jerk_p99_before": 0.0,
                "jerk_p99_after": 0.0, "dof_rms_delta_rad": 0.0}
        return out, info

    monkeypatch.setattr(bmb.motion_quality, "clean_motion", blunted_clean)
    out = tmp_path / "out"
    out.mkdir()
    (out / "bundle.json").write_text("{}")    # stale manifest must be removed
    with pytest.raises(SystemExit) as exc:
        bmb.build_bundle(src_csv, out)
    assert exc.value.code == 2
    assert not (out / "bundle.json").exists()
    score = json.loads((out / "scorecard.json").read_text())
    fails = score["gates"]["failures"]
    assert any("retention" in f for f in fails)


def test_feasibility_gate_trips(built, monkeypatch, tmp_path):
    src_csv, _ = built
    import tools.build_motion_bundle as bmb

    real_analyze = bmb.motion_dynamics.analyze

    def hot_analyze(csv_path, fps=FPS, ground=True):
        res = real_analyze(csv_path, fps=fps, ground=ground)
        res["dynamic"]["any_joint_frames_over_pct"] = 5.0
        return res

    monkeypatch.setattr(bmb.motion_dynamics, "analyze", hot_analyze)
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        bmb.build_bundle(src_csv, out)
    assert exc.value.code == 2
    assert not (out / "bundle.json").exists()
    score = json.loads((out / "scorecard.json").read_text())
    assert any("feasibility" in f for f in score["gates"]["failures"])


# ---- launcher wiring ------------------------------------------------------------

def test_run_attempt9_wiring():
    """The launcher must default to the bundle CSV and verify the manifest
    BEFORE any retime/convert step (audit F2: no training on unverified bytes)."""
    text = (Path(__file__).resolve().parent.parent
            / "cloud" / "run_attempt9.sh").read_text()
    assert "motions/v12_bundle/final.csv" in text
    assert text.index("verify motion bundle manifest") \
        < text.index("retime tempo variants")
    assert "bundle_realized.json" in text
    # stale-npz guard: tempo npz are re-generated when the source sha changes
    assert "thriller_v12_tempo.src.sha256" in text
