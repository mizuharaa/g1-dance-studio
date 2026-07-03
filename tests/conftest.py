"""Shared fixtures: synthetic G1 motions, body-model zips, video fixtures.

Everything runs headless — no robot, no cloud box, no GPU. The MuJoCo-based
tests need third_party/mujoco_menagerie (a symlink to the main checkout is
fine); they self-skip when it is absent.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

WORKTREE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE))

MODEL_XML = WORKTREE / "third_party/mujoco_menagerie/unitree_g1/scene.xml"
HAVE_MODEL = MODEL_XML.exists()

# CSV convention (LAFAN1 / project-wide): 36 columns =
#   0:3 root xyz | 3:7 root quat (xyzw) | 7:36 the 29 joint angles
N_COLS = 36
STAND_Z = 0.79


def make_motion(frames: int = 15, *, z: float = STAND_Z,
                drift_xy: tuple[float, float] = (0.0, 0.0),
                joint_overrides: dict[int, float] | None = None) -> np.ndarray:
    """A standing-still motion: identity orientation, zero joints.

    drift_xy is the total XY translation reached linearly by the last frame.
    joint_overrides sets joint column j (0..28) to a constant value.
    """
    m = np.zeros((frames, N_COLS))
    t = np.linspace(0.0, 1.0, frames)
    m[:, 0] = drift_xy[0] * t
    m[:, 1] = drift_xy[1] * t
    m[:, 2] = z
    m[:, 6] = 1.0  # quat w (stored last in xyzw)
    for j, val in (joint_overrides or {}).items():
        m[:, 7 + j] = val
    return m


@pytest.fixture
def motion_csv(tmp_path):
    """Factory: write a motion array (or make_motion kwargs) to a CSV path."""
    def _write(motion: np.ndarray | None = None, name: str = "motion.csv",
               **kwargs) -> Path:
        if motion is None:
            motion = make_motion(**kwargs)
        p = tmp_path / name
        np.savetxt(p, motion, delimiter=",", fmt="%.6f")
        return p
    return _write


def run_vet(csv_path: Path) -> tuple[int, dict]:
    """Run the vet gate CLI with --json; returns (exit_code, report)."""
    import json
    proc = subprocess.run(
        [sys.executable, str(WORKTREE / "pipeline/vet_motion.py"),
         str(csv_path), "--json"],
        capture_output=True, text=True, timeout=120, cwd=WORKTREE)
    assert proc.stdout, f"vet produced no output; stderr: {proc.stderr[-400:]}"
    return proc.returncode, json.loads(proc.stdout)


# ---- body-model zip builders ---------------------------------------------------

def _zip_with(path: Path, members: dict[str, int]) -> Path:
    """Create a zip whose members have the given uncompressed sizes."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, size in members.items():
            z.writestr(name, b"\0" * size)
    return path


BIG = 1_200_000  # comfortably over body_models.MIN_BYTES


@pytest.fixture
def smpl_zip(tmp_path):
    """A structurally valid SMPL v1.1.0 zip (neutral + male + female)."""
    return _zip_with(tmp_path / "SMPL_python_v.1.1.0.zip", {
        "SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl": BIG,
        "SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl": BIG,
        "SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl": BIG,
    })


@pytest.fixture
def smpl_v100_zip(tmp_path):
    """The wrong download: v1.0.0 has male/female but no neutral model."""
    return _zip_with(tmp_path / "SMPL_python_v.1.0.0.zip", {
        "smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl": BIG,
        "smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl": BIG,
    })


@pytest.fixture
def smplx_zip(tmp_path):
    return _zip_with(tmp_path / "models_smplx_v1_1.zip", {
        f"models/smplx/SMPLX_{g}.{ext}": BIG
        for g in ("NEUTRAL", "MALE", "FEMALE") for ext in ("npz", "pkl")
    })


@pytest.fixture
def bm_env(tmp_path, monkeypatch):
    """Point pipeline.body_models at an isolated directory tree."""
    from pipeline import body_models as bm
    bmdir = tmp_path / "body_models"
    bmdir.mkdir()
    monkeypatch.setattr(bm, "BM_DIR", bmdir)
    monkeypatch.setattr(bm, "MANIFEST", bmdir / "manifest.json")
    monkeypatch.setattr(bm, "GMR_LINK", tmp_path / "gmr" / "smplx")
    return bm, bmdir


# ---- app / store isolation ------------------------------------------------------

@pytest.fixture
def jobs_env(tmp_path, monkeypatch):
    """Isolate the job store in a temp directory."""
    from pipeline import store
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr(store, "JOBS_DIR", jobs_dir)
    return store, jobs_dir


@pytest.fixture
def client(tmp_path, monkeypatch, jobs_env):
    """TestClient over ui.server with the runner stubbed out (no real stage
    execution) and the store isolated. Yields (client, server_module)."""
    from starlette.testclient import TestClient

    from ui import server
    monkeypatch.setattr(server._runner, "run_job",
                        lambda job, until=None: None)
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    (tmp_path / "videos").mkdir(exist_ok=True)
    with TestClient(server.app) as c:
        yield c, server
