"""Body-model zip detection/rejection paths on synthetic mini-zips."""
import shutil
import zipfile

import pytest

from .conftest import _zip_with, BIG


def test_status_empty_dir_not_ready(bm_env):
    bm, _ = bm_env
    st = bm.status()
    assert st["ready"] is False
    assert st["zips"] == []
    assert "Drop the SMPL" in st["hint"]


def test_classify_smpl_and_smplx(bm_env, smpl_zip, smplx_zip):
    bm, bmdir = bm_env
    shutil.copy(smpl_zip, bmdir)
    shutil.copy(smplx_zip, bmdir)
    st = bm.status()
    detected = {z["file"]: z["detected"] for z in st["zips"]}
    assert detected[smpl_zip.name] == "smpl"
    assert detected[smplx_zip.name] == "smplx"


def test_install_happy_path(bm_env, smpl_zip, smplx_zip):
    bm, bmdir = bm_env
    shutil.copy(smpl_zip, bmdir)
    shutil.copy(smplx_zip, bmdir)
    report = bm.install()
    assert report["problems"] == []
    assert "smpl/SMPL_NEUTRAL.pkl" in report["installed"]
    assert (bmdir / "smpl" / "SMPL_NEUTRAL.pkl").stat().st_size == BIG
    assert (bmdir / "smplx" / "SMPLX_NEUTRAL.npz").exists()
    assert report["status"]["ready"] is True
    assert (bmdir / "manifest.json").exists()
    # GMR symlink created and points at our smplx dir
    assert bm.GMR_LINK.is_symlink()
    assert bm.GMR_LINK.resolve() == (bmdir / "smplx").resolve()


def test_install_is_idempotent(bm_env, smpl_zip, smplx_zip):
    bm, bmdir = bm_env
    shutil.copy(smpl_zip, bmdir)
    shutil.copy(smplx_zip, bmdir)
    bm.install()
    report2 = bm.install()          # second run must not raise or duplicate
    assert report2["status"]["ready"] is True


def test_v100_wrong_download_explained(bm_env, smpl_v100_zip):
    bm, bmdir = bm_env
    shutil.copy(smpl_v100_zip, bmdir)
    with pytest.raises(RuntimeError, match="v1.0.0"):
        bm.install()
    st = bm.status()
    assert st["ready"] is False


def test_unrecognized_zip_explained(bm_env, tmp_path):
    bm, bmdir = bm_env
    _zip_with(bmdir / "holiday_photos.zip", {"IMG_1234.jpg": 5000})
    with pytest.raises(RuntimeError, match="right download"):
        bm.install()


def test_truncated_model_flagged_as_corrupt(bm_env):
    bm, bmdir = bm_env
    _zip_with(bmdir / "SMPL_python_v.1.1.0.zip", {
        "smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl": 1000,  # tiny
        "smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl": BIG,
        "smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl": BIG,
    })
    with pytest.raises(RuntimeError, match="corrupt"):
        bm.install()


def test_bad_zipfile_is_unrecognized_not_crash(bm_env):
    bm, bmdir = bm_env
    (bmdir / "broken.zip").write_bytes(b"PK\x03\x04 not really a zip")
    st = bm.status()
    assert st["zips"][0]["detected"] == "unrecognized"
    with pytest.raises(RuntimeError):
        bm.install()


def test_good_zip_installs_despite_bad_neighbor(bm_env, smpl_zip, smplx_zip):
    bm, bmdir = bm_env
    shutil.copy(smpl_zip, bmdir)
    shutil.copy(smplx_zip, bmdir)
    (bmdir / "junk.zip").write_bytes(b"garbage")
    report = bm.install()           # must not raise: something installed
    assert any("junk.zip" in p for p in report["problems"])
    assert report["status"]["ready"] is True
