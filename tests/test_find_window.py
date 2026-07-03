"""Window selection + XY re-centering (pure numpy + the CLI --out path)."""
import subprocess
import sys

import numpy as np

from .conftest import WORKTREE, make_motion

from pipeline.find_window import longest_window  # noqa: E402

SCRIPT = WORKTREE / "pipeline/find_window.py"


def test_whole_motion_when_everything_fits():
    m = make_motion(frames=30, drift_xy=(1.0, 0.0))
    assert longest_window(m) == (0, 29)


def test_floorwork_frame_splits_and_longer_side_wins():
    m = make_motion(frames=30)
    m[10, 2] = 0.10  # one floorwork frame
    s, e = longest_window(m)
    assert (s, e) == (11, 29)  # 19 frames beat the 10 before the dip
    assert 10 not in range(s, e + 1)


def test_excursion_break_starts_new_window():
    # 60 frames drifting to 4 m: no window from frame 0 can hold 1.5 m,
    # so the best window must be a proper sub-range re-anchored mid-motion.
    m = make_motion(frames=60, drift_xy=(4.0, 0.0))
    s, e = longest_window(m)
    xy = m[:, 0:2]
    span = np.linalg.norm(xy[s:e + 1] - xy[s], axis=1).max()
    assert span <= 1.5
    assert (e - s) < 59


def test_all_floorwork_returns_empty_window():
    m = make_motion(frames=10, z=0.1)
    s, e = longest_window(m)
    assert e - s == 0


def test_cli_out_recenters_xy(tmp_path):
    m = make_motion(frames=45, drift_xy=(1.0, 0.5))
    m[:, 0] += 7.0   # start far from origin
    m[:, 1] -= 3.0
    src = tmp_path / "in.csv"
    out = tmp_path / "seg.csv"
    np.savetxt(src, m, delimiter=",", fmt="%.6f")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(src), "--out", str(out),
         "--min-seconds", "1"],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    seg = np.loadtxt(out, delimiter=",")
    assert seg[0, 0] == 0.0 and seg[0, 1] == 0.0     # re-centered
    assert seg[0, 2] == m[0, 2]                       # z untouched
    assert len(seg) == 45
    # relative trajectory preserved
    assert np.allclose(seg[-1, 0:2], [1.0, 0.5], atol=1e-5)
