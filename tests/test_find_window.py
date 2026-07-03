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
    # 60 frames drifting to 4 m: the whole thing has footprint radius 2 m, which
    # overflows 1.5 m, so the best window is a proper sub-range. Validity is now
    # the window's enclosing-circle radius (not span from the window start).
    from pipeline.venue import minimal_enclosing_circle
    m = make_motion(frames=60, drift_xy=(4.0, 0.0))
    s, e = longest_window(m)
    _, r = minimal_enclosing_circle(m[s:e + 1, 0:2])
    assert r <= 1.5 + 1e-6
    assert (e - s) < 59
    # a straight walk can now use the full diameter: ~3 m of travel fits 1.5 m
    span = np.linalg.norm(m[e, 0:2] - m[s, 0:2])
    assert span > 1.5


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
    # XY is now re-centred on the window's FOOTPRINT (enclosing-circle) centre, not
    # the first frame, so the dance sits centred in the venue with maximum margin.
    # For this straight 1.0x0.5 m drift the centre is the midpoint => first frame
    # lands at -half the travel.
    from pipeline.venue import minimal_enclosing_circle
    c, _ = minimal_enclosing_circle(seg[:, 0:2])
    assert np.allclose(c, [0.0, 0.0], atol=1e-5)     # footprint centred at origin
    assert np.allclose(seg[0, 0:2], [-0.5, -0.25], atol=1e-5)
    # z is GROUNDED (audit HIGH fix): floor-referenced standing height.
    assert 0.3 < seg[0, 2] < 1.2
    assert len(seg) == 45
    # relative XY trajectory preserved (endpoint minus start = the full drift)
    assert np.allclose(seg[-1, 0:2] - seg[0, 0:2], [1.0, 0.5], atol=1e-5)
