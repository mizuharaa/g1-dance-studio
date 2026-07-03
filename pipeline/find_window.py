"""Find the longest contiguous window of a G1 motion CSV that fits the dance area.

A window passes if root XY stays within MAX_EXCURSION of the window's first frame
and pelvis never drops below the floorwork limit. Useful for turning a traveling
LAFAN1 dance into a deployable segment for the 2 m-radius area.

Usage: python find_window.py motion.csv [--out segment.csv] [--min-seconds 20]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

CSV_FPS = 30.0
MAX_EXCURSION_M = 1.5
MIN_PELVIS_HEIGHT_M = 0.35


def longest_window(m):
    """Longest contiguous window whose root XY stays within MAX_EXCURSION of the
    window start and whose pelvis never drops below the floorwork limit.

    NOTE: the z test is absolute (floor at z=0), so the caller must pass a
    GROUNDED motion (see pipeline.grounding). The retarget stage and this
    module's CLI ground before calling; callers with raw retarget output must
    ground first or the floorwork check is meaningless (audit HIGH)."""
    xy = m[:, 0:2]
    z_ok = m[:, 2] >= MIN_PELVIS_HEIGHT_M
    best = (0, 0)
    start = 0
    while start < len(m):
        if not z_ok[start]:
            start += 1
            continue
        # grow end while constraints hold relative to this start
        end = start
        while end + 1 < len(m) and z_ok[end + 1] and \
                np.linalg.norm(xy[end + 1] - xy[start]) <= MAX_EXCURSION_M:
            end += 1
        if end - start > best[1] - best[0]:
            best = (start, end)
        # next candidate start: first frame that broke the excursion bound
        start = end + 1
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", help="write the windowed segment as a new CSV")
    ap.add_argument("--min-seconds", type=float, default=20.0)
    args = ap.parse_args()

    # Runs as a standalone script too — import via the absolute package so a
    # relative import doesn't blow up the CLI.
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from pipeline.grounding import ground_motion, have_model
    from pipeline.motion_io import load_motion_csv
    m = load_motion_csv(args.csv)
    if have_model():
        m, _ = ground_motion(m)  # window's z test is absolute — ground first
    s, e = longest_window(m)
    dur = (e - s + 1) / CSV_FPS
    print(f"{args.csv}: best window frames {s}..{e} = {dur:.1f}s "
          f"(of {len(m)/CSV_FPS:.1f}s total)")
    if dur < args.min_seconds:
        print(f"WARNING: shorter than --min-seconds {args.min_seconds}")
    if args.out:
        seg = m[s:e + 1].copy()
        seg[:, 0:2] -= seg[0, 0:2]  # re-center XY on the window start
        np.savetxt(args.out, seg, delimiter=",", fmt="%.6f")
        print(f"wrote {args.out} ({len(seg)} frames, XY re-centered)")


if __name__ == "__main__":
    main()
