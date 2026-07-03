"""Find the longest contiguous window of a G1 motion CSV that fits a venue.

A window fits if its root-XY trajectory's MINIMAL ENCLOSING CIRCLE has radius
<= max_excursion_m (i.e. the whole window fits the venue when the robot is
placed at that circle's centre), and the pelvis never drops below the floorwork
limit. This is the enclosing-circle upgrade over the old "distance from the
window's first frame": a dance that circles a point off to one side is no longer
penalised for merely starting off-centre — what matters is the space it needs.

`max_excursion_m` defaults to 1.5 m (the historical 2 m-radius area minus a
0.5 m safety margin) so existing callers are unchanged; pass a venue's
`max_excursion_m` (see pipeline.venue) for other spaces.

Usage: python find_window.py motion.csv [--out seg.csv] [--min-seconds 20]
                                        [--max-excursion 1.5]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

CSV_FPS = 30.0
MAX_EXCURSION_M = 1.5
MIN_PELVIS_HEIGHT_M = 0.35


def _mec(xy):
    # local import so this module has no hard dependency at import time
    from pipeline.venue import minimal_enclosing_circle
    return minimal_enclosing_circle(xy)


def window_center(m, s, e):
    """(cx, cy) enclosing-circle centre of the window's root-XY — the point the
    robot should be placed at so the window fits the venue with maximum margin."""
    c, _ = _mec(np.asarray(m)[s:e + 1, 0:2])
    return float(c[0]), float(c[1])


def longest_window(m, max_excursion_m=MAX_EXCURSION_M):
    """Longest contiguous window whose root-XY minimal enclosing circle has
    radius <= max_excursion_m and whose pelvis never drops below the floorwork
    limit. Returns (start, end) inclusive frame indices.

    NOTE: the z test is absolute (floor at z=0), so the caller must pass a
    GROUNDED motion (see pipeline.grounding). Enclosing radius is translation-
    invariant, so recentering does not affect the result.

    The window growth is greedy (each search jumps past the frame that broke the
    bound); this matches the historical behaviour and is intentionally not the
    globally-optimal window (a known, accepted limitation)."""
    xy = np.asarray(m)[:, 0:2]
    z_ok = np.asarray(m)[:, 2] >= MIN_PELVIS_HEIGHT_M
    n = len(m)
    best = (0, 0)
    start = 0
    while start < n:
        if not z_ok[start]:
            start += 1
            continue
        centre = xy[start].astype(float)
        r = 0.0
        end = start
        while end + 1 < n and z_ok[end + 1]:
            p = xy[end + 1]
            if np.linalg.norm(p - centre) <= r + 1e-9:
                cand_c, cand_r = centre, r          # already inside — MEC unchanged
            else:
                cand_c, cand_r = _mec(xy[start:end + 2])
            if cand_r <= max_excursion_m + 1e-9:
                centre, r, end = cand_c, cand_r, end + 1
            else:
                break
        if end - start > best[1] - best[0]:
            best = (start, end)
        start = end + 1
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", help="write the windowed segment as a new CSV")
    ap.add_argument("--min-seconds", type=float, default=20.0)
    ap.add_argument("--max-excursion", type=float,
                    default=float(os.environ.get("G1_MAX_EXCURSION_M",
                                                 MAX_EXCURSION_M)),
                    help="venue's max root excursion in metres (default 1.5)")
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
    s, e = longest_window(m, max_excursion_m=args.max_excursion)
    dur = (e - s + 1) / CSV_FPS
    print(f"{args.csv}: best window frames {s}..{e} = {dur:.1f}s "
          f"(of {len(m)/CSV_FPS:.1f}s total, venue max {args.max_excursion} m)")
    if dur < args.min_seconds:
        print(f"WARNING: shorter than --min-seconds {args.min_seconds}")
    if args.out:
        seg = m[s:e + 1].copy()
        cx, cy = window_center(seg, 0, len(seg) - 1)
        seg[:, 0] -= cx      # re-centre XY on the enclosing-circle centre so the
        seg[:, 1] -= cy      # dance sits centred in the venue (max margin)
        np.savetxt(args.out, seg, delimiter=",", fmt="%.6f")
        print(f"wrote {args.out} ({len(seg)} frames, XY re-centred on footprint "
              f"centre ({cx:.3f}, {cy:.3f}))")


if __name__ == "__main__":
    main()
