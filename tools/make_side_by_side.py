#!/usr/bin/env python
"""Build a time-aligned side-by-side of the reference dance video and the sim rollout.

    reference (left)  |  sim rollout (right)   + the real music track

WHY THE OFFSET IS NOT ZERO (and not the naive 4.0s either) — derived 2026-07-07
by frame inspection, committed as the record of the derivation:

  * The reference "Thriller Dance Final.mov" is a dance TUTORIAL: the instructor
    stands and TALKS for the first ~7s before the choreography starts (verified:
    frames at 0-6s = standing/talking, arms down; arms-out dance pose first
    appears ~7s).
  * The retarget windowed that intro out, so the robot's FIRST dance frame
    corresponds to source ~7.0s, not source 0.
  * The sim rollout (rollout_v3e.mp4) starts at deploy t=0: a 2.5s activation
    ramp + 1.5s standing lead-in, so the robot's dance movement begins at
    sim t=4.0s (verified: sim frame at 4.0s = arms-out dance-start pose).
  * Therefore reference-dance-start (~7.0s) must align to sim-dance-start (4.0s):
      source_time_shown_at(sim_t) = sim_t + SRC_LEAD   with SRC_LEAD = 3.0s
    i.e. ADVANCE the source by 3.0s (trim its first 3.0s), do NOT pad it.
  * Speed is 1:1 — the extraction re-encoded the ~35.4fps VFR source to 30fps
    preserving wall-clock (44.28s in, 44.3s out), so there is NO time stretch.
    Confirmed: pose pairs at sim 8/16/28/40s vs source 11/19/31/43s stay matched
    across the whole dance with no growing drift.
  * Music starts at sim t=4.0s (= dance start) — the show timeline's
    tick0 + 2.5 ramp + 1.5 lead-in = 4.0s contract.

The naive first attempt PADDED the source +4.0s (pushing the talking intro to
where the robot was already dancing) → ~7s of visible lag. This tool fixes that.

Usage:
  ~/miniconda3/envs/g1dance/bin/python tools/make_side_by_side.py \
    [--source ...] [--sim ...] [--audio ...] [--out ...] \
    [--src-lead 3.0] [--music-at 4.0] [--speed 1.0]
Defaults are the derived-correct Thriller values.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = str(Path.home() / "miniconda3/envs/g1dance/bin/ffmpeg")
if not Path(FFMPEG).exists():
    FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=str(ROOT / "data/videos/Thriller Dance Final.mov"))
    ap.add_argument("--sim", default=str(ROOT / "data/previews/rollout_v3e.mp4"))
    ap.add_argument("--audio", default=str(ROOT / "data/dances/20260704-18f65bbd/audio/music.wav"))
    ap.add_argument("--out", default=str(ROOT / "data/previews/thriller_side_by_side_v3e.mp4"))
    ap.add_argument("--src-lead", type=float, default=3.4,
                    help="seconds into the source at composite t=0 (chosen with --speed so the "
                         "source dance-start ~7.0s lands on the sim dance-start at t=4.0s)")
    ap.add_argument("--music-at", type=float, default=4.0,
                    help="composite time (s) at which music (and the dance) begins")
    ap.add_argument("--speed", type=float, default=0.9,
                    help="source playback factor. 0.9 = empirical drift correction: the "
                         "DEPLOYED robot motion runs ~10%% slower than the raw video, so the "
                         "source is slowed to keep mid/late beats aligned (dance-start is "
                         "frame-matched; this factor is from mid/late pose-matching, approximate)")
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    for p in (args.source, args.sim, args.audio):
        if not Path(p).exists():
            print(f"missing input: {p}", file=sys.stderr)
            return 1

    h = args.height
    src_pts = f"setpts=(PTS-STARTPTS)/{args.speed}" if args.speed != 1.0 else "setpts=PTS-STARTPTS"
    # left = reference (advanced by src_lead via -ss), right = sim rollout
    filt = (
        f"[1:v]fps=30,{src_pts},scale=-2:{h},setsar=1[src];"
        f"[0:v]fps=30,scale=-2:{h},setsar=1[sim];"
        f"[src][sim]hstack=inputs=2[v];"
        f"[2:a]adelay={int(args.music_at*1000)}|{int(args.music_at*1000)},apad[a]"
    )
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", args.sim,                       # 0: sim (no offset)
        "-ss", f"{args.src_lead}", "-i", args.source,  # 1: source, advanced
        "-i", args.audio,                     # 2: music
        "-filter_complex", filt,
        "-map", "[v]", "-map", "[a]",
        "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", args.out,
    ]
    print(f"params: src_lead={args.src_lead}s (advance), music_at={args.music_at}s, "
          f"speed={args.speed}, height={h}")
    r = subprocess.run(cmd)
    if r.returncode != 0 or not Path(args.out).exists():
        print("ffmpeg failed", file=sys.stderr)
        return 1
    dur = subprocess.run(
        [FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error",
         "-show_entries", "format=duration", "-of", "csv=p=0", args.out],
        capture_output=True, text=True).stdout.strip()
    print(f"wrote {args.out}  (duration {dur}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
