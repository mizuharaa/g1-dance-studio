"""ffprobe-based intake validation for reference dance videos.

Runs locally the moment a video job is created, so obviously-unusable footage
fails in seconds with a human-readable reason instead of after a cloud round
trip. Limits mirror the product constraints (PROJECT_STATE.md):

  hard:      readable file with a video stream; duration 15 s .. 4 min
  advisory:  resolution >= 1280x720; variable frame rate (VFR) warning
"""
from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

MIN_SECONDS = 15
MAX_SECONDS = 240
ADVISORY_MIN_HEIGHT = 720
# Sane aspect band: 9:16 portrait (0.56) to 21:9 ultrawide (2.33), with margin.
MIN_ASPECT = 0.4
MAX_ASPECT = 2.5


def probe(path: Path) -> dict:
    """Raw ffprobe metadata (format + streams). Raises RuntimeError if the
    file is not readable video."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError("ffprobe cannot read this file — not a valid "
                           f"video? ({proc.stderr.strip()[-200:]})")
    return json.loads(proc.stdout)


def _fps(stream: dict, key: str) -> float:
    try:
        return float(Fraction(stream.get(key, "0/1")))
    except (ValueError, ZeroDivisionError):
        return 0.0


def validate(path: Path) -> dict:
    """Validate a reference video. Returns a metadata dict with 'advisories';
    raises RuntimeError with a human-readable reason when unusable."""
    meta = probe(path)
    video = next((s for s in meta.get("streams", [])
                  if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError("file contains no video stream")

    duration = float(meta.get("format", {}).get("duration")
                     or video.get("duration") or 0)
    if duration <= 0:  # don't misreport a duration-less/corrupt file as "0.0s too short"
        raise RuntimeError(
            "could not read the video's duration — the file may be corrupt or "
            "truncated; try re-exporting it")
    if duration < MIN_SECONDS:
        raise RuntimeError(
            f"video is {duration:.1f}s — too short, the pipeline needs at "
            f"least {MIN_SECONDS}s of continuous dance")
    # +5 s tolerance: a keyframe-copy trim of a "4:00" segment lands ~240.2 s, which must not
    # fail this gate (the built-in trimmer already holds the user to <=4 min).
    if duration > MAX_SECONDS + 5:
        raise RuntimeError(
            f"video is {duration / 60:.1f} min — longer than the current "
            f"{MAX_SECONDS // 60} min limit; trim it to the segment you want "
            "the robot to learn")

    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    r_fps, avg_fps = _fps(video, "r_frame_rate"), _fps(video, "avg_frame_rate")

    # HARD: degenerate or extreme geometry is unusable footage — reject locally in
    # seconds rather than burning a paid cloud cycle (audit MEDIUM). The old advisory
    # used AND so 1920x400 / 4000x100 slipped through with no signal at all.
    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"video has invalid dimensions ({width}x{height}) — unreadable geometry")
    aspect = width / height
    if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
        raise RuntimeError(
            f"video aspect ratio {aspect:.2f} ({width}x{height}) is extreme — a "
            "full-body dance needs a roughly normal frame (portrait to landscape), "
            "not a sliver")

    advisories = []
    if height < ADVISORY_MIN_HEIGHT or width < 1280:  # EITHER dim small = advise
        advisories.append(f"resolution {width}x{height} is below 720p — pose "
                          "extraction quality may suffer")
    if r_fps and avg_fps and abs(r_fps - avg_fps) / max(r_fps, avg_fps) > 0.01:
        advisories.append(
            f"variable frame rate detected ({avg_fps:.2f} avg vs {r_fps:.2f} "
            "declared) — timing may drift; a constant-frame-rate re-encode "
            "will happen before extraction")

    return {
        "duration_s": round(duration, 2),
        "width": width, "height": height,
        "fps": round(avg_fps or r_fps, 3),
        "codec": video.get("codec_name"),
        "size_bytes": path.stat().st_size,
        "advisories": advisories,
    }
