#!/usr/bin/env python
"""Audio/music sync for dance shows.

A dance performance needs its music, and the music must stay locked to the
robot's motion through the whole pipeline. This module owns three jobs:

  1. INGEST — get an audio track for a dance, either extracted from the source
     video (if it has an audio stream) or supplied as a separate music file.
  2. ALIGN — compute how to trim/offset that track so it lines up with the
     FINAL prepped motion timeline. This is the load-bearing part: prep_motion
     prepends a standing pad + blend-in before the dance starts, so the music
     must be delayed by exactly that much or every beat lands early.
  3. MUX — lay the aligned audio onto a (silent) MuJoCo preview MP4 so the
     preview actually plays with music — a concrete, watchable result — and,
     at show time, define when playback starts relative to the performance.

Timing model (see docs/audio_sync_design.md for the full derivation):

    prepped motion timeline (what the robot performs):
    | PAD_IN | BLEND_IN |          DANCE (music)          | BLEND_OUT | HOLD_OUT |
    0       1.0s      1.5s                              45.8s       46.8s     49.3s
                       ^ music starts here                ^ music ends
    audio_delay_s   = PAD_IN_S + BLEND_IN_S            (1.5s for the default prep)
    audio_trim      = [window_start_s, +dance_duration_s]  (source-relative)

Real-time duration is preserved end to end (30fps retarget -> 50fps train ->
50Hz deploy all keep wall-clock seconds), so alignment computed in seconds is
valid from preview through real-robot playback.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline.config import PROJECT_ROOT

# Must match pipeline/prep_motion.py. If prep constants change, an emitted
# prep-info dict (preferred) overrides these; they are the documented default.
PREP_FPS = 30
PREP_PAD_IN_S = 1.0
PREP_BLEND_IN_S = 0.5
PREP_BLEND_OUT_S = 1.0
PREP_HOLD_OUT_S = 2.5


@dataclass
class AudioAlignment:
    """How to place a source music track onto the performance timeline."""

    audio_delay_s: float      # music starts this many s into the performance
    trim_start_s: float       # take source audio from here...
    trim_duration_s: float    # ...for this long (the danced span)
    music_end_s: float        # when music stops in the performance timeline
    performance_s: float      # full prepped-motion duration (video length)

    def to_dict(self) -> dict:
        return asdict(self)


def _ffprobe(video: Path, args: list[str]) -> str:
    return subprocess.run(
        ["ffprobe", "-v", "error", *args, str(video)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def has_audio(video: Path) -> bool:
    """True if the media file carries at least one audio stream."""
    out = _ffprobe(video, ["-select_streams", "a",
                           "-show_entries", "stream=index",
                           "-of", "csv=p=0"])
    return bool(out)


def compute_alignment(
    dance_duration_s: float,
    *,
    pad_in_s: float = PREP_PAD_IN_S,
    blend_in_s: float = PREP_BLEND_IN_S,
    blend_out_s: float = PREP_BLEND_OUT_S,
    hold_out_s: float = PREP_HOLD_OUT_S,
    window_start_s: float = 0.0,
) -> AudioAlignment:
    """Pure alignment math. Given the danced span's duration (and where in the
    source it started, if the motion was windowed), return where the music sits
    on the final prepped-motion timeline.

    The dance content in the prepped motion begins after the pad + blend-in, so
    the music is delayed by (pad_in_s + blend_in_s). The blend-out + hold-out
    tail plays out in silence (the robot returns to standing after the song).
    """
    if dance_duration_s <= 0:
        raise ValueError("dance_duration_s must be positive")
    for name, v in (("pad_in_s", pad_in_s), ("blend_in_s", blend_in_s),
                    ("blend_out_s", blend_out_s), ("hold_out_s", hold_out_s),
                    ("window_start_s", window_start_s)):
        if v < 0:
            raise ValueError(f"{name} must be >= 0")
    delay = pad_in_s + blend_in_s
    performance = delay + dance_duration_s + blend_out_s + hold_out_s
    return AudioAlignment(
        audio_delay_s=round(delay, 6),
        trim_start_s=round(window_start_s, 6),
        trim_duration_s=round(dance_duration_s, 6),
        music_end_s=round(delay + dance_duration_s, 6),
        performance_s=round(performance, 6),
    )


def alignment_from_prep_info(info: dict, *, window_start_s: float = 0.0,
                             **prep_overrides) -> AudioAlignment:
    """Derive alignment from a prep_motion info dict (its 'in_frames' is the
    danced-span length at PREP_FPS). Prep constants can be overridden if a
    non-default prep was used."""
    fps = prep_overrides.pop("fps", PREP_FPS)
    dance_s = info["in_frames"] / fps
    return compute_alignment(dance_s, window_start_s=window_start_s,
                             **prep_overrides)


def extract_audio(video: Path, out_wav: Path) -> Path | None:
    """Extract the source audio track to WAV. Returns None if the video is
    silent (no audio stream) — a common real case: exported dance clips are
    often muted, so the operator must then supply a separate music file."""
    if not has_audio(video):
        return None
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-vn", "-acodec", "pcm_s16le",
         "-ar", "44100", "-ac", "2", str(out_wav)],
        check=True, capture_output=True, text=True,
    )
    return out_wav


def mux_audio_onto_video(
    video: Path, audio: Path, align: AudioAlignment, out: Path,
) -> Path:
    """Lay the aligned music onto a (silent) preview video and write `out`.

    Audio is trimmed to the danced span, delayed to the dance start, and the
    video's own audio (if any) is replaced. Video is stream-copied; only audio
    is re-encoded. `-shortest` keeps output at the video length.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    delay_ms = int(round(align.audio_delay_s * 1000))
    afilter = (
        f"atrim=start={align.trim_start_s}:duration={align.trim_duration_s},"
        f"asetpts=PTS-STARTPTS,"
        f"adelay={delay_ms}:all=1,"
        f"apad"                       # pad tail with silence to video length
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-i", str(audio),
         "-filter_complex", f"[1:a]{afilter}[a]",
         "-map", "0:v", "-map", "[a]",
         "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)],
        check=True, capture_output=True, text=True,
    )
    return out


def make_placeholder_track(duration_s: float, out_wav: Path,
                           bpm: float = 118.0) -> Path:
    """Generate a royalty-free click/tone track standing in for real music
    (the Thriller source is silent and the real song is licensed). A periodic
    tone at `bpm` lets you audibly verify motion<->music sync in the preview."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    beat_hz = bpm / 60.0
    # a sine carrier gated by a beat-rate square envelope = a metronome-ish pulse
    expr = f"sin(2*PI*330*t)*(0.4*(1+sin(2*PI*{beat_hz}*t)))*lt(mod(t*{beat_hz}\\,1)\\,0.15)"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"aevalsrc={expr}:s=44100:d={duration_s}",
         "-ac", "2", str(out_wav)],
        check=True, capture_output=True, text=True,
    )
    return out_wav


def build_thriller_demo() -> dict:
    """Produce a music-synced Thriller preview end to end (best-effort, using
    whatever assets exist). Returns a report dict."""
    data = PROJECT_ROOT / "data"
    video = data / "videos" / "Thriller Dance Final.mov"
    show_csv = data / "motions" / "thriller" / "thriller_show.csv"
    dance_csv = data / "motions" / "thriller" / "thriller_g1.csv"
    # The preview MUST be of the PREPPED show motion (49.3s, with the standing
    # intro) — the 1.5s music delay is calibrated to that timeline. A preview of
    # the raw retarget (44.3s, no intro) would need delay=0 instead.
    preview = data / "previews" / "thriller_show_prepped.mp4"
    audio_dir = data / "audio" / "thriller"
    out = data / "previews" / "thriller_with_music.mp4"

    report: dict = {"ok": False}
    if not preview.exists():
        report["error"] = (
            f"no prepped-show preview at {preview} — render it first: "
            "python pipeline/playback_csv.py "
            "data/motions/thriller/thriller_show.csv --render "
            "data/previews/thriller_show_prepped.mp4")
        return report

    dance_frames = sum(1 for _ in open(dance_csv)) if dance_csv.exists() else 1329
    align = compute_alignment(dance_frames / PREP_FPS)
    report["alignment"] = align.to_dict()
    report["source_video_has_audio"] = video.exists() and has_audio(video)

    # ingest: real track if the video has one, else a labelled placeholder
    track = audio_dir / "music.wav"
    if report["source_video_has_audio"]:
        extract_audio(video, track)
        report["audio_source"] = "extracted_from_video"
    else:
        make_placeholder_track(align.trim_duration_s, track)
        report["audio_source"] = "placeholder_click_track (source video is silent)"

    mux_audio_onto_video(preview, track, align, out)
    report["ok"] = True
    report["output"] = str(out)
    report["note"] = (
        "Drop the real licensed music at data/audio/thriller/music.wav and "
        "re-run to replace the placeholder; alignment is unchanged."
    )
    return report


AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}


def attach_audio_for_dance(
    dance, *,
    source_audio: Path | None = None,
    extract_from_video: Path | None = None,
    placeholder_bpm: float | None = None,
    window_start_s: float = 0.0,
) -> dict:
    """Give a dance its music and return the audio record to store on it.

    Exactly one source is used, tried in order: an explicit audio file, extraction
    from a video, or a generated placeholder click track. The track is aligned to
    the dance's prepped-motion timeline (music delayed past the standing intro) and,
    if the dance has a silent preview, muxed onto a copy so the preview plays WITH
    the music. Returns the dict for shows.set_audio(); it does NOT persist itself.
    """
    from pipeline.config import DATA_DIR
    if not dance.duration_s or dance.duration_s <= 0:
        raise ValueError("dance has no duration — cannot align music to it")
    danced_s = float(dance.duration_s)
    align = compute_alignment(danced_s, window_start_s=window_start_s)

    audio_dir = DATA_DIR / "dances" / dance.id / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if source_audio is not None:
        src = Path(source_audio).expanduser()
        if not src.is_file():
            raise ValueError(f"audio file not found: {src}")
        if src.suffix.lower() not in AUDIO_EXTS:
            raise ValueError(f"unsupported audio type: {src.suffix} "
                             f"(want one of {sorted(AUDIO_EXTS)})")
        track = audio_dir / ("music" + src.suffix.lower())
        shutil.copyfile(src, track)
        source = "attached_file"
    elif extract_from_video is not None:
        vid = Path(extract_from_video).expanduser()
        if not vid.is_file():
            raise ValueError(f"video not found: {vid}")
        track = audio_dir / "music.wav"
        if extract_audio(vid, track) is None:
            raise ValueError("that video has no audio track to extract — "
                             "attach a separate music file instead")
        source = "extracted_from_video"
    else:
        if not shutil.which("ffmpeg"):
            raise ValueError("ffmpeg not found — cannot generate a placeholder track")
        track = audio_dir / "music.wav"
        make_placeholder_track(align.trim_duration_s, track,
                               bpm=placeholder_bpm or 118.0)
        source = "placeholder_click_track"

    rel = lambda p: str(Path(p).relative_to(PROJECT_ROOT)) if str(p).startswith(str(PROJECT_ROOT)) else str(p)
    record = {"track": rel(track), "source": source, "align": align.to_dict(),
              "muxed_preview": None, "attached_at": None}

    # Mux onto the dance's preview if we have a local silent preview to lay it on.
    preview = _resolve_local_preview(dance)
    if preview is not None and preview.exists() and shutil.which("ffmpeg"):
        out = DATA_DIR / "previews" / f"{dance.id}_with_music.mp4"
        try:
            mux_audio_onto_video(preview, track, align, out)
            record["muxed_preview"] = "/previews/" + out.name
        except subprocess.CalledProcessError:
            record["muxed_preview"] = None  # non-fatal: keep the audio, skip the mux
    return record


def _resolve_local_preview(dance) -> Path | None:
    """Find a local preview MP4 for a dance to mux music onto, if any."""
    from pipeline.config import DATA_DIR
    prev = dance.preview
    if not prev:
        return None
    if prev.startswith("/previews/"):
        return DATA_DIR / "previews" / prev.split("/")[-1]
    p = Path(prev)
    return p if p.is_absolute() else PROJECT_ROOT / prev


def main() -> None:
    ap = argparse.ArgumentParser(description="Dance audio sync")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("thriller-demo", help="build the music-synced Thriller preview")
    a = sub.add_parser("align", help="print alignment for a danced duration")
    a.add_argument("--dance-seconds", type=float, required=True)
    a.add_argument("--window-start", type=float, default=0.0)
    args = ap.parse_args()

    if args.cmd == "thriller-demo":
        if not shutil.which("ffmpeg"):
            raise SystemExit("ffmpeg not found")
        print(json.dumps(build_thriller_demo(), indent=2))
    elif args.cmd == "align":
        al = compute_alignment(args.dance_seconds, window_start_s=args.window_start)
        print(json.dumps(al.to_dict(), indent=2))


if __name__ == "__main__":
    main()
