#!/usr/bin/env python3
"""Drop-in real music for a dance: convert -> replace -> re-attach.

Usage: attach_music.py <audio-file> [--dance-id 20260704-18f65bbd] [--slug thriller]
Converts any ffmpeg-readable audio to 44.1k stereo WAV, replaces
data/audio/<slug>/music.wav, and re-runs attach_audio_for_dance so the dance
record and its preview mux pick up the real track. Refuses placeholder-like
input (a click track: mostly silence + one pure tone).
"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
FF = str(Path.home() / "miniconda3/envs/g1dance/bin/ffmpeg")

def looks_like_click_track(wav: Path) -> bool:
    r = subprocess.run([FF, "-v", "error", "-i", str(wav), "-f", "s16le", "-ac", "1",
                        "-ar", "8000", "-"], capture_output=True)
    x = np.frombuffer(r.stdout, np.int16).astype(float)
    if len(x) < 8000: return True
    env = np.abs(x[: len(x) // 400 * 400]).reshape(-1, 400).mean(1)
    silent = (env < env.max() * 0.05).mean()
    spec = np.abs(np.fft.rfft(x[: 8000 * 10]))
    top = spec[50:].max() / (spec[50:].mean() + 1e-9)
    return bool(silent > 0.5 and top > 200)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--dance-id", default="20260704-18f65bbd")
    ap.add_argument("--slug", default="thriller")
    a = ap.parse_args()
    src = Path(a.audio).expanduser()
    assert src.is_file(), f"not found: {src}"
    dst = Path(f"data/audio/{a.slug}/music.wav")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".new.wav")
    subprocess.run([FF, "-v", "error", "-y", "-i", str(src), "-ar", "44100", "-ac", "2",
                    str(tmp)], check=True)
    if looks_like_click_track(tmp):
        tmp.unlink(); sys.exit("REFUSED: input looks like a click track / placeholder")
    tmp.replace(dst)
    from pipeline import shows
    from pipeline.audio import attach_audio_for_dance
    d = shows.load_dance(a.dance_id)
    rec = attach_audio_for_dance(d, source_audio=dst)
    shows.set_audio(a.dance_id, rec)
    print(f"attached real track to {a.dance_id}: {dst} ({src.name})")

if __name__ == "__main__":
    main()
