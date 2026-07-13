"""On-demand H.264-mp4 -> VP9/WebM transcode so the desktop app's PySide6 QtWebEngine
(which ships WITHOUT the H.264 codec) can play preview footage INLINE instead of punting to
an external browser.

The WebM is written next to its source (``foo.mp4`` -> ``foo.webm``) so the existing
``/previews`` static mount serves it (with HTTP Range, so the player can seek). Transcodes
are cached (reused if newer than the source) and de-duplicated across concurrent requests.
Uses imageio-ffmpeg's bundled ffmpeg — there is no system ffmpeg on this laptop.
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

_ffmpeg: str | None = None
_jobs: dict[str, str] = {}          # source-path -> "running" | "error: ..."
_lock = threading.Lock()


def ffmpeg_exe() -> str:
    global _ffmpeg
    if _ffmpeg is None:
        import imageio_ffmpeg
        _ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    return _ffmpeg


def webm_sibling(mp4: Path) -> Path:
    return mp4.with_suffix(".webm")


def _fresh(webm: Path, mp4: Path) -> bool:
    try:
        return webm.is_file() and webm.stat().st_mtime >= mp4.stat().st_mtime
    except OSError:
        return False


def _transcode(mp4: Path, webm: Path) -> None:
    """Fast VP9 (realtime, cpu-used 8) — plenty for a fixed-size preview; ~10x realtime.
    Audio (dance previews have music; sim previews are silent) is kept as Opus when present."""
    tmp = webm.with_name(webm.name + ".tmp")
    # -f webm forces the muxer (the .tmp extension would otherwise defeat format detection).
    # libvorbis is the reliably-working audio encoder in this static build (a no-op when the
    # source is silent, e.g. sim previews); dance previews keep their music.
    cmd = [ffmpeg_exe(), "-y", "-i", str(mp4),
           "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "36",
           "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1",
           "-pix_fmt", "yuv420p", "-c:a", "libvorbis", "-f", "webm", str(tmp)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=600)
    tmp.replace(webm)


def ensure_webm(mp4: Path) -> dict:
    """Return {ready: bool, status?: str}. On the first request for a not-yet-transcoded
    source, kicks off a background transcode and returns ready=False; poll again until ready."""
    if not mp4.is_file():
        return {"ready": False, "status": "source-missing"}
    webm = webm_sibling(mp4)
    if _fresh(webm, mp4):
        return {"ready": True}
    key = str(mp4)
    with _lock:
        state = _jobs.get(key)
        if state is None or state.startswith("error"):
            _jobs[key] = "running"

            def _run() -> None:
                try:
                    _transcode(mp4, webm)
                    with _lock:
                        _jobs.pop(key, None)
                except Exception as e:  # noqa: BLE001 — surfaced via status, never crashes the app
                    with _lock:
                        _jobs[key] = f"error: {type(e).__name__}: {e}"[:160]

            threading.Thread(target=_run, daemon=True).start()
        status = _jobs.get(key, "running")
    return {"ready": False, "status": status}
