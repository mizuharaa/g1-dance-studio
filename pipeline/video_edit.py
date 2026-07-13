"""Server-side video trimming + thumbnail extraction for the upload trimmer.

Clips longer than MAX_UNTRIMMED_S must be cut to a <=4 min segment before the pipeline runs
(the training/endurance envelope targets 2-3 min). The desktop webview can't play H.264, so the
trimmer UI scrubs via still FRAMES (jpg — which the webview CAN render) that this module extracts
on demand; the final TRIM is a fast keyframe copy. Uses imageio-ffmpeg's bundled ffmpeg.
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

MAX_UNTRIMMED_S = 240          # clips longer than this (4 min) are gated for trimming
MIN_SEGMENT_S = 5              # a segment must be at least this long

_ffmpeg: str | None = None
_proxy_jobs: dict[str, str] = {}
_proxy_lock = threading.Lock()


def ffmpeg_exe() -> str:
    global _ffmpeg
    if _ffmpeg is None:
        import imageio_ffmpeg
        _ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    return _ffmpeg


def probe_duration(path: str | Path) -> float:
    """Duration in seconds (0.0 if unknown). Uses imageio's ffmpeg metadata."""
    try:
        import imageio.v2 as imageio
        r = imageio.get_reader(str(path))
        meta = r.get_meta_data(); r.close()
        return float(meta.get("duration") or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def extract_frame(src: Path, t: float, out_jpg: Path) -> bool:
    """Write a single ~480px-wide jpg frame at time `t` (seconds). Cached by the caller.
    Fast input-seek (`-ss` before `-i`). Returns True on success."""
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_jpg.with_name(out_jpg.name + ".part.jpg")
    cmd = [ffmpeg_exe(), "-y", "-ss", f"{max(0.0, t):.3f}", "-i", str(src),
           "-frames:v", "1", "-q:v", "4", "-vf", "scale=480:-2", "-f", "image2", str(tmp)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=60)
        tmp.replace(out_jpg)
        return True
    except Exception:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return False


def _proxy_transcode(src: Path, out_webm: Path) -> None:
    """Small, fast, seekable VP9/WebM proxy (640-wide, 15 fps, no audio) for the trimmer's live
    scrubber — the desktop webview can't decode H.264 but plays VP9. ~15-30x realtime."""
    tmp = out_webm.with_name(out_webm.name + ".part.webm")
    out_webm.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-i", str(src), "-vf", "scale=640:-2", "-r", "15",
           "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "40", "-deadline", "realtime",
           "-cpu-used", "8", "-row-mt", "1", "-an", "-f", "webm", str(tmp)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=600)
    tmp.replace(out_webm)


def ensure_proxy(src: Path, out_webm: Path) -> dict:
    """Ensure a scrubbable webm proxy of `src` exists (background + cached). {ready, status}."""
    if not src.is_file():
        return {"ready": False, "status": "source-missing"}
    try:
        if out_webm.is_file() and out_webm.stat().st_mtime >= src.stat().st_mtime:
            return {"ready": True}
    except OSError:
        pass
    key = str(out_webm)
    with _proxy_lock:
        state = _proxy_jobs.get(key)
        if state is None or state.startswith("error"):
            _proxy_jobs[key] = "running"

            def _run() -> None:
                try:
                    _proxy_transcode(src, out_webm)
                    with _proxy_lock:
                        _proxy_jobs.pop(key, None)
                except Exception as e:  # noqa: BLE001
                    with _proxy_lock:
                        _proxy_jobs[key] = f"error: {type(e).__name__}"
            threading.Thread(target=_run, daemon=True).start()
        status = _proxy_jobs.get(key, "running")
    return {"ready": False, "status": status}


def trim(src: Path, dst: Path, start_s: float, length_s: float) -> None:
    """Copy the [start_s, start_s+length_s] segment into dst (fast, keyframe-aligned).
    Re-mux only (no re-encode) so it's near-instant even for a 4 min HD clip."""
    tmp = dst.with_name(dst.name + ".part.mp4")
    cmd = [ffmpeg_exe(), "-y", "-ss", f"{max(0.0, start_s):.3f}", "-i", str(src),
           "-t", f"{max(MIN_SEGMENT_S, length_s):.3f}", "-c", "copy",
           "-avoid_negative_ts", "make_zero", "-f", "mp4", str(tmp)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=300)
    tmp.replace(dst)
