"""G1 Dance Studio — desktop entry point.

Starts the FastAPI engine (ui/server.py) on localhost in a background thread,
waits until it answers, then opens it in a native pywebview window (Qt backend,
installed via `pip install pywebview[qt]` — no system WebKit2GTK needed).
"""
from __future__ import annotations

import argparse
import socket
import threading
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def _wait_for_port(host: str, port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"server did not come up on {host}:{port}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8735)
    ap.add_argument("--debug", action="store_true",
                    help="enable webview devtools / right-click inspect")
    args = ap.parse_args()

    # A stale server from a previous launch would still be holding the port; our
    # uvicorn thread would fail to bind (error swallowed) and the window would attach
    # to that STALE server, silently serving old code. Fail loudly instead of
    # attaching to someone else's process (audit: stale-process/port issue).
    if _port_in_use("127.0.0.1", args.port):
        raise SystemExit(
            f"port {args.port} is already in use — a previous G1 Dance Studio "
            f"is likely still running.\n"
            f"  Close it (pkill -f ui/desktop.py) or start with --port <other>.\n"
            f"  Refusing to attach to a stale server (it may serve out-of-date code).")

    import uvicorn
    from ui.server import app

    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=args.port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    _wait_for_port("127.0.0.1", args.port)

    import webview

    class DesktopBridge:
        """Python functions exposed to the web app as ``window.pywebview.api``.

        The bundled PySide6 QtWebEngine ships WITHOUT proprietary codecs (H.264/AAC), so
        the H.264 .mp4 previews will not decode inside this window. ``open_external`` hands
        a preview URL to the operator's real system browser (Chrome/Firefox — which HAVE
        H.264) so previews are always watchable. ``is_desktop`` lets the frontend detect the
        native shell and offer the button proactively instead of after a silent failure."""
        def is_desktop(self) -> bool:
            return True

        def open_external(self, url: str) -> bool:
            import webbrowser
            # Only ever open http/https (the local engine's own preview URLs) — never a
            # file:// path or a shell string.
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                webbrowser.open(url)
                return True
            return False

    webview.create_window("G1 Dance Studio", f"http://127.0.0.1:{args.port}/",
                          width=1280, height=860, min_size=(980, 640),
                          js_api=DesktopBridge())
    webview.start(gui="qt", debug=args.debug)


if __name__ == "__main__":
    main()
