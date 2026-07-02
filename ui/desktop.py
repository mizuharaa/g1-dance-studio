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

    import uvicorn
    from ui.server import app

    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=args.port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    _wait_for_port("127.0.0.1", args.port)

    import webview
    webview.create_window("G1 Dance Studio", f"http://127.0.0.1:{args.port}/",
                          width=1280, height=860, min_size=(980, 640))
    webview.start(gui="qt", debug=args.debug)


if __name__ == "__main__":
    main()
