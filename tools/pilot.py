#!/usr/bin/env python
"""Browser pilot: lets Claude drive a headed Chrome window on the user's display.

Runs a persistent Chrome (separate profile in .secrets/browser-profile) and polls
PILOT_DIR/cmd.json for commands; writes result.json and shot.png after each command.
The user can type in the window at any time (captcha/OTP); Claude only acts when
a command is issued. Kill with: rm PILOT_DIR/alive  (or the process).

Command format (cmd.json): {"id": <int>, "action": "...", ...args}
Actions:
  goto {url}                       navigate
  click {selector | x,y}           click element or coordinates
  fill {selector, text}            fill input
  type {text}                      type into focused element
  press {key}                      keyboard key (e.g. "Enter", "Tab")
  scroll {dy}                      scroll by pixels
  eval {js}                        evaluate JS, return value
  text {}                          page inner text (trimmed to 15k chars)
  html {selector}                  outerHTML of first match (trimmed)
  shot {}                          screenshot only
Every command also refreshes shot.png (full viewport) and result.json:
  {"id", "ok", "value", "url", "title", "error"}
"""
import json
import time
import traceback
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / ".secrets" / "pilot"
PROFILE = ROOT / ".secrets" / "browser-profile"
PILOT.mkdir(parents=True, exist_ok=True)
PROFILE.mkdir(parents=True, exist_ok=True)
CMD = PILOT / "cmd.json"
RES = PILOT / "result.json"
SHOT = PILOT / "shot.png"
ALIVE = PILOT / "alive"


def run() -> None:
    ALIVE.write_text(str(time.time()))
    last_id = None
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(PROFILE),
            channel="chrome",
            headless=False,
            viewport={"width": 1280, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("about:blank")
        while ALIVE.exists():
            time.sleep(0.4)
            if not CMD.exists():
                continue
            try:
                cmd = json.loads(CMD.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if cmd.get("id") == last_id:
                continue
            last_id = cmd.get("id")
            out = {"id": last_id, "ok": True, "value": None, "error": None}
            try:
                page = ctx.pages[-1] if ctx.pages else ctx.new_page()
                act = cmd["action"]
                if act == "goto":
                    page.goto(cmd["url"], timeout=45000)
                elif act == "click":
                    if "selector" in cmd:
                        page.click(cmd["selector"], timeout=8000)
                    else:
                        page.mouse.click(cmd["x"], cmd["y"])
                elif act == "fill":
                    page.fill(cmd["selector"], cmd["text"], timeout=8000)
                elif act == "type":
                    page.keyboard.type(cmd["text"], delay=30)
                elif act == "press":
                    page.keyboard.press(cmd["key"])
                elif act == "scroll":
                    page.mouse.wheel(0, cmd["dy"])
                elif act == "eval":
                    out["value"] = page.evaluate(cmd["js"])
                elif act == "text":
                    out["value"] = page.inner_text("body")[:15000]
                elif act == "html":
                    el = page.query_selector(cmd["selector"])
                    out["value"] = el.evaluate("e => e.outerHTML")[:15000] if el else None
                elif act == "shot":
                    pass
                else:
                    raise ValueError(f"unknown action {act!r}")
                page.wait_for_timeout(700)
            except Exception:
                out["ok"] = False
                out["error"] = traceback.format_exc()[-1500:]
            try:
                out["url"] = page.url
                out["title"] = page.title()
                page.screenshot(path=str(SHOT))
            except Exception:
                pass
            RES.write_text(json.dumps(out))
        ctx.close()


if __name__ == "__main__":
    run()
