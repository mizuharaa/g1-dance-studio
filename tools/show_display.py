#!/usr/bin/env python
"""Show-time VIDEO player: full-screen the side-by-side comparison on the big screen.

The dance is the star, but the audience also gets a WOW screen: the pre-rendered
side-by-side (reference | simulation, data/previews/thriller_side_by_side_v3e.mp4)
plays FULL-SCREEN on the EXTERNAL display, started in lockstep with the robot so it
reads as a real-time comparison.

SYNC CONTRACT (mirrors pipeline/show_audio.py):
    The wrapper (tools/show_run.sh) captures `date +%s.%N` the instant
    pipeline/deploy_runtime prints its "starting leg-odometry policy" line (tick0)
    and hands it to us as --at-epoch. We sleep until that wall-clock instant and
    launch the player then. The side-by-side already carries the 4.0 s pre-dance
    lead-in (2.5 s activation ramp + 1.5 s standing lead-in) baked into its own head,
    so it must start at tick0 itself — NOT at tick0+4 like the music cue. That way the
    video's on-screen dance and the robot's dance land together, and the video's baked
    lead-in stays aligned with the music that show_audio fires at tick0+4.

DISPLAY SELECTION:
    Preferring an EXTERNAL monitor if one is connected. --display (env SHOW_DISPLAY)
    names an xrandr output explicitly; otherwise we auto-pick the first connected
    NON-primary output; failing that we fall back to plain full-screen on the primary.
    In dev only the laptop panel (eDP-1) is connected; the external attaches at show
    time, so this handles both 1- and 2-monitor states.

PLAYER:
    Whichever of mpv / vlc / ffplay is installed (preference in that order). Each takes
    a different flag to full-screen on a chosen monitor; build_player_argv() maps the
    xrandr output to that player's screen index.

SAFETY: this module NEVER touches the robot or DDS. It only spawns a local media
player. A SIGTERM/SIGINT (the wrapper sends SIGTERM on runtime STOP/exit) terminates
the player immediately so the screen never outlives the dance.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Player preference order. Each name is resolved via shutil.which; the first that
# exists wins. build_player_argv() knows the full-screen + screen-select flags per
# player. VLC is LAST resort: on this box it renders colourful static / logs "Too high
# level of recursion" (2026-07-13). Prefer mpv, then ffplay. `SHOW_PLAYER` (env) forces
# a specific one. Install the good player with:  sudo apt install -y mpv
PLAYERS = ("mpv", "ffplay", "vlc")


# ---- xrandr parsing (pure, unit-tested) ------------------------------------------------
@dataclass(frozen=True)
class Monitor:
    """One connected xrandr output. `index` is its position among connected outputs —
    the screen NUMBER that mpv --screen / vlc --qt-fullscreen-screennumber expect."""
    name: str
    primary: bool
    index: int


def parse_monitors(xrandr_text: str) -> list[Monitor]:
    """Parse `xrandr --query` output into the list of CONNECTED outputs.

    Output-header lines are unindented and look like:
        eDP-1 connected primary 1920x1200+0+0 (normal ...) 302mm x 189mm
        HDMI-1 connected 1920x1080+1920+0 (normal ...) 520mm x 320mm
        DP-2 disconnected (normal ...)
    Mode lines are indented and ignored; the leading "Screen 0: ..." line has a second
    token of "0:" (not "connected") and is ignored. `disconnected` is NOT `connected`
    (we match the whole token, so the shared substring never trips us up)."""
    monitors: list[Monitor] = []
    idx = 0
    for line in xrandr_text.splitlines():
        if not line or line[0].isspace():        # indented mode line
            continue
        parts = line.split()
        if len(parts) < 2 or parts[1] != "connected":
            continue                              # "disconnected", "Screen 0:", junk
        primary = len(parts) > 2 and parts[2] == "primary"
        monitors.append(Monitor(name=parts[0], primary=primary, index=idx))
        idx += 1
    return monitors


def pick_display(monitors: list[Monitor], prefer: str | None = None) -> Monitor | None:
    """Choose the monitor to play on:
      1. `prefer` (env SHOW_DISPLAY) if it names a connected output;
      2. else the first connected NON-primary output (the external big screen);
      3. else None -> caller uses plain full-screen on the primary/default monitor.
    A named-but-not-connected `prefer` warns and falls through to auto-pick (the
    external is often plugged in only moments before showtime)."""
    if prefer:
        for m in monitors:
            if m.name == prefer:
                return m
        print(f"[show_display] SHOW_DISPLAY={prefer!r} not connected — auto-picking",
              file=sys.stderr)
    externals = [m for m in monitors if not m.primary]
    if externals:
        return externals[0]
    return None


# ---- player discovery + argv (pure, unit-tested) ---------------------------------------
def find_player(players: tuple[str, ...] = PLAYERS, which=shutil.which,
                forced: str | None = None) -> str:
    """Return the player to use. `forced` (or env SHOW_PLAYER) pins a specific player —
    use it to force mpv/ffplay when the auto-pick would otherwise land on the buggy VLC.
    Otherwise the first installed player in preference order wins, else SystemExit."""
    forced = forced or os.environ.get("SHOW_PLAYER")
    if forced:
        if which(forced):
            return forced
        raise SystemExit(f"SHOW_PLAYER={forced!r} is not installed "
                         f"(have: {', '.join(p for p in players if which(p)) or 'none'})")
    for p in players:
        if which(p):
            return p
    raise SystemExit(f"no video player found — install one of: {', '.join(players)} "
                     f"(recommended: sudo apt install -y mpv)")


def build_player_argv(player: str, video, screen_index: int | None = None) -> list[str]:
    """Full-screen argv for `player`. When screen_index is not None the video is
    pinned to that monitor (via each player's own screen-select flag); when None the
    player full-screens on its default/primary monitor.

    Every player is told to quit at end-of-file so the wrapper's `wait` returns and the
    screen never lingers past the dance."""
    video = str(video)
    if player == "mpv":
        argv = ["mpv", "--fullscreen", "--no-terminal", "--no-osc"]
        if screen_index is not None:
            argv += [f"--screen={screen_index}", f"--fs-screen={screen_index}"]
        argv.append(video)                              # mpv exits at EOF by default
        return argv
    if player == "vlc":
        # VLC is the LAST-RESORT player (prefer mpv/ffplay). Two documented VLC bugs on
        # this box, both defended against here:
        #  * --avcodec-hw=none forces SOFTWARE decode. On Intel iGPUs vlc's VA-API path
        #    hands the filter chain a hardware surface it can't read ("Unknown input chroma
        #    VAOP") and renders colourful static (2026-07-08 live run).
        #  * "Too high level of recursion" (2026-07-13): handing the file to an ALREADY
        #    running VLC instance and the subtitle/OSD filter chain recurse. --no-one-instance
        #    launches a fresh process; --no-spu/--no-sub-autodetect-file/--no-osd drop the
        #    filters that recurse. A pre-rendered side-by-side needs none of them.
        argv = ["vlc", "--fullscreen", "--no-video-title-show", "--play-and-exit",
                "--avcodec-hw=none", "--no-one-instance", "--no-osd", "--no-spu",
                "--no-sub-autodetect-file"]
        if screen_index is not None:
            argv.append(f"--qt-fullscreen-screennumber={screen_index}")
        argv.append(video)
        return argv
    if player == "ffplay":
        # ffplay has no output-name/screen selector — full-screen on the current screen.
        return ["ffplay", "-fs", "-autoexit", "-loglevel", "error", video]
    raise ValueError(f"unknown player {player!r}")


# ---- xrandr runner ---------------------------------------------------------------------
def run_xrandr(binary: str = "xrandr") -> str:
    """`xrandr --query` stdout. Injected/patched away in tests (needs an X display)."""
    exe = shutil.which(binary) or binary
    out = subprocess.run([exe, "--query"], capture_output=True, text=True, check=True)
    return out.stdout


# ---- waiting (pure, unit-tested) -------------------------------------------------------
def wait_until(target_epoch: float, *, now=time.time, sleep=time.sleep) -> float:
    """Sleep until wall-clock target_epoch (coarse sleeps then fine). Returns the
    (signed) firing error in seconds — >=0 means we fired at/after target. Same pacing
    as pipeline/show_audio.wait_until so audio and video align."""
    while True:
        dt = target_epoch - now()
        if dt <= 0:
            return -dt
        sleep(min(dt, 0.05) if dt < 0.25 else dt - 0.2)


# ---- resolution + playback -------------------------------------------------------------
@dataclass
class Playback:
    player: str
    argv: list[str]
    monitor: Monitor | None
    monitors: list[Monitor] = field(default_factory=list)


def resolve_playback(video, *, prefer_output: str | None = None,
                     xrandr_text: str | None = None, xrandr_runner=None,
                     which=shutil.which, players: tuple[str, ...] = PLAYERS) -> Playback:
    """Decide player + display + argv without spawning anything. If xrandr is
    unavailable (no X display in a test/headless shell) we degrade to plain
    full-screen on the primary rather than failing the show."""
    if xrandr_text is None:
        try:
            xrandr_text = (xrandr_runner or run_xrandr)()
        except Exception as e:  # noqa: BLE001 — a broken xrandr must not kill the show
            print(f"[show_display] xrandr failed ({e}) — plain full-screen", file=sys.stderr)
            xrandr_text = ""
    monitors = parse_monitors(xrandr_text)
    chosen = pick_display(monitors, prefer_output)
    player = find_player(players, which)
    if player == "vlc":
        print("[show_display] falling back to VLC, which can render colourful static / "
              "'Too high level of recursion' on this box. For a clean show display install "
              "mpv:  sudo apt install -y mpv", file=sys.stderr)
    screen_index = chosen.index if chosen is not None else None
    argv = build_player_argv(player, video, screen_index)
    return Playback(player=player, argv=argv, monitor=chosen, monitors=monitors)


# a stack of live player procs so the signal handler can stop them (single-threaded CLI)
_LIVE: list = []


def play(video, *, at_epoch: float | None = None, prefer_output: str | None = None,
         popen=subprocess.Popen, now=time.time, sleep=time.sleep, **resolve_kw):
    """Resolve the display, wait until at_epoch (if given), then launch the player
    full-screen. Returns the player Popen; the caller (main) waits on it. Registers the
    proc in _LIVE so a SIGTERM aborts it even mid-launch."""
    video = Path(video)
    if not video.is_file():
        raise SystemExit(f"show video not found: {video}")
    pb = resolve_playback(video, prefer_output=prefer_output, **resolve_kw)
    where = pb.monitor.name if pb.monitor else "(primary/default)"
    print(f"[show_display] player={pb.player} output={where} argv={' '.join(pb.argv)}")
    if at_epoch is not None:
        late = wait_until(at_epoch, now=now, sleep=sleep)
        print(f"[show_display] launching video ({late:+.3f}s vs tick0 target)")
    proc = popen(pb.argv)
    _LIVE.append(proc)
    return proc


def _stop_all():
    while _LIVE:
        p = _LIVE.pop()
        try:
            if p.poll() is None:
                p.terminate()
        except Exception as e:  # noqa: BLE001 — best effort, the show is ending
            print(f"[show_display] stop failed: {e}", file=sys.stderr)


def _abort_handler(signum, frame):  # pragma: no cover - exercised via unit call
    print(f"\n[show_display] signal {signum} -> stopping video", file=sys.stderr)
    _stop_all()
    os._exit(143)  # 128 + SIGTERM(15)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", nargs="?", default=os.environ.get("SHOW_VIDEO"),
                    help="video file to full-screen (default: env SHOW_VIDEO)")
    ap.add_argument("--display", default=os.environ.get("SHOW_DISPLAY"),
                    help="xrandr output name (e.g. HDMI-1); default env SHOW_DISPLAY, "
                         "else auto-pick a non-primary monitor")
    ap.add_argument("--at-epoch", type=float, default=None,
                    help="wall-clock epoch seconds to launch at (tick0 anchor); sleeps "
                         "until then so the video aligns with the robot + music")
    args = ap.parse_args(argv)

    if not args.video:
        raise SystemExit("no video: set SHOW_VIDEO or pass a path")

    signal.signal(signal.SIGTERM, _abort_handler)
    signal.signal(signal.SIGINT, _abort_handler)

    proc = play(args.video, at_epoch=args.at_epoch, prefer_output=args.display)
    rc = proc.wait()
    if proc in _LIVE:
        _LIVE.remove(proc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
