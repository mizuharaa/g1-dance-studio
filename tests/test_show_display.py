"""Offline tests for the show-time VIDEO player — no robot, no X display, no player.

tools/show_display.py full-screens the pre-rendered side-by-side comparison on the
EXTERNAL monitor, launched at the same tick0 anchor as the music so the video, the
music and the robot all start together. These tests pin, with xrandr + the player +
the clock all faked:

  * xrandr parsing (connected vs disconnected, the primary flag, the "Screen 0:" and
    indented mode lines ignored) and display selection — EXTERNAL when a non-primary
    output is connected, the primary/default when only the laptop panel is (the dev
    state; the external attaches at show time);
  * player discovery in preference order (mpv > ffplay > vlc — VLC is last resort) plus
    the SHOW_PLAYER override, and the exact full-screen argv each player gets, including
    the per-player screen-select flag;
  * --at-epoch: play() sleeps until the tick0 wall-clock instant before spawning, so it
    aligns with pipeline/show_audio's cue (mock clock);
  * abort: a live player is terminated on stop (the wrapper SIGTERMs us on runtime STOP).

The module is loaded straight from its file (tools/ is not a package) — same pattern
as tests/test_edit_choreography.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "show_display.py"
_spec = importlib.util.spec_from_file_location("show_display", TOOL)
sd = importlib.util.module_from_spec(_spec)
sys.modules["show_display"] = sd   # dataclasses need the module registered to resolve types
_spec.loader.exec_module(sd)


# ---- xrandr fixtures (real `xrandr --query` shapes) ------------------------------------
# dev box: only the laptop panel is connected (the external attaches at show time).
DEV_XRANDR = """\
Screen 0: minimum 320 x 200, current 1920 x 1200, maximum 16384 x 16384
eDP-1 connected primary 1920x1200+0+0 (normal left inverted right x axis y axis) 302mm x 189mm
   1920x1200     60.00*+
   1920x1080     60.00
HDMI-1 disconnected (normal left inverted right x axis y axis)
DP-1 disconnected (normal left inverted right x axis y axis)
"""

# show time: the big screen (HDMI-1) is now connected, laptop panel is primary.
SHOW_XRANDR = """\
Screen 0: minimum 320 x 200, current 3840 x 1200, maximum 16384 x 16384
eDP-1 connected primary 1920x1200+0+0 (normal left inverted right x axis y axis) 302mm x 189mm
   1920x1200     60.00*+
HDMI-1 connected 1920x1080+1920+0 (normal left inverted right x axis y axis) 520mm x 320mm
   1920x1080     60.00*+
   1280x720      60.00
DP-1 disconnected (normal left inverted right x axis y axis)
"""


def which_of(*present):
    """A shutil.which stand-in: resolves only the named binaries."""
    have = set(present)
    return lambda name: (f"/usr/bin/{name}" if name in have else None)


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def now(self):
        return self.t

    def sleep(self, s):
        assert s >= 0
        self.t += s


class FakeProc:
    """Minimal Popen stand-in: poll() is None until terminate()d."""
    def __init__(self, rc=0):
        self._rc = rc
        self.terminated = False

    def poll(self):
        return None if not self.terminated else self._rc

    def wait(self):
        return self._rc

    def terminate(self):
        self.terminated = True


@pytest.fixture(autouse=True)
def _clear_live():
    """Never let a registered proc leak between tests (module-level _LIVE)."""
    sd._LIVE.clear()
    yield
    sd._LIVE.clear()


@pytest.fixture(autouse=True)
def _no_show_player_env(monkeypatch):
    """find_player reads SHOW_PLAYER from the env — keep the default-order tests hermetic."""
    monkeypatch.delenv("SHOW_PLAYER", raising=False)


# ---- xrandr parsing --------------------------------------------------------------------
def test_parse_dev_only_primary_panel():
    mons = sd.parse_monitors(DEV_XRANDR)
    assert [m.name for m in mons] == ["eDP-1"]        # disconnected outputs dropped
    assert mons[0].primary is True and mons[0].index == 0


def test_parse_showtime_two_monitors_external_not_primary():
    mons = sd.parse_monitors(SHOW_XRANDR)
    assert [m.name for m in mons] == ["eDP-1", "HDMI-1"]
    assert mons[0].primary is True and mons[0].index == 0
    assert mons[1].primary is False and mons[1].index == 1  # screen number for the player


def test_parse_ignores_screen_header_and_mode_lines():
    # the "Screen 0:" line and every indented "   1920x1200 ..." mode line are skipped;
    # "disconnected" must not be mistaken for "connected" despite the shared substring.
    assert sd.parse_monitors(DEV_XRANDR) == [sd.Monitor("eDP-1", True, 0)]
    assert sd.parse_monitors("") == []


# ---- display selection -----------------------------------------------------------------
def test_pick_external_when_connected():
    chosen = sd.pick_display(sd.parse_monitors(SHOW_XRANDR))
    assert chosen is not None and chosen.name == "HDMI-1" and chosen.index == 1


def test_pick_none_when_only_primary():
    # dev state: nothing but the primary panel -> None -> caller plain full-screens.
    assert sd.pick_display(sd.parse_monitors(DEV_XRANDR)) is None


def test_pick_honors_named_output():
    mons = sd.parse_monitors(SHOW_XRANDR)
    chosen = sd.pick_display(mons, prefer="eDP-1")     # explicit override wins
    assert chosen is not None and chosen.name == "eDP-1" and chosen.index == 0


def test_pick_named_but_absent_falls_back_to_auto(capsys):
    mons = sd.parse_monitors(SHOW_XRANDR)
    chosen = sd.pick_display(mons, prefer="DP-9")       # not connected
    assert chosen is not None and chosen.name == "HDMI-1"  # auto-picks the external
    assert "not connected" in capsys.readouterr().err


def test_pick_named_absent_and_no_external_is_none(capsys):
    mons = sd.parse_monitors(DEV_XRANDR)
    assert sd.pick_display(mons, prefer="HDMI-1") is None
    assert "not connected" in capsys.readouterr().err


# ---- player discovery ------------------------------------------------------------------
def test_find_player_preference_order():
    assert sd.find_player(which=which_of("mpv", "vlc")) == "mpv"
    assert sd.find_player(which=which_of("vlc")) == "vlc"
    assert sd.find_player(which=which_of("ffplay")) == "ffplay"


def test_find_player_demotes_vlc_below_ffplay():
    # VLC is last resort: when both ffplay and vlc are present, ffplay wins.
    assert sd.find_player(which=which_of("ffplay", "vlc")) == "ffplay"


def test_find_player_none_installed_is_actionable():
    with pytest.raises(SystemExit, match="no video player"):
        sd.find_player(which=which_of())


def test_show_player_env_override_forces_installed_player(monkeypatch):
    monkeypatch.setenv("SHOW_PLAYER", "vlc")
    # even though mpv is present and preferred, the override forces vlc
    assert sd.find_player(which=which_of("mpv", "vlc")) == "vlc"


def test_show_player_override_not_installed_is_actionable(monkeypatch):
    monkeypatch.setenv("SHOW_PLAYER", "mpv")
    with pytest.raises(SystemExit, match="SHOW_PLAYER='mpv' is not installed"):
        sd.find_player(which=which_of("vlc"))


def test_vlc_argv_has_recursion_defenses():
    # the "Too high level of recursion" / colourful-static defenses must be present
    argv = sd.build_player_argv("vlc", "/v.mp4", screen_index=1)
    for flag in ("--avcodec-hw=none", "--no-one-instance", "--no-spu"):
        assert flag in argv, f"missing VLC defense flag {flag}"


# ---- argv construction -----------------------------------------------------------------
def test_mpv_argv_fullscreen_on_screen():
    argv = sd.build_player_argv("mpv", "/v.mp4", screen_index=1)
    assert "--fullscreen" in argv
    assert "--screen=1" in argv and "--fs-screen=1" in argv
    assert argv[0] == "mpv" and argv[-1] == "/v.mp4"


def test_mpv_argv_no_screen_when_default():
    argv = sd.build_player_argv("mpv", "/v.mp4", screen_index=None)
    assert not any(a.startswith("--screen") or a.startswith("--fs-screen") for a in argv)
    assert "--fullscreen" in argv and argv[-1] == "/v.mp4"


def test_vlc_argv_fullscreen_on_screen():
    argv = sd.build_player_argv("vlc", "/v.mp4", screen_index=1)
    assert argv[0] == "vlc" and "--fullscreen" in argv
    assert "--qt-fullscreen-screennumber=1" in argv
    assert "--play-and-exit" in argv          # quits at EOF so the wrapper's wait returns
    assert argv[-1] == "/v.mp4"


def test_vlc_argv_no_screennumber_when_default():
    argv = sd.build_player_argv("vlc", "/v.mp4", screen_index=None)
    assert not any(a.startswith("--qt-fullscreen-screennumber") for a in argv)
    assert "--fullscreen" in argv and argv[-1] == "/v.mp4"


def test_ffplay_argv():
    argv = sd.build_player_argv("ffplay", "/v.mp4")
    assert argv[0] == "ffplay" and "-fs" in argv and "-autoexit" in argv
    assert argv[-1] == "/v.mp4"


def test_unknown_player_rejected():
    with pytest.raises(ValueError, match="unknown player"):
        sd.build_player_argv("totem", "/v.mp4")


# ---- full resolution (display + player + argv, no spawn) --------------------------------
def test_resolve_showtime_targets_external():
    pb = sd.resolve_playback("/v.mp4", xrandr_text=SHOW_XRANDR, which=which_of("vlc"))
    assert pb.player == "vlc"
    assert pb.monitor is not None and pb.monitor.name == "HDMI-1"
    assert "--qt-fullscreen-screennumber=1" in pb.argv


def test_resolve_dev_falls_back_to_primary():
    pb = sd.resolve_playback("/v.mp4", xrandr_text=DEV_XRANDR, which=which_of("vlc"))
    assert pb.monitor is None                                   # only the primary panel
    assert not any(a.startswith("--qt-fullscreen-screennumber") for a in pb.argv)
    assert "--fullscreen" in pb.argv


def test_resolve_survives_broken_xrandr(capsys):
    def boom():
        raise FileNotFoundError("xrandr")
    pb = sd.resolve_playback("/v.mp4", xrandr_runner=boom, which=which_of("mpv"))
    assert pb.monitor is None and pb.player == "mpv"            # degrades gracefully
    assert "xrandr failed" in capsys.readouterr().err


# ---- wait_until ------------------------------------------------------------------------
def test_wait_until_sleeps_to_target_and_reports_lateness():
    clk = FakeClock(1000.0)
    late = sd.wait_until(1004.0, now=clk.now, sleep=clk.sleep)
    assert clk.t == pytest.approx(1004.0, abs=0.06)
    assert 0.0 <= late < 0.06


# ---- play(): the tick0-anchored launch -------------------------------------------------
def test_play_waits_for_at_epoch_then_spawns(tmp_path):
    video = tmp_path / "sbs.mp4"
    video.write_bytes(b"\0")
    clk = FakeClock(1000.0)
    spawned = []

    def fake_popen(argv):
        spawned.append((list(argv), clk.t))     # capture WHEN the player launched
        return FakeProc()

    proc = sd.play(video, at_epoch=1000.0 + 4.0, popen=fake_popen,
                   now=clk.now, sleep=clk.sleep,
                   xrandr_text=SHOW_XRANDR, which=which_of("vlc"))
    assert len(spawned) == 1
    argv, launched_at = spawned[0]
    assert launched_at == pytest.approx(1004.0, abs=0.06)       # not before tick0-anchor
    assert "--qt-fullscreen-screennumber=1" in argv and argv[-1] == str(video)
    assert proc in sd._LIVE                                     # registered for abort


def test_play_without_at_epoch_launches_immediately(tmp_path):
    video = tmp_path / "sbs.mp4"
    video.write_bytes(b"\0")
    clk = FakeClock(2000.0)
    spawned = []
    sd.play(video, at_epoch=None, popen=lambda a: spawned.append(a) or FakeProc(),
            now=clk.now, sleep=clk.sleep, xrandr_text=DEV_XRANDR, which=which_of("vlc"))
    assert len(spawned) == 1 and clk.t == 2000.0               # no waiting


def test_play_missing_video_is_actionable(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        sd.play(tmp_path / "nope.mp4", popen=lambda a: FakeProc(),
                xrandr_text=DEV_XRANDR, which=which_of("vlc"))


# ---- abort -----------------------------------------------------------------------------
def test_stop_all_terminates_live_player(tmp_path):
    video = tmp_path / "sbs.mp4"
    video.write_bytes(b"\0")
    proc = sd.play(video, popen=lambda a: FakeProc(),
                   xrandr_text=SHOW_XRANDR, which=which_of("vlc"))
    assert proc in sd._LIVE
    sd._stop_all()
    assert proc.terminated is True and sd._LIVE == []


# ---- CLI surface -----------------------------------------------------------------------
def test_main_requires_a_video(monkeypatch):
    monkeypatch.delenv("SHOW_VIDEO", raising=False)
    with pytest.raises(SystemExit, match="no video"):
        sd.main([])
