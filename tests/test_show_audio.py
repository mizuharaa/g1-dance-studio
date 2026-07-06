"""Offline tests for the show-audio cue engine — no robot, no DDS, no SDK.

pipeline/show_audio.py starts the show music in lockstep with the policy
(tick0 + 2.5s activation ramp + 1.5s standing lead-in = 4.0s). These tests pin:
  * the timeline math (offset from a dance record's audio.align, latency comp,
    the RAMP_S cross-check against pipeline/deploy_ramp);
  * the robot stream format + chunking (16kHz mono s16, sample-aligned chunks);
  * mode dispatch and the exact SDK calls each mode makes — with the AudioClient
    fully faked (pattern: tests/test_arm_dance.py). connect() is never called
    with a real client, so DDS is never initialized here;
  * abort behaviour: stop() silences PlayStream / turns the LED off.

The SDK contract being faked (from ~/robot/unitree_sdk2_python, read-only):
    PlayStream(app_name, stream_id, pcm bytes) -> (code, data)   # g1_audio_client.py:63
    PlayStop(app_name)  SetVolume(v)  LedControl(R,G,B)          # :68 :47 :54
"""
import json
import math
import struct
import wave
from pathlib import Path

import pytest

sa = pytest.importorskip("pipeline.show_audio")

THRILLER_RECORD = Path(__file__).resolve().parent.parent / \
    "data/dances/20260704-18f65bbd/dance.json"


# ---- module hygiene --------------------------------------------------------------------
def test_sdk_only_imported_lazily():
    """unitree_sdk2py must never be imported at module import (tests/CI have no DDS;
    importing it can open sockets). Every sdk import line must live inside a function."""
    src = Path(sa.__file__).read_text().splitlines()
    offenders = [l for l in src
                 if "unitree_sdk2py" in l and "import" in l and not l.startswith((" ", "\t"))]
    assert offenders == []


# ---- timeline math ---------------------------------------------------------------------
def test_ramp_constant_matches_deploy_ramp():
    deploy_ramp = pytest.importorskip("pipeline.deploy_ramp")
    assert sa.RAMP_S == deploy_ramp.RAMP_S


def test_default_offset_is_the_contract_4s():
    assert sa.cue_offset_for_align(None) == pytest.approx(4.0)
    assert sa.DEFAULT_OFFSET_S == pytest.approx(4.0)


def test_offset_reads_align_from_record():
    assert sa.cue_offset_for_align({"audio_delay_s": 1.5}) == pytest.approx(4.0)
    assert sa.cue_offset_for_align({"audio_delay_s": 2.25}) == pytest.approx(4.75)
    with pytest.raises(ValueError):
        sa.cue_offset_for_align({"audio_delay_s": -1.0})


@pytest.mark.skipif(not THRILLER_RECORD.exists(), reason="thriller record absent")
def test_thriller_record_offset_is_4s():
    align = json.loads(THRILLER_RECORD.read_text())["audio"]["align"]
    assert sa.cue_offset_for_align(align) == pytest.approx(4.0)


def test_compute_start_time():
    assert sa.compute_start_time(100.0, 4.0) == pytest.approx(104.0)
    assert sa.compute_start_time(100.0, 4.0, latency_comp_s=0.25) == pytest.approx(103.75)
    assert sa.compute_start_time(100.0, 4.0, lead_s=0.4) == pytest.approx(103.6)
    with pytest.raises(ValueError):  # comp so large the cue would beat tick0 = config bug
        sa.compute_start_time(100.0, 4.0, latency_comp_s=5.0)


def test_latency_comp_env_knob():
    assert sa.latency_comp_from_env({}) == 0.0
    assert sa.latency_comp_from_env({"AUDIO_LATENCY_COMP": "0.35"}) == pytest.approx(0.35)
    with pytest.raises(SystemExit):
        sa.latency_comp_from_env({"AUDIO_LATENCY_COMP": "fast"})
    with pytest.raises(SystemExit):
        sa.latency_comp_from_env({"AUDIO_LATENCY_COMP": "-0.2"})


# ---- PCM format + chunking -------------------------------------------------------------
def test_chunking_preserves_bytes_and_sample_alignment():
    pcm = bytes(range(256)) * 782  # 200192 bytes
    chunks = sa.chunk_pcm(pcm)
    assert b"".join(chunks) == pcm
    assert all(len(c) <= sa.CHUNK_BYTES for c in chunks)
    assert all(len(c) % 2 == 0 for c in chunks)  # never tear an int16 sample
    assert [len(c) for c in chunks[:-1]] == [sa.CHUNK_BYTES] * (len(chunks) - 1)
    with pytest.raises(ValueError):
        sa.chunk_pcm(pcm, chunk_bytes=95999)  # odd -> would tear samples
    with pytest.raises(ValueError):
        sa.chunk_pcm(pcm, chunk_bytes=0)


def test_pcm_duration():
    one_second = b"\x00\x00" * sa.PCM_RATE
    assert sa.pcm_duration_s(one_second) == pytest.approx(1.0)
    assert sa.chunk_pcm(one_second * 3, sa.CHUNK_BYTES) == [one_second * 3]  # 3s = 1 chunk


def _write_test_wav(path: Path, seconds: float = 1.0, rate: int = 44100, channels: int = 2):
    """A stereo 44.1kHz sine — deliberately NOT the robot format."""
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / rate))) * channels
            for i in range(n))
        w.writeframes(frames)


def _have_ffmpeg() -> bool:
    try:
        sa.find_ffmpeg()
        return True
    except SystemExit:
        return False


@pytest.mark.ffmpeg
@pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg unavailable")
def test_wav_converts_to_16k_mono_s16(tmp_path):
    wav = tmp_path / "song.wav"
    _write_test_wav(wav, seconds=1.0)
    pcm = sa.wav_to_pcm16(wav)
    expect = sa.PCM_RATE * sa.PCM_SAMPLE_BYTES  # 1.0s of 16kHz mono s16 = 32000 B
    assert abs(len(pcm) - expect) <= expect * 0.02
    assert len(pcm) % 2 == 0
    # sanity: it still looks like a sine (nonzero, bounded int16)
    samples = struct.unpack(f"<{len(pcm)//2}h", pcm)
    assert max(samples) > 5000 and min(samples) < -5000


def test_wav_to_pcm16_missing_file_is_clear(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        sa.wav_to_pcm16(tmp_path / "nope.wav")


# ---- fakes -----------------------------------------------------------------------------
class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def now(self):
        return self.t

    def sleep(self, s):
        assert s >= 0
        self.t += s


class FakeAudioClient:
    """Records every SDK call with the fake-clock timestamp. Mirrors the real
    AudioClient return conventions (PlayStream -> (code, data), others -> code)."""

    def __init__(self, clock=None, fail_stream_at=None):
        self.clock = clock
        self.calls = []
        self.fail_stream_at = fail_stream_at

    def _t(self):
        return self.clock.t if self.clock else None

    def PlayStream(self, app_name, stream_id, pcm_data):
        i = sum(1 for c in self.calls if c[0] == "PlayStream")
        self.calls.append(("PlayStream", app_name, stream_id, bytes(pcm_data), self._t()))
        return (7 if self.fail_stream_at == i else 0), None

    def PlayStop(self, app_name):
        self.calls.append(("PlayStop", app_name, self._t()))
        return 0

    def SetVolume(self, volume):
        self.calls.append(("SetVolume", volume, self._t()))
        return 0

    def LedControl(self, R, G, B):
        self.calls.append(("LedControl", (R, G, B), self._t()))
        return 0


# ---- wait_until ------------------------------------------------------------------------
def test_wait_until_sleeps_to_target_and_reports_lateness():
    clk = FakeClock(1000.0)
    late = sa.wait_until(1004.0, now=clk.now, sleep=clk.sleep)
    assert clk.t == pytest.approx(1004.0, abs=0.06)
    assert 0.0 <= late < 0.06
    # already-past target: no sleep, positive lateness
    late = sa.wait_until(1000.0, now=clk.now, sleep=clk.sleep)
    assert late == pytest.approx(clk.t - 1000.0, abs=1e-9)


# ---- robot speaker mode ----------------------------------------------------------------
def test_robot_stream_sends_exact_pcm_in_order():
    clk = FakeClock()
    fake = FakeAudioClient(clk)
    pcm = b"\x01\x02" * 80000  # 160000 B = 96000 + 64000 -> 2 chunks
    cue = sa.RobotSpeakerCue(Path("x.wav"), client=fake, sleep=clk.sleep, now=clk.now)
    cue.pcm = pcm
    cue.connect()  # injected client -> returns it, NO DDS
    sid = cue.fire()
    streams = [c for c in fake.calls if c[0] == "PlayStream"]
    assert len(streams) == 2
    assert all(c[1] == sa.APP_NAME for c in streams)
    assert {c[2] for c in streams} == {sid}          # one stream id for the whole song
    assert b"".join(c[3] for c in streams) == pcm    # byte-exact payload
    # example-proven pacing: 1s between sends
    assert streams[1][4] - streams[0][4] == pytest.approx(sa.SEND_INTERVAL_S)


def test_robot_stream_error_code_raises():
    clk = FakeClock()
    fake = FakeAudioClient(clk, fail_stream_at=1)
    cue = sa.RobotSpeakerCue(Path("x.wav"), client=fake, sleep=clk.sleep, now=clk.now)
    cue.pcm = b"\x00\x00" * 96000  # 3 chunks
    with pytest.raises(RuntimeError, match="code 7"):
        cue.fire()


def test_robot_fire_refuses_without_prepare_or_client():
    cue = sa.RobotSpeakerCue(Path("x.wav"), client=FakeAudioClient())
    with pytest.raises(RuntimeError, match="prepare"):
        cue.fire()
    cue2 = sa.RobotSpeakerCue(Path("x.wav"))
    cue2.pcm = b"\x00\x00"
    with pytest.raises(RuntimeError, match="connect"):
        cue2.fire()


def test_robot_stop_playstops():
    fake = FakeAudioClient()
    cue = sa.RobotSpeakerCue(Path("x.wav"), client=fake)
    cue.stop()
    assert ("PlayStop", sa.APP_NAME, None) in fake.calls


# ---- LED cue mode ----------------------------------------------------------------------
def test_led_cue_countdown_then_green_at_the_moment():
    clk = FakeClock(1000.0)
    fake = FakeAudioClient(clk)
    cue = sa.LedCue(client=fake, sleep=clk.sleep, now=clk.now)
    cue.connect()
    target = 1005.0
    cue.run(target)
    leds = [c for c in fake.calls if c[0] == "LedControl"]
    colors = [c[1] for c in leds]
    # 3 blue countdown flashes (each blue->off), then GREEN, then off
    assert colors == [(0, 0, 255), (0, 0, 0)] * 3 + [(0, 255, 0), (0, 0, 0)]
    green_t = leds[6][2]
    assert green_t == pytest.approx(target, abs=0.06)   # green IS the cue moment
    blue_ts = [leds[i][2] for i in (0, 2, 4)]
    assert blue_ts == pytest.approx([target - 3, target - 2, target - 1], abs=0.06)
    assert colors[-1] == (0, 0, 0)                      # always ends dark


def test_led_stop_turns_off():
    fake = FakeAudioClient()
    cue = sa.LedCue(client=fake)
    cue.stop()
    assert fake.calls[-1][0] == "LedControl" and fake.calls[-1][1] == (0, 0, 0)


# ---- mode dispatch (run_cue) -----------------------------------------------------------
def test_run_cue_rejects_unknown_mode():
    with pytest.raises(SystemExit, match="AUDIO_MODE"):
        sa.run_cue("boombox", t0_epoch=0.0, offset_s=4.0)


def test_run_cue_robot_needs_a_track():
    with pytest.raises(SystemExit, match="track"):
        sa.run_cue("robot", t0_epoch=0.0, offset_s=4.0, track=None)


def test_run_cue_banner_fires_after_offset_minus_lead(capsys):
    clk = FakeClock(2000.0)
    rep = sa.run_cue("banner", t0_epoch=2000.0, offset_s=4.0, cue_lead_s=0.4,
                     now=clk.now, sleep=clk.sleep)
    assert "PLAY MUSIC NOW" in capsys.readouterr().out
    assert clk.t == pytest.approx(2003.6, abs=0.06)
    assert rep["late_s"] >= 0


def test_run_cue_led_dispatch():
    clk = FakeClock(3000.0)
    fake = FakeAudioClient(clk)
    rep = sa.run_cue("led", t0_epoch=3000.0, offset_s=4.0, client=fake,
                     now=clk.now, sleep=clk.sleep)
    assert rep["sequence"][-1] == (0, 0, 0)
    assert (0, 255, 0) in rep["sequence"]
    green_i = next(i for i, c in enumerate(fake.calls) if c[1] == (0, 255, 0))
    assert fake.calls[green_i][2] == pytest.approx(3004.0, abs=0.06)


def test_run_cue_robot_end_to_end_timing(monkeypatch):
    """Full robot-mode path with the SDK faked: convert (mocked), wait to
    t0+offset-latency, stream, hold past the buffered tail, PlayStop."""
    clk = FakeClock(5000.0)
    fake = FakeAudioClient(clk)
    two_s = b"\x00\x00" * (2 * sa.PCM_RATE)
    monkeypatch.setattr(sa, "wav_to_pcm16", lambda track, ffmpeg=None: two_s)
    rep = sa.run_cue("robot", t0_epoch=5000.0, offset_s=4.0, track=Path("song.wav"),
                     latency_comp_s=0.25, volume=70, client=fake,
                     now=clk.now, sleep=clk.sleep)
    assert rep["duration_s"] == pytest.approx(2.0)
    assert ("SetVolume", 70) == fake.calls[0][:2]
    first_chunk = next(c for c in fake.calls if c[0] == "PlayStream")
    assert first_chunk[4] == pytest.approx(5000.0 + 4.0 - 0.25, abs=0.06)  # latency comp
    stop = next(c for c in fake.calls if c[0] == "PlayStop")
    # stream closed only after the tail has PLAYED (target + duration + 2s margin)
    assert stop[2] >= 5003.75 + 2.0 + 2.0 - 0.06
    assert rep["late_s"] >= 0


def test_run_cue_laptop_spawns_player_at_cue_time(tmp_path):
    clk = FakeClock(7000.0)
    track = tmp_path / "music.wav"
    track.write_bytes(b"RIFF")
    spawned = []

    class FakeProc:
        def __init__(self):
            self.terminated = False

        def wait(self):
            return 0

        def poll(self):
            return 0

    def fake_popen(argv):
        spawned.append((list(argv), clk.t))
        return FakeProc()

    cue = sa.LaptopCue(track, player=["true"], popen=fake_popen)
    # drive it the way run_cue does, but with the fake clock
    target = sa.compute_start_time(7000.0, 4.0, 0.1)
    sa.wait_until(target, now=clk.now, sleep=clk.sleep)
    cue.fire()
    cue.wait()
    assert spawned[0][0][-1] == str(track)
    assert spawned[0][1] == pytest.approx(7003.9, abs=0.06)


# ---- dance record resolution -----------------------------------------------------------
def test_load_dance_audio_resolves_track_and_align(tmp_path):
    data = tmp_path / "data"
    ddir = data / "dances" / "d1"
    (ddir / "audio").mkdir(parents=True)
    (ddir / "audio" / "music.wav").write_bytes(b"RIFF")
    (ddir / "dance.json").write_text(json.dumps({
        "id": "d1",
        "audio": {"track": "data/dances/d1/audio/music.wav",
                  "align": {"audio_delay_s": 1.5}}}))
    track, align = sa.load_dance_audio("d1", data_dir=data)
    assert track.is_file()
    assert sa.cue_offset_for_align(align) == pytest.approx(4.0)


def test_load_dance_audio_missing_audio_is_actionable(tmp_path):
    data = tmp_path / "data"
    ddir = data / "dances" / "d2"
    ddir.mkdir(parents=True)
    (ddir / "dance.json").write_text(json.dumps({"id": "d2"}))
    with pytest.raises(SystemExit, match="no attached audio"):
        sa.load_dance_audio("d2", data_dir=data)
    with pytest.raises(SystemExit, match="no dance record"):
        sa.load_dance_audio("nope", data_dir=data)


@pytest.mark.skipif(not THRILLER_RECORD.exists(), reason="thriller record absent")
def test_thriller_track_resolves_to_real_file():
    track, align = sa.load_dance_audio("20260704-18f65bbd")
    assert track.is_file()
    assert align["audio_delay_s"] == pytest.approx(1.5)


# ---- CLI surface ----------------------------------------------------------------------
def test_cli_banner_cue_smoke(capsys):
    import time as _time
    rc = sa.main(["cue", "--mode", "banner", "--t0-epoch", str(_time.time() - 10),
                  "--offset", "4.0"])
    out = capsys.readouterr().out
    assert rc == 0 and "PLAY MUSIC NOW" in out and "armed: mode=banner" in out


@pytest.mark.skipif(not THRILLER_RECORD.exists(), reason="thriller record absent")
def test_cli_offset_subcommand(capsys):
    rc = sa.main(["offset", "--dance-id", "20260704-18f65bbd"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["offset_s"] == pytest.approx(4.0)
