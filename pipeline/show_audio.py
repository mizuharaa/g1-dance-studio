#!/usr/bin/env python
"""Show-time audio/cue engine: start the music in lockstep with the robot.

Today's rehearsal proved the operator watches the ROBOT, not the terminal — a
printed "PLAY NOW" banner is useless (user: "had no idea when to start the
music"). The product answer is that the robot itself carries the show audio
(its speaker travels with it) and, failing that, the robot itself becomes the
cue light. Four modes, selected by AUDIO_MODE in tools/show_run.sh:

  robot  — stream the dance's music out of the G1's own chest speaker
           (AudioClient.PlayStream over DDS). THE product mode.
  led    — flash the G1 head LED as a silent visual cue (AudioClient
           LedControl): blue countdown flashes at T-3/-2/-1, GREEN at the
           music moment. Operator presses play on an external speaker when
           the head turns green.
  laptop — play the track through the laptop (paplay) — once the SOF
           firmware fix lands (docs/SHOW_AUDIO.md).
  banner — legacy terminal bell + banner (tools/rehearsal_cue.sh behaviour).

TIMELINE CONTRACT (docs/SHOW_AUDIO.md):
    music_start = tick0 + RAMP_S (2.5 s activation ramp, pipeline/deploy_ramp)
                        + audio_delay_s (1.5 s standing lead-in, dance record
                          audio.align — pipeline/audio.py)
                = tick0 + 4.0 s for the default prep.
    tick0 is anchored by the wrapper: it captures `date +%s.%N` the moment
    pipeline/deploy_runtime prints its "starting leg-odometry policy" line and
    passes it as --t0-epoch, so python startup time cannot skew the cue.
    AUDIO_LATENCY_COMP (seconds, default 0) shifts the robot/laptop audio
    EARLIER to absorb the playback chain's startup latency — calibrate on
    hardware (next-session checklist in docs/SHOW_AUDIO.md).

SDK CONTRACT (verified against ~/robot/unitree_sdk2_python, READ-ONLY):
  unitree_sdk2py/g1/audio/g1_audio_client.py
    :63 PlayStream(app_name, stream_id, pcm_data: bytes) -> (code, data)
    :68 PlayStop(app_name)              :47 SetVolume(volume)
    :54 LedControl(R, G, B)             service "voice", api ids 1001..1010
  example/g1/audio/g1_audio_client_play_wav.py:25 — the stream MUST be
  16 kHz mono 16-bit PCM; example/g1/audio/wav.py:125 streams it in
  96000-byte chunks (3 s) with 1.0 s sleeps between sends.

SAFETY: this module NEVER touches the robot at import time. DDS init happens
only inside RobotCueBase.connect(), which tests never call (they inject fake
clients). Audio/LED cannot move the robot; motion still goes through
deploy_runtime's own human-confirmation gates.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- timeline constants ---------------------------------------------------------------
# MUST match pipeline/deploy_ramp.py RAMP_S (2.5 s activation ramp prepended to every
# deployable CSV) — cross-checked by tests/test_show_audio.py so drift cannot go unseen.
RAMP_S = 2.5
# Default standing lead-in if a dance record carries no audio.align (pipeline/audio.py
# PREP_PAD_IN_S + PREP_BLEND_IN_S).
DEFAULT_AUDIO_DELAY_S = 1.5
DEFAULT_OFFSET_S = RAMP_S + DEFAULT_AUDIO_DELAY_S  # 4.0

# ---- robot stream format (from the SDK example, see module docstring) ------------------
PCM_RATE = 16000
PCM_CHANNELS = 1
PCM_SAMPLE_BYTES = 2
CHUNK_BYTES = 96000          # 3 s at 16 kHz mono s16 — example-proven
SEND_INTERVAL_S = 1.0        # example-proven pacing between chunk sends
APP_NAME = "g1dance"
IFACE = "enp0s31f6"          # same NIC as pipeline/deploy_runtime.IFACE

FFMPEG_FALLBACKS = [
    Path.home() / "miniconda3/envs/g1dance/bin/ffmpeg",
    Path.home() / "miniconda3/envs/tv/bin/ffmpeg",
]


# ---- pure helpers (unit-tested, no I/O) ------------------------------------------------
def cue_offset_for_align(align: dict | None, ramp_s: float = RAMP_S) -> float:
    """Seconds from policy tick0 to music start. align is the dance record's
    audio.align dict (pipeline/audio.py AudioAlignment) or None."""
    delay = DEFAULT_AUDIO_DELAY_S if not align else float(align["audio_delay_s"])
    if delay < 0 or ramp_s < 0:
        raise ValueError("negative timeline component")
    return ramp_s + delay


def compute_start_time(t0_epoch: float, offset_s: float, latency_comp_s: float = 0.0,
                       lead_s: float = 0.0) -> float:
    """Wall-clock time the cue must FIRE: tick0 + offset, pulled earlier by the
    playback chain's startup latency (robot/laptop) or a human-reaction lead
    (banner). Refuses to fire before tick0 — a comp that large is a config bug."""
    t = t0_epoch + offset_s - latency_comp_s - lead_s
    if t < t0_epoch:
        raise ValueError(
            f"cue would fire {t0_epoch - t:.2f}s before tick0 — "
            f"offset {offset_s} vs latency_comp {latency_comp_s} + lead {lead_s}")
    return t


def latency_comp_from_env(env: dict | None = None) -> float:
    """AUDIO_LATENCY_COMP env knob (seconds, >= 0, < offset). Default 0.0."""
    raw = (env if env is not None else os.environ).get("AUDIO_LATENCY_COMP", "0.0")
    try:
        v = float(raw)
    except ValueError:
        raise SystemExit(f"AUDIO_LATENCY_COMP must be a number, got {raw!r}")
    if v < 0:
        raise SystemExit(f"AUDIO_LATENCY_COMP must be >= 0, got {v}")
    return v


def chunk_pcm(pcm: bytes, chunk_bytes: int = CHUNK_BYTES) -> list[bytes]:
    """Split a PCM byte string into send-chunks. Chunk boundaries stay
    sample-aligned (even byte counts) so no int16 sample is ever torn."""
    if chunk_bytes < PCM_SAMPLE_BYTES or chunk_bytes % PCM_SAMPLE_BYTES:
        raise ValueError(f"chunk_bytes must be a positive multiple of "
                         f"{PCM_SAMPLE_BYTES}, got {chunk_bytes}")
    return [pcm[i:i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)]


def pcm_duration_s(pcm: bytes) -> float:
    return len(pcm) / (PCM_RATE * PCM_CHANNELS * PCM_SAMPLE_BYTES)


def find_ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    for p in FFMPEG_FALLBACKS:
        if p.is_file():
            return str(p)
    raise SystemExit("ffmpeg not found (PATH or known conda envs) — needed to "
                     "convert music to the robot's 16kHz mono PCM format")


def wav_to_pcm16(track: Path, ffmpeg: str | None = None) -> bytes:
    """Convert any audio file to the robot stream format: raw s16le 16 kHz mono.
    (The SDK example rejects everything but 16 kHz mono 16-bit — see docstring.)"""
    track = Path(track)
    if not track.is_file():
        raise SystemExit(f"music track not found: {track}")
    out = subprocess.run(
        [ffmpeg or find_ffmpeg(), "-v", "error", "-i", str(track),
         "-f", "s16le", "-acodec", "pcm_s16le",
         "-ar", str(PCM_RATE), "-ac", str(PCM_CHANNELS), "-"],
        check=True, capture_output=True)
    pcm = out.stdout
    if len(pcm) < PCM_RATE * PCM_SAMPLE_BYTES // 10:  # < 0.1 s is not music
        raise SystemExit(f"conversion of {track} produced only {len(pcm)} bytes")
    if len(pcm) % PCM_SAMPLE_BYTES:
        pcm = pcm[:-(len(pcm) % PCM_SAMPLE_BYTES)]
    return pcm


def load_dance_audio(dance_id: str, data_dir: Path | None = None) -> tuple[Path, dict | None]:
    """Resolve a dance record's music track path + audio.align dict."""
    data = data_dir or (PROJECT_ROOT / "data")
    rec_path = data / "dances" / dance_id / "dance.json"
    if not rec_path.is_file():
        raise SystemExit(f"no dance record at {rec_path}")
    rec = json.loads(rec_path.read_text())
    audio = rec.get("audio") or {}
    track = audio.get("track")
    if not track:
        raise SystemExit(f"dance {dance_id} has no attached audio track — "
                         "attach music first (pipeline/audio.py)")
    tp = Path(track)
    if not tp.is_absolute():
        tp = (data.parent / tp) if not (data / track).exists() else data / track
    return tp, audio.get("align")


# ---- waiting --------------------------------------------------------------------------
def wait_until(target_epoch: float, *, now=time.time, sleep=time.sleep) -> float:
    """Sleep until wall-clock target_epoch (coarse sleeps then fine). Returns the
    (signed) firing error in seconds — >=0 means we fired at/after target."""
    while True:
        dt = target_epoch - now()
        if dt <= 0:
            return -dt
        sleep(min(dt, 0.05) if dt < 0.25 else dt - 0.2)


# ---- cue implementations ---------------------------------------------------------------
class RobotCueBase:
    """Shared lazy-DDS plumbing. `client` is injectable so tests NEVER touch DDS."""

    def __init__(self, iface: str = IFACE, client=None):
        self.iface = iface
        self.client = client

    def connect(self, timeout_s: float = 10.0):
        """DDS init + AudioClient handshake. ROBOT-FACING — never in tests."""
        if self.client is not None:
            return self.client
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
        ChannelFactoryInitialize(0, self.iface)
        c = AudioClient()
        c.SetTimeout(timeout_s)
        c.Init()
        self.client = c
        return c


class RobotSpeakerCue(RobotCueBase):
    """Stream the music out of the G1's own speaker (PlayStream)."""

    def __init__(self, track: Path, iface: str = IFACE, client=None, *,
                 chunk_bytes: int = CHUNK_BYTES, send_interval_s: float = SEND_INTERVAL_S,
                 sleep=time.sleep, now=time.time):
        super().__init__(iface, client)
        self.track = Path(track)
        self.chunk_bytes = int(chunk_bytes)
        self.send_interval_s = float(send_interval_s)
        self._sleep, self._now = sleep, now
        self.pcm: bytes = b""
        self.stream_id: str | None = None

    def prepare(self) -> float:
        """Convert the track up front (before the cue moment). Returns duration s."""
        self.pcm = wav_to_pcm16(self.track)
        return pcm_duration_s(self.pcm)

    def set_volume(self, volume: int):
        code = self.client.SetVolume(int(volume))
        if code != 0:
            print(f"[show_audio] WARNING SetVolume({volume}) -> code {code}")

    def fire(self) -> str:
        """Send the PCM stream, first chunk NOW. Example-proven pacing: chunks of
        `chunk_bytes` every `send_interval_s` (robot side buffers). Returns stream id."""
        if not self.pcm:
            raise RuntimeError("fire() before prepare()")
        if self.client is None:
            raise RuntimeError("fire() before connect()")
        self.stream_id = str(int(self._now() * 1000))
        for i, chunk in enumerate(chunk_pcm(self.pcm, self.chunk_bytes)):
            code, _ = self.client.PlayStream(APP_NAME, self.stream_id, chunk)
            if code != 0:
                raise RuntimeError(f"PlayStream chunk {i} failed with code {code} "
                                   "(is the audio service up? volume set?)")
            if i == 0:
                print(f"[show_audio] first chunk sent (stream {self.stream_id}) — "
                      "music should be starting NOW")
            self._sleep(self.send_interval_s)
        return self.stream_id

    def stop(self):
        """Immediate stop (abort path) / end-of-show stream close."""
        if self.client is not None:
            self.client.PlayStop(APP_NAME)


class LedCue(RobotCueBase):
    """The robot's head becomes the cue light: blue countdown flashes at
    T-3/-2/-1 s, solid GREEN exactly at the music moment (operator presses
    play on green), then off."""

    COUNTDOWN_AT_S = (3.0, 2.0, 1.0)
    FLASH_S = 0.15
    GREEN_HOLD_S = 1.2

    def __init__(self, iface: str = IFACE, client=None, *, sleep=time.sleep, now=time.time):
        super().__init__(iface, client)
        self._sleep, self._now = sleep, now
        self.sequence: list[tuple[int, int, int]] = []  # for tests/inspection

    def _led(self, r: int, g: int, b: int):
        self.sequence.append((r, g, b))
        code = self.client.LedControl(r, g, b)
        if code not in (0, None):
            print(f"[show_audio] WARNING LedControl({r},{g},{b}) -> code {code}")

    def run(self, target_epoch: float):
        """Blocking: countdown flashes, then green at target_epoch, then off."""
        if self.client is None:
            raise RuntimeError("run() before connect()")
        for t_minus in self.COUNTDOWN_AT_S:
            wait_until(target_epoch - t_minus, now=self._now, sleep=self._sleep)
            self._led(0, 0, 255)
            self._sleep(self.FLASH_S)
            self._led(0, 0, 0)
        wait_until(target_epoch, now=self._now, sleep=self._sleep)
        self._led(0, 255, 0)
        print("[show_audio] LED GREEN — operator starts the music NOW")
        self._sleep(self.GREEN_HOLD_S)
        self._led(0, 0, 0)

    def stop(self):
        if self.client is not None:
            self.client.LedControl(0, 0, 0)


class LaptopCue:
    """Play the track through the laptop (paplay) — needs working laptop audio
    (SOF firmware fix, docs/SHOW_AUDIO.md)."""

    def __init__(self, track: Path, player: list[str] | None = None, popen=subprocess.Popen):
        self.track = Path(track)
        self.player = player or ["paplay"]
        self._popen = popen
        self.proc = None

    def prepare(self):
        if not self.track.is_file():
            raise SystemExit(f"music track not found: {self.track}")
        if not shutil.which(self.player[0]):
            raise SystemExit(f"{self.player[0]} not found — laptop audio mode needs it")

    def fire(self):
        self.proc = self._popen([*self.player, str(self.track)])

    def wait(self):
        if self.proc is not None:
            self.proc.wait()

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()


class BannerCue:
    """Legacy terminal cue (rehearsal_cue.sh behaviour) — kept as last resort."""

    def fire(self):
        sys.stdout.write("\a\a\a\n")
        print("██████████████████████████████████████")
        print("██  ▶▶▶  PLAY MUSIC NOW  ◀◀◀        ██")
        print("██████████████████████████████████████")
        sys.stdout.flush()

    def stop(self):
        pass


MODES = ("robot", "led", "laptop", "banner")


# ---- orchestration ---------------------------------------------------------------------
def run_cue(mode: str, *, t0_epoch: float, offset_s: float, track: Path | None = None,
            iface: str = IFACE, latency_comp_s: float = 0.0, cue_lead_s: float = 0.4,
            volume: int | None = None, client=None,
            now=time.time, sleep=time.sleep) -> dict:
    """The wrapper's entry point: given the tick0 anchor (wall-clock epoch seconds
    captured by tools/show_run.sh the moment deploy_runtime printed its policy-start
    line), fire the selected cue at tick0 + offset. Blocking; returns a report dict.

    A SIGTERM/SIGINT during the run stops audio/LED immediately (the wrapper sends
    SIGTERM when the runtime aborts) — the handler is installed by main(), not here.
    """
    if mode not in MODES:
        raise SystemExit(f"unknown AUDIO_MODE {mode!r} (want one of {MODES})")
    report = {"mode": mode, "t0_epoch": t0_epoch, "offset_s": offset_s}

    if mode == "banner":
        cue = BannerCue()
        target = compute_start_time(t0_epoch, offset_s, lead_s=cue_lead_s)
        _CURRENT_CUE.append(cue)
        report["late_s"] = wait_until(target, now=now, sleep=sleep)
        cue.fire()
        return report

    if mode == "led":
        cue = LedCue(iface, client, sleep=sleep, now=now)
        target = compute_start_time(t0_epoch, offset_s)
        cue.connect()
        _CURRENT_CUE.append(cue)
        cue.run(target)
        report["sequence"] = list(cue.sequence)
        return report

    if track is None:
        raise SystemExit(f"mode {mode} needs a music track")

    if mode == "laptop":
        cue = LaptopCue(track)
        cue.prepare()
        target = compute_start_time(t0_epoch, offset_s, latency_comp_s)
        _CURRENT_CUE.append(cue)
        report["late_s"] = wait_until(target, now=now, sleep=sleep)
        cue.fire()
        cue.wait()
        return report

    # mode == "robot"
    cue = RobotSpeakerCue(track, iface, client, sleep=sleep, now=now)
    dur = cue.prepare()                      # convert BEFORE the cue moment
    report["duration_s"] = dur
    target = compute_start_time(t0_epoch, offset_s, latency_comp_s)
    cue.connect()                            # DDS up BEFORE the cue moment
    _CURRENT_CUE.append(cue)
    if volume is not None:
        cue.set_volume(volume)
    report["late_s"] = wait_until(target, now=now, sleep=sleep)
    cue.fire()
    # hold until the buffered tail has PLAYED, then close the stream. PlayStop right
    # after the last send could truncate robot-side buffered audio (sends run ~3x
    # faster than real time).
    wait_until(target + dur + 2.0, now=now, sleep=sleep)
    cue.stop()
    return report


# a stack of live cues so the signal handler can stop them (single-threaded CLI use)
_CURRENT_CUE: list = []


def _abort_handler(signum, frame):  # pragma: no cover - exercised via unit call
    print(f"\n[show_audio] signal {signum} -> stopping audio/LED")
    while _CURRENT_CUE:
        try:
            _CURRENT_CUE.pop().stop()
        except Exception as e:  # noqa: BLE001 — best effort, we are dying anyway
            print(f"[show_audio] stop failed: {e}")
    os._exit(130)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cue", help="fire the show cue at t0+offset (wrapper entry point)")
    c.add_argument("--mode", default=os.environ.get("AUDIO_MODE", "banner"), choices=MODES)
    c.add_argument("--t0-epoch", type=float, required=True,
                   help="wall-clock epoch seconds of policy tick0 (date +%%s.%%N at the "
                        "runtime's policy-start line)")
    c.add_argument("--dance-id", help="dance record id — track + align from dance.json")
    c.add_argument("--track", type=Path, help="explicit music file (overrides --dance-id)")
    c.add_argument("--offset", type=float, default=None,
                   help="seconds tick0->music (default: 2.5 ramp + record's audio_delay_s)")
    c.add_argument("--iface", default=os.environ.get("G1_IFACE", IFACE))
    c.add_argument("--cue-lead", type=float, default=float(os.environ.get("CUE_LEAD", "0.4")),
                   help="banner mode: fire this early (human reaction time)")
    c.add_argument("--volume", type=int,
                   default=int(os.environ["AUDIO_VOLUME"]) if os.environ.get("AUDIO_VOLUME") else None,
                   help="robot mode: SetVolume 0-100 before playing")

    s = sub.add_parser("stop", help="emergency: stop robot playback + LED off")
    s.add_argument("--iface", default=os.environ.get("G1_IFACE", IFACE))

    o = sub.add_parser("offset", help="print the tick0->music offset for a dance")
    o.add_argument("--dance-id", required=True)

    v = sub.add_parser("convert", help="debug: convert a track to robot PCM, print stats")
    v.add_argument("--track", type=Path, required=True)

    args = ap.parse_args(argv)

    if args.cmd == "offset":
        _, align = load_dance_audio(args.dance_id)
        print(json.dumps({"offset_s": cue_offset_for_align(align),
                          "ramp_s": RAMP_S,
                          "audio_delay_s": (align or {}).get("audio_delay_s",
                                                             DEFAULT_AUDIO_DELAY_S)}))
        return 0

    if args.cmd == "convert":
        pcm = wav_to_pcm16(args.track)
        print(json.dumps({"track": str(args.track), "pcm_bytes": len(pcm),
                          "duration_s": round(pcm_duration_s(pcm), 3),
                          "chunks": len(chunk_pcm(pcm)),
                          "format": f"s16le {PCM_RATE}Hz mono"}))
        return 0

    if args.cmd == "stop":
        cue = RobotSpeakerCue(Path("/dev/null"), args.iface)
        cue.connect()
        cue.stop()
        cue.client.LedControl(0, 0, 0)
        print("[show_audio] PlayStop sent + LED off")
        return 0

    # cue
    track, align = None, None
    if args.track:
        track = args.track
    elif args.dance_id:
        track, align = load_dance_audio(args.dance_id)
    offset = args.offset if args.offset is not None else cue_offset_for_align(align)
    signal.signal(signal.SIGTERM, _abort_handler)
    signal.signal(signal.SIGINT, _abort_handler)
    print(f"[show_audio] armed: mode={args.mode} offset={offset:.2f}s "
          f"latency_comp={latency_comp_from_env():.3f}s track={track}")
    report = run_cue(args.mode, t0_epoch=args.t0_epoch, offset_s=offset, track=track,
                     iface=args.iface, latency_comp_s=latency_comp_from_env(),
                     cue_lead_s=args.cue_lead, volume=args.volume)
    print(f"[show_audio] done: {json.dumps({k: v for k, v in report.items() if k != 'sequence'})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
