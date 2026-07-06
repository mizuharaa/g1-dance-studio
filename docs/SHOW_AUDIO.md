# Show audio — the robot carries its own music (and its own cue light)

**Problem (2026-07-06 rehearsal):** the operator watches the ROBOT, not the
terminal. The printed "PLAY NOW" banner of `tools/rehearsal_cue.sh` was useless
(user: *"had no idea when to start the music"*). And the laptop has no working
audio output (missing SOF firmware — fixed below).

**Product answer:** the G1 has a speaker and an RGB head LED, both reachable over
the same DDS link the deploy runtime already uses. The speaker travels with the
robot, so the show audio should come FROM the robot; if an external PA must be
used, the robot's head LED becomes the cue light the operator is already looking at.

Built: `pipeline/show_audio.py` (cue engine) + `tools/show_run.sh` (wrapper,
successor of `rehearsal_cue.sh`) + `tests/test_show_audio.py` (all SDK-mocked).

## The four modes (`AUDIO_MODE`)

| mode | what happens at the music moment | needs |
|---|---|---|
| `robot` | music streams out of the G1's own speaker (`AudioClient.PlayStream`) | robot LAN up; music attached to the dance |
| `led` | head LED: blue flash at T-3/-2/-1 s, **solid GREEN exactly at the music moment**, then off. Operator presses play on the external speaker when the head turns green. | robot LAN up |
| `laptop` | `paplay` plays the track through the laptop | SOF firmware fix installed (below) |
| `banner` | legacy terminal bell + banner (default — zero new deps) | nothing |

```bash
# the product mode: robot plays its own music
AUDIO_MODE=robot AUDIO_VOLUME=85 tools/show_run.sh

# external PA, robot is the cue light
AUDIO_MODE=led tools/show_run.sh

# after the firmware fix
AUDIO_MODE=laptop tools/show_run.sh
```

`show_run.sh` runs the exact same deploy command as `rehearsal_cue.sh`
(`deploy_runtime --mode ground-run-legodom --max-secs 52 --i-will-watch-the-robot`,
all human-confirmation gates unchanged) and never touches
`pipeline/deploy_runtime.py`: it watches the runtime's stdout for the
policy-start line, anchors tick0 with `date +%s.%N` **in the shell at the moment
the line appears** (python spawn time cannot skew the cue — measured scheduling
error in dry-run: 0.1 ms), and hands that epoch to `pipeline/show_audio.py cue`.

**Abort:** any runtime `STOP:` line or exit SIGTERMs the cue helper, whose
handler immediately `PlayStop`s the speaker and turns the LED off. Music can
never keep playing over a damped robot. (Proven in dry-run with a fake runtime.)

Emergency silence by hand: `python -m pipeline.show_audio stop`

## Timeline contract

```
tick0 = policy loop start (the runtime's "starting leg-odometry policy" line)
| ACTIVATION RAMP 2.5s | PAD+BLEND-IN 1.5s |   DANCE = music 44.3s   | tail |
t0                    t0+2.5             t0+4.0                    t0+48.3
                                           ^ music_start = tick0 + 4.0 s
music_start = tick0 + RAMP_S (2.5, pipeline/deploy_ramp.py)
                    + audio.align.audio_delay_s (1.5, dance record / pipeline/audio.py)
```

The offset is read from the dance record (`data/dances/<id>/dance.json` →
`audio.align.audio_delay_s`), not hardcoded — a non-default prep stays correct.
`python -m pipeline.show_audio offset --dance-id 20260704-18f65bbd` → `4.0`.
A test cross-checks `show_audio.RAMP_S == deploy_ramp.RAMP_S` so the constants
cannot drift apart silently.

### Latency compensation (`AUDIO_LATENCY_COMP`)

The robot's playback chain (RPC → audio service → DAC) has an unknown startup
latency; the SDK documents none. `AUDIO_LATENCY_COMP` (seconds, default `0.0`)
starts the robot/laptop audio that much EARLIER. Calibrate once on hardware
(procedure below), then run shows with e.g. `AUDIO_LATENCY_COMP=0.15`.

### Robot stream format (SDK contract, verified in source)

`~/robot/unitree_sdk2_python` (READ-ONLY reference):

- `unitree_sdk2py/g1/audio/g1_audio_client.py:63` — `PlayStream(app_name,
  stream_id, pcm_data) -> (code, data)`; `:68 PlayStop`; `:47 SetVolume`;
  `:54 LedControl(R,G,B)`; service `"voice"`, api ids 1001–1010
  (`g1_audio_api.py`).
- `example/g1/audio/g1_audio_client_play_wav.py:25` — stream MUST be
  **16 kHz mono 16-bit PCM**.
- `example/g1/audio/wav.py:125` — send in 96 000-byte chunks (3 s) with 1.0 s
  sleeps; the robot buffers ahead (sends run ~3× faster than real time), so the
  stream is closed (`PlayStop`) only after the tail has *played*.

`show_audio` converts any track with ffmpeg (`-f s16le -ar 16000 -ac 1`);
Thriller's `music.wav` → 1 417 600 bytes = 44.3 s = 15 chunks (verified).

## Next-session hardware validation (~30 s of robot speaker, human present)

Robot IDLE in damp, secured, e-stop in hand. Laptop on the robot LAN
(`enp0s31f6`). **No motion involved — but a human stays present.**

```bash
cd ~/g1-dance
# 1. offline sanity (no robot): format + offset
~/miniconda3/envs/tv/bin/python -m pipeline.show_audio convert --track data/dances/20260704-18f65bbd/audio/music.wav
~/miniconda3/envs/tv/bin/python -m pipeline.show_audio offset --dance-id 20260704-18f65bbd   # -> 4.0

# 2. 5-second speaker smoke test (first ever robot-audio contact)
~/miniconda3/envs/g1dance/bin/ffmpeg -y -t 5 -i data/dances/20260704-18f65bbd/audio/music.wav /tmp/clip5.wav
AUDIO_VOLUME=70 ~/miniconda3/envs/tv/bin/python -u -m pipeline.show_audio cue \
    --mode robot --track /tmp/clip5.wav --t0-epoch "$(date +%s.%N)" --offset 1.0
#    music should come from the CHEST SPEAKER ~1 s later. If not: volume? service up?
#    silence anytime: python -m pipeline.show_audio stop

# 3. measure AUDIO_LATENCY_COMP: film the laptop screen + robot with the phone.
#    show_audio prints "first chunk sent ... music should be starting NOW" at send
#    time; the gap between that line appearing and sound in the recording is the
#    latency. Export AUDIO_LATENCY_COMP=<that> (typ. 0.0–0.5 s). Re-run step 2 to
#    confirm by ear; iterate once.

# 4. LED cue check (silent):
~/miniconda3/envs/tv/bin/python -u -m pipeline.show_audio cue \
    --mode led --t0-epoch "$(date +%s.%N)" --offset 5.0
#    expect blue flashes at 2/3/4 s, GREEN at 5 s, then off.

# 5. dress rehearsal (robot secured per ROBOT_DAY_RUNBOOK, user confirms):
AUDIO_MODE=robot AUDIO_VOLUME=85 AUDIO_LATENCY_COMP=<measured> tools/show_run.sh
#    verify the first beat lands as the robot exits the standing lead-in.
```

Knobs: `DANCE_ID` (default Thriller `20260704-18f65bbd`), `AUDIO_VOLUME` 0–100,
`CUE_LEAD` (banner human-reaction lead, default 0.4 s), `PY` (python override).

## Laptop audio firmware fix (Arrow Lake SOF)

Diagnosis (journal): `sof-audio-pci-intel-mtl 0000:00:1f.3` requests
`intel/sof-ipc4/arl/sof-arl.ri` — absent → `sof_probe_work failed err: -2` → no
sound card. The requested topology `intel/sof-ace-tplg/sof-hda-generic-2ch.tplg`
IS already installed; only the DSP blob is missing.

The Intel-signed production firmware is **staged and verified** (magic `$AE1`,
real binary, not HTML):

```
/tmp/claude-1000/-home-alois-g1-dance/9308416e-0894-4c9c-a2df-535bed1144aa/scratchpad/fw_install/sof-ipc4/arl/sof-arl.ri
    (sof-bin v2025.01, 1,002,132 bytes)
/tmp/claude-1000/-home-alois-g1-dance/9308416e-0894-4c9c-a2df-535bed1144aa/scratchpad/fw_install/sof-ipc4/sof-mtl-2025.12.2-intel-signed.ri
    (alternate: sof-bin v2025.12.2 blob, 1,080,064 bytes — ARL uses the MTL image;
     upstream sof-bin ships arl/sof-arl.ri as a symlink to mtl/intel-signed/sof-mtl.ri)
```

**Install (user, sudo). Do this BEFORE rebooting — /tmp is cleared on reboot:**

```bash
sudo install -D -m0644 \
  /tmp/claude-1000/-home-alois-g1-dance/9308416e-0894-4c9c-a2df-535bed1144aa/scratchpad/fw_install/sof-ipc4/arl/sof-arl.ri \
  /lib/firmware/intel/sof-ipc4/arl/sof-arl.ri \
&& sudo modprobe -r snd_sof_pci_intel_mtl && sudo modprobe snd_sof_pci_intel_mtl
```

If `modprobe -r` fails ("module in use"), just **reboot** — the firmware file is
already installed at that point and survives; the driver picks it up at boot.

Fallback that needs NO staged file (works offline anytime, replicates the
upstream linux-firmware symlink — the distro already ships the MTL blob at
`/lib/firmware/intel/sof-ipc4/mtl/intel-signed/sof-mtl.ri`):

```bash
sudo mkdir -p /lib/firmware/intel/sof-ipc4/arl
sudo ln -s ../mtl/intel-signed/sof-mtl.ri /lib/firmware/intel/sof-ipc4/arl/sof-arl.ri
```

Verify after reload/reboot:

```bash
sudo dmesg | grep -iE "sof.*(firmware|boot|version)" | tail   # expect "Firmware info: version ..., booted"
aplay -l                                                      # expect a sof-hda-dsp card
paplay /usr/share/sounds/alsa/Front_Center.wav                # audible test
```

Then `AUDIO_MODE=laptop tools/show_run.sh` works as a third playback path.

## Test coverage

`tests/test_show_audio.py` (29 tests, SDK fully faked — DDS is never initialized;
`unitree_sdk2py` import is asserted to be lazy): timeline math incl. the 4.0 s
contract against the real Thriller record, latency-comp env parsing, PCM
chunk/sample alignment, ffmpeg conversion round-trip, byte-exact PlayStream
payload + pacing, error-code handling, LED countdown/green timing on a fake
clock, mode dispatch, laptop-player spawn timing, dance-record resolution, CLI
smoke. Full suite: 365 passed, 3 skipped.
