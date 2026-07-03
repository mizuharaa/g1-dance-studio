# Audio / music sync design

A dance show is motion **and** music. This document specifies how music enters
the pipeline, how it stays locked to the robot's motion through every resampling
and padding step, and how it plays at show time. Prototype: `pipeline/audio.py`
(+ `tests/test_audio.py`). Produced artifact: a music-synced Thriller preview
(`data/previews/thriller_with_music.mp4`).

## 0. Load-bearing finding: the source video is SILENT

`ffprobe` on `data/videos/Thriller Dance Final.mov` shows a **video stream only —
no audio track**. Exported dance clips are frequently muted, so the product
**cannot assume it can extract music from the video**. Two ingest paths are
required (and both are implemented):

- **Extract** the audio track from the source video (when present).
- **Attach** a separate music file per dance (the common case here).

The Thriller demo therefore muxes a clearly-labelled **placeholder click track**
(royalty-free, generated) so motion↔music sync is audibly verifiable now; the
real licensed song is dropped in at `data/audio/thriller/music.wav` later and
the alignment is identical.

## 1. The timing problem

The dance passes through several time transforms before the robot performs it:

| Stage | Rate | Effect on timing |
|---|---|---|
| Source video | ~35 fps (VFR) | normalized to constant 30 fps first (duration preserved) |
| GVHMR → GMR retarget | 30 fps CSV | real-time duration preserved |
| `prep_motion.py` | 30 fps | **prepends** standing pad + blend-in, **appends** blend-out + hold |
| training resample | 50 fps | duration preserved (resample, not retime) |
| deploy control | 50 Hz | real-time playback |

Every stage preserves **wall-clock seconds** except `prep_motion`, which **shifts
the dance start later** by adding a standing intro. That shift is the whole
problem: if the music starts at performance t=0 it will run ahead of the dance by
the length of the intro.

### The prepped performance timeline

```
| PAD_IN 1.0s | BLEND_IN 0.5s |         DANCE = music 44.3s        | BLEND_OUT 1.0s | HOLD_OUT 2.5s |
0            1.0s           1.5s                                 45.8s            46.8s          49.3s
                             ^ music starts (dance pose 1)         ^ music ends
```

Verified against the real Thriller assets: `thriller_g1.csv` = 1329 frames = 44.3s
danced; `thriller_show.csv` = 1479 frames = 49.3s; the 150 extra frames = 45
prepended (1.5s) + 105 appended (3.5s). So:

- **`audio_delay_s = PAD_IN_S + BLEND_IN_S = 1.5s`** — music is delayed to the moment
  the robot finishes the blend-in and hits the first dance pose.
- The **blend-out + hold-out tail plays in silence** — the robot returns to standing
  after the song ends (a clean show finish).

### The offset formula (`pipeline.audio.compute_alignment`)

```
audio_delay_s   = pad_in_s + blend_in_s                    # from prep constants
trim_start_s    = window_start_s                           # if the motion was windowed
trim_duration_s = dance_duration_s                         # danced span length
music_end_s     = audio_delay_s + dance_duration_s
performance_s   = audio_delay_s + dance_duration_s + blend_out_s + hold_out_s
```

If a future dance is **windowed** (only part of the source used), `window_start_s`
trims the source audio to the same span before delaying it — so beat alignment
survives windowing too. Prep constants are read from the prep-info dict when
available (`alignment_from_prep_info`) rather than hard-coded, so a non-default
prep stays correct.

**Why seconds are safe:** because every non-prep stage preserves wall-clock
duration, an alignment computed in seconds on the prepped timeline is valid all
the way to real-robot playback. The velocity-clamp and ground-shift in prep alter
values, not frame counts, so they don't move the timeline.

## 2. Show-time playback (Show mode)

At performance time the app must start the audio in lockstep with the robot:

1. Operator confirms the pre-show checklist and hits deploy.
2. The controller begins the motion (which opens with the 1.5s standing intro).
3. The app starts audio playback at **motion t=0** but the track itself has the
   1.5s lead-in baked in (silence then music), OR the app waits `audio_delay_s`
   and starts the trimmed track — equivalent. Baking the delay into the file is
   simpler and avoids a second timer.
4. **Latency budget:** audio output latency (tens of ms) is negligible against a
   1.5s intro; a short **countdown/arm** before motion start gives the operator a
   known t=0. If the robot start is **delayed** (controller not ready), audio must
   not start — gate playback on the controller's "motion started" signal, not a
   wall clock.
5. **Abort:** any abort (kill / e-stop) must also stop audio immediately — wire
   audio stop into the same abort path.

## 3. Proposed dance-library schema additions (for `pipeline/shows.py` owner)

Add to each dance record (do **not** edit `shows.py` here — this is the proposed
contract):

```json
"audio": {
  "track": "data/audio/thriller/music.wav",   // ingested music file
  "source": "placeholder | extracted | attached",
  "delay_s": 1.5,            // audio_delay_s on the performance timeline
  "trim_start_s": 0.0,
  "trim_duration_s": 44.3,
  "muxed_preview": "data/previews/thriller_with_music.mp4"
}
```

Show-readiness (future): a dance intended for a paid show should require an audio
track present and a confirmed muxed-preview review, alongside the existing sim-exam
gate.

## 4. Beat-awareness (future, optional)

Out of scope for v1 but noted: detect beats (e.g. `librosa.beat`) in the music and
in the motion (velocity peaks), then score how well the choreography lands on the
beat. Useful as a **quality metric** for retargeted-from-video dances and to
auto-suggest a small global time-offset if the dancer in the source video was
slightly off the track. Not needed to ship — the source motion already carries the
dancer's own timing.

## 5. What the prototype does now

- `has_audio` / `extract_audio` — ingest from video when present.
- `make_placeholder_track` — labelled click track (video is silent).
- `compute_alignment` / `alignment_from_prep_info` — the offset math (unit-tested).
- `mux_audio_onto_video` — trim + delay + pad, lay onto a preview.
- `build_thriller_demo` / CLI `python -m pipeline.audio thriller-demo` — end to end.

**Produced:** `data/previews/thriller_with_music.mp4` (49.3s, click track entering
at 1.5s exactly when the dance starts). Replace `data/audio/thriller/music.wav`
with the real song and re-run — alignment unchanged.

## 6. Merge / conflict notes

- New files only: `pipeline/audio.py`, `tests/test_audio.py`, this doc. No edits to
  `shows.py`, `sim_exam.py`, `PROJECT_STATE.md`, or `ui/server.py` (the schema
  additions above are proposed for the shows.py owner; a Show-mode "play with music"
  button is a small follow-up once the schema lands). No merge conflicts expected.
- Rendered artifacts (`thriller_with_music.mp4`, `thriller_show_prepped.mp4`) live
  under the gitignored `data/` in the main checkout, not committed.
