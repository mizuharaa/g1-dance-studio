# Retarget fidelity — where Thriller's sharpness dies before the policy sees it

**Question (2026-07-06):** the robot's Thriller "looks like Thriller but is not as good
as the video". How much of that gap is created by the FRONT-END
(video → GVHMR → GMR → velocity-limit → 30 fps CSV → prep/blends → 50 fps npz),
before the policy ever gets the reference?

**Answer: the sharpness is deleted almost entirely in ONE place — the blanket
velocity clamp (prep_motion, 0.9·3π ≈ 8.48 rad/s on every joint). It blunted 58 real
dance accents by 40–85 % and removed 97 % of the high-frequency velocity energy in
the sharpest section. GVHMR captured those accents fine; the 30→50 fps npz step is
essentially lossless. A prototype reference with per-joint clamps at the true motor
limits restores 96 % of the accent velocity with identical frame count / music timing:
`data/motions/edits/thriller_deploy_v2_sharp.csv` (vet PASS).**

Analysis tool: `tools/retarget_fidelity_analysis.py` → `data/reports/retarget_fidelity.json`.
Prototype tool: `tools/make_sharp_reference.py` → `data/reports/thriller_v2_sharp.json`.

---

## 1. The chain, as actually run (verified from artifacts)

| Stage | Artifact | What it does to sharpness |
|---|---|---|
| Original video | `Thriller Dance Final.mov` — **VFR ~35.4 fps** (1572 frames on a 120 Hz clock, 44.28 s) | reference truth |
| 30 fps normalize | `thriller_30fps.mp4` (1329 frames) | **drops 243 frames (15.5 %)** before GVHMR — timing judder up to 33 ms, single-frame velocity overshoot at hits |
| GVHMR (4090) | `hmr4d_results.pt` + side-by-side mp4 | small loss — see §3; accents are captured |
| GMR retarget | `thriller_g1.csv` (1329×36 @30 fps) | the raw retarget; accent peaks 10–56 rad/s |
| GMR `--velocity-limit` | `thriller_vlim.csv` | **NO-OP: byte-identical to `thriller_g1.csv`.** The recorded "re-retargeted with use_velocity_limit=True → 0 % over-velocity" was actually achieved by the prep clamp below, not GMR. (GMR's limiter, when it works, is also a blanket 3π on all motors — `third_party/GMR/.../motion_retarget.py:100`.) |
| prep_motion | `thriller_show.csv` (1479 = 45 pad/blend + 1329 + 105 out/hold) | **THE KILLER**: sequential per-frame delta clamp at 0.9·3π ≈ 8.48 rad/s on ALL 29 joints + 3-frame moving average around touched frames. Modified 156/1329 frames, worst-joint pose deviation up to **89.9°** vs the retarget |
| deploy ramp | `thriller_deploy.csv` (1554 = 75-frame activation ramp + show) | timing only, no sharpness effect |
| mjlab csv_to_npz | `thriller_deploy.npz` (2589 @50 fps) | ~lossless: linear interp preserves segment slopes exactly (max vel 8.48 preserved); stored `joint_vel` (central differences) smooths p99 by ~2.5 % |

## 2. Quantified loss table (30 fps joint velocities, rad/s)

Global (dance segment only, 1329 frames):

| Metric | raw retarget | deploy ref (old) | deploy npz | v2 sharp (prototype) |
|---|---|---|---|---|
| max \|v\| any joint | **56.4** | 8.48 | 8.48 | 33.3 |
| max \|v\| arms | 56.4 | 8.48 | 8.48 | 33.3 |
| max \|v\| legs | 13.7 | 8.37 | 8.37 | 13.7 |
| p99 \|v\| | 5.81 | 5.62 | 5.50 | 5.82 |
| max \|acc\| (rad/s²) | 1735 | 264 | 416* | — |
| frames with any joint > 9.42 | 3.1 % | 0 % | 0 % | 3.2 % |

\* 50 fps central-difference corner artifact, not real content.

Per section (times in SHOW timeline = video + 1.5 s; HF = 5–15 Hz velocity-spectrum energy kept vs raw):

| Section | what it is | raw peak | old deploy peak | HF kept (old) | accents clamped | v2 peak | HF kept (v2) |
|---|---|---|---|---|---|---|---|
| **13–17 s** | side-step + fast arm swings/punches | 56.4 | 8.5 | **2.8 %** | **35** | 33.3 | 71.6 % |
| **25–36 s** | X-pose arms, kicks | 21.5 | 8.3 | 45.1 % | 11 | 21.5 | 100 % |
| **40–49 s** | claw finale | 9.8 | 8.2 | 88.7 % | 1 | 9.8 | 100 % |

Event-level: 60 over-blanket events in the raw retarget; **58 are real MOVES**
(sustained displacement — dance accents), only **2 are glitches** (single-frame
hip-roll spikes at show 39.2 s). The review's "isolated retarget spike" theory for the
56 rad/s peak was wrong — it is the 15.2 s arm-swing accent. The old deploy kept a mean
**41 %** of the raw accent peak velocity (attenuation 16–86 %, worst on exactly the
biggest hits: every raw peak ≥ 25 rad/s was cut ~70–85 %).

Pose-domain equivalent (how far the reference itself is from the retarget, worst joint per frame):

| | whole dance | the 156 modified frames | 13–17 s section |
|---|---|---|---|
| old deploy vs raw | 0.22° mean | **16.6° mean, 63° p95, 89.9° max** | 13.9° mean/frame |
| v2 sharp vs raw | 0.006° mean | 0.9° mean, 44.1° max | 1.1° mean/frame |

For scale: the a1 policy's own tracking error is ~6.7° mean; a2's is looser
(mpkpe 0.221 m vs a1 0.168 m). **At the accent beats the reference is 2–10× more wrong
than the policy's typical tracking error — and a perfect policy cannot recover it.**

## 3. GVHMR (SMPL estimate) — qualitative, from the side-by-side video

Sampled at the measured accent timestamps (video time; frames in
side-by-side `thriller_30fps_3_incam_global_horiz.mp4`):

- **11.6–14.8 s** (the butchered section): mesh hugs the dancer through the arm
  swings; extremes captured (arms fully extended at 13.70 s, fully down 4 frames later
  at 13.83 s — a real ≥ 12 rad/s average, >20 rad/s peak swing). Hands show motion
  blur in the video; per-frame estimates around such frames carry overshoot noise —
  the supra-physical 40–56 rad/s single-frame peaks (a human shoulder peaks ~20–25
  rad/s) are estimator snap + the 15.5 % dropped-frame judder, ON TOP of a genuinely
  fast move.
- **21.7 s** X-pose, **25.3 s** arms-up V, **33.5 s** kick (blurred foot): all matched
  well by the mesh.
- **37.7 s** head-snap/hair-whip + arm punch: body and stance matched; the head is
  visibly under-rotated vs the dancer's whip — a real GVHMR loss, but moot for the G1:
  **the robot has no neck DOF**, so head snaps are lost to morphology regardless.
- Hands: body-only SMPL — the claw FINGERS were never extracted (known; hands must be
  authored per the hands-spike verdict).

Verdict: GVHMR is NOT where the sharpness died. Its residual contributions: slight
peak-noise at motion-blur frames, under-rotated head snap, no fingers.

## 4. The 30 fps intermediate and the 50 fps npz

- The npz's `joint_pos` is linear interpolation of the 30 fps CSV: piecewise-linear
  resampling **preserves every segment slope**, so peak frame-to-frame velocity is
  bit-identical (8.48 rad/s in, 8.48 out). No filtering found in mjlab's converter
  (`csv_to_npz.py`: lerp/slerp + `torch.gradient`, verified on the box).
- The stored `joint_vel` channel (part of the tracking command) uses central
  differences → mild smoothing: p99 5.62→5.41 (−2.5 %), peaks −0.02 %. Negligible
  next to the clamp.
- Retargeting at higher output fps buys nothing: GVHMR output is video-locked at
  30 fps. The real (small) win is upstream: **stop dropping 15.5 % of the source
  frames** — normalize VFR to its ~35.4 fps average (or feed GVHMR native VFR-decoded
  frames) and let the retarget resample 35.4→30 by interpolation instead.

## 5. Prototype: `data/motions/edits/thriller_deploy_v2_sharp.csv`

Built by `tools/make_sharp_reference.py` from the RAW retarget + the existing deploy
CSV (ramp rows copied verbatim; show assembly reuses prep_motion's own pad/blend/
grounding code → **identical 1554-frame count, identical beat alignment, seam
error < 1e-6**). Changes:

1. **Despike** (out-and-back single-frame spikes, both legs > 8 rad/s, ≥ 50 %
   reversal): fixed exactly **1 cell** — confirming the raw retarget is clean.
2. **Per-joint velocity clamp at 0.9 × true motor limits** (wbt/BeyondMimic G1
   actuator config = Unitree motor classes): hips pitch/yaw 28.8, hip-roll/knee 18.0,
   ankles/waist-r/p/shoulders/elbows/wrist-roll 33.3, waist-yaw 28.8,
   wrist-pitch/yaw 19.8 rad/s. Touched **5 frames** (vs 156) — only the
   supra-physical > 33 rad/s estimator overshoots.
3. **No moving-average pass.**

Validation: `pipeline/vet_motion.py` **PASS** (same hard-gate results as the old
deploy); FK dance-segment min contact height = 0.000 m (grounded); the −1.98 cm dip
in the blend-out/hold segment is **inherited unchanged from the old deploy** (prep
grounds only the dance before appending blends — pre-existing, out of scope).

Numbers (dance segment):

| | old deploy | v2 sharp |
|---|---|---|
| mean accent peak kept (58 real MOVES) | 41 % | **96 %** |
| accents fully restored (≥ 99 % of raw peak) | 0/58 | **46/58** |
| HF energy kept, 13–17 s | 2.8 % | 71.6 % |
| HF energy kept, 25–36 s | 45 % | 100 % |
| HF energy kept, 40–49 s | 89 % | 100 % |
| max pose deviation from retarget | 89.9° | 44.1° (only at the 5 capped supra-physical spikes) |

**This is a reference prototype for the NEXT training run — not a drop-in for the
current policy** (a2 is sha-bound to the old deploy CSV and was trained to the blunted
reference). Nothing existing was replaced.

## 6. Verdict — front-end vs policy share of the visual gap

- **At the accent beats** (the moments that read as "not as good as the video":
  the 13–17 s arm swings, the 23–29 s hits, the 39 s punch): the front-end deleted
  **59 % of the peak velocity on average (up to 85 %)** before training. This loss is
  BINDING — even a perfect policy dances the blunted version. Front-end owns most of
  the sharpness gap here.
- **Between accents (~88 % of frames):** the front-end is transparent (mean 0.22°
  deviation); the a2 policy's diffuse tracking error (~mpkpe 0.221 m; a1 0.168 m,
  ~6.7° mean joint error) plus its action-rate smoothing owns the gap. (Policy side
  is another workstream.)
- **Unfixable by either:** no neck DOF (head snaps), no extracted fingers (claw
  hands), both signature Thriller elements — set expectations or author hand/head
  proxies (waist accent, authored Inspire claw once collision-gated).

## 7. Recommendation for the standard many-dances retarget stage

1. **Retarget WITHOUT `--velocity-limit`** (it was a no-op this run anyway, and GMR's
   limiter is the same wrong blanket 3π). Keep the raw CSV as the fidelity source.
2. **Replace prep_motion's blanket clamp** with despike + per-joint clamp at
   0.9 × true motor velocity limits (the `make_sharp_reference.py` algorithm). Fold it
   into `prep_motion.py` as the default path for new dances.
3. **Vet gate:** make the velocity advisory per-joint against the true motor limits
   table, not blanket 3π — "3.1 % frames over" hid 58 real accents this time.
4. **VFR normalization:** normalize source video to its native average fps
   (here 35.4), not hard-30; resample to 30 fps AFTER retarget if the CSV convention
   needs it.
5. **Keep 30→50 fps mjlab conversion as-is** (verified ~lossless).
6. Policy side (handoff): retrain on the sharp reference; revisit the a2
   action-rate penalty (a1's tighter tracking looked crisper) — the reference is no
   longer the binding constraint after (1)–(2).
