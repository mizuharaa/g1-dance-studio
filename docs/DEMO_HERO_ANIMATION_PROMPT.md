# Build prompt — G1 Dance Studio hero animation (20–30 s GIF)

> Paste this prompt into a Claude Code session that has this repo checked out.
> Deliverable: a 20–30 s looping hero animation showcasing the pipeline
> end-to-end, exported as `docs/assets/pipeline_hero.gif` (≤ 8 MB, 720 px wide).

---

You are building a hero animation for **G1 Dance Studio** — a video-to-robot
pipeline: a phone video of a dancer goes in; a Unitree G1 humanoid performs the
choreography, balance-robust, out. Build it as a single self-contained HTML file
(`docs/assets/pipeline_hero.html`) with pure CSS/JS animation (no external
assets, no network), then record it to GIF.

## Scene flow (total 20–30 s, loop seamlessly)

1. **Drop the video** (0–4 s): a cursor drags a phone-video file card onto a
   drop zone labeled "G1 Dance Studio". The card thumbnail is a dancer
   silhouette. On drop: a subtle pulse, then the card unfolds into a filmstrip.
2. **Landmark extraction** (4–8 s): the filmstrip frame shows a dancing figure;
   animated green joint dots + orange bone lines snap onto it one by one
   (GVHMR pose estimation). Caption chip: "Pose estimation — GVHMR → SMPL".
3. **Retargeting** (8–12 s): the human skeleton morphs into the robot's
   29-joint stick figure (different proportions — shorter arms, wider hips);
   joint-limit arcs flash where angles clamp. Caption: "Retarget — GMR IK,
   29 DoF". Show feet snapping FLAT to a ground line (our foot-flattening).
4. **Cloud training** (12–18 s): the stick figure multiplies into a 4×4 grid of
   tiny robots in a "GPU box" frame, each jittering then progressively dancing
   in sync; a reward curve climbs; small chips tick past: "latency DR 0–80 ms",
   "keypoint termination", "4096 envs". Caption: "RL training — PPO, MuJoCo".
5. **The gate** (18–22 s): the best robot steps through a gate frame with bars
   lighting green one by one: "survival 100%", "drift 0.9 m", "legs ≥ 80%",
   "60 ms ✓". One red bar variant flashes a reject (a robot bounced back) —
   fail-closed, half a second, then green path continues.
6. **Showtime** (22–28 s): the robot walks onto a small stage, spotlight on,
   does 2–3 iconic Thriller poses (claw hands, arm swing) in sync with a
   pulsing beat bar; an operator console card shows "SHOW MODE — GO". End
   card: "video in → robot dances out" + repo name. Fade back to scene 1.

## Style

- Dark studio background (#0b101a), electric blue accents (#3b82f6), amber for
  warnings, green for passes — matches the app's console UI.
- Flat vector look, rounded cards, monospace chips for technical labels.
- Smooth easing (cubic-bezier), nothing strobes; 30 fps target.
- All motion CSS keyframes or rAF JS; the robot figures can be simple
  articulated SVG stick-bots (torso + head + 4 two-segment limbs) — charming
  beats accurate.

## Export

1. Open the HTML in a headless browser (puppeteer or playwright is fine to add
   as a dev-dependency) at 1280×720, record 20–30 s of frames at 15–30 fps.
2. Encode with ffmpeg palettegen/paletteuse to `docs/assets/pipeline_hero.gif`,
   ≤ 8 MB (drop to 12 fps / 640 px if needed).
3. Also keep the HTML committed — it is the editable source of truth.
4. Embed the GIF in README.md under the existing hero image with the caption:
   "The pipeline, end to end: drop a video → pose landmarks → retarget →
   cloud RL training → verification gate → showtime."

## Accuracy notes (do not invent)

- Stage names must match the real pipeline: GVHMR (pose), GMR (retarget),
  mjlab/MuJoCo + PPO (training), sim gate (verification), Show Mode (deploy).
- The gate bars listed in scene 5 are the project's real bars.
- The robot is a Unitree G1 (29 DoF) — humanoid, not a quadruped.
