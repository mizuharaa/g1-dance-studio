# Hand expressiveness — design, safety gate, and honest verdict

R&D spike. Framing (per owner concerns): **the Inspire hands are expensive and
must never be driven by an unvetted motion, and they may look oversized.** So this
is not "add hands" — it is "*can* we use the hands safely, and do they even look
good, proven in sim with zero hardware risk."

## Bottom line up front

- **v1 recommendation: use the hands SPARINGLY — a few authored expressive
  moments (the Thriller claw), not continuous finger motion — and only after the
  motion's hand track passes the collision gate on a *real Inspire model*.** The
  size concern makes constant finger animation a liability; a held signature
  gesture turns the bulk into an intentional statement instead of a distraction.
- Two hard blockers below must be cleared before any hand ever moves for real.

## Finding 1 — hand pose is NOT recoverable from our current pipeline (verified)

GVHMR predicts a **body-only** SMPL: the Thriller prediction
(`data/motions/thriller/hmr4d_results.pt`) contains only `body_pose (1329, 63)`,
`betas`, `global_orient`, `transl` — **no finger data anywhere**. The GMR loader
(`third_party/GMR/.../utils/smpl.py`) then hardcodes
`left_hand_pose = right_hand_pose = zeros(N, 45)`. So although we retarget through
the SMPL-X body model, its hand DoF are fed zeros.

⇒ There is **nothing to "recover."** Expressive hand motion must be **authored**
(pose library + keyframes) until the front-end is upgraded to a **whole-body +
hands** estimator (SMPLer-X / OSX, or a HaMeR hand-pose add-on) that actually
populates the finger joints. `hands.probe_gvhmr_hands()` proves absence on any pred.

## Finding 2 — MODEL MISMATCH: our sim hand ≠ the robot's hand (verified)

The only G1+hands MuJoCo model available (menagerie `g1_with_hands.xml`) is the
**Unitree Dex3** 3-finger hand (7 DoF/hand, joints thumb×3 / index×2 / middle×2).
The **real robot has Inspire RH56DFTP** 5-finger hands (6 DoF/hand, `angle_set[6]`
0–1000 over DDS `rt/inspire_hand/ctrl/{l,r}`, confirmed in `~/robot` runbooks +
`inspire_hand_sdk`).

⇒ The collision gate machinery is correct and model-agnostic, but on the Dex3
substrate it **cannot certify the real Inspire hands.** Sourcing an Inspire
RH56DFTP MJCF (with collision geometry) is the top prerequisite — then
`HandModel(mjcf=<inspire>, is_real_inspire=True)` and the gate certifies for real.

## The safety deliverable — hand-collision vet gate (built + tested)

`pipeline/hands.check_hand_collisions(track, HandModel)` replays a hand-pose track
frame-by-frame and rejects it on any of:
- **self-collision** — geoms on *different fingers* closer than a margin (pure
  `mj_geomDistance`, so it ignores the model's contype/conaffinity filtering),
- **hand↔body collision** — hand geoms vs torso/legs/other-arm (the same-arm
  wrist/forearm attachment chain is excluded so neutral poses don't false-alarm),
- **joint-limit violation.**
Verified on the Dex3 substrate: an open/neutral track **passes** (0 collisions);
fingers driven to their limits **fail** with 24 self-collisions at frame 0;
over-range targets flag limit violations. A motion's hand track must PASS this
before it could ever reach hardware — this is the hardware-protecting core.

## Aesthetics — honest read on "comically large"

I **cannot faithfully render the Inspire proportions** (no Inspire MJCF). The Dex3
render (`design/hands_shots/g1_dex3_claw.png`) looks reasonably proportioned — but
that is the *3-finger Dex3*, not the user's hand, so it does not settle the
question. The authoritative data point is the owner's real-world observation that
the **Inspire RH56DFTP looks oversized**, which is consistent with it being a
full adult-hand-sized 5-finger hand on the G1's relatively slim forearms. Taken at
face value: **don't animate fingers constantly.** Reserve hands for a few held,
deliberate gestures where the size reads as intent.

## How hands integrate (design)

- **Retarget:** emit a hand-pose track (6-DoF/hand Inspire, or model DoF) alongside
  the 29-DoF body CSV, time-aligned to the body timeline. `build_hand_track()`
  interpolates named keyframes and applies `lead_in_s` to match prep_motion's
  standing pad/blend so hands hit their marks in sync with the body dance.
- **Training:** hands are low-mass and non-balance-critical → keep them **out of
  the RL balance policy** and play them **open-loop, time-synced** to the body
  controller. (Training fingers in RL adds cost for no balance benefit.)
- **Deploy:** stream the vetted hand track to `rt/inspire_hand/ctrl/{l,r}` from a
  small player clocked off the body controller's "motion started" signal — same
  sync discipline as the music track. NEVER without a passing collision-gate verdict.

## Path to the Thriller claw in a real show

1. Source an Inspire RH56DFTP MJCF with collision geometry; point `HandModel` at it.
2. Author the claw keyframes (`INSPIRE_POSES["claw"]`) held over the iconic section,
   open hands elsewhere; build the time-synced track.
3. Run the collision gate → must PASS. 4. Gantry-test the hand playback with the
   body policy. 5. Only then include it in the show.

## Follow-ups for owners (not done here)
- Source/author the Inspire MJCF (blocks real certification + honest render).
- Whole-body+hands estimator as a front-end option (blocks video→hands capture).
- Wire a hand-collision check into the show-ready gate once an Inspire model exists.
