# G1 Dance — Operator Manual (Show Mode)

For the person running a live performance. No robotics knowledge assumed.
Read once fully before your first show; afterwards the app walks you through it.

## What you are operating

A Unitree G1 humanoid robot performing a pre-trained dance. The dance was
trained and safety-checked in simulation beforehand. Your job on show day is
NOT to program anything — it is to confirm the venue is safe, run the pre-show
checklist in the app, start the performance, and stay ready to stop it.

## The golden rules

1. **You hold the e-stop for the entire performance.** Thumb on the button,
   eyes on the robot. If anything looks wrong — press it. A stopped show is a
   minor problem; a robot falling into an audience is not. Never hand the
   e-stop to anyone else mid-show.
2. **2 meters.** The dance area is a circle of 2 m radius on hard flat ground.
   Nobody and nothing inside it while the robot dances. No exceptions —
   the dance was safety-checked for exactly this space.
3. **Below 30% battery, no show.** Performance quality and balance degrade
   with a weak battery.
4. **Only "show-ready" dances in front of an audience.** The app marks each
   dance: `draft` (not ready), `sim-verified` (passed simulation once),
   `show-ready` (passed repeatedly + approved). The deploy flow works for
   lower statuses in rehearsal, but a paying audience sees show-ready only.

## Running a show, step by step

1. Open the app (double-click "G1 Dance Studio" or run
   `~/g1-dance/scripts/dance-studio`). Click **Show** in the top-right corner.
2. **Pick the dance** from the library. Check its status badge and the
   "N/3 clean" badge (consecutive clean simulation runs).
3. Watch the preview video if you want to know what the robot will do.
4. Enter **your name** (it goes in the show log) and press
   **Start pre-show check**.
5. The app walks you through five checks, one at a time — robot health,
   area clear, battery %, e-stop in hand, venue limits. Answer honestly;
   every answer is recorded with a timestamp. You cannot skip or reorder.
6. When all five are green, the **Deploy** button unlocks. Position the robot
   at the center of the dance area, step out, press Deploy and type `DEPLOY`.
7. During the performance: stay within reach of the area boundary, e-stop in
   hand, watch the robot's feet (sliding or staggering = press the button).
8. Afterwards, record the outcome — **Clean run**, **Aborted**, or
   **Incident** — plus any notes. This history is how the team keeps the
   service reliable.

## If something goes wrong

| Situation | What you do |
|---|---|
| Robot staggers, recovers | Let it finish; note it in the outcome. |
| Robot staggers repeatedly / drifts out of the area | E-stop. Record "aborted". |
| Robot falls | E-stop immediately (cuts motor torque). Keep people away. Photograph the scene, record "incident" with details. Do not restart without the team's OK. |
| Robot won't respond to the app | Do not improvise. E-stop, power off, contact the team. |
| Audience member enters the area | E-stop first, apologize later. |

## What this app can and cannot do

- The app **never** starts the robot on its own. Every performance requires
  a human completing the checklist and typing DEPLOY.
- (Current build) the Deploy button records the authorization but hardware
  delivery is handled by the deployment tooling of Phase 6 — until that
  ships, Deploy is a rehearsal of the real flow.
