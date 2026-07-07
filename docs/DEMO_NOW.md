# INVESTOR DEMO — crib card (2026-07-07)

## The one command
```
cd ~/g1-dance && bash demo.sh
```
Reads out a pre-flight, waits for you to confirm the tether/remote, then on ENTER:
robot moves to ready pose (~4 s) → dances Thriller to the real track, music
auto-synced (~50 s) → smooth ramp to damping.

## Before you start
- Robot standing on the tether, tether taut enough to catch a fall.
- **Damping remote in your hand.** It is the only stop.
- Laptop on robot-lan; aux speaker plugged into the laptop, volume up.
- 2 m clear — the robot may catch-step ~1 m rightward at the very end (onboard
  catches it; expected, harmless).

## If something looks wrong
- **Damp with the remote.** Every fault also auto-damps in software.
- To kill from the laptop: `pkill -f "python.*deploy_runtime"` (signal the
  python pid, not the wrapper).

## What you're showing
- v3e policy: the crispest arms we've trained (−38% tracking error vs the prior
  hardware-proven policy), tracking the true dance dynamics with the accents.
- Full RL whole-body controller — balanced, push-robust, not open-loop playback.
- Proven today: full dance + real music ran clean end to end (2589/2589 ticks).

## NOT in this demo (deliberately — untested on hardware = not for a live stage)
- The "stand at the end instead of damp" polish and the one-button app were mid-
  build when the demo came up; that code is unfinished and unproven on the robot.
  Do not run it live. It resumes after the demo, tested properly first.
```
```
