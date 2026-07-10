# Motion feasibility — the "60-70%" is a POLICY gap, not a feasibility one (2026-07-10)

Tool: tools/motion_feasibility.py. Reports per-joint velocity vs the 9.4 rad/s motor limit.

## Evidence (3 independent signals — B2 retime will NOT fix the Thriller)
1. **The Thriller DEPLOY motion is already feasible**: peak joint vel ~8.5 rad/s < 9.4 limit,
   0% of frames over. (Raw retargets DO exceed it — thriller_g1 / dance1 below — so the tool
   still matters for FUTURE raw dances + the vet gate; the Lane-B clamp already handled Thriller.)
2. **Real hardware confirms a TRACKING gap**, not infeasibility: run 20260710-145111 shows mean
   |target-q| = 10-16 deg/joint (max 80-102) — the robot CAN reach the poses, the policy just
   doesn't command them accurately (waist, shoulders, hips, ankles worst).
3. **Sandbox A/B: slowing the motion does NOT help.** tools/sim_sandbox achieved fraction stays
   79.7% at 1.0x / 1.25x / 1.5x time-stretch (rms err flat ~0.455 rad). The policy under-reaches
   the same amount regardless of reference speed.

## Conclusion
The motion rides near (but under) the motor limit with little torque headroom, but retiming it
does not raise fidelity — the bottleneck is the POLICY (systematic under-reach on subtle joints).
=> **Lane E (arm-scoped-reward retrain) is the fix; it should NOT wait for a B2 feasibility motion.**
Lane E should train on the Lane-B DE-GLITCHED Thriller motion now. B2's feasibility tool stays as
a vet-gate/authoring aid for future raw extractions (which are genuinely infeasible pre-clamp).
