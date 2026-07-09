# CSV-ankle side-by-side verification (2026-07-09)

Sim panel re-rendered from the DEPLOYED motion (thriller_csv_ankle_penalty_deploy.npz)
via tools/render_deploy_sim.py — replaces the old v3e composite whose sim panel was a
different Thriller take (2589-frame lineage) that matched neither the robot nor the
reference (the live-run "sim not in sync" complaint).

Composite: tools/make_side_by_side.py --sim data/previews/rollout_ankle_csv.mp4 \
  --source "data/videos/Thriller Dance Final.mov" \
  --audio data/dances/20260708-71711415/audio/music.wav \
  --src-lead 3.76 --music-at 4.0 --speed 0.9 --height 480
Output: data/previews/thriller_side_by_side_csv.mp4

cmp_*s.png = composite frames at t = 5/12/22/34/44 s (left=reference, right=sim).
Verified: pelvis upright_z ~1.00 all frames (no fall/glitch); mid-dance (12-34s) aligns
well; start/end phase offset is inherent human-tutorial-vs-robot tempo drift (same
approximate-alignment caveat the v3e composite carried).
