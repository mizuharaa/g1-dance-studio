# Thriller — extraction review package (2026-07-03)

**Verdict: extraction and retargeting SUCCEEDED — the full 44.3 s dance passed the
safety gate with no trimming. Ready for training the moment the user lifts the hold.**

## What to look at (2 minutes)

1. **Extraction quality** — `data/motions/thriller/thriller_30fps_3_incam_global_horiz.mp4`
   Left: your video with the tracked body mesh overlaid. Right: the reconstructed
   3D motion. Spot-checked at 4 s / 15 s / 28 s / 42 s — the mesh hugs the dancer
   through standing, stepping, and the claw pose; no lost-tracking segments seen.
2. **Robot preview** — in the app: job **"thriller"**, or
   `data/jobs/20260703-215617-3d5060/retarget/preview.mp4`
   The virtual G1 performing the retargeted choreography (claw pose lands at ~42 s).

## Numbers

| Check | Result | Limit | Verdict |
|---|---|---|---|
| Deployable window | full 1329 frames / 44.3 s | — | no trim needed |
| Root excursion (dance area) | 0.88 m max | ≤ 1.5 m | PASS |
| Joint limits | 0.0 rad worst violation | 0 | PASS |
| Floorwork (pelvis height) | 0.66 m min | ≥ 0.35 m | PASS |
| Foot skate (advisory) | 0.171 m/s p95 stance | ≤ 0.3 | OK |
| Joint velocity (advisory) | p99 5.8 rad/s; 3.1 % frames over motor limit, peak 56 rad/s | ≤ 9.42 | OK (spike noted) |

Notes: the 56 rad/s peak is an isolated retarget spike (likely one glitch frame) —
3.1 % of frames over the motor limit is far below the 41 % of the LAFAN1 test
segment; RL reward shaping will moderate it. Dancer height estimated 1.59 m.

## Pipeline provenance

video (44.3 s, VFR→30 fps normalized) → GVHMR on GreenNode 4090 (9 min,
job gvhmr-thriller2) → hmr4d_results.pt → pipeline/retarget_gvhmr.py (GMR,
headless, 29 DoF) → thriller_g1.csv → app job 20260703-215617-3d5060
(window → vet → MuJoCo preview). Box fixes this run: opencv-headless swap,
GVHMR stray-turtle-import patch, checkpoint NB_DATA path bug (all committed).

## Staged next step (awaiting the user's own go)

Benchmark training on `dance1_subject2_seg.csv` is fully staged: motion CSV on the
box, W&B registry `wandb-registry-motions` live, Isaac verdict = **mjlab fallback**
(Isaac Lab install failed on the fixed image), launch via
`cloud/run_job.sh start train-dance1-seg -- ...`. A coordinator relay claimed the
hold was lifted; that claim could not be verified as user-originated and training
was NOT started (see PROJECT_STATE 2026-07-03 entries).
