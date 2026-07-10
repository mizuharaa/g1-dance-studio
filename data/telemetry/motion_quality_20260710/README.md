# Motion-quality measurements — 2026-07-10 (Lane B, twitch/glitch fix)

Script: `tools/motion_quality.py` (committed). Reproduce any file here with:

    python -m tools.motion_quality <csv> --json <name>_raw.json \
        --clean <name>_cleaned.csv --plot <name>_before_after.png

`*_raw.json` = full analysis of the RAW csv + a `clean` block with the
before/after numbers of `clean_motion` (accel-spike outlier rejection +
Savitzky-Golay window 7 / poly 3, tangent-space SG on the root quat).
Cleaned CSVs are derived artifacts and not committed (regenerate as above).

The Thriller deploy CSV is not on this clone; its committed vet record
(`data/dances/20260708-71711415/dance.json`) shows the same signature:
peak joint vel 56.4 rad/s vs p99 5.8 — isolated spikes.

## Summary (jerk in rad/s^3, deltas in rad)

| motion              | frames | vel peak | spikes raw→clean | jerk peak raw→clean | jerk p99 raw→clean | dof RMS delta | vel-clamp frames raw→clean |
|---------------------|-------:|---------:|-----------------:|--------------------:|-------------------:|--------------:|---------------------------:|
| dance1_subject1     |   3945 |    32.4  |        341 → 2   |       39359 → 3796  |       4726 → 1708  |        0.018  |                  626 → 407 |
| dance1_subject2     |   3945 |    32.7  |        436 → 7   |       40945 → 10562 |       8843 → 3241  |        0.034  |                1795 → 1401 |
| dance1_subject2_seg |    863 |    26.2  |        114 → 1   |       38557 → 11050 |      11238 → 4255  |        0.037  |                  487 → 369 |
| dance2_subject4     |   6771 |    27.9  |        459 → 17  |       37052 → 4827  |       5527 → 2102  |        0.023  |                 1120 → 722 |
| acro_backflip       |    167 |    40.6  |         20 → 0   |       68447 → 3161  |      12284 → 1178  |        0.039  |                    17 → 3  |

Spike = per-joint |accel| robust-z > 10 vs the joint's own MAD AND > 150 rad/s^2.

Notes:
- Spike counts drop 96–100 %; jerk peak drops 4–20×; tracking fidelity delta
  stays ≤ 0.04 rad RMS (the residual delta concentrates AT the glitch frames,
  which is the point).
- `vel_clamped_frames` does NOT reach ~0 on these LAFAN1 mocap dances because
  they genuinely exceed 0.9·3π rad/s on sustained fast moves (the vet already
  tolerates ~30 % over-limit frames as advisory). Glitch-driven single-frame
  clamps are what disappears (acro 17 → 3).
- Vet smoothness gate thresholds derived from this table: jerk peak 20000,
  spike frames 2 % — every raw file trips both, every cleaned file passes both.
