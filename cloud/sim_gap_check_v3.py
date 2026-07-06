#!/usr/bin/env python
"""sim_gap_check with the v3 task variants registered.

Thin wrapper: cloud/sim_gap_check.py (gate v3 lives there, unedited) only
imports mjlab.tasks, so custom task ids can't be passed to --task. This wrapper
registers cloud/sim2real_task_v3.py's tasks first, then delegates.

Use --task Mjlab-Tracking-Flat-Unitree-G1-S2R-V3B-GAPEVAL for V3B candidates
(stock harness + the x2.5 arm gains the deploy contract requires). V3A/V3C
candidates use the default stock task — identical to the s2r/a2 baselines.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mjlab.tasks  # noqa: F401
import sim2real_task_v3  # noqa: F401  registers V3A/V3B/V3C + V3B-GAPEVAL

import sim_gap_check

if __name__ == "__main__":
  sim_gap_check.main()
