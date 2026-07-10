#!/usr/bin/env python
"""Train entry for the v5 "fidelity" recipe (Lane E).

Registers Mjlab-Tracking-Flat-Unitree-G1-S2R-V5 (cloud/sim2real_task_v5.py)
then delegates to mjlab's stock train CLI. Delay curriculum is driven by the
G1_CMD_DELAY_MAX_LAG / G1_OBS_DELAY_MAX_LAG env vars — see
cloud/train_v5_curriculum.sh for the staged launch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mjlab.tasks  # noqa: F401  populate the stock registry first
import sim2real_task_v5  # noqa: F401  registers Mjlab-Tracking-Flat-Unitree-G1-S2R-V5

from mjlab.scripts.train import main

if __name__ == "__main__":
  main()
