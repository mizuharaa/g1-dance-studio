#!/usr/bin/env python
"""Train entry for the sim2real v3 (dance-quality) task variants.

Registers Mjlab-Tracking-Flat-Unitree-G1-S2R-V3{A,B,C} (cloud/sim2real_task_v3.py)
then delegates to mjlab's stock train CLI.

Usage (on the box):
  NB=/workspace/notebook-data
  $NB/envs/mjlab/bin/python $NB/cloud/train_sim2real_v3.py \
      Mjlab-Tracking-Flat-Unitree-G1-S2R-V3A \
      --env.scene.num-envs 4096 \
      --env.commands.motion.motion-file $NB/motions/thriller_deploy.npz \
      [any other mjlab train.py args]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mjlab.tasks  # noqa: F401  populate the stock registry first
import sim2real_task_v3  # noqa: F401  registers the V3A/V3B/V3C(+GAPEVAL) tasks

from mjlab.scripts.train import main

if __name__ == "__main__":
  main()
