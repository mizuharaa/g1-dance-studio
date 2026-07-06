#!/usr/bin/env python
"""Train entry for the sim2real v4 'calm-legs' task (cloud/sim2real_task_v4.py).

Usage (on the box):
  NB=/workspace/notebook-data
  $NB/envs/mjlab/bin/python $NB/cloud/train_sim2real_v4.py \
      Mjlab-Tracking-Flat-Unitree-G1-S2R-V4 \
      --env.scene.num-envs 4096 \
      --env.commands.motion.motion-file $NB/motions/thriller_deploy_v2_sharp.npz \
      [any other mjlab train.py args]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mjlab.tasks  # noqa: F401  populate the stock registry first
import sim2real_task_v4  # noqa: F401  registers Mjlab-Tracking-Flat-Unitree-G1-S2R-V4

from mjlab.scripts.train import main

if __name__ == "__main__":
  main()
