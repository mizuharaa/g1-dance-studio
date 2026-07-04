#!/usr/bin/env python
"""Train entry for the sim2real retrain task.

Registers Mjlab-Tracking-Flat-Unitree-G1-Sim2Real (cloud/sim2real_task.py)
then delegates to mjlab's stock train CLI — train.py builds its task list
from the registry at call time, so the custom task appears like any other.

Usage (on the box):
  NB=/workspace/notebook-data
  $NB/envs/mjlab/bin/python $NB/cloud/train_sim2real.py \
      Mjlab-Tracking-Flat-Unitree-G1-Sim2Real \
      --env.scene.num-envs 4096 \
      --env.commands.motion.motion-file $NB/motions/thriller_deploy.npz \
      [any other mjlab train.py args]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mjlab.tasks  # noqa: F401  populate the stock registry first
import sim2real_task  # noqa: F401  registers Mjlab-Tracking-Flat-Unitree-G1-Sim2Real

from mjlab.scripts.train import main

if __name__ == "__main__":
  main()
