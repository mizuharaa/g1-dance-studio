#!/usr/bin/env python
"""Train entry for the DYNAMIC-SKILLS (acro) task.

Registers Mjlab-Tracking-Flat-Unitree-G1-Acro (cloud/dynamic_skills_task.py)
then delegates to mjlab's stock train CLI (same pattern as train_sim2real.py).

Usage (on the box):
  NB=/workspace/notebook-data
  $NB/envs/mjlab/bin/python $NB/cloud/train_dynamic_skills.py \
      Mjlab-Tracking-Flat-Unitree-G1-Acro \
      --env.scene.num-envs 4096 \
      --env.commands.motion.motion-file $NB/motions/<acro>.npz \
      --agent.max-iterations 10000 --agent.run-name train-acro-1 --video False
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mjlab.tasks  # noqa: F401  populate the stock registry first
import dynamic_skills_task  # noqa: F401  registers Mjlab-Tracking-Flat-Unitree-G1-Acro

from mjlab.scripts.train import main

if __name__ == "__main__":
  main()
