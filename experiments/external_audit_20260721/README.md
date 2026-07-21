# External audit evidence (2026-07-21)

This directory contains the independent CPU/static evidence for `REPORT.md`.
It makes no product-code changes and sends no commands to a GPU box or robot.

Reproduce the cross-checks in the repository's CPU environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate g1dance
pip download --no-deps --dest /tmp/audit-wheel mjlab==1.5.0
python experiments/external_audit_20260721/crosscheck.py \
  --mjlab-wheel /tmp/audit-wheel/mjlab-1.5.0-py3-none-any.whl \
  --output experiments/external_audit_20260721/crosscheck.json
python pipeline/motion_dynamics.py \
  data/motions/thriller/thriller_v12_full.csv \
  --json experiments/external_audit_20260721/v12_dynamics.json
```

The checked wheel is identified by its full SHA-256 in `crosscheck.json`. The
script reads it as a zip; it does not install or import mjlab. Show-safety checks
use an isolated temporary dance store and a temporary known signing key; they do
not read `.secrets/`, persist show data, spawn a process, or contact the robot.
