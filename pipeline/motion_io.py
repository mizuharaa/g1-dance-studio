"""Shared, shape-validated loader for G1 motion CSVs.

The vet / window / retarget paths all did a bare ``np.loadtxt`` and immediately
sliced fixed columns (audit: "malformed files crash with cryptic tracebacks /
HTTP 500"). A header row, a non-numeric cell, a ragged file, or the wrong column
count produced an IndexError/ValueError deep in numpy instead of a message the
operator can act on. This module centralizes the load + validation so every
caller fails the same clear way.

CSV convention (LAFAN1 / project-wide): 36 columns =
    0:3 root xyz | 3:7 root quat (xyzw) | 7:36 the 29 joint angles
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

N_COLS = 36


def load_motion_csv(path: str | Path) -> np.ndarray:
    """Load a G1 motion CSV as an (N, 36) float array, or raise RuntimeError with
    a human-readable reason. Never lets a malformed file crash with a raw
    numpy traceback."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"motion file not found: {path}")
    try:
        m = np.loadtxt(path, delimiter=",")
    except ValueError as e:
        raise RuntimeError(
            f"'{path.name}' isn't a numeric CSV — a header row, non-numeric cell, "
            f"or ragged rows will do this ({str(e)[:120]})") from e
    if m.ndim == 1:
        # a single row loads 1-D; a single column also loads 1-D
        if m.size == N_COLS:
            m = m.reshape(1, N_COLS)
        else:
            raise RuntimeError(
                f"'{path.name}' has {m.size} value(s) in a single row — expected a "
                f"table with {N_COLS} columns per frame")
    if m.ndim != 2 or m.shape[1] != N_COLS:
        cols = m.shape[1] if m.ndim == 2 else "?"
        raise RuntimeError(
            f"'{path.name}' has {cols} columns — a G1 motion CSV needs exactly "
            f"{N_COLS} (3 root xyz + 4 root quat + 29 joints)")
    if not np.isfinite(m).all():
        n = int((~np.isfinite(m)).sum())
        raise RuntimeError(
            f"'{path.name}' contains {n} non-finite value(s) (NaN/inf) — the "
            "motion is corrupt")
    return m
