#!/usr/bin/env python3
"""Tempo-resample a 30 fps G1 motion CSV for the v10 SPEED CURRICULUM.

WHY THIS EXISTS (v10 mechanism decision, 2026-07-20)
----------------------------------------------------
mjlab 1.5.0's MotionCommand plays the motion npz back at exactly one frame per
50 Hz control step (`self.time_steps += 1`, integer index — see
third_party/mjlab_mdp_ref/mdp/commands.py). There is NO phase-rate / playback-
speed hook in MotionCommandCfg, so the only non-fork way to train the same dance
at a different tempo is to hand the task a TIME-RESAMPLED copy of the motion.
This script produces those copies from the NATIVE csv: cubic interpolation of
root position + joint angles, proper quaternion slerp for root orientation,
NO frame duplication. Velocities are NOT stored in the CSV; the box-side
csv_to_npz recomputes joint/body velocities by differentiating the retimed
positions at 50 fps (and body kinematics via FK), so they scale correctly and
consistently (a 0.75x-tempo copy has ~0.75x the joint velocities).

SPEED SEMANTICS: --speed is a TEMPO multiplier.
  speed 1.00 -> byte-identical copy of the input (native).
  speed 0.60 -> the dance plays at 60% tempo; duration = native / 0.60.
Output stays at the input fps (default 30), same column layout:
  cols 0:3 root pos xyz, 3:7 root quat XYZW, 7:36 the 29 LAFAN1-order joints.

Usage:
  python cloud/retime_motion.py --input native.csv --speed 0.75 \
      --output native_0p75.csv [--fps 30] [--check-json checks.json]

The run prints (and optionally writes) a self-check: frame counts, quaternion
norms, and the max-joint-velocity ratio out/in (must be ~= speed).
Pure numpy + optional scipy (falls back to a Catmull-Rom cubic if scipy is
missing on the box), so it runs on the no-GPU laptop AND inside the mjlab env.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

try:  # preferred: C2-continuous natural cubic spline
  from scipy.interpolate import CubicSpline  # type: ignore

  _HAVE_SCIPY = True
except Exception:  # pragma: no cover - box fallback
  _HAVE_SCIPY = False


def _catmull_rom(x_src: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
  """Numpy-only cubic (Catmull-Rom / Hermite) interpolation, axis 0.

  Fallback when scipy is absent. C1-continuous cubic — still NOT linear and NOT
  frame duplication; tangents are central differences.
  """
  n = y.shape[0]
  dt = x_src[1] - x_src[0]
  # central-difference tangents (one-sided at the ends)
  m = np.empty_like(y)
  m[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
  m[0] = (y[1] - y[0]) / dt
  m[-1] = (y[-1] - y[-2]) / dt

  s = (x_new - x_src[0]) / dt
  i0 = np.clip(np.floor(s).astype(int), 0, n - 2)
  f = (s - i0)[:, None]

  p0, p1 = y[i0], y[i0 + 1]
  m0, m1 = m[i0] * dt, m[i0 + 1] * dt
  f2, f3 = f * f, f * f * f
  h00 = 2 * f3 - 3 * f2 + 1
  h10 = f3 - 2 * f2 + f
  h01 = -2 * f3 + 3 * f2
  h11 = f3 - f2
  return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def _interp_cubic(t_src: np.ndarray, y: np.ndarray, t_new: np.ndarray) -> np.ndarray:
  if _HAVE_SCIPY:
    return CubicSpline(t_src, y, axis=0)(t_new)
  return _catmull_rom(t_src, y, t_new)


def _slerp_track(quats_xyzw: np.ndarray, fps: float, t_new: np.ndarray) -> np.ndarray:
  """Piecewise slerp of a quaternion track (xyzw) sampled at fps, at times t_new."""
  q = quats_xyzw.copy().astype(np.float64)
  q /= np.linalg.norm(q, axis=1, keepdims=True)
  # hemisphere continuity: flip whenever consecutive dot < 0
  dots = np.sum(q[1:] * q[:-1], axis=1)
  flips = np.cumprod(np.where(dots < 0, -1.0, 1.0))
  q[1:] *= flips[:, None]

  n = q.shape[0]
  s = t_new * fps
  i0 = np.clip(np.floor(s).astype(int), 0, n - 2)
  f = (s - i0)[:, None]

  q0, q1 = q[i0], q[i0 + 1]
  dot = np.clip(np.sum(q0 * q1, axis=1, keepdims=True), -1.0, 1.0)
  theta = np.arccos(dot)
  small = theta[:, 0] < 1e-6
  sin_t = np.sin(np.where(theta > 1e-6, theta, 1.0))  # guarded
  w0 = np.where(small[:, None], 1.0 - f, np.sin((1.0 - f) * theta) / sin_t)
  w1 = np.where(small[:, None], f, np.sin(f * theta) / sin_t)
  out = w0 * q0 + w1 * q1
  return out / np.linalg.norm(out, axis=1, keepdims=True)


def _max_joint_vel(joints: np.ndarray, fps: float) -> float:
  if joints.shape[0] < 2:
    return 0.0
  return float(np.max(np.abs(np.diff(joints, axis=0)) * fps))


def retime(input_csv: Path, output_csv: Path, speed: float, fps: float) -> dict:
  data = np.loadtxt(input_csv, delimiter=",")
  if data.ndim != 2 or data.shape[1] != 36:
    raise SystemExit(f"expected 36 columns (3 pos + 4 quat + 29 joints), got {data.shape}")
  n_in = data.shape[0]
  dur_in = (n_in - 1) / fps

  if abs(speed - 1.0) < 1e-9:
    shutil.copyfile(input_csv, output_csv)
    out = np.loadtxt(output_csv, delimiter=",")
  else:
    t_src = np.arange(n_in) / fps
    n_out = int(round((n_in - 1) / speed)) + 1
    t_new = np.clip(np.arange(n_out) / fps * speed, 0.0, dur_in)

    pos = _interp_cubic(t_src, data[:, 0:3], t_new)
    quat = _slerp_track(data[:, 3:7], fps, t_new)
    joints = _interp_cubic(t_src, data[:, 7:36], t_new)
    out = np.hstack([pos, quat, joints])
    np.savetxt(output_csv, out, delimiter=",", fmt="%.6f")

  # ---- self-checks -------------------------------------------------------
  qn = np.linalg.norm(out[:, 3:7], axis=1)
  vin = _max_joint_vel(data[:, 7:36], fps)
  vout = _max_joint_vel(out[:, 7:36], fps)
  check = {
    "input": str(input_csv),
    "output": str(output_csv),
    "speed": speed,
    "interp": "scipy.CubicSpline" if _HAVE_SCIPY else "numpy Catmull-Rom (scipy absent)",
    "frames_in": int(n_in),
    "frames_out": int(out.shape[0]),
    "frames_out_expected": int(round((n_in - 1) / speed)) + 1,
    "duration_in_s": round(dur_in, 3),
    "duration_out_s": round((out.shape[0] - 1) / fps, 3),
    "quat_norm_min": float(qn.min()),
    "quat_norm_max": float(qn.max()),
    "max_joint_vel_in_rad_s": round(vin, 4),
    "max_joint_vel_out_rad_s": round(vout, 4),
    "vel_ratio_out_over_in": round(vout / vin, 4) if vin > 0 else None,
    "vel_ratio_expected": speed,
  }
  ok = (
    check["frames_out"] == check["frames_out_expected"]
    and abs(check["quat_norm_min"] - 1.0) < 1e-3
    and abs(check["quat_norm_max"] - 1.0) < 1e-3
    # cubic interp can slightly over/undershoot local peaks; 15% band on the ratio
    and (vin == 0 or abs(vout / vin - speed) < 0.15 * speed)
  )
  check["pass"] = bool(ok)
  return check


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--input", required=True)
  ap.add_argument("--output", required=True)
  ap.add_argument("--speed", type=float, required=True,
                  help="tempo multiplier: 1.0 native, 0.6 = 60%% tempo (longer)")
  ap.add_argument("--fps", type=float, default=30.0)
  ap.add_argument("--check-json", default="", help="also write the self-check here")
  a = ap.parse_args()
  if not (0.1 <= a.speed <= 2.0):
    raise SystemExit(f"--speed {a.speed} out of the sane 0.1-2.0 band")

  check = retime(Path(a.input), Path(a.output), a.speed, a.fps)
  print(json.dumps(check, indent=2))
  if a.check_json:
    Path(a.check_json).parent.mkdir(parents=True, exist_ok=True)
    with open(a.check_json, "w") as f:
      json.dump(check, f, indent=2)
  if not check["pass"]:
    print("RETIME CHECK FAIL", file=sys.stderr)
    return 1
  print("RETIME CHECK PASS")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
