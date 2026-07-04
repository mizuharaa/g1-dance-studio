#!/usr/bin/env python3
"""Validate LegOdometry against the reference ground truth — NO robot.

The reference carries the TRUE base (pelvis) linear velocity (body_lin_vel_w) and pelvis
height (body_pos_w). Feed the reference's joint q/dq + pelvis orientation to LegOdometry
and compare its estimated base_lin_vel + height to the truth. Since base_lin_vel trained
with ±0.5 m/s noise, "good enough" = the estimate stays within that band on most frames.

    conda activate g1dance && PYTHONPATH=. python tools/validate_leg_odom.py
"""
from __future__ import annotations
import numpy as np
import pipeline.deploy_runtime as dr
from pipeline.leg_odometry import LegOdometry

PELVIS_NPZ = 0   # model body 1 (pelvis) minus the dropped world body


def main():
    meta = dr.Meta(dr.DEFAULT_META)
    d = np.load(dr.DEFAULT_MOTION)
    jp, jv = d["joint_pos"], d["joint_vel"]
    bpos, bquat = d["body_pos_w"], d["body_quat_w"]
    blin, bang = d["body_lin_vel_w"], d["body_ang_vel_w"]
    T = jp.shape[0]

    odo = LegOdometry(list(meta.joint_order))

    err_v, err_h, mag_true, within_tol = [], [], [], 0
    for t in range(T):
        R = dr.quat_wxyz_to_mat(bquat[t, PELVIS_NPZ])       # pelvis body->world
        gyro_body = R.T @ bang[t, PELVIS_NPZ]               # world ang vel -> body
        v_est, h_est, _ = odo.estimate(jp[t], jv[t], R, gyro_body)
        v_true_body = R.T @ blin[t, PELVIS_NPZ]             # true pelvis vel in body frame
        h_true = float(bpos[t, PELVIS_NPZ, 2])
        e = v_est - v_true_body
        err_v.append(e)
        err_h.append(h_est - h_true)
        mag_true.append(np.linalg.norm(v_true_body))
        within_tol += int(np.all(np.abs(e) <= 0.5))         # inside the trained noise band

    err_v = np.array(err_v); err_h = np.array(err_h); mag_true = np.array(mag_true)
    per_axis_rmse = np.sqrt((err_v ** 2).mean(axis=0))
    mag_err = np.linalg.norm(err_v, axis=1)
    print(f"frames={T}  true base speed: mean {mag_true.mean():.3f}  max {mag_true.max():.3f} m/s")
    print("--- base_lin_vel estimate error (body frame) ---")
    print(f"  per-axis RMSE (x,y,z) = {np.round(per_axis_rmse,3).tolist()} m/s")
    print(f"  |error| mean {mag_err.mean():.3f}  median {np.median(mag_err):.3f}  "
          f"p95 {np.percentile(mag_err,95):.3f}  max {mag_err.max():.3f} m/s")
    print(f"  frames with ALL axes within ±0.5 m/s (trained noise band): "
          f"{within_tol}/{T} ({100*within_tol/T:.1f}%)")
    print("--- base height estimate error ---")
    print(f"  RMSE {np.sqrt((err_h**2).mean()):.3f} m  mean {err_h.mean():+.3f}  "
          f"p95|err| {np.percentile(np.abs(err_h),95):.3f} m")
    good = per_axis_rmse.max() < 0.5 and (within_tol / T) > 0.8
    print("\nVERDICT:", "USABLE (within trained tolerance on most frames)" if good
          else "MARGINAL/NO — see error above")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
