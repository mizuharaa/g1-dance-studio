#!/usr/bin/env python3
"""Offline validation of the odometry-fed ground obs path — NO robot.

Open-loop: assume the robot tracks the reference, synthesize the onboard estimate
(rt/odommodestate) from the reference trajectory (+ optional noise/drift), build the
HONEST 160-D obs with build_obs_odom, and run the REAL ONNX policy through the whole
motion. Checks that every obs/action is finite and bounded, and quantifies how much the
honest base_lin_vel / anchor terms change the policy vs the gantry fakes (build_obs).

This does NOT prove ground stability (that needs the tethered bring-up) — it proves the
obs pipeline feeds the policy sane inputs and the policy stays within the action cap.

    conda activate tv && python tools/sim_ground_odom.py [--noise]
"""
from __future__ import annotations
import argparse
import numpy as np
import onnxruntime as ort
import pipeline.deploy_runtime as dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", action="store_true", help="add realistic odom noise+drift")
    ap.add_argument("--vel-noise", type=float, default=0.05, help="m/s std on world velocity")
    ap.add_argument("--pos-drift", type=float, default=0.02, help="m drift over the whole run")
    a = ap.parse_args()

    meta = dr.Meta(dr.DEFAULT_META)
    ref = dr.Reference(dr.DEFAULT_MOTION)
    sess = ort.InferenceSession(str(dr.DEFAULT_POLICY), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)

    # reference torso world velocity (finite diff), used as ideal v_world
    v_ref = np.zeros((ref.T, 3))
    v_ref[1:] = np.diff(ref.apos, axis=0) * dr.CONTROL_HZ

    last_a_h = np.zeros(meta.n)   # honest (odom) path
    last_a_g = np.zeros(meta.n)   # gantry-fake path
    max_a_h = 0.0
    n_over_ground = 0             # honest actions exceeding GROUND_MAX_ACTION
    n_over_gantry = 0            # honest actions exceeding MAX_ACTION (gantry cap)
    blv_mag = []                 # honest base_lin_vel magnitude per tick
    act_delta = []               # |honest action - fake action| per tick
    bad = 0

    for t in range(ref.T):
        q = ref.jp[t].copy()            # robot perfectly tracks -> q = reference (absolute)
        dq = ref.jv[t].copy()
        imu_quat = ref.aquat[t].copy()
        gyro = np.zeros(3)
        robot_disp = (ref.apos[t] - ref.apos[0]).copy()
        v_world = v_ref[t].copy()
        if a.noise:
            v_world = v_world + rng.normal(0, a.vel_noise, 3)
            robot_disp = robot_disp + (t / ref.T) * a.pos_drift * rng.standard_normal(3)

        obs_h, terms_h = dr.build_obs_odom(meta, ref, q, dq, imu_quat, gyro, last_a_h, t,
                                           robot_disp, v_world)
        obs_g, _ = dr.build_obs(meta, ref, q, dq, imu_quat, gyro, last_a_g, t)
        if not (np.all(np.isfinite(obs_h)) and np.all(np.isfinite(obs_g))):
            bad += 1
            continue
        act_h = dr.run_policy(sess, obs_h, t)
        act_g = dr.run_policy(sess, obs_g, t)
        if not (np.all(np.isfinite(act_h)) and np.all(np.isfinite(act_g))):
            bad += 1
            continue
        last_a_h, last_a_g = act_h, act_g
        max_a_h = max(max_a_h, float(np.abs(act_h).max()))
        n_over_ground += int(np.any(np.abs(act_h) > dr.GROUND_MAX_ACTION))
        n_over_gantry += int(np.any(np.abs(act_h) > dr.MAX_ACTION))
        blv_mag.append(float(np.linalg.norm(terms_h["base_lin_vel"])))
        act_delta.append(float(np.abs(act_h - act_g).mean()))

    print(f"ticks={ref.T}  noise={'on' if a.noise else 'off'}  non-finite={bad}")
    print(f"honest base_lin_vel magnitude: mean {np.mean(blv_mag):.3f}  max {np.max(blv_mag):.3f} m/s "
          f"(vs gantry fake = 0)")
    print(f"honest action |a|max over run = {max_a_h:.2f}  "
          f"(GROUND cap {dr.GROUND_MAX_ACTION}, gantry cap {dr.MAX_ACTION})")
    print(f"ticks with |action| > GROUND cap: {n_over_ground}/{ref.T}  "
          f"| > gantry cap: {n_over_gantry}/{ref.T}")
    print(f"mean |honest_action - gantry_fake_action| = {np.mean(act_delta):.4f} "
          f"(how much feeding real vel/pos changes the policy)")
    ok = bad == 0 and max_a_h < dr.MAX_ACTION
    print("RESULT:", "PASS (finite, within gantry cap)" if ok else "CHECK (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
