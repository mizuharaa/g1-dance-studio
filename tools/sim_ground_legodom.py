#!/usr/bin/env python3
"""Offline end-to-end smoke of the LEG-ODOMETRY ground path — no robot.
Reproduces mode_ground_run_legodom's obs (real base_lin_vel + hybrid anchor from leg odom)
over the full motion using the reference as 'perfect tracking', runs the REAL policy, and
checks actions stay finite and within the action cap."""
import numpy as np, onnxruntime as ort
import pipeline.deploy_runtime as dr
from pipeline.leg_odometry import LegOdometry
PELVIS=0
meta=dr.Meta(dr.DEFAULT_META); ref=dr.Reference(dr.DEFAULT_MOTION)
d=np.load(dr.DEFAULT_MOTION); bq=d["body_quat_w"]; ba=d["body_ang_vel_w"]
sess=ort.InferenceSession(str(dr.DEFAULT_POLICY),providers=["CPUExecutionProvider"])
odo=LegOdometry(list(meta.joint_order))
last=np.zeros(meta.n); maxa=0.0; over=0; bad=0; blv=[]
R0=dr.quat_wxyz_to_mat(bq[0,PELVIS]); h0=odo.estimate(ref.jp[0],ref.jv[0],R0,R0.T@ba[0,PELVIS])[1]
for t in range(ref.T):
    q=ref.jp[t]; dq=ref.jv[t]; imu=bq[t,PELVIS]; R=dr.quat_wxyz_to_mat(imu); gyro=R.T@ba[t,PELVIS]
    v_body,h_est,_=odo.estimate(q,dq,R,gyro); v_world=R@v_body
    rd=ref.at(t)[2]-ref.apos[0]; robot_disp=np.array([rd[0],rd[1],h_est-h0])
    obs,terms=dr.build_obs_odom(meta,ref,q,dq,imu,gyro,last,t,robot_disp,v_world)
    if not np.all(np.isfinite(obs)): bad+=1; continue
    a=dr.run_policy(sess,obs,t)
    if not np.all(np.isfinite(a)): bad+=1; continue
    last=a; maxa=max(maxa,float(np.abs(a).max())); over+=int(np.any(np.abs(a)>dr.GROUND_MAX_ACTION))
    blv.append(float(np.linalg.norm(terms["base_lin_vel"])))
print(f"ticks={ref.T} non-finite={bad}")
print(f"base_lin_vel (leg odom) mag: mean {np.mean(blv):.3f} max {np.max(blv):.3f} m/s")
print(f"action |a|max={maxa:.2f} (GROUND cap {dr.GROUND_MAX_ACTION}); ticks over GROUND cap: {over}/{ref.T}")
print("RESULT:", "PASS (finite, bounded)" if bad==0 and maxa<dr.MAX_ACTION+4 else "CHECK")
