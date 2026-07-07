# Onboard policy deploy (the wireless show) — status + debug runbook

**Why onboard:** running the 50 Hz balance loop on the laptop over wifi is a fall risk
(jitter/dropout → stale commands). The correct design (what Unitree does) is to run the
policy **onboard PC2** (the Jetson) on the local control net (eth1), so the real-time loop
never touches wifi; wifi/tailscale carries only the trigger (start/stop). This removes the
wireless-control risk entirely.

## What is already SET UP on PC2 (2026-07-07)
- `teleimager` conda env (Python 3.10) has `unitree_sdk2py` + `cyclonedds` + numpy 1.26.
- **onnxruntime 1.23.2 installed** into `teleimager` (aarch64 wheel — the tiny MLP runs on CPU
  in sub-ms; the Jetson is plenty).
- **Code + policy bundled** to `~/g1-dance` on PC2: `pipeline/{__init__,deploy_runtime,leg_odometry}.py`
  + `data/policies/thriller_standtail_candidate/{policy.onnx,policy_meta.json,thriller_deploy.npz}`.
  deploy_runtime is nearly standalone (numpy at top; SDK lazy-imported) so this minimal set runs it.
- `IFACE` is env-configurable (`ROBOT_IFACE`); onboard we use **eth1** (192.168.123.164 = the local
  control net). wlan0 (192.168.21.237) + tailscale are available for the trigger.

## The BLOCKER (needs a session with the operator at the robot)
Running the onboard subscriber fails at topic creation:
```
CYCLONEDDS_URI=/home/kc_ws/cyclonedds.xml \
  ~/miniconda3/envs/teleimager/bin/python -m pipeline.deploy_runtime --mode read --iface eth1 ...
-> cyclonedds.core.DDSException: [DDS_RETCODE_PRECONDITION_NOT_MET]
   Occurred upon initialisation of a cyclonedds.topic.Topic  (rt/lowstate)
```
`PRECONDITION_NOT_MET` on a Topic almost always = **a topic of that name already exists with an
incompatible type descriptor**. PC2's onboard `master_service` (running) owns `rt/lowstate` with the
type from ITS SDK build. Our Python subscriber uses PC2's **`kc_ws` SDK (sha 58c3f62)**, whose
`LowState_` IDL evidently does not match.

**Key evidence pointing at SDK/IDL version, not config:**
- The **laptop reads `rt/lowstate` fine** over ethernet using `~/robot/unitree_sdk2_python` — so that
  SDK's `LowState_` type IS compatible with `master_service`. PC2's `kc_ws` SDK is a *different* build.
- Matching the robot's own DDS XML (`CYCLONEDDS_URI=/home/kc_ws/cyclonedds.xml`) did **not** fix it
  (and that XML even names `eth0` while the control net is `eth1` — likely not master_service's actual
  config anyway).

## ROOT CAUSE FOUND (2026-07-07, six approaches tried)
It is **not** the SDK version and **not** the `rt/lowstate` type specifically. A minimal probe
subscribing to a **benign topic name** (`rt/PROBE_benign_xyz`) with the `LowState_` type **also**
fails `PRECONDITION_NOT_MET`. `ChannelFactoryInitialize` succeeds; the FIRST `ChannelSubscriber.Init()`
(topic/type creation) fails. So it is a **domain-level TYPE-registration conflict**: `master_service`
(C++) has already registered the `unitree_hg` `LowState_` type in the DDS domain, and our co-located
Python participant registering the same fully-qualified type name with a different sertype is rejected —
for ANY topic name. The laptop avoids this because it is a separate host whose own domain instance
negotiates the type over the wire (XTypes), never re-registering into `master_service`'s registry.

Tried and FAILED (all same PRECONDITION_NOT_MET): default DDS config; `/home/kc_ws/cyclonedds.xml`;
the laptop's working `~/robot` SDK on PC2; an explicit eth1 + SharedMemory-off config; benign topic
name; both SDKs. => A **parallel Python DDS subscriber co-located with `master_service` cannot register
the type.** This is the wrong integration shape, not a tuning problem.

## THE RIGHT PATH (needs Unitree's onboard method + operator; this is how their demos work)
Unitree's onboard policy demos do NOT stand up a competing participant. Options, best first:
1. **Run the policy inside the robot's control framework/container** (the motion_tracking_controller /
   `qiayuanl/unitree:jazzy` image referenced in docs/architecture.md), which already owns the compatible
   `LowState_` type and the control loop — deploy the ONNX + our obs/action glue there. Needs `sudo`
   docker access + Unitree's onboard-deploy docs.
2. **Ask Unitree** for the supported way to read `rt/lowstate` from a second onboard process (a
   CycloneDDS XTypes / type-discovery config that permits type coexistence, or a shared type library).
3. **Fallback if wireless is required sooner:** laptop-in-the-loop but WIRELESS — see docs/WIRELESS_SHOW.md
   + tools/wireless_preflight.py. HARD constraint: the control net (192.168.123.x) is physically the
   ethernet/eth1; wifi does not reach it without a bridge, and tailscale adds VPN latency unfit for 50 Hz.
   So this needs the robot to BRIDGE the control net onto wifi (or a wifi AP on the control subnet) AND
   the preflight (RTT + DDS staleness p99 < ~10 ms, 0 loss, sustained) to PASS first. Higher risk than
   onboard; the comms-loss deadman is the only backstop.

## Superseded debug plan (kept for history — the SDK-swap hypothesis, now DISPROVEN)
1. **Align the SDK.** Put the laptop's WORKING SDK (`~/robot/unitree_sdk2_python`, the one whose
   `LowState_` matches `master_service`) onto PC2 and import IT instead of `kc_ws`'s (PYTHONPATH or a
   venv install). Re-run `--mode read --iface eth1`. This is the leading hypothesis: same SDK the
   laptop uses → same type descriptor → topic compatible. (unitree_sdk2py is pure-Python IDL, so
   arch-independent; only cyclonedds is native and already present.)
2. If still failing, **inspect the live type**: which SDK/commit built `master_service` (ask Unitree
   / the robot image docs), and match `unitree_sdk2py` to it. Compare the `LowState_` IDL hash.
3. **SHM/iceoryx check:** confirm whether `iox-roudi` is running (co-located SHM transport). If so, a
   type mismatch is fatal over SHM; align types OR force network transport via a CYCLONEDDS_URI that
   disables shared memory, and retest.
4. **Domain/participant:** confirm the domain id `master_service` uses; our participant must join the
   same one. The kc_ws XML uses `Domain id="any"`.

## Once `--mode read` works onboard (still no motor commands)
- It prints finite/bounded actions from the real onboard `rt/lowstate` → the onboard policy path is
  proven. Then, and only then, with the OPERATOR PRESENT + remote + tether:
  - onboard `--mode ground-run-legodom` (the full safety spine — entry catch, fall detector, exit
    stand handoff, start-pose guard — is in the bundled deploy_runtime), tethered first, exactly like
    the laptop staircase we already validated.
- **Trigger:** wrap the onboard run in a small script on PC2; fire it wirelessly (ssh over tailscale/
  wlan0, or map a remote button). The trigger is not real-time; only the local eth1 control loop is.

## Safety notes
- This G1 has no torque-cut e-stop; the remote's B-damp + power switch are the only hard stops.
- Onboard motion is a first-of-its-kind run for this project — tether-first, operator-present, and
  the comms path in the loop is now LOCAL (eth1), which is the whole point (no wifi jitter in control).
- Do the SDK/DDS debugging near the live control service ONLY with the operator aware.

## BREAKTHROUGH (2026-07-07): the robot ALREADY has the onboard controller — use it, don't fight DDS
The `g1-siu-deploy:jazzy` docker image (unitree is in the docker group — no sudo) contains
`/ws/src/motion_tracking_controller` — the **BeyondMimic MotionTrackingController** (the architecture's
original onboard target). It runs the tracking policy onboard inside the ROS2 control framework that
OWNS the compatible DDS types — so it sidesteps the type-registration conflict entirely (that conflict
was from a *competing* Python participant; this is the *right* participant).
COMPATIBILITY with our mjlab policy is HIGH:
- joint_names: IDENTICAL 29-dof order.
- default_position: MATCHES (hip_pitch -0.312, knee 0.669, ...; ours -0.363 vs its -0.33 ankle — trivial).
- obs terms present incl. `motion_anchor_pos_b`; anchor = torso_link (== ours).
- GAINS DIFFER and MUST be overridden: controller default kp/kd = 350/300 (BeyondMimic); OUR policy
  trained at kp 40/99/28, kd 2.6/6.3/1.8 — deploying at 350 would be wildly out-of-distribution -> fall.
  Config values generated from policy_meta and staged at PC2 `~/onboard_deploy/onboard_controller_cfg.txt`.
STAGED on PC2 `~/onboard_deploy/`: policy.onnx (standtail), thriller_deploy.npz, controller config values.
LAUNCH shape (config-only, no build): `ros2 launch motion_tracking_controller real.launch.py
robot_type:=g1 policy_path:=~/onboard_deploy/policy.onnx start_step:=...` after patching
config/g1/controllers.yaml walking_controller.{kp,kd,action_scale,default} to OUR values.
REMAINING (needs the operator at the robot — first onboard control run):
1. VERIFY the controller's motion format: does it ingest our thriller_deploy.npz (body_pos_w/quat/joint_*)
   or a different BeyondMimic motion? Check MotionCommand.h / how the motion is loaded. Adapt if needed.
2. VERIFY the obs construction byte-matches ours (velocimeter lever-arm at imu_in_pelvis, no
   projected-gravity, action_scale per-joint) — subtle diffs false-fail a good policy.
3. Patch controllers.yaml gains -> our values; run `real.launch.py` with feet OFF / on the gantry first.
4. Full tethered staircase (same as the laptop path we already validated), THEN the trigger goes wireless
   (ros2 action/topic over wifi/tailscale — control loop stays 100% onboard on eth1).
This is the CORRECT wireless answer: the 50 Hz loop is onboard (never on wifi); wifi carries only the trigger.

## VERIFIED (2026-07-07): our policy is a DROP-IN for the onboard controller (metadata was the only gap)
Read the controller's loader (`MotionOnnxPolicy.cpp`, `OnnxPolicy.h`) + a reference g1 policy in
the container. Findings:
- The controller's `MotionOnnxPolicy::forward` feeds `time_step` and reads outputs
  `joint_pos, joint_vel, body_pos_w, body_quat_w` — the REFERENCE MOTION IS BAKED INTO THE ONNX.
  OUR mjlab export ALREADY has exactly these (input obs[160]+time_step; those 4 outputs + actions).
  Smoke-tested: t0 starts at default pose, motion advances 0.680 rad by t167 (== the clip's known
  0.68 rad range), actions finite |max|=1.75. So the motion is NOT a separate-file problem.
- Every obs term in our layout has a handler: `command`(=MotionCommandTerm joint_pos+joint_vel),
  `base_lin_vel/base_ang_vel/joint_pos/joint_vel/actions` (base `RlController::parserObservation`),
  `motion_anchor_pos_b/ori_b` (MotionTracking). Our order matches training.
- THE ONLY GAP: our ONNX had EMPTY metadata; the controller reads policy config from ONNX metadata
  (`joint_names, joint_stiffness, joint_damping, default_joint_pos, action_scale, observation_names,
  command_names` + `anchor_body_name, body_names`). FIXED on the laptop: `tools/make_onboard_onnx.py`
  injects them from policy_meta (gains 40/99/28, per-joint action_scale, obs order, torso_link anchor,
  14 body_names). Verified read-back. => `policy_onboard.onnx` (staged PC2 `~/onboard_deploy/`).
REMAINING to actually run it (operator-present, first onboard control run):
1. Point `config/g1/controllers.yaml` policy_path at policy_onboard.onnx (gains now travel IN the onnx,
   but confirm the yaml doesn't override with the 350/300 defaults — set yaml gains to ours too / or
   confirm metadata wins).
2. `colcon build` is NOT needed (config + onnx only). `ros2 launch motion_tracking_controller
   real.launch.py robot_type:=g1 policy_path:=~/onboard_deploy/policy_onboard.onnx motion.start_step:=0`.
3. Gantry/feet-off first -> tethered staircase (same as the laptop path we validated) -> then wireless
   trigger (ros2 topic/action over wlan0/tailscale; the 50 Hz loop stays onboard on eth1). ACTIVATION
   HAZARD: start_step must begin at the 2.5s default->dance ramp (our thriller_deploy motion has it), or
   interpolate standby->frame0 for 2-3s before activating, else a 0.68 rad lurch.
This is the wireless answer AND the onboard bundle: verified-compatible policy, injected, staged.
