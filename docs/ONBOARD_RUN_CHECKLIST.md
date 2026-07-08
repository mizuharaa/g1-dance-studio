# Onboard wireless dance — FIRST RUN operator checklist

**Purpose:** the exact, in-order steps to run our Thriller policy onboard the robot (PC2 /
Jetson) inside the container's BeyondMimic `motion_tracking_controller`, with the trigger
wireless. This is the **first-of-its-kind onboard control run** — tether-first, operator
present, remote in hand. Do not skip a phase.

> Status feeding this: `policy_onboard.onnx` is VERIFIED drop-in compatible and STAGED on PC2
> (`~/onboard_deploy/`). Launch is **config + onnx only, no `colcon build`**. Full analysis:
> `docs/ONBOARD_DEPLOY.md` (BREAKTHROUGH + VERIFIED sections).

---

## 0. SAFETY GATE — do not proceed until ALL true
- [ ] Robot on the **gantry** (or firmly hung), **feet OFF the ground** for the first launch.
- [ ] **Remote in hand**, thumb near **B (damp)**; you know the **power switch** location.
      (This G1 has **no torque-cut hardware e-stop** — B-damp + power switch are the only hard stops.)
- [ ] Clear space, nobody within the robot's arm/leg sweep.
- [ ] You (operator) are physically present and running these steps — Claude does NOT send
      motor commands; every activation below is a human keypress.
- [ ] The robot's normal control stack (`master_service`) is up and the remote already holds
      the robot **standing / damped** as usual before we hand to the controller.

## 1. Get onto PC2 and into the container
```bash
# from the laptop (or directly on PC2):
ssh unitree@<PC2>            # PC2 on the control net / tailscale
docker ps | grep g1-siu-deploy         # confirm the container is up (id was 477c232a485c)
docker exec -it <container> bash       # 'unitree' is in the docker group — no sudo
# inside the container:
source /opt/ros/jazzy/setup.bash
ls ~/onboard_deploy/                    # policy_onboard.onnx must be here (staged)
```
If `policy_onboard.onnx` is missing from the container's view, it's staged at the PC2 host
`~/onboard_deploy/` and `~/g1-dance/data/policies/thriller_standtail_candidate/` — copy it in.

## 2. Gains — VERIFIED they ride in the ONNX metadata (no yaml patch needed)
File: `/ws/src/motion_tracking_controller/config/g1/controllers.yaml`
- **VERIFIED 2026-07-08 on PC2**: `real.launch.py` loads our policy into the **`walking_controller`**
  block (`walking_controller.policy.path`, set from the `policy_path:=` launch arg) and spawns it
  **INACTIVE**; the robot comes up in `standby_controller`. The `walking_controller` block has **NO
  kp/kd/action_scale/default in the yaml** — only `update_rate` + `ramp_seconds`. So its gains come
  **from the ONNX metadata** (`Policy::getJointStiffness/getJointDamping`, parsed by
  `OnnxPolicy::parseMetadata`). The **350/200/300** gains you may see in the yaml belong to the
  separate `standby_controller`, NOT to our policy.
- **VERIFIED**: `~/onboard_deploy/policy_onboard.onnx` metadata carries OUR values —
  `joint_stiffness 40.179/99.098/28.501…`, `joint_damping 2.558/6.309/1.814…`,
  `default_joint_pos −0.312/0.669/−0.363…`, per-joint `action_scale`, `anchor_body_name torso_link`,
  correct obs order + joint/body names. So **do NOT edit the yaml gains** — they aren't used for
  walking_controller and editing standby's would be wrong.
- ⚠️ **Residual unknown (cross-check at launch, below)**: the code that *applies* metadata gains to
  the motors lives in the **compiled** `RlController.cpp` (only headers ship in `/opt/ros`), so
  "on_activate pushes metadata kp/kd to the actuators" is a strong inference, NOT source-verified.
  The startup-log check in §3 is the mandatory cross-check before any feet-down activation.

## 3. Launch (feet OFF / on the gantry)
```bash
ros2 launch motion_tracking_controller real.launch.py \
  robot_type:=g1 \
  policy_path:=~/onboard_deploy/policy_onboard.onnx \
  motion.start_step:=0
```
- [ ] Watch the startup log: policy loaded, metadata parsed (anchor=torso_link, 14 body_names,
      obs order `command,motion_anchor_pos_b,motion_anchor_ori_b,base_lin_vel,base_ang_vel,joint_pos,joint_vel,actions`).
- [ ] **MANDATORY cross-check** (resolves the §2 residual unknown): before activating
      `walking_controller`, confirm from the controller log / `ros2 param` that the gains it will
      apply are **OUR 40/99/28** (NOT 350/300, NOT 0). If you cannot confirm the applied gains are
      ours → **STOP**, do not activate. (Robot is safely in standby until you switch.)
- [ ] Robot comes up in `standby_controller` (feet off, gantry): holds the standby pose, no runaway.

## 4. ACTIVATION HAZARD — avoid the 0.68 rad lurch
The clip's frame 0 differs from the standby default pose by up to **0.68 rad**. Activating
straight onto raw frame 0 lurches. Mitigation (either):
- [ ] Use the **`thriller_deploy` motion** (has a 2.5 s default→dance ramp prepended) and
      `start_step:=0` so the first 2.5 s is a gentle ramp; **or**
- [ ] Interpolate standby→frame0 over 2–3 s in the controller before activating.
Never activate at full pose delta cold.

## 5. Tethered staircase (SAME as the laptop path we already validated)
Bring up in increasing exposure; abort (B-damp) at any wrongness:
- [ ] **Feet off, gantry** — activate, watch the arms/legs track the ramp then early motion. Damp.
- [ ] **Feet on ground, gantry still bearing weight** — activate, watch balance response. Damp.
- [ ] **Gantry slack (robot bearing own weight, tether present)** — short activation, first
      seconds of the dance only. Damp. Inspect. Repeat lengthening the window.
- [ ] **Full run tethered** — the complete Thriller with the standing-end tail; robot stays
      standing at the end (EXIT_MODE=stand handoff), remote takes back over.

## 6. Go wireless (only after a clean tethered full run)
The control loop is **already 100% onboard on eth1** — going "wireless" only moves the
**trigger** off a wire. **VERIFIED 2026-07-08 on PC2**: eth1 UP (192.168.123.164 control net),
wlan0 UP (192.168.21.237), and **tailscale live** — PC2 = `unitree-g1` @ **100.111.44.110**. So
the wireless path exists now.
- [ ] Trigger transport = ros2 topic/action over **tailscale (100.111.44.110) / wlan0** (NOT eth1).
      The laptop must share that path (join tailscale, or be on the same AP) to send the trigger.
- [ ] Preflight the link first (RTT + DDS staleness GO/NO-GO — the show app's wireless preflight).
- [ ] Confirm: pulling wifi mid-dance does **not** stall the 50 Hz loop (it's on eth1) — the
      robot keeps balancing; only new triggers are lost. Verify once, deliberately, tethered.
- [ ] Then: press GO wirelessly → dance → stands at end → remote takes over.

## Abort / stop at any point
1. **B (damp)** on the remote — first reflex.
2. Power switch if damp is insufficient.
3. `Ctrl-C` the ros2 launch (stops new commands; does NOT physically stop a falling robot —
   use the remote first).

## Notes
- Do DDS/onboard work near the live control service only with the operator aware (this checklist
  assumes that).
- If the controller's motion ingestion differs from our export, the motion is baked INTO
  `policy_onboard.onnx` (advances with `time_step`) — there is no separate motion file to load
  for our policy. See `docs/ONBOARD_DEPLOY.md` VERIFIED section.
- After a successful run, record it in `PROJECT_STATE.md` + `logs/jobs.md` (measurement discipline).
