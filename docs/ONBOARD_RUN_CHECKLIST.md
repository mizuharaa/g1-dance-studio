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

## 2. Point the controller at OUR policy + confirm gains
File: `/ws/src/motion_tracking_controller/config/g1/controllers.yaml`
- [ ] Set `policy_path` → `~/onboard_deploy/policy_onboard.onnx`.
- [ ] **CRITICAL — gains.** The controller's yaml defaults are BeyondMimic **kp 350 / kd 300**.
      OUR policy trained at **kp 40.2 / 99.1 / 28.5, kd 2.56 / 6.31 / 1.81** (per-joint, in the
      onnx metadata). Deploying at 350/300 is wildly out-of-distribution → **fall**.
      Our gains now travel *inside* `policy_onboard.onnx` metadata, BUT the yaml may still
      override. **Confirm which wins** before feet-down:
      - Preferred: set the yaml `kp/kd/action_scale/default_position` to OUR values too
        (source of truth: `data/policies/thriller_standtail_candidate/policy_meta.json`, and the
        staged `~/onboard_deploy/onboard_controller_cfg.txt`), so yaml and metadata agree.
      - At minimum, log the gains the controller actually loaded at startup and eyeball them.
- [ ] `default_position`: ours hip_pitch −0.312, knee 0.669 (yaml ankle −0.33 vs ours −0.363 is trivial).

## 3. Launch (feet OFF / on the gantry)
```bash
ros2 launch motion_tracking_controller real.launch.py \
  robot_type:=g1 \
  policy_path:=~/onboard_deploy/policy_onboard.onnx \
  motion.start_step:=0
```
- [ ] Watch the startup log: policy loaded, metadata parsed (anchor=torso_link, 14 body_names,
      obs order `command,motion_anchor_pos_b,motion_anchor_ori_b,base_lin_vel,base_ang_vel,joint_pos,joint_vel,actions`),
      **gains = OUR 40/99/28** (NOT 350/300). If gains read 350/300 → **STOP**, fix step 2.
- [ ] With feet off: the controller should hold near the default pose without runaway.

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
**trigger** off a wire:
- [ ] Trigger transport = ros2 topic/action over **wlan0 / tailscale** (NOT the control net).
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
