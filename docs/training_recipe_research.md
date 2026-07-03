# Training recipe — FULL (ultracode research, 2026-07-03)

Supersedes the partial version. 59 findings across 5 source angles.

# G1 Dance Tracking Training Recipe (BeyondMimic/mjlab)

## 1. Baseline — reference implementation exact values (all **verified** from HybridRobotics/whole_body_tracking, mirrored by mjlab tracking task)

**Rewards (9 terms, exp(-mse/std²) kernels over 14 bodies: pelvis, hip_roll/knee/ankle_roll L+R, torso, shoulder_roll/elbow/wrist_yaw L+R; anchor = torso_link):**

| Term | Weight | std / param |
|---|---|---|
| motion_global_anchor_pos | 0.5 | std=0.3 |
| motion_global_anchor_ori | 0.5 | std=0.4 |
| motion_body_pos (anchor-relative, yaw/xy-aligned) | 1.0 | std=0.3 |
| motion_body_ori | 1.0 | std=0.4 |
| motion_body_lin_vel (global) | 1.0 | std=1.0 |
| motion_body_ang_vel (global) | 1.0 | std=3.14 |
| action_rate_l2 | -0.1 | — |
| joint_pos_limits | -10.0 | — |
| undesired_contacts (WBT) / self_collision (mjlab, w=-10.0, 10N threshold, history_length=4) | -0.1 | threshold 1.0N |

**Terminations:** anchor z error >0.25 m; anchor ori error >0.8 rad; end-effector (ankle_roll + wrist_yaw) z error >0.25 m; 10 s timeout. Position terminations are **z-only** — xy drift never terminates (deliberate for long clips).

**Episode/RSI/sampling:** episode_length_s=10.0 (fixed, independent of clip length); RSI on every reset with perturbation pose ±0.05 m xy / ±0.01 m z / ±0.1 rad roll-pitch / ±0.2 yaw, joints ±0.1 rad, velocity ±0.5 m/s xy / ±0.2 z / ±0.52 rp / ±0.78 yaw rad/s. Adaptive sampling over 1-s bins: adaptive_alpha=0.001, adaptive_uniform_ratio=0.1, adaptive_kernel_size=1, adaptive_lambda=0.8.

**PPO (RSL-RL):** num_envs=4096, num_steps_per_env=24 (≈98k samples/iter), max_iterations=30000, save_interval=500, MLP [512,256,128] ELU actor+critic (asymmetric: critic gets clean privileged obs incl. body pos/ori), init_noise_std=1.0, empirical/obs normalization ON, lr=1e-3 adaptive-KL desired_kl=0.01, clip=0.2, entropy_coef=0.005, epochs=5, minibatches=4, gamma=0.99, lam=0.95, value_loss_coef=1.0, max_grad_norm=1.0.

**Sim/control:** dt=0.005, decimation=4 → 50 Hz policy; motion resampled to 50 fps. mjlab solver: iterations=10, ls_iterations=20, nconmax=35, njmax=250.

**Actuators (G1):** armature from rotor inertia × gear² (5020: 0.0036097; 7520_14: 0.0101775; 7520_22: 0.0251019; 4010: 0.00425; ankles/waist_rp use 2× 5020). kp = armature·(2π·10)² (10 Hz natural freq), kd = 2·ζ·armature·2π·10 with ζ=2 (overdamped, deliberately LOW gains). Effort limits: hip_yaw/pitch 88, hip_roll/knee 139, ankles/waist_rp 50, waist_yaw 88, shoulders/elbow/wrist_roll 25, wrist_pitch/yaw 5 Nm. Velocity limits 20–37 rad/s. action_scale = 0.25·effort_limit/kp per joint. soft_joint_pos_limit_factor=0.9, self-collisions ON.

**DR/events (the full set that shipped on real G1 — "surprisingly simple" per maintainer):** friction static U(0.3,1.6) / dynamic U(0.3,1.2) / restitution U(0,0.5) (mjlab: foot friction 0.3–1.2 + encoder bias); joint default-pos offset ±0.01 rad; torso CoM ±0.025 x / ±0.05 y,z m; push every 1–3 s (±0.5 m/s xy, ±0.2 z, ±0.52 rp, ±0.78 yaw rad/s). Obs noise: joint_pos ±0.01, joint_vel ±0.5–1.5, base_lin_vel ±0.5, base_ang_vel ±0.2, anchor_pos ±0.25 m (deliberate — odometry robustness), anchor_ori ±0.05. **No** PD-gain randomization, **no** delay, **no** action filtering.

Sources: `tracking_env_cfg.py`, `rsl_rl_ppo_cfg.py`, `robots/g1.py`, `mdp/commands.py` in HybridRobotics/whole_body_tracking; mjlab `src/mjlab/tasks/tracking/*`.

---

## 2. Recommended deltas for the 44.3 s Thriller run

Priority order — fix the motion, not the RL:

1. **Clean the reference before any training** (verified — arxiv 2510.02252, KungfuBot). Your 3.1% frames >3π rad/s and the velocity spike are at the level shown to measurably degrade tracking. Re-run GMR with `use_velocity_limit=True` (clamps to 3π; added Aug 2025) or clamp joint velocities to ≤90% of 3π and smooth the spike; apply FK ground-height correction (min body height subtracted from global z). One 7.5 rad/s spike can dominate the adaptive-sampling bins and stall convergence.
2. **Add standing-pose transitions** (verified — g1_spinkick_example): prepend 0.5 s standing→pose blend + ~1 s padding and append pose→standing + **2–3 s static hold** at the end (verified code behavior: final pose is never held during training otherwise — no clean finish for the show).
3. **Convert with mjlab's own converter, never WBT's** (verified — mjlab issue #777): MuJoCo depth-first vs PhysX breadth-first body ordering makes WBT NPZs silently wrong. Symptom: episode_length pinned at ~1, robot vibrates.
4. **Otherwise run stock config** (verified — maintainer claims LAFAN1 trains "without tuning any parameters"; mjlab nightly hits 95–99% success with defaults). Do not touch rewards or PPO first.
5. Conditional deltas only if problems appear:
   - Jitter/foot chatter on hardware: action_rate_l2 −0.1 → −0.2 (likely; mjlab docs suggestion); optionally ExBody2-style ankle-action −0.1 and waist roll/pitch −0.5 (verified ExBody2 used them on a 43 s G1 dance).
   - Policy rigid on expressive arm sections: entropy_coef 0.005 → 0.01 (verified RobotDancing used 0.01).
   - Poor foot placement: upweight feet ~2× body per ASAP (verified ASAP values).
   - Stage-position drift matters: raise global anchor weights 0.5 → 1.0 (verified mechanism; value is assumption).
6. **Budget:** ~10k iterations (verified BeyondMimic Table II for long clips); checkpoints every 500 iters, ONNX auto-exported.

---

## 3. Long-dance (2–3 min) strategy — ranked

**Option 1 (commit to this): single-clip training, stock pipeline.** *Verified.* Episode length is decoupled from clip length (episode_length_s stays 10.0 — maintainer explicitly confirmed in WBT issue #21); RSI + adaptive sampling over ~120–180 one-second bins covers the clip. Direct existence proofs: BeyondMimic trained multi-minute LAFAN1 (dance1_subject1, 118 s) on real G1; RobotDancing trained eight ~3-min dances as single sequences on real G1 zero-shot (~15k iters each, 89 mm mean body pos error, ~0.5 m global drift); OmniTrack ran hour-long G1 tracking with the same bin sampling. Memory is a non-issue (~9000 frames ≈ tens of MB). **Do not segment; do not raise episode_length_s; do not shrink num_envs.**

Tuning within option 1 (likely): raise adaptive_kernel_size 1 → 4–8 (lambda 0.8) so failure credit spreads to run-up bins — default 1 disables the paper's kernel; consider adaptive_uniform_ratio 0.1 → 0.3 because of the normalization bug in WBT issue #40 that makes the uniform floor nearly ineffective (hard bins starve easy bins on long clips). Monitor `Metrics/motion/sampling_entropy` / `sampling_top1_prob` / `sampling_top1_bin` (already logged): a pinned top-1 bin = reference-feasibility problem in that choreography section, not an RL problem.

**Option 2 (escalation if hard bins won't converge): reference-side + RobotDancing additions.** *Verified techniques, untested in your stack.* (a) physics-consistent reference via sim rollout (OmniTrack); (b) residual actions q_tar = q_ref + a_t (cut position error 15–21% vs absolute); (c) termination-threshold curriculum loose→tight (ASAP-style 1.5 m → 0.3 m anneal) instead of weakening rewards; (d) episode-length-gated penalty curriculum (OmniH2O: penalty scale ×0.9999 when mean ep len <40, ×1.0001 when >120).

**Option 3 (last resort): chunk + stitch separate policies.** Nobody in the BeyondMimic/G1 lineage needed this at ≤3 min; introduces seam-blending problems. Only for >5 min or disjoint skill regimes starving each other. (likely)

Per-dance specialist policies (one per choreography), not one multi-dance generalist — ExBody2 showed specialists win on per-frame DoF error. (verified)

What actually breaks at 2–3 min: (1) iterations to convergence (~1.5–3× your 44 s clip), (2) hardware odometry drift feeding motion_anchor_pos_b — mitigated by the No-State-Estimation task variant; xy drift is architecturally tolerated (z-only terminations, yaw/xy-aligned body rewards, global anchor reward pulls back). Plan choreography tolerant of ±0.25–0.5 m drift and a floor-mark reset between songs. (verified mechanism, likely magnitudes)

---

## 4. mjlab translation — exact commands/config

Pin **mjlab ≥ v1.5.0** (fixes: #1069 forward-sim after resample, #761 command compute on reset, #1006-8 MPKPE metric, #1078 NaN in bad_orientation). `uv sync` from mjlab's lockfile — a stale mujoco_warp pin caused a 6–7× slowdown (discussion #220). CUDA ≥ 12.4 for CUDA graphs.

```bash
# 1. Convert (MANDATORY: mjlab's converter; GMR CSV = base pos, quat xyzw, 29 joints Unitree order)
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
  --input-file thriller.csv --output-name thriller \
  --input-fps 30 --output-fps 50 --render True
# watch the rendered mp4 before training; grab /tmp/motion.npz for local use

# 2. Sanity-check motion kinematics
uv run play Mjlab-Tracking-Flat-Unitree-G1 --agent zero \
  --motion-file ~/motions/thriller.npz --no-terminations

# 3. Train (tmux/nohup on GreenNode)
WANDB_API_KEY=$(cat ~/g1-dance/.secrets/wandb.key) MUJOCO_GL=egl \
uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file ~/motions/thriller.npz \
  --env.scene.num-envs 4096 --agent.max-iterations 15000 \
  --agent.run-name thriller_v1
# resume: --agent.resume True --wandb-run-path <entity>/mjlab/<run-id>

# 4. Evaluate / play
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path entity/mjlab/run-id
# ship gate: evaluate script, 1024 rollouts from frame 0, success_rate >= 0.95
```

CLI override surface (all verified): `--env.rewards.action_rate_l2.weight=-0.2`, `--env.events.push_robot.interval-range-s '(1.0,2.0)'`, adaptive sampling params under `env.commands.motion` (MotionCommandCfg: adaptive_kernel_size, adaptive_uniform_ratio, adaptive_lambda, adaptive_alpha). Task IDs: `Mjlab-Tracking-Flat-Unitree-G1` and `...-No-State-Estimation` (drops motion_anchor_pos_b + base_lin_vel from actor obs — the hardware variant).

**First-500-iters health check (verified failure signatures):** Episode/episode_length must climb past ~100; if pinned at ~1 → bad NPZ (body ordering), stop and reconvert. `Metrics/motion/error_body_pos` should trend down. Add nan_detection termination for overnight runs (MuJoCo Warp NaN crashes are a known class; `--enable-nan-guard True` for debugging only). Avoid pseudo_inertia DR (+4 GB VRAM, #753/#757). Runs are not bitwise-reproducible across seeds (#1023).

**Reference quality bars (mjlab nightly, verified):** success_rate 0.95–0.99, MPKPE ~0.11 m, r-MPKPE ~0.03 m, EE pos error ~0.046 m at 6000 iters on a short motion.

---

## 5. Sim2real checklist for G1 deploy

- [ ] **Identical kp/kd/action_scale train↔deploy** — the #1 G1 failure. "Adjusting these parameters during deployment is fundamentally incorrect" (WBT issue #16/#17, verified). If hardware vibrates at high frequency, raise gains *in training* and retrain.
- [ ] Read kp/kd/action_scale/joint order **from ONNX metadata** in motion_tracking_controller, don't assume WBT's values — mjlab's derived gains differ from WBT's hand-tuned ones, and bug #347 shipped wrong metadata once. Inspect: `python -c "import onnx; m=onnx.load('policy.onnx'); print({p.key:p.value for p in m.metadata_props})"` (verified)
- [ ] Train the **No-State-Estimation** variant for shows (no estimated base_lin_vel/anchor_pos consumed → kills the odometry-drift failure mode over minutes). (verified variant exists; recommendation likely)
- [ ] Keep default DR exactly (Section 1) — it produced 29 real-G1 clips; more DR = worse dance fidelity (verified). Keep 1–3 s pushes and ±0.25 m anchor-pos obs noise; widen pushes to ±0.7–1.0 m/s xy only *after* tracking converges (likely).
- [ ] **No test-time action filtering, no Kalman on actions, no PD retune, no rescaling** — none of the successful G1 systems (BeyondMimic, RobotDancing) do it; smoothness comes from train-time action_rate penalty (verified). Clamp joint targets to 0.9× range and torques to effort limits.
- [ ] 50 Hz policy on Jetson + ~500 Hz low-level PD; ONNX runs <1 ms/step (verified BeyondMimic). Controls: L1+A standby, R1+A start, B damping e-stop; robot 192.168.123.11.
- [ ] **Sim2sim gate before hardware** every export: MuJoCo replay (unitree_rl_mjlab play → unitree_mujoco, or RoboJuDo g1_beyondmimic_with_ctrl), full-song length, watch global drift. (verified practice)
- [ ] Escalation kit if transfer is poor (verified OmniH2O values, apply only on evidence): PD-gain rand ±25%, control delay 20–60 ms (in mjlab: actuator delay 1 policy step = 20 ms, needs ≥ v1.5.x for #1035 perf fix), torque RFI 0.1× limit, torso body_mass ±10% if battery/backpack varies.
- [ ] Choreography risk triage: no hand/floor contact (finger collision geometry is wrong in sim — WBT #67, mjlab #496); ankle-heavy moves (toe stands, heel pivots) are highest risk (G1 ankle linkage gap, MOSAIC) — test isolated first. (verified)

---

## 6. Expected 4090 wall-clock

Throughput anchors: mjlab nightly ~219k env-steps/s @ 4096 envs on tracking (~0.45–0.7 s collection/iter, ~1–1.5 s/iter total); discussion #220 measured ~190k steps/s on an actual 4090 after fixing the mujoco_warp pin (11k steps/s before — **verify pin first and confirm ≥100k steps/s in the first 100 iters**). Per-iteration cost is independent of clip length; only iterations-to-convergence grow. (likely — GPU on nightly box unrecorded)

| Motion | Iterations to usable | Wall-clock |
|---|---|---|
| 28.8 s test | 2–6k | ~1–3 h |
| 44.3 s Thriller | 10–15k | ~4–8 h (overnight safe) |
| 2–3 min dance | 15–30k+ | ~8–16 h, budget 1–2 days first time |

All wall-clock figures **likely**; iteration counts **verified** (BeyondMimic ~10k long clips, RobotDancing ~15k per 3-min dance, defaults cap 30k). Benchmark s/iter early and extrapolate; eval checkpoints every 500 iters and stop at success plateau rather than running to max_iterations.

---

## 7. Open uncertainties to watch during training

1. **mjlab port fidelity** — diff your installed mjlab config against Section 1 (reward stds 0.3/0.4/1.0/3.14, adaptive sampling params, RSI ranges, z-only terminations 0.25/0.8, actuator table); adaptive sampling and push_robot are the most commonly dropped pieces in reimplementations. mjlab pins WBT commit f8e20c8 — later WBT fixes may not be ported. (likely)
2. **Sampling collapse on long clips** — sampling_top1_prob pinned near 1.0 for a sustained period = infeasible reference segment; fix the CSV (csv_to_npz `--line-range` to slice/debug), don't tune RL. The uniform-floor normalization bug (WBT #40) means easy bins may be under-rehearsed — check late-clip success via the frame-0 evaluate script, which exposes what adaptive-sampled training curves hide.
3. **NaN crashes** (MuJoCo Warp beta) — "normal expects all elements of std >= 0.0" mid-run; mitigate with nan_detection termination + resume from last 500-iter checkpoint.
4. **Wall-clock trap is environment config, not the task** — stale mujoco_warp, delay-term slowdown (#1035), pseudo_inertia VRAM blowup.
5. **Play "looks broken"** — fragmented out-of-order segments during play is adaptive resampling, not policy failure (WBT #54); use start-mode/no-terminations for demos.
6. **Global drift magnitude on hardware over 3 min** — expect ~0.5 m (RobotDancing); whether the global-anchor reward + G1 odometry keeps it stage-acceptable is unproven for your venue — measure in sim2sim, then hardware; knob = anchor weights 0.5→1.0. (assumption at your scale)
7. **kp/kd source of truth** — mjlab's derived gains vs WBT's hand-tuned gains differ; whichever you train with must be what motion_tracking_controller applies (read from ONNX metadata, verify against `export-scene g1` MJCF dump).
8. **W&B dependency** — evaluate script loads motion from the W&B artifact; local-file runs need the motion linked or a small patch. Confirm GreenNode egress or set `logger=tensorboard`.
9. **3π rad/s frames** — 3π≈9.4 rad/s is *within* motor velocity limits (20–37 rad/s); it's a retarget-quality flag, not an actuator limit. Still clean them (Section 2) — they poison tracking rewards and adaptive bins.
