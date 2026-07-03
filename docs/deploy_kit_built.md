# Deploy kit + sim exam — build summary (2026-07-03, deploy-kit worktree)

For main to integrate into PROJECT_STATE. Two deliverables, both verified locally.

## 1. Sim2sim exam gate — `pipeline/sim_exam.py`

Closed-loop exam of a trained policy in plain MuJoCo (unitree_mujoco
`scene_29dof.xml`), PD torque control at 50 Hz (dt 0.005 × decimation 4, matching
tracking_env_cfg). Three phases: nominal (survival, tracking error, ≤1.5 m
excursion), push (randomized 250 N / 0.1 s horizontal shoves, recovery = anchor
error < 0.25 m within 2 s, pass ≥ 0.8), repeatability (N seeded reruns with
±0.02 rad initial jitter, consecutive-clean counter). Termination mirrors
training's `bad_anchor_pos_z_only` (0.25 m) + `bad_anchor_ori` (0.8). EGL video
of the nominal run. Verdict JSON = `sim_exam/v1` contract, defined in
**docs/show_mode_contracts.md** (also defines `deploy_bundle/v1` for show-mode).

Policy interface: pluggable `PolicyAdapter`. Implemented: `WbtOnnxPolicy`
(whole_body_tracking exporter format — obs+time_step in, actions out, gains/
defaults/obs-layout parsed from onnx metadata, obs-history tiling supported) and
`StubPolicy` (reference replay for harness verification).

Verification: stub run on dance1_subject2_seg — joint mapping proven correct
(holds 0.075 rad tracking on a frozen pose), robot falls because blind PD replay
has no balance intelligence — the exam fails it for exactly the right reason.
Video path verified (EGL, 44 KB clip). `onnxruntime` installed into g1dance env.

## 2. Robot-day kit — `deploy/`

`lib.sh` (gates: `CONFIRMED_BY_HUMAN=alois` env + per-script flags + dry-run
default), `gen_config.py` (bundle builder, HARD-GATED: refuses without a passing
sim_exam verdict sha-matched to both policy AND motion; emits manifest +
damping-hold entrypoint that refuses to run until its launch line is verified on
PC2 — `LAUNCH_LINE_VERIFIED` marker), `01_pc2_install.sh`, `02_push_bundle.sh`
(integrity re-check), `10_gantry_test.sh` (most-gated: env + 3 flags + typed
"ROBOT IS SECURED" phrase; starts controller in DAMPING HOLD only),
`kill_now.sh` (instant abort; env var only — safety-positive, no dry-run),
`README.md` (interlock model). Plus **docs/ROBOT_DAY_RUNBOOK.md** — full
step-0-to-8 procedure with per-step abort criteria and the abort ladder.

Verification: shellcheck clean (`cd deploy && shellcheck -x ./*.sh`); interlocks
exercised — kill refuses without env var, gantry test refuses without
--estop-confirmed, gen_config refuses missing files / missing exam / FAIL exam
and accepts a sha-matched pass; 02_push integrity check + dry-run scp verified.
Test bundle removed after verification.

## Open interface questions for the TRAINING track

1. **mjlab export format**: sim_exam's `WbtOnnxPolicy` assumes the
   whole_body_tracking exporter graph (obs[1,D]+time_step[1,1] → actions + baked
   motion, metadata_props with joint_names/gains/obs-layout). Does mjlab's export
   match? If not, send one sample .onnx (or its format notes) and a
   `PolicyAdapter` subclass slots in — the exam loop is adapter-agnostic.
2. **Obs term coverage**: supported terms are command, motion_anchor_pos_b,
   motion_anchor_ori_b, base_lin_vel, base_ang_vel, joint_pos, joint_vel,
   actions (G1FlatEnvCfg and the WoStateEstimation variant both covered). Any
   custom terms in the final config → extend `ExamEnv._term_value`.
3. **Motion npz vs CSV retiming**: exam rebuilds the 50 Hz reference from the
   30 fps CSV (nlerp + ghost FK for the torso anchor). If training's csv_to_npz
   resamples differently (e.g. cubic), tracking-error metrics shift slightly —
   verdict thresholds may need one calibration pass against the first real
   policy.
4. **Controller launch line** (robot day, not training): pinned as a TODO gated
   by `LAUNCH_LINE_VERIFIED`; needs the motion_tracking_controller README read
   on PC2 (no internet on robot LAN; repo not cloned locally).

## Merge notes

New files only, except: none of ui/, none of PROJECT_STATE.md touched. New:
pipeline/sim_exam.py, docs/show_mode_contracts.md, docs/ROBOT_DAY_RUNBOOK.md,
docs/deploy_kit_built.md, deploy/{README.md,lib.sh,gen_config.py,01_pc2_install.sh,
02_push_bundle.sh,10_gantry_test.sh,kill_now.sh}. No dependency changes beyond
`onnxruntime` (+ shellcheck, conda) already installed into the g1dance env.
