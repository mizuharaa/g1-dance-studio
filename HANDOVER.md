# G1 Dance — Project Handover (2026-07-10)

**Written for the incoming developer.** This is the "read me first" for taking over. It supersedes
the 2026-07-06 handover. Read all three of:
- `CLAUDE.md` — operating rules (safety, measurement discipline, cloud handling).
- `PROJECT_STATE.md` — the running source of truth (mission, decision log, phase, resume protocol).
- `HANDOVER.md` (this file) — a narrative of everything done, what's in flight, the findings, the traps.

---

## 1. Mission

Make a Unitree **G1 humanoid dance the full Thriller, untethered, to real music**, driven by a
**one-button software app**, with a side-by-side *reference | simulation* video on an external
display, wirelessly, and the **robot always self-balancing** (entry → GO → dance → keep standing →
onboard balance takes back over). RL whole-body motion tracking via **BeyondMimic / mjlab**
(task `Mjlab-Tracking-Flat-Unitree-G1-Sim2Real`).

**Current headline:** the dance works on hardware (ran ~44 s cleanly). The blocker is a **sim2real
latency gap** that makes the robot drift and fall late in the dance. A **latency-robust retrain is
RUNNING RIGHT NOW on the GPU box** (see §3) — the single most important in-flight item.

---

## 2. Architecture (what talks to what)

- **App**: FastAPI (`ui/server.py`) + pywebview desktop wrapper (`ui/desktop.py`, port **8735**) +
  vanilla JS (`ui/static/app.js`). Pipeline stages: extract (GVHMR) → retarget → **train** → verify → export.
  - Headless option: `python3 ui/server.py --host 127.0.0.1 --port 8735` serves the same UI; the
    desktop window attaches to it. (Use when a GUI window can't launch, e.g. over SSH.)
- **Cloud training**: GreenNode GPU notebook (RTX 4090). **No API/CLI** — see §6. Training scripts
  live in `cloud/`; the app orchestrates them over SSH (`pipeline/cloud.py`, `pipeline/stages/cloud_motion.py`).
  - Transports: `ssh` (files via scp — the ONLY file-transfer path) and `jupyter` (command-only, no files).
- **Deploy runtime** (`pipeline/deploy_runtime.py`): `--mode ground-run-legodom` runs a leg-odometry
  policy, releases the onboard motion service, commands `rt/lowcmd` at 50 Hz, and **guarantees damping
  on any exit** (clean, SIGTERM, or crash). Modes: `read` (safe default), `move-to-default`, `run`,
  `stand-hold`, `ground-run`, `ground-run-odom`, `ground-run-legodom`. Key env knobs: `EXIT_MODE=stand`,
  `HANDOFF_HOLD_S`, `HANDOFF_OVERLAP_S`, `ENTRY_CATCH_S`, `START_POSE_MAX_DELTA_RAD` (entry guard, default OFF).
- **Show flow**: `pipeline/show_runner.py` → `tools/show_run.sh` → `deploy_runtime`. Music via
  `pipeline/show_audio.py` (tick0 + 4.0 s). Video via `tools/show_display.py` (side-by-side at tick0).
  The non-free show deploys the **selected dance's** bundle (`--policy/--meta/--motion-npz`).
- **Conda envs**: `tv` = robot runtime (unitree_sdk2py, onnxruntime, mujoco). `g1dance` = pipeline/tests.

---

## 3. ⚠️ IN FLIGHT RIGHT NOW — the latency-robust retrain

Running on the GPU box since 2026-07-10 ~09:57 ICT. **Do not lose track of this.**

- **Box**: `g1-retrain-latency`, id `nb-9c7ba766-f5bf-4e42-8091-7542b9372da6`.
  **SSH**: `ssh -i .secrets/greennode_rsa -p 59613 root@103.245.250.152` (also in `.secrets/cloud.json`).
- **Run**: tmux session `train` on the box; log `/workspace/notebook-data/train_lat.log`.
  run-name `train-thriller_lat80-2607`, task Sim2Real, 4096 envs, 5000 iters. **ETA ~1h35m** from launch.
- **What changed vs the previous policy** (`thriller_csv_ankle_penalty`, the one that fell):
  1. Latency domain-randomization widened **0–20 ms → 0–80 ms** (`cloud/sim2real_task.py`:
     `CMD_DELAY_MAX_LAG 4→16`, `OBS_DELAY_MAX_LAG 1→4`). **This is the fix for the fall.**
  2. Root-position reward weight **1.0** (drift fix; in `DEFAULT_PARAMS.extra_train_args`).

**When it finishes** (all runnable from the laptop over SSH; templates in `pipeline/stages/cloud_motion.py`):
1. **Export** ONNX: `cloud/export_policy.py <ckpt> <npz> <exports>`.
   Checkpoint: `/workspace/notebook-data/logs/rsl_rl/g1_tracking/*train-thriller_lat80*/model_4999.pt`.
2. **Verify** `cloud/sim_gap_check.py` — now **gates survival at 40 ms + push** (was 20 ms) + 60/80 ms
   stress lines. If it can't survive hardware-range latency it must NOT pass.
3. **Held-out exams** `cloud/heldout_eval.py` ×3 (seeds 90001/90011/90021, 256 envs).
4. Pull → `pipeline/mjlab_verify.py` sign → attach + `record_sim_run_from_verdict` → promote in Shows page.
5. **DELETE the box** (§6 — Alois is emphatic).

A background poller (scratchpad `wait_train.sh`) was watching for completion.

---

## 4. THE headline finding — sim2real LATENCY gap (2026-07-09 fall diagnosis)

The robot danced ~44 s, **drifted side/back/forward, then fell** (on the ground; fall-detector damped
+ handed to onboard; tether caught it). Diagnosed **decisively** (4 independent signals):

1. **Sim `gap_check`**: 0 falls at nominal/noise/10 ms/20 ms delay; **many falls at 40 ms**.
2. **Hardware IMU**: normal ≤16° tilt through 40 s, then climbs; **right knee buckles** to 135° at
   ~45 s; torso drops 0.24 m below choreographed height → abort. Choreography wanted it *upright*
   there — so it genuinely deviated (not a false trigger).
3. **Telemetry cross-correlation** (`tools/measure_latency_from_telemetry.py`): effective
   command→response latency **80 ms median on legs**, 60–100 ms even on light arm joints (≈no
   mechanical lag) → **pure sensorimotor latency ≥40 ms**.
4. **Comms ruled out**: wired, ping RTT **0.16 ms**, DDS staleness ~2 ms baseline.

**Root cause, pinned to config**: `cloud/sim2real_task.py` only randomized latency to **20 ms**
(40 ms was "eval-only"), AND `sim_gap_check` only *gated* survival at ≤20 ms — so the policy that fell
passed verification. Both fixed (§3). Full evidence + scripts committed under
`data/telemetry/latency_diag_20260709/` (DIAGNOSIS.md + raw outputs).

---

## 5. Everything else done this session

- **Sim/ref video desync FIXED** (`73c74ad`). The show played the old **v3e** side-by-side, whose sim
  panel is a *different* Thriller take (2589-frame lineage) than the deployed policy (2789-frame CSV) —
  matched neither robot nor reference. New local kinematic renderer `tools/render_deploy_sim.py`
  (mujoco EGL, name-based joint map, **no GPU**) renders the sim panel from the *actual deploy motion*.
  New composite `data/previews/thriller_side_by_side_csv.mp4`; `FREE_SHOW_VIDEO` points to it. Verify
  frames: `data/telemetry/side_by_side_csv_verify/`.
- **Ankle-penalty Thriller** trained + verified + promoted earlier (the policy that fell). Dance record
  `data/dances/20260708-71711415/`; policy `data/policies/thriller_csv_ankle_penalty/`.
- **Show deploys the SELECTED dance's policy** (was hardcoded default) — `_dance_policy_args()`.
- **Stand-exit handback** wired: `EXIT_MODE=stand` holds final pose `HANDOFF_HOLD_S=3.0` then overlaps
  `HANDOFF_OVERLAP_S=5.0` while onboard 'ai' restores — operator engages onboard stand in that ~8 s
  window. **STILL UNVALIDATED** (the run aborted at 44 s before the clean finish).
- **App STOP button** (`#runStopBtn` → `POST /api/shows/runs/current/stop` → SIGTERM → damp).
- **Entry-fall guard** (`START_POSE_MAX_DELTA_RAD`, default OFF).
- **Training-env cascade permanently fixed**: mjlab installs into an **isolated venv**
  (no `--system-site-packages`) + GLVND loaders + `LD_LIBRARY_PATH=/opt/conda/lib` on every cloud
  script. Killed a long tail of libstdc++/matplotlib/GL/scipy failures.

---

## 6. GreenNode GPU box — operational reality (READ THIS)

- **No API, no CLI.** Created/deleted/connected **only through the web console**, driven by a
  headed-Chrome automation: `tools/pilot.py` (Playwright), file-driven via `tools/pilot_cmd.sh`.
  Launch: `nohup python3 tools/pilot.py &`; drive: `bash tools/pilot_cmd.sh '{"action":...}'`.
  Console login in `.secrets/greennode.cred` (root user, VNG postpaid — "0 credits" is normal).
- **SSH keys must be RSA.** The console's SSH-key import **rejects ed25519** ("Invalid Public Key").
  Working keypair `.secrets/greennode_rsa(.pub)`, registered as `g1dance-laptop`. `cloud.json`
  `ssh.key_path` points to it. (Old `.secrets/greennode_ssh_key` is ed25519 → unusable there.)
- **At create you MUST**: pick `GPU-CODE-RTX4090` → `aiplatform-standard-16x64-1rtx4090`; attach volume
  `g1dance-data` (folder `notebook-data` → `/workspace/notebook-data`); select the RSA key; **add TCP
  port 22** (SSH is otherwise unreachable — creation-time only); image PyTorch 2.5.1/CUDA 12.4. The
  custom Angular dropdowns are flaky to automate (instance type silently resets to CPU; verify the
  Summary shows `GPU: 1 × RTX4090` before Create). A human clicking the form is more reliable than the
  pilot for the create step; the pilot is solid for login/navigation/delete-hunting.
- **Provisioning** (fresh volume = full path; `docs/BOX_RECREATE_RUNBOOK.md` Part 4): push `cloud/*` +
  `.wandb_key` + motion; run `cloud/00_bootstrap.sh` then `cloud/20_training.sh mjlab`; confirm
  `reports/training_stack.json` → `mjlab_ready`.
- **DELETE the box when done** (billing runs creation→deletion). If deletion fails, keep it busy —
  never idle. New SSH host/port each recreate → update `.secrets/cloud.json`.

---

## 7. Known open issues / next steps

1. **Validate the latency retrain** (once §3 finishes + passes gap_check) on hardware — the main goal.
   Bring tether slack (drift), watch for the ~45 s buckle recurring.
2. **Exit-fix still unvalidated** — the stand-handback (§5) never got a clean end-of-dance test.
3. **Video is still colourful static on the show display** — VLC "Too high level of recursion" filter
   bug on this machine. The *content* is correct (verified frames); the **player** is broken. Fix by
   switching `tools/show_display.py` off VLC to `mpv`/`ffplay` (VLC is the only one currently installed).
4. **Drift** — addressed by root-pos weight 1.0 in the retrain; confirm on hardware.

---

## 8. Robot specifics + safety (NON-NEGOTIABLE)

- This G1 (~35 kg) has **NO torque-cut hardware e-stop**. Only hard stops: remote **B-damp**, **power
  switch**, **app STOP button**. During a show the onboard service is *released*, so the phone/app can't
  re-pair mid-show — the damping remote in hand is the safety net.
- **Never** command motion without: human present, tether rigged to catch, **damping remote in hand**,
  and the motion MuJoCo-verified. All motion modes require `--i-will-watch-the-robot` AND env
  `CONFIRMED_BY_HUMAN=alois`. Deploy always needs explicit human confirmation (the app RUN SHOW phrase).
- **Never modify `~/robot/`** (working teleop setup — read-only reference).
- Robot iface `enp0s31f6`; onboard PCs `192.168.123.161` / `192.168.123.164` (wired subnet 192.168.123.x).
- **Stop a run by signalling the PYTHON pid, not the bash wrapper** (else the child orphans and holds
  the robot energized — happened twice): `pgrep -f "python.*deploy_runtime"`. The app STOP does this via
  process-group SIGTERM.
- **Motion-service gotcha**: releasing it for low-level control freezes `rt/odommodestate` and can
  strand the remote — the runtime auto-restores `SelectMode("ai")` on exit. If the remote won't pair,
  run `SelectMode("ai")` from the laptop or reboot the robot.
- **Thermal**: read `motor_state[i].temperature`; warn ~80 °C, fault ~90 °C. The monitor must drain DDS
  to the LATEST msg (a stale-backlog bug once let a motor hit 80 °C blind — fixed).
- **Training gains == deploy gains** (ankle kp 29, knee 99, hip 40 — verified; NOT a gain bug). Robot
  model + gains on box: `repos/mjlab/.../unitree_g1/g1_constants.py`.
- **Measurement discipline**: no finding is "decisive" without an independent cross-check/replication;
  commit every measurement script AND its raw output (see `data/telemetry/`).

---

## 9. Access the new developer needs (NOT in git — get from Alois)

`.secrets/` is gitignored and holds everything sensitive — obtain directly from Alois:
- `.secrets/cloud.json` — box SSH host/port/key.  `.secrets/greennode_rsa(.pub)` — RSA SSH key.
- `.secrets/greennode.cred` — GreenNode console login (for `tools/pilot.py`).
- `.secrets/wandb.key` — Weights & Biases key.  `.secrets/pilot/` — Chrome-pilot runtime state.

**Security note:** a GitHub PAT for this repo was exposed in a chat transcript earlier in the project —
**rotate it** if not already done. Never commit tokens; use a fresh PAT or an SSH deploy key.

---

## 10. Key files map

| Area | Files |
|---|---|
| Source of truth | `PROJECT_STATE.md`, `CLAUDE.md`, `logs/jobs.md` |
| App | `ui/server.py`, `ui/desktop.py`, `ui/static/app.js` |
| Deploy runtime (robot) | `pipeline/deploy_runtime.py`, `tools/show_run.sh`, `pipeline/show_runner.py`, `pipeline/leg_odometry.py` |
| Show media | `pipeline/show_audio.py`, `tools/show_display.py`, `tools/make_side_by_side.py`, `tools/render_deploy_sim.py` |
| Cloud training | `pipeline/stages/cloud_motion.py`, `pipeline/cloud.py`, `cloud/*.sh`, `cloud/sim2real_task.py`, `cloud/train_sim2real.py`, `cloud/sim_gap_check.py`, `cloud/heldout_eval.py`, `cloud/export_policy.py` |
| GreenNode automation | `tools/pilot.py`, `tools/pilot_cmd.sh`, `docs/BOX_RECREATE_RUNBOOK.md`, `docs/GREENNODE_SETUP.md` |
| Fall diagnosis (this session) | `data/telemetry/latency_diag_20260709/`, `tools/measure_latency_from_telemetry.py` |
| Operator runbooks | `docs/ONBOARD_RUN_CHECKLIST.md`, `docs/GROUND_TETHERED_RUNBOOK.md` |

---

## 11. How to resume in a fresh session

Start in `~/g1-dance` and: read this file → `PROJECT_STATE.md` decision log (newest first) → check the
in-flight retrain (§3: SSH to the box, `tmux ls`, tail `train_lat.log`). If training finished, run the
export→verify→exam→promote chain (§3). Robot motion only with a human present + damping remote in hand.
Keep the GPU box busy or delete it — never idle. Measurement discipline per `CLAUDE.md`.
