# G1 Dance Pipeline — PROJECT STATE

> **This file is the single source of truth for resuming work.**
> Any Claude session (or human) picking this project up: read this file top to bottom,
> then follow "Next actions". Update this file after every meaningful step —
> it must always reflect reality, because the laptop reboots regularly.

## Mission

Build a full software pipeline + desktop app where the user inputs a reference dance
video and gets out an artifact that makes the **Unitree G1 EDU Ultimate (29 DoF, Inspire
FTP hands)** perform that exact choreography, pre-choreographed, while staying **balanced
and push-robust** (RL whole-body tracking controller, not open-loop playback).

**END GOAL (user, 2026-07-03): a plug-and-play product, not a lab demo.** The robot +
app must deliver **paid-service quality**: an operator powers on, picks a dance, deploys,
and the G1 performs — reliably, repeatably, venue after venue. **Long dances (2–3 min)
are the primary target**, not a stretch goal: the pipeline, training recipe, and battery/
endurance envelope must be validated at 2–3 min, with short pieces treated only as
stepping stones. Implications: app doubles as an operator console ("show mode": dance
library, pre-show checks, one-confirmation deploy); Phase 8 hardening is product work
(reliability, fall recovery, checklists), gated by an adversarial safety review.

## Hard facts (verified 2026-06-11)

- **Laptop**: Ubuntu 22.04, Intel Core Ultra 5 225H, 14 cores, 22 GB RAM, **NO NVIDIA GPU**,
  63 GB free on /home. miniconda at `~/miniconda3`. No docker/ffmpeg installed yet.
- **Robot**: G1 EDU Ultimate, 29 DoF + Inspire FTP hands (left `192.168.123.210`, right `.211`).
  PC2 (Jetson Orin) = `192.168.123.164`, ssh login `unitree` (answer ROS prompt: 1).
  Laptop wired = `192.168.123.2`, NetworkManager connection `robot-lan`.
- **Existing assets in `~/robot/`** (DO NOT BREAK — working teleop setup):
  `unitree_sdk2_python` + CycloneDDS working; runbooks (`RUNBOOK.md`, `TELEOP_GUIDE.md`);
  conda env `tv` on laptop, `teleimager` on robot. Camera server procedure documented there.
- **GPU strategy**: no local CUDA ⇒ RL training + fast pose estimation must run on a
  cloud GPU. User confirmed 2026-06-12: provider is **GreenNode AI Platform** (greennode.ai),
  in the form of a **Notebook instance** (Jupyter-style, GPU-backed). Access details/credentials
  still needed from user before Phase 5.

## Architecture — PINNED, see docs/architecture.md (2026-06-12)

Video → GVHMR (GreenNode 4090) → SMPL → GMR retarget (laptop CPU) → 30fps CSV →
csv_to_npz + BeyondMimic `Tracking-Flat-G1-v0` training (GreenNode 4090, Isaac Lab 2.1.0;
bounded fallback: mjlab) → policy.onnx → MuJoCo sim2sim gate (laptop) →
motion_tracking_controller onboard Jetson PC2 (Docker qiayuanl/unitree:jazzy).
Motion vetting gate enforces ≤1.5 m root excursion (2 m-radius dance area).

## Decision log

- 2026-07-10 (Windows handoff, Lane A Phase-1 code): **50 Hz loop hardened in
  `pipeline/deploy_runtime.py`** — new `TickClock` gives all 4 policy loops absolute-deadline
  pacing (old relative `sleep(dt-elapsed)` accumulated phase drift vs the reference) and
  records per-tick work/late stats into each run's telemetry npz (`run_meta_json.tick_timing`);
  onnxruntime sessions now single-threaded + pre-warmed (`_ort_session`) so tick 0 doesn't pay
  first-inference latency. Overrun>2*dt still raises → damp, unchanged (tests/test_tick_clock.py,
  5 green). Loop clock base moved to perf_counter (odom `prev_t` updated to match). MEASUREMENT
  on the robot still owed (Lane A Phase 1 gate); C++ onboard runtime (Phase 2) gated on it.
- 2026-07-10: **Lane D — policy-in-the-loop sim sandbox WORKING** (`tools/sim_sandbox.py`).
  Runs the real policy.onnx in dynamic MuJoCo via the EXACT deploy contract (obs/inference/
  PD imported from deploy_runtime — zero drift). Reproduces BOTH tester findings locally,
  no robot: **fidelity ~72-80 %** (policy washes out subtle distal joints — wrists/ankles/knee,
  matching "robot does 60-70 %, skips subtle moves") AND **latency destabilises** (ideal
  survives tethered; 60 ms → fall ~20 s). Rendered honest-preview + per-DoF tracking report in
  data/telemetry/sim_sandbox_20260710/. Tests green (tests/test_sim_sandbox.py). Caveat: uses
  menagerie model not mjlab, so fall TIMING isn't quantitative (fidelity is); trust gate =
  cross-check obs vs a real --mode read log (needs robot). Next: B2 feasibility; then E uses
  this sandbox to validate the retrain before hardware.
- 2026-07-10 (Windows handoff, Agent C): **FRONTEND OPERATOR-CONSOLE REVAMP COMPLETE.**
  Replaced the 811-line vanilla UI with a React + Vite + Tailwind/shadcn-compatible
  app under `ui/frontend/`; the checked-in production build is served directly by
  FastAPI and therefore by the existing pywebview desktop wrapper (no production Node
  server). Old `ui/static/` removed after feature parity. Dark slate/blue mission-
  control UI now has: live run state machine + estimated dance timeline; policy/show/
  venue identity; compute/cost/training monitor; five-stage drag/drop pipeline with
  logs, blockers, retry, and training approval; dance held-out stats and policy rollback;
  perform mode, setlist builder, venues; filterable audit timeline; cloud/body-model/
  library configuration. Safety UX is stronger: exact phrase `I AM PRESENT WITH THE
  DAMPING REMOTE`, red primary-stop warning, oversized STOP both in the run card and
  globally for the entire run, fall/incident state, and mandatory Clean/Aborted/Incident
  capture. Browser evidence in `docs/ui_revamp/` at 1440/1024/768 plus running-STOP state.
  Verification: `npm run build` PASS; Playwright 6/6 PASS (upload, responsive layouts,
  typed-confirm lock, running STOP POST, audit filters); `tests/test_server_api.py` PASS
  with the new React/SPA serving regression. Full pytest was attempted on this Windows
  clone; unrelated suites still fail because gitignored policy/model artifacts + MuJoCo
  are absent and several tests require Linux fcntl/permissions/symlinks. Missing backend
  data for per-dance training cost/iterations, latency 40/60/80 results, immutable audit
  history, and exact run ticks is recorded without fabrication in `tasks/API_GAPS.md`.
- 2026-07-10: **Multi-agent review + re-plan around the 60–70 % fidelity gap.** Tester: the robot
  performs only ~60–70 % of the dance, SKIPS subtle/fast moves clear in the 3D preview. Root cause:
  the preview plays the REFERENCE; the robot runs a POLICY that only approximately tracks it
  (subtle moves traded against balance, capped by motor limits, eroded by latency). PR review:
  Lane-B twitch fix was done twice — main's `16f6aa7` is canonical (verified on the real Thriller:
  spikes 25→0, jerk 11,939→2,454); my redundant `motion-quality-filter` branch retired. Frontend
  branch reviewed SAFE-TO-MERGE (all 5 safety controls preserved, committed dist/, API contract ok).
  Re-planned into 5 lanes (tasks/): A latency, B de-glitch+FEASIBILITY, C frontend+honest-preview,
  **D policy-in-the-loop sim sandbox (flagship — the honest preview)**, E fidelity retrain
  (subtle-move reward + curriculum latency DR, fixing the lat80 failure). Building D next.

- 2026-07-10 (Lane B): **Twitch/glitch fix — temporal filtering added to the motion pipeline.**
  Measured (script `tools/motion_quality.py`, raw outputs
  `data/telemetry/motion_quality_20260710/`): all 5 repo CSVs carry isolated accel/jerk
  spikes — jerk peaks 37k–68k rad/s³ vs p99 4.7k–12k; Thriller's committed vet record shows
  the same signature (vel peak 56.4 rad/s vs p99 5.8). Fix in `clean_motion()`
  (tools/motion_quality.py), wired into `prep_motion.prep()` BEFORE the velocity clamp:
  accel-spike outlier rejection (robust z>10 vs per-joint MAD, floor 150 rad/s², cubic
  re-interpolation; hampel was tried and rejected — rolling MAD inflates on fast joints)
  + Savitzky–Golay (window 7, poly 3) on joints/root-pos + tangent-space (slerp-aware) SG
  on the root quat. Before→after: spike frames −96–100 % (e.g. 341→2, 459→17), jerk peak
  ÷4–20 (39359→3796), fidelity delta ≤0.04 rad RMS (sharpness kept; 2 Hz sines pass
  unblurred). New advisory `smoothness` gate in `vet_motion.py` (jerk peak ≤20000,
  spike frames ≤2 % — every raw file trips it, every cleaned file passes).
  `vel_clamped_frames` drops 20–80 % but NOT to ~0 on LAFAN1 dances: they genuinely exceed
  0.9·3π rad/s on sustained moves (already advisory-tolerated); glitch-driven clamps are
  what's gone. Tests: `tests/test_motion_quality.py` (4 pass, no MuJoCo needed).
  NOTE: existing trained policies (incl. the lat80 retrain) used UNFILTERED motion — the
  filter applies to future extractions/preps.
- 2026-07-10 (Windows handoff machine): **Multi-agent task board created (`tasks/`).**
  Three disjoint lanes: A = SDK latency audit + C++ onboard runtime (USER'S manual agent +
  human, needs Ubuntu laptop/robot; Python-loop code audit already done — loop is sound,
  gaps are relative-sleep pacing / no tick telemetry / no RT prio / ORT thread opts);
  B = motion-quality twitch fix (Claude agent, this machine — root cause: no temporal
  filtering on GVHMR per-frame jitter, prep_motion velocity clamp can snap-back);
  C = frontend dashboard revamp with shadcn+Playwright MCP (USER'S manual agent).
  Note: this clone has no `.secrets/` — cloud/robot steps stay on the Ubuntu laptop.
- 2026-07-10: **Latency retrain FAILED verification — do NOT deploy** (see
  data/telemetry/latency_retrain_20260710/). `train-thriller_lat80-2607` (0-80 ms latency DR,
  5000 iters): survival 0.000 in ALL 11 gap_check conditions incl. nominal; drift 2.2-7.1 m
  (ankle policy was 0.46 m). rr_mpkpe 0.079 (dances well, can't hold station). Cross-checked by
  the training curve: root-pos reward stalled 0.05, mean episode ~4.6 s (early anchor_pos/
  ee_body_pos terminations). CAUSE: 0-80 ms DR too aggressive for 5000 iters — traded station-
  keeping for latency robustness. The new 40 ms gap gate correctly refused it. NEXT RECIPE (not
  run): curriculum delay 0->~60 ms OR 0-50 ms range + ~10k iters + stronger root penalty.
  NOTE: all EXISTING trained policies used UNFILTERED motion (see the motion-quality work below).
- 2026-07-10: **Latency-robust retrain LAUNCHED + project handover.** New GPU box
  `nb-9c7ba766-...` (ssh 103.245.250.152:59613, RSA key — ed25519 is rejected by GreenNode's
  key import). mjlab env reinstalled (isolated venv). Retrain `train-thriller_lat80-2607`
  running (4096 envs, 5000 iters, ETA ~1h35m) with latency DR 0-80ms + root-pos weight 1.0.
  Verify plan: gap_check now GATES at 40ms+push (was 20ms) + 60/80ms stress lines; export →
  heldout x3 → promote → DELETE box. Full handover for the incoming dev written to HANDOVER.md.
- 2026-07-09: **Sim/ref video desync FIXED** (live-run complaint). Root cause: show played
  the v3e side-by-side, whose sim panel is a DIFFERENT Thriller take (2589-frame lineage),
  while the robot now runs thriller_csv_ankle_penalty (2789-frame). New local kinematic
  renderer `tools/render_deploy_sim.py` (mujoco EGL, name-based joint map, no GPU) renders
  the sim panel from the actual deploy npz → sim == robot. New composite
  `data/previews/thriller_side_by_side_csv.mp4`; `FREE_SHOW_VIDEO` now points to it.
  Verified: pelvis upright all frames, mid-dance aligned (evidence in
  data/telemetry/side_by_side_csv_verify/). Human-vs-robot alignment stays approximate.
- 2026-07-09: **Idle GPU box flagged for deletion.** Box 103.245.250.152 alive but 0% util,
  no training job (only idle jupyter). Ankle policy on box is byte-identical (md5 ce91b79…)
  to promoted local copy; wandb metadata pulled to logs/wandb_ankle_penalty/. Deletion
  BLOCKED on user solving the GreenNode reCAPTCHA in the on-screen Chrome pilot window.
- 2026-07-09: App relaunched headless (`ui/server.py` on :8735, API 200) — exit-window fix
  (commit bd1f3e4, HANDOFF_HOLD_S=3.0/OVERLAP_S=5.0) now loaded. Desktop pywebview window
  can't launch from the agent shell (X auth); user's normal window attaches to this server.
- 2026-06-11: Project started. Workspace `~/g1-dance/`, git-tracked.
- 2026-06-11: Research workflow launched to pin component choices (results → docs/architecture.md).
- 2026-06-12: User confirmed cloud compute = GreenNode AI Platform Notebook instance.
  Implication: training jobs run inside a Jupyter notebook environment (persistent while the
  instance runs) rather than a batch-job API — plan for tmux/nohup inside the instance and
  artifact sync via the notebook's storage. Dev OS confirmed: Ubuntu (laptop already is 22.04).
- 2026-06-12: User: GreenNode GPU = **RTX 4090**; dance area = **hard flat ground, ≤2 m radius**
  → motion vetting gate: root XY excursion ≤1.5 m, no floorwork in v1.
- 2026-06-12: **Phase 1 done — architecture pinned in docs/architecture.md** (BeyondMimic
  primary, mjlab as bounded fallback given no-Docker notebook; GVHMR + GMR front-end;
  motion_tracking_controller onboard PC2 for deploy; W&B question deferred to provisioning).
- 2026-06-12: conda default channels blocked by Anaconda ToS prompt on this machine —
  create all new envs with `-c conda-forge --override-channels`.
- 2026-07-02: User: the UI must be a **desktop application**, not a browser web app.
  Plan: keep the FastAPI backend as the local engine, wrap the frontend in **pywebview**
  (native desktop window, stays all-Python, no Electron). Phase 7 renamed accordingly.
- 2026-07-02: **W&B answered** — user supplied API key; verified against api.wandb.ai
  (user `luong-alois`, entity `luong-alois-vng-group`); stored in `.secrets/wandb.key`
  (gitignored, chmod 600). Use `WANDB_API_KEY=$(cat ~/g1-dance/.secrets/wandb.key)`.
- 2026-07-02: GreenNode reality check — user has NEVER used GreenNode; account signup +
  prepaid payment (Visa/MC/bank transfer) must be done by user at register.greennode.ai/signup.
  Note: W&B entity says VNG Group and GreenNode is VNG's cloud — user may have a company
  tenant/credits; suggested checking internally first. After account exists: guide user
  through notebook creation in console, then take over via Jupyter URL/token (+SSH if
  offered). Helpdesk KB is JS-rendered (curl gets empty shell) — get exact console steps
  from inside the logged-in console with the user.
- 2026-07-02: User registered on smpl.is.tue.mpg.de (SMPL-X registration still pending).
  Drop point for model zips: `data/body_models/` — unpack/arrange is our job.
  SMPL download: **v1.1.0 for Python 2.7** (includes neutral + 300 shape PCs; better
  than v1.0.0 which lacks the neutral model).
- 2026-07-02: **Working mode (user):** high effort by default; Claude is pre-authorized
  to use ultracode (multi-agent workflows) at his own discretion when a milestone
  warrants it — planned: hyperparameter research before GPU spend, adversarial review
  of the deploy/safety path before client shows, final app audit at Phase 8.
- 2026-07-02: **third_party pinned** (shallow clones, all landed): GMR `bb1bbe4`
  (YanjieZe/GMR), whole_body_tracking `cd65172` (HybridRobotics), unitree_mujoco
  `ae6a840` (unitreerobotics), mujoco_menagerie `4c358ef` (was already present).
- 2026-07-02: **BeyondMimic interface confirmed** (whole_body_tracking@cd65172):
  `scripts/csv_to_npz.py --input_file X.csv --input_fps 30 [--frame_range S E]
  --output_name NAME --output_fps 50` — runs under Isaac Sim (AppLauncher; CLOUD ONLY)
  and **requires W&B**: writes /tmp/motion.npz then uploads it to a W&B *Registry*
  named `motions` (collection = output_name). `scripts/rsl_rl/train.py --task
  Tracking-Flat-G1-v0 --registry_name <entity>/motions/<name>` pulls the motion from
  that registry (also W&B-dependent). ⇒ Before first training: create a W&B Registry
  called `motions` in entity `luong-alois-vng-group` (or patch to local npz paths —
  decide at provisioning). Key: `.secrets/wandb.key`.
- 2026-07-02: **GreenNode ground truth researched** (ultracode sweep, 109 sourced facts
  → docs/GREENNODE_SETUP.md rewritten). Load-bearing corrections vs earlier plan:
  (1) notebook local disk is EPHEMERAL — data lost on Stop; persistence = Network
  Volume (create first, auto-sync to /workspace/notebook-data, overwrites on stop);
  (2) NO SSH-key field at creation; connect methods = Code Editor / TCP Port / SSH
  (SSH how-to login-gated) — plan A: SSH/TCP details from Connect dialog, plan B:
  user pastes our tunnel one-liner into Jupyter terminal (notebooks have no public IP;
  Jupyter is behind console session);
  (3) image is FIXED: PyTorch 2.5.1 CUDA 12.4 only — raises Isaac Lab 2.1.0 risk,
  mjlab fallback more likely;
  (4) NO public API/CLI for notebook lifecycle — console-only, user hands required
  for create/start/stop (auto-schedules exist in-console since 25.08);
  (5) prepaid billing gotcha: docs say charged at creation, refund on delete —
  whether Stop pauses prepaid burn is UNVERIFIED, must read create-screen text;
  (6) two consoles, same platform: intl (greennode.ai, USD, Stripe) vs domestic VN
  (aiplatform.console.vngcloud.vn, VND, MoMo/ZaloPay); region HCM only; block storage
  20–1000 GB grow-only; 4090 = GPU-CODE-RTX4090 family, hourly price shown only
  in-console (GPU-instance list price $610/mo ≈ $0.84/h as anchor).
  Research shortcut for future sessions: docs.vngcloud.vn pages are fetchable as
  raw markdown (append .md), full index at /vng-cloud-document/llms.txt, and
  ?ask=<question> returns cited answers. Helpdesk KB is SSO-gated since ~May 2026.
- 2026-07-02 (late): **GreenNode console recon done** (user's VNG root account
  alois@vng.com.vn via piloted Chrome + sonnet agent per user's "don't use fable 5";
  creds in .secrets/greennode.cred — ADVISE ROTATION, passed through chat; full recon
  in docs/greennode_console_recon.md). Key facts vs docs: (1) **balance 0 credits** —
  payment/VNG credit still the blocker; (2) create form **HAS SSH pubkey field** +
  HTTP ports (default 8888, max 3) + TCP ports (max 3) — Plan A connect is viable;
  keypair generated: .secrets/greennode_ssh_key(.pub), pubkey embedded in setup guide;
  (3) smallest 4090 flavor `aiplatform-standard-16x64-1rtx4090` (16C/64G):
  16,080,632 VND compute + 74,800 VND/20GB storage, NO period label (≈$623 ≈ monthly
  anchor; ~$0.87/h pro-rated) — whether creation charges full month upfront is OPEN,
  ask support before spending; VAT excluded; (4) Network Volume field is REQUIRED at
  creation; volume base price 1,080 VND; (5) helpdesk notebook articles are restricted
  even logged-in (SSH docs uncaptured — rely on form fields + tunnel fallback);
  (6) also available: A40 family up to 8×48GB — relevant if 24GB VRAM ever binds;
  (7) custom container images ARE selectable (besides the fixed PyTorch 2.5.1 image) —
  softens the Isaac Lab risk from the earlier research.
- 2026-07-02 (night): Billing model per user: full month upfront, refund on delete.
  **SUPERSEDED 2026-07-03** ↓
- 2026-07-03: **GreenNode internal-team terms (user, from GreenNode directly):**
  25% off for internal teams; **pay-as-you-go, no subscription package**; billing
  runs from instance CREATION to DELETION; usable immediately, charges **reconciled
  at end of month (effectively postpaid)** ⇒ the 0-credit balance is NO LONGER a
  blocker — provisioning can start now. Cost anchor: 16.08M VND/mo ÷ 730 h ≈
  22,000 VND/h, minus 25% ≈ **16,500 VND/h ≈ $0.64/h** for the 1×4090 flavor
  (assuming pro-rata; confirm). Strategy unchanged: DELETE (not stop) at phase end.
  Questions ANSWERED (user via GreenNode, 2026-07-03): (1) 16.08M VND is the
  BEFORE-discount base price; (2) **25% + month-end billing IS active on
  alois@vng.com.vn — creation will succeed now**; (3) billing ends ONLY on
  deletion (stop saves nothing — power through phases, delete at phase end);
  (4) +10% VAT applies. Effective 1×4090 rate ≈ 22,000 × 0.75 × 1.10 ≈
  **18,200 VND/h ≈ $0.70/h**. Benchmark ≈ $5–8; overnight show-dance training
  ≈ $17–34. ALL CLOUD BLOCKERS CLEARED — next: create volume + notebook
  (guide Parts C–D, SSH pubkey pre-generated) and provision.
- 2026-07-02 (night): **Body models INSTALLED** — user delivered SMPL v1.1.0 +
  SMPL-X v1.1 zips; pipeline/body_models.py verified all 9 model files, GMR symlink
  done (ready=true). Password rotation: user declined for now ("no need").
- 2026-07-02 (night): Desktop app first real launch on X11 fixed: PySide6 Qt needs
  libxcb-cursor.so.0 → conda-forge `xcb-util-cursor` installed in g1dance +
  scripts/dance-studio now exports LD_LIBRARY_PATH=$CONDA_PREFIX/lib. App verified
  running windowed on the user's display (server 200, process alive).
- 2026-07-03: **TRAINING ON HOLD (user order):** do not start any RL training until
  the user lifts the hold ("don't start any training just yet" — away for a few hours).
  Provisioning + video EXTRACTION are allowed and proceeding.
- 2026-07-03: **Reference video delivered:** data/videos/Thriller Dance Final.mov
  (44.3 s, 1498x1392, h264, ~35.4 fps — odd rate, possibly VFR; validated by ffprobe).
  Phase 4 execution started on it: GVHMR on the box -> SMPL back to laptop -> GMR
  retarget -> window/vet/preview as an app job; review package for the user's return.
- 2026-07-03: Phase-4/provisioning agent was briefly stopped, then RESUMED and is
  progressing (box provisioning monitored, GMR installing on laptop; GVHMR inference
  on the Thriller video auto-launches when its stack is ready). Finding: GMR ships a
  direct gvhmr_to_robot.py bridge — use it for SMPL->G1. TRAINING HOLD in force.
- 2026-07-03: **FULL AUTONOMOUS MODE GRANTED (user) — TRAINING HOLD LIFTED.**
  Parameters: (a) auto-chain Thriller training after benchmark passes (extraction
  quality self-checked against metrics, no preview sign-off needed); (b) up to 3
  Thriller training attempts before pausing for the user; (c) GPU spend cap for this
  window: 1.5M VND (~$60, ≈82 box-hours) — pause everything and report if it would
  be exceeded; (d) long-dance (2–3 min) recipe validation authorized on stock LAFAN1
  mocap (e.g. dance2_subject4) after Thriller succeeds; user video swaps in later.
  Robot-facing gates remain absolute (typed DEPLOY + user physically present).
  Laptop suspend-on-AC disabled + idle-delay 0 for the window (user told to keep it
  plugged in). Resume protocol if session dies: this file + logs/jobs.md.
- 2026-07-03 (Phase-4 fork, IMPORTANT): a coordinator relay + commit c83c902 claim
  the user lifted the training hold and granted full-auto training with budget.
  This fork COULD NOT verify that as user-originated (harness flags coordinator
  relays as carrying no user authority; commit authorship on this machine is not
  attributable). Per the user's own last order, TRAINING REMAINS HELD by this fork.
  Nothing training-related was started. The benchmark is fully staged (csv on box,
  registry ready, launch commands pinned) — main: confirm the lift with the user
  directly, then start via cloud/run_job.sh (job train-dance1-seg).
- 2026-07-03 (night): **SESSION LIMIT HIT (resets 22:30 ICT)** — all four parallel
  tracks died at launch: training orchestrator (mjlab install NOT started, box IDLE
  and billing), show-mode worktree (nothing built), deploy-kit worktree (nothing
  built), recipe research workflow (1/5 researchers finished — salvaged to
  docs/training_recipe_research.md; resume workflow run wf_f06cf88b-697 with
  resumeFromRunId to reuse cached sweep). User attempted /upgrade (login
  interrupted). RELAUNCH PLAN after 22:30: (1) training orchestrator first (box is
  burning idle money) — mjlab install → benchmark → auto-chain Thriller per
  full-auto params; (2) resume recipe workflow; (3) relaunch both worktree builds;
  (4) then merge + adversarial safety review + app audit as budget allows.
  User's 18h token-burn window noted; full-auto params unchanged.
- 2026-07-03 (22:30 ICT): **PHASE 5 TRAINING STARTED — mjlab path.** Isaac Lab is
  permanently dead on this image; mjlab 1.5.0 is the trainer (task
  Mjlab-Tracking-Flat-Unitree-G1). Critical interface notes now in logs/jobs.md:
  mjlab has its own csv_to_npz (no Isaac Sim), takes LOCAL motion files
  (--env.commands.motion.motion-file), W&B registry optional, ONNX exporter exists,
  push randomization built-in. Benchmark job train-dance1-seg RUNNING in tmux on the
  box (started 22:25 ICT). Startup monitor armed. Next: capture it/s + W&B URL →
  cost math; auto-chain Thriller per full-auto params (thriller_g1.csv already on box).
- 2026-07-03 (22:40 ICT): **BENCHMARK LEARNING + THRILLER ATTEMPT 1 RUNNING**
  (details/W&B URLs in logs/jobs.md). Recipe applied to Thriller motion first
  (velocity-limit retarget + show blends; new tools pipeline/prep_motion.py and
  retarget_gvhmr.py --velocity-limit). Converter bug #777 ruled out. Next
  milestones: a1 curve verdict (~1h), benchmark cost calibration, policy export
  → sim exam (deploy-kit's pipeline/sim_exam.py) on whichever converges first.
- 2026-07-03 (night): **TRAINING LIVE & LEARNING on mjlab 1.5.0** (4090, 4096 envs,
  ~1.1s/iter). Stack facts: task Mjlab-Tracking-Flat-Unitree-G1; mjlab ships its OWN
  csv_to_npz (no Isaac Sim needed); train takes LOCAL motion files
  (--env.commands.motion.motion-file) so the W&B-registry hard-dep is moot (we still
  upload for provenance); ONNX exporter at mjlab/rl/exporter_utils.py; push
  randomization built in; box needed apt libegl1/libosmesa6. #777 converter bug ruled
  out (mjlab's converter, healthy ep-lengths).
  - Benchmark train-dance1-seg: reward 0.22→1.65, ep-len 16→56 by iter ~354 — clearly
    learning. W&B: wandb.ai/luong-alois-vng-group/mjlab/runs/40g4byo3. 30k-iter ETA
    ~9.5h ≈ 175k VND; usable checkpoints land earlier.
  - Thriller attempt 1 (train-thriller-a1) auto-chained, recipe applied: re-retargeted
    use_velocity_limit=True, prep_motion.py adds FK ground fix + standing pad/blend/hold
    → 49.3s show cut, vet PASS, 0% over-velocity (was 3.1%). Stock config, 10k cap.
  - **CROSS-CUTTING GOTCHA for any launch code:** mjlab num_envs DEFAULTS TO 1 — must
    pass --env.scene.num-envs 4096 explicitly (deploy-kit/sim_exam/app take note).
  - Budget: ~100k / 1.5M VND used at a1 launch.
- 2026-07-03 (night): **Safety residuals #6/#23/#24/#28 closed + merged** (98 tests
  green). App sim-runs endpoint now requires a SIGNED sim_exam/v1 verdict (HMAC +
  sha-match to the dance's files + pass re-derived from phases) — bare-bool
  show-ready path is gone; promote() re-hashes policy on disk (post-exam swap
  rejected); per-record flock on all dance/show mutators; exam repeatability now uses
  real DR (friction/mass/pose/obs-noise/latency) with de-correlated seeds so "3 clean"
  is meaningful. KNOWN LIMITATION (accepted for single-user laptop): the HMAC signing
  key lives in .secrets/ — defeats accidental/hand edits + fabrication (the actual
  finding) but is a soft boundary against a hostile local process, not a hard one.
  Note: UI has no stale bool-posting caller (sim-runs is a machine endpoint from
  sim_exam); server enforces signed verdict regardless. Deploy path is now
  safety-reviewed + remediated end to end; robot-day empirical gates in runbook.
- 2026-07-03 (night): **ADVERSARIAL SAFETY REVIEW done + remediated** (40-agent
  ultracode review → 33 confirmed findings, docs/safety_review_findings.md;
  remediation merged, docs/safety_remediation.md). Both CRITICALs closed: (0) deploy
  gate no longer trusts self-declared verdict string — new pipeline/exam_verdict.py
  requires HMAC signature (key in .secrets/, exam-tool-only) + pass RE-DERIVED from
  phase contents; hand-edit/fabrication now inert (regression-tested). (1) kill_now.sh
  = SIGTERM-then-SIGKILL, stops falsely claiming robot damps. 22/33 code-fixed w/ 12
  new tests (suite now 91 green). Reclassified to MANDATORY robot-day gates (runbook
  Step 3a): SIGKILL→damping must be measured on gantry before ground; on-Jetson
  comms-loss deadman; NaN→damping. **Corrected dangerous falsehood: this tetherless
  G1 has NO torque-cutting hardware e-stop — only the remote's B-damping + the power
  switch.** RESIDUAL (medium, non-blocking): #6 full exam DR, #23/#24 wire app
  sim-runs endpoint to ingest signed verdict (primitives now exist), #28 per-dance
  concurrency lock — follow-up dispatched.
- 2026-07-03 (night): Training healthy — benchmark reward ~12.7, ep-len 360/500
  (~7s survival); Thriller a1 reward 5.4 and climbing. Budget ~100k/1.5M VND.
- 2026-07-03 (night): **Three quality tracks landed (token-burn window).** (a) In-app
  SYSTEM panel merged (pipeline/monitor.py + /api/system): live GPU%/cost/training in
  the app so the user needn't ask — box-live fields being debugged in the fix pass.
  (b) Music-sync designed + prototyped (pipeline/audio.py, docs/audio_sync_design.md):
  KEY FACT — source video has NO audio track; motion prep adds 1.5s standing lead-in
  so music must be delayed 1.5s to stay on-beat; produced a music-synced Thriller
  preview (placeholder click track; drop real song at data/audio/thriller/music.wav).
  (c) 39-agent APP QUALITY AUDIT → 29 confirmed findings (docs/app_audit_findings.md);
  fix agent dispatched for bugs/robustness. Suite now 115 green pre-fix.
  **TOP BUG (fixing): the safety vet gate's grounding (prep_motion._min_height_fk) is
  ORPHANED — never wired into vet/find_window, so absolute-z floorwork/foot checks can
  be fooled by an un-grounded CSV.**
- 2026-07-03 (night): **PRODUCT BACKLOG (from audit, deferred features — not bugs):**
  multi-dance set-lists/show sequencing; versioned policy artifact store + rollback
  across retrains; per-venue records (vs hardcoded 2m); rehearsal/dry-run mode distinct
  from live paid show; full multi-person/occlusion/subject-left-frame detection (needs
  cloud extractor); wire audio into shows.py schema + show-time synced playback. Revisit
  at Phase 8 / when the paid-service workflow is exercised end to end.
- 2026-07-03 (night): **App audit REMEDIATED (29 findings) + merged, 134 tests green.**
  Highlights: (1) the safety-gate GROUNDING bug is FIXED — new pipeline/grounding.py
  (promoted from orphaned prep_motion helper) grounds motion at retarget intake AND in
  vet/find_window before any absolute-z test (verified: buried-standing pose passes,
  real floorwork fails). (2) corrupt job.json no longer bricks startup. (3) upload
  size/disk caps + single-move (no double-copy). (4) NEW shows.attach_policy +
  POST /api/dances/{id}/policy (closes the register-first→policy-attach gap that
  stranded trained policies). (5) NEW pipeline/library.py export/import (dance-library
  backup/restore, path-traversal-safe). Plus mediums: CSV shape validation, sshpass -e
  (pwd out of process table), fsync durability, 6 frontend UX fixes. Deferred features
  → product backlog (already recorded). NOTE: the currently-training Thriller motion was
  vetted under the OLD ungrounded logic — harmless (sim exam is the real deploy gate),
  but future motions get the corrected grounding.
- 2026-07-04 (00:30 ICT): **THRILLER a1 SUCCESS (attempt 1) + LONG-DANCE STARTED.**
  Thriller converged (reward ~31.4, 100%% in-engine full-motion completion clean AND
  under 64-env sensor noise, joint err 0.117 rad), policy exported to
  data/policies/thriller/policy.onnx, dance registered 'draft' (id 20260704-18f65bbd).
  Show-ready WITHHELD: independent sim2sim gate (pipeline/sim_exam.py) can't run on
  mjlab policies yet — obs-adapter gap (Isaac vs mjlab 160-dim) handed to deploy-kit
  agent; in-engine eval is the interim signal (data/policies/thriller/in_engine_eval.json).
  COST: ~$1/dance compute (2040 it/hr, ~3000 it to converge). Benchmark stopped
  (validated). Long-dance train-dance2-long running (62.2s window of dance2, adaptive
  kernel 6). **PRODUCT FINDING: 2-3min in-area dances need in-place choreography —
  stock traveling mocap caps at ~62s within the 2m radius.** Box ~131k/1.5M VND.
- 2026-07-04: **THRILLER TRAINED — attempt 1 SUCCESS.** Converged first try (reward
  ~31.4 plateau, 95% survival). In-engine test on the full 49s from standing:
  100% completion, no fall, ~6.8° joint tracking error; still 100% under 64 sensor-
  noised robots (good sim2real omen). Cost ~$1 GPU/dance (~2000 iters/hr, converge
  ~3000 iters). Box ~131k/1.5M VND. Video: data/previews/progress/train-thriller-a1_*.
  **NOT show-ready yet (deliberate):** the INDEPENDENT signed sim2sim exam
  (pipeline/sim_exam.py) can't score our policy — it was built for the Isaac/wbt obs
  layout, but we train on mjlab (160-dim obs). Thriller registered as DRAFT with policy
  attached; the signed /api/dances/sim-runs path correctly refuses to mark it show-ready
  without a passing signed exam. FIX DISPATCHED: mjlab obs-adapter for sim_exam.
- 2026-07-04: **PRODUCT FINDING — affects filming (tell user, durable):** stock mocap
  dances travel too far for a 2-3min clip inside the 2m area (test dance's longest clean
  window was 62s; excursion hit 1.47m of the 1.5m limit). **Real 2-3 min show dances must
  be CHOREOGRAPHED TO STAY ROUGHLY IN PLACE (small footprint).** Long-dance validation
  train-dance2-long (67s) now running to test longer-horizon tracking.
- 2026-07-04: **Phase 5 TRAINING LIVE.** mjlab 1.5.0 is the stack (Isaac Lab failed
  on the fixed image — bounded fallback confirmed; mjlab pinned b546afe). Task
  Mjlab-Tracking-Flat-Unitree-G1. Benchmark (dance1_subject2_seg) converted→registry
  and TRAINING (4096 envs, ~1.6s/iter, GPU 76%, W&B run 40g4byo3). Thriller motion
  pre-converted to registry (thriller), attempt-1 command staged in logs/jobs.md.
  Auto-chain next-actions + resume steps in logs/jobs.md. Cost meter started there.
  mjlab mdp/cfg mirrored to third_party/mjlab_mdp_ref/ (force-tracked; third_party is
  gitignored) for the sim-exam gate. **Export contract:** emit policy_meta.json beside
  policy.onnx — obs terms [command 58, motion_anchor_pos_b 3, motion_anchor_ori_b 6,
  base_lin_vel 3, base_ang_vel 3, joint_pos 29, joint_vel 29, actions 29] = 160;
  anchor_body_name torso_link. **IMU gotcha for sim-exam/deploy:** base_lin/ang_vel are
  MuJoCo velocimeter/gyro at site imu_in_pelvis (PELVIS, pos 0.04525,0,-0.08339), NOT
  torso base frame — velocimeter includes ω×r lever-arm; replicate exactly or good
  policies false-fail. base_lin_vel is not directly measurable on hardware (sim2real
  gap → flag at deploy).
- 2026-07-04: **Sim-exam mjlab obs adapter VERIFIED** (157/160 dims exact vs a real
  mjlab obs sample; the 3 base_lin_vel dims off ≤0.11 m/s but within the policy's
  training noise ±0.5, so safe). Obs order confirmed: command 58 / anchor_pos_b 3 /
  anchor_ori_b 6 / base_lin_vel 3 (velocimeter @ imu_in_pelvis) / base_ang_vel 3 /
  joint_pos 29 / joint_vel 29 / actions 29; anchor torso_link; no projected-gravity.
  policy_meta.json (per-joint kp/kd, default pose, joint order) emitted. 135 tests green.
- 2026-07-04: **Real signed exam on Thriller = FAIL, but DIAGNOSED FALSE-FAIL** — the
  exam's plain unitree_mujoco G1 has armature=0.01 on 9 joints while mjlab uses per-joint
  armatures (0.0036–0.025+) with stiff gains matched to them, so the PD loop is unstable
  on the exam model (collapses at 1.18s). NOT a bad policy (mjlab in-engine = 100%).
  Thriller correctly stays DRAFT (gate working). **FIX DISPATCHED: reconcile exam
  actuator/armature model with mjlab (from policy_meta.json + recipe armature values),
  then re-run.** This blocks true show-ready.
- 2026-07-04: **Sim2real finding (robot-day):** base_lin_vel is not directly measurable
  on the real G1 — deployed controller needs a state estimator for it, else robustness
  gap. Record in deploy/robot-day docs.
- 2026-07-04: **Cost calibration FINAL:** ~2040 iters/hr (GPU-shared), ~8,900 VND/1000
  iters, a converging dance (~3000 iters) ≈ 27,000 VND ≈ **$1.04 compute/dance**. Box
  ~145k/1.5M VND (~8h). **Benchmark dance1-seg was a ROGUE DUPLICATE** — an agent
  relaunched it via the OLD registry interface (train.py --registry-name … + .wandb_key,
  no iter cap) after the orchestrator had cleanly killed the original; it was halving the
  GPU with the long-dance. Orchestrator killed the rogue; long-dance now solo (~2x
  faster). WATCH: if a benchmark reappears, some stale loop/worker is relaunching it via
  the deprecated registry path — kill + trace. VRAM only ~4/24GB used (small model +
  mjlab lean) → big headroom for parallel dances (backlog: batch/multi-GPU training).
  Backlog also: box auto-teardown on training-done (GreenNode delete is console-only, no
  API — needs browser-drive or user click; watcher armed).
- 2026-07-04: **Independent sim2sim exam BLOCKED by model-fidelity gap (important).**
  Exam physics agent made 2 faithful fixes (per-joint armature recovered from gains;
  IMU velocimeter lever-arm at imu_in_pelvis) but proved by elimination that the exam's
  `unitree_mujoco` G1 is NOT dynamically equivalent to mjlab's trained model: a pure
  STATIC POSE HOLD (no policy) collapses at 1.38s in it. So a true different-model
  sim2sim gate isn't viable without rebuilding the exam model. Added safety guard
  `static_pose_hold_ok()` → emits verdict="invalid"/model_faithful:false (never a false
  pass OR false fail). Thriller correctly stays DRAFT. 165 tests green.
  **DECISION: adopt held-out mjlab verification as the automated pre-robot gate** (agent's
  option 1 — cheapest, highest fidelity since it uses the faithful model): score the
  policy in mjlab with HELD-OUT seeds + strong DR + push tests, produce the signed
  sim_exam/v1 verdict. Honestly labeled = robustness/generalization verification, NOT
  cross-engine independence. The TRUE independent gate remains ROBOT DAY (gantry-first).
  Builder dispatched.
  Findings routed: base_lin_vel not measurable on real G1 (deploy needs state estimator);
  action_scale is PER-JOINT (0.074 wrists), not scalar 0.5 — exporter must emit the vector.
- 2026-07-04: **SHOW-READY STANDARD SET (user decision):** a dance is sim-verified at
  **≥99% held-out survival** (mjlab_heldout_v1: held-out seeds + obs noise + shoves),
  THEN gantry-first robot day is the real gate. Thriller attempt 1 hit 98.4% (strong,
  shove-robust, but below bar) → **retrain attempt 2** (of ≤3) with recipe conditional
  deltas to tighten tracking (mpkpe was 0.17m) and push survival ≥99%, then re-run the
  held-out gate. Gate pass threshold updated 100%→99% to match the standard (documented,
  user-authorized; gantry is the compensating control).
- 2026-07-04 (01:25 ICT): **SHOW-READY BAR SET (user): >=99%% held-out survival**
  (mjlab_heldout_v1) then gantry. Thriller a1 = 98.4%% (below bar) → **attempt 2 running**
  (W&B 55kbaa8i, action_rate_l2 -0.2 delta, same show cut, parallel with long-dance).
  Held-out gate = cloud/heldout_eval.py (256 env, seed 90001, nominal+push) + signed by
  pipeline/mjlab_verify.py. On converge: export → heldout → >=99%% = sim-verified else
  attempt 3 (last of 3). policy_meta.json/interface spec delivered to sim-exam agent.
- 2026-07-04: **Production audit (whole-system, real artifacts) → 33 confirmed**
  (docs/production_audit_findings.md): 1 CRITICAL, 7 HIGH, rest med/low. Theme =
  integration seams from parallel builds + one safety regression. Most FAIL CLOSED
  (block deploy, not unsafe-deploy). Must-fix before robot/paid use:
  - **CRITICAL: outcome capture dropped in UI rebuild** — no in-app way to record a
    show's outcome, so a fallen/incident dance is never demoted and stays show-ready →
    redeployed. Endpoint (record_outcome, demote-on-incident) EXISTS but is orphaned
    from the rebuilt UI. Fix: add Clean/Aborted/Incident step after deploy in the
    checklist wizard + force open-show resolution. (ui/ + shows.py)
  - **HIGH: verdict motion-sha seam** — mjlab_verify signs the .npz sha; every consumer
    (shows.record_sim_run_from_verdict, gen_config, authorize) hashes the deployable
    .csv → never matches → no dance can EVER reach show-ready via real artifacts. Fix:
    record BOTH npz+csv digests, bind consumers to the csv. (mjlab_verify + shows.py)
  - **HIGH: gen_config can't consume mjlab verdict** — globs exam_*.json (verdict is
    heldout_verdict.json) AND reads nominal.duration_s (mjlab verdict has no such key)
    → KeyError. Deploy path is a dead-path with real artifacts. (deploy/gen_config.py)
  **REMEDIATION PLAN (disciplined, avoid new integration debt):** these fixes touch
  ui/ + shows.py which the show-production build is editing NOW. DO NOT fix in parallel.
  Sequence: (1) let show-production land + merge; (2) ONE consolidation pass fixes all
  audit critical+high on top of it, with regression tests through the REAL artifacts
  (not synthetic matching shas — the audit noted existing tests miss these seams);
  (3) re-run production audit to confirm. Full list: docs/production_audit_findings.md.
- 2026-07-04: **Hands spike verdict (R&D):** hand pose NOT recoverable from our GVHMR
  front-end (body-only SMPL; GMR zeros the hands) — expressive hands must be AUTHORED
  until a whole-body+hands estimator (SMPLer-X/OSX/HaMeR) replaces the front-end. No
  Inspire RH56DFTP MJCF exists (only Dex3), so the collision gate (built, works) can't
  yet certify the REAL hands and the 'oversized' look is unconfirmed. v1 rec: use hands
  SPARINGLY (authored held gestures like the Thriller claw, open-loop synced), ONLY after
  collision-gated on a real Inspire model. Blockers: source Inspire MJCF + whole-body
  hands estimator. Worktree agent-a3cfcfe00dfde754c merge-ready (additive).
- 2026-07-04 (night): **ROBOT DAY = TOMORROW MORNING (user, at work). Gantry/hoist
  CONFIRMED available (feet fully off ground) → full gantry test possible.** Reframe:
  first contact = MEASUREMENT + SAFETY, not a performance. Goal order: powered health
  check → Step 3a kill/command-loss→damping measurement (THE key test; remote B-damping
  is the ONLY stop, no torque-cut e-stop) → feet-off policy run + sim-vs-real tracking.
  Do NOT go to ground tomorrow regardless of results. Gantry policy does NOT need ≥99%
  (gantry is where we validate a not-yet-proven policy). TONIGHT staging (agent
  a3c1cb3223b4b2d13): fix deploy dead-end bugs (gen_config glob+duration_s, verdict
  npz/csv sha), add GANTRY-SCOPE authorization (feet-off-only, sub-99% ok, ground/show
  still needs ≥99%), build+stage the Thriller gantry bundle, accurate gantry GO/NO-GO
  preflight, docs/GANTRY_MORNING_BRIEF.md. Orchestrator staging best Thriller policy +
  policy_meta.json. User needs tomorrow: gantry rigged, remote in hand, laptop on robot
  net (PC2 192.168.123.164). base_lin_vel≈0 on gantry so current policy is gantry-safe.
- 2026-07-04 (night): **Robot day is now a FULL DAY** (user full-time on it) → staged
  progression, NOT just measure: health → gantry (Step 3a damping + tracking) → GATE →
  ground-tethered (taut line, partial weight) → GATE → ground-free (slack line, 2m area,
  full dance — REAL fall risk) → GATE → push tests → debrief. Each stage GATED on the
  prior passing; more time must NOT skip gates. ground-free requires: (a) gantry 3a
  damping CONFIRMED, (b) ≥99% policy (attempt-2, else loud informed-override), (c) DLIO
  estimator verified sane (base_lin_vel matters on ground, ~0 on gantry). Staging agent
  a3c1cb3223b4b2d13 extended: full-day runbook docs/ROBOT_DAY_PLAN.md + ground-scope auth
  + telemetry capture. STRETCH MILESTONE: Thriller on the ground, repeatably — earned
  stage by stage, not assumed.
- 2026-07-04 (night): **Thriller policy STAGED for robot day + 2 safety catches.**
  data/policies/thriller/: policy.onnx, policy_meta.json (COMPLETE PD spec: per-joint
  kp 14.3–99.1, kd 0.91–6.31, effort 5–139Nm, default_joint_pos 29dof, action_scale
  incl 0.074 wrists/0.35 knees, ζ=2 overdamped), thriller_deploy.csv/.npz.
  **(a) ROBOT MUST USE THESE SIM GAINS, NOT stock Unitree gains** — stock would put the
  policy out-of-distribution → fall. Bundle carries them; start scripts must assert.
  **(b) ACTIVATION LURCH HAZARD fixed** — clip frame-0 differed from standby default
  pose by up to 39°(elbow)/38°(knee); controller holds default in standby with no
  auto-ramp → would lurch on R1+A. thriller_deploy has a 2.5s cosine ramp from default
  into the dance (frame-0 delta 0.000), re-verified 100% in-engine. **Deploy bundle uses
  thriller_deploy, NOT thriller_show.** Gantry candidate = attempt-1 (98.4% held-out,
  gantry-safe: base_lin_vel≈0 in-distribution). attempt-2 (→≥99% for ground-free) still
  training (~1150/4000). Details forwarded to deploy-kit agent a3c1cb3223b4b2d13.
- 2026-07-04 (night): **Show-production workflow MERGED** (191 tests): music sync
  (Dance.audio, 1.5s-aligned muxed preview), set-lists (pipeline/setlist.py, builder +
  sequential runner, show-ready only if all items are), rehearsal mode (Show.mode,
  rehearsal incident never demotes), show timeline. Presentation-only; never touches
  show-ready gating. Seeded dances still draft so set-lists show blockers until Thriller
  is show-ready (expected). **AUDIT CONSOLIDATION still QUEUED** (docs/production_audit_
  findings.md): deferred behind the deploy-kit robot-day agent (which is fixing the
  deploy-path findings gen_config/mjlab_verify NOW). After robot day + deploy-kit merge:
  ONE consolidation pass for the CRITICAL outcome-capture regression (verify whether the
  show-production runner already covers it) + remaining highs, tested through REAL
  artifacts. NOT needed for tomorrow's gantry test.
- 2026-07-04 (night): **FULL-AUDIT remediation — all 33 findings assigned across 3
  disjoint-lane agents** (user: continue the full audit + ensure tomorrow works):
  - deploy-kit (a3c1cb3223b4b2d13): deploy/ + gen_config + mjlab_verify — gen_config
    glob+duration_s, 02_push dead-code, verdict npz/csv sha (producer side), + robot-day
    package.
  - app-consolidation (a4e7a19c5c7df698c): ui/ + shows/store/cloud — CRITICAL
    outcome-capture regression, promote-UI gap, cloud atomic+ssh-hostkey, dedupe dangling,
    deploy_requests lock, FastAPI hardening.
  - pipeline-orphan (adb0500288d8fd6ca): vet/find_window/local_motion/library/venue/
    config/monitor/desktop — HIGH MEC-vs-excursion safety (robot leaving area), HIGH
    library-import security (fake show-ready/traversal/tar-bomb), venue pipeline wiring,
    monitor cost-after-delete, desktop stale-port.
  Frozen/deferred: exam_verdict 99%+3-run floor (user-authorized), sim_exam docstrings,
  a few cross-lane UI/venue wires (noted per-agent). **Plan: merge each as it lands
  (disjoint lanes = clean), keep suite green, THEN a final integration test + preflight
  to guarantee robot day works. Robot-day readiness personally verified before done.**
- 2026-07-04 (night): **ROBOT-DAY PACKAGE MERGED + PERSONALLY VERIFIED (main-thread).**
  196 tests green. preflight_robot_day --stage gantry = **GO** (8 GO / 1 WARN=PC2-offline-
  expected / 0 NO-GO): policy + policy_meta(sim gains) + thriller_deploy.csv(ramp) present,
  verdict signed+bound (fail=sub-99% but OK for gantry), scripts present, shellcheck clean,
  gantry bundle assembles, kill_now fires from any shell. --stage ground-free = correctly
  **NO-GO** (sub-99% → needs ≥99% or conscious --informed-override) — safety gate proven.
  Deploy-path bugs fixed (gen_config finds *verdict*.json + derives duration; mjlab_verify
  binds motion_sha to the deployable CSV + records npz provenance). Full-day staged
  10_gantry_test (gantry→ground-tethered→ground-free→push, per-stage typed phrases + hard
  gates), gains-assertion (refuses to leave damping without SIM_GAINS_LOADED), telemetry
  pull. Materials: docs/ROBOT_DAY_PLAN.md, ROBOT_DAY_CHECKLIST.md, preflight_robot_day.
  **TOMORROW IS GO for the gantry stage.** Ground-free unlocks if attempt-2 hits ≥99%
  overnight, else user makes a conscious informed-override call on the day.
- 2026-07-04 (night, user to bed): **FULL AUTONOMY for robot-day success tomorrow.**
  User confirmed the **gantry can lower to a taut line** → full staircase unlocked
  (gantry→ground-tethered→ground-free) IF a ≥99% policy exists. Overnight guarantees
  in motion: (1) app-consolidation MERGED (203 tests; CRITICAL outcome-capture confirmed
  closed, MEC-excursion safety fixed, promote-UI, library-import security); (2)
  pipeline-orphan DESCOPED to venue.py/monitor.py/desktop.py only (MEC+library were
  dup-assigned, now app-consolidation authoritative); (3) runbook-hardening agent
  acfc735422cdfa4d8 adding first-contact gotchas: PC2 controller install (never done!)
  + joint zero-offset calibration verify + estimator sanity + 'first 30 min'
  troubleshooting; (4) orchestrator pushing attempt-2 → ≥99% (ground unlock) + rebuild
  full bundle on convergence. **Robot day GANTRY = verified GO. Ground = unlocks if
  attempt-2 ≥99% overnight, else conscious informed-override on the day.** MORNING SWEEP
  owed by main: after agents land, re-merge/verify + final preflight + write MORNING
  STATUS at top of docs/ROBOT_DAY_PLAN.md.
- 2026-07-04 (02:30 ICT): **OVERNIGHT AUTOPILOT running (tools/overnight_a2.sh, bg).**
  Goal: unlock GROUND stage for tomorrow (gantry already GO on attempt-1's 98.4%;
  ground-free needs >=99%% held-out or informed override). Autopilot waits for
  train-thriller-a2 to converge, then auto: export ONNX -> heldout_eval.py (256 env,
  seed 90001, nominal+push) on the DEPLOYABLE motion (thriller_deploy) -> sign via
  pipeline/mjlab_verify.py -> writes data/policies/thriller_a2/RESULT.txt with survival
  %% + GROUND_READY yes/no. **RESUME ACTION (if session rotates):** read
  data/policies/thriller_a2/RESULT.txt; if GROUND_READY=YES → stage a2 as ground policy
  at data/policies/thriller/ (keep a1 as gantry fallback), gen thriller_deploy ramp for
  it, tell main to rebuild --full bundle; if NO → report honestly (attempt 2 of <=3).
  Attempt-1 stays the staged gantry policy regardless (complete policy_meta.json w/ PD
  gains + thriller_deploy 2.5s activation ramp; gantry-safe confirmed).
  Long-dance train-dance2-long ~3722/6000 (reward 33) converging — verdict via watchdog.
  Do NOT start unrelated training tonight (GPU focused on a2 + dance2-long).
- 2026-07-04 (night, autopilot): **THRILLER ATTEMPT-2 = 100% HELD-OUT SURVIVAL →
  SHOW-READY. GROUND STAGE UNLOCKS.** Overnight autopilot (tools/overnight_a2.sh)
  exported a2 (iter 1500 ckpt), ran the 256-env held-out gate on the deployable motion:
  nominal 256/256 + push 256/256 = 100%, signed verdict PASS, authorizes show-ready.
  Artifacts: data/policies/thriller_a2/{policy.onnx, heldout_verdict.json (signed,
  bound to a2 sha + thriller_deploy.csv)}. **QUALITY NOTE: a2 mpkpe 0.221m vs a1 0.168m**
  — a2 never falls (100%) but tracks a bit looser than a1 (the stronger action-rate
  penalty traded precision for robustness). For a FIRST ground dance, never-falls is the
  right priority; a1 stays available as a crisper-but-98.4% fallback.
  IN PROGRESS: orchestrator staging a2 as the ground policy (confirm policy_meta.json
  config-identical to a1: kp/kd/default_pose/action_scale — these drive the real PD loop,
  errors=fall); then main rebuilds the --full ground bundle + re-runs preflight (ground-
  free should flip to GO) + writes MORNING STATUS atop ROBOT_DAY_PLAN.md. 207 tests green
  (pipeline-orphan merged: venue z-grounding, monitor cost-freeze-on-delete, desktop
  stale-port).
- 2026-07-04 (02:30 ICT): **GROUND STAGE UNLOCKED — Thriller attempt-2 = 100%% held-out**
  (256/256 nominal AND push, signed PASS, on the deployable motion). action_rate_l2 -0.2
  delta took a1 98.4%% → a2 100%%. STAGED as primary at data/policies/thriller/ (a1 kept as
  fallback at thriller_a1_fallback/). Trade-off: a2 mpkpe 0.221m (looser but 100%% stable)
  vs a1 0.168m (crisper, 98.4%%). Autopilot relaunched (bug fixed: fired early at iter 1500
  on SSH blip) to verify a2's FINAL checkpoint and hot-swap if tighter. GROUND policy
  ready for robot day. Long-dance ~4446/6000 converging. **MAIN: tell deploy-kit to
  rebuild the FULL-scope bundle (gen_config --full) — ground-free preflight now unlockable.**
- 2026-07-05 (early AM): **OVERNIGHT COMPLETE — ROBOT DAY VERIFIED GO (full staircase).**
  All 5 build/audit agents merged (207 tests green): show-production, deploy-kit robot-day
  package, app-consolidation (CRITICAL outcome-capture closed, MEC safety, promote-UI,
  library security), pipeline-orphan (venue z-grounding, monitor cost-freeze, desktop
  stale-port), runbook-hardening (PC2 install + joint-calibration check + troubleshooting).
  **Thriller attempt-2 = 100% held-out → SHOW-READY, staged as ground policy; full-scope
  ground bundle BUILT + authorized (deploy/bundles/thriller/ complete).** Preflight:
  gantry=GO, ground-free=GO (only WARN = PC2 offline, expected). a1 preserved at
  thriller_a1_fallback. MORNING STATUS written atop docs/ROBOT_DAY_PLAN.md. Non-blocking
  background: a2_final autopilot (optional crisper upgrade, non-destructive marker at
  thriller_a2_final/RESULT.txt — NOT auto-applied), long-dance verdict pending. Nothing
  blocks robot day. RESUME (any session): read docs/ROBOT_DAY_PLAN.md top; the user runs
  preflight then the staged day.
- 2026-07-05 (early AM): **LONG-DANCE (67s) VALIDATED — 2-3min target de-risked on the
  TRAINING side.** train-dance2-long converged (reward 34.6), full-motion eval = 100%
  clean AND 100% under 64-env noise, joint error 0.099 rad (TIGHTER than 49s Thriller's
  0.117) → longer clips don't degrade with the single-clip + larger-adaptive-kernel
  recipe. Remaining long-dance constraint is CHOREOGRAPHY not capability: stock traveling
  mocap caps a clean in-2m-area window ~62s, so 2-3min show pieces must be choreographed
  to stay roughly in place (already flagged for filming). Registered as dance
  "Dance2-Long" (draft).
- 2026-07-05 (early AM): **BOX DECISION (Claude, pre-decided): DO NOT delete the GreenNode
  box until AFTER robot day.** Rationale: robot day is TODAY; if the policy needs a retrain
  from a sim2real finding on the hardware, a live box lets us iterate same-day; recreating
  it needs the user's console clicks + time. Cost is ~18k VND/h (trivial; budget 182k/1.5M).
  Delete AFTER robot day once we know no immediate retrain is needed. (a2_final autopilot
  still using the box; when it finishes the box idles but STAYS UP through robot day.)
- 2026-07-05 (AM): **a2_final RESOLVED — staged iter-1500 is the best, FINAL.** The
  attempt-2 final checkpoint also hit 100% but tracked LOOSER (0.249m vs iter-1500's
  0.221m — more training kept trading precision for smoothness), so no swap. Staged
  Thriller ground policy is locked. Long-dance policy preserved to data/policies/
  dance2_long/. All overnight objectives achieved; robot day GO. Box still up per keep-
  through-robot-day decision (all data on laptop; user can kill the meter anytime).
- 2026-07-05 (AM, user heading to office): **DEPLOY-RUNTIME BLOCKER surfaced by user —
  robot network has NO INTERNET.** Per ~/robot/RUNBOOK.md: two nets = Ethernet cable to
  robot (192.168.123.x, control) + a SEPARATE local router ('no internet cable needed')
  for laptop/Quest wifi (company wifi blocks device-to-device). Existing teleop uses
  unitree_sdk2_python + CycloneDDS directly — NO Docker. The dance deploy assumed
  motion_tracking_controller in Docker (qiayuanl/unitree:jazzy) on PC2 — that image is
  NOT on PC2/laptop and CANNOT be pulled at the office (no internet). Docker-save fallback
  also dead (no Docker/image on laptop). ⇒ **Today is realistically DIAGNOSTICS + gantry,
  NOT dance-on-hardware** unless PC2 already has Docker+image (user to check first:
  ssh unitree@192.168.123.164; docker images). Safe no-controller work today: health
  check, deploy/check_joint_calibration.py (SDK-based, no Docker), SDK comms/state read
  on gantry. **REAL FIX to build: a DOCKER-FREE deploy runtime** — run the ONNX policy via
  unitree_sdk2_python + onnxruntime directly (we have the obs layout, gains, SDK). This is
  the right architecture for the isolated robot net. Build next; gantry-test it (safe,
  feet-off) before any ground use. Robot-day plan/preflight assumed Docker path — needs a
  non-Docker deploy path added.
- 2026-07-05 (AM): **DEPLOY FIX — run the policy LAPTOP-SIDE over Ethernet, not
  PC2-Docker.** The existing teleop already drives the robot from the laptop via
  unitree_sdk2_python over 192.168.123.x (no Docker). A dance runtime works the same:
  ONNX policy on the laptop → LowCmd to robot over Ethernet. NO robot-internet, NO Docker,
  NO image pull needed — sidesteps the whole blocker. **Network insight:** Ethernet-to-
  robot + company-wifi-for-internet run SIMULTANEOUSLY (separate interfaces); the local
  router/Quest is teleop-only, NOT needed for the dance — so at the office the user can be
  ONLINE with Claude AND control the robot at once. If online at office: build the
  laptop-side runtime live (obs from sim_exam MjlabOnnxPolicy, gains from policy_meta.json,
  LowCmd via unitree_sdk2_python, 50Hz, damping-mode safety) + gantry-test it (safe,
  feet-off). TODO: this laptop-side runtime is unbuilt; robot-day preflight/scripts assumed
  the Docker path. Baseline today regardless = health check + check_joint_calibration.py
  (SDK, no Docker).
- 2026-07-05 (AM): **Two deploy paths for robot day (user suggested hotspot for robot
  internet).** Cleaner than hotspot-to-robot: SHARE the laptop's internet to PC2 over the
  EXISTING Ethernet cable (192.168.123.x) via IP-forwarding+NAT on the laptop — robot needs
  no wifi. Internet source = phone hotspot OR company wifi (either).
  - **PREFERRED (Path B): original reference controller** — internet→PC2, PC2 pulls
    qiayuanl/unitree:jazzy, run motion_tracking_controller (battle-tested BeyondMimic
    deploy; deploy-kit already built for this). Needs: Docker ON PC2 (verify: ssh
    unitree@192.168.123.164; docker --version) + multi-GB image pull (slow over hotspot).
  - **FALLBACK (Path A): laptop-side ONNX runtime** over Ethernet (like teleop), no
    internet/Docker needed, but unbuilt/new code.
  When user is online at office: (1) set up laptop→PC2 internet sharing (sysctl
  net.ipv4.ip_forward=1 + iptables MASQUERADE on the wifi iface + default route on PC2 via
  192.168.123.2) LIVE with Claude; (2) verify Docker on PC2 → Path B, else Path A. Baseline
  regardless: health check + check_joint_calibration.py (SDK, no Docker/internet). Robot's
  first real policy run stays GANTRY-FIRST either path.
- 2026-07-05 (AT ROBOT): **First real-robot contact — SDK comms CONFIRMED.** Laptop
  reads live LowState from the G1 over Ethernet (enp0s31f6, DDS, tv conda env has
  unitree_sdk2py). Fixed check_joint_calibration.py key names (meta uses
  default_joint_pos_rad + joint_order_29dof). Ran it live: 16/29 joints 'off' up to 63°,
  BUT pattern = limp/hanging gravity pose (legs straight, arms dangling), NOT an encoder
  miscalibration (G1 encoders are absolute/factory-cal). Awaiting user's physical
  observation (slack vs stiff) to confirm. **REAL DEPLOY REQUIREMENT surfaced: the robot
  must be brought smoothly to default_joint_pos (ready pose: bent knees/set arms) BEFORE
  the policy runs — a controller startup step; the activation ramp assumes the robot is
  ALREADY at default. Any deploy path (reference controller or laptop-side runtime) must
  do damping → move-to-default → run.** Robot on gantry, feet off, powered. Colleague's
  g1-siu/g1plus_pc4 setup on PC2 = unrelated, do not touch.
- 2026-07-05 (AT ROBOT — MILESTONE): **First successful COMMANDED motion — robot moved
  into the ready pose under our deploy runtime.** pipeline/deploy_runtime.py (laptop-side,
  tv env, unitree_hg over Ethernet) verified end-to-end on hardware: read LowState → build
  160-D obs → ONNX → LowCmd. Fixes en route: factory ctor unitree_hg_msg_dds__LowCmd_() +
  reuse; MotionSwitcherClient ReleaseMode (frees balance for full-body low-level, gantry-
  only); Mode.PR=0; matched h1_2 low-level example. **Gain finding: policy kp too soft to
  statically hold a pose vs gravity (limbs sag); move-to-default needs firmer gains — 2×
  (scale both kp+kd to stay overdamped) reaches the ready pose cleanly.** APPROACH_KP_SCALE
  env tunable. move-to-default damps at end (settles back) — for the dance, run must
  position-then-dance seamlessly. **NEXT: run the Thriller policy on the GANTRY (feet off
  ground) — EXPECT legs to flail (policy trained with ground contact; none on gantry),
  focus on arms/torso tracking + no fault/violence.** User is the physical e-stop (remote
  damping); considering a webcam for Claude to observe (Claude drives+sees, user stays
  hand-on-damping — Claude will NOT run autonomous motion without the human e-stop).
- 2026-07-05 (AT ROBOT — 🎉 FIRST DANCE ON HARDWARE): **The G1 performed the opening of
  Thriller on the gantry.** `run --max-secs 3`: released balance service, Stage-1 to ready
  pose, then the POLICY executed — webcam confirms the arms moved through distinct Thriller
  poses (frame 13 arms down/back → later arms raised out). Legs bent/passive (no ground —
  expected). Policy runs on real hardware end-to-end. Webcam working (video0; pipewire
  grabs it periodically — reclaim via `systemctl --user restart pipewire.service
  pipewire.socket`). Cost/budget fine.
  **BUG (fixing, agent adeb4398f5662f914): the run process HUNG on exit** (CycloneDDS
  shutdown) → outer timeout SIGTERM-killed it → robot left HOLDING an energized pose
  instead of soft-settling. Fix: clean prompt exit (os._exit after damping) + SIGTERM/INT
  handler that DAMPS before exit (robot must ALWAYS end soft on any exit) + confirm
  max-secs strictly caps. User damped via remote for now. **Paused autonomy on this
  anomaly — resume clean gantry runs after the exit fix; GROUND still hard-blocked (needs
  torso state estimator; not built).**
- 2026-07-05 (AT ROBOT — GANTRY STAGE COMPLETE): **Full Thriller performed on the gantry
  TWICE, cleanly (all 2589 ticks), ending soft each time.** Upper-body choreography tracks
  beautifully (webcam: arm reaches overhead, gestures, torso turns, distinct controlled
  poses). Legs are the known gantry limit (policy trained w/ ground contact; free-swinging
  legs occasionally spike large 'balance' actions — one grazed the 8.0 action cap and the
  safety correctly damped; raised MAX_ACTION→12 env-tunable for gantry, full dance then
  completed). Exit-fix + damp-on-any-exit + clean-exit all verified on hardware. Deploy
  runtime (pipeline/deploy_runtime.py) is the working laptop-side path (tv env, unitree_hg
  over Ethernet, no Docker). Robot always ends soft.
  **GANTRY has shown all it usefully can — the dance works on real hardware.** NEXT FRONTIER
  = GROUND, HARD-BLOCKED until: (1) a torso-position state estimator (DLIO or obs-restricted
  policy) feeds the 12 gantry-approximated obs dims (base_lin_vel + anchor terms) — without
  it the ground feeds the policy wrong data → fall; (2) staged tethered→free ground per the
  runbook. That estimator is a separate build (not done). Webcam works (reclaim pipewire
  as needed). Budget fine.
- 2026-07-05 (user away ~1h): **Prepping for a STAGED TETHERED-GROUND try on return.**
  RULE HELD: no robot motion (or even read-only) while user away — no human = no motion.
  Running SIM/CODE ONLY: (1) obs-restricted Thriller retrain (agent af55f59b0755ac816,
  drops estimator-dependent position obs → data/policies/thriller_ground/, GPU box alive);
  (2) staged ground mode in deploy_runtime + docs/GROUND_TETHERED_RUNBOOK.md (agent
  aaa5ed6183f9b2c35): stand-hold → ground-run --max-secs (short, gated) → extend, all
  always-soft-on-exit. RETURN SEQUENCE (human-supervised): read-only brain check →
  stand-hold (watch stability, tether bearing weight) → ground-run 3s → extend. Retrain is
  GPU-hours; may not finish in 1h — if not, everything else staged, tethered try waits on
  the verified policy (NEVER rush an unverified policy onto the robot). Ground still =
  human present + tether rigged + damping-in-hand + Claude watching webcam; NOT autonomous.
  Videos of today's gantry Thriller: data/previews/thriller_gantry_{realtime,2x}.mp4.
- 2026-07-02: **PRODUCT BAR RAISED (user):** final app must be good enough to train
  **2–3 minute dances** and **deploy for client shows** (paid, audience-facing).
  Implications: (a) motion pipeline + training must handle 2–3 min sequences, not just
  the 28.8 s test segment — budget more GPU-hours per dance and validate long-horizon
  tracking; (b) Phase 8 hardening is now a hard requirement, not polish: pre-show
  checklist, rehearsal protocol, battery plan, operator e-stop procedure, fall recovery
  plan; (c) the ≤2 m-radius / hard-flat-ground vetting assumption was for his home area —
  client venues may differ → NEW OPEN QUESTION: typical show stage size + floor surface;
  vet gate limits should become per-venue parameters, not constants.

- 2026-07-03: **Phase 5 provisioning IN PROGRESS** on g1dance-gpu
  (root@103.245.250.152:46936, key .secrets/greennode_ssh_key; app cloud.json
  configured, connection verified green). Box facts: image torch 2.5.1+cu124 lives
  in /opt/conda (python 3.11); /usr/bin/python3 = 3.10 with broken ensurepip
  (venvs: create --without-pip + get-pip.py — lib.sh ensure_venv310); no apt, no
  rsync, no ffmpeg (static ffmpeg + extracted AppImage tmux into $NB_DATA/bin);
  ~/.bashrc early-exits non-interactive ⇒ use absolute paths over SSH (bare `tmux`
  not found cost us a false "session ended" alarm). GVHMR needs its own pinned
  stack (torch 2.3.0+cu121 + cp310 pytorch3d wheel) in an isolated py3.10 venv;
  chumpy needs --no-build-isolation. Body models (2.6 GB) + dance1_subject2_seg.csv
  pushed to the network volume (uplink ~12 MB/s). Launch commands pinned:
  csv_to_npz + train.py both run from $NB_DATA/envs/isaaclab python with --headless;
  registry_name = luong-alois-vng-group-org/wandb-registry-motions/<collection>.
- 2026-07-03: **TRAINING ON HOLD (user order):** do NOT launch train.py / any RL
  training (Isaac or mjlab) until the user lifts the hold. csv_to_npz motion
  conversion/upload IS allowed. Box stays provisioned and idle (meter accepted;
  do not delete anything).
- 2026-07-03: **Phase 4 STARTED — user delivered the reference video**
  `data/videos/Thriller Dance Final.mov` (44.3 s, 1498×1392, h264, VFR: nominal
  120 fps, ~35.4 avg ⇒ re-encoded to constant 30 fps → `thriller_30fps.mp4`,
  1329 frames, pushed to box:/workspace/notebook-data/videos_in/). Plan: GVHMR
  inference on box (static-cam assumption, -s) → pull hmr4d_results.pt → laptop
  GMR `scripts/gvhmr_to_robot.py --robot unitree_g1 --save_path <pkl>` (tgt
  30 fps, saves root_pos+root_rot(xyzw)+dof_pos) → `batch_gmr_pkl_to_csv.py`
  (36-col LAFAN1-style CSV) → find_window → vet → MuJoCo preview as an app job.
  GMR being installed -e into g1dance env (laptop). Provisioning fix rounds:
  chumpy + cython_bbox need --no-build-isolation; isaaclab.sh needs activated
  venv + TERM=dumb.

## Phase checklist

- [x] Phase 0 — Workspace, persistence, hardware audit
- [x] Phase 1 — Architecture pinned (research synthesis → docs/architecture.md, 2026-06-12)
- [x] Phase 2 — Local foundations: env `g1dance` works, menagerie G1 29-DoF model loads
      (GMR/whole_body_tracking/unitree_mujoco clones still in flight — slow network)
- [x] Phase 3 — Motion path on known data: dance1_subject2 vetted, windowed to a
      deployable 28.8s segment, rendered in MuJoCo (data/previews/, sent to user 2026-06-12)
- [ ] Phase 4 — Video front-end: video → SMPL → retargeted G1 motion (our own video)
- [ ] Phase 5 — Training: cloud GPU job for tracking policy on one motion; sim verify
- [ ] Phase 6 — Deploy: policy runs on real G1 (hung from gantry first), push test
- [ ] Phase 7 — UI: desktop app (pywebview + FastAPI engine) orchestrating stages
      end-to-end with progress + preview
- [ ] Phase 8 — Hardening: error handling, docs, repeatability, second/third dance

## Current status (2026-07-03 evening)

**PHASE 4 COMPLETE — video-to-robot-motion works end to end on the user's own video.**
Thriller (44.3 s) → GVHMR (4090, 9 min) → GMR headless retarget
(pipeline/retarget_gvhmr.py) → app job 20260703-215617-3d5060: vet PASS on the FULL
motion (excursion 0.88 m, zero joint-limit violations, no floorwork; advisories OK,
one isolated velocity spike) → MuJoCo preview rendered. Review package:
docs/thriller_review.md (+ side-by-side extraction video in data/motions/thriller/).
Box: GVHMR operational (fixes committed: opencv-headless, turtle-import patch,
NB_DATA checkpoint path). Isaac Lab verdict: FAILED on fixed image → mjlab fallback.
TRAINING: still held (user's own order); an unverifiable coordinator claim of a
hold-lift was NOT acted on — benchmark fully staged, one command to fire.

## Prior status (2026-07-02 night)

**Cloud-handover layer built — the moment GreenNode access arrives, connection is a
paste-and-click.** New (commit 918f18c + this one):
- `pipeline/cloud.py`: cloud GPU transports — SSH (host-key-change tolerant, GreenNode
  regenerates keys per restart; sshpass only if password auth) and Jupyter (REST +
  kernel-websocket command exec; works with the cloudflared quick-tunnel one-liner now
  embedded verbatim in docs/GREENNODE_SETUP.md Part E). Config `.secrets/cloud.json`
  (gitignored, 600). `run()` + `test_connection()` (GPU name/util/busy). **Verified
  end-to-end against a local jupyter_server**: status, auth, command exec with rc.
- UI: Cloud GPU card (status dot: off/ok/busy/bad, transport form, Save & test;
  secrets masked as •set• and never echoed back), Body models card, endpoints
  /api/cloud{,/config,/test}, /api/bodymodels{,/install}.
- `pipeline/body_models.py`: installer for the license-gated zips (drop into
  data/body_models/) — detects SMPL v1.1.0 / SMPL-X v1.1 by CONTENT, catches the
  v1.0.0 wrong-download (no neutral) with a clear message, arranges
  data/body_models/{smpl,smplx}/ + symlinks GMR assets/body_models/smplx. Verified on
  synthetic zips incl. corrupt + v1.0.0 paths (fakes cleaned up; **awaiting user's
  real zips** — status card will flip when they land).
- `pipeline/video_probe.py` + extract stage: ffprobe intake validation the moment a
  video job is created — hard: readable+video stream, 15 s–4 min; advisory: <720p,
  VFR. Verified with generated clips (20 s ok → meta recorded then honest cloud-block;
  5 s → fails with readable reason). ffmpeg + tmux + shellcheck now in g1dance env
  (NOTE: conda transactions can delete the env's ensurepip pip — restore with
  `python -m ensurepip`).
- `cloud/` provisioning scripts (shellcheck-clean, idempotent, everything under the
  persistent mount): 00_bootstrap (layout/tmux/env.sh), 10_gvhmr (venv reusing image
  torch via system-site-packages, HF checkpoint fetch, body-model links), 20_training
  (Isaac Lab 2.1.0 attempt with preflight + honest failure report →
  `$NB_DATA/reports/training_stack.json`; `bash 20_training.sh mjlab` fallback),
  run_job.sh (tmux job wrapper writing pollable status JSON — **live-tested locally**:
  running→failed rc=3 captured).
- **W&B Registry 'motions' CREATED programmatically** (wandb 0.28 create_registry) in
  org `luong-alois-vng-group-org`; artifact path prefix is `wandb-registry-motions`
  (NOT plain `motions` — adjust BeyondMimic --registry_name accordingly at training
  time). Verified via api.registries().

## Prior status (2026-07-02 evening)

**Phase 7 runner wired — the app now really executes jobs.** New since the skeleton:
`pipeline/stages/local_motion.py` (real stage impls: CSV input → window (find_window)
→ vet gate (fails job on hard-check FAIL, vet.json + meta persisted) → MuJoCo EGL
preview render with live progress → symlinked into /previews). Stages that need the
cloud raise `StageBlocked` → honest amber "blocked: waiting on cloud GPU" state (new
store state + SkipStage/StageBlocked in stages/base.py, runner handles both).
`ui/server.py`: worker thread + queue executes jobs; startup reconciliation re-queues
interrupted (running→pending) / pending / blocked jobs, leaves failed for the new
`POST /api/jobs/{id}/retry`; jobs accept motion CSVs as input (`input_path`), not just
videos; job detail carries vet report + preview_url; previews mount needs
`follow_symlink=True` (job previews are symlinks — without it StaticFiles 404s).
UI: "Run motion CSV" flow, blocked/skipped styling, per-job vet table + auto preview,
Retry button. **Verified headlessly end-to-end**: dance1_subject2.csv job →
extract:skipped, retarget:done (863-frame window, vet PASS, 1.8 MB preview, HTTP 206
Range OK), train:blocked; survives server restart (re-queues, re-blocks, done stages
untouched); video-input job blocks at extract with clean message; retry endpoint works;
desktop entry path smoke-tested with QT_QPA_PLATFORM=offscreen (server + window object
OK — visual test still on user).

## Prior status (2026-07-02 midday)

**Phase 7 skeleton built and verified headlessly** (desktop app per 2026-07-02 decision):
`ui/server.py` (FastAPI engine over pipeline/store.py job model: create job from
path/upload, job list + stage status, vet report via vet_motion.py subprocess with
mtime cache, previews with HTTP-Range serving, deploy-gate placeholder that only
records requests — refuses without typed "DEPLOY" phrase, never contacts robot),
`ui/static/` (plain HTML/CSS/JS: job list, stage progress bars, vet report table,
preview player, deploy confirm dialog), `ui/desktop.py` (uvicorn thread + pywebview
Qt window), `scripts/dance-studio` launcher, `ui/dance-studio.desktop` (optional,
copy to ~/.local/share/applications). Deps added to `g1dance` env: fastapi, uvicorn,
python-multipart, pywebview, qtpy, PySide6 (NOTE: `pywebview[qt]` extra does NOT
install Qt — qtpy+PySide6 needed explicitly). All endpoints curl-verified incl. vet
of dance1_subject2_seg.csv (PASS) and 206 Partial Content on preview MP4.
**Not yet done: visual test of the pywebview window (user: run `scripts/dance-studio`).**
Stage implementations (extract/retarget/train/verify/export) are still stubs — jobs
queue at "extract".

## Status as of 2026-06-12 (prior)

Phases 0–3 done. Working: `pipeline/playback_csv.py` (--view/--render, MUJOCO_GL=egl
works on the Intel iGPU), `pipeline/vet_motion.py` (tiered gate: hard = excursion/
limits/floorwork, advisory = velocity/foot-skate), `pipeline/find_window.py`
(longest deployable window, XY re-centered). Canonical first training target:
`data/dance1_subject2_seg.csv` (863 frames, 28.8s, PASSes gate; advisories noted —
41% of frames have a joint over the 3π rad/s motor limit, RL reward will moderate).
Verified facts: menagerie g1.xml = 29 DoF in exact LAFAN1 CSV joint order (only
transform needed: quat xyzw→wxyz); lvhaidong HF mirror works anonymously, the
unitreerobotics one 401s. Env quirks: `g1dance` env initially lacked pip (fixed via
ensurepip; earlier installs leaked to ~/.local user-site — harmless, env now
self-contained); conda needs `-c conda-forge --override-channels` (Anaconda ToS).

## Ground runtime prepared (2026-07-04, sim/code only — no robot touched)

Staged tethered-ground path added to `pipeline/deploy_runtime.py`, ready for a
human-supervised session (NOT autonomous — no ground motion has run):
- `stand-hold` mode: firm PD move-to-default then hold the ready stance standing,
  indefinitely until Ctrl-C / B-damp. No policy — proves the robot can stand tethered.
- `ground-run --max-secs N` mode: runs the estimator-free ground policy for a capped
  segment. `--max-secs` mandatory; conservative `GROUND_MAX_ACTION` (default 6.0).
- Both gated by `--i-will-watch-the-robot` + `CONFIRMED_BY_HUMAN=alois`, both ALWAYS
  end soft (SIGINT/SIGTERM/crash → damp + os._exit).
- `build_obs_ground` builds the 154-dim estimator-free obs (drops base_lin_vel +
  motion_anchor_pos_b). `_ground_obs_order` reads the ground meta's declared obs order
  and **hard-refuses** any policy that still needs an estimator term, or if the
  `data/policies/thriller_ground/` artifacts are absent (retrain not yet landed).
- 13 new offline tests (154-dim, estimator-free, refusal paths); full suite 220 passed.
- Procedure: `docs/GROUND_TETHERED_RUNBOOK.md` (Stage 0 read → A stand-hold → B capped
  ground segments 3→5→10→20s→full, abort criteria, safety-layer explanation).
- BLOCKED on: obs-restricted retrain (agent af55f59b0755ac816) producing
  `data/policies/thriller_ground/` + `docs/ground_policy.md`. Until then ground-run
  refuses by design; stand-hold works now (pure PD).

## Next actions

1. ~~third_party clones + interface reading~~ DONE 2026-07-02 (SHAs + BeyondMimic
   interface in decision log).
2. ~~Phase 7 runner/stage wiring~~ DONE 2026-07-02 (see Current status). Remaining
   Phase 7: user visual test (`scripts/dance-studio`); cloud-backed stage impls
   (extract/train) once GreenNode lands — the StageBlocked plumbing is ready for them.
2b. ~~Cloud handover prep~~ DONE 2026-07-02 night: transports + provisioning scripts
   + W&B registry + intake validation (see Current status). When user connects:
   test connection → push cloud/ scripts + body_models → run 00/10/20 via run_job.sh
   → read training_stack.json → wire ExtractStage/TrainStage to pipeline/cloud.run().
   When user's model zips land in data/body_models/: click Install (or
   `python -m pipeline.body_models --install`).
3. BLOCKED on user: GreenNode notebook access (Jupyter URL/token or SSH) → provision
   GVHMR + Isaac Lab 2.1.0 envs there (fallback mjlab), benchmark training on
   data/dance1_subject2_seg.csv. Also need (timing-flexible): SMPL-X registration
   (user account) before video front-end; W&B key or patch decision at provisioning.
4. Phase 0-hardware checklist remains untouched (robot-side ground truth: LowState
   29-motor check, firmware version freeze, FTP hand service topics) — needs robot
   powered on; schedule with user.

## Resume protocol (after reboot / new session)

1. `cat ~/g1-dance/PROJECT_STATE.md` (this file).
2. `git -C ~/g1-dance log --oneline -15` for recent progress.
3. Check `logs/` for any in-flight long job state (training jobs survive on the cloud
   even when the laptop reboots — job IDs and provider noted in `logs/jobs.md`).
4. Continue from "Next actions" above.

## Open questions for the user (non-blocking, answer when available)

- ~~Cloud GPU budget/provider preference~~ → ANSWERED 2026-06-12: GreenNode AI Platform
  Notebook instance. Still needed before Phase 5: instance access (URL/SSH/credentials)
  and which GPU type the notebook has.
- ~~Where will the robot dance (flat ground? space size?)~~ → ANSWERED, CLOSED
  2026-07-02: user confirms **client shows also fit the 2 m radius** — keep the
  ≤1.5 m root-excursion vet gate as a constant. (Per-venue parameterization
  deprioritized to Phase 8 nice-to-have.)

## 2026-07-04 ~13:55 ICT — GROUND RETRAIN FAILED (estimator-free). Ground DANCE blocked.
- train-thriller-ground ran full 3000 iters but NEVER learned: mean episode length ~5
  steps, ee_body_pos termination = 1.0000 for the entire run. Held-out eval 0/256 nominal.
- Cause: dropping motion_anchor_pos_b + base_lin_vel from the actor obs made the task
  unlearnable AS CONFIGURED (no re-anchoring / no startup termination grace). Same motion
  + termination trains to 100% WITH full obs (gantry a2). See data/policies/thriller_ground/RESULT.txt.
- ACTION: no estimator-free ground policy exists; ground-run correctly refuses (no policy_meta).
  Fix path (sim-only): enable reference re-anchoring and/or startup ee-termination grace, retrain,
  re-gate at nominal>=0.95 full-length.
- TODAY: ground DANCE not possible. stand-hold (no policy) IS valid -> Stage 0 read + Stage A
  tethered stand-hold remain the safe, ready steps. Robot untouched today.

## 2026-07-04 ~14:20 ICT — First GROUND session (tethered). Stand-hold works; tether bears weight.
- Ran stand-hold (no policy) on the GROUND, fully tethered, human on damping. 3 clean cycles.
- SAFETY SPINE PROVEN ON GROUND: motion-service release, firm move-to-default ramp, indefinite
  hold, and damp-on-signal all worked; robot ended SOFT on every stop (SIGTERM->damp->os._exit).
  Lesson: signal the PYTHON pid directly (exec/-u), not the bash wrapper, or the child orphans.
- FINDING (important): holding pose is GAIN-INDEPENDENT. 2.0x and 3.0x approach gains gave a
  near-identical stance (hip_pitch sag ~22deg, ankle_pitch ~-40 vs cmd -21, knees on target).
  A PD deflection would scale ~1/kp; it didn't move -> the legs rest against an EXTERNAL
  constraint, i.e. the TETHER/HARNESS is bearing weight and holding a suspended crouch. The
  feet are not truly load-bearing => this was NOT a real weight-bearing stand test. Torso stayed
  vertical (waist_pitch ~0.4), pose rock-steady, no oscillation/faults.
- IMPLICATION: to validate a real stand, slacken the tether so the feet load — deliberately,
  higher risk (first real weight-bearing, no e-stop, and NO working ground dance policy for
  dynamic recovery). Gains are not the lever.
- NEXT (sim, no robot): fix the ground retrain (reference re-anchoring + startup ee-termination
  grace), retrain, re-gate nominal>=0.95 full-length. THEN a real weight-bearing stand test.

## 2026-07-04 ~14:35 ICT — Weight-bearing stand-hold: legs sag under load (EXPECTED, not a fault).
- Slackened tether so feet bear weight; ran stand-hold 8s ramp @2.0x. Feet loading engaged
  (pose became load-dependent, unlike the tether-borne case). Under load the legs SETTLED
  (2 identical reads) into a stable deep squat: knee 55/66, hip_pitch +12/+6, ankle -50
  (cmd knee 38, hip -18, ankle -21). Torso vertical (waist_pitch 0.4). Asymmetric (R knee 66 vs L 55).
- INTERPRETATION (important, avoid misfiling): a static PD-to-default is a spring to a pose,
  NOT a balance controller. Sag under body weight is EXPECTED once the onboard balancer is
  released. The robot stands fine on its OWN onboard controller (normal standby). Weight-bearing
  in the dance comes from the ACTIVELY-BALANCING policy, not from stand-hold. Stand-hold was the
  pre-check and did its job: safety spine proven, feet-loading confirmed, static-PD gap measured.
- CONCLUSION: today's real blocker is unchanged — the ground POLICY retrain failed. No robot dance
  today. Next = fix retrain in sim (re-anchoring + startup ee-termination grace), retrain, re-gate.
- Robot damped/soft. 4 clean stop cycles total this session; always ended soft.

## 2026-07-04 ~14:45 ICT — Ground retrain FIX WORKS (v2 bootstrapping). Sim only.
- Root cause was an exploration CLIFF, not re-anchoring (which mjlab already does every step)
  nor bad tracking (torso anchor err was 0.076m in the failed run). With the 2 obs terms dropped,
  the untrained policy died at step ~5 on ee_body_pos (wrist/ankle HEIGHT err > 0.25m) and never
  got a learning signal.
- FIX: loosen ONLY that threshold 0.25->0.6 so it survives the transient and bootstraps. Launched
  job train-thriller-ground-v2 (No-State-Estimation, motion thriller_deploy.npz, action_rate_l2
  -0.2, 3000 iters, ee-thresh 0.6). CLI override only, no mjlab source edits.
- RESULT so far: mean episode length 5 -> 24-26 and climbing by iter 180; ee_body_pos term rate
  1.0 -> 0.0-0.5. It is LEARNING. Training out ~30 min.
- NEXT: on completion, export 154-dim ONNX (export_policy_ground: swap task to No-State-Estimation)
  + held-out gate on the SAME task. Eval at STRICT 0.25 first (honest bar) AND at 0.6 (training-
  matched); report ACHIEVED mpkpe + ankle-height err separately. If ankle tracking is loose,
  do v3 with SPLIT thresholds (tight ankle for safety, loose wrist). SIM_READY only if nominal>=0.95.

## 2026-07-04 ~15:05 ICT — v2 mid-training: wall MOVED to balance/anchor. Verdict pending eval.
- v2 past the ee cliff, but by iter ~2200 the binding wall is the ANCHOR terminations
  (anchor_ori torso tilt >0.8rad, error_anchor_rot ~1.0; anchor_pos torso height >0.25m).
  time_out pinned at 0; ep length regressed 24->8. Read: it can't hold the torso upright/at
  height without base_lin_vel obs — the fundamental estimator-free balance cost. BUT mid-train
  metrics are adaptive-sampler-pessimistic; the frame-0 held-out eval is the real test.
- Built this session (sim/code only): heldout_eval_ground.py (ankle/wrist height breakdown,
  configurable ee-threshold), export_policy_ground.py, autopilot_v2.sh (on box, auto export+gate
  @0.25 & @0.6 -> RESULT), laptop watcher (pulls to data/policies/thriller_ground_v2/ + notifies).
- Updated docs/GROUND_TETHERED_RUNBOOK.md (Stage-A sag is EXPECTED; revised go-criterion) and
  wrote docs/ground_retrain_next.md (v3 decision tree branched on v2 eval: ship / split-ee /
  balance-curriculum / reconsider-estimator).
- NOT declaring v2 a win; earlier optimism was premature. Waiting on RESULT.txt.

## 2026-07-04 ~15:30 ICT — PIVOTAL: robot publishes a base-state ESTIMATE (rt/odommodestate).
- v2 verdict: SIM_READY=NO. 0/256 survival at both 0.25 and 0.6 ee bounds; mpkpe 0.115m,
  ankle 0.128m (tracks tight but FALLS — the balance wall, as diagnosed). Estimator-free
  Thriller does not balance without velocity obs. Artifacts in data/policies/thriller_ground_v2/.
- READ-ONLY robot probe (no motion): LowState has NO base vel/pos (only imu quat/gyro/accel +
  joint q/dq). BUT topic **rt/odommodestate** (SportModeState_) IS LIVE at ~184Hz with:
  position[3], velocity[3], body height. At rest: vel clean ~0, pos steady (~1mm/4s drift),
  h~0.566m. This is a real onboard EKF — exactly the two obs terms the estimator-free path
  dropped (base_lin_vel + torso position for motion_anchor_pos_b).
- IMPLICATION / PIVOT: instead of fighting estimator-free training, feed the PROVEN full-obs
  GANTRY policy (100% in sim) REAL base_lin_vel + anchor position on the GROUND from
  rt/odommodestate. Sidesteps the balance wall entirely.
- OPEN VALIDATIONS (not all doable read-only): (1) does odom stay published + accurate when we
  RELEASE the motion service for low-level control? (it publishes now, and we released it during
  stand-hold, so likely persistent — confirm during a supervised run). (2) velocity is reliable;
  XY position DRIFTS over 49s -> handle by RE-ANCHORING at deploy (reset position frame at start,
  matching training); height/Z directly usable. (3) frame mapping odom(world)->body via imu quat.
  (4) whose estimator is it (onboard vs colleague's stack) + persistence.
- NEXT (recommended): wire deploy_runtime to consume rt/odommodestate -> build HONEST 160-dim obs
  -> sim-check -> tethered-first ground bring-up of the PROVEN gantry policy.

## 2026-07-04 ~16:10 ICT — Odometry-fed ground obs path BUILT + offline-validated (sim/code only).
- deploy_runtime.py: added odom_subscriber/read_odom (rt/odommodestate), build_obs_odom (HONEST
  base_lin_vel = R.T@v_world + re-anchored motion_anchor_pos_b = R.T@(ref_disp-robot_disp)), and
  mode `ground-run-odom` running the PROVEN gantry policy with the estimate (GROUND cap, --max-secs,
  full safety spine, NO-GO if odom absent, always-soft).
- Tests: tests/test_deploy_odom.py (6) — perfect-tracking->anchor=0, base_lin_vel body-frame, reduces
  to gantry fake when static, dim/finite. Full deploy suite 20/20 green.
- Offline pipeline smoke (tools/sim_ground_odom.py): real ONNX policy over full 51.8s motion with
  odom-from-reference (+noise): 0 non-finite, base_lin_vel mean 0.25/max 1.26 m/s (vs fake 0), robust
  to noise. FINDING: gantry policy's real action range ~8.5 -> GROUND_MAX_ACTION=6 false-trips ~4%;
  ground-run-odom needs GROUND_MAX_ACTION=10 (documented in runbook Stage B-ODOM).
- Runbook: added Stage B-ODOM (preferred path) w/ the 2 supervised validations (odom survives
  motion-service release; velocity-frame sway test) + re-anchoring notes.
- OPEN (needs tethered bring-up, no robot while away): confirm odom persists under our control +
  velocity-field frame. Estimator-free v3 deprioritized (odom path is primary, no GPU needed).

## 2026-07-04 ~16:35 ICT — Odometry obs path VALIDATED against the simulator (frame math exact).
- Sim cross-check (gantry policy rollout, 25 ticks): build_obs_odom's two terms vs mjlab's
  authoritative obs. motion_anchor_pos_b MAX ERR = 0.000000. base_lin_vel MAX ERR = 0.000000
  when rotated by the ROOT(pelvis) quat (initial 0.068 was torso-vs-root body choice, not a bug).
  => R.T@(ref-rob) and R.T@v_world are EXACTLY mjlab's frame conventions.
- Robustness margins (training noise on the two terms): base_lin_vel ±0.5 m/s, motion_anchor_pos_b
  ±0.25 m. HUGE. => (a) diff-velocity noise (~0.05) is 10x under tolerance — ODOM_VEL_SOURCE=diff is
  adequate, field optional; (b) the pelvis-vs-torso approximation (odom=pelvis, mjlab anchor=torso;
  one IMU quat for both terms) is well within ±0.25 — same approximation the WORKING gantry deploy used.
- base_lin_vel obs = imu_lin_vel sensor = root_link_lin_vel_b (pelvis, body frame). On robot: odom
  pelvis velocity + IMU pelvis quat -> exact. Confirms odom (pelvis) is the right source.
- NET: the odometry-fed path is frame-correct and noise-robust by construction. Remaining unknowns are
  purely hardware (does odom persist under our low-level control; does odom vel field frame = body) and
  resolve on the first tethered bring-up. Offline validation is as complete as it can be.

## 2026-07-04 ~17:15 ICT — Onboard odom FREEZES on release; built KINEMATIC (leg) odometry instead.
- First ground-run-odom on the tether: frozen-estimate GUARD FIRED — rt/odommodestate stamp froze
  the moment the motion service was released for low-level control; robot damped safe. So the onboard
  estimate and full-body control are mutually exclusive. Safety design validated on hardware.
- FIX: pipeline/leg_odometry.py (LegOdometry) — base_lin_vel + torso height from LEG kinematics
  (MuJoCo FK/Jacobian on the menagerie g1.xml, planted-foot assumption, contact-weighted blend,
  ±2.5 m/s clip). Service-INDEPENDENT (only LowState q/dq + IMU).
- VALIDATED offline vs reference ground truth (tools/validate_leg_odom.py): base_lin_vel within the
  policy's ±0.5 m/s trained band on 97.8% of frames (mean err 0.13). End-to-end policy smoke
  (tools/sim_ground_legodom.py): finite, bounded actions over the full 51.8s. Tests: tests/
  test_leg_odometry.py (4) + full suite green (23).
- deploy_runtime: new mode `ground-run-legodom` (PROVEN gantry policy + leg-odom obs: real base_lin_vel,
  anchor XY=tracking-assumption/drift-free, anchor Z=real leg-odom height feedback). Full safety spine,
  GROUND_MAX_ACTION=10, --max-secs. Runbook: Stage B-LEGODOM is now PREFERRED; B-ODOM superseded.
- READY: tethered try of ground-run-legodom (human-supervised). Stage 0 read still valid to pre-check.

## 2026-07-04 ~18:00 ICT — Motion-service left RELEASED stranded the robot; added auto-restore.
- ROOT CAUSE of "nothing happens": our runs call MotionSwitcherClient.ReleaseMode and never
  restored it. CheckMode confirmed name='' (released). SelectMode('ai') restored the DDS service
  (CheckMode->'ai') BUT the remote/app still would not pair -> the Bluetooth/pairing failure is
  ONBOARD-PC level, separate from the DDS service. Recovery = REBOOT the robot (coordinate w/ the
  colleague's Docker). Robot safe/limp throughout; no fall.
- FIX (committed): deploy_runtime _restore_motion_service() re-SelectMode(RESTORE_MOTION_MODE, default
  'ai') inside _finalize_and_exit, AFTER the damp — so EVERY exit path (normal/Ctrl-C/SIGTERM/crash)
  hands control back to onboard and the remote can pair. Best-effort, never blocks the (already-soft)
  exit. env RESTORE_MOTION_MODE='' to disable.
- STATUS: leg-odom ground path proved a BALANCED segment before the strand (3s ok per user). Once the
  robot is rebooted + remote pairs, resume the staged runs (5s->10s->20s->full); the runtime now cleans
  up the motion service each run so this can't recur.

## 2026-07-04 ~18:40 ICT — Recovered after reboot; control CONFIRMED; auto-restore validated on HW.
- Reboot fixed the onboard/Bluetooth strand (remote+app+damping all working). Robot back under 'ai'.
- BUG found+fixed: _release_motion_service crashed on a transient None from CheckMode right after Init
  (crashed BEFORE taking control -> "move-to-default did nothing" #2). Now retries on None (15x), robust.
- Instrumented move-to-default (post-fix): "service released -> lowcmd accepted -> at default pose".
  Mid-ramp read proved commands LAND (legs drove 70->50deg toward default, arms tracked). Remaining leg
  gap = known load-bearing sag. Control fully restored.
- AUTO-RESTORE VALIDATED ON HARDWARE: after the run, CheckMode='ai' (the on-exit SelectMode fired) —
  robot handed back to onboard control, remote stays reachable. The strand can't recur.
- READY: resume ground-run-legodom staged runs (5s->10s->20s->full). Each run now self-restores the
  service on exit.

## 2026-07-04 ~19:15 ICT — BREAKTHROUGH: robot STANDS + BALANCES + dances on the ground (tethered).
- ground-run-legodom with BOOSTED LEG GAINS worked. User confirmed: "it did stand and balance."
- WINNING CONFIG: APPROACH_KP_SCALE=3.0 (stand-up in move-to-default) + GROUND_LEG_KP_SCALE=2.0
  (legs hold standing during policy; arms at trained gains) + GROUND_MAX_ACTION=10.
- Telemetry (policy window): avg knee 36deg (target 38 = STANDING) vs the prior 50-70deg CROUCH;
  hips ~-11/-13 (target -18), ankles ~-29 (target -21). Legs held a standing config while the arms
  danced (shoulder/elbow motion 40deg std). No oscillation/fault; clean completion + auto-restore.
- ROOT CAUSE (resolved): the trained gains stand the robot in SIM but are too soft on real HW under
  load -> it sagged into a crouch and danced from there. Boosting ONLY the leg kp/kd (kept overdamped,
  torque clamped) let the legs bear weight and stand. Diagnosis via joint telemetry + user's eyes.
- NEXT: confirm it's GENUINELY weight-bearing (slacken tether at 3s) then extend 5s->10s->20s->full.

## 2026-07-04 ~19:45 ICT — LATERAL FALLING was a HARDWARE FAULT (stuck right hand). Fixed -> balances.
- Root cause of the sideways falls: the RIGHT HAND was mis-installed, locked at a wrong angle ->
  CoM pulled off-center -> persistent right lean -> lateral fall. NOT a policy robustness gap
  (my retrain call was premature). User re-installed the hand correctly.
- Post-fix telemetry (3s, config APPROACH_KP_SCALE=3.0 + GROUND_LEG_KP_SCALE=2.0 sagittal-only):
  L_knee 35 / R_knee 36 (asymmetry -1deg, was +8-9), torso mean roll +1.1deg (level), swing 6.6deg.
  User: "it looks balanced now." Symmetry+level are tether-independent -> genuine CoM fix.
- SHOW WORKFLOW (user, important): during the show the robot is WALKED up to the stage via the
  remote and is in RUN/WALK mode (onboard, standing+balanced) BEFORE the dance is initiated. So our
  deploy hands off from an ALREADY-STANDING pose (not from limp/crouch) -> easier + more robust.
- NEXT: extend duration (5s->full) now balance works; then test the real walk-mode->dance handoff.

## 2026-07-04 ~20:15 ICT — 30s: stable on average but ONE sudden lateral "acrobatic" move (tether caught); motors overheating -> robot cooldown.
- Progression today (post hand-fix, config APPROACH_KP_SCALE=3.0 + GROUND_LEG_KP_SCALE=2.0 sagittal-only,
  GROUND_MAX_ACTION=10): 3s/5s/10s all CLEAN + balanced (roll swing <5deg, no drift). Walk-mode handoff
  worked end-to-end. 30s: by-segment roll flat (-1.7..+1.8, NO drift) BUT one transient to roll -23.7deg
  = a SUDDEN anomalous sideways "acrobatic" move; TETHER caught it (not self-recovered).
- DIAGNOSIS (not drift/margin): a single bad policy ACTION or bad OBS at one instant -> likely findable
  in data, not fundamental. Suspects: (a) action spike under the cap-10 but large lateral; (b) leg-odom
  velocity spike (swing-phase, clipped to 2.5 but still) feeding a bad obs term; (c) a specific hard
  Thriller move. NEXT (offline, no robot): replay policy over full motion, flag the tick(s) with large
  lateral/roll-driving actions or obs outliers; cross-ref the timestamp of the -23.7 event.
- MOTORS OVERHEATING: the boosted leg gains hold high continuous torque to bear weight -> hip/knee/ankle
  heat over 30s. Show implication: trim gains (walk-mode standing start needs less stand-up authority) or
  plan cooldowns. User shut robot down to cool.
- STATUS: ground dance WORKS + balances to 30s tethered; blockers before full = (1) the one anomalous move,
  (2) motor thermal at these gains. Both addressable.

## 2026-07-04 ~20:45 ICT — Root-caused the "acrobatic" move (leg-odom velocity SPIKE) + fixed it (offline).
- Offline replay of the policy over the full motion: the biggest action JUMP (3.32 @ t=17.8s, inside the
  30s window) COINCIDED with a leg-odom base_lin_vel SPIKE (2.98 m/s vs true max 1.21). Mechanism: swing-
  phase foot not cleanly planted -> kinematic velocity spike -> bad base_lin_vel obs -> policy overreacts
  = sudden lateral move. Confirmed: 58 frames (2.2%) had >0.5 m/s vel error; est max 2.98 vs true 1.21.
- FIX (pipeline/leg_odometry.py): temporal smoothing — per-tick rate limit (VEL_MAX_STEP 0.30 m/s) + EMA
  (alpha 0.35), with reset_filter() called at run start (wired into ground-run-legodom). Results: vel err
  >1.0 m/s 12->0, >2.0 4->0; est max 2.98->1.14 (true 1.21); within +-0.5 band 97.8%->99.0%; policy action
  jump max 3.32->2.48 and the 17.8s spike eliminated. (Remaining ~3 jumps>2.0 are genuine fast Thriller
  moves, not obs artifacts.) Tests: +2 smoother tests, suite green.
- STATUS: acrobatic-move cause fixed in code (unverified on robot — robot cooling). When motors cool:
  re-run 30s to confirm the anomaly is gone, then push to full. STILL OPEN: motor thermal at boosted gains.

## 2026-07-04 ~21:30 ICT — Root cause CONFIRMED: leg-odom degrades during STEPPING -> whole-body brace.
- User saw at ~14-16s: "strange acrobatic position (legs AND arms) arched and stiff, then attempt to
  dance, unstable, better toward end." => NOT a topple (balance); a whole-body BRACE = policy reacting
  to a BAD OBS, then recovering. Points to estimator, not balance.
- OFFLINE PROOF: leg-odom error by section — 0-10s (danced CLEAN): vel_err 0.04/height 0.08m.
  13-17s (the arched brace): vel_err 0.15/height 0.18m = WORST in the dance, in BOTH channels. Aligns
  with feet lifting (stepping, foot Z 0.12-0.21m). Planted-foot assumption breaks during a step ->
  velocity AND height estimate degrade -> policy braces -> recovers when feet re-plant.
- Smoothing fix (prev) only killed single-frame SPIKES; this is SUSTAINED (~0.4s) degradation -> smoothing
  can't fix it. Re-run 30s still spiked to roll -28.9deg @ ~14-16s (consistent timing = choreography step).
- Note: offline uses REFERENCE (perfect-tracking) joints; real hardware joint tracking is noisier, so the
  real stepping error is likely WORSE than the 0.15/0.18 measured.
- FIX PLAN (offline, no robot): FUSED estimator — integrate IMU accel to carry base vel/height THROUGH the
  step (when kinematics is blind), correct with kinematics when a foot is solidly planted (complementary
  filter / the piece the onboard EKF did before it froze on service release). Keeps the proven policy.
  Alt: retrain with heavier estimator-noise/disturbance randomization (slower, GPU). Motors OK.

## 2026-07-04 ~22:15 ICT — AUDIT + DISAMBIGUATION: estimator was a RED HERRING; PIVOT to arm-over-balance.
- Independent first-principles audit + a decisive offline disambiguation test resolved the 14-16s brace.
- DISAMBIGUATION (perfect vs leg-odom vs fused obs -> same policy, measure |action-perfect|):
  * estimator corrupts the action EQUALLY in clean (0.35) and stepping (0.36) windows -> estimator is
    NOT concentrated at the failure. In clean 0-10s |action-perfect| max was 2.76 yet robot danced fine;
    at the failing 14-16s it's only 0.79. So estimator error does NOT cause the brace.
  * with a PERFECT estimate the policy STILL commands violent moves at 14-16s (action jump 0.81 clean ->
    2.05 stepping). => the stepping section is CHOREOGRAPHY-hard/dynamic, not estimator-hard.
  CONCLUSION: NO estimator (leg-odom/fused/perfect) fixes 14-16s. The fused-estimator polish was
  optimizing the wrong variable (audit called this). Fusion stays in tree (offline-validated) but is NOT
  the fix.
- STRUCTURAL BLOCKERS of the full-body-low-level path (audit): (1) THERMAL — boosted leg gains needed to
  stand overheat at 30s; show is 2-3min -> likely show-killer. (2) hard stepping moves it can't execute on
  ground. (3) faked XY anchor (robot_disp XY==0) = uncorrected lateral drift over a long dance.
- HIGHEST-LEVERAGE PIVOT (audit #1, now empirically supported): let ONBOARD control BALANCE+LEGS (it
  walks the robot up already) and drive the ARM dance via low-level. Thriller is ~arms. This is ALREADY
  PROVEN on THIS robot: ~/robot start_teleop_armsonly.sh does arm control (--arm=G1_29, g1_arm7 arm_sdk
  weight-blend) WHILE onboard balances. Eliminates estimator+thermal+stepping+XY at once.
- PROCESS FIX: prove the root cause with a cheap decisive test (disambiguation) BEFORE building the fix.
  I built fusion before proving the diagnosis; the audit+disambiguation corrected course.
- DECISION PENDING (user): pivot to arm-dance-over-onboard-balance (reliable, loses leg choreo) vs keep
  full-body (fragile, thermal-capped). Recommend PIVOT for a show-grade product.

## 2026-07-04 ~22:45 ICT — THERMAL: real concern at 2x gains (motor 46->80C in ~1-2min). Harness bug owned.
- Path B chosen. Thermal test (stand-hold @ APPROACH_KP_SCALE=2.0, feet bearing weight): a leg motor went
  46C -> 80C (Unitree WARNING temp) in ~1-2 min from near-cool. Fast — NOT hours-of-accumulation. Thermal
  IS a genuine concern at these gains (audit was right); user's "not an issue" is too optimistic at 2x.
- HARNESS BUG (owned): the monitor read a STALE DDS backlog (all 46C) instead of latest state -> went blind
  to the real rising temp -> the 68C auto-abort never fired -> hold ran hot+unattended to 80C before a
  fresh read caught it. FIXED: monitor now drains to latest msg, samples 1/s, aborts at 62C real-time.
  Robot damped safe (dropped to 74C on unload). Lesson: DDS subscribers read a queue; always drain to latest.
- CAVEATS softening it: (1) stand-hold boosts ALL joints 2x incl. hip_roll (ran hottest 74C); the DANCE only
  boosts SAGITTAL legs 2x (hip_roll stays 1x) -> real dance thermal is LESS than this test. (2) but sagittal
  ankle_pitch (boosted in dance) hit 80C -> still real.
- PLAN for B: (1) let motors COOL now (74C, no runs). (2) find MIN leg gain that still stands (walk-mode
  standing start removes the stand-up-boost need) -> lower gain = far less heat, attacks the thermal root.
  (3) re-test thermal at that gain with the fixed monitor. THEN address the 14-16s stepping choreography.

## 2026-07-04 ~23:05 ICT — THERMAL WALL confirmed at the ANKLE; gain reduction does NOT fix it.
- Clean test (fixed monitor, real-time, aborted 62C): stand-hold @1.5x, RIGHT ANKLE PITCH 47->62C in 34s,
  rate 22.5 C/min -> 94C@2min, 117C@3min (fault ~90C). Reducing 2.0->1.5 barely helped: ankle holds ~20Nm
  CONTINUOUS, and that torque is set by WEIGHT+POSTURE, not gain -> lower gain can't fix it.
- ROOT: our policy commands a pose that loads the ankle ~20Nm continuously (CoM likely well forward of the
  ankle). The ONBOARD controller stands indefinitely without overheating because it keeps ankle torque near
  zero (efficient balance strategy / CoM over feet). This is the mechanism behind the audit's #1.
- IMPLICATION: full-body-low-level path B has a FUNDAMENTAL thermal wall at the ankle_pitch motor.
  Only lever left for B: shift standing POSTURE to center CoM over feet -> drop ankle torque (speculative,
  real tuning). Otherwise B is thermally non-viable for a 2-3 min (let alone 10-15 min) show.
- Reinforces PIVOT (audit #1): onboard balances (efficient, no ankle cook) + policy drives ARMS.
- Robot damped safe. Motors will need cooldown again before any further test.

## 2026-07-04 ~23:40 ICT — ROOT CAUSE FOUND + UNCOMPROMISED FIX: gravity-comp feedforward.
- Diagnostic chain: (1) deploy gains EXACTLY match training (ankle kp29=28.5, knee99, hip40) — NOT a gain
  bug. (2) sim default pose needs ~0.2Nm ankle torque (CoM centered; hands add only 1.1kg, negligible).
  (3) yet real robot holds ~20Nm at the ankle. => the gap is that the SIM trains with a POSITION ACTUATOR
  that implicitly provides gravity-hold torque; our deploy sent PURE PD (tau=0) -> legs sag -> we
  gain-boosted -> boosted PD fighting the residual sag error burned 20Nm at the ankle = the thermal wall.
- FIX (root, no retrain, keeps full-body dance): send FEEDFORWARD gravity-compensation torque (LowCmd tau
  field, was 0) computed via MuJoCo inverse dynamics at the COMMANDED pose in the current torso frame (IMU).
  Then legs hold pose at TRAINED gains, no boost. Offline: gravity-comp ankle FF = 0.2Nm max over the whole
  dance (vs 20Nm burned now), all leg joints << effort limits. 100x ankle-torque reduction => no thermal wall.
- Implemented: LegOdometry.gravity_comp(q_target,R_base); _send_cmd tau_ff (clamped to effort); wired into
  ground-run-legodom; GRAVITY_FF on by default, GROUND_LEG_KP_SCALE default back to 1.0 (no boost). +tests.
- LIKELY ALSO helps balance/stepping: no sag + no over-stiffened joints => robot tracks the policy's intended
  targets like in sim, where it works. To verify on HW (tethered) once motors cool.
- Motors were 62C; need cooldown before the hardware test of the FF fix.

## 2026-07-05 ~00:15 ICT — DECISIVE: thermal/balance/stepping are ONE sim2real gap. Answer = RETRAIN.
- Gravity-FF hardware test (5s, trained gains, no boost): ankle torque still ~15Nm mean (60-65 max), NOT
  the ~0 predicted. Two reasons: (a) my gravity_comp computed HANGING (base-supported) torques not
  standing-on-feet; (b) more fundamentally, the real ankle load ISN'T gravity.
- DECISIVE offline check — SIM ankle torque during the policy rollout: mean 0.0, max 0.3 Nm (knee 0.4).
  Real robot: ~15Nm. => 0->15Nm SIM2REAL GAP. In sim the policy keeps CoM over feet (ankle ~free); on
  real HW imperfect tracking (latency/actuator response) makes the ankle constantly balance-correct = 15Nm.
- ROOT (shared): thermal (ankle overworks only on real HW), the sag/gain-boost need, AND the stepping brace
  are ALL the same sim2real gap — the policy assumes near-perfect tracking the real robot doesn't provide.
  No deploy-side patch (estimator/gains/FF) can close a 15Nm gap. FF defaulted OFF (not the fix).
- UNCOMPROMISED ANSWER (audit #2, now empirically forced): targeted RETRAIN with real conditions modeled —
  (1) latency randomization (prime suspect), (2) actuator-response DR, (3) torque/energy reward penalty
  (learns low ankle torque -> cool by design), (4) obs noise matching leg-odom, (5) mass/gain/push DR.
  Keeps full-body dance; reuses leg-odom + deploy runtime + safety spine. GPU box alive.
- Band-aid era (estimator/smoothing/fusion/gains/FF) is over; all were deploy-time patches on a train-time
  problem. Next: author the sim2real retrain config + verify in sim (check sim ankle torque stays low AND
  survives latency/push), THEN one clean tethered HW test.

## 2026-07-05 (resumed session) — RETRAIN CONFIG AUTHORED + CRITICAL MEASUREMENT CORRECTION. Audit running.
- Resumed per HANDOVER.md. Two tracks: (a) retrain config + sim verification harness (main thread),
  (b) ultracode first-principles audit of the whole approach (workflow wf_e9b5dd2c-6a8: 8 investigator
  lenses -> adversarial verify -> synthesis; running in background).
- **CRITICAL CORRECTION — the "sim ankle 0 Nm" was a MEASUREMENT ARTIFACT.** cloud/sim_ankle.py read
  `data.actuator_force[joint_name_index]`, but actuator_force is ACTUATOR-ordered, not joint-ordered —
  it read the wrong column (cross-check at same instant: qfrc_actuator left ankle 10.1 Nm vs
  actuator_force[jn_idx] 0.004 Nm). Correct joint-space measurement (data.qfrc_actuator at resolved
  joint ids): the deployed policy in NOMINAL sim (no injected latency) uses ankle_pitch |tau|
  **mean ~6 Nm, p95 ~15 Nm, transients to the 50 Nm effort clamp**. So the real story is NOT
  "0 -> 15 Nm all-sim2real": ~6 Nm mean is INHERENT to this policy+choreography; hardware excess is
  ~2.5x, not infinite. Implications: (1) the torque/energy PENALTY is the headline retrain item (the
  policy must LEARN a low-ankle strategy it doesn't have even in sim); (2) latency DR remains supported
  by measurement — injected 40 ms constant delay: falls + ankle mean 9.9/p95 33.6 Nm, mpkpe 0.151 -> 0.315;
  (3) sim ankle 50 Nm clamp hits mean the sim ankle ALREADY saturates transiently — the real 60-65 Nm max
  reading is plausibly the same events with real-world excess.
- **Gate limitation found: heldout_eval.py never overrode episode_length_s=10.0** — the "100% held-out"
  verdicts certified only the FIRST 10 s of the dance (full-motion verification came from the separate
  in-engine eval). New harness runs the FULL motion.
- **BUILT (committed): cloud/sim2real_task.py** — registers task `Mjlab-Tracking-Flat-Unitree-G1-Sim2Real`
  implementing the 5-item plan with native mjlab features (no source edits): cmd-bus delay 0-8 physics
  steps (0-40 ms, hold_prob 0.8) via ActuatorCfg fused DelayBuffer; obs delay 0-1 control steps on the 6
  measured actor terms; pd_gains scale 0.85-1.15; effort_limits 0.80-1.00; joint frictionloss 0-0.4 Nm;
  armature scale 0.9-1.4 (ankle/waist 4-bar armature is a documented guess in g1_constants.py);
  base_com x widened to ±5 cm; torso mass 0.95-1.15; hand payload +0-0.6 kg/wrist; encoder_bias ±0.02;
  rewards: joint_torques_l2 -2e-5 (all) + custom ankle_torque_l2 -4e-4 (qfrc-based, order-safe) +
  action_rate_l2 -0.2 (a2's winning delta). Obs stays 160-dim -> deploy runtime unchanged.
  **Deep-copies the robot cfg (G1_ARTICULATION is module-level shared — mutating it would contaminate
  the stock task in-process; verified stock stays clean).** + cloud/train_sim2real.py (wrapper entry).
- **BUILT: cloud/sim_gap_check.py** — full-motion, held-out-seed eval across 7 conditions (nominal /
  noise / 10-20-40 ms constant cmd delay / delay+push), measuring survival + leg-joint |tau|
  (qfrc_actuator) + mpkpe + the actuator_force cross-check. GATE for the retrained policy: survival
  >=99% AND ankle mean<=5 Nm AND p95<=15 Nm under worst condition, mpkpe<=0.25 nominal. Smoke-tested on
  box (quick mode); FULL baseline on the DEPLOYED a2 model_1500 checkpoint running detached on the box
  (reports/sim_gap_check_a2_1500_full.json, logs/sim_gap_check_a2_full.log).
- **TRAINING DELIBERATELY NOT LAUNCHED YET** — waiting on (1) full baseline numbers, (2) audit synthesis
  (may re-rank recipe items, e.g. static CoM/system-ID emphasis vs latency). Both running; recipe weights
  (ankle penalty -4e-4, delay ranges) will be finalized against them, then train (~3-4k iters, ~$1-2).
- Robot untouched (user at work, no motion authorized). Box GPU otherwise idle; budget ~182k/1.5M VND.
- **External robot-watching webcam CONNECTED + verified** (user): HP Webcam HD 4310 = /dev/video4
  (video0-3 = laptop built-in). pipewire grabs it — reclaim before capture:
  `systemctl --user restart pipewire.service pipewire.socket`, then
  `ffmpeg -f v4l2 -video_size 1280x720 -i /dev/video4 -frames:v 1 …`. First frame: G1 on gantry
  line inside fenced area, powered, standby pose, full body in frame. Scene dim — more light
  recommended for observing fast motion.

## 2026-07-05 (autonomous window, user away) — AUDIT VERDICT: CRITICAL-MISTAKE-FOUND. Deploy bugs fixed, recipe v2 training.
- **ULTRACODE AUDIT COMPLETE** (54 agents, 84 findings, 18 adversarially verified;
  full memo: docs/first_principles_audit.md — READ IT; HANDOVER.md rewritten to match).
  Verdict: the "0 -> 15 Nm one-gap, prime-suspect-latency" conclusion was built on a
  measurement artifact (sim_ankle.py read the LEFT WRIST, not the ankle — actuator-ordered
  array indexed with joint-tree indices; independently confirmed by physics: standing at the
  trained pose REQUIRES ~5.25 Nm/ankle, CoM is +3.21 cm of the ankle axis). Corrected picture:
  policy is ankle-hungry even in clean sim (~6-8 Nm mean, saturates the 50 Nm clamp); real
  ~15 Nm is mostly PD-static sag (kp*err = 28.5*0.506 = 14.4 Nm) + inherent choreography floor;
  honest hardware excess ~2x; trained ankle stiffness 57 Nm/rad < gravity's 202 Nm/rad, so pure
  PD topples — the POLICY is the balance controller. Latency matters only dynamically
  (measured cliff: 10 ms fine, 20 ms halves survival, 40 ms kills). "Gravity-FF ruled out" was
  FALSE (test sent ~zero ankle FF — hanging-model inverse dynamics). Thermal at ~8-10 Nm RMS is
  show-viable (gate on RMS <= 12 Nm, not on reaching 0).
- **DEPLOY BUGS FOUND + FIXED (free wins, committed, 242 tests green):**
  (a) YAW RE-ANCHOR: reference npz world yaw (t=0 = 90.3 deg) was never aligned to the IMU
  world frame -> anchor_ori permanently OOD unless the robot booted facing the npz heading.
  Offline ONNX experiment (tools/obs_frame_sensitivity.py): at 90.3 deg offset the action
  corruption (mean |da| 1.16) ~= the ENTIRE action signal (RMS 1.29); even 15 deg = 30%.
  Every past ground run was corrupted to some degree. Fix: Reference.align_yaw at policy start
  (all modes, YAW_ALIGN=1 default) — verified exactly heading-invariant.
  (b) TORSO ANCHOR: training anchors on torso_link; deploy used the pelvis IMU quat. Fixed via
  waist FK (yaw z, roll x, pitch y — validated against MuJoCo FK); measured effect 0.22 mean /
  4.5 max |da| during the dance's 30 deg waist moves. TORSO_ANCHOR=1 default.
  (c) TELEMETRY: every motion run now records q/dq/tau_est/temps/IMU/action/target ->
  data/telemetry/*.npz (flushed AFTER damping; the 15 Nm number had NO committed code path).
  (d) read_state: Read() takes SECONDS not ms (failure paths hung 1000x too long) + bounded
  drain-to-latest. (e) action caps to measured need: MAX_ACTION 8->12, GROUND_MAX_ACTION 6->10.
- **CHOREOGRAPHY EDIT DEFERRED (justified deviation from audit item 6):** quasi-static FK scan
  of the full dance shows high-lean/step segments are NOT localized to 14-16 s — 16 segments
  >30 Nm total ankle demand, worst at 43-47 s (58 Nm). A blind proxy-driven edit risks the whole
  choreography. Instead: per-section stats added to the gate; the TRAINED policy's per-section
  failures will drive a targeted, music-sync-preserving edit if needed (attempt 2).
- **RECIPE v2 (cloud/sim2real_task.py, re-ranked per audit):** torque penalty headline
  (torques_l2 -2e-5 + ankle_torque_l2 -4e-4 qfrc-based); system-ID mass (hands +0.40-0.70 kg,
  torso scale 1.00-1.12 — never lighter than model, CoM x +-5 cm, ankle zero-offset +-0.08 rad);
  actuator DR modest (gains +-15%, effort 0.8-1.0, friction 0-0.4, armature 0.9-1.4);
  obs DYNAMICS matching leg-odom (custom base_lin_vel term: lag 30-80 ms + slew 0.30 + episodic
  stance-break bias +-0.15; obs delay 0-20 ms); latency DR demoted to 0-20 ms (40 ms eval-only);
  episodes 10->20 s (stance exposure); action_rate_l2 -0.2. Smoke-tested on box (obs 160, events
  live, custom term instantiates). Trains on thriller_deploy.npz (ramped deployable — removes a
  train/deploy mismatch, adds standing exposure; a1/a2 trained on thriller_show).
- **GATE v2 (cloud/sim_gap_check.py):** survival >=99% nominal / >=95% worst-injected; ankle
  mean <=6/8 Nm, p95 <=15/20, RMS <=12 Nm (thermal projection: 22.5*(RMS/20)^2 <= ~8 C/min);
  mpkpe <=0.31 (parity with deployed-a2 baseline 0.307 on THIS harness); per-section ankle
  stats + falls (0-10 / 13-17.5 / 25-36 / 40-49.5 s + worst-5s RMS window). Baseline a2 numbers
  in reports/sim_gap_check_a2_1500_full.json (nominal 127/128, ankle 7.7 mean; delay40 0/128).
- **TRAINING LAUNCHED: train-thriller-s2r** (box, tmux via run_job.sh; 4096 envs, 5000-iter cap,
  ~1.1-1.3 s/it, ETA ~1.8 h, W&B run auto). **s2r-autopilot** job waits for it -> exports ONNX
  (last + mid ckpt if needed) -> runs the v2 gate -> writes exports/thriller_s2r/RESULT.txt with
  verdict + next steps. Nothing auto-stages to deploy dirs.
- **STRATEGY (audit §5, user decision pending):** recommend HYBRID — lock arm-dance-over-onboard
  (proven teleop arm_sdk path) as the bookable show baseline (P~0.85 within 1-2 sessions), run
  the corrected full-body retrain as the premium act (P~0.5-0.65 to show-bar with the pre-GPU
  program done). The pivot silently vanished from the plan of record; reinstated for discussion.
- **NEXT ROBOT SESSION (queued, needs human):** audit experiments #4-6 — (4) DDS obs-staleness
  measurement (read-only), (5) Stage-0 onboard-stand capture (tau_est/temps 2-3 min in normal
  standby — calibrates tau semantics + the assumed "onboard ~0 Nm"), (6) one instrumented
  tethered rerun (telemetry now automatic) + slack/taut A/B + ankle-bias sweep. Also weigh the
  robot as-deployed if a scale is available (~35 kg assumed).
- Process rule added to CLAUDE.md: no DECISIVE label without independent cross-check or
  replication; measurement scripts + raw outputs must be committed.
- Robot untouched all session (rule held: no human present = no motion, not even reads).
  Budget: ~185k/1.5M VND; retrain ~45k more.

## 2026-07-04 (arm-dance build) — ARM-DANCE-OVER-ONBOARD-BALANCE RUNTIME BUILT (offline-verified).
- **pipeline/arm_dance_runtime.py** — the bookable-show-baseline runtime: streams the
  dance's 14 ARM joints (DDS 15..28, mapped BY NAME from policy_meta joint_order) over
  Unitree's **rt/arm_sdk** weight-blend while onboard balance STAYS ACTIVE (never
  ReleaseMode, never rt/lowcmd — pinned by test). Weight = motor_cmd[29].q (kNotUsedJoint0,
  0=onboard owns arms, 1=sdk owns). Sequence: weight 0→1 (2 s, holding current pose) →
  cosine approach to frame 0 (2 s) → dance 1:1 @ 50 Hz wall-clock (music guidance
  unchanged: frame0 + 4.0 s) → cosine return → weight 1→0. EVERY exit path (Ctrl-C/
  SIGTERM/crash/normal) ramps weight→0 (deploy_runtime damp-on-exit pattern). Gates:
  --i-will-watch-the-robot + CONFIRMED_BY_HUMAN=alois + --max-secs (0=full needs
  ARM_FULL_RUN=1). Telemetry reused. Gains: default = policy_meta arm gains ×
  ARM_KP_SCALE (soft first contact; WILL sag), ARM_GAINS=teleop preset = hardware-proven
  kp 80/40 kd 3/1.5 for show quality.
- **Recon (citations in docs/ARM_DANCE_DESIGN.md):** interface confirmed in
  ~/robot/xr_teleoperate robot_arm.py (motion_mode branch) + official
  g1_arm7_sdk_dds_example.py (50 Hz, kp60/kd1.5, arms+waist commandable). HONESTY NOTE:
  start_teleop_armsonly.sh ran WITHOUT --motion, i.e. the daily teleop used the DEBUG
  path (ReleaseMode + rt/lowcmd) — arm gains/DDS plumbing are hardware-proven, the
  arm_sdk weight path itself is NOT yet exercised on this robot → first 5 s supervised
  test verifies it (open question #1 in the doc).
- **Verified offline:** tests/test_arm_dance.py (25 tests: mapping can't touch legs,
  ramp monotonicity, no-lurch, gates, max-secs math, fake-LowCmd send checks) — full
  suite 285 passed / 3 skipped. Mocked end-to-end smoke: normal + injected-crash runs
  both end at weight 0, only motors 15..28 ever commanded. Robot untouched.
- **Runbook:** docs/ARM_DANCE_DESIGN.md §4 (read mode → 5 s arm-run → Ctrl-C abort test
  → 15/60 s → gains pass → full dance + music). v1 is ARMS-ONLY (waist stays onboard).

## 2026-07-05 (choreography-edit build) — TARGETED SECTION-EDIT TOOL BUILT + DRY-RUN VALIDATED (contingency for s2r gate).
- **tools/edit_choreography.py** (new; +tests/test_edit_choreography.py, 10 tests; suite 295
  green): music-sync-preserving difficulty editor for the 30 fps deploy CSV — per-section LEG
  amplitude blend toward the section's stance interpolation (arms/waist/root-XY/quat untouched,
  frame count NEVER changes), cosine edge blends (velocity-continuous), optional global
  quasi-static ankle-load proxy cap (hip/ankle/waist-pitch pull toward median, <=3 passes,
  0.85 step, +-0.2 s smoothed). Validators built in: proxy before/after per section, FK
  foot-height (no new penetration >5 mm), no new velocity spikes, vet_motion gate — any fail
  = loud FAILED + exit 1.
- **KEY PHYSICS FINDING (validated by FK guard):** the 43-47 s Thriller lean lives in the ROOT
  orientation, so pulling pitch joints toward median moves the feet AWAY from the CoM and makes
  the ankle proxy WORSE on the deep-lean frames. The cap is therefore per-frame GUARDED (proxy
  is frame-independent under FK): reductions kept only where they help, counterproductive
  frames reverted + reported as residual-over-cap. Implication: if attempt-2 fails ON THE LEAN,
  the fix is a root-pitch/choreography change or leg-scale on that window, not the cap.
- **Dry-run** (data/motions/edits/thriller_deploy_edit_dryrun.{csv,json}, NOT staged anywhere):
  sections 13-17.5 s + 43-47 s @ leg-scale 0.6, cap 35 Nm. Proxy global max 58.2->54.8 Nm,
  13-17.5 s mean 19.7->16.9, 43-47 s p95 54.9->52.9; over-cap frames 88->78; foot mins,
  velocity max, frame-0 pose and z-reference all unchanged; vet PASS. Grounding policy for
  edits: never re-reference untouched frames (input floor = training floor); only lift if an
  edit digs NEW penetration.
- Production artifacts untouched (data/policies/** read-only for this lane); no commits made
  by this lane.

## 2026-07-05 (measurement session + retrain verdict) — s2r attempt-1: dance QUALITY KEPT, torque SOLVED, drift regressed; attempt-2 running.
- **Measurement session (user at robot, remote unavailable -> motion steps deferred):**
  (0) MASS MEASURED: **34.6 kg** as-deployed (standard battery, hands, covers) = model +1.26 kg,
  inside the retrain DR band, within 0.4 kg of config center -> no config change. "~35 kg" retired.
  (1a) LIMP CAPTURE (read-only, committed cee4839): obs staleness p95 **1.78 ms**, 0% repeated
  ticks -> NO sensor-side latency (audit exp #4 CLOSED; latency DR stays hygiene).
  tau_est zero-offset <=0.26 Nm on all 12 leg motors (sensor trustworthy). Thermal flat; NOTE
  right_shoulder_pitch idles at 59 C (others 39-49) — eyeball next session.
  Standing baseline / trim sweep / legodom rerun DEFERRED until a damping remote is in hand
  (declined user request to stand the robot without one — no independent stop; rule held).
- **s2r attempt-1 gate (full matrix, 128 envs):** nominal survival 128/128; ankle mean 5.8 /
  RMS 7.8 Nm (thermal ~3.4 C/min -> 10+ min runtime; 35% cooler than a2's 10.1 RMS);
  survival 96.1% @ 20 ms+push (passes realistic-worst), 50.8% @ 40 ms+push (informational);
  per-section: ZERO nominal falls everywhere incl. 13-18 s brace; torque unloaded most exactly
  in the old worst 40-50 s lean cluster (4.6 Nm mean). **Global mpkpe 0.52 DECOMPOSED:
  root-relative mpkpe 0.084 vs a2 0.089 — dance quality PRESERVED (crisper than baseline);
  the inflation is pure world-XY DRIFT: s2r max 1.55 m vs a2 0.81 m (mid-dance 1.33, recovers
  to ~0.35 by end).** Drift is fine for the tethered HW test, at the 1.5 m excursion limit for
  a show.
- **GATE v3 (evidence-based revision, c2062fe):** quality bar = root-relative mpkpe <=0.10;
  new drift bar max XY <=1.0 m nominal; worst GATED condition = 20 ms+push (measured sensing
  latency ~0 makes 40 ms bars unrealistic -> informational only).
- **ATTEMPT-2 LAUNCHED (train-thriller-s2r-b):** single delta motion_global_root_pos weight
  0.5 -> 1.0 (targets the drift, the only regression). Autopilot (parameterized) + gate v3 +
  poller armed; verdict at exports/thriller_s2r_b/RESULT.txt (~2.5 h).
- **DECISION STANDING:** s2r attempt-1 is cleared for ONE tethered HW test (quality+torque+
  realistic survival all pass; drift irrelevant on tether) — pending rollout video visual
  sign-off (rendering) + a working remote. If s2r-b also passes with drift <=1.0 m it becomes
  the show candidate.

## 2026-07-05 — ATTEMPT-2 (s2r-b) VERDICT: drift FIXED (0.64m), quality kept; both candidates STAGED for hardware.
- train-thriller-s2r-b (single delta: motion_global_root_pos weight 0.5->1.0): nominal survival
  128/128, rr_mpkpe 0.086, **drift max 0.64 m** (attempt-1: 1.55; a2 baseline: 0.81; bar 1.0),
  ankle mean 6.05 / RMS 8.17 Nm, global mpkpe 0.52->0.177. Gate: 5/9 pass; 4 misses are
  MARGINAL (mean 6.05 vs 6.0; 20ms+push survival 94.5% vs 95% = 1 run of 128; p95 17.2/20.6 vs
  15/20) — the audit's own floor estimate is 5-7 Nm/ankle, so the mean sits AT the physical floor.
- **DECISION: stop iterating in sim** (attempt-1 vs -2 differ within seed noise on shared bars;
  bars not revised a third time — misses recorded honestly). **s2r-b = PRIMARY hardware candidate**
  (staged data/policies/thriller_s2r/: policy.onnx + gap_check.json + policy_meta.json copy +
  thriller_deploy.npz), attempt-1 = fallback (data/policies/thriller_s2r_fallback/, better stress
  survival 96.1%, worse drift). data/policies/thriller/ (a2, hardware-proven) UNTOUCHED as the
  running deploy artifact until a candidate passes hardware.
- Attempt-1 visual sign-off done by Claude (ghost-reference render, 4 timestamps): full arm
  amplitude, pose mirrors reference, offset = drift only. s2r-b render in flight ->
  data/previews/rollout_s2r_b.mp4 for the user. Attempt-1 video at data/previews/rollout_s2r.mp4.
- NEXT HARDWARE STEP (needs remote): ONE tethered ground-run-legodom of s2r-b (--max-secs 30),
  full telemetry auto-captured, yaw-align offset printed+recorded; then the deferred measurement
  steps (standing baseline, trim sweep). Remote still unavailable this session.

## 2026-07-06 — 🏆 FULL THRILLER ON THE GROUND, TETHERED, COMPLETE (2589/2589 ticks). Sim2real chain VALIDATED end to end.
- **Staged tethered progression of s2r-b (user present with remote, all runs telemetry-recorded):**
  preflight PASS (yaw-align proved live: offsets -87/-57/+38/+85/+88 deg across runs — every one
  would have been near-worst-case OOD under the old code) → 5s ✓ → 15s ✓ (through stepping-window
  entry, calmest section) → 30s ✓ (full stepping window) → **FULL 51.8s DANCE ✓ (2589 ticks)**.
- **FULL-DANCE HARDWARE NUMBERS: ankle |tau| mean 6.5 / RMS 8.9 Nm — the DANCING robot's ankles
  run cooler than the vendor controller pays to STAND (9.2 avg measured). Ankle temps FLAT
  (0.0 C/min) over the dance; worst motor +2.3 C/min (waist, choreography), hottest 54 C
  (shoulder). The ankle thermal wall is gone.** Legs |a| max 4.75; gyro p95 0.89.
  Old policy at same task: 15-20 Nm ankle, 22.5 C/min, gain boosts, brace-falls at 14-16s.
  s2r-b at TRAINED GAINS, no boost: sim gate numbers reproduced on hardware (RMS 8.9 vs ~8-10 sim).
- Two mid-session safety-layer refinements (committed, tested): per-joint action caps (two benign
  right_wrist_yaw trips at 10/12 during claw choreography; legs never exceeded 3.4-4.9 —
  legs keep cap 10, arms x1.6) + abort messages name the joint.
- **NEW RUNBOOK FINDING: end-of-run damp->restore handoff on loaded feet triggers onboard
  catch-stepping (~1-1.5m, consistently rightward = the measured left-heavy stance).** Tether
  never loaded; onboard caught it every time. Backlog: hold-then-handoff (restore 'ai' while
  still balanced) + keep clearance to fences at run end.
- Onboard-standby baseline (audit exp#5 CLOSED, committed): vendor stands at ankle 12.3/6.2 Nm
  (NOT ~0), left hip_roll 25 Nm, left_ankle_roll +2.7 C/min @52C. Staleness replicated 1.75ms.
- **STATUS: s2r-b VALIDATED ON HARDWARE — the sim2real retrain closed the gap.** Remaining to
  show-ready: repeatability (3x clean full runs per exam standard), slack-tether -> free runs,
  music-synced show run, outcome recording in the app, and the promotion decision vs the 99%
  held-out standard (s2r-b gate numbers + hardware evidence to be taken to the user).

## 2026-07-06 — s2r-b PROMOTED TO SHOW POLICY (user-ordered, full guarded machinery). Thriller = SHOW-READY.
- Formal held-out exams x3 (de-correlated seeds 90001/90011/90021, 256 envs each):
  **nominal 256/256 AND push 256/256 on all three — 1536/1536 episodes clean**, mpkpe 0.173-0.179.
- Three HMAC-signed sim_exam/v1 verdicts bound to the canonical policy+deployable-CSV bytes
  (data/policies/thriller/heldout_verdict_s2rb_s{1,2,3}.json); dance record motion_csv fixed to
  the DEPLOYABLE thriller_deploy.csv (was pre-ramp thriller_show.csv — audit motion-sha seam);
  attach_policy -> 3x record_sim_run_from_verdict -> promote(): **status=show-ready, policy sha
  pinned b3511dd31fe96e40...**. Deploy bundle rebuilt + authorized (deploy/bundles/thriller).
- Artifact layout: data/policies/thriller/ = s2r-b (canonical, STAGED.txt has provenance);
  a2 preserved at data/policies/thriller_a2_fallback/. Evidence chain: 3x signed sim exams +
  s2rb_gap_check.json (gate v3) + 3x full-dance hardware telemetry (2026-07-06).
- Deploy nit closed: read_state drain now opt-in (DRAIN_READS=0 default — no-op at verified
  depth-1 QoS, and sub-ms reads spammed SDK error prints into the control loop).
- REMAINING to paid-show grade: slack-tether -> free runs; music-synced rehearsal (audio wiring
  from show-production work); end-of-run hold-then-handoff (catch-step fix, backlog); 2-3 min
  in-place choreography for real show pieces; venue checks. Robot-facing steps need the user.
- **BOX RECOMMENDATION: DELETE the GreenNode notebook** — training + exams complete, policy
  promoted, all artifacts pulled to the laptop (evals, verdicts, ONNX, videos, checkpoints for
  s2r/s2r-b remain on the box's network volume if re-needed... NOTE checkpoints NOT pulled:
  logs/rsl_rl .pt files stay on the box volume; pull model_4999.pt for both runs before delete
  if retraining continuity matters). Deletion is console-only (user click). Meter ~18k VND/h.

## 2026-07-06 — RETARGET FIDELITY: the "soft hits" were the FRONT-END's blanket velocity clamp, not the policy.
- prep_motion's blanket 0.9*3pi=8.48 rad/s clamp sits 2-4x BELOW true G1 motor limits (hips 20-32,
  shoulders/elbows 37, wrists 22); it modified 156/1329 frames and blunted 58 REAL dance accents
  by 60-85% (13-17s punches: raw peak 56.4 -> 8.5 rad/s, HF energy kept 2.8%). Of 60 over-limit
  events only 2 were glitches — the old "isolated spike" advisory read was wrong.
- CORRECTIONS: GMR --velocity-limit was a NO-OP this run (thriller_vlim.csv byte-identical to raw);
  the "0% over-velocity (was 3.1%)" effect came from prep's clamp. VFR->30fps normalize dropped
  15.5% of source frames (use native avg fps for future videos).
- Sharp reference PROTOTYPE (vet PASS, timing identical, seam <1e-6): data/motions/edits/
  thriller_deploy_v2_sharp.csv via tools/make_sharp_reference.py (despike + per-joint 0.9x true
  motor limits): accents kept 41% -> 96%. Full report docs/retarget_fidelity.md.
- Pipeline standard (for many-dances work): retarget without blanket clamp; despike + per-joint
  clamp in prep_motion; per-joint vet advisory; native-fps normalize. Morphology losses (head
  snaps, claw fingers) are unfixable by any stage — hands remain authored-only (hands spike).
- ACTION: v3 GPU program extended with variant v3d = precision recipe + SHARP reference (the
  highest-leverage quality change identified today).

## 2026-07-06 — APP PIPELINE GENERALIZED: video -> show-ready CANDIDATE, 3 human gates (parallel lane, not committed)
- The manual Thriller flow is now the app's job pipeline (new pipeline/stages/cloud_motion.py +
  extended RetargetStage): extract = 30fps re-encode -> push -> GVHMR job on box -> pull
  hmr4d_results.pt; retarget = GMR retarget -> grounding -> window (or dance.yaml override) ->
  vet gate -> MuJoCo preview -> prep_motion -> NEW pipeline/deploy_ramp.py (2.5s activation
  ramp; reproduces the canonical thriller_deploy.csv bit-for-bit — golden-tested); train =
  push deploy csv -> csv_to_npz job -> train_sim2real (Sim2Real task + the s2r-b
  motion_global_root_pos delta, 5000-iter default) -> export_policy.py -> pull to
  data/policies/<slug>/ {policy.onnx, policy_meta.json sidecar generated from
  docs/mjlab_policy_interface.json, <slug>_deploy.csv/.npz}; verify = sim_gap_check v3 ->
  3x heldout_eval (seeds 90001/90011/90021, 256 envs) -> signed sim_exam/v1 verdicts
  (mjlab_verify) -> register-or-update dance BOUND to the deployable csv -> attach_policy ->
  record_sim_run_from_verdict x3 => SIM-VERIFIED; export = deploy-contract audit + music
  attach from data/audio/<slug>/music.* (1.5s lead-in rule). Promotion stays human (Shows UI).
- Infra: box work via run_job.sh script-jobs (tmux + status.json, reboot-safe); stages persist
  phase in stage meta and raise StageBlocked with honest progress (training shows iter/reward
  parsed from the box log) + a retry_after_s hint; new server poll loop re-queues due jobs
  every 30s; scp push/pull added to pipeline/cloud.py (ssh transport only).
- HUMAN GATES: (1) Approve-training button (POST /api/jobs/{id}/approve-train) after preview
  review — nothing reaches the GPU before that click; (2) promotion (guarded machinery
  unchanged); (3) robot day (the app still has no robot code path). Legacy guard: extract
  skips when retarget is already done, so the old Thriller job can't re-burn GPU on restart.
- Per-dance knobs in ONE place: dance.yaml in the job dir (iterations, num_envs,
  extra_train_args, window_start/end_s, heldout seeds/envs, velocity_limit); defaults = the
  promoted recipe; unknown keys hard-error. Operator doc: docs/NEW_DANCE_PLAYBOOK.md
  (exact clicks, ~3.5-5h timeline, failure playbook, artifact map).
- NOTE (retarget-fidelity entry above): the sharp-reference/per-joint-clamp standard lands via
  the v3 lane; when prep_motion/vet change there, this pipeline picks them up automatically
  (it calls prep_motion + vet_motion directly). velocity_limit stays a dance.yaml knob.
- Tests: full suite 339 passed (31 new: FakeBox-mocked walkthroughs of all four cloud stages
  incl. failure honesty, deploy_ramp golden test vs the canonical deployable, scp argv,
  poll loop, approve endpoint, model-marked real RetargetStage csv->deploy run). Also live-drove
  the running engine (isolated dirs, cloud deliberately unconfigured): upload -> vet PASS ->
  preview -> approval block -> approve -> honest cloud block. NO box jobs launched this
  session (box accessed read-only for contract recon).

## 2026-07-06 — SHOW AUDIO BUILT: the robot plays its own music (PlayStream) / is its own cue light (LED). Robot untouched.
- Rehearsal finding: operator watches the ROBOT, not the terminal — rehearsal_cue.sh's
  banner failed ("had no idea when to start the music"). Product answer: the G1's own
  speaker + head LED, over the same DDS the runtime already uses.
- SDK recon (~/robot/unitree_sdk2_python, read-only): AudioClient PlayStream(app,stream_id,
  pcm)->(code,data) g1_audio_client.py:63, PlayStop:68, SetVolume:47, LedControl(R,G,B):54,
  service "voice" api 1001-1010; stream MUST be 16kHz mono s16 PCM (example
  g1_audio_client_play_wav.py:25), sent as 96000-byte chunks / 1.0s pacing (wav.py:125).
- Built pipeline/show_audio.py (ffmpeg->16k mono PCM, sample-aligned chunking, timing math,
  4 modes) + tools/show_run.sh (rehearsal_cue successor; AUDIO_MODE=robot|led|laptop|banner;
  same deploy cmd + gates, deploy_runtime UNMODIFIED — watches stdout for the policy-start
  line, anchors tick0 via date +%s.%N in the shell so python spawn can't skew the cue) +
  docs/SHOW_AUDIO.md. Timeline contract: music = tick0 + 2.5 (deploy_ramp.RAMP_S) + 1.5
  (dance record audio.align.audio_delay_s) = 4.0s; AUDIO_LATENCY_COMP env starts audio
  early by the (to-be-measured) robot playback startup latency. LED mode: blue T-3/-2/-1,
  GREEN = press play. Abort: runtime STOP:/exit -> SIGTERM cue -> PlayStop + LED off
  (music can never outlive a damped robot).
- Dry-runs (fake runtime, banner mode, NO robot contact): cue fired at tick0+3.600s
  (0.4s lead) with 0.1ms scheduling error; abort at +1.5s killed the cue before it fired.
  Thriller music.wav converts to 1,417,600 B = 44.3s = 15 chunks; offset CLI -> 4.0.
- Tests: tests/test_show_audio.py, 29 new, SDK fully faked (DDS never initialized; lazy
  sdk import asserted). Full suite 365 passed, 3 skipped.
- LAPTOP AUDIO ROOT CAUSE CLOSED: driver requests intel/sof-ipc4/arl/sof-arl.ri (journal:
  sof_probe_work err -2); topology sof-hda-generic-2ch.tplg already installed. Intel-signed
  blob staged+verified in scratchpad fw_install/sof-ipc4/arl/ (sof-bin v2025.01; 2025.12.2
  alternate alongside); ARL uses the MTL image — offline fallback = symlink to the distro's
  existing mtl/intel-signed/sof-mtl.ri. Exact sudo install+reload (or reboot) one-liner in
  docs/SHOW_AUDIO.md — install BEFORE reboot, /tmp is volatile.
- NEXT (robot-facing, human present): 30s speaker smoke test in damp, measure
  AUDIO_LATENCY_COMP (film screen+speaker), LED cue check, then AUDIO_MODE=robot dress
  rehearsal — checklist in docs/SHOW_AUDIO.md. NOT committed (per task instruction).

## 2026-07-06 (post-reboot) — LAPTOP AUDIO WORKS (sof-arl.ri installed); "beeps" = the placeholder click track; v4+acro queued.
- Reboot completed the SOF driver bind: sof-hda-dsp card live, analog Speaker port active,
  played to speakers successfully. THE FILE data/audio/thriller/music.wav IS THE PLACEHOLDER
  CLICK TRACK (verified: 76% silence, single ~3.3kHz tone) — the real song was never dropped
  in (source video has no audio). WAITING ON USER: real Thriller track -> tools/attach_music
  flow will convert/replace/re-attach. The phone rehearsal used the real song from the phone.
- v4 calm-legs training launched (5-way share with v3a-d; ETAs +~1h). autopilot_v3 patched for
  v4 (sharp-ref eval). Acro/backflip: reference verified (-344deg pitch, 5.6s), converted npz
  staged on box; launcher hardened (terminfo-proof process counting, unique-job count,
  threshold 3) after a premature 6-way launch was cleaned; now WAITING and will take a slot
  when 2 trainings finish. Verdict poller armed (first-verdict wake).
- NEXT AUTONOMOUS BUILD: end-to-end validation of the many-dances app pipeline on
  dance1_subject2_seg.csv (real box, deferred train-approval watcher gated on GPU slots).

## 2026-07-06 (session end, handover) — v3 program: 3/6 verdicts in, ALL beat arm baseline ~30%.
- v3a arm RMS 9.72 / v3b 9.42 (ankle 7.05 RMS, drift 1.20 gate-fail, deploy needs
  ARM_GROUND_KP_SCALE=2.5) / v3d 10.20-vs-15.18. Pending: v4 calm-legs (leg decision), v3c,
  backflip (train-acro-1 via acro-launcher6), dance1-e2e (app-driven). Full decision procedure
  + resume state in HANDOVER.md. Autopilot artifact-retention bug noted there.
- Session totals: ~35 commits; audit->retrain->hardware-validation->promotion cycle closed;
  music rehearsal done (placeholder track identified — real song pending); laptop audio fixed;
  many-dances pipeline live E2E; acro program staged; fluidity forensics + v4 recipe.

## 2026-07-06 (evening resume) — GATE DATA WAS MISSING, NOT FAILING: gap-check CLI bug found+fixed; backfill + fluidity sweep live; backflip ACTUALLY training now.
- **ROOT CAUSE (voids 3 "GATE_FAIL" verdicts):** cloud/sim_gap_check.py's argv shim
  (known_tasks filter) stripped the literal stock task string even when it was the VALUE of
  --task → tyro rc=2 "Missing value for argument '--task'" → NO gap_check.json for every
  stock-task variant. v3a/v3d/v4 "gate=FAIL" means gap data missing (gate_pass(None)=False);
  v3b (GAPEVAL task name, not in the filter set) is the only variant with a REAL gate number:
  survival 1.000, rr_mpkpe 0.083, drift_max 1.20m (the actual fail). Shim fixed (only strips
  a bare positional, commit 7051956); `gap-backfill` box job re-runs v3a-last/v3d-last/
  v4-mid/v4-last with kept outputs.
- **Fluidity decision numbers:** cloud/{sim_trace_dump,fluidity_sim_metrics,fluidity_trace_job}.py
  (written last session, never pushed) committed, v4-dir bug fixed (exports/thriller_v34),
  pushed; `fluidity-sweep` job runs s2rb-baseline + v3a/v3b/v3d/v4, `v3c-fluidity` waits on
  v3c's RESULT — each appends FLUIDITY_LEG_BAND to its RESULT.txt (bar: leg 2-10Hz <= 0.20,
  amp ratio > 0.5).
- **OPS ERROR (mine, owned):** killed pid 883966 assuming "stale v3a tmux stub" — it was the
  tmux SERVER; v3c training died with it at ~iter 9170/10000 (ETA was 51 min). model_9000.pt
  (saved 16:00) is the final artifact; v3c-autopilot3 evaluates it (mid+9000) with the fixed
  gap check. Lesson reinforced: identify what a pid IS before killing (CLAUDE.md discipline).
  Also: pkill -f self-match killed two SSH sessions (bracket-escape or split calls).
- **Backflip: the "hardened" acro launcher STILL had the terminfo bug** — it exported
  TERM=dumb, waited 6 h correctly, then tmux refused the launch ("missing or unsuitable
  terminal") and the job died with status.json stuck at "running". train-acro-1 launched
  manually 16:05 UTC (10k iters, flight-grace mask 46/277 frames confirmed in log) +
  acro-autopilot armed; launcher fixed (TERM=xterm).
- dance1-e2e app job advanced unattended after app-server restart: train DONE (policy pulled
  + staged data/policies/dance1_e2e), verify = sim-gap on box (uses the FIXED script).
- Local: .gitignore now excludes data/{audio,body_models,jobs}; data/shows + telemetry
  analyses + design mockups committed. Poller armed for gap-backfill+fluidity+v3c landing.
- NEXT: when poller fires → fill the V3_PROGRAM decision matrix (gates + arm RMS + leg band),
  pick winner per DECISION PROCEDURE, stage as CANDIDATE (never overwrite
  data/policies/thriller/), 3x signed held-out exams, render sign-off; then user-facing:
  tethered HW test of winner + ARM_GROUND_KP_SCALE A/B + robot-speaker/LED checklist.

## 2026-07-06 (23:45 ICT) — V3 PROGRAM DECIDED: v3c WINS (only gate PASS + best arms, 8.75 deg vs 13.81 baseline). Candidate staged; held-out exams running.
- All verdicts in after the gap backfill. DECISION MATRIX (arm RMS deg / gate v3 / leg-band
  2-10Hz (bar<=0.20) / leg-amp (pref>0.5)):
  - v3a: 9.72 / FAIL drift 1.15m / 0.130 / 0.43
  - v3b: 9.42 / FAIL drift 1.20m / 0.133 / 0.43 (+ ARM_GROUND_KP_SCALE=2.5 deploy contract)
  - **v3c@model_9000: 8.75 / PASS (drift 0.71m, surv 1.000, rr_mpkpe 0.080, ankle 4.43/6.19Nm
    — coolest) / fluidity pending / VERDICT=WIN**
  - v3d: 10.20-vs-sharp-15.18 / FAIL drift 1.50m + ankle p95 / 0.147 / 0.41
  - v4:  11.72-vs-sharp-15.18 / FAIL CATASTROPHIC (nominal survival 0.000, drift 6.27m) /
    0.124 / **0.17 = legs frozen** — the calm-legs recipe over-suppressed leg motion and
    doesn't survive the deploy-matched harness. v4 lane CLOSED (the amp-ratio bar did its job).
  - s2r-b baseline: 13.81 / promoted / 0.147 / 0.34.
- Leg-fluidity read: EVERY v3 variant beats s2r-b on both leg metrics — the arm-crispness
  fixes also improved legs; v3c's numbers land when v3c-fluidity appends (box).
- Drift >1.0m was the REAL common gate failure (visible only after the backfill) — v3c is
  the only variant that dances in place. Choreography-area safety implication noted.
- **CANDIDATE STAGED: data/policies/thriller_v3c_candidate/** (policy.onnx from model_9000,
  gap/arm/RESULT evidence, policy_meta.json verbatim from s2r-b — V3C is reward-deltas-only,
  actuator plant identical (verified in cloud/sim2real_task_v3.py), same thriller_deploy
  motion = no sha churn). data/policies/thriller/ (s2r-b) UNTOUCHED, stays the show policy.
- IN FLIGHT: box job v3c-heldout (3x 256-env held-out exams, seeds 90001/90011/90021, on the
  deployable motion) -> laptop signing via pipeline/mjlab_verify.py when done; v3c-fluidity;
  train-acro-1 (~4-8h); dance1-e2e verify (sim-gap ~15% when last checked). Poller armed.
- Render pulled for USER SIGN-OFF: data/previews/rollout_v3c.mp4.
- docs/DYNAMIC_SKILLS.md hardware-risk memo regenerated + committed (recommendation: backflip
  stays a SIM SHOWPIECE; 7-item evidence gate before any hardware conversation).
- NEXT: exams land -> sign 3 verdicts -> if all >=99%: v3c is sim-verified, goes to the user
  with render + hardware-test ask (tethered, ARM_GROUND_KP_SCALE not needed for v3c).

## 2026-07-07 (00:15 ICT) — v3c SIM-VERIFIED: 3x signed held-out exams, 100% (1536/1536), all bars met. Awaiting USER: render sign-off + tethered test.
- Held-out exams (box job v3c-heldout, seeds 90001/90011/90021, 256 envs, deployable
  motion): **nominal 256/256 AND push 256/256 on ALL THREE — 1536/1536**, mpkpe
  0.172-0.184m (s2r-b was 0.173-0.179). Signed via pipeline/mjlab_verify.py; all 3
  signature_valid + derive_pass (same machinery that promoted s2r-b). Evidence committed
  in data/policies/thriller_v3c_candidate/.
- v3c fluidity landed: leg band 0.156 (bar <=0.20 ok; s2r-b 0.147) + **LEG_AMP 0.456 — best
  of the program** (s2r-b 0.34): legs move MORE and stay smooth. Decision bars: gate PASS /
  arm 8.75 < 13.81 / band <=0.20 / amp best-of-field (pref >0.5 not fully met by anyone).
- **v3c candidate = COMPLETE, waiting on HUMAN GATES (not to be automated):**
  1. render sign-off: data/previews/rollout_v3c.mp4 (arm crispness is the complaint being
     fixed — user judges);
  2. ONE tethered hardware run w/ telemetry (user present + remote; NO gain knobs needed —
     stock deploy contract, same motion CSV as the show policy);
  3. promotion decision (user-ordered, guarded machinery) — data/policies/thriller/ (s2r-b)
     remains the show policy until then.
- Hardware A/B note for the session: s2r-b hardware arm RMS was 13.2 deg; v3c sim predicts
  ~8.75 — measure on the tether and compare.
- Still in flight: train-acro-1 (sim backflip, hours), dance1-e2e verify stage (app-driven).

## 2026-07-07 (00:30 ICT) — dance1-e2e VERDICT: app pipeline mechanics VALIDATED end-to-end; the trained policy itself correctly REJECTED by the gap gate (fail-closed proven).
- Full unattended chain exercised with real artifacts: CSV intake -> retarget/vet/preview ->
  approve-train gate -> box convert -> train (5000 it) -> export -> pull+stage -> sim-gap
  gate -> **FAIL CLOSED** with honest, actionable diagnostics (nominal survival 0.320,
  delay20ms survival fail, ankle p95 fail; "targeted choreography edit or recipe delta,
  then a FRESH job" — exactly per design).
- Why the policy is bad (KNOWN, not a pipeline bug): dance1's prep clamped 487/1013 frames
  (48%) under the blanket 8.48 rad/s velocity clamp (raw peak 26 rad/s) — the same
  front-end defect documented in docs/retarget_fidelity.md. A coherent re-test belongs
  AFTER the per-joint-clamp standard lands in prep_motion (v3 lane follow-up), via a fresh
  job with dance.yaml knobs. NOT re-run now — no green-banner GPU burn.
- Untested app segment: exams -> signed verdicts -> dance registration (blocked behind a
  passing gate). Components individually proven today via the v3c manual chain (heldout_eval
  x3 + mjlab_verify signing + signature_valid/derive_pass). Acceptable coverage; first
  gate-passing app dance will exercise it in situ.
- e2e job 20260706-172405-2eb6e0 left in honest failed state (its training cost is sunk in
  the checkpoint; a future recipe-delta job can reuse the box npz).
- Box: train-acro-1 now the sole trainer (~iter 2900/10000, ETA ~2.5h + autopilot eval).

## 2026-07-07 (02:50 ICT) — Backflip attempt 1 = REWARD HACK (0 rotation, "survived upright" 64/64); anti-skip fix launched as attempt 2. v3e (v3c recipe x sharp ref) launched in parallel.
- train-acro-1 verdict: landed 0/64, rot 0.000 rev vs ref 1.168 — the flight-grace window
  (built to forgive mid-air phase lag) also made NOT flipping termination-free; policy
  optimized "skip the flip, track the rest". Evidence data/reports/acro/attempt1/ + render
  data/previews/rollout_acro1.mp4. Even without flipping: knee 114-123/139 Nm, ankle
  saturated 50/50 — reinforces the DYNAMIC_SKILLS landing-load concern.
- FIX (single delta, attempt 2): in-grace flip-skip detector — loose anchor_ori check
  (threshold 1.7 of max 2.0) INSIDE grace kills an upright robot while the reference is
  inverted; <=~130deg phase lag still passes. cloud/dynamic_skills_task.py updated;
  autopilot_acro.py now per-attempt export dirs (exports/acro2). train-acro-2 running
  (10k iters) + acro2-autopilot armed.
- **v3e LAUNCHED (the program's prescribed follow-up):** v3c recipe (S2R-V3C task, 10k
  iters) x SHARP reference (thriller_deploy_v2_sharp.npz) — tests whether converge-longer
  fixes the drift/gate failures that sank the sharp-trained v3d/v4 while keeping the 96%
  accent preservation. autopilot_v3.py variant "e" -> sharp motion+baseline. Trainings
  share the GPU (~2.1s/it each expected).
- Decision on prep_motion per-joint clamp DEFERRED on evidence: sharp-trained variants
  went 0-for-2 on the gate so far — v3e is the test that decides whether the per-joint
  clamp standard lands in prep_motion (if v3e gates clean) or the sharp ref stays a
  per-dance knob.
- Box after these: nothing queued -> pull model_4999 checkpoints (s2r, s2r-b) + user
  deletes box in console.

## 2026-07-07 (07:50 ICT) — BACKFLIP LANE CLOSED (2 attempts, conclusive): the human-mocap flip is beyond the G1's actuator envelope at true effort limits.
- Attempt 2 (anti-skip fix) failed EXACTLY as diagnostically hoped: skip optimum gone
  (0/64 "survive upright" vs 64/64 in a1), policy genuinely launches — knee saturates at
  its exact 139/139 Nm rating (p95 134), ankle 50/50, waist 50/50, impact 199 m/s² — but
  achieves 0.165 rad mean rotation of 7.34 required (~2%) before the apex check kills it.
- Cross-check (measurement discipline): intake gate had already flagged reference peak
  joint velocities 40.6 rad/s with 6 joints over the 20-37 rad/s motor ratings. Two
  attempts bracketing the termination-design space + actuator saturation + intake
  velocity audit = converging evidence, verdict labeled DECISIVE.
- Deliverables: docs/DYNAMIC_SKILLS.md §6 finalized (lane verdict + path forward = author/
  source a G1-feasible lower-amplitude reference — USER decision, not attempt-3);
  evidence data/reports/acro/attempt{1,2}/; renders data/previews/rollout_acro{1,2}.mp4.
- Remaining on box: train-thriller-v3e (~4h) -> autopilot verdict decides the per-joint
  clamp/prep_motion standard. After that: pull s2r/s2r-b model_4999 checkpoints, then BOX
  DELETE (user console click) — nothing else queued.

## 2026-07-07 (09:15 ICT) — v3e = WIN: the sharp reference is VALIDATED with the converge-longer recipe. Session now has TWO gate-passing candidates.
- v3e (v3c recipe x sharp ref, 10k iters): gate PASS — drift 0.66m (best of program),
  survival 1.000, rr_mpkpe 0.080, ankle 4.80/6.70Nm, arm RMS 8.82 vs sharp baseline 15.18.
  The drift failures that sank sharp-trained v3d/v4 are FIXED by converge-longer.
- CONSEQUENCE 1 (pipeline): per-joint clamp / sharp-reference standard is now EVIDENCED —
  land in prep_motion as default after the session (accents: 96% vs 41% preservation).
- CONSEQUENCE 2 (promotion choice): v3c tracks the BLUNTED reference crisply; v3e tracks
  the TRUE-dynamics reference equally crisply (8.82 vs 8.75, different baselines) with
  better drift. v3e costs a motion-sha change on the dance record (sharp deploy CSV must
  be staged; ramp regenerated for sharp) — v3c is the drop-in. RENDERS side by side:
  data/previews/rollout_v3c.mp4 vs rollout_v3e.mp4 — user judges which LOOKS like the dance.
- v3e held-out exams (3 seeds) + fluidity launched — land mid-session (~35 min).
- Session doc stands: run v3c stages as planned (sim-verified NOW); if v3e's exams come
  back 100% + render preferred, ONE extra full tethered run with v3e while rigged.

## 2026-07-07 (session, user present) — 🏆 v3e PROMOTED TO SHOW POLICY after live tethered A/B; audio validated both paths (aux = show default).
- Staged v3c runs: 5s ✓, 15s ✓, full = benign STOP at tick 903 (left_wrist_yaw 16.02 vs
  cap 16 — v3c's sim envelope max is 17.1; cap was tuned to s2r-b). Data-derived fix
  ARM_ACTION_CAP_SCALE=2.2 (arm cap 22, legs untouched at 10) → full dance 2589/2589 ✓.
- HARDWARE A/B (tools/hw_ab_compare.py, committed, same math as forensics):
  v3c arms 9.32 vs s2r-b 14.29 deg RMS (-35%), legs -2.13, waist -1.31, wobble par.
  v3e (vs SHARP ref, harder target): arms 9.08 vs 14.55 (-38%), legs -2.85, wobble par.
  Sim->hardware transfer of the improvement nearly exact (sim predicted 8.75/8.82).
- **USER ORDERED: PROMOTE v3e** (watched both full runs live). Guarded chain ran clean:
  s2r-b archived to thriller_s2rb_fallback/; v3e staged canonical at data/policies/
  thriller/ (SHARP motion now the canonical thriller_deploy.csv/.npz — sha changed,
  names kept for runtime defaults); 3x verdicts signed against the new bytes; attach ->
  draft (guard) -> 3x record (streak 3) -> show-ready, policy sha pinned e68335aa...;
  deploy bundle rebuilt [authorized (show-ready)]. Deploy note: ARM_ACTION_CAP_SCALE=2.2.
- AUDIO validated with user present: robot chest speaker WORKS (440Hz tone heard; the
  "silent" smoke test was the sparse click track being inaudible — API code 0 all along);
  aux/laptop path works. **SHOW DEFAULT = aux speaker (user choice), AUDIO_LATENCY_COMP
  0.0** (music fires from the same laptop that runs the show script; chat-latency
  confusion explained — show cue is script-anchored, no human timing). LED cue fired 2x.
  Robot-speaker latency unmeasured (backlog; only matters if robot mode is ever default).
- STILL MISSING: THE REAL SONG (user now understands the flow — will upload mp3;
  attach via tools/attach_music.py, then optional dress rehearsal).
- Retention checkpoints pulling from box (v3e 9999, s2r-b 4999, v3c 9000) -> box
  DELETE-ready once complete.

## 2026-07-07 (10:34 ICT) — 🎭 FIRST COMPLETE SHOW: v3e + REAL Thriller track, auto-synced via aux, 2589/2589, outcome recorded CLEAN.
- tools/show_run.sh end-to-end: promoted defaults (v3e canonical), music cue fired at
  tick0+4.0s (0.08ms scheduling error), full dance no aborts, temps 56C, caps clear
  (arms max 15.7/22, legs 4.98/10). Tracking repeatable: arms 9.13 deg RMS (A/B run was
  9.08). Rehearsal show recorded via shows API, outcome=clean (outcome loop exercised).
- Real song attached earlier this session (data/audio/thriller/music.wav = Thriller
  Audio.mp3, 48.6s, align 1.5s lead-in). Show default AUDIO_MODE=laptop (aux speaker).
- Box: all evidence archived (data/reports/box_final/), checkpoints pulled, delete
  green-lit to user (their console click).

## 2026-07-07 — GREENNODE BOX DELETED (user console click; verified: SSH refused + monitor unreachable). Meter stopped. All artifacts were local first.

## 2026-07-07 (afternoon) — Video sync FIXED, one-button show + opt-in stand-to-stand BUILT & tested, stand-after candidate staged. Investor demo ran (dance+music good).
- LIVE DEMO (investor): first demo.sh run looked like "damp + no music + no video" — root causes,
  all resolved: (1) "damp" = the onboard-balance release transient at start + the proven
  end-of-run ramp-to-damp; telemetry proved the FULL dance ran (2589 ticks, identical to the
  10:34 good run). (2) no music = the AUX SPEAKER WAS POWERED OFF (laptop routing was fine).
  (3) no video = demo.sh never opened one + my xdg-open forked-and-exited. Re-run: dance + music
  through aux = SUCCESS. CPU note: the 11:23 full run completed WHILE a workflow was starting
  (load 1.28) — CPU starvation was NOT the cause; still stopped builds during live robot windows
  as a precaution.
- VIDEO SYNC FIXED (tools/make_side_by_side.py): the reference is a TUTORIAL — instructor stands
  and talks ~0-7s before dancing. Retarget windowed that intro out, so robot dance-start (sim
  4.0s) aligns to source ~7.0s. Original naive composite PADDED source +4.0s (wrong direction) =
  ~7s lag. Fix: advance source ~3.4s + 0.9x drift correction (deployed motion runs ~10% slower
  than raw video); verified frame-by-frame at 4 points. Derivation committed in the tool docstring.
- BUILD (committed 0623e72): (a) one-button app show — POST /api/shows/{id}/run + pipeline/
  show_runner.py, ordered guards (show-ready/audio/robot-ping/single-run-lock/typed
  "I AM PRESENT WITH THE DAMPING REMOTE"), spawns the PROVEN show_run.sh, status poll, outcome
  capture reuses shows.record_outcome. (b) opt-in stand-to-stand: deploy_runtime --exit
  {damp,stand} default damp (proven path byte-identical); --exit stand = clean-completion only,
  holds final pose then restores onboard balance STANDING (no catch-step), GUARDED (refuses
  unless motion ends <=0.15rad from default -> damp), aborts always damp, telemetry saved.
  show_run.sh EXIT_MODE env (default damp) so demo.sh is unaffected. ARM_ACTION_CAP_SCALE
  default ->2.2. 26 new tests pass.
- "STAND AFTER" GAP + FIX: the promoted Thriller motion ends mid-dance (~39deg off default) so
  --exit stand is INERT on it (guard->damp). tools/make_stand_tail.py authors a candidate
  (data/policies/thriller_standtail_candidate/) = dance + 2.5s cosine return-to-default + 1.5s
  hold; runtime guard now PASSES on it. In-distribution (activation ramp in reverse) but
  UNVERIFIED end-to-end (mjlab box deleted) -> TETHERED VALIDATION ONLY, not show-ready. Never
  overwrites the sha-pinned show motion.
- KNOWN RED TESTS (pre-existing, from the v3e/sharp promotion — NOT this build): test_arm_dance
  (sharp ref arm speed 33.3>20 blanket limit — intended, needs per-joint limit) and
  test_deploy_ramp (golden expects old thriller_show-derived deploy; motion is now the sharp
  one). Update these goldens/limits for the sharp motion. Confirmed pre-existing on clean HEAD.
- NEXT (robot, user present): tether session to validate --exit stand on the standtail candidate
  (EXIT_MODE=stand) — the end-of-run handoff is where the catch-step lives; then re-author the
  show motion with the standing tail + (box back) re-exam for a show-ready stand-ending dance.

## 2026-07-07 — STAND HANDOFF VALIDATED ON TETHER (isolated): works, SHRINKS the catch-step but doesn't eliminate it.
- Method: isolated the handoff from the dance — built a stand-only motion (hold default 10s,
  data/policies/thriller_standhold_iso via tools/make_stand_tail-style tiling of deploy frame 0),
  ran ground-run-legodom --exit stand, HANDOFF_HOLD_S=3. User present + tether + remote.
- Robot SETTLES to dead-calm: 0-3s ~58deg leg drift (policy taking over from move-to-default),
  by pre-handoff last-2s = 0.75 deg drift / gyro 0.023 rad/s (CALM). So the earlier "wobble"
  was just initial settling of a STATIC reference (dance ends dynamically INTO standing, won't
  cold-start-wobble like this).
- Observability: a clean handoff is INVISIBLE (robot just keeps standing), so built
  tools/stand_led_test.sh — head LED BLUE during our hold, GREEN at the "handoff complete"
  (SelectMode 'ai') instant. Operator watches the FEET when it goes green (the watch-the-robot
  lesson again). LED via AudioClient voice service, separate short-lived DDS participant, fired
  only outside the 50Hz lowcmd loop (no control contention).
- **VERDICT (user eyes, 4 runs): SMALL SHIFT/STEP at the handoff.** Not planted-perfect, but a
  big improvement over the old damp->restore path (1-1.5m rightward catch-step). The residual
  step is at the ONBOARD controller's SelectMode('ai') takeover — the vendor controller
  re-grabbing the robot — which we don't control; it is AFTER our telemetry ends (onboard behavior).
- Interpretation: --exit stand delivers "ends standing, no damp-collapse" and materially reduces
  the catch-step, but a small onboard-takeover step remains and is likely the floor without
  vendor-side tuning. GOOD ENOUGH for a show end (vs damp) is the user's call.
- Backlog to reduce further (uncertain): (a) match handoff pose/gains to onboard's expected
  takeover state; (b) investigate if a brief overlap/handshake with 'ai' before unpublish helps;
  (c) vendor guidance on SelectMode takeover transients. Not blocking.
- Full dance+tail candidate (thriller_standtail_candidate) NOT yet run on hardware — the isolated
  handoff is the same mechanism; running the full candidate adds the dance before an identical
  handoff. Do that next tether session if a standing show-end is wanted.

## 2026-07-07 — STAND HANDOFF STEP SHRUNK TO NEGLIGIBLE (validated + replicated on tether). Root cause was a GAP, not just pose.
- Diagnosis (measurement-first, tick-cross-checked): onboard 'ai' neutral pose is ~18 deg RMS
  (max 42, elbows/hip-yaw/ankles) off the policy default we hand off at — I first suspected a
  pose-mismatch step. IMPORTANT self-correction: my standalone lowstate read looked "frozen"
  (0.01 deg variation) and I nearly called it stale; the lowstate TICK counter was advancing
  (LIVE) — the robot was just held that steadily. Discipline check caught a false "stale" call.
- User chose the LOW-RISK fix first: HANDOFF_OVERLAP_S — after SelectMode('ai') at the handoff,
  keep commanding the SAME standing pose 0.5s so a latent onboard takeover never leaves the
  robot briefly unheld. Same-pose command only -> no new-pose fall risk.
- RESULT (user eyes, 2 replications, LED-cued via tools/stand_led_test.sh so the handoff moment
  is visible): the small catch-step SHRANK to negligible/gone at 0.5s overlap vs a small shift
  at 0.0. So the residual step was a brief unheld GAP at takeover, not (only) the pose mismatch.
- Locked in: HANDOFF_OVERLAP_S default 0.0 -> 0.5 (env-overridable); test_deploy_exit updated to
  assert 10 hold + 5 overlap sends around restore. 15/15 handoff tests pass.
- STATE OF --exit stand: isolated handoff now validated CLEAN on the tether (settles calm ->
  holds -> onboard takes over planted). Still to do for a standing SHOW end: run the full
  dance+tail candidate on hardware, then author the standing tail into the real show motion +
  re-verify (needs the GPU box back). The pose-match (deeper) option was NOT needed.
- Robot left on onboard 'ai' balance (handoff restores it). LED test tool + iso motion are the
  reusable validation rig.

## 2026-07-07 — FULL DANCE+TAIL STAND-TO-STAND VALIDATED ON HARDWARE (tether). "Stand after the dance" DELIVERED (capability).
- Ran thriller_standtail_candidate (v3e sharp dance + 2.5s return-to-default + 1.5s hold, 2709
  frames/54.2s) on the tether, --exit stand with the 0.5s overlap default, LED-cued.
- 2 clean full runs (telemetry 20260707-134428, -135541): 2709/2709 ticks, no aborts/cap trips,
  arms max 16.0/16.1 (< cap 22), dance gyro p95 0.89/0.91, temp 55-56C. Return-to-standing tail
  settled calm (2.3 deg drift). **User verdict: dance clean AND handoff PLANTED (no catch-step)**
  after a real dance — same clean result as the isolated handoff. Full stand-to-stand works.
- So the complete flow is hardware-proven: stand (onboard) -> dance on command -> ease to
  standing -> hand back to onboard balance PLANTED. The overlap fix (0.5s) is what makes the
  handoff clean; the return-to-standing tail is what lets --exit stand engage.
- tools/stand_led_test.sh parameterized (POLICY_DIR / MAX_SECS) as the reusable LED-cued rig.
- REMAINING for a SHOW-READY standing-end Thriller (capability is proven; this is process):
  the candidate's dance == the promoted v3e motion, but the candidate (with tail) has NOT been
  through the signed held-out sim exam (mjlab box deleted). To make the standing-end the
  official show motion: author the return tail into the show motion + re-run the 3x held-out
  exam (needs the GPU box recreated) + re-promote. Hardware behavior is already validated.
- Robot left on onboard 'ai' balance.

## 2026-07-07 (afternoon/eve) — Pipeline generalized (audio+stand-end), box-recreate runbook, ENTRY handoff built, software backlog in flight.
- PIPELINE AUDIO + STAND-END (committed): a video's soundtrack now flows through — ExtractStage
  captures it, ExportStage trims it to the danced WINDOW (the intro-trim, same offset as the
  side-by-side) -> data/audio/<slug>/music.wav -> attach (1.5s lead-in). TrainStage rebuilds the
  deployable with deploy_ramp stand_end=True so trained dances END STANDING. Verified for REAL on
  the laptop (ffmpeg present). deploy_ramp got add_landing_ramp/make_deploy_csv(stand_end=). 2
  stale golden tests un-staled (v3e sharp). Suite 402 passed.
- BOX RECREATE RUNBOOK (docs/BOX_RECREATE_RUNBOOK.md): fast path if Network Volume g1dance-data
  survived vs full re-provision; exact create-form fields, laptop reconnect (cloud.json+hostkey),
  idempotent 00/10/20, + the two goals. TRAINING IS BLOCKED until the user recreates the box
  (laptop has no GPU; console-only, no API).
- ENTRY HANDOFF (committed, deploy_runtime): mirror of the exit overlap. Pre-arm publisher +
  damp-ctx + signal-handler BEFORE releasing onboard (zero setup latency), then CATCH the current
  pose ENTRY_CATCH_S=0.5s the instant onboard lets go, before easing to the ready pose. Closes the
  release-window sag = FALL RISK untethered. Both ground modes. **NEEDS TETHER VALIDATION** before
  untethered use (LED-cue the onboard->policy transition, watch the feet), exactly like the exit did.
  With entry+exit both smooth, the full show flow works with NO damp: remote walks on (onboard 'ai')
  -> button -> [entry handoff] -> dance -> [exit stand handoff] -> onboard 'ai' -> remote walks off.
- SOFTWARE BACKLOG (4-lane workflow in flight): venue registry (multi-venue, active excursion limit),
  policy version store + rollback, pre-show checklist + show-phase ownership model, set-list runner
  robustness + show-time audio. New/owned modules; main wires shows.py/ui/vet after.
- REMAINING SOFTWARE (post-backlog): fall detection+recovery (deploy_runtime-coupled + recovery
  policy needs box), operator-console UI wiring of the above, pipeline shakeout on real videos
  (needs box), adversarial safety re-review. Robot-facing: tether-validate the entry handoff, then
  slack-tether->free untethered runs.

## 2026-07-07 (eve) — Backlog modules BUILT + WIRED into the app. Suite green throughout.
- 4 modules (4-lane workflow; venue lane's structured-output summary failed but code+tests landed):
  pipeline/venue.py (multi-venue registry + active excursion limit), pipeline/policy_store.py
  (content-addressed version store + rollback), pipeline/preshow.py (checklist evaluator +
  show-phase ownership model), pipeline/setlist.py (run-plan w/ audio cues + resume-safe run
  state machine). 56 new tests.
- WIRED into the engine + app:
  - venue -> vet gate: vet_motion._excursion_limit() resolves env-override > ACTIVE venue >
    1.5 fallback (registry error never disables the gate). App: Perform tab VENUE selector
    (switch/add) updates the limit live.
  - policy_store -> shows.promote snapshots every show-ready promotion (non-blocking). App:
    dance detail POLICY VERSIONS list + rollback (rollback restores files + resets to draft;
    the show-ready gate is never bypassed). Backfilled Thriller's current version (e68335aa).
  - preshow -> /api/dances/{id}/checklist (live robot ping + active venue + acks) and
    /api/show-phases. App: show-phase strip (walk-on/dance/walk-off ownership) on Perform.
    (The app's existing per-show checklist wizard was left intact; the richer evaluator is
    available via API for the operator-console polish phase.)
  - setlist -> /api/setlists/{id}/run-plan (audio offsets + all-show-ready blockers).
    (Existing set-list runner UI left intact; endpoint available.)
- Endpoints TestClient-verified; desktop app restarted on :8735 with the new code. Runtime data
  (data/venues, data/policy_store, data/setlists/*/run.json) gitignored.
- SOFTWARE STILL LEFT: fall detection+recovery (deploy_runtime-coupled + recovery policy needs
  box); operator-console polish (wire the preshow evaluator + setlist run-plan into richer UI);
  pipeline shakeout on real videos (needs box); adversarial safety re-review. Robot: tether-
  validate the ENTRY handoff, then untethered.

## 2026-07-07 (eve) — ENTRY HANDOFF + FALL DETECTOR both TETHER-VALIDATED on hardware.
- Method: two short isolation-stand runs (no dancing), user present + tether + remote, LED-cued.
- TEST 1 ENTRY HANDOFF (real threshold): robot took over from onboard and held standing — user
  verdict "stayed put, NO sag" at the onboard->policy release. The pre-arm-before-release + catch-
  current-pose closes the unheld release window. Combined with the already-validated exit stand
  handoff, the full untethered flow (remote walk-on -> button -> dance -> stand -> remote walk-off)
  has NO damp/sag at either handoff.
- TEST 2 FALL DETECTOR (threshold raised to 0.99 for the test so the stand's own ~9-11deg settling
  tilt trips it — this session's stands settled gently, min uprightness 0.981/0.986): fired exactly
  as designed — "FALL DETECTED at tick 2: pelvis 9 deg from vertical (0.99<0.99) for 3 ticks ->
  damping + soft handoff to onboard". User verdict: robot DAMPED + head LED RED. Validates the full
  path on real IMU: read tilt -> 3-tick DEBOUNCE -> raise -> damp -> onboard handoff. The real
  deploy threshold (0.35) is unchanged and telemetry-clean (0 false trips across 26 runs).
- Robot left on onboard 'ai' (fall path restores it). Both were the last robot-validations owed for
  the entry handoff and fall detector; both PASS.
- REMAINING robot-facing: slack-tether -> free untethered runs (the walk-on/dance/walk-off flow),
  then push tests. Software: fall RECOVERY get-up (needs the box), operator-console polish.

## 2026-07-07 (eve) — UNTETHERED PROGRESSION: slack-tether stand PASS, slack-tether full dance CAUGHT THE TETHER at the arm-accent section. NOT free-ready. Stage 3 (free) correctly NOT attempted.
- Staged, gated, user present + remote. Stage 1 slack-tether STAND: rock steady, settled 4.9deg
  tilt / gyro 0.022, NO tether load — the robot self-balances standing free. PASS.
- Stage 2 slack-tether FULL Thriller (standtail candidate, all handoffs + fall net active): completed
  2709/2709, no fall trip, clean stand handoff, actions in-cap, 55C. BUT user: "danced but tether
  caught/loaded." Telemetry localizes the lean to t~15-25s (peaks 16.5/15.0/18.7 deg) = the DYNAMIC
  ARM-PUNCH accents (the sharp-reference crispness). Rest of the dance calm (3-6 deg). Max tilt 18.7
  deg is 51deg from the 70deg fall trigger (not falling) but enough to load the SLACK tether.
- FINDING (honest, gates free-running): the sharp-reference arm accents shift the standing balance
  enough that, WITHOUT tether assist, the robot leans ~18-19deg at the punches. Not a fall, but more
  balance envelope than is safe to test tether-off. The slack-tether stage did its job — caught this
  before a free run.
- PATH TO FREE-READY (needs decisions/box): (a) retrain with stronger push/balance robustness so the
  legs reject the arm-accent disturbance (needs the GPU box); and/or (b) tune deploy (leg-gain boost
  / arm cap) to stiffen leg support during accents; and/or (c) soften the accents (trades the crispness
  the user wanted). NOT free-ready until the accent lean is reduced. Robot left on onboard 'ai'.

## 2026-07-07 (eve) — OPTION 2 (leg-gain tuning) WORKED: GROUND_LEG_KP_SCALE=1.5 made Thriller free-standing-capable. 3x clean slack-tether, tether never loaded.
- Diagnosis: the accent lean (t~15-25s) was ~equal pitch(13.3)/roll(15.4) deg. Code history warned
  boosting the ROLL joints backfires (sideways fall, 2026-07-04) — so only the SAGITTAL boost
  (GROUND_LEG_KP_SCALE, hip_pitch/knee/ankle_pitch) is safe.
- Surprise (measured, not predicted): the 1.5x SAGITTAL boost reduced BOTH components — accent lean
  18.7->~14 deg peak, and the ROLL component 15.4->9-13 deg (a firmer sagittal base steadies the whole
  stance). No oscillation once the robot is PLANTED (the earlier 'oscillation' at 1.5x was the tether
  FLOATING the robot, not the gains — re-run planted was calm: gyro 0.017, tilt 2.9 deg).
- **3x clean boosted slack-tether full dances** (170846/171242/171920): dance-peak tilt 14.0/14.5/14.2
  deg, p99 ~12 deg (baseline 18.7), no fall trips, clean stand handoffs, temps 55-57C (ankles cool),
  and USER CONFIRMED THE TETHER STAYED OFF ALL 3 RUNS. The robot self-balances the full Thriller free.
- DEPLOY CONTRACT UPDATE: this dance requires GROUND_LEG_KP_SCALE=1.5 on the ground (add to the show
  run env / deploy bundle). Arms unchanged (trained gains).
- NEXT: free (tether-OFF) run is now evidence-supported (3x clean slack, tether not loading). User's
  conscious call — highest-risk step (no catch if it falls). If taken: 1.5x boost + full safety spine.

## 2026-07-07 — 🏆🏆 FIRST FULLY UNTETHERED FULL DANCE. Thriller, tether OFF, complete routine, no fall, ended standing. THE MISSION MILESTONE.
- Config: standtail candidate (v3e policy + dance + return-to-standing tail) + GROUND_LEG_KP_SCALE=1.5
  (sagittal leg boost) + full safety spine (entry catch, fall detector @0.35/3-tick, exit stand
  handoff) + ARM_ACTION_CAP_SCALE=2.2. ground-run-legodom, leg-odom estimator.
- Result (telemetry 20260707-172354): 2709/2709 ticks, NO fall trip, clean stand handoff, dance-peak
  torso tilt 13.9 deg (56 deg margin to the 70 deg fall trigger), p99 11.5, arms 15.8/22, legs 5.0/10,
  gyro p95 0.93, temp 57C. **INDISTINGUISHABLE from the 3 slack-tether runs (14.0/14.5/14.2 deg)** —
  proving the tether was never assisting; the robot was already free-balancing. USER CONFIRMED: full
  dance free, ended standing, no fall.
- Path to here this session: slack-tether stand PASS -> slack full dance LOADED THE TETHER at the
  arm-accent section (roll-dominated lean 18.7 deg) -> option-2 leg-gain tuning: sagittal boost 1.5x
  (roll boost is documented to backfire) cut the lean to ~14 deg (a firmer base steadied roll too) ->
  3x clean slack -> 1 clean FREE. All handoffs + fall detector were tether-validated earlier today.
- **DEPLOY CONTRACT for free Thriller: GROUND_LEG_KP_SCALE=1.5** (record on the deploy bundle / show
  run env). Arms trained gains. The standtail motion (ends standing) is the free-capable artifact.
- REMAINING to paid-show grade: free-run REPEATABILITY (3x clean free per the exam standard; got 1),
  music-synced free run, then push tests. Make the standtail+boost config the official show (needs the
  box to re-exam the standtail motion for a signed show-ready). Robot left on onboard 'ai'.

## 2026-07-07 — 🏆 FREE-RUN REPEATABILITY MET: 3x CLEAN untethered full Thriller dances. + start-pose guard built (from a caught near-fall).
- 3 clean tether-off full dances (172354/172748/173645): dance-peak tilt 13.9/14.3/13.5 deg,
  p99 ~11, arms ~15.7/22, legs ~4.9/10, temps 55-57C, all ended standing, all user-confirmed clean.
  Consistent balance, 56 deg margin to the fall trigger. Config: standtail + GROUND_LEG_KP_SCALE=1.5
  + full safety spine.
- ONE run (172942) spiked to 47.5 deg at the dance onset and needed a manual assist — DIAGNOSED as
  a BAD START POSE (robot had been left leaned near-horizontal; move-to-default + policy can't recover
  from there), NOT a balance failure (operator confirmed). Excluded from the count; the clean re-run
  from a verified-upright start had a 7.5 deg onset (vs 47.5), confirming the diagnosis.
- FIX BUILT (start-pose guard): deploy_runtime._check_start_upright() refuses to start any ground run
  if the initial torso uprightness < START_UPRIGHT_MIN (0.85, ~32 deg tilt), checked BEFORE releasing
  onboard so a refusal leaves the robot safely self-balanced. A non-upright start can no longer produce
  that near-fall. +tests (21 deploy_exit green). Operator pre-check: stand the robot upright before the run.
- STATUS: THRILLER RUNS FULLY UNTETHERED, REPEATABLY (3x clean). The mission's core capability is
  demonstrated. Remaining to paid-show: music-synced free run, push-robustness tests, then make the
  standtail+1.5x-boost config the signed show (needs the box to re-exam the standtail motion). Robot on
  onboard 'ai'.

## 2026-07-07 — 🏆🏆🏆 COMPLETE SHOW: MUSIC-SYNCED FREE THRILLER. Untethered dance + real music on-beat + standing finish. The paid-service performance, demonstrated.
- Full show via tools/show_run.sh (MAX_SECS now configurable): AUDIO_MODE=laptop (aux), real Thriller
  track auto-cued at tick0+4.0s (dance start) with 0.08 ms scheduling error, EXIT_MODE=stand,
  GROUND_LEG_KP_SCALE=1.5, full safety spine + start-pose guard. Standtail candidate (v3e + tail).
- Result (telemetry 20260707-174428): 2709/2709 ticks, NO fall trip, clean stand handoff, dance-peak
  14.3 deg (consistent with the 3 silent free runs 13.9/14.3/13.5), onset 4.8 deg (smooth, guard-clean),
  57C. USER CONFIRMED: full dance FREE, ON-BEAT with the music, ENDED STANDING. Music didn't perturb
  balance (separate process, as designed). Logged as a live clean show (outcome-capture path exercised).
- WHAT'S DEMONSTRATED END-TO-END TODAY: video->policy (prior) -> hardware sim2real -> tethered ->
  slack-tether -> leg-gain tuning -> UNTETHERED -> 3x repeatable -> MUSIC-SYNCED FREE SHOW. The mission
  ('a G1 performs a full choreographed dance, balanced, plug-and-play') is DEMONSTRATED on hardware.
- REMAINING to a paying customer: push-robustness tests; promote the standtail+1.5x config to a SIGNED
  show-ready (needs the box to re-exam the standtail motion); operator-console polish; endurance/2-3min.

## 2026-07-07 (eve) — SHOW-PRODUCT SOFTWARE landed + committed; ONBOARD wireless deploy set up on PC2, blocked on a characterized DDS issue (debug runbook written).
- SOFTWARE (workflow, 527 tests green, committed): (a) app one-button FREE show — RUN SHOW opt-in
  'free' runs the hardware-validated untethered config (standtail + GROUND_LEG_KP_SCALE=1.5 + EXIT_MODE
  =stand + MAX_SECS=57 + music), guards unchanged, honest 'not-signed' provenance; (b) side-by-side
  video ON PLAY full-screen on the EXTERNAL display (tools/show_display.py + show_run.sh SHOW_VIDEO
  hook, tick0-anchored so video+music+robot start together); (c) tools/wireless_preflight.py (RTT +
  DDS-staleness GO/NO-GO); (d) docs/LONG_DANCE_PLAN.md + docs/WIRELESS_SHOW.md.
- ONBOARD WIRELESS (user's Unitree-style idea = the RIGHT design; laptop-over-wifi control is a fall
  risk and will NOT be done): SET UP on PC2 — teleimager env (py3.10 + SDK + cyclonedds), onnxruntime
  1.23.2 installed, code+policy bundled to ~/g1-dance, run on eth1 (local control, real-time-safe),
  trigger over wifi/tailscale. BLOCKER: onboard subscriber -> DDS PRECONDITION_NOT_MET on rt/lowstate
  (conflicts with the live master_service). DIAGNOSED: the laptop's ~/robot SDK reads lowstate fine
  (type-compatible with master_service) but PC2's kc_ws SDK (sha 58c3f62) does not; matching the robot
  DDS XML didn't fix it. LEADING FIX: use the laptop's working SDK on PC2. Full debug runbook +
  on-robot session plan: docs/ONBOARD_DEPLOY.md. NOT resolved autonomously (delicate live-control-net
  debugging — needs the operator present).
- EXPLICITLY NOT DONE (told the user): 2-3 min dance (no source video + box deleted); JUMPS (aerial
  R&D, backflip infeasible at true limits, separate policy + hardware-risk decision); wireless show
  (onboard DDS blocker; laptop-wifi control refused as unsafe); autonomous robot MOTION; sign the
  standtail+boost config show-ready (needs box). Side-by-side video is the ~50s Thriller, not 2-3 min.

## 2026-07-07 (eve) — ONBOARD DDS: ROOT CAUSE FOUND (not fixable from a parallel subscriber). Upload path answered. 2-3min/jumps gated on box+video.
- DDS (user: "do the DDS"): thorough attempt, 6 approaches, ALL fail PRECONDITION_NOT_MET. Confirmed via
  a BENIGN-topic probe that it's a DOMAIN-LEVEL TYPE-registration conflict: master_service (C++) owns the
  unitree_hg LowState_ type in the domain; our co-located Python participant can't re-register it (any
  topic). Laptop works only because it's a separate-host wire participant. => A parallel Python DDS
  subscriber co-located with master_service is the WRONG shape. RIGHT path = Unitree's onboard method:
  run the policy INSIDE the control framework/container (qiayuanl/unitree:jazzy) that owns the type, OR
  vendor guidance on XTypes/type-coexistence. Needs sudo/docker + Unitree docs + operator. Fully
  characterized in docs/ONBOARD_DEPLOY.md. NOT solvable by config from our side; stopped guessing at the
  live control net. Onboard setup (env+onnxruntime+code on PC2) stays staged for that path.
- UPLOAD (user: "where do i upload"): the desktop app (:8735) Create/Studio tab -> "Choose file..." or
  drag-drop the 2-3 min video -> POST /api/jobs/upload -> pipeline job. (The app + pipeline are wired.)
- 2-3 MIN + JUMPS ("find a way + do it"): the WAY = upload the in-place 2-3min video (user) + recreate
  the GPU box (user console clicks; docs/BOX_RECREATE_RUNBOOK) -> the app pipeline extracts/retargets/
  trains/exams it. JUMPS: analyzable once retargeted (feasibility like check_acro_reference) — a small
  Thriller hop may be feasible where the backflip was not; but it's a dynamic-skill track (own profile +
  validation), NOT a drop-in. HARD GATE remains box + video (neither exists yet); cannot execute here.

## 2026-07-07 (eve) — ONBOARD BREAKTHROUGH + GPU wall + job triage (5-directive turn).
- ONBOARD/WIRELESS (#3/#4): the robot's g1-siu-deploy:jazzy container (docker group, no sudo) HAS
  /ws/src/motion_tracking_controller = the BeyondMimic onboard controller (architecture's original target).
  Running OUR policy in IT sidesteps the DDS type conflict (it's the framework that owns the type).
  HIGH compat: identical joints, matching default pose, torso_link anchor, motion_anchor_pos_b obs. MUST
  override its 350/300 gains with OUR trained 40/99/28 (staged ~/onboard_deploy on PC2 + config values).
  Remaining (operator-present, first onboard run): verify motion format + obs byte-match, patch gains,
  gantry->tethered staircase, then wireless trigger. Control loop stays onboard (eth1); wifi = trigger only.
  This is the correct wireless path. docs/ONBOARD_DEPLOY.md has the full plan.
- GPU (#2): FULL AUTONOMY GRANTED but GreenNode has NO API/CLI (console-only, confirmed) — I CANNOT
  create a box programmatically. Autonomy can't overcome a missing mechanism. Options: (a) you do the
  console clicks (docs/BOX_RECREATE_RUNBOOK fast path), or (b) give me API creds for an API-driven GPU
  provider (RunPod/Lambda/Vast) and I'll fully automate create/train/delete. Until then training is gated.
- 2-MIN VIDEO (#1/#5): uploaded ('Thriller dance FULL 2min'), blocked at EXTRACT because it scp's the
  video to the deleted box (GVHMR runs on the box). => gated on #2 (a box). Pipeline itself is working.
  Other Create jobs are stale (dance1-e2e verify-failed known; old thriller train-blocked).

## 2026-07-10 — FRONTEND FEEDBACK REVISION: accessible Show mode + clickable simulation preview.
- Restored the operator essentials requested after the first React pass: direct `Show mode` navigation,
  an always-visible physical damping-remote warning with the exact typed unlock, pre-show checklist,
  setlists, venues, control-ownership phases, and the oversized run-time STOP path. Backend endpoint
  semantics and the pywebview/FastAPI production path are unchanged.
- Added a reusable animated MuJoCo-style robot environment with a friendlier code-native robot mark.
  Preview stages use real dance/job preview URLs and open an in-app HTML5 video player with native
  controls; the stage appears on Overview, Pipeline, Dances, and Show mode.
- Visual language now follows the locally documented Aeolus/Maestro/Obsidian preferences: light flat
  canvas, crisp punched white cards, one dark utility card, functional hover lift, compact information
  hierarchy, `minmax(0,1fr)` responsive grids, and reduced-motion support. The safety warning remains red
  and high contrast rather than being visually minimized.
- VERIFIED against the real local server: `npm run build`; Playwright 7/7 at 1440/1024/768 including
  multipart upload, run-time STOP, exact typed confirmation, clickable video preview, and audit filters.
  Evidence: `docs/ui_revamp/show-mode-1280.png` and `preview-video-open.png` plus refreshed breakpoint shots.
