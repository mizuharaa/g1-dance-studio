# New Dance Playbook — video file in, show-ready candidate out

**Status:** wired 2026-07-06 (app pipeline generalized from the hand-driven Thriller flow).
This is the operator's dry-run script. The app does everything except the three human
gates: **preview approval**, **promotion**, and **robot day**.

The pipeline reproduces the exact recipe that made Thriller show-ready
(PROJECT_STATE 2026-07-05..06): GVHMR → GMR (`--velocity-limit`) → window/vet →
prep (`prep_motion` + 2.5 s activation ramp) → `csv_to_npz` → sim2real training
(`Mjlab-Tracking-Flat-Unitree-G1-Sim2Real` + the s2r-b root-pos delta) → ONNX export
with `policy_meta.json` sidecar → sim-gap gate v3 → 3× held-out exams → signed
`sim_exam/v1` verdicts → dance registered sim-verified with the deployable-CSV binding.

---

## 0. One-time prerequisites (already true today)

- GPU box alive and configured in the app (Studio → Cloud GPU → green dot).
  Box: `root@103.245.250.152:46936`, key `.secrets/greennode_ssh_key`, ssh transport.
- Box provisioned: `cloud/00_bootstrap.sh`, `cloud/10_gvhmr.sh` (GVHMR env + ckpts),
  `bash cloud/20_training.sh mjlab` (mjlab env), body models synced, and the
  `cloud/*.py` scripts present in `/workspace/notebook-data/cloud/`
  (`train_sim2real.py`, `sim2real_task.py`, `export_policy.py`, `sim_gap_check.py`,
  `heldout_eval.py`, `run_job.sh`).
- Laptop: `g1dance` conda env, `third_party/GMR` installed, MuJoCo G1 model present.

## 1. Film / pick the video

Single continuous shot, tripod, one person, full body in frame, 15 s – 4 min.
In-place choreography strongly preferred (vet gate: ≤1.5 m root excursion; venue = 2 m
radius). VFR/odd fps is fine — the app re-encodes to constant 30 fps itself.

## 2. Operator steps (app clicks — there is no training command)

1. **Upload** the video in the Studio page (or `POST /api/jobs` with a path).
   The job name becomes the dance name; `data/policies/<slug>/` derives from it.
2. *(Optional)* drop a **`dance.yaml`** in `data/jobs/<job-id>/` to override knobs
   (see §4). No file = the promoted Thriller recipe.
3. Wait for **extract → retarget** to finish (fully automatic):
   intake probe → 30 fps re-encode → push → GVHMR on the box (~9 min for a 45 s
   clip) → pull → GMR retarget (laptop, velocity-limited) → grounding → window →
   **vet gate** (hard-fails the job if the motion is undeployable) → MuJoCo preview
   → show prep (pads/blends/ground fix) → `<slug>_deploy.csv` (2.5 s activation ramp).
4. **HUMAN GATE 1 — watch the preview** in the job card, check the vet table, then
   click **Approve training** (confirms a ~2–3 h GPU spend). Nothing reaches the GPU
   before this click.
5. Walk away. The app runs, honestly reporting each blocked/polling state
   (survives laptop reboots — box jobs run in tmux, the app re-polls on restart):
   - `csv_to_npz` on the box (minutes),
   - `train_sim2real` (~2–3 h at 4096 envs / 5000-iter cap; live iter/reward in the
     job card and the System panel; ~45 k VND ≈ $1.7 of GPU),
   - ONNX export + artifact pull → `data/policies/<slug>/{policy.onnx,
     policy_meta.json, <slug>_deploy.csv, <slug>_deploy.npz}`,
   - **sim-gap gate v3** (full motion, 7 injected conditions, ankle-torque +
     drift + quality bars; ~30–45 min),
   - **3× held-out exams** (seeds 90001/90011/90021, 256 envs each, nominal+push;
     ~15 min each),
   - verdicts signed (`pipeline/mjlab_verify.py`) and recorded through the guarded
     shows API → dance appears in the library as **sim-verified** with 3 clean runs.
6. *(Optional music)* put the licensed track at **`data/audio/<slug>/music.wav`**
   (any of .wav/.mp3/.m4a/.aac/.ogg/.flac named `music.*`) before the export stage,
   or attach later from the Shows page. Alignment is automatic: 1.5 s lead-in on the
   show timeline (the deploy runtime adds the 2.5 s activation ramp before that —
   music cue = policy start + 4.0 s).
7. **HUMAN GATE 2 — promote**: Shows page → dance → *Promote to show-ready*.
   The guard rails require the 3 consecutive clean **signed** exams and re-hash the
   policy on disk against the exam-pinned sha. The app never auto-promotes.
8. **HUMAN GATE 3 — robot day** (outside the app, user present, damping remote in
   hand): build/authorize the deploy bundle, then the staged tether progression per
   `docs/GROUND_TETHERED_RUNBOOK.md` (5 s → 15 s → 30 s → full). The app has no
   robot code path — by design.

## 3. Expected timeline

| Step | Wall clock |
|---|---|
| upload + intake + 30 fps re-encode | ~1–3 min |
| GVHMR on the box (45 s clip) | ~9 min (+ upload) |
| GMR retarget + vet + preview + prep (laptop) | ~5 min |
| **human: preview review + approve** | you |
| csv_to_npz (box) | ~2–5 min |
| sim2real training (5000 iters, 4096 envs) | ~2–3 h |
| ONNX export + pull | ~5 min |
| sim-gap gate v3 (128 envs, full matrix) | ~30–45 min |
| 3× held-out exams (256 envs each) | ~45 min |
| **total** | **≈ 3.5–5 h**, two touchpoints before promotion |

## 4. Per-dance knobs — `dance.yaml` (all optional)

Defaults are the promoted Thriller recipe; unknown keys are a hard error.

```yaml
# data/jobs/<job-id>/dance.yaml
iterations: 5000          # training cap (cost ~8.9k VND / 1000 iters)
num_envs: 4096
task: Mjlab-Tracking-Flat-Unitree-G1-Sim2Real   # the CURRENT BEST recipe
eval_task: Mjlab-Tracking-Flat-Unitree-G1        # exams run on the stock task id
extra_train_args:         # appended to train_sim2real; default carries the
  - "--env.rewards.motion_global_root_pos.weight"   # s2r-b drift-fix delta
  - "1.0"
heldout_seeds: [90001, 90011, 90021]
heldout_num_envs: 256
gap_check_num_envs: 128
window_start_s: null      # explicit motion window (seconds into the retarget);
window_end_s: null        #   null = the vet gate's longest deployable window
velocity_limit: true      # GMR joint-velocity clamp during retarget (keep on)
```

Note on **action caps**: the per-joint deploy action caps (legs 10 / arms ×1.6) are a
*runtime* safety knob in `pipeline/deploy_runtime.py`, calibrated from telemetry on
robot day — they are intentionally NOT a training parameter here.

## 5. When a gate fails (honest failure states, all resumable via Retry)

- **Vet gate fail** (job fails at retarget): excursion/limits/floorwork problem.
  Re-shoot, or set `window_start_s`/`window_end_s` to cut the offending section.
- **Sim-gap gate fail** (job fails at verify with the numbers in the message):
  read `data/jobs/<id>/verify/gap_check.json` per-section stats. Options:
  a targeted, music-sync-preserving choreography edit
  (`tools/edit_choreography.py`), or a recipe delta via `extra_train_args`.
  Either way, start a **fresh job** with the `dance.yaml` — this job's training
  is already complete, so Retry on it only re-runs the verify gates.
- **Held-out exam below the ≥99 % bar**: the dance stays draft; the s2r-b history
  says a single reward delta is usually enough — put it in a fresh job's
  `dance.yaml`. (Retry on the failed verify stage re-runs the exams as-is, which
  is only useful after a transient box problem.)
- **Box unreachable / laptop reboot**: stages re-block with the reason and the poll
  loop (or startup reconciliation) resumes automatically; box jobs keep running in
  tmux either way.

## 6. Where things land

```
data/jobs/<job-id>/            job state, per-stage outputs, log.txt
  retarget/<slug>_deploy.csv   the deployable motion (also copied below)
  verify/heldout_verdict_s*.json  signed verdicts (copies in the policy dir)
data/policies/<slug>/          THE deploy consumption contract:
  policy.onnx, policy_meta.json, <slug>_deploy.csv, <slug>_deploy.npz,
  gap_check.json, heldout_verdict_s{1,2,3}.json
data/dances/<dance-id>/        library record: motion_csv BOUND to the deployable
                               csv, policy sha pinned after exams, audio record
box:/workspace/notebook-data/  motions/<slug>_deploy.{csv,npz}, exports/app_<slug>_*,
                               jobs/*.status.json + logs (pollable, reboot-safe)
```

## 7. What stays human, forever

1. Approve training (after watching the preview).
2. Promote to show-ready (guarded by the signed-exam machinery).
3. Everything robot-facing: bundle authorization, tether rig, damping remote,
   staged runs. `CLAUDE.md` safety rules apply unchanged.
