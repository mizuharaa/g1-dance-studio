# AGENT HANDOFF — start here

You are an AI agent taking over the G1 Dance project. This is your action-oriented resume.
For the full narrative read `HANDOVER.md`; for the day-by-day log read `PROJECT_STATE.md`;
for the rules read `CLAUDE.md`. **Credentials are NOT in this repo** — see the last section.

Goal of the project: a Unitree G1 humanoid dances the full Thriller **untethered, to real music,
one-button**, side-by-side ref|sim video on an external screen, robot always self-balancing.
RL motion tracking via BeyondMimic/mjlab (`Mjlab-Tracking-Flat-Unitree-G1-Sim2Real`).

---

## ✅ WHAT'S DONE
- Full training pipeline works (video→retarget→train→verify→export) and the GreenNode GPU flow
  is permanently fixed (isolated mjlab venv — no more GL/scipy cascade).
- A Thriller policy (`thriller_csv_ankle_penalty`) was trained, verified, promoted, and **ran on
  hardware for ~44 s cleanly** before falling.
- **Fall root-caused (decisively): a sim2real LATENCY gap.** Trained latency ≤20 ms; real hardware
  latency is 40–80 ms (measured 4 ways). Evidence: `data/telemetry/latency_diag_20260709/`.
- **Fixes committed**: (1) latency DR widened 0→80 ms (`cloud/sim2real_task.py`); (2) `sim_gap_check`
  now GATES survival at 40 ms+push (was 20 ms) + adds 60/80 ms stress lines; (3) sim/ref video
  desync fixed (`tools/render_deploy_sim.py`); (4) app STOP button; (5) stand-exit handback wired.
- Repo pushed to the handoff remote `git@github.com:mizuharaa/unitree_g1_dance.git` (full history).

## 🔄 CURRENT STATE (as of 2026-07-10 ~11:00 ICT)
- **A latency-robust retrain is RUNNING on the GPU box** — this is the live task.
  - Box `nb-9c7ba766-...`, ssh `root@103.245.250.152:59613` (RSA key). tmux session `train`,
    log `/workspace/notebook-data/train_lat.log`. run-name `train-thriller_lat80-2607`.
  - 4096 envs, 5000 iters. Was ~40 min from finishing. Verify plan below.
- App can run headless: `python3 ui/server.py --host 127.0.0.1 --port 8735`.

## ❗ CURRENT PROBLEM (what the retrain is fixing)
The deployed policy drifts and falls ~45 s into the dance because it was never trained for the
robot's true 40–80 ms sensorimotor latency (actuation + leg-odometry estimation; NOT comms — comms
is 0.16 ms wired). The retrain adds 0–80 ms latency DR + a stronger root-position weight (drift).
**Until validated on hardware, the untethered dance is not solved.**

## 📋 WHAT NEEDS TO BE DONE (in order)
1. **Finish the retrain** when training stops (checkpoint `logs/rsl_rl/g1_tracking/*train-thriller_lat80*/model_4999.pt`).
   Run over SSH (templates in `pipeline/stages/cloud_motion.py`: EXPORT/GAP/EXAM scripts):
   - `cloud/export_policy.py <ckpt> <npz> <exports>` → `policy.onnx`
   - `cloud/sim_gap_check.py --checkpoint <ckpt> --motion-file <npz> --num-envs 128 --output-file gap.json`
     → **must pass the new 40 ms+push gate**; inspect 60/80 ms survival too.
   - `cloud/heldout_eval.py <task> --checkpoint <ckpt> ...` ×3 (seeds 90001/90011/90021, 256 envs).
2. **Pull artifacts, sign** (`pipeline/mjlab_verify.py`), attach + `record_sim_run_from_verdict`,
   **promote** in the Shows page.
3. **DELETE the GPU box** the moment artifacts are pulled + md5-verified local (billing; the owner is
   emphatic). If deletion fails, keep it busy — never idle.
4. **Hardware validation** (needs the human + damping remote): run the show with the new policy.
   This finally tests BOTH the balance fix AND the still-unvalidated stand-exit handback. Bring
   tether slack (drift). Watch for the ~45 s buckle recurring.
5. **Fix the show video player**: it renders colourful static (VLC "Too high recursion" bug). Content
   is correct; switch `tools/show_display.py` off VLC to `mpv`/`ffplay`.

## ⚠️ HARD RULES (from CLAUDE.md — do not violate)
- Robot has NO torque-cut e-stop. Motion ONLY with human present + tether + **damping remote in hand**,
  motion MuJoCo-verified, `--i-will-watch-the-robot` + `CONFIRMED_BY_HUMAN=alois`.
- Never modify `~/robot/`. Stop a run by signalling the **python** pid (not the bash wrapper).
- Measurement discipline: no "decisive" finding without an independent cross-check; commit every
  measurement script AND its raw output.
- GreenNode: no API — drive the console via `tools/pilot.py` (Chrome). SSH keys must be **RSA**
  (ed25519 rejected). At box-create you MUST add **TCP port 22** + select the RSA key.
- DELETE the GPU box when done. Update `PROJECT_STATE.md` after every meaningful step; commit often.

## 🔑 CREDENTIALS
Not in this repo (`.secrets/` is gitignored). On the original laptop they are in `.secrets/`; the
full list + login values is in **`.secrets/CREDENTIALS.md`** (also gitignored). To operate on a new
machine, obtain the entire `.secrets/` directory from the owner over a secure channel. It contains:
box SSH RSA key, GreenNode console login, W&B key, the GitHub handoff deploy key, and the Chrome-pilot
profile. **A GitHub PAT was exposed earlier in the project — rotate it.**
