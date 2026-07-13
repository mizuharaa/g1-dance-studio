# RETRAIN RUNBOOK — Thriller (latency-curriculum v5 + clean motion)

Step-by-step to retrain the Thriller policy so it (a) is robust to the real 40–80 ms
sensorimotor latency and (b) is trained on the **de-glitched** motion (fixes the twitchy
limb-snapping). Written 2026-07-13. Assumes you are on the laptop with `~/g1-dance/.secrets/`
present (SSH RSA key, GreenNode login, W&B key). If `.secrets/` is missing, get it from the
owner first — nothing here works without it.

> **Why retrain at all?** Current status: the `lat80` retrain FAILED (survival 0.000, drift
> 2–7 m — `data/telemetry/latency_retrain_20260710/RESULT.md`); the deployed policy was
> trained on **unfiltered** motion (jerk peak 101,701 rad/s³). The v5 recipe fixes both:
> staged latency curriculum (not a blunt 0–80 ms band) + arm-fidelity terms, run on the
> cleaned motion.

---

## 0. Prerequisites (once)
- `.secrets/` on the laptop. `export WANDB_API_KEY=$(cat ~/g1-dance/.secrets/wandb.key)`.
- The v5 recipe is already in the repo: `cloud/sim2real_task_v5.py`, `cloud/train_v5_curriculum.sh`.
- Budget check: ~10k iters @ ~2040 it/hr ≈ 5 h box time ≈ **~90k VND (~$3.5)**. Owner cap 1.5M VND.

## 1. Prepare the clean motion (LAPTOP, CPU — no GPU)
The twitch filter (`tools/motion_quality.clean_motion`) is wired into `prep_motion` as of
commit 16f6aa7, but the deployed policy predates it. Re-prep the raw Thriller retarget:

```bash
cd ~/g1-dance
python -m pipeline.prep_motion --in data/motions/thriller/thriller_g1.csv \
                               --out data/motions/thriller/thriller_g1_clean.csv
# expect: jerk_peak 101701 -> ~4800, spike_frames 67 -> 4, dof_rms_delta ~0.03 rad (sharp kept)
python tools/vet_motion.py data/motions/thriller/thriller_g1_clean.csv --json | tail -30   # must PASS
```
Convert to the training `.npz` (mjlab's own converter, no Isaac):
```bash
python cloud/csv_to_npz_mjlab.py data/motions/thriller/thriller_g1_clean.csv \
       --input-fps 30 --output-fps 50 --output data/motions/thriller/thriller_clean.npz
```
(If `csv_to_npz_mjlab.py` is not the exact name on your tree, use the converter referenced in
`pipeline/stages/cloud_motion.py`.)

## 2. Create + provision the GPU box (GreenNode — console only, NO API)
GreenNode has no CLI/API — the create form is clicked in the console (drive it with
`tools/pilot.py` Chrome, or by hand). Follow `docs/BOX_RECREATE_RUNBOOK.md`. Non-negotiables:
- **RSA** SSH key (ed25519 is rejected). Add **TCP port 22** on the create form. Select the RSA key.
- Flavor `aiplatform-standard-16x64-1rtx4090`. Attach the Network Volume (required).
- After boot: `ssh -p <port> root@<ip>` (RSA key). Reinstall the isolated mjlab venv per
  `docs/GREENNODE_SETUP.md` (the PyTorch-2.5.1 image needs the isolated venv — no Isaac Lab).
- `scp` the clean `.npz` to `$NB/notebook-data/motions/`.

## 3. Train — v5 latency CURRICULUM (ON THE BOX, in tmux)
```bash
tmux new -s train
export NB=/workspace/notebook-data
MOTION=$NB/motions/thriller_clean.npz bash cloud/train_v5_curriculum.sh
```
Stages (the script drives them; ~10k iters total, resuming each stage):
1. **0–20 ms** cmd/obs, 4000 iters — learn the dance + station-keeping first.
2. **0–50 ms**, +3000 iters (resume s1).
3. **0–60 ms**, +3000 iters (resume s2). *60, not 80 — sim PD already models mechanical lag.*

> ⚠️ **UNVERIFIED on this mjlab version:** the resume flag names
> (`--agent.resume/--agent.load-run`). Before stage 2 runs, confirm:
> `$PY $NB/cloud/train_sim2real_v5.py <TASK> --help | grep -i resume`. If they differ, the
> curriculum silently restarts from scratch — check each stage resumed (reward continues, not resets).

Watch: `motion_global_root_pos` reward should CLIMB (the lat80 failure stalled it at 0.05);
mean episode length should approach the full dance, not ~4.6 s.

## 4. Verify (ON THE BOX) — the gate BEFORE anything touches hardware
The curriculum script auto-runs the verify chain at the end; if you run it by hand:
```bash
CKPT=$(ls -t $NB/logs/rsl_rl/g1_tracking/*v5fid-s3*/model_*.pt | head -1)
$PY $NB/cloud/export_policy.py "$CKPT" "$MOTION" "$NB/exports/v5fid"           # -> policy.onnx + meta
$PY $NB/cloud/sim_gap_check.py --checkpoint "$CKPT" --motion-file "$MOTION" \
    --num-envs 128 --output-file "$NB/exports/v5fid/gap.json"
for S in 90001 90011 90021; do
  $PY $NB/cloud/heldout_eval.py Mjlab-Tracking-Flat-Unitree-G1-S2R-V5 --checkpoint "$CKPT" --seed $S --num-envs 256
done
```
**HARD GATES (all must hold, else do NOT deploy):**
- `gap.json`: survival passes at **40 ms + push** (the gate that correctly refused lat80). Inspect 60/80 ms lines.
- **nominal root drift < 1 m** (lat80 drifted 2–7 m — the exact failure to avoid).
- held-out survival **≥ 99%** across the 3 seeds (the show-ready bar).

## 5. Pull, sign, promote (LAPTOP)
```bash
scp -P <port> root@<ip>:$NB/exports/v5fid/* data/policies/thriller_v5fid/     # onnx+meta+gap
md5sum data/policies/thriller_v5fid/policy.onnx                                # record; verify local == box
python pipeline/mjlab_verify.py --exports data/policies/thriller_v5fid --gap data/policies/thriller_v5fid/gap.json
#   -> writes a SIGNED heldout_verdict.json (records BOTH npz + csv sha)
```
Then in the app (Shows page): attach the policy → the signed verdict flows through
`record_sim_run_from_verdict` → **promote** to show-ready (the gate refuses without the signed ≥99% verdict).

## 6. DELETE THE BOX (billing — owner is emphatic)
The moment artifacts are pulled + md5-verified local: delete the notebook in the GreenNode
console (no API — click it, or `tools/pilot.py`). Stop does NOT stop billing; **delete**. If
deletion fails, keep the box busy — never leave it idle and billing.

## 7. Hardware validation (needs the human + damping remote — CANNOT be automated)
Gantry-first, then slack-tether, then free — per `PROJECT_STATE.md` progression. Bring tether
slack (drift). **Watch for the ~45 s buckle recurring** (the drift/latency failure this retrain
targets). This run also finally tests the still-unvalidated **stand-exit handback** (§ exit-fix).
Start-pose upright + **feet flat on the ground** (see Safety tab) before arming.

---

### Quick "do I even need to retrain?" checklist
- Twitchy/limb-snapping only? → retrain on the **clean** motion (§1) is enough; latency curriculum is additive.
- Drifts/buckles ~45 s in? → you need the **latency curriculum** (§3) — this is the primary fix.
- Both → the v5 recipe on the clean motion does both in one run.
