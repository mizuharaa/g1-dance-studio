# GreenNode box — recreate & re-provision runbook

**When to use:** the GreenNode 4090 notebook was DELETED (2026-07-07). This brings a
fresh one back to the exact working state so the app's `video → train → sim-verified`
pipeline (and the standing-end re-exam) can run. Account creation/billing is NOT
repeated here — see `docs/GREENNODE_SETUP.md` Parts A–B if credit ran out.

**Time:** ~10 min if the Network Volume `g1dance-data` survived (fast path); ~45–60 min
if it was also deleted (full re-provision, mostly install waits).
**Cost while up:** ~18k VND/h (~$0.70/h). **DELETE it again when done** — billing runs
creation→deletion; a Stop saves nothing meaningful for us.

---

## What's already on the laptop (nothing to recover from the box)
- Provisioning scripts: `cloud/00_bootstrap.sh`, `cloud/10_gvhmr.sh`, `cloud/20_training.sh`,
  `cloud/run_job.sh`, all task/recipe/eval scripts. These are idempotent and re-run every
  time a box is created (block storage is wiped on Stop/Delete).
- SSH keypair: `.secrets/greennode_ssh_key(.pub)` (public half also in GREENNODE_SETUP.md).
- W&B key: `.secrets/wandb.key`. Body models: `data/body_models/` (license-gated — synced
  up, never downloaded on the box).
- Retention checkpoints: `data/checkpoints/{v3e_model_9999,s2rb_model_4999,v3c_model_9000,
  s2r_model_4999}.pt` — restore these to the box only if you need to resume a prior run.
- Connection config the app reads: `.secrets/cloud.json` (`ssh.host` / `ssh.port` — you WILL
  update these). Network Volume S3 creds: `.secrets/cloud-connect.txt` (volume `g1dance-data`).

---

## Part 1 — Did the Network Volume survive? (decides fast vs full path)
In the console, open **Storage management → Network Volume**. Look for **`g1dance-data`**
(HCM region).
- **Present** → FAST PATH. Re-attach it at creation; everything under
  `/workspace/notebook-data` (`envs/mjlab`, `repos/`, `cloud/`, `motions/`) is intact and
  the provision scripts will be near-instant no-ops.
- **Gone** → FULL PATH. Create a new Network Volume (Part C of GREENNODE_SETUP.md: name it
  `g1dance-data`, ~50–100 GB) and expect the full re-provision in Part 4.

## Part 2 — Create the notebook (~5 min; ref GREENNODE_SETUP.md Part D for screenshots)
Console → **AI Platform → Notebook → Create**. Key fields:
- **Resource**: family `GPU-CODE-RTX4090`, smallest type `aiplatform-standard-16x64-1rtx4090`
  (16 CPU / 64 GB / 1×4090). Read the hourly price on-screen before confirming (~18k VND/h
  after the 25% internal-team discount; +10% VAT; month-end reconciled).
- **Data mount**: select Network Volume `g1dance-data`; mount folder = **`notebook-data`**
  (→ `/workspace/notebook-data`). This field is REQUIRED.
- **SSH public key** (paste inline):
  `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICTYBj0plfDnYAhJ5HKoH8yP2ePz7d1Z6HIKz/+kScfJ g1dance-laptop`
- **HTTP ports**: leave 8888. **TCP ports**: add **22** (SSH).
- Image: the fixed **PyTorch 2.5.1 / CUDA 12.4** image is fine (mjlab is our trainer;
  Isaac Lab is dead on it — do not waste time on Isaac).
- Create, wait for **Running**, open **Connect** and copy the **SSH host + port**.

## Part 3 — Point the laptop at the new box (~2 min)
The new box has a new IP/port and new SSH host keys.
```bash
cd ~/g1-dance
# 1) update the app's connection config with the new host + port from the Connect dialog:
python3 - <<'PY'
import json
p=".secrets/cloud.json"; d=json.load(open(p))
d["ssh"]["host"]="<NEW_HOST>"; d["ssh"]["port"]=<NEW_PORT>
open(p,"w").write(json.dumps(d,indent=2)); print("updated", d["ssh"])
PY
# 2) clear any stale host key for that host:port, then trust the new one on first connect:
ssh-keygen -R "[<NEW_HOST>]:<NEW_PORT>" 2>/dev/null || true
# 3) verify SSH + GPU:
ssh -i .secrets/greennode_ssh_key -p <NEW_PORT> -o StrictHostKeyChecking=accept-new \
  root@<NEW_HOST> 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```
Expect `NVIDIA GeForce RTX 4090, 24564 MiB`.

## Part 4 — Re-provision (fast no-op if the volume survived)
```bash
cd ~/g1-dance
H=<NEW_HOST>; P=<NEW_PORT>; K=.secrets/greennode_ssh_key; NB=/workspace/notebook-data
SSH="ssh -i $K -p $P root@$H"; SCP="scp -i $K -P $P"

# a) always re-push the cloud scripts (they may have changed on the laptop):
$SSH "mkdir -p $NB/cloud"
$SCP -r cloud/* root@$H:$NB/cloud/
# b) W&B key + body models (body models only needed for NEW video extraction):
$SCP .secrets/wandb.key root@$H:$NB/.wandb_key
$SSH "test -d $NB/body_models" || $SCP -r data/body_models root@$H:$NB/    # skip if present
# c) run the idempotent provisioners (fast if the volume survived):
$SSH "cd $NB/cloud && bash 00_bootstrap.sh"
$SSH "cd $NB/cloud && bash 10_gvhmr.sh"          # only needed for VIDEO input; ~10min first time
$SSH "cd $NB/cloud && bash 20_training.sh mjlab" # mjlab is the trainer; ~15min first time
# d) confirm the trainer is ready:
$SSH "cat $NB/reports/training_stack.json"       # expect mjlab_ready: true
```
If the volume was recreated fresh, also re-stage the motions you want to train/verify, e.g.:
```bash
$SSH "mkdir -p $NB/motions"
$SCP data/policies/thriller/thriller_deploy.npz root@$H:$NB/motions/
```

## Part 5 — Smoke test (30 s of GPU, proves the stack)
```bash
$SSH "cd $NB && MUJOCO_GL=egl ./envs/mjlab/bin/python -c \
  'import mjlab, torch; print(\"mjlab ok, cuda\", torch.cuda.is_available())'"
```

---

## Part 6 — Now do the two things the box was needed for

### A) Make the standing-end Thriller show-ready (re-exam, ~light)
The stand handoff is hardware-validated; what's missing is a signed sim exam on a
standing-end motion. Cleanest is to RETRAIN Thriller on the sharp motion **with the
return-to-standing tail** (the pipeline now emits this via `deploy_ramp stand_end=True`),
then the standard gate. On the box, per `cloud/V3_PROGRAM.md` + `logs/jobs.md`:
1. Build the deploy motion with the tail on the laptop:
   `python -m pipeline.deploy_ramp --in <thriller_sharp_show>.csv --out thriller_standend_deploy.csv --stand-end`,
   push it, `csv_to_npz` on the box.
2. `run_job.sh start train-thriller-standend -- "... train_sim2real_v3.py ...-S2R-V3C
   --env.commands.motion.motion-file motions/thriller_standend_deploy.npz ..."` (v3e recipe).
3. On converge: `export_policy.py` → `heldout_eval.py` ×3 (seeds 90001/90011/90021, 256 env)
   → pull → `pipeline/mjlab_verify.py` sign → `attach_policy` + 3× `record_sim_run_from_verdict`
   → promote in the Shows page. Deploy with `EXIT_MODE=stand` for the standing finish.

### B) Train a new dance from a video (the general pipeline)
In the desktop app (already launched, or `bash scripts/dance-studio`): **New dance → upload
video (with audio)**. The pipeline now: extracts the soundtrack + aligns it to the danced
window (music), retargets with a standing-end tail, blocks at the **Approve training** gate
(review the preview), then on approval runs on THIS box: convert → train → export → held-out
exams → sim-verified CANDIDATE with music. Promotion to show-ready stays a human click on the
Shows page. Per-dance knobs: `dance.yaml` in the job dir (`docs/NEW_DANCE_PLAYBOOK.md`).

---

## Teardown
When the training/exam you needed is done and all artifacts are pulled to the laptop
(policies, `heldout_verdict_*.json`, ONNX, checkpoints), **DELETE the notebook in the
console** (not just Stop). The Network Volume `g1dance-data` can stay (cheap) so the next
recreate is a fast path — or delete it too if you're done for a long while.
Verify it's gone: `ssh … 'echo up'` should give `Connection refused`.
