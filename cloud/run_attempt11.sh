#!/usr/bin/env bash
# Attempt 9 (v12 fidelity-motion) — LEG-FIDELITY fix on the WARM v10 box. Native tempo.
#
# v11 = v10 + a dedicated leg-tracking reward + softened stance/drift so the
# lower body stops losing to the arms (v10 rollout: legs 43-59% vs arms 77-137%
# of the reference amplitude in the arm-swing section — a reward asymmetry, not a
# capability limit). See cloud/sim2real_task_v14.py header.
#
# Motion input (audit F2): the hash-bound v12 bundle built by
# tools/build_motion_bundle.py — motions/v12_bundle/{bundle.json,final.csv,
# scorecard.json,source.csv}. The manifest is verified below BEFORE any
# retime/convert; tempo npz are regenerated whenever the source CSV hash
# changes (a stale prior-attempt npz would silently train the OLD motion — its
# frame count matches, so the frame check alone can't catch it).
#
# Push first (laptop) — FRESH BOX (all prior boxes deleted 2026-08): the v14
# task imports the whole chain, push ALL of it + the shared eval/verify scripts
# (simplest: push the entire cloud/ dir), and the v12_bundle motion dir:
#   scp -P <PORT> -i .secrets/greennode_rsa cloud/*.py cloud/*.sh \
#     root@<IP>:/workspace/notebook-data/cloud/
#   scp -P <PORT> -i .secrets/greennode_rsa -r data/motions/thriller/v12_bundle \
#     root@<IP>:/workspace/notebook-data/motions/
# (chain: sim2real_task.py -> v5 -> v6 -> v8 -> v10 -> v11 -> v13 -> v14; plus
#  pick_checkpoint / sim_gap_check / heldout_eval / export_ckpt_onnx /
#  verify_obs_layout / retime_motion — all in cloud/.)
# Then on the box:
#   export G1_GATE_ANKLE_P95_NOMINAL_NM=22 G1_GATE_ANKLE_P95_WORST_NM=25
#   cd /workspace/notebook-data && nohup bash cloud/run_attempt11.sh > attempt11.out 2>&1 &
#
# $$$ Box already billing since the v10 run — this reuses it. Pull + DELETE after.
set -uo pipefail
NB=${NB:-/workspace/notebook-data}
PY=$NB/envs/mjlab/bin/python
CSV=${G1_MOTION_CSV:-$NB/motions/v12_bundle/final.csv}
BUNDLE=$NB/motions/v12_bundle
CSV2NPZ=$NB/repos/mjlab/src/mjlab/scripts/csv_to_npz.py
[ -f "$NB/.wandb_key" ] && export WANDB_API_KEY=$(tr -d '[:space:]' < "$NB/.wandb_key")

cd "$NB"
say() { echo "== $* == $(date -Is)"; }
die() { echo "!! $*"; exit 1; }

nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader >/dev/null || die "no GPU"

# ---- bundle manifest verification (audit F2: train only hash-bound bytes) ----
# Self-contained on purpose: pipeline/ is NOT pushed to the boxes, so this
# re-implements the load-bearing part of pipeline.artifacts.verify_manifest
# with nothing but hashlib/json — every {path,sha256} member is re-hashed, the
# content-addressed bundle_id is recomputed (sha256 of the canonical JSON minus
# the bundle_id field, exactly as artifacts.write_manifest stamps it), and the
# training CSV is required to BE the manifest's final_csv. The full schema
# check runs laptop-side via pipeline.artifacts; member hashes are the
# load-bearing part here.
say "verify motion bundle manifest ($BUNDLE/bundle.json)"
[ -f "$BUNDLE/bundle.json" ] || die "bundle manifest missing: $BUNDLE/bundle.json — scp -r data/motions/thriller/v12_bundle from the laptop"
python3 - "$BUNDLE" "$CSV" << 'PYEOF' || die "bundle verification FAILED — refusing to train on unverified motion bytes"
import hashlib, json, sys
from pathlib import Path
bundle, csv = Path(sys.argv[1]), Path(sys.argv[2])
m = json.loads((bundle / "bundle.json").read_text())

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

errs = []
body = {k: v for k, v in m.items() if k != "bundle_id"}
digest = hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if m.get("bundle_id") != digest:
    errs.append("bundle_id does not match manifest content (manifest edited?)")

def walk(node, crumb):
    if isinstance(node, dict):
        if "path" in node and "sha256" in node:
            f = Path(node["path"])
            if not f.is_absolute():
                f = bundle / f
            if not f.exists():
                errs.append(f"{crumb}: missing file {node['path']}")
            elif sha(f) != node["sha256"]:
                errs.append(f"{crumb}: sha mismatch for {node['path']}")
        for k, v in node.items():
            walk(v, f"{crumb}.{k}" if crumb else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, f"{crumb}[{i}]")

walk(m, "")
fin = m.get("motion", {}).get("final_csv") or {}
if not csv.is_file() or sha(csv) != fin.get("sha256"):
    errs.append(f"training CSV {csv} is not the manifest's final_csv "
                "(override or tamper) — rebuild/re-push the bundle")
for e in errs:
    print("MANIFEST ERROR:", e)
if errs:
    sys.exit(1)
print(f"manifest OK: bundle_id {m['bundle_id'][:12]}... members verified")
PYEOF

# Tempo npz: reuse ONLY if they were generated from this exact source CSV
# (recorded by sha stamp). A prior attempt's npz from a different CSV has a
# near-identical frame count, so existence/frame checks alone would happily
# train the wrong motion (audit F2).
declare -A SPEED=( [060]=0.60 [075]=0.75 [090]=0.90 [100]=1.00 )
CSVSHA=$(sha256sum "$CSV" | awk '{print $1}')
STAMP=$NB/motions/thriller_v12_tempo.src.sha256
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$CSVSHA" ]; then
  say "tempo npz stale or unstamped for source $CSVSHA — regenerating"
  rm -f "$NB"/motions/thriller_v12_{060,075,090,100}.{csv,npz} \
        "$NB"/motions/thriller_v12_{060,075,090,100}_check.json
fi
if [ ! -f "$NB/motions/thriller_v12_100.npz" ]; then
  [ -f "$CSV" ] || die "motion CSV missing: $CSV — push it from the laptop"
  head -1 "$CSV" | awk -F, 'NF != 36 { exit 1 }' || die "CSV not 36-column"
  say "retime tempo variants (fresh box: 0.60/0.75/0.90/1.00 from $CSV)"
  NATIVE_FRAMES=$(wc -l < "$CSV")
  for k in 060 075 090 100; do
    OUT=$NB/motions/thriller_v12_${k}.csv
    [ -f "$OUT" ] || "$PY" "$NB/cloud/retime_motion.py" --input "$CSV" --output "$OUT" \
        --speed "${SPEED[$k]}" --check-json "$NB/motions/thriller_v12_${k}_check.json" \
      || die "retime ${SPEED[$k]}x FAILED"
    NPZ=$NB/motions/thriller_v12_${k}.npz
    if [ ! -f "$NPZ" ]; then
      # csv_to_npz hardcodes its output to /tmp/motion.npz and never clears it, and
      # the conversion is `|| true`. Clear any stale prior-tempo npz FIRST so a flaky
      # conversion can't leave the previous variant's file to be copied here (audit B).
      rm -f /tmp/motion.npz
      MUJOCO_GL=egl "$PY" "$CSV2NPZ" --input-file "$OUT" \
          --output-name "thriller_v12_${k}" --input-fps 30 --output-fps 50 || true
      [ -f /tmp/motion.npz ] && cp /tmp/motion.npz "$NPZ"
    fi
    [ -f "$NPZ" ] || die "npz absent after conversion: $NPZ"
    # Tempo sanity: reject a stale/wrong-tempo npz BEFORE accepting it. Expected
    # 50fps frames ~ native_csv_rows / speed * 50/30 (+-2%) — same formula as
    # run_attempt7.sh. Existence alone (the old guard) can't catch a stale copy.
    FRAMES=$("$PY" - "$NPZ" << 'EOF'
import numpy as np, sys
d = np.load(sys.argv[1], allow_pickle=True)
print(int(d["joint_pos"].shape[0]))
EOF
)
    EXPF=$(python3 -c "print(int($NATIVE_FRAMES / ${SPEED[$k]} * 50 / 30))")
    LO=$((EXPF * 98 / 100)); HI=$((EXPF * 102 / 100))
    [ "$FRAMES" -ge "$LO" ] && [ "$FRAMES" -le "$HI" ] \
      || die "npz $k frame count $FRAMES outside expected $LO-$HI (stale/wrong-tempo conversion)"
    echo "  thriller_v12_${k}.npz ready ($FRAMES frames, ~$((FRAMES / 50))s @50fps)"
  done
  echo "$CSVSHA" > "$STAMP"
fi
say "preflight OK (tempo npz present)"
for k in 060 075 090 100; do
  [ -f "$NB/motions/thriller_v12_${k}.npz" ] || die "missing thriller_v12_${k}.npz"
done

# ---- record REALIZED training inputs (motions/bundle_realized.json) ----------
# Box-local manifest binding the tempo npz that will actually be trained on.
# Self-contained for the same reason as the verifier above (no pipeline/ on the
# box): copies the verified manifest's motion section, re-bases its member
# paths under v12_bundle/, fills tempo_npz with fresh hashes, and stamps
# schema/created_at/bundle_id with the same canonical-JSON recipe as
# pipeline.artifacts.write_manifest — so the laptop can later run
# artifacts.verify_manifest on the pulled file as-is.
say "record realized training inputs"
python3 - "$NB/motions" << 'PYEOF' || die "failed to write motions/bundle_realized.json"
import hashlib, json, sys, time
from pathlib import Path
motions = Path(sys.argv[1])
src = json.loads((motions / "v12_bundle" / "bundle.json").read_text())

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

motion = src["motion"]

def rebase(node):
    if isinstance(node, dict):
        if ("path" in node and "sha256" in node
                and not Path(node["path"]).is_absolute()):
            node["path"] = "v12_bundle/" + node["path"]
        for v in node.values():
            rebase(v)
    elif isinstance(node, list):
        for v in node:
            rebase(v)

rebase(motion)
motion["tempo_npz"] = {
    k: {"path": f"thriller_v12_{k}.npz",
        "sha256": sha(motions / f"thriller_v12_{k}.npz")}
    for k in ("060", "075", "090", "100")}
m = {"schema": src["schema"], "motion": motion,
     "source_bundle_id": src.get("bundle_id"),
     "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
body = {k: v for k, v in m.items() if k != "bundle_id"}
m["bundle_id"] = hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
(motions / "bundle_realized.json").write_text(json.dumps(m, indent=2, sort_keys=True))
print("bundle_realized.json written: bundle_id", m["bundle_id"][:12] + "...")
PYEOF

say "selfcheck (v11 task, stage-1 clock 0.60x -> G1_SLOWDOWN=1.6667)"
G1_SLOWDOWN=1.6667 G1_OBS_HISTORY=5 G1_PHASE_OBS=${G1_PHASE_OBS:-0} \
  "$PY" "$NB/cloud/sim2real_task_v14.py" --selfcheck || die "selfcheck FAIL"

say "64-env GPU smoke test (v11, stage-1 conditions, 3 iters)"
G1_SLOWDOWN=1.6667 G1_OBS_HISTORY=5 G1_PHASE_OBS=${G1_PHASE_OBS:-0} \
G1_CMD_DELAY_MAX_LAG=4 G1_OBS_DELAY_MAX_LAG=1 G1_DRIFT_TERM_M=0.8 \
  "$PY" "$NB/cloud/sim2real_task_v14.py" Mjlab-Tracking-Flat-Unitree-G1-S2R-V14 \
    --env.scene.num-envs 64 --env.commands.motion.motion-file "$NB/motions/thriller_v12_060.npz" \
    --agent.max-iterations 3 --agent.run-name smoketest-v14 || die "smoke test FAIL"
# NOTE: the smoke also exercises AUDIT F1's scoped_effort_limits startup readback —
# a wrong realized forcerange on ANY control kills the smoke here, before real spend.

say "obs-layout gate (audit F3/E: deploy HistoryStacker vs live actor obs, byte-compare)"
# Frozen CLI per tasks/audit_fixes_20260721/CONVENTIONS.md §3.5. Exit 0 PASS /
# 1 FAIL / 3 API-unavailable. A transposed flatten is SILENT (no crash) and
# would drive a fall on the robot — hard-fail on 1; treat 3 as a loud warning
# (the layout is separately pinned CPU-side to HistoryStacker by unit test).
G1_SLOWDOWN=1.6667 G1_OBS_HISTORY=5 G1_PHASE_OBS=${G1_PHASE_OBS:-0} \
  "$PY" "$NB/cloud/verify_obs_layout.py" --task Mjlab-Tracking-Flat-Unitree-G1-S2R-V14 \
    --task-module sim2real_task_v14 --motion-file "$NB/motions/thriller_v12_100.npz" \
    --num-envs 2
_OBS_RC=$?
if [ "$_OBS_RC" -eq 1 ]; then die "obs-layout gate FAILED — deploy contract mismatch"; fi
[ "$_OBS_RC" -eq 3 ] && echo "  (obs-layout gate: mjlab API unavailable — layout remains CPU-pinned only)"

say "LAUNCH v12 = v11 recipe on the FIDELITY motion (full intro, unwarped, guard-cleaned)"
export G1_WAIST_SRC_WINDOWS="15.5-20.5,27.5-38.5"   # +2.5s: v12 restores the intro
# RUN_NAME respects a pre-set value so a multi-box sweep gives each box a distinct
# name (W&B + exports); falls back to a dated default for a single run.
export RUN_NAME=${RUN_NAME:-train-thriller_v14style-$(date +%m%d)}
# STYLE fix (2026-07-22): commit harder to the sharper reference
export G1_LEG_POS_STD=${G1_LEG_POS_STD:-0.26}
export G1_LEG_ORI_STD=${G1_LEG_ORI_STD:-0.34}
M060=$NB/motions/thriller_v12_060.npz \
M075=$NB/motions/thriller_v12_075.npz \
M090=$NB/motions/thriller_v12_090.npz \
M100=$NB/motions/thriller_v12_100.npz \
  bash "$NB/cloud/train_v14_curriculum.sh"
say "run_attempt11 DONE — pull exports, judge vs calibrated bars, sign, DELETE THE BOX"
