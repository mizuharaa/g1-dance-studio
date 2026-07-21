# A3 — Fix F2: rebuild v12 from an immutable source + hash-bound manifest (HIGH)

**Depends on A2 (fixed grounding).**

**Finding:** REPORT §F2. Current `thriller_v12_full.csv` (sha `1bf7…`) matches NO
scorecard (`2ee0…` recorded; source was a deleted /tmp path); post-scorecard fix-A
rewrite changed 692 rows; fresh feasibility: binding-max 2.865, 74.8% floaty feet
vs scorecard 1.179/–. The launcher checks only existence/frame-count.

**Owned files:** `tools/build_motion_bundle.py` (NEW), `data/motions/thriller/**`
(regenerated artifacts, force-add), `cloud/run_attempt9.sh` (hash verification),
`tests/test_motion_bundle.py` (new).

## Spec
1. **`tools/build_motion_bundle.py`** — ONE command, deterministic, from an
   immutable committed source:
   `python tools/build_motion_bundle.py --source data/motions/thriller/thriller_deploy.csv --out-dir data/motions/thriller/v12_bundle`
   Steps in order: guard-clean (`tools/motion_quality.clean_motion`) →
   flight-aware ground (A2) → feasibility scorecard (`pipeline/motion_dynamics`)
   → per-joint & per-5s-window fidelity retention vs the SOURCE (amp + peakvel,
   not just group means; warn <0.90, fail <0.75 on any moving joint) → write
   `final.csv`, `scorecard.json`, and `bundle.json` via `pipeline/artifacts`
   (CONVENTIONS §3.1 `motion` section: source_csv/final_csv/scorecard entries +
   grounding params). NO repair-ladder mutation steps in this build (unwarped
   lineage decision stands); if feasibility any-joint-over% > 3%, FAIL with the
   report — a human decides on repair, the tool never silently softens.
2. **Regenerate the real bundle** into `data/motions/thriller/v12_bundle/` from
   `thriller_deploy.csv` (committed, immutable). Commit `final.csv`,
   `scorecard.json`, `bundle.json` (git add -f per repo convention). Leave the
   old `thriller_v12_full.csv` in place but append a DEPRECATED note to
   `logs/jobs.md` (it must never be trained on again).
3. **`cloud/run_attempt9.sh`:** point `G1_MOTION_CSV` default at
   `v12_bundle/final.csv`; before ANY retime/convert, run
   `python -c "from pipeline.artifacts import verify_manifest; ..."` against
   `v12_bundle/bundle.json` and DIE on any error; after generating each tempo
   npz, append its `{"path","sha256"}` into a box-local copy
   `motions/bundle_realized.json` (write via artifacts.write_manifest with the
   motion section copied + tempo_npz filled) so training inputs are recorded.
   Keep existing frame-count checks.
4. **Tests:** build a bundle from a small synthetic CSV → verify_manifest([])
   passes; tamper with final.csv → launcher-style verification fails; fidelity
   gate trips on an artificially blunted clean stub (monkeypatch clean_motion).

## Acceptance
One command reproduces the bundle bit-for-bit from the committed source (run it
twice, byte-compare); `verify_manifest` returns `[]`; scorecard matches the
final bytes by construction; run_attempt9 refuses a tampered bundle; new
feasibility numbers recorded honestly in the scorecard + `logs/jobs.md` note.

## Validation
```bash
python -m pytest tests/test_motion_bundle.py -q && bash -n cloud/run_attempt9.sh
python tools/build_motion_bundle.py --source data/motions/thriller/thriller_deploy.csv --out-dir /tmp/claude-1000/-home-alois/9447aaaf-123a-4f13-8d70-6e25dbe4e703/scratchpad/v12b && python tools/build_motion_bundle.py --source data/motions/thriller/thriller_deploy.csv --out-dir /tmp/claude-1000/-home-alois/9447aaaf-123a-4f13-8d70-6e25dbe4e703/scratchpad/v12b2 && diff <(sha256sum /tmp/claude-1000/-home-alois/9447aaaf-123a-4f13-8d70-6e25dbe4e703/scratchpad/v12b/final.csv | cut -d' ' -f1) <(sha256sum /tmp/claude-1000/-home-alois/9447aaaf-123a-4f13-8d70-6e25dbe4e703/scratchpad/v12b2/final.csv | cut -d' ' -f1)
```
