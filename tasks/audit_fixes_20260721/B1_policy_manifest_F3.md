# B1 — Fix F3 (part 1): policy-specific metadata + bundle manifest (HIGH)

**Finding:** REPORT §F3. All three shipped ONNX policies are `[1,770]` while
their `policy_meta.json` (copied from the shared 160-dim gantry-era file by
`pipeline/publish_policy.py:127-150`) says `obs:[1,160]` and names
estimator-only terms. Deploy either refuses (good) or, with terms omitted,
silently loses the `requires_ground_contact` guard. Export writes ONNX only.

**Owned files:** `cloud/export_ckpt_onnx.py`, `cloud/export_policy.py`,
`pipeline/publish_policy.py`, `pipeline/deploy_runtime.py` (validation ONLY —
see CONVENTIONS §2 invariants), `tests/test_bundle_meta.py` (new).

## Spec
1. **Export generates meta v2 (CONVENTIONS §3.2)** from the LIVE task cfg at
   export time: ordered actor terms + widths, `obs_per_frame`, `history_length`,
   `flatten_layout: "term-major-oldest-first"`, real `onnx_inputs` read from the
   exported ONNX, `requires_ground_contact: true` for ground-contract policies,
   plus all existing fields (joint order, kp/kd, effort, action scale) sourced
   from the cfg — NOT copied from another policy dir. Write it next to the ONNX
   in exports/, and fill the `policy` section of `bundle.json` via
   `pipeline/artifacts` (create the manifest if the motion side is absent —
   sections are independent).
2. **publish_policy:** if a pulled dir has meta whose `onnx_inputs` disagree
   with the actual ONNX input shape → REFUSE to publish that meta (log, fall
   back to registering the dance WITHOUT preview rather than with a lying
   contract — never copy the shared meta over a 770-dim policy). Keep the
   never-raise publish contract.
3. **deploy_runtime bundle validation (BEFORE `make_dds` / human confirm; damp
   spine untouched):** load policy+meta+npz; REFUSE (SystemExit, reason) if:
   ONNX obs width is dynamic or not an exact multiple of meta `obs_per_frame`;
   meta `onnx_inputs` ≠ actual ONNX; meta names estimator terms for a ground
   run (existing check stays); `requires_ground_contact` absent for a
   770-contract policy. Delete the silent "degrade to history 1" on
   non-multiple widths (`n_hist_for` may stay for the sandbox, but deploy must
   REJECT instead of degrading — pass strict=True from deploy).
4. **Tests:** meta-v2 generation from a fake cfg; publish refusal on mismatched
   meta; deploy validation table (good bundle passes; each defect class refuses
   with a message naming the defect). Use tiny hand-built ONNX (onnx.helper) to
   avoid big fixtures.

## Acceptance
No code path left that copies `data/policies/thriller/policy_meta.json` onto a
7xx-dim policy; strict refusal precedes any DDS/motion-service call and any
human prompt; all existing deploy tests green; `tests/test_bundle_meta.py` green.

## API delta
`policy_meta.json` gains the §3.2 fields (additive). `bundle.json` policy
section per §3.1.
