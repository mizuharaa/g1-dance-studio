# B5 — Fix F5: stop calling the preview scene "the training model" (HIGH, small)

**Finding:** REPORT §F5. `tools/assets/g1_faithful/` is built from the OFFICIAL
Unitree XML (35.11 kg, 9.60 kg torso, feet-only collisions, friction 1.0), not
the pinned mjlab model (33.34 kg, 7.82 kg torso, full-body collisions, 0.6 base
foot friction + DR). The sandbox/studio banners call it "the mjlab training
model", turning an alternate-model stress test into false corroboration.

**Owned files:** `tools/sim_sandbox.py`, `tools/sim_studio.py` (labels/banners/
docstrings), `pipeline/sim_preview.py` (stored labels, if any),
`experiments/g1_model_reconciliation.md` (append a correction note — do NOT
rewrite history, add a dated section), tests that assert banner text.

## Spec (labels + provenance only — NO physics/scene changes this wave)
1. Rename the concept everywhere user-visible: the current scene is the
   **"hardware-uncertainty scene (official Unitree model)"**. Banner text:
   `PREVIEW on the hardware-uncertainty scene (official Unitree XML; NOT the
   pinned mjlab training model — treat disagreement as signal, not error)`.
   Update the `_banner`/caveat helpers and any "faithful (= mjlab-aligned)"
   phrasing in code comments/docstrings you own.
2. Record scene identity in outputs: every sandbox/studio report json gains
   `scene: {"name": "hardware-uncertainty-v1", "xml_sha256": ...}` (hash the
   XML via `pipeline.artifacts.sha256_file`).
3. Append to `experiments/g1_model_reconciliation.md`: a dated correction
   citing the audit's mass comparison (33.341 vs 35.112 kg; torso 7.818 vs
   9.598) and the two-scene plan (exact-training dump = future GPU task; this
   wave is truth-in-labeling only).
4. Tests: report-json contains the scene block; banner contains
   "hardware-uncertainty" and does NOT contain "training model".

## Acceptance
`grep -rn "training model" tools/sim_sandbox.py tools/sim_studio.py` returns
only negated/explanatory phrasing; scene hash present in fresh report jsons;
existing preview tests green.
