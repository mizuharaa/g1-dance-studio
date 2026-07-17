# Preview assets for this policy

This directory holds a policy pulled from a cloud training run. To render the
Simulation-tab preview (tools/sim_studio), pipeline/sim_preview needs, next to
`policy.onnx`:

  - `policy_meta.json`  — joint order / gains, IDENTICAL across Thriller policies
                          (policy-independent), copied from data/policies/thriller/.
  - `*_deploy.npz`      — the reference Thriller motion the preview plays as the
                          "intended dance" (left pane). This is the SHARED
                          `thriller_deploy` motion copied from data/policies/thriller/,
                          NOT this policy's own trajectory — it only drives the
                          reference/left side; the right side is this policy rolled out.

Both are added automatically by pipeline/publish_policy.py on pull if missing.
