# Preview assets for this policy

This directory holds a policy pulled from a cloud training run. To render the
Simulation-tab preview (tools/sim_studio), pipeline/sim_preview needs, next to
`policy.onnx`:

  - `policy_meta.json`  — this policy's export-time contract. Its `onnx_inputs`
                          must exactly match this directory's ONNX graph.
  - `*_deploy.npz`      — the reference motion the preview plays as the "intended dance"
                          (left pane) AND feeds as the policy's command input (right pane).
                          Preferred source is THIS policy's own lineage: the staged npz
                          pulled from the run, else a conversion of the pulled deploy CSV.
                          Only if neither is available is the SHARED `thriller_deploy`
                          motion copied as a last resort (wrong-lineage — see finding C).

The motion may be added automatically. Metadata is never synthesized or borrowed;
without a matching sidecar this dance is registered without a preview.
