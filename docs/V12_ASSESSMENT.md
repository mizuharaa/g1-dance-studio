# V12 assessment — did the training pass?

**Verdict: 8/9 honest-gate PASS — the strongest, most trustworthy policy of the
project — but NOT a full gate pass, and the user's eye test correctly caught real
style problems the gate never measured.** Winner iter 8000 (native-tempo stage),
attempt 9, box 40711, 2026-07-22. Artifacts:
`exports/train-thriller_v12fid-0722/` (in the repo, policy.onnx included).

## What passed (harness v2 — first trustworthy numbers; NOT comparable to v5–v11)

| Check (bar) | v12 result |
|---|---|
| Clean survival ≥99% | **100%** (128/128) PASS |
| Drift ep-p95 ≤1.5 m | **0.91 m** PASS — drift SOLVED on the honest metric |
| Ankle p95 ≤22 Nm | **14.3** PASS — first time inside the real robot's measured 15–19 band |
| Ankle mean ≤6 Nm | **5.1** PASS (first ever) |
| Thermal RMS ≤12 Nm | **6.9** PASS |
| rr_mpkpe ≤0.1 m | **0.057** PASS (best ever) |
| Heldout nominal | 100/100/100 (3 seeds); push 89–92% |

One-factor robustness rows all strong: noise 100%, cmd_delay 40 ms 100%,
push-only 89.8%, DR-only 93.8%.

## What failed

**The full composite `dr_delay40ms_push`: 57% vs bar 95%.** DR + 40 ms delay +
push + noise stacked. Each factor alone is fine; the stack is not. (The old
~70%-IRL anchor scored 34.4% on the *contaminated* version of this row, so the
bar may be stricter than deployed reality — re-derive the calibrated bar on
harness v2 before spending GPU chasing this number.)

## The finding that matters for hardware: a sharp latency wall

| cmd delay | survival |
|---|---|
| 0–40 ms | 100% |
| 60 ms | 72.7% |
| 80 ms | **0%** |

Same for obs delay (80 ms → 0%). **The real robot's measured sensorimotor
latency is 40–80 ms** (`data/telemetry/latency_diag_20260709/`). v12 is robust
only to the LOWER HALF of the real latency band. If the hardware sits near the
top of that band on a given day, the policy is operating at or past its cliff —
a plausible mechanical explanation for the "always about to lose balance" look,
independent of the style-stiffness explanation. Worth measuring live latency
FIRST on the next robot day (the diag scripts exist) before blaming the recipe.

## What the gate never measured (user eye test, confirmed by measurement)

`experiments/style_gap_20260722/`: leg command HF content 3.97% vs 1.2–2.3%
real-robot baseline (visible twitching), leg reach 60–79% of reference, base
sway 3× too stiff (never commits weight), clean drift 0.53 m (in-bounds but
visible). These are now recorded per-run by the gate (audit fix J) but have no
bars yet.

## Bottom line

- **As a training run: PASS with one asterisk** — every nominal and single-factor
  bar cleared, composite-stress row failed.
- **As a show policy: not yet.** The style gap (stiff sway, short legs, twitch)
  is real, measured, and unfixed — v13 targeted it but its artifacts died with
  the box; **v14** (this repo, `cloud/sim2real_task_v14.py`) is the successor
  carrying the HoloSoma-informed fixes (keypoint termination forcing leg reach,
  leaner shaping) plus the v13 anti-chatter term.
- **Deploy candidate ranking today**: v3e remains the only hardware-proven
  policy; v12 is the best sim policy and next in line for a gantry session
  (770-dim contract — needs the HistoryStacker deploy path, box-verified but
  never hardware-run).
