"""Best-checkpoint selection for the attempt-4 verify chain.

WHY: v6 auto-exported the LAST checkpoint (model_9997), whose mean episode length
(388) was a low point in an oscillating late-stage reward — a nearby checkpoint
survives/tracks better. We already learned this once (the a2 story: iter-1500 beat
iter-3999). This screens the last K checkpoints of the final stage with a CHEAP
gap_check (2 gate-critical conditions, few envs) and prints the winner, ranked by
how well it satisfies the actual gate. Pure evaluation — zero training risk.

Usage (run on the box, inside the verify chain, MUJOCO_GL=egl):
  python cloud/pick_checkpoint.py --python <PY> --gap-check cloud/sim_gap_check.py \
      --rundir <s3 run dir> --motion-file <npz> --last 6 --num-envs 64 \
      --workdir <EXP>/screen
Prints one line "WINNER <abs ckpt path>" on success (also writes screen_summary.json).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# The two gate-critical conditions we screen on (cheap: skips the 60/80 ms stress
# lines that don't gate). Names must match cloud/sim_gap_check.py CONDITIONS.
SCREEN_ONLY = "nominal,delay40ms_push"

# gate thresholds (mirror cloud/sim_gap_check.py + PROJECT_STATE gate)
GATE = {
  "nominal_survival": 0.99,
  "nominal_drift_max": 1.0,
  "nominal_ankle_p95": 15.0,
  "push_survival": 0.95,     # delay40ms_push
  "push_ankle_p95": 20.0,
}


def _last_checkpoints(rundir: Path, k: int) -> list[Path]:
  ck = sorted(rundir.glob("model_*.pt"),
              key=lambda p: int(p.stem.split("_")[1]))
  return ck[-k:]


def _score(gap: dict) -> dict:
  """Turn a screened gap.json into gate-pass counts + tiebreak metrics."""
  n = gap["conditions"]["nominal"]
  p = gap["conditions"].get("delay40ms_push", {})
  nsurv = n.get("success_rate", 0.0)
  ndrift = n.get("drift", {}).get("max_m", 9e9)
  nank = n.get("ankle_pitch", {}).get("p95_abs", 9e9)
  psurv = p.get("success_rate", 0.0)
  pank = p.get("ankle_pitch", {}).get("p95_abs", 9e9)
  passes = sum([
    nsurv >= GATE["nominal_survival"],
    ndrift <= GATE["nominal_drift_max"],
    nank <= GATE["nominal_ankle_p95"],
    psurv >= GATE["push_survival"],
    pank <= GATE["push_ankle_p95"],
  ])
  return {
    "gate_passes": passes, "nominal_survival": nsurv, "nominal_drift_max": ndrift,
    "nominal_ankle_p95": nank, "push_survival": psurv, "push_ankle_p95": pank,
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--python", required=True)
  ap.add_argument("--gap-check", required=True)
  ap.add_argument("--rundir", required=True)
  ap.add_argument("--motion-file", required=True)
  ap.add_argument("--last", type=int, default=6)
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--workdir", required=True)
  a = ap.parse_args()

  rundir = Path(a.rundir)
  work = Path(a.workdir); work.mkdir(parents=True, exist_ok=True)
  cks = _last_checkpoints(rundir, a.last)
  if not cks:
    print(f"!! no checkpoints in {rundir}", file=sys.stderr)
    return 2

  rows = []
  for ck in cks:
    it = int(ck.stem.split("_")[1])
    out = work / f"screen_{it}.json"
    print(f"[screen] checkpoint {it} -> {out.name}", flush=True)
    r = subprocess.run(
      [a.python, a.gap_check, "--checkpoint", str(ck), "--motion-file", a.motion_file,
       "--num-envs", str(a.num_envs), "--only", SCREEN_ONLY, "--output-file", str(out)],
      capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
      print(f"  !! screen failed for {it} (rc={r.returncode}); skipping\n{r.stderr[-500:]}",
            file=sys.stderr)
      continue
    try:
      sc = _score(json.load(open(out)))
    except Exception as e:  # noqa: BLE001
      print(f"  !! could not score {it}: {e}", file=sys.stderr)
      continue
    sc["iter"] = it; sc["checkpoint"] = str(ck)
    rows.append(sc)
    print(f"  iter {it}: passes={sc['gate_passes']}/5 surv={sc['nominal_survival']:.3f} "
          f"drift={sc['nominal_drift_max']:.2f} ankle={sc['nominal_ankle_p95']:.1f} "
          f"push_surv={sc['push_survival']:.3f}", flush=True)

  if not rows:
    print("!! every screen failed — falling back to the newest checkpoint", file=sys.stderr)
    win = cks[-1]
    json.dump({"winner": str(win), "reason": "all-screens-failed", "rows": []},
              open(work / "screen_summary.json", "w"), indent=2)
    print(f"WINNER {win}")
    return 0

  # rank: most gate checks passed, then highest nominal survival, then lowest drift,
  # then lowest ankle p95. (Latest iter breaks exact ties.)
  rows.sort(key=lambda s: (s["gate_passes"], s["nominal_survival"],
                           -s["nominal_drift_max"], -s["nominal_ankle_p95"], s["iter"]),
            reverse=True)
  win = rows[0]
  json.dump({"winner": win["checkpoint"], "winner_iter": win["iter"],
             "gate": GATE, "rows": rows}, open(work / "screen_summary.json", "w"), indent=2)
  print(f"[pick] winner = iter {win['iter']} ({win['gate_passes']}/5 gate checks, "
        f"surv {win['nominal_survival']:.3f}, drift {win['nominal_drift_max']:.2f} m)")
  print(f"WINNER {win['checkpoint']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
