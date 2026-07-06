#!/usr/bin/env python
"""Convert a KungfuAthleteBot G1 29-DoF org npz into our 36-col LAFAN1-style CSV.

Input  (data/acro_refs/kungfuathletebot_*/<id>_org.npz):
    { fps: 30, qpos: (T,36) }  qpos = [x y z | qw qx qy qz | 29 joints]
    Joint order verified identical to policy_meta.json joint_order_29dof
    (see data/acro_refs/kungfuathletebot_backflip/PROVENANCE.txt).

Output (36-col CSV, 30 fps): [x y z | qx qy qz qw | 29 joints] — the exact
convention pipeline/motion_io.load_motion_csv and mjlab csv_to_npz expect.

Processing (kept minimal & auditable):
  1. optional --trim-start/--trim-end (org-clip frames, end exclusive): the
     KungfuAthleteBot clips flow into the NEXT kungfu move after the flip
     recovery — trim at the end of the post-landing upright window so the
     reference ends standing (the landing-success eval needs that). All kept
     frames are real mocap; nothing is synthesized.
  2. quaternion reorder wxyz -> xyzw + per-frame normalization;
  3. XY re-origin (frame 0 at 0,0) — footprint placement is a deploy concern;
  4. clamp joints to model position limits (0.005 rad inner margin), report
     the worst clamp — video-mocap retargets carry a few hundredths of a rad
     of violation; mjlab clips to soft limits at RSI anyway, this just makes
     the reference consistent with what the sim can represent;
  5. ground the motion with the same pipeline.grounding.ground_motion used by
     the vet gate (constant z-shift so the lowest robot geom touches z=0);
  6. pad: hold frame 0 for --lead-in seconds (default 1.5) and the last frame
     for --hold-out seconds (default 2.0). Gives the RSI sampler quiet stance
     bins around the skill and gives landing-success eval a settle window.

NO smoothing, NO velocity clamping (docs/retarget_fidelity.md: blanket clamps
blunt dynamic accents — for a flip they would delete the skill itself).

Usage:
  ~/miniconda3/envs/g1dance/bin/python tools/convert_acro_reference.py \
      data/acro_refs/kungfuathletebot_backflip/280_org.npz \
      data/acro_refs/converted/acro_backflip.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CSV_FPS = 30.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("csv_out")
    ap.add_argument("--lead-in", type=float, default=1.5, help="s of frame-0 hold")
    ap.add_argument("--hold-out", type=float, default=2.0, help="s of last-frame hold")
    ap.add_argument("--trim-start", type=int, default=0, help="org frame to start at")
    ap.add_argument("--trim-end", type=int, default=None,
                    help="org frame to end BEFORE (exclusive)")
    args = ap.parse_args()

    import mujoco

    from pipeline.grounding import ground_motion

    data = np.load(args.npz)
    fps = float(np.array(data["fps"]).reshape(-1)[0])
    if abs(fps - CSV_FPS) > 1e-6:
        raise SystemExit(f"expected 30 fps org npz, got fps={fps} — use the *_org.npz")
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise SystemExit(f"expected qpos (T,36), got {qpos.shape}")

    # 1. trim to the skill (real mocap frames only)
    n_org = qpos.shape[0]
    qpos = qpos[args.trim_start:args.trim_end]
    print(f"trim: org {n_org} frames -> kept [{args.trim_start}:"
          f"{args.trim_end if args.trim_end is not None else n_org}) = {len(qpos)}")

    # 2. wxyz -> xyzw + normalize
    m = np.empty_like(qpos)
    m[:, 0:3] = qpos[:, 0:3]
    m[:, 3:6] = qpos[:, 4:7]   # x y z
    m[:, 6] = qpos[:, 3]       # w
    m[:, 7:] = qpos[:, 7:]
    qn = np.linalg.norm(m[:, 3:7], axis=1, keepdims=True)
    if np.any(np.abs(qn - 1.0) > 0.01):
        print(f"note: renormalizing quats (worst |q|-1 = {np.abs(qn - 1).max():.4f})")
    m[:, 3:7] /= qn

    # 3. XY re-origin
    m[:, 0] -= m[0, 0]
    m[:, 1] -= m[0, 1]

    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "third_party/mujoco_menagerie/unitree_g1/scene.xml")
    )

    # 4. clamp joints to model limits (inner margin 0.005 rad)
    lo = model.jnt_range[1:, 0] + 0.005
    hi = model.jnt_range[1:, 1] - 0.005
    before = m[:, 7:].copy()
    m[:, 7:] = np.clip(m[:, 7:], lo, hi)
    print(f"joint-limit clamp: worst change {np.abs(m[:, 7:] - before).max():.4f} rad")

    # 5. ground (same machinery as the vet gate)
    m, shift = ground_motion(m, model)
    print(f"ground shift: {shift:+.4f} m")

    # 6. pad with stance holds (zero velocity at the seams by construction)
    n_in = int(round(args.lead_in * CSV_FPS))
    n_out = int(round(args.hold_out * CSV_FPS))
    m = np.concatenate([np.repeat(m[:1], n_in, axis=0), m,
                        np.repeat(m[-1:], n_out, axis=0)], axis=0)

    out = Path(args.csv_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, m, delimiter=",", fmt="%.8f")
    print(f"WROTE {out}  {m.shape[0]} frames = {m.shape[0] / CSV_FPS:.2f} s "
          f"(lead-in {n_in}f, skill {qpos.shape[0]}f, hold-out {n_out}f)")


if __name__ == "__main__":
    main()
