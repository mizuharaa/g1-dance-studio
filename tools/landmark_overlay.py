#!/usr/bin/env python
"""Render the pose-estimation landmarks overlaid on the ORIGINAL dance video.

The earliest, cheapest place to catch garbage-in: before retargeting/training, the
operator can SEE whether GVHMR actually tracked the dancer. We take the GVHMR pred
file (hmr4d_results.pt) — which carries in-camera SMPL-X params + per-frame camera
intrinsics K_fullimg but NO 2D keypoints — run the SMPL-X body model to get 3D body
joints in camera space, project them through K onto each frame, and draw the skeleton
+ joint dots on the source video.

There is NO system ffmpeg on this laptop; encoding goes through imageio-ffmpeg.

Usage:
  python -m tools.landmark_overlay \
      --pred data/jobs/<id>/extract/hmr4d_results.pt \
      --video data/jobs/<id>/extract/input_30fps.mp4 \
      --out data/previews/landmarks/<id>.mp4
  # or point at a job dir and let it find both:
  python -m tools.landmark_overlay --job data/jobs/<id> --out <id>.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GMR_DIR = ROOT / "third_party" / "GMR"
SMPLX_DIR = GMR_DIR / "assets" / "body_models"

# SMPL/SMPL-X body kinematic tree — the first 22 joints are the body (shared layout).
# (child -> parent); used to draw bones.
_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
_BONES = [(i, p) for i, p in enumerate(_PARENTS) if p >= 0]
_NB = len(_PARENTS)  # 22


def _joints_incam(pred_file: Path, max_frames: int | None) -> np.ndarray:
    """3D SMPL-X body joints (N, 22, 3) in CAMERA coordinates + per-frame K (N, 3, 3)."""
    import torch
    import smplx

    pred = torch.load(str(pred_file), map_location="cpu", weights_only=False)
    inc = pred["smpl_params_incam"]
    K = pred["K_fullimg"].float().numpy()
    n = inc["body_pose"].shape[0]
    if max_frames:
        n = min(n, max_frames)
    body_model = smplx.create(str(SMPLX_DIR), "smplx", gender="neutral", use_pca=False)
    # Match GMR's working call: model shapedirs expect 16 shape coeffs, so pad the
    # GVHMR 10-vector betas to 16 and pass a single (1,16) betas (broadcast over frames);
    # no expression (would change the concat width and break the shape blend).
    betas = torch.tensor(np.pad(inc["betas"][0].numpy(), (0, 6))).float().view(1, -1)
    joints = np.empty((n, _NB, 3), np.float32)
    B = 256  # batch to bound memory (SMPL-X also returns vertices)
    for s in range(0, n, B):
        e = min(s + B, n)
        m = e - s
        with torch.no_grad():
            out = body_model(
                betas=betas,
                global_orient=inc["global_orient"][s:e].float(),
                body_pose=inc["body_pose"][s:e].float(),
                transl=inc["transl"][s:e].float(),
                left_hand_pose=torch.zeros(m, 45), right_hand_pose=torch.zeros(m, 45),
                jaw_pose=torch.zeros(m, 3), leye_pose=torch.zeros(m, 3),
                reye_pose=torch.zeros(m, 3),
            )
        joints[s:e] = out.joints[:, :_NB, :].detach().numpy()
    return joints, K[:n]


def _project(joints: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Perspective-project camera-space joints (N,22,3) with per-frame K -> pixel (N,22,2)."""
    uvw = np.einsum("nij,nkj->nki", K, joints)          # (N,22,3)
    z = np.clip(uvw[..., 2:3], 1e-4, None)
    return uvw[..., :2] / z


def render(pred_file: Path, video: Path, out_path: Path, max_frames: int | None = None) -> dict:
    import cv2
    import imageio

    joints, K = _joints_incam(pred_file, max_frames)
    uv = _project(joints, K)                            # (N,22,2)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = min(len(uv), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(uv))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                quality=7, macro_block_size=None,
                                ffmpeg_log_level="error")
    BONE = (255, 210, 40)     # cyan-ish bones (RGB)
    DOT = (60, 255, 120)      # green joint dots
    drawn = 0
    for k in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        pts = uv[k]
        for a, b in _BONES:
            pa, pb = pts[a], pts[b]
            if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                         BONE, 2, cv2.LINE_AA)
        for p in pts:
            if 0 <= p[0] < w and 0 <= p[1] < h:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, DOT, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (w, 20), (0, 0, 0), -1)
        cv2.putText(frame, "POSE-ESTIMATION LANDMARKS (GVHMR) on source video",
                    (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        writer.append_data(frame)
        drawn += 1
    cap.release()
    writer.close()
    info = {"frames": drawn, "fps": float(fps), "out": str(out_path)}
    print(info)
    return info


def _resolve(args) -> tuple[Path, Path]:
    if args.job:
        job = Path(args.job)
        pred = job / "extract" / "hmr4d_results.pt"
        video = job / "extract" / "input_30fps.mp4"
        if not video.exists():
            video = job / "input.mp4"
        return pred, video
    return Path(args.pred), Path(args.video)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job", type=Path, default=None, help="job dir (finds pred + video)")
    ap.add_argument("--pred", type=Path, default=None, help="hmr4d_results.pt")
    ap.add_argument("--video", type=Path, default=None, help="source video (30 fps clip)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=None, help="cap frames (smoke test)")
    args = ap.parse_args()
    pred, video = _resolve(args)
    if not pred.exists():
        print(f"pred file not found: {pred}", file=sys.stderr)
        return 2
    if not video.exists():
        print(f"source video not found: {video}", file=sys.stderr)
        return 2
    render(pred, video, args.out, max_frames=args.max_frames)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
