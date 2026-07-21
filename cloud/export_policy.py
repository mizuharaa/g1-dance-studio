#!/usr/bin/env python3
"""Stage a trained mjlab tracking policy's ONNX as <exports>/policy.onnx.

mjlab's `MotionTrackingOnPolicyRunner` already exports the trained policy to ONNX at
training time — the `_OnnxMotionModel`: inputs `obs` + `time_step`; outputs `actions`
plus the motion tensors (joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w,
body_ang_vel_w) with the reference motion baked in and metadata attached
(`attach_metadata_to_onnx`). That file lives next to the checkpoints in the run dir.
The app's export stage calls this script to copy that export into the exports dir as
`policy.onnx` (the artifact it then pulls to the laptop).

This exists because the app's export stage referenced `cloud/export_policy.py` but the
file was never committed (the export stage had never run end-to-end on a fresh box).

Besides the ONNX, this stages the run's OWN reference motion npz (the <motion.npz> arg,
the exact file training was conditioned on) into the exports dir as `*_deploy.npz`, so the
manual pull (scripts/retrain_pull.sh) carries a policy-correct motion for its preview. Before
this, the arg was accepted but ignored and the pull fell back to a shared, wrong-lineage npz
(audit 2026-07-21 finding C).

Usage: export_policy.py <checkpoint.pt> <motion.npz> <exports_dir>
"""
import glob
import os
import shutil
import sys


def _stage_motion(npz: str, exports: str) -> None:
    """Copy the run's reference motion npz into `exports` as a `*_deploy.npz` so the pull
    carries THIS policy's own motion (both the preview reference pane and the sandbox
    command input read it). Never fails the export — a missing/odd npz only means the
    downstream preview falls back to a converted-from-csv or shared motion."""
    if not npz or not os.path.isfile(npz):
        print(f"  (motion npz not staged — not a file: {npz!r})")
        return
    base = os.path.basename(npz)
    if not base.endswith("_deploy.npz"):   # guarantee the *_deploy.npz preview glob matches
        base = f"{os.path.splitext(base)[0]}_deploy.npz"
    dst = os.path.join(exports, base)
    try:
        shutil.copyfile(npz, dst)
    except OSError as e:
        print(f"  (motion npz not staged — copy failed: {e})")
        return
    print(f"EXPORT motion: {npz} -> {dst} ({os.path.getsize(dst)} bytes)")


def _verify(dst: str) -> None:
    """Best-effort check that the staged ONNX has the BeyondMimic I/O contract.
    Never fails the export for a missing verifier — only for a genuinely wrong file."""
    ins = outs = None
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
        ins = {i.name for i in s.get_inputs()}
        outs = [o.name for o in s.get_outputs()]
    except Exception:
        try:
            import onnx
            m = onnx.load(dst)
            ins = {i.name for i in m.graph.input}
            outs = [o.name for o in m.graph.output]
        except Exception as e:
            print(f"  (contract verify skipped — no onnx/onnxruntime: {e})")
            return
    assert {"obs", "time_step"} <= ins, f"exported ONNX missing inputs obs/time_step: {ins}"
    assert len(outs) >= 5, f"exported ONNX has too few outputs: {outs}"
    print(f"  verified: inputs={sorted(ins)} outputs={outs}")


def main(ckpt: str, npz: str, exports: str) -> None:
    run_dir = os.path.dirname(os.path.abspath(ckpt))
    onnxs = sorted(glob.glob(os.path.join(run_dir, "*.onnx")), key=os.path.getmtime)
    if not onnxs:
        sys.exit(
            f"ERROR: no mjlab-exported .onnx in {run_dir}. mjlab's tracking runner "
            f"exports the policy at train time; none was found — training may have "
            f"exited before the final save/export."
        )
    src = onnxs[-1]  # newest = the final trained-policy export (matches the newest checkpoint)
    os.makedirs(exports, exist_ok=True)
    dst = os.path.join(exports, "policy.onnx")
    shutil.copyfile(src, dst)
    if os.path.getsize(dst) <= 0:
        sys.exit(f"ERROR: staged ONNX is empty: {dst}")
    print(f"EXPORT: {os.path.basename(src)} -> {dst} ({os.path.getsize(dst)} bytes)")
    _verify(dst)
    _stage_motion(npz, exports)   # stage the run's OWN reference motion alongside policy.onnx


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("usage: export_policy.py <checkpoint.pt> <motion.npz> <exports_dir>")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
