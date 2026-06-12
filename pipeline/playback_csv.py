"""Kinematic playback of a LAFAN1-convention G1 motion CSV in MuJoCo.

CSV format (30 fps, 36 cols): root xyz (0-2), root quat xyzw (3-6),
29 joint angles (7-35) ordered legs(12), waist yrp(3), Larm(7), Rarm(7) --
identical to the menagerie g1.xml joint order. qpos wants quat as wxyz.

Usage:
  python playback_csv.py motion.csv --view              # interactive window
  python playback_csv.py motion.csv --render out.mp4    # offscreen video
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_XML = ROOT / "third_party/mujoco_menagerie/unitree_g1/scene.xml"
CSV_FPS = 30.0


def load_motion(csv_path):
    m = np.loadtxt(csv_path, delimiter=",")
    assert m.shape[1] == 36, f"expected 36 cols, got {m.shape[1]}"
    qpos = np.empty_like(m)
    qpos[:, 0:3] = m[:, 0:3]                      # root xyz
    qpos[:, 3] = m[:, 6]                          # quat w
    qpos[:, 4:7] = m[:, 3:6]                      # quat xyz
    qpos[:, 7:] = m[:, 7:]                        # 29 joints
    return qpos


def render(model, data, qpos, out_path, width=854, height=480):
    import imageio
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance, cam.elevation = 3.5, -15
    writer = imageio.get_writer(out_path, fps=int(CSV_FPS))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        cam.lookat[:] = [q[0], q[1], 0.8]          # follow the root
        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())
        if i % 300 == 0:
            print(f"  frame {i}/{len(qpos)}")
    writer.close()
    print(f"wrote {out_path}")


def view(model, data, qpos):
    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as v:
        t0 = time.time()
        while v.is_running():
            i = int(((time.time() - t0) * CSV_FPS) % len(qpos))
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(model, data)
            v.sync()
            time.sleep(1 / CSV_FPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--render", metavar="OUT_MP4")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    qpos = load_motion(args.csv)
    assert model.nq == qpos.shape[1], f"model nq={model.nq} vs csv {qpos.shape[1]}"
    print(f"{args.csv}: {len(qpos)} frames = {len(qpos)/CSV_FPS:.1f}s")

    if args.render:
        render(model, data, qpos, args.render)
    if args.view:
        view(model, data, qpos)


if __name__ == "__main__":
    main()
