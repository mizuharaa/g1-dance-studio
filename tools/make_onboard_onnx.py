#!/usr/bin/env python3
"""Make an mjlab-exported dance policy DROP-IN for the robot's onboard
BeyondMimic `motion_tracking_controller` (in the g1-siu-deploy:jazzy container).

Why this exists
---------------
Our mjlab export already produces a BeyondMimic-native ONNX: input `obs`(160)+`time_step`,
outputs `actions`,`joint_pos`,`joint_vel`,`body_pos_w`,`body_quat_w`,... (the reference
motion is BAKED INTO the ONNX and advances with `time_step`). The onboard controller's
`MotionOnnxPolicy` reads exactly those tensors. The ONLY gap: our export writes NO ONNX
metadata, and the controller's `OnnxPolicy::parseMetadata` + `MotionOnnxPolicy::parseMetadata`
read the policy's config FROM ONNX metadata:

    base (legged_rl_controllers/OnnxPolicy):  joint_names, joint_stiffness, joint_damping,
                                              default_joint_pos, action_scale,
                                              observation_names, command_names
    motion (motion_tracking_controller):      anchor_body_name, body_names

Format verified against the container's reference g1 policy (comma-separated names / floats).
This tool copies our policy.onnx and injects those keys from policy_meta.json, so the
controller can load OUR gains/obs-order/motion without any C++ change — only a
config/g1/controllers.yaml that points policy_path at the output.

Usage:  python tools/make_onboard_onnx.py <policy_dir>
        (expects <dir>/policy.onnx + <dir>/policy_meta.json; writes <dir>/policy_onboard.onnx)
"""
import json, re, sys, onnx


def per_joint_action_scale(meta):
    joints = meta['joint_order_29dof']
    asd = meta['action_scale']
    if not isinstance(asd, dict):        # already scalar / vector
        return [asd] * len(joints) if not isinstance(asd, list) else asd
    out = []
    for j in joints:
        hit = next((v for pat, v in asd.items() if re.fullmatch(pat, j)), None)
        if hit is None:
            raise KeyError(f"no action_scale regex matches joint {j}")
        out.append(hit)
    return out


def build_metadata(meta):
    def csv(v, p): return ",".join(f"{x:.{p}f}" for x in v)
    return {
        "joint_names":       ",".join(meta['joint_order_29dof']),
        "body_names":        ",".join(meta['tracked_body_names']),
        "anchor_body_name":  meta['anchor_body_name'],
        # obs term order MUST equal training order or the 160-vec is scrambled:
        "observation_names": ",".join(meta['actor_obs_terms_in_order']),
        "command_names":     "motion",
        "joint_stiffness":   csv(meta['kp_stiffness'], 3),
        "joint_damping":     csv(meta['kd_damping'], 3),
        "default_joint_pos": csv(meta['default_joint_pos_rad'], 4),
        "action_scale":      csv(per_joint_action_scale(meta), 6),
    }


def main(pdir):
    meta = json.load(open(f"{pdir}/policy_meta.json"))
    md = build_metadata(meta)
    model = onnx.load(f"{pdir}/policy.onnx")
    del model.metadata_props[:]
    for k, v in md.items():
        p = model.metadata_props.add(); p.key = k; p.value = v
    out = f"{pdir}/policy_onboard.onnx"
    onnx.save(model, out)
    # verify read-back
    import onnxruntime as ort
    s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    got = s.get_modelmeta().custom_metadata_map
    assert set(got) == set(md), f"metadata mismatch: {set(md) ^ set(got)}"
    ins = {i.name for i in s.get_inputs()}
    assert {"obs", "time_step"} <= ins, f"missing ONNX inputs: {ins}"
    print(f"wrote {out}")
    print(f"  metadata keys: {sorted(got)}")
    print(f"  observation_names: {got['observation_names']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else "data/policies/thriller_standtail_candidate"))
