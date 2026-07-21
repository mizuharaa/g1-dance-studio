"""Policy-specific metadata and strict deploy bundle regression tests (F3)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from cloud import export_ckpt_onnx
from pipeline import artifacts, deploy_runtime, publish_policy


GROUND_TERMS = [
    ("command", 58),
    ("motion_anchor_ori_b", 6),
    ("base_ang_vel", 3),
    ("joint_pos", 29),
    ("joint_vel", 29),
    ("actions", 29),
]


def _tiny_onnx(path: Path, obs_width: int | str = 770) -> Path:
    obs = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_width])
    time_step = helper.make_tensor_value_info("time_step", TensorProto.FLOAT, [1, 1])
    actions = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 29])
    value = helper.make_tensor("action_value", TensorProto.FLOAT, [1, 29], [0.0] * 29)
    node = helper.make_node("Constant", [], ["actions"], value=value)
    graph = helper.make_graph([node], "tiny_policy", [obs, time_step], [actions])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )
    onnx.save(model, path)
    return path


def _motion_npz(path: Path) -> Path:
    np.savez(
        path,
        joint_pos=np.zeros((2, 29)),
        joint_vel=np.zeros((2, 29)),
        body_pos_w=np.zeros((2, 16, 3)),
        body_quat_w=np.zeros((2, 16, 4)),
    )
    return path


def _meta(inputs: dict | None = None) -> dict:
    return {
        "schema": "g1.policy_meta/2",
        "policy_meta_version": 2,
        "onnx_inputs": inputs or {"obs": [1, 770], "time_step": [1, 1]},
        "obs_per_frame": 154,
        "history_length": 5,
        "flatten_layout": "term-major-oldest-first",
        "actor_obs_terms_in_order": [name for name, _ in GROUND_TERMS],
        "actor_obs_term_widths": [[name, width] for name, width in GROUND_TERMS],
        "requires_ground_contact": True,
    }


class _FakeObservationManager:
    active_terms = {"actor": [name for name, _ in GROUND_TERMS]}
    group_obs_term_dim = {
        "actor": [(width * 5,) for _, width in GROUND_TERMS]
    }

    @staticmethod
    def get_term_cfg(group, name):
        assert group == "actor"
        assert name in _FakeObservationManager.active_terms["actor"]
        return SimpleNamespace(history_length=5, flatten_history_dim=True)


def _fake_live_env():
    names = [f"joint_{idx}" for idx in range(29)]
    actuators = [SimpleNamespace(target=f"robot/{name}", id=idx)
                 for idx, name in enumerate(names)]
    robot = SimpleNamespace(
        joint_names=names,
        spec=SimpleNamespace(actuators=actuators),
        data=SimpleNamespace(default_joint_pos=np.arange(29, dtype=float)[None] / 100),
    )
    gain = np.zeros((29, 3)); gain[:, 0] = np.arange(29) + 10
    bias = np.zeros((29, 3)); bias[:, 2] = -(np.arange(29) + 1)
    force = np.column_stack((-(np.arange(29) + 20), np.arange(29) + 20))
    action = SimpleNamespace(
        target_names=names,
        _scale=np.linspace(0.1, 0.3, 29)[None],
        cfg=SimpleNamespace(use_default_offset=True),
    )
    return SimpleNamespace(
        observation_manager=_FakeObservationManager(),
        action_manager=SimpleNamespace(get_term=lambda name: action),
        scene={"robot": robot},
        sim=SimpleNamespace(mj_model=SimpleNamespace(
            actuator_gainprm=gain,
            actuator_biasprm=bias,
            actuator_forcerange=force,
        )),
    )


def test_export_builds_meta_v2_from_live_cfg_and_manifest(tmp_path):
    onnx_path = _tiny_onnx(tmp_path / "policy.onnx")
    env_cfg = SimpleNamespace(sim=SimpleNamespace(dt=0.005), decimation=4)
    meta = export_ckpt_onnx.build_policy_meta(
        env_cfg,
        _fake_live_env(),
        onnx_path,
        task_id="fake-ground-task",
        task_module="fake_task",
        checkpoint="runs/fake/model_42.pt",
        requires_ground_contact=True,
    )

    assert meta["schema"] == "g1.policy_meta/2"
    assert meta["actor_obs_terms_in_order"] == [name for name, _ in GROUND_TERMS]
    assert meta["actor_obs_term_widths"] == [[name, width] for name, width in GROUND_TERMS]
    assert meta["obs_per_frame"] == 154
    assert meta["history_length"] == 5
    assert meta["onnx_inputs"] == {"obs": [1, 770], "time_step": [1, 1]}
    assert meta["joint_names"] == [f"joint_{idx}" for idx in range(29)]
    assert meta["kp"][0] == 10
    assert meta["kd"][0] == 1
    assert meta["effort_limit_nm"][-1] == 48
    assert meta["action_scale"] == pytest.approx(np.linspace(0.1, 0.3, 29))

    manifest = export_ckpt_onnx.write_policy_artifacts(
        tmp_path,
        onnx_path,
        meta,
        task_id="fake-ground-task",
        task_module="fake_task",
        checkpoint="runs/fake/model_42.pt",
    )
    assert manifest["schema"] == artifacts.SCHEMA
    assert manifest["policy"]["obs_per_frame"] == 154
    assert manifest["policy"]["history_length"] == 5
    assert manifest["model"]["task_id"] == "fake-ground-task"
    assert artifacts.verify_manifest(tmp_path / "bundle.json") == []


def test_publish_never_copies_shared_meta(tmp_path, monkeypatch):
    policy_dir = tmp_path / "pulled"
    policy_dir.mkdir()
    _tiny_onnx(policy_dir / "policy.onnx")
    (policy_dir / "own_deploy.npz").write_bytes(b"motion-placeholder")
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "policy_meta.json").write_text(json.dumps(_meta()))
    monkeypatch.setattr(publish_policy, "_SHARED", shared)

    assert publish_policy.ensure_preview_assets(policy_dir, log=lambda _msg: None)
    assert not (policy_dir / "policy_meta.json").exists()


def test_publish_mismatched_meta_registers_without_preview(
    tmp_path, monkeypatch, dances_env
):
    shows, _ = dances_env
    policy_dir = tmp_path / "pulled"
    policy_dir.mkdir()
    _tiny_onnx(policy_dir / "policy.onnx")
    (policy_dir / "own_deploy.npz").write_bytes(b"motion-placeholder")
    bad = _meta({"obs": [1, 160], "time_step": [1, 1]})
    (policy_dir / "policy_meta.json").write_text(json.dumps(bad))

    old = shows.new_dance("Mismatched policy")
    old.preview = "/previews/stale.mp4"
    old.save()
    rendered = []
    monkeypatch.setattr(
        publish_policy.sim_preview,
        "render_sync",
        lambda dance: rendered.append(dance.id) or {"status": "ready", "sha": "x"},
    )

    logs = []
    dance = publish_policy.publish(policy_dir, old.name, log=logs.append)
    assert dance is not None
    assert dance.id == old.id
    assert dance.policy_path == str(policy_dir / "policy.onnx")
    assert dance.preview is None
    assert rendered == []
    assert any("REFUSED preview" in line and "onnx_inputs" in line for line in logs)


@pytest.mark.parametrize(
    ("obs_width", "mutate", "message"),
    [
        ("dynamic_obs", lambda meta: None, "dynamic"),
        (771, lambda meta: meta.update(onnx_inputs={"obs": [1, 771], "time_step": [1, 1]}),
         "not an exact multiple"),
        (770, lambda meta: meta.update(onnx_inputs={"obs": [1, 160], "time_step": [1, 1]}),
         "onnx_inputs"),
        (770, lambda meta: meta.update(history_length=4), "history_length"),
        (770, lambda meta: (
            meta["actor_obs_terms_in_order"].__setitem__(1, "base_lin_vel"),
            meta["actor_obs_term_widths"][1].__setitem__(0, "base_lin_vel"),
        ), "estimator"),
        (770, lambda meta: meta.pop("requires_ground_contact"), "requires_ground_contact"),
    ],
)
def test_deploy_bundle_validation_refuses_each_contract_defect(
    tmp_path, obs_width, mutate, message
):
    case = tmp_path / str(message).replace(" ", "_")
    case.mkdir()
    policy = _tiny_onnx(case / "policy.onnx", obs_width)
    motion = _motion_npz(case / "motion.npz")
    actual_width = obs_width
    meta = _meta({"obs": [1, actual_width], "time_step": [1, 1]})
    mutate(meta)
    meta_path = case / "policy_meta.json"
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(SystemExit, match=message):
        deploy_runtime.validate_policy_bundle(
            meta_path, policy, motion, "ground-run"
        )


def test_deploy_good_bundle_passes_and_preflight_precedes_robot_access(
    tmp_path, monkeypatch
):
    policy = _tiny_onnx(tmp_path / "policy.onnx")
    motion = _motion_npz(tmp_path / "motion.npz")
    meta_path = tmp_path / "policy_meta.json"
    meta_path.write_text(json.dumps(_meta()))
    parsed = deploy_runtime.validate_policy_bundle(
        meta_path, policy, motion, "ground-run"
    )
    assert parsed["history_length"] == 5

    bad = _meta({"obs": [1, 160], "time_step": [1, 1]})
    meta_path.write_text(json.dumps(bad))
    touched = []
    monkeypatch.setattr(deploy_runtime, "make_dds", lambda *_args: touched.append("dds"))
    monkeypatch.setattr(
        deploy_runtime, "_require_human", lambda *_args: touched.append("human")
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "deploy_runtime",
            "--mode", "ground-run",
            "--ground-meta", str(meta_path),
            "--ground-policy", str(policy),
            "--ground-motion", str(motion),
            "--i-will-watch-the-robot",
            "--max-secs", "1",
        ],
    )
    with pytest.raises(SystemExit, match="onnx_inputs"):
        deploy_runtime.main()
    assert touched == []
