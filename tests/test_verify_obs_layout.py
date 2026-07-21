"""CPU contract tests for the box-only actor history verifier (F3)."""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from cloud import verify_obs_layout
from pipeline.deploy_runtime import HistoryStacker


def test_verifier_rebuild_is_byte_identical_to_deploy_history_stacker():
    order = [("alpha", 2), ("beta", 1), ("gamma", 3)]
    frames = [
        {
            "alpha": np.array([10.0 + tick, 20.0 + tick]),
            "beta": np.array([30.0 + tick]),
            "gamma": np.array([40.0 + tick, 50.0 + tick, 60.0 + tick]),
        }
        for tick in range(3)
    ]
    stacker = HistoryStacker(order, n_hist=3)
    for frame in frames:
        deployed = stacker.push(frame)

    histories = {
        name: np.stack([frame[name] for frame in frames])
        for name, _width in order
    }
    verified = verify_obs_layout.rebuild_term_major(
        histories, [name for name, _width in order]
    )
    assert verified.tobytes() == deployed.tobytes()
    np.testing.assert_array_equal(
        verified,
        [10, 20, 11, 21, 12, 22, 30, 31, 32,
         40, 50, 60, 41, 51, 61, 42, 52, 62],
    )


def test_verifier_rejects_missing_term_history():
    try:
        verify_obs_layout.rebuild_term_major(
            {"present": np.zeros((2, 3))}, ["present", "missing"]
        )
    except KeyError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing actor term history was accepted")


def test_cli_api_failure_returns_3_and_names_exact_attribute(
    monkeypatch, capsys, tmp_path
):
    mjlab = ModuleType("mjlab")
    envs = ModuleType("mjlab.envs")
    tasks = ModuleType("mjlab.tasks")
    registry = ModuleType("mjlab.tasks.registry")
    envs.ManagerBasedRlEnv = object
    # Deliberately omit commands["motion"] after the preceding cfg access succeeds.
    registry.load_env_cfg = lambda _task: SimpleNamespace(
        scene=SimpleNamespace(num_envs=0),
        commands={},
        observations={"actor": SimpleNamespace(history_length=5)},
    )
    monkeypatch.setitem(sys.modules, "mjlab", mjlab)
    monkeypatch.setitem(sys.modules, "mjlab.envs", envs)
    monkeypatch.setitem(sys.modules, "mjlab.tasks", tasks)
    monkeypatch.setitem(sys.modules, "mjlab.tasks.registry", registry)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_obs_layout.py",
            "--task", "fake",
            "--motion-file", str(tmp_path / "motion.npz"),
        ],
    )

    assert verify_obs_layout.main() == 3
    assert 'cfg.commands["motion"].motion_file' in capsys.readouterr().err
