"""Truth-in-labeling and scene-provenance regressions for preview reports (F5)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")

from tools import sim_sandbox, sim_studio  # noqa: E402


def test_hardware_uncertainty_banner_is_explicit_not_a_positive_training_claim():
    banner = sim_sandbox.model_caveat(sim_sandbox.HARDWARE_UNCERTAINTY)
    studio_banner, _color = sim_studio._banner(sim_sandbox.HARDWARE_UNCERTAINTY)
    assert banner == sim_sandbox.HARDWARE_UNCERTAINTY_BANNER
    assert studio_banner == banner
    assert "hardware-uncertainty" in banner
    assert "NOT the pinned mjlab training model" in banner
    assert "PREVIEW on the mjlab TRAINING model" not in banner


def test_sandbox_report_carries_default_scene_name_and_full_xml_hash():
    scene = sim_sandbox.scene_identity(sim_sandbox.HARDWARE_UNCERTAINTY)
    expected = hashlib.sha256(sim_sandbox.HARDWARE_UNCERTAINTY.read_bytes()).hexdigest()
    assert scene == {
        "name": "hardware-uncertainty-v1",
        "xml_sha256": expected,
    }

    out = {
        "q": np.zeros((2, 1)),
        "ref_jp": np.ones((2, 1)),
        "fell_at_tick": None,
        "scene": scene,
    }
    assert sim_sandbox.tracking_report(out)["scene"] == scene


def test_studio_report_carries_same_scene_identity():
    left = {"kind": "REFERENCE", "achieved": 1.0}
    right = {"kind": "POLICY", "achieved": 0.8, "fell_at": None}
    report = sim_studio._report_payload(
        left, right, sim_sandbox.HARDWARE_UNCERTAINTY
    )
    assert report["scene"] == sim_sandbox.scene_identity(
        sim_sandbox.HARDWARE_UNCERTAINTY
    )
