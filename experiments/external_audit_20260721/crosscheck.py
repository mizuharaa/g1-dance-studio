#!/usr/bin/env python3
"""Independent CPU/static cross-checks for the 2026-07-21 external audit.

Run from the repository root in the ``g1dance`` conda environment:

    python experiments/external_audit_20260721/crosscheck.py \
      --mjlab-wheel /path/to/mjlab-1.5.0-py3-none-any.whl \
      --output experiments/external_audit_20260721/crosscheck.json

The wheel is inspected as a zip and temporarily extracted only to load its pinned
G1 XML.  Nothing is installed and no product files are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V12 = ROOT / "data/motions/thriller/thriller_v12_full.csv"
V12_CARD = ROOT / "data/motions/thriller/thriller_v12_full_scorecard.json"
PREVIEW_XML = ROOT / "tools/assets/g1_faithful/g1_mjlab_faithful.xml"
GROUND_XML = ROOT / "third_party/mujoco_menagerie/unitree_g1/scene.xml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str, binary: bool = False):
    out = subprocess.check_output(["git", *args], cwd=ROOT)
    return out if binary else out.decode().strip()


def load_git_csv(revision: str, path: str) -> np.ndarray:
    raw = git("show", f"{revision}:{path}", binary=True)
    return np.loadtxt(io.BytesIO(raw), delimiter=",")


def model_summary(model: mujoco.MjModel) -> dict:
    body_mass = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}":
        float(model.body_mass[i])
        for i in range(1, model.nbody)
    }
    robot_geom_ids = np.flatnonzero(model.geom_bodyid != 0)
    # Both source XMLs leave the four sole spheres per foot unnamed. Identify them
    # by their ankle-roll parent body, sphere type, and enabled contact bit.
    foot_ids = []
    for raw_i in robot_geom_ids:
        i = int(raw_i)
        body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])
        ) or ""
        if (
            "ankle_roll_link" in body
            and int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
            and int(model.geom_contype[i]) != 0
        ):
            foot_ids.append(i)
    return {
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "njnt": int(model.njnt),
        "nu": int(model.nu),
        "total_robot_mass_kg": round(float(model.body_mass[1:].sum()), 6),
        "body_mass_kg": body_mass,
        "robot_geom_count": int(len(robot_geom_ids)),
        "sole_sphere_count": len(foot_ids),
        "sole_sphere_slide_friction": sorted({
            round(float(model.geom_friction[i, 0]), 6) for i in foot_ids
        }),
    }


def pinned_wheel_checks(wheel: Path) -> tuple[dict, dict[str, str]]:
    wanted = {
        "effort": "mjlab/envs/mdp/dr/actuator.py",
        "entity": "mjlab/managers/scene_entity_config.py",
        "obs": "mjlab/managers/observation_manager.py",
        "buffer": "mjlab/utils/buffers/circular_buffer.py",
        "command": "mjlab/tasks/tracking/mdp/commands.py",
        "events": "mjlab/envs/mdp/events.py",
        "event_manager": "mjlab/managers/event_manager.py",
        "g1": "mjlab/asset_zoo/robots/unitree_g1/g1_constants.py",
        "cfg": "mjlab/envs/manager_based_rl_env.py",
    }
    with zipfile.ZipFile(wheel) as zf:
        sources = {key: zf.read(name).decode() for key, name in wanted.items()}
        assertions = {
            "effort_uses_actuator_ids": (
                "asset_cfg.actuator_ids" in sources["effort"]
                and "asset_cfg.joint_ids" not in sources["effort"].split(
                    "def effort_limits", 1
                )[1].split("\ndef ", 1)[0]
            ),
            "effort_scales_pristine_default": (
                'get_default_field("actuator_forcerange")' in sources["effort"]
            ),
            "joint_and_actuator_selectors_are_independent": (
                '"joint_names", "joint_ids"' in sources["entity"]
                and '"actuator_names", "actuator_ids"' in sources["entity"]
            ),
            "cfg_commands_are_dict": 'commands: dict[str, CommandTermCfg]' in sources["cfg"],
            "history_flattens_each_term_before_concat": (
                "circular_buffer.buffer.reshape" in sources["obs"]
            ),
            "history_buffer_oldest_to_newest": "index 0 is oldest" in sources["buffer"],
            "history_first_frame_backfill": "Backfill entire history with first frame" in sources["buffer"],
            "motion_wrap_resamples": (
                "self.time_steps >= self.motion.time_step_total" in sources["command"]
                and "self._resample_command(env_ids)" in sources["command"]
            ),
            "resample_writes_reference_state": "self._write_reference_state_to_sim(" in sources["command"],
            "push_is_mass_independent_velocity_overwrite": (
                "instantaneous, mass-independent" in sources["events"]
                and "write_root_link_velocity_to_sim" in sources["events"]
            ),
            "startup_events_preserve_dict_order": (
                "for term_name, term_cfg in self.cfg.items()" in sources["event_manager"]
                and "self._mode_term_cfgs[term_cfg.mode].append(term_cfg)"
                in sources["event_manager"]
                and "for index, term_cfg in enumerate(self._mode_term_cfgs[mode])"
                in sources["event_manager"]
            ),
            "g1_full_collision_friction_06": (
                "collisions=(FULL_COLLISION,)" in sources["g1"]
                and "friction={r\"^(left|right)_foot[1-7]_collision$\": (0.6,)}" in sources["g1"]
            ),
        }
        xml_prefix = "mjlab/asset_zoo/robots/unitree_g1/xmls/"
        with tempfile.TemporaryDirectory(prefix="g1-audit-mjlab-") as tmp:
            for member in zf.namelist():
                if member.startswith(xml_prefix) and not member.endswith("/"):
                    zf.extract(member, tmp)
            xml = Path(tmp) / xml_prefix / "g1.xml"
            pinned_model = model_summary(mujoco.MjModel.from_xml_path(str(xml)))

    source_hashes = {
        wanted[key]: hashlib.sha256(text.encode()).hexdigest()
        for key, text in sources.items()
    }
    return {
        "wheel": str(wheel),
        "wheel_sha256": sha256(wheel),
        "source_sha256": source_hashes,
        "semantic_assertions": assertions,
        "all_semantic_assertions_pass": all(assertions.values()),
        "raw_g1_xml": pinned_model,
    }, sources


def effort_scope_check() -> dict:
    source = (ROOT / "cloud/sim2real_task_v8.py").read_text()
    configured = {
        "stock_event_joint_names": 'joint_names=NON_ANKLE_JOINT_PATTERNS' in source,
        "clamp_event_joint_names": 'joint_names=base.ANKLE_JOINT_NAMES' in source,
        "composed_range": "ANKLE_EFFORT_DR_COMPOSED" in source,
    }
    nominal_classes = {
        "wrists_pitch_yaw": 5.0,
        "arms": 25.0,
        "waist_roll_pitch_and_ankles": 50.0,
        "hip_pitch_yaw_and_waist_yaw": 88.0,
        "knees_and_hip_roll": 139.0,
    }
    final_scale = (0.52, 0.76)
    realized = {
        name: [round(value * final_scale[0], 2), round(value * final_scale[1], 2)]
        for name, value in nominal_classes.items()
    }
    return {
        "repo_selector_configuration": configured,
        "final_event_scale": list(final_scale),
        "effect_given_pinned_effort_semantics": "all actuator groups selected",
        "realized_effort_range_nm_by_nominal_class": realized,
    }


def v12_checks() -> dict:
    from pipeline.grounding import per_contact_height

    current = np.loadtxt(V12, delimiter=",")
    parent = load_git_csv("57e4b2c^", "data/motions/thriller/thriller_v12_full.csv")
    delta = current - parent
    card = json.loads(V12_CARD.read_text())
    contact = per_contact_height(current, mujoco.MjModel.from_xml_path(str(GROUND_XML)))
    adaptive_path = ROOT / "data/motions/thriller/thriller_g1_grounded_adaptive.csv"
    adaptive = np.loadtxt(adaptive_path, delimiter=",")
    adaptive_contact = per_contact_height(
        adaptive, mujoco.MjModel.from_xml_path(str(GROUND_XML))
    )
    return {
        "current_csv": {
            "sha256": sha256(V12),
            "shape": list(current.shape),
        },
        "scorecard": {
            "recorded_repaired_sha256": card.get("repaired_sha256"),
            "recorded_source_sha256": card.get("source_sha256"),
            "recorded_source": card.get("source"),
            "hash_matches_current": card.get("repaired_sha256") == sha256(V12),
            "recorded_final": card.get("final"),
            "recorded_style_similarity": card.get("style_similarity"),
        },
        "fix_a_commit_delta": {
            "parent_sha256": hashlib.sha256(
                git("show", "57e4b2c^:data/motions/thriller/thriller_v12_full.csv", binary=True)
            ).hexdigest(),
            "changed_rows": int(np.any(delta != 0, axis=1).sum()),
            "changed_joint_columns": int(np.any(delta[:, 7:] != 0, axis=0).sum()),
            "root_xyz_quat_max_abs_delta": float(np.abs(delta[:, :7]).max()),
            "joint_max_abs_delta_rad": float(np.abs(delta[:, 7:]).max()),
            "joint_rms_delta_rad": float(np.sqrt(np.mean(delta[:, 7:] ** 2))),
        },
        "contact_height_current": {
            "min_m": float(contact.min()),
            "max_m": float(contact.max()),
            "median_m": float(np.median(contact)),
            "p95_m": float(np.percentile(contact, 95)),
            "range_mm": float(np.ptp(contact) * 1000),
            "frames_gt_100mm_pct": float(np.mean(contact > 0.1) * 100),
        },
        "contact_height_known_per_frame_grounded_comparator": {
            "file": str(adaptive_path.relative_to(ROOT)),
            "min_m": float(adaptive_contact.min()),
            "max_m": float(adaptive_contact.max()),
            "median_m": float(np.median(adaptive_contact)),
            "p95_m": float(np.percentile(adaptive_contact, 95)),
            "range_mm": float(np.ptp(adaptive_contact) * 1000),
            "frames_gt_100mm_pct": float(np.mean(adaptive_contact > 0.1) * 100),
        },
    }


def synthetic_flight_check() -> dict:
    from pipeline.grounding import (
        FLIGHT_BAND_M,
        FLIGHT_MIN_S,
        GROUND_SMOOTH_WIN,
        _sg_smooth_1d,
    )

    fps = 30.0
    contact = np.zeros(180)
    contact[60:120] = 0.2  # two-second, 20 cm airborne plateau
    ground_line = _sg_smooth_1d(contact, GROUND_SMOOTH_WIN)
    air = (contact - ground_line) > FLIGHT_BAND_M
    runs = []
    start = None
    for i, value in enumerate(np.r_[air, False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append(i - start)
            start = None
    remaining = contact - ground_line
    return {
        "input": "2.0 s plateau, +0.20 m at 30 Hz",
        "input_peak_m": float(contact.max()),
        "flight_band_m": FLIGHT_BAND_M,
        "minimum_flight_frames": int(round(FLIGHT_MIN_S * fps)),
        "detected_frames": int(air.sum()),
        "longest_detected_run": max(runs, default=0),
        "ground_line_peak_m": float(ground_line.max()),
        "remaining_peak_m_before_residual_lift": float(remaining.max()),
        "remaining_plateau_median_m": float(np.median(remaining[65:115])),
    }


def policy_contract_checks() -> list[dict]:
    dirs = [
        "exports/train-thriller_v8s2r-0716",
        "exports/train-thriller_v10spd-0720",
        "exports/train-thriller_v11leg-0720",
    ]
    out = []
    for rel in dirs:
        path = ROOT / rel
        if not (path / "policy.onnx").is_file() or not (path / "policy_meta.json").is_file():
            continue
        session = ort.InferenceSession(
            str(path / "policy.onnx"), providers=["CPUExecutionProvider"]
        )
        meta = json.loads((path / "policy_meta.json").read_text())
        out.append({
            "dir": rel,
            "policy_sha256": sha256(path / "policy.onnx"),
            "onnx_inputs_actual": {i.name: i.shape for i in session.get_inputs()},
            "onnx_inputs_meta": meta.get("onnx_inputs"),
            "task_meta": meta.get("task"),
            "actor_obs_terms_meta": meta.get("actor_obs_terms_in_order"),
            "requires_ground_contact_meta": meta.get("requires_ground_contact"),
        })
    return out


def eval_horizon_check() -> dict:
    gap_path = ROOT / "exports/train-thriller_v11leg-0720/gap.json"
    npz_path = ROOT / "exports/train-thriller_v11leg-0720/thriller_v11_deploy.npz"
    gap = json.loads(gap_path.read_text())
    motion = np.load(npz_path)
    frames = int(motion["joint_pos"].shape[0])
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    nominal = gap["conditions"]["nominal"]
    return {
        "motion_frames": frames,
        "motion_fps": fps,
        "motion_duration_s": frames / fps,
        "gate_episode_length_s": gap.get("episode_length_s"),
        "nominal_steps_run": nominal.get("steps_run"),
        "steps_after_motion_end": int(nominal.get("steps_run", 0) - frames),
        "reported_nominal_delay_ms": nominal.get("delay_ms"),
        "reported_nominal_success_rate": nominal.get("success_rate"),
    }


def show_safety_checks() -> dict:
    """Exercise show authorization seams without reading secrets or spawning a process."""
    from pipeline import exam_verdict, show_runner, shows

    original_dances_dir = shows.DANCES_DIR
    original_show_root = show_runner.PROJECT_ROOT
    original_key_path = exam_verdict._SIGNING_KEY_PATH
    try:
        with tempfile.TemporaryDirectory(prefix="g1-audit-show-") as tmp_raw:
            tmp = Path(tmp_raw)

            # An incomplete selected bundle returns no CLI overrides.  The real launcher
            # consequently invokes deploy_runtime's parser defaults.
            bundle = tmp / "bundle"
            bundle.mkdir()
            (bundle / "policy.onnx").write_bytes(b"audit-placeholder")
            show_runner.PROJECT_ROOT = tmp
            incomplete_args = show_runner._dance_policy_args(
                SimpleNamespace(policy_path="bundle/policy.onnx")
            )
            deploy_source = (ROOT / "pipeline/deploy_runtime.py").read_text()
            parser_has_defaults = all(
                token in deploy_source
                for token in (
                    'ap.add_argument("--meta", default=str(DEFAULT_META))',
                    'ap.add_argument("--motion-npz", default=str(DEFAULT_MOTION))',
                    'ap.add_argument("--policy", default=str(DEFAULT_POLICY))',
                )
            )

            # Record the same authentic verdict three times in an isolated store.  Use
            # a known temporary key so this check never opens or creates .secrets/.
            key = b"external-audit-temporary-key-32b"
            key_path = tmp / "exam-signing.key"
            key_path.write_bytes(key)
            exam_verdict._SIGNING_KEY_PATH = key_path
            shows.DANCES_DIR = tmp / "dances"
            policy = tmp / "policy.onnx"
            motion = tmp / "motion.csv"
            policy.write_bytes(b"policy")
            motion.write_bytes(b"motion")
            now = 1_721_500_000.0
            dance = shows.Dance(
                id="audit-dance",
                name="Audit dance",
                created_at=now,
                updated_at=now,
                policy_path=str(policy),
                motion_csv=str(motion),
            )
            dance.save()
            verdict = exam_verdict.sign_verdict(
                {
                    "schema": "sim_exam/v1",
                    "policy_sha256": sha256(policy),
                    "motion_sha256": sha256(motion),
                    "nominal": {"pass": True},
                    "push": {"pass": True, "force_n": 875.0},
                    "repeatability": {"pass": True, "runs": 128, "clean": 128},
                },
                key=key,
            )
            credit_after_each_submission = []
            for _ in range(3):
                credited = shows.record_sim_run_from_verdict(dance.id, verdict)
                credit_after_each_submission.append(
                    credited.repeatability["consecutive_clean"]
                )
            stored = shows.load_dance(dance.id)
            exam_ids = [item["exam_id"] for item in stored.repeatability["history"]]

            return {
                "incomplete_bundle": {
                    "policy_exists": (bundle / "policy.onnx").exists(),
                    "metadata_exists": (bundle / "policy_meta.json").exists(),
                    "deploy_npz_exists": any(bundle.glob("*_deploy.npz")),
                    "returned_cli_overrides": incomplete_args,
                    "deploy_parser_has_default_bundle": parser_has_defaults,
                    "therefore_launch_uses_unselected_defaults": (
                        incomplete_args == [] and parser_has_defaults
                    ),
                },
                "signed_verdict_replay": {
                    "signature_valid": exam_verdict.signature_valid(verdict),
                    "has_unique_eval_id": "at" in verdict,
                    "credit_after_each_identical_submission": credit_after_each_submission,
                    "stored_exam_ids_newest_first": exam_ids,
                    "promotion_streak_target": shows.REPEATABILITY_TARGET,
                    "identical_submissions_reach_target": (
                        stored.repeatability["consecutive_clean"]
                        >= shows.REPEATABILITY_TARGET
                    ),
                },
            }
    finally:
        shows.DANCES_DIR = original_dances_dir
        show_runner.PROJECT_ROOT = original_show_root
        exam_verdict._SIGNING_KEY_PATH = original_key_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjlab-wheel", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    wheel_info, _sources = pinned_wheel_checks(args.mjlab_wheel.resolve())
    preview = model_summary(mujoco.MjModel.from_xml_path(str(PREVIEW_XML)))
    pinned = wheel_info["raw_g1_xml"]
    common = sorted(set(pinned["body_mass_kg"]) & set(preview["body_mass_kg"]))
    body_delta = {
        name: round(preview["body_mass_kg"][name] - pinned["body_mass_kg"][name], 6)
        for name in common
        if abs(preview["body_mass_kg"][name] - pinned["body_mass_kg"][name]) > 1e-9
    }
    result = {
        "audit_baseline": {
            "git_head": git("rev-parse", "HEAD"),
            "git_head_subject": git("log", "-1", "--format=%s"),
        },
        "pinned_mjlab": wheel_info,
        "effort_scope": effort_scope_check(),
        "v12_motion": v12_checks(),
        "grounding_synthetic_flight": synthetic_flight_check(),
        "policy_contracts": policy_contract_checks(),
        "eval_horizon": eval_horizon_check(),
        "show_safety": show_safety_checks(),
        "model_comparison": {
            "pinned_raw_training_xml": pinned,
            "claimed_faithful_preview": preview,
            "preview_minus_pinned_total_mass_kg": round(
                preview["total_robot_mass_kg"] - pinned["total_robot_mass_kg"], 6
            ),
            "preview_minus_pinned_body_mass_kg_nonzero": body_delta,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "head": result["audit_baseline"]["git_head"],
        "wheel_sha256": wheel_info["wheel_sha256"],
        "semantic_assertions_pass": wheel_info["all_semantic_assertions_pass"],
        "v12_hash_matches_scorecard": result["v12_motion"]["scorecard"]["hash_matches_current"],
        "v12_contact_range_mm": result["v12_motion"]["contact_height_current"]["range_mm"],
        "policies_checked": len(result["policy_contracts"]),
    }, indent=2))


if __name__ == "__main__":
    main()
