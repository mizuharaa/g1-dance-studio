"""Sim2sim exam gate: run a trained tracking policy closed-loop in plain MuJoCo.

This is the Stage-4 "different simulator" exam (architecture.md): policies train in
Isaac Lab / mjlab, but must pass here — plain MuJoCo, unitree_mujoco G1 model, PD
torque control at the deploy rate — before anything is allowed near the robot.

Exam phases
-----------
1. nominal        full-motion rollout: survival, tracking error, <=1.5 m excursion
2. push           randomized horizontal shoves on the torso; recovery rate
3. repeatability  N seeded nominal reruns; consecutive-clean counter

Verdict JSON follows the sim_exam/v1 contract in docs/show_mode_contracts.md.

INTERFACE ASSUMPTIONS — training orchestrator, please confirm against the real export
--------------------------------------------------------------------------------------
Written against whole_body_tracking@cd65172 utils/exporter.py (BeyondMimic reference):

* policy.onnx graph:  inputs  obs[1,D] f32, time_step[1,1] f32
                      outputs actions[1,29] first, then baked reference tensors
                      (joint_pos, joint_vel, body_pos_w, body_quat_w, ...) - unused
                      here; we take reference truth from the motion CSV instead.
* onnx metadata_props (csv strings): joint_names, joint_stiffness, joint_damping,
  default_joint_pos, action_scale, observation_names, observation_history_lengths,
  anchor_body_name, body_names.
* action -> PD target:  q_des = default_joint_pos + action_scale * action   (Isaac
  joint order); torque tau = kp*(q_des-q) - kd*qd on MuJoCo torque motors.
* control at 50 Hz = sim dt 0.005 x decimation 4 (tracking_env_cfg.py); time_step
  input = control tick index (motion npz is 50 fps).
* obs terms supported (order from metadata observation_names):
    command                 2x29  motion joint_pos+joint_vel at t (Isaac order)
    motion_anchor_pos_b     3     motion anchor pos in robot-anchor frame
    motion_anchor_ori_b     6     first two rot-matrix columns of same
    base_lin_vel/base_ang_vel  3+3  root velocities in base frame
    joint_pos/joint_vel     29+29 relative to default / absolute vel
    actions                 29    previous raw action
  History lengths > 1 are supported by tiling the per-term buffer (oldest first),
  matching Isaac's ObservationManager flattening.
* If mjlab exports a different graph, add a PolicyAdapter subclass — the exam loop,
  metrics and verdict logic are adapter-agnostic. Open question tracked in
  docs/deploy_kit_built.md.

Known sim2sim fidelity gaps (accepted): implicit-actuator PD (PhysX) vs explicit
torque PD here; no observation noise replay; capsule-vs-mesh collision differences.
A policy that only works with Isaac quirks SHOULD fail here — that is the point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from pipeline.config import PROJECT_ROOT, THIRD_PARTY

CSV_FPS = 30.0
CTRL_HZ = 50.0
SIM_DT = 0.005
DECIMATION = 4  # 4 * 0.005 s = 20 ms = 50 Hz
EXCURSION_LIMIT_M = 1.5  # vet gate / 2 m dance-area rule
ANCHOR_Z_FAIL_M = 0.25  # mirrors bad_anchor_pos_z_only termination
ANCHOR_ORI_FAIL = 0.8  # mirrors bad_anchor_ori termination (rad, geodesic)
PUSH_DURATION_S = 0.1
PUSH_RECOVERY_S = 2.0
PUSH_RECOVER_ERR_M = 0.25


def _resolve_third_party() -> Path:
    """Worktrees do not carry the (gitignored) third_party clones — fall back."""
    if (THIRD_PARTY / "unitree_mujoco").exists():
        return THIRD_PARTY
    fallback = Path.home() / "g1-dance" / "third_party"
    if (fallback / "unitree_mujoco").exists():
        return fallback
    raise FileNotFoundError("third_party/unitree_mujoco not found (main tree or worktree)")


G1_XML = _resolve_third_party() / "unitree_mujoco" / "unitree_robots" / "g1" / "scene_29dof.xml"


# --------------------------------------------------------------------------- motion
@dataclass
class Motion:
    """Reference motion resampled from the 30 fps CSV to control ticks (50 Hz)."""

    root_pos: np.ndarray  # [T,3]
    root_quat_wxyz: np.ndarray  # [T,4]
    joint_pos: np.ndarray  # [T,29] CSV/MuJoCo joint order
    joint_vel: np.ndarray  # [T,29]
    anchor_pos: np.ndarray = field(default=None)  # [T,3] torso_link world (ghost FK)
    anchor_quat_wxyz: np.ndarray = field(default=None)  # [T,4]

    @property
    def ticks(self) -> int:
        return len(self.joint_pos)


def load_motion(csv_path: Path, model: mujoco.MjModel, anchor_body: str) -> Motion:
    m = np.loadtxt(csv_path, delimiter=",")
    if m.shape[1] != 36:
        raise ValueError(f"{csv_path}: expected 36 cols (7 root + 29 joints), got {m.shape[1]}")
    src_t = np.arange(len(m)) / CSV_FPS
    dst_t = np.arange(0, src_t[-1], 1.0 / CTRL_HZ)

    def interp(cols: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(dst_t, src_t, cols[:, i]) for i in range(cols.shape[1])], axis=1)

    root_pos = interp(m[:, 0:3])
    quat_xyzw = interp(m[:, 3:7])  # nlerp; fine at 30->50 fps for dance data
    quat_xyzw /= np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    jp = interp(m[:, 7:36])
    jv = np.gradient(jp, 1.0 / CTRL_HZ, axis=0)

    # ghost forward kinematics for the anchor (torso) reference trajectory
    ghost = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_body)
    if body_id < 0:
        raise ValueError(f"anchor body {anchor_body!r} not in model")
    a_pos = np.empty((len(dst_t), 3))
    a_quat = np.empty((len(dst_t), 4))
    for i in range(len(dst_t)):
        ghost.qpos[0:3] = root_pos[i]
        ghost.qpos[3:7] = quat_wxyz[i]
        ghost.qpos[7:] = jp[i]
        mujoco.mj_kinematics(model, ghost)
        a_pos[i] = ghost.xpos[body_id]
        a_quat[i] = ghost.xquat[body_id]
    return Motion(root_pos, quat_wxyz, jp, jv, a_pos, a_quat)


# --------------------------------------------------------------------------- adapters
class PolicyAdapter(ABC):
    """Everything the exam needs from a policy, sim-agnostic.

    joint_names are in the POLICY's own order; the exam maps to MuJoCo by name.
    """

    joint_names: list[str]
    kp: np.ndarray
    kd: np.ndarray
    default_pos: np.ndarray
    action_scale: np.ndarray
    obs_terms: list[tuple[str, int]]  # (term_name, history_length)

    @abstractmethod
    def act(self, obs: np.ndarray, tick: int) -> np.ndarray:
        """obs [D] -> action [29] (raw, policy joint order)."""

    def reset(self) -> None:  # pragma: no cover - default no-op
        pass


class WbtOnnxPolicy(PolicyAdapter):
    """whole_body_tracking exporter format (see module docstring)."""

    def __init__(self, path: Path):
        import onnxruntime as ort  # lazy: heavy import

        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        if "joint_names" not in meta:
            raise ValueError("onnx lacks whole_body_tracking metadata; wrong export or adapter")
        self.joint_names = meta["joint_names"].split(",")
        n = len(self.joint_names)
        for k, attr in [
            ("joint_stiffness", "kp"),
            ("joint_damping", "kd"),
            ("default_joint_pos", "default_pos"),
            ("action_scale", "action_scale"),
        ]:
            v = np.array([float(x) for x in meta[k].split(",")], dtype=np.float64)
            if len(v) != n:
                raise ValueError(f"metadata {k}: {len(v)} values for {n} joints")
            setattr(self, attr, v)
        names = meta["observation_names"].split(",")
        hist = [int(x) for x in meta["observation_history_lengths"].split(",")]
        self.obs_terms = list(zip(names, hist))
        self.anchor_body_name = meta.get("anchor_body_name", "torso_link")

    def act(self, obs: np.ndarray, tick: int) -> np.ndarray:
        out = self.sess.run(
            ["actions"],
            {"obs": obs[None].astype(np.float32), "time_step": np.array([[float(tick)]], np.float32)},
        )
        return out[0][0].astype(np.float64)


class StubPolicy(PolicyAdapter):
    """Harness-verification policy: replays the reference with optional noise.

    noise=0 approximates a perfect tracker (still fails realistically — pure PD
    replay is not balance-aware); noise>0 guarantees a fail. Exists so the exam
    machinery can be exercised before the first real export lands.
    """

    # armature-derived analytic gains, whole_body_tracking robots/g1.py
    _ARMATURE = {
        "hip_pitch": 0.010177520, "hip_roll": 0.025101925, "hip_yaw": 0.010177520,
        "knee": 0.025101925, "ankle": 2 * 0.003609725, "waist_yaw": 0.010177520,
        "waist": 2 * 0.003609725, "shoulder": 0.003609725, "elbow": 0.003609725,
        "wrist_roll": 0.003609725, "wrist": 0.00425,
    }

    def __init__(self, model: mujoco.MjModel, motion: Motion, noise: float = 0.0, seed: int = 0):
        w0 = 2 * np.pi * 10.0
        self.joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(1, model.njnt)  # 0 is the free joint
        ]
        arm = np.array([self._armature_for(n) for n in self.joint_names])
        self.kp = arm * w0**2
        self.kd = 2.0 * 2.0 * arm * w0
        self.default_pos = motion.joint_pos[0].copy()
        self.action_scale = np.ones(len(self.joint_names)) * 0.25
        self.obs_terms = [("command", 1), ("joint_pos", 1), ("joint_vel", 1), ("actions", 1)]
        self.anchor_body_name = "torso_link"
        self._motion = motion
        self._rng = np.random.default_rng(seed)
        self._noise = noise

    def _armature_for(self, joint_name: str) -> float:
        for key, a in self._ARMATURE.items():
            if key in joint_name:
                return a
        return 0.003609725

    def act(self, obs: np.ndarray, tick: int) -> np.ndarray:
        i = min(tick, self._motion.ticks - 1)
        act = (self._motion.joint_pos[i] - self.default_pos) / self.action_scale
        if self._noise:
            act = act + self._rng.normal(0, self._noise, act.shape)
        return act


def load_policy(spec: str, model: mujoco.MjModel, motion: Motion, seed: int = 0) -> PolicyAdapter:
    if spec == "stub":
        return StubPolicy(model, motion, noise=0.0, seed=seed)
    if spec == "stub-noisy":
        return StubPolicy(model, motion, noise=2.0, seed=seed)
    return WbtOnnxPolicy(Path(spec))


# --------------------------------------------------------------------------- exam env
def _quat_geodesic(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    d = abs(float(np.dot(q1_wxyz, q2_wxyz)))
    return 2.0 * np.arccos(min(1.0, d))


def _mat_first_two_cols_b(q_ref_wxyz, q_rob_wxyz) -> np.ndarray:
    """Rotation of ref frame expressed in robot frame; first two columns, flattened."""
    r_ref = np.zeros(9)
    r_rob = np.zeros(9)
    mujoco.mju_quat2Mat(r_ref, q_ref_wxyz)
    mujoco.mju_quat2Mat(r_rob, q_rob_wxyz)
    rel = r_rob.reshape(3, 3).T @ r_ref.reshape(3, 3)
    return rel[:, :2].reshape(-1)


class ExamEnv:
    def __init__(self, model: mujoco.MjModel, policy: PolicyAdapter, motion: Motion):
        self.model = model
        self.data = mujoco.MjData(model)
        self.policy = policy
        self.motion = motion
        self.model.opt.timestep = SIM_DT

        # name maps: policy joint order <-> mujoco qpos/dof/actuator indices
        self.n = len(policy.joint_names)
        self.qadr = np.empty(self.n, dtype=int)
        self.vadr = np.empty(self.n, dtype=int)
        self.aadr = np.empty(self.n, dtype=int)
        for i, name in enumerate(policy.joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"policy joint {name!r} not in MuJoCo model")
            self.qadr[i] = model.jnt_qposadr[jid]
            self.vadr[i] = model.jnt_dofadr[jid]
            act_name = name.removesuffix("_joint")
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if aid < 0:
                raise ValueError(f"actuator {act_name!r} not in model")
            self.aadr[i] = aid
        # CSV order == mujoco joint-definition order (playback_csv.py invariant)
        mj_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
        self.csv_to_policy = np.array([mj_names.index(nm) for nm in policy.joint_names])
        self.anchor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, getattr(policy, "anchor_body_name", "torso_link")
        )
        self.torso_id = self.anchor_id
        self.ctrl_lo = model.actuator_ctrlrange[:, 0]
        self.ctrl_hi = model.actuator_ctrlrange[:, 1]

    # ---- state helpers (policy joint order) ----
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.qadr]

    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.vadr]

    def reset(self, seed: int = 0, jitter: float = 0.0) -> None:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = self.motion.root_pos[0]
        self.data.qpos[3:7] = self.motion.root_quat_wxyz[0]
        self.data.qpos[7:] = self.motion.joint_pos[0]
        if jitter:
            self.data.qpos[7:] += rng.uniform(-jitter, jitter, self.data.qpos[7:].shape)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.last_action = np.zeros(self.n)
        self.policy.reset()
        self._obs_hist: dict[str, list[np.ndarray]] = {}

    # ---- observation ----
    def _term_value(self, term: str, tick: int) -> np.ndarray:
        i = min(tick, self.motion.ticks - 1)
        mo = self.motion
        if term == "command":
            jp = mo.joint_pos[i][self.csv_to_policy]
            jv = mo.joint_vel[i][self.csv_to_policy]
            return np.concatenate([jp, jv])
        if term == "motion_anchor_pos_b":
            r = np.zeros(9)
            mujoco.mju_quat2Mat(r, self.data.xquat[self.anchor_id])
            return r.reshape(3, 3).T @ (mo.anchor_pos[i] - self.data.xpos[self.anchor_id])
        if term == "motion_anchor_ori_b":
            return _mat_first_two_cols_b(mo.anchor_quat_wxyz[i], self.data.xquat[self.anchor_id])
        if term == "base_lin_vel":
            r = np.zeros(9)
            mujoco.mju_quat2Mat(r, self.data.qpos[3:7])
            return r.reshape(3, 3).T @ self.data.qvel[0:3]
        if term == "base_ang_vel":
            return self.data.qvel[3:6].copy()  # free joint ang vel is body-frame
        if term == "joint_pos":
            return self.joint_pos() - self.policy.default_pos
        if term == "joint_vel":
            return self.joint_vel().copy()
        if term == "actions":
            return self.last_action.copy()
        raise ValueError(f"unsupported observation term {term!r} — extend ExamEnv._term_value")

    def obs(self, tick: int) -> np.ndarray:
        parts = []
        for term, hist in self.policy.obs_terms:
            v = self._term_value(term, tick)
            buf = self._obs_hist.setdefault(term, [v] * max(1, hist))
            buf.append(v)
            del buf[:-max(1, hist)]
            parts.append(np.concatenate(buf) if hist > 1 else buf[-1])
        return np.concatenate(parts)

    # ---- stepping ----
    def step(self, tick: int, push: np.ndarray | None = None) -> None:
        action = self.policy.act(self.obs(tick), tick)
        self.last_action = action
        q_des = self.policy.default_pos + self.policy.action_scale * action
        for _ in range(DECIMATION):
            tau = self.policy.kp * (q_des - self.joint_pos()) - self.policy.kd * self.joint_vel()
            self.data.ctrl[self.aadr] = np.clip(tau, self.ctrl_lo[self.aadr], self.ctrl_hi[self.aadr])
            self.data.xfrc_applied[self.torso_id, :3] = push if push is not None else 0
            mujoco.mj_step(self.model, self.data)

    # ---- per-tick metrics ----
    def metrics(self, tick: int) -> dict:
        i = min(tick, self.motion.ticks - 1)
        anchor_pos_err = float(np.linalg.norm(self.data.xpos[self.anchor_id] - self.motion.anchor_pos[i]))
        anchor_z_err = float(abs(self.data.xpos[self.anchor_id][2] - self.motion.anchor_pos[i][2]))
        ori_err = _quat_geodesic(self.data.xquat[self.anchor_id], self.motion.anchor_quat_wxyz[i])
        joint_err = float(np.mean(np.abs(self.joint_pos() - self.motion.joint_pos[i][self.csv_to_policy])))
        excursion = float(np.linalg.norm(self.data.qpos[0:2] - self.motion.root_pos[0][:2]))
        return {
            "anchor_pos_err": anchor_pos_err,
            "anchor_z_err": anchor_z_err,
            "anchor_ori_err": ori_err,
            "joint_err": joint_err,
            "excursion": excursion,
            "failed": anchor_z_err > ANCHOR_Z_FAIL_M or ori_err > ANCHOR_ORI_FAIL,
        }


# --------------------------------------------------------------------------- phases
def run_nominal(env: ExamEnv, seed: int = 0, jitter: float = 0.0, frames_out: list | None = None,
                renderer: "mujoco.Renderer | None" = None) -> dict:
    env.reset(seed=seed, jitter=jitter)
    T = env.motion.ticks
    errs, joint_errs, max_exc = [], [], 0.0
    survived = T
    for t in range(T):
        env.step(t)
        m = env.metrics(t)
        errs.append(m["anchor_pos_err"])
        joint_errs.append(m["joint_err"])
        max_exc = max(max_exc, m["excursion"])
        if renderer is not None and frames_out is not None and t % 2 == 0:  # 25 fps video
            renderer.update_scene(env.data)
            frames_out.append(renderer.render())
        if m["failed"]:
            survived = t
            break
    dur = T / CTRL_HZ
    return {
        "pass": survived == T and max_exc <= EXCURSION_LIMIT_M,
        "survived_s": round(survived / CTRL_HZ, 2),
        "duration_s": round(dur, 2),
        "excursion_m": round(max_exc, 3),
        "mean_anchor_pos_err_m": round(float(np.mean(errs)), 4),
        "max_anchor_pos_err_m": round(float(np.max(errs)), 4),
        "mean_joint_err_rad": round(float(np.mean(joint_errs)), 4),
    }


def run_push_test(env: ExamEnv, num_pushes: int, force_n: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    T = env.motion.ticks
    push_ticks = sorted(rng.choice(np.arange(int(2 * CTRL_HZ), max(T - int(3 * CTRL_HZ), 3 * int(CTRL_HZ)), 1),
                                   size=min(num_pushes, max(1, T // int(4 * CTRL_HZ))), replace=False))
    dur_ticks = int(PUSH_DURATION_S * CTRL_HZ)
    recover_ticks = int(PUSH_RECOVERY_S * CTRL_HZ)
    recovered, applied, active_until, fell = 0, 0, -1, False
    force_vec = np.zeros(3)
    pending: int | None = None
    for t in range(T):
        if push_ticks and t == push_ticks[0]:
            theta = rng.uniform(0, 2 * np.pi)
            force_vec = np.array([np.cos(theta), np.sin(theta), 0.0]) * force_n
            active_until = t + dur_ticks
            pending = t + dur_ticks + recover_ticks
            push_ticks.pop(0)
            applied += 1
        push = force_vec if t < active_until else None
        env.step(t, push=push)
        m = env.metrics(t)
        if m["failed"]:
            fell = True
            break
        if pending is not None and t >= pending:
            if m["anchor_pos_err"] < PUSH_RECOVER_ERR_M:
                recovered += 1
            pending = None
    rate = recovered / applied if applied else 0.0
    return {
        "num_pushes": applied,
        "recovered": recovered,
        "recovery_rate": round(rate, 3),
        "force_n": force_n,
        "fell": fell,
        "pass": not fell and rate >= 0.8,
    }


def run_repeatability(env: ExamEnv, runs: int) -> dict:
    clean, consecutive, best = 0, 0, 0
    per_run = []
    for k in range(runs):
        r = run_nominal(env, seed=100 + k, jitter=0.02)
        per_run.append({"seed": 100 + k, "pass": r["pass"], "survived_s": r["survived_s"]})
        if r["pass"]:
            clean += 1
            consecutive += 1
            best = max(best, consecutive)
        else:
            consecutive = 0
    return {"runs": runs, "clean": clean, "consecutive_clean": best, "pass": clean == runs, "per_run": per_run}


# --------------------------------------------------------------------------- main
def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", required=True, help="path to policy.onnx, or 'stub' / 'stub-noisy'")
    ap.add_argument("--motion", required=True, help="30fps G1 motion CSV (vet-passed)")
    ap.add_argument("--dance", default=None, help="dance name for the report")
    ap.add_argument("--runs", type=int, default=5, help="repeatability runs")
    ap.add_argument("--pushes", type=int, default=4)
    ap.add_argument("--force", type=float, default=250.0, help="push force (N)")
    ap.add_argument("--video", default=None, help="output MP4 (EGL) of the nominal run")
    ap.add_argument("--json", dest="json_out", default=None, help="verdict JSON path")
    ap.add_argument("--skip-push", action="store_true")
    ap.add_argument("--skip-repeat", action="store_true")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    motion_path = Path(args.motion)
    # anchor known only after policy load for onnx; stub uses torso_link — load in 2 steps
    tmp_motion = load_motion(motion_path, model, "torso_link")
    policy = load_policy(args.policy, model, tmp_motion)
    anchor = getattr(policy, "anchor_body_name", "torso_link")
    motion = tmp_motion if anchor == "torso_link" else load_motion(motion_path, model, anchor)
    env = ExamEnv(model, policy, motion)

    print(f"[sim_exam] {args.policy} vs {motion_path.name}: {motion.ticks} ticks "
          f"({motion.ticks / CTRL_HZ:.1f}s) @ {CTRL_HZ:.0f}Hz, anchor={anchor}")

    frames: list | None = None
    renderer = None
    if args.video:
        renderer = mujoco.Renderer(model, height=480, width=640)
        frames = []

    t0 = time.time()
    nominal = run_nominal(env, frames_out=frames, renderer=renderer)
    print(f"[sim_exam] nominal: {'PASS' if nominal['pass'] else 'FAIL'} {nominal}")
    push = None if args.skip_push else run_push_test(env, args.pushes, args.force)
    if push:
        print(f"[sim_exam] push: {'PASS' if push['pass'] else 'FAIL'} {push}")
    repeat = None if args.skip_repeat else run_repeatability(env, args.runs)
    if repeat:
        print(f"[sim_exam] repeatability: {'PASS' if repeat['pass'] else 'FAIL'} "
              f"{repeat['clean']}/{repeat['runs']} clean")

    video_path = None
    if args.video and frames:
        import imageio.v2 as imageio

        video_path = str(args.video)
        imageio.mimsave(video_path, frames, fps=25, macro_block_size=1)
        print(f"[sim_exam] video: {video_path} ({len(frames)} frames)")

    verdict = {
        "schema": "sim_exam/v1",
        "dance": args.dance or motion_path.stem,
        "policy": str(args.policy),
        "policy_sha256": sha256(Path(args.policy)) if Path(args.policy).exists() else None,
        "motion_csv": str(motion_path),
        "motion_sha256": sha256(motion_path),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "control_hz": CTRL_HZ,
        "nominal": nominal,
        "push": push,
        "repeatability": repeat,
        "verdict": "pass" if nominal["pass"] and (push is None or push["pass"]) and (repeat is None or repeat["pass"]) else "fail",
        "video": video_path,
        "wall_s": round(time.time() - t0, 1),
    }
    out = Path(args.json_out) if args.json_out else PROJECT_ROOT / "data" / "exports" / f"exam_{verdict['dance']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=1))
    print(f"[sim_exam] VERDICT: {verdict['verdict'].upper()} -> {out}")
    return 0 if verdict["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
