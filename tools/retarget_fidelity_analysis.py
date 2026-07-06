#!/usr/bin/env python
"""Retarget-fidelity analysis for a dance reference chain (docs/retarget_fidelity.md).

Quantifies how much motion "sharpness" (joint-velocity peaks, high-frequency
spectral energy) each front-end stage removes before the policy ever sees the
reference:

    raw GMR retarget CSV  ->  prep_motion clamp+blends (show CSV)  ->  50 fps npz

Outputs a JSON report (data/reports/retarget_fidelity.json by default) with:
  * global + per-section velocity/acceleration stats per stage
  * every clamp "event" in the raw motion (|v| over the blanket limit), each
    classified as GLITCH (single-frame pose spike that reverses immediately —
    estimator noise, good to remove) or MOVE (sustained directional motion —
    a real dance accent that the clamp blunted)
  * velocity spectra (band energies) per stage and section

Timeline convention: the raw CSV is the un-padded dance; the show CSV prepends
pad+blend (default 45 frames) before the same 1329 frames; the deploy npz
prepends a further activation ramp (default 75 frames @30fps) and runs at 50 fps.

Usage:
  python tools/retarget_fidelity_analysis.py \
      --raw data/motions/thriller/thriller_g1.csv \
      --show data/motions/thriller/thriller_show.csv \
      --npz data/policies/thriller/thriller_deploy.npz \
      --show-offset-frames 45 --npz-extra-frames 75 \
      --sections "13-17,25-36,40-49" \
      --out data/reports/retarget_fidelity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FPS = 30.0
BLANKET_LIMIT = 3 * np.pi          # rad/s — GMR/vet "motor class" advisory limit
CLAMP_LIMIT = 0.9 * 3 * np.pi      # rad/s — prep_motion actual clamp

# Joint order for the 36-col CSV convention (cols 7:36), menagerie G1 29 DoF.
JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

# True per-motor velocity limits (rad/s) — BeyondMimic/wbt G1 actuator config
# (matches Unitree motor classes: 7520-14=32, 7520-22=20, 5020=37, 4010=22).
TRUE_VEL_LIMITS = {
    "hip_pitch": 32.0, "hip_roll": 20.0, "hip_yaw": 32.0, "knee": 20.0,
    "ankle_pitch": 37.0, "ankle_roll": 37.0,
    "waist_yaw": 32.0, "waist_roll": 37.0, "waist_pitch": 37.0,
    "shoulder_pitch": 37.0, "shoulder_roll": 37.0, "shoulder_yaw": 37.0,
    "elbow": 37.0, "wrist_roll": 37.0, "wrist_pitch": 22.0, "wrist_yaw": 22.0,
}

GROUPS = {
    "legs": [i for i, n in enumerate(JOINT_NAMES)
             if any(k in n for k in ("hip", "knee", "ankle"))],
    "waist": [i for i, n in enumerate(JOINT_NAMES) if "waist" in n],
    "arms": [i for i, n in enumerate(JOINT_NAMES)
             if any(k in n for k in ("shoulder", "elbow", "wrist"))],
}


def true_limit(joint_name: str) -> float:
    for key, lim in TRUE_VEL_LIMITS.items():
        if key in joint_name:
            return lim
    raise KeyError(joint_name)


def vel(dof: np.ndarray, fps: float = FPS) -> np.ndarray:
    return np.diff(dof, axis=0) * fps


def acc(dof: np.ndarray, fps: float = FPS) -> np.ndarray:
    return np.diff(dof, n=2, axis=0) * fps * fps


def band_energy(v: np.ndarray, fps: float, bands=((0, 2), (2, 5), (5, 15))) -> dict:
    """Mean per-joint velocity-spectrum energy in Hz bands (rfft power)."""
    n = v.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(np.fft.rfft(v, axis=0)) ** 2 / n
    out = {}
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        out[f"{lo}-{hi}Hz"] = float(power[m].sum(axis=0).mean())
    return out


def stage_stats(v: np.ndarray, a: np.ndarray | None = None) -> dict:
    av = np.abs(v)
    st = {
        "vel_p99": round(float(np.percentile(av, 99)), 3),
        "vel_max": round(float(av.max()), 3),
        "pct_frames_over_blanket": round(
            100 * float((av > BLANKET_LIMIT).any(axis=1).mean()), 2),
    }
    for g, idx in GROUPS.items():
        st[f"vel_max_{g}"] = round(float(av[:, idx].max()), 3)
        st[f"vel_p99_{g}"] = round(float(np.percentile(av[:, idx], 99)), 3)
    if a is not None:
        st["acc_p99"] = round(float(np.percentile(np.abs(a), 99)), 1)
        st["acc_max"] = round(float(np.abs(a).max()), 1)
    return st


def classify_event(dof: np.ndarray, frame: int, joint: int) -> str:
    """GLITCH = the over-limit step largely reverses within 2 frames
    (transient pose spike). MOVE = net displacement is kept (real motion)."""
    step = dof[frame + 1, joint] - dof[frame, joint]
    end = min(frame + 3, len(dof) - 1)
    net = dof[end, joint] - dof[frame, joint]
    return "GLITCH" if abs(net) < 0.5 * abs(step) else "MOVE"


def find_events(dof: np.ndarray, limit: float = BLANKET_LIMIT,
                gap: int = 3) -> list[dict]:
    """Group over-limit (frame, joint) cells into time-contiguous events."""
    v = vel(dof)
    over = np.argwhere(np.abs(v) > limit)          # (frame, joint)
    events: list[dict] = []
    for f, j in over:
        placed = False
        for ev in events:
            if ev["joint_idx"] == int(j) and f - ev["frames"][-1] <= gap:
                ev["frames"].append(int(f))
                placed = True
                break
        if not placed:
            events.append({"joint_idx": int(j), "frames": [int(f)]})
    for ev in events:
        j = ev["joint_idx"]
        f0 = ev["frames"][int(np.argmax([abs(v[f, j]) for f in ev["frames"]]))]
        ev["joint"] = JOINT_NAMES[j]
        ev["t_video_s"] = round(f0 / FPS, 2)
        ev["peak_vel"] = round(float(abs(v[f0, j])), 2)
        ev["peak_frame"] = f0
        ev["kind"] = classify_event(dof, f0, j)
        ev["true_motor_limit"] = true_limit(JOINT_NAMES[j])
        ev["frames"] = [int(f) for f in ev["frames"]]
    return sorted(events, key=lambda e: e["t_video_s"])


def section_slice(n: int, t0: float, t1: float, offset_frames: int = 0,
                  fps: float = FPS) -> slice:
    a = max(0, int(round(t0 * fps)) - offset_frames)
    b = min(n, int(round(t1 * fps)) - offset_frames)
    return slice(a, max(a, b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--show", required=True, type=Path)
    ap.add_argument("--npz", type=Path)
    ap.add_argument("--show-offset-frames", type=int, default=45,
                    help="frames of pad+blend the show CSV prepends to the dance")
    ap.add_argument("--npz-extra-frames", type=int, default=75,
                    help="extra 30fps frames (activation ramp) before the show cut in the npz")
    ap.add_argument("--sections", default="13-17,25-36,40-49",
                    help="show-timeline sections 't0-t1,...' (seconds)")
    ap.add_argument("--out", type=Path,
                    default=Path("data/reports/retarget_fidelity.json"))
    args = ap.parse_args()

    raw = np.loadtxt(args.raw, delimiter=",")[:, 7:]
    show_full = np.loadtxt(args.show, delimiter=",")[:, 7:]
    off = args.show_offset_frames
    show = show_full[off:off + len(raw)]
    assert len(show) == len(raw), "show CSV does not contain the full raw dance"

    report: dict = {
        "inputs": {"raw": str(args.raw), "show": str(args.show),
                   "npz": str(args.npz) if args.npz else None},
        "limits": {"blanket_rad_s": round(BLANKET_LIMIT, 3),
                   "prep_clamp_rad_s": round(CLAMP_LIMIT, 3),
                   "true_motor_limits": TRUE_VEL_LIMITS},
        "global": {
            "raw": stage_stats(vel(raw), acc(raw)),
            "show": stage_stats(vel(show), acc(show)),
        },
        "clamp_footprint": {
            "frames_modified": int((np.abs(show - raw).max(axis=1) > 1e-9).sum()),
            "frames_total": int(len(raw)),
            "max_pose_deviation_rad": round(float(np.abs(show - raw).max()), 3),
            "max_pose_deviation_deg": round(
                float(np.degrees(np.abs(show - raw).max())), 1),
        },
    }

    if args.npz:
        z = np.load(args.npz)
        zfps = float(z["fps"][0]) if "fps" in z else 50.0
        jp = z["joint_pos"]
        z0 = int(round((args.npz_extra_frames + off) / FPS * zfps))
        z1 = int(round((args.npz_extra_frames + off + len(raw)) / FPS * zfps))
        npz_dance = jp[z0:z1]
        report["global"]["npz"] = stage_stats(
            vel(npz_dance, zfps), acc(npz_dance, zfps))
        if "joint_vel" in z:
            report["global"]["npz_stored_joint_vel"] = stage_stats(
                np.asarray(z["joint_vel"][z0:z1]))

    # ---- events (in the raw retarget)
    events = find_events(raw)
    v_show = vel(show)
    for ev in events:
        f, j = ev["peak_frame"], ev["joint_idx"]
        lo, hi = max(0, f - 2), min(len(v_show), f + 3)
        ev["show_peak_vel"] = round(float(np.abs(v_show[lo:hi, j]).max()), 2)
        ev["attenuation_pct"] = round(
            100 * (1 - ev["show_peak_vel"] / ev["peak_vel"]), 1)
        ev["t_show_s"] = round(ev["t_video_s"] + off / FPS, 2)
    report["events"] = events
    report["events_summary"] = {
        "n_events": len(events),
        "n_glitch": sum(e["kind"] == "GLITCH" for e in events),
        "n_move": sum(e["kind"] == "MOVE" for e in events),
    }

    # ---- per-section stats (sections given in SHOW timeline)
    sections = {}
    for spec in args.sections.split(","):
        t0, t1 = (float(x) for x in spec.split("-"))
        sl = section_slice(len(raw), t0, t1, offset_frames=off)
        if sl.stop - sl.start < 10:
            continue
        r, s = raw[sl], show[sl]
        sec = {
            "video_time_s": [round(sl.start / FPS, 2), round(sl.stop / FPS, 2)],
            "raw": stage_stats(vel(r), acc(r)),
            "show": stage_stats(vel(s), acc(s)),
            "spectra_raw": band_energy(vel(r), FPS),
            "spectra_show": band_energy(vel(s), FPS),
            "events": [e["t_show_s"] for e in events
                       if t0 <= e["t_show_s"] < t1],
        }
        hf_r = sec["spectra_raw"]["5-15Hz"]
        hf_s = sec["spectra_show"]["5-15Hz"]
        sec["hf_energy_kept_pct"] = round(100 * hf_s / hf_r, 1) if hf_r else None
        sections[spec] = sec
    report["sections"] = sections

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("global", "clamp_footprint", "events_summary")}, indent=2))
    print(f"full report -> {args.out}")


if __name__ == "__main__":
    main()
