"""Stage implementations that run on this laptop.

CSV motion inputs (LAFAN1-convention robot motion) run fully locally:
    retarget = window -> vet gate -> MuJoCo preview render.
Video inputs need the cloud GPU (GreenNode) for extract/retarget, and every
job needs it for train — until it is provisioned those stages raise
StageBlocked so the UI shows an honest "waiting on cloud" state instead of
fake progress. A re-queue (app restart or the Retry button) retries them.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..config import DATA_DIR, PROJECT_ROOT
from ..find_window import CSV_FPS, longest_window, window_center
from ..store import Job
from .base import Reporter, SkipStage, StageBlocked

CLOUD_MSG = "waiting on cloud GPU (GreenNode not provisioned yet)"
PREVIEWS_DIR = DATA_DIR / "previews"


def _input_csv(job: Job) -> Path | None:
    p = job.dir / "input.csv"
    return p if p.exists() else None


def _run_tool(script: str, args: list[str], job: Job, on_line=None,
              env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run a pipeline/*.py tool as a subprocess, streaming stdout to the job log."""
    env = dict(os.environ, **(env_extra or {}))
    proc = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "pipeline" / script), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        out_lines.append(line)
        if on_line:
            on_line(line.rstrip())
    _, err = proc.communicate()
    return subprocess.CompletedProcess(proc.args, proc.returncode,
                                       "".join(out_lines), err)


class ExtractStage:
    """Local part: intake validation via ffprobe (fail fast on unusable
    footage). The GPU pose extraction itself is cloud-blocked until
    GreenNode is provisioned."""

    name = "extract"

    def run(self, job: Job, report: Reporter) -> None:
        if _input_csv(job):
            raise SkipStage("input is already robot motion (CSV) — no video to extract")
        st = job.stages[self.name]
        if "video" not in st.meta:
            from ..video_probe import validate
            report(0.05, "checking video file")
            st.meta["video"] = validate(job.dir / "input.mp4")  # raises w/ reason
            for adv in st.meta["video"]["advisories"]:
                job.log(f"extract: ADVISORY — {adv}")
            job.log(f"extract: video ok — {st.meta['video']['duration_s']}s "
                    f"{st.meta['video']['width']}x{st.meta['video']['height']} "
                    f"@ {st.meta['video']['fps']}fps")
            report(0.1, "video valid — waiting on cloud for pose extraction")
        raise StageBlocked(CLOUD_MSG)


class RetargetStage:
    """CSV path: window -> vet -> preview. Video path: cloud (blocked)."""

    name = "retarget"

    def run(self, job: Job, report: Reporter) -> None:
        csv = _input_csv(job)
        if csv is None:
            raise StageBlocked(CLOUD_MSG)
        out_dir = job.stage_dir(self.name)
        st = job.stages[self.name]

        report(0.05, "finding deployable window")
        from ..motion_io import load_motion_csv
        m = load_motion_csv(csv)  # clear error on a malformed CSV, not a traceback
        # Ground-reference before window/vet so the absolute-z gate is meaningful
        # (audit HIGH: GMR retarget output is not floor-referenced). Idempotent.
        from ..grounding import UNGROUNDED_FLAG_M, ground_motion, have_model
        if have_model():
            m, shift = ground_motion(m)
            if abs(shift) > UNGROUNDED_FLAG_M:
                job.log(f"retarget: grounded motion (input contact was "
                        f"{shift:+.3f} m off the floor)")
        s, e = longest_window(m)
        if e <= s:
            raise RuntimeError("no frame window satisfies the dance-area limits")
        seg = m[s:e + 1].copy()
        # Re-center XY on the window's enclosing-circle centre (NOT frame 0), so the
        # deployed motion's max excursion-from-origin equals the radius the vet gate
        # certifies. Frame-0 recentering could leave the robot drifting up to ~2x the
        # certified footprint and out of the dance area (production audit, HIGH).
        cx, cy = window_center(m, s, e)
        seg[:, 0] -= cx
        seg[:, 1] -= cy
        motion_csv = out_dir / "motion.csv"
        np.savetxt(motion_csv, seg, delimiter=",", fmt="%.6f")
        st.meta["window"] = {
            "start_frame": int(s), "end_frame": int(e),
            "seconds": round(len(seg) / CSV_FPS, 1),
            "input_seconds": round(len(m) / CSV_FPS, 1),
        }
        job.log(f"retarget: window frames {s}..{e} = {len(seg) / CSV_FPS:.1f}s "
                "(XY re-centered)")

        report(0.25, "vetting motion (hard gate)")
        vet = _run_tool("vet_motion.py", [str(motion_csv), "--json"], job)
        if not vet.stdout:
            raise RuntimeError(f"vet_motion produced no report: {vet.stderr[-500:]}")
        vet_report = json.loads(vet.stdout)
        (out_dir / "vet.json").write_text(json.dumps(vet_report, indent=2))
        st.meta["vet_pass"] = vet_report["pass"]
        job.log(f"retarget: vet gate {'PASS' if vet_report['pass'] else 'FAIL'}")
        if not vet_report["pass"]:
            failed = [n for n, c in vet_report["hard"].items() if not c["pass"]]
            raise RuntimeError(f"motion rejected by vet gate: {', '.join(failed)} "
                               "(see vet report)")

        report(0.35, "rendering MuJoCo preview")
        total = len(seg)

        def on_line(line: str) -> None:
            #  playback prints "  frame i/N" every 300 frames
            if line.strip().startswith("frame "):
                try:
                    i = int(line.split()[1].split("/")[0])
                    report(0.35 + 0.6 * i / total, f"rendering preview {i}/{total}")
                except (ValueError, IndexError):
                    pass

        preview = out_dir / "preview.mp4"
        render = _run_tool("playback_csv.py", [str(motion_csv), "--render",
                           str(preview)], job, on_line=on_line,
                           env_extra={"MUJOCO_GL": "egl"})
        if render.returncode != 0 or not preview.exists():
            raise RuntimeError(f"preview render failed: {render.stderr[-500:]}")

        # expose via the previews static mount (StaticFiles follows symlinks)
        PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
        link = PREVIEWS_DIR / f"job-{job.id}.mp4"
        link.unlink(missing_ok=True)
        link.symlink_to(preview)
        st.meta["preview"] = f"/previews/{link.name}"
        job.log(f"retarget: preview rendered ({preview.stat().st_size} bytes)")
        report(1.0, "reference motion ready")


class TrainStage:
    name = "train"

    def run(self, job: Job, report: Reporter) -> None:
        raise StageBlocked(CLOUD_MSG)


class VerifyStage:
    name = "verify"

    def run(self, job: Job, report: Reporter) -> None:
        raise StageBlocked("needs a trained policy first")


class ExportStage:
    name = "export"

    def run(self, job: Job, report: Reporter) -> None:
        raise StageBlocked("needs a verified policy first")


def build_stages() -> dict:
    stages = [ExtractStage(), RetargetStage(), TrainStage(), VerifyStage(),
              ExportStage()]
    return {s.name: s for s in stages}
