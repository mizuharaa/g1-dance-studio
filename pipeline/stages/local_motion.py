"""Stage implementations that run on this laptop.

CSV motion inputs (LAFAN1-convention robot motion) run fully locally:
    retarget = window -> vet gate -> MuJoCo preview -> show prep -> deploy ramp.
Video inputs run GMR retargeting here (from the GVHMR output the cloud extract
stage pulled), then the same window/vet/preview/prep flow. The cloud-backed
extract/train/verify/export stages live in stages/cloud_motion.py; when the
GPU box is not configured they raise StageBlocked so the UI shows an honest
"waiting on cloud" state instead of fake progress.
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
from .base import Reporter, StageBlocked

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


class RetargetStage:
    """CSV path: window -> vet -> preview -> prep. Video path: GMR retarget of
    the GVHMR output first, then the same flow. Ends with the show CSV and the
    DEPLOYABLE <slug>_deploy.csv (2.5 s activation ramp) the train stage pushes."""

    name = "retarget"

    def run(self, job: Job, report: Reporter) -> None:
        from .cloud_motion import _slug, load_params
        params = load_params(job)
        out_dir = job.stage_dir(self.name)
        st = job.stages[self.name]

        csv = _input_csv(job)
        if csv is None:
            pred = job.dir / "extract" / "hmr4d_results.pt"
            if not pred.exists():
                raise StageBlocked("needs the GVHMR extraction output "
                                   "(extract stage incomplete)")
            csv = out_dir / "raw_g1.csv"
            if not csv.exists():
                report(0.02, "retargeting human motion to the G1 (GMR)")

                def on_retarget_line(line: str) -> None:
                    if line.startswith("retarget "):
                        try:
                            i, total = line.split()[1].split("/")
                            report(0.02 + 0.15 * int(i) / max(1, int(total)),
                                   f"GMR retarget {i}/{total}")
                        except (ValueError, IndexError):
                            pass

                args = ["--pred", str(pred), "--out", str(csv)]
                if params["velocity_limit"]:
                    args.append("--velocity-limit")
                r = _run_tool("retarget_gvhmr.py", args, job,
                              on_line=on_retarget_line)
                if r.returncode != 0 or not csv.exists():
                    raise RuntimeError(f"GMR retarget failed: {r.stderr[-500:]}")
                job.log("retarget: GMR retarget done (30 fps, 29 DoF, "
                        f"velocity_limit={params['velocity_limit']})")

        report(0.2, "finding deployable window")
        from ..motion_io import load_motion_csv
        m = load_motion_csv(csv)  # clear error on a malformed CSV, not a traceback
        # Ground-reference before window/vet so the absolute-z gate is meaningful
        # AND the support foot doesn't float (§3.3 'floaty feet' source defect).
        # Per-frame grounding removes the retarget's slow vertical drift so the
        # planted foot sits on z≈0 every frame (a single global offset — the old
        # behaviour — plants only the one lowest instant and left the foot
        # floating >0.10 m in ~78% of the Thriller). Relative heights (root-above-
        # foot) are preserved exactly; idempotent on an already-grounded motion.
        from ..grounding import ground_motion_per_frame, have_model
        if have_model():
            m, ginfo = ground_motion_per_frame(m)
            if ginfo["drift_removed_mm"] > 5.0 or ginfo["mean_shift_m"] > 0.01:
                job.log(f"retarget: per-frame grounded motion (removed "
                        f"{ginfo['drift_removed_mm']:.0f} mm of vertical drift, "
                        f"mean shift {ginfo['mean_shift_m']:+.3f} m"
                        + (f", {ginfo['flight_frames']} flight frames held"
                           if ginfo['flight_frames'] else "") + ")")
        if params["window_start_s"] is not None and params["window_end_s"] is not None:
            # explicit per-dance window (dance.yaml) — still subject to the vet gate
            s = max(0, int(round(float(params["window_start_s"]) * CSV_FPS)))
            e = min(len(m) - 1,
                    int(round(float(params["window_end_s"]) * CSV_FPS)) - 1)
            job.log(f"retarget: window override from dance params: frames {s}..{e}")
        else:
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

        # Show prep + deployable motion (the exact Thriller recipe): residual
        # velocity clamp + FK ground fix + standing pad/blends (prep_motion),
        # then the 2.5 s standby->frame-0 activation ramp (deploy_ramp). The
        # train stage pushes ONLY the _deploy.csv — it is the artifact every
        # downstream consumer (training, exams, dance record, bundle) binds to.
        report(0.96, "prepping show + deployable motion")
        from ..deploy_ramp import make_deploy_csv
        from ..prep_motion import prep
        slug = _slug(job.name)
        show_csv = out_dir / f"{slug}_show.csv"
        deploy_csv = out_dir / f"{slug}_deploy.csv"
        st.meta["prep"] = prep(motion_csv, show_csv)
        st.meta["ramp"] = make_deploy_csv(show_csv, deploy_csv)
        st.meta["show_csv"] = str(show_csv)
        st.meta["deploy_csv"] = str(deploy_csv)

        # Post-clean HARD backstop (cost guard): the vet gate above runs on the
        # RAW motion and only WARNS on glitch (prep de-glitches downstream). If
        # clean_motion STILL couldn't get this motion under the severe floor,
        # refuse now — before the human ever sees an "approve training" button —
        # rather than pay for a 5 h GPU run on unfixable data. Safe on the
        # cleaned Thriller (7.6k jerk); only a broken source trips it.
        from ..vet_motion import severe_after_clean
        severe = severe_after_clean(str(show_csv))
        st.meta["severe_after_clean"] = severe
        if severe:
            raise RuntimeError(
                "motion still severely glitchy/infeasible AFTER de-glitch "
                f"({', '.join(severe)}) — fix the source clip or retarget; "
                "refusing to spend GPU on it")
        job.log(f"retarget: deployable motion ready — {deploy_csv.name} "
                f"({st.meta['ramp']['seconds']}s incl. 2.5s activation ramp)")
        report(1.0, "motion ready — review the preview, then approve training")


def build_stages() -> dict:
    from .cloud_motion import ExportStage, ExtractStage, TrainStage, VerifyStage
    stages = [ExtractStage(), RetargetStage(), TrainStage(), VerifyStage(),
              ExportStage()]
    return {s.name: s for s in stages}
