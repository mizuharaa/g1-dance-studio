"""One-button live-show runner for the desktop app.

The desktop app's "RUN SHOW" button lands here (via ui/server.py): for a
show-ready dance with music attached, this module launches the PROVEN live path
`tools/show_run.sh` — which drives `pipeline.deploy_runtime --mode
ground-run-legodom` and cues the music — and tracks the single running show so
the app can display its phase/log without a terminal.

Safety posture (CLAUDE.md deploy rule):
  * This module NEVER talks to the robot itself. It only spawns show_run.sh.
  * The runtime's ONLY stop is the operator's hand-held damping remote — there is
    no hardware torque-cut e-stop on this tetherless G1. The API therefore refuses
    to start unless the operator has typed the exact damping-remote confirmation
    phrase; that typed phrase PLUS the operator physically holding the remote IS
    the explicit human confirmation the deploy stage must always require.
  * Exactly ONE run may be active at a time (single-run lock), and a finished
    run's outcome must be recorded (pipeline.shows.record_outcome, via the
    existing /outcome endpoint) before another run can start — an unresolved open
    show blocks the next run.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import artifacts, preshow, shows
from .config import PROJECT_ROOT, ROBOT_PC2_IP

# The proven live path (see the context handover + tools/show_run.sh header).
SHOW_RUN_SH = PROJECT_ROOT / "tools" / "show_run.sh"

# ---- untethered ("free") show config ---------------------------------------------
# HARDWARE-VALIDATED on 2026-07-07: the G1 dances Thriller FULLY UNTETHERED, ends
# standing, music on-beat, repeated 3x clean (see PROJECT_STATE.md 2026-07-07). The
# free config swaps the SHOW policy for the standtail candidate (v3e dance + a
# return-to-standing tail) and adds the sagittal leg-gain boost that fixed the
# arm-accent lean, plus a stand-at-end handoff.
#
# PROVENANCE — READ BEFORE TRUSTING THIS AS "SIGNED": the free config is validated on
# the robot but the standtail motion is NOT yet a SIGNED show-ready artifact (the mjlab
# box must re-exam it). `free` therefore remains an explicitly enabled REHEARSAL
# trial only. Live mode always refuses it; rehearsal additionally requires
# G1_ALLOW_UNSIGNED_FREE=1 and an internally valid standtail bundle manifest.
FREE_POLICY_DIR = "data/policies/thriller_standtail_candidate"   # project-relative
# Side-by-side reference video (Lane B's show_run.sh reads SHOW_VIDEO/SHOW_DISPLAY to
# launch it; here we only set the env contract — the launch itself is Lane B's).
# 2026-07-09: switched from the v3e composite to the CSV-ankle composite. The v3e sim
# panel was a DIFFERENT Thriller take (2589-frame lineage); the deployed dance is now
# thriller_csv_ankle_penalty (2789-frame), so the v3e sim matched neither the robot nor
# the reference (the "sim not in sync" live-run complaint). The new panel is rendered
# from the actual deploy motion (tools/render_deploy_sim.py) so sim == robot; human-vs-
# robot reference alignment stays approximate (src_lead=3.76,speed=0.9 — see make_side_by_side).
FREE_SHOW_VIDEO = "data/previews/thriller_side_by_side_csv.mp4"
RUN_CONFIRMATION_PHRASE = "I AM PRESENT WITH THE DAMPING REMOTE"


class ShowBundleError(ValueError):
    """A recorded show artifact is missing, changed, or incompletely authorized."""

    def __init__(self, member: str, expected=None, actual=None):
        self.member, self.expected, self.actual = member, expected, actual
        if expected is None:
            detail = f"{member}: no promotion-time path/hash was recorded"
        elif actual is None:
            detail = f"{member}: missing (expected sha256 {expected})"
        else:
            detail = f"{member}: expected sha256 {expected}, actual {actual}"
        super().__init__(f"show bundle authorization failed — {detail}")


class FreeModeError(RuntimeError):
    """The explicitly unsigned free-show escape hatch is not authorized."""


class RunConsentError(PermissionError):
    """Typed operator consent is absent or incorrect."""


class PreShowError(RuntimeError):
    """A machine-checkable pre-show blocker failed under the run lock."""


@dataclass(frozen=True)
class ResolvedBundle:
    policy: Path
    meta: Path
    npz: Path
    motion_csv: Path | None = None
    bundle_manifest: Path | None = None


def _abs(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else shows.PROJECT_ROOT / value


def _verify_member(member: str, path_value, expected) -> Path:
    if not path_value or not expected:
        raise ShowBundleError(member)
    path = _abs(path_value)
    if not path.is_file():
        raise ShowBundleError(member, expected, None)
    try:
        actual = artifacts.sha256_file(path)
    except OSError as exc:
        raise ShowBundleError(member, expected, f"unreadable: {exc}") from exc
    if actual != expected:
        raise ShowBundleError(member, expected, actual)
    return path


def resolve_bundle(dance: "shows.Dance") -> ResolvedBundle:
    """Re-authorize the exact promotion-recorded show bundle; never glob/fallback."""
    policy = _verify_member("policy.onnx", dance.policy_path, dance.policy_sha256)
    meta = _verify_member("policy_meta.json", dance.meta_path, dance.meta_sha256)
    npz = _verify_member("motion NPZ", dance.npz_path, dance.npz_sha256)
    motion = _verify_member("motion CSV", dance.motion_csv, dance.motion_sha256)

    manifest_path: Path | None = None
    if dance.bundle_id:
        manifest_path = policy.parent / "bundle.json"
        if not manifest_path.is_file():
            raise ShowBundleError("bundle.json", dance.bundle_id, None)
        try:
            manifest = json.loads(manifest_path.read_text())
            errors = artifacts.verify_manifest(manifest_path, base_dir=manifest_path.parent)
        except (OSError, json.JSONDecodeError, AttributeError, TypeError) as exc:
            raise ShowBundleError("bundle.json", dance.bundle_id, f"unreadable: {exc}") from exc
        actual_id = manifest.get("bundle_id")
        if actual_id != dance.bundle_id:
            raise ShowBundleError("bundle.json", dance.bundle_id, actual_id)
        if errors:
            raise ShowBundleError("bundle.json", dance.bundle_id, "; ".join(errors))
    elif not dance.legacy_bundle:
        raise ShowBundleError("bundle.json / legacy_bundle authorization")
    return ResolvedBundle(policy, meta, npz, motion, manifest_path)


def _manifest_entries(node, base: Path) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    if isinstance(node, dict):
        if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
            path = Path(node["path"])
            entries.append((path if path.is_absolute() else base / path, node["sha256"]))
        else:
            for value in node.values():
                entries.extend(_manifest_entries(value, base))
    elif isinstance(node, list):
        for value in node:
            entries.extend(_manifest_entries(value, base))
    return entries


def _resolve_unsigned_free_bundle() -> ResolvedBundle:
    """Resolve the standtail trial from one self-consistent (explicitly unsigned) manifest."""
    policy_dir = _abs(FREE_POLICY_DIR)
    manifest_path = policy_dir / "bundle.json"
    if not manifest_path.is_file():
        raise ShowBundleError("free bundle.json", "present and internally valid", None)
    try:
        manifest = json.loads(manifest_path.read_text())
        errors = artifacts.verify_manifest(manifest_path, base_dir=policy_dir)
    except (OSError, json.JSONDecodeError, AttributeError, TypeError) as exc:
        raise ShowBundleError("free bundle.json", "readable", str(exc)) from exc
    if errors:
        raise ShowBundleError("free bundle.json", manifest.get("bundle_id"), "; ".join(errors))
    policy_entries = manifest.get("policy") or {}
    all_entries = _manifest_entries(manifest, policy_dir)

    def entry_path(entry, member):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ShowBundleError(member)
        path = Path(entry["path"])
        return path if path.is_absolute() else policy_dir / path

    policy = entry_path(policy_entries.get("onnx"), "free policy.onnx")
    meta = entry_path(policy_entries.get("meta"), "free policy_meta.json")
    tempo = ((manifest.get("motion") or {}).get("tempo_npz") or {})
    if isinstance(tempo, dict) and tempo.get("100"):
        npz = entry_path(tempo["100"], "free motion NPZ")
    else:
        npzs = [path for path, _sha in all_entries if path.suffix.lower() == ".npz"]
        if len(npzs) != 1:
            raise ShowBundleError("free motion NPZ")
        npz = npzs[0]
    expected_by_path = {path.resolve(): sha for path, sha in all_entries}
    for member, path in (("free policy.onnx", policy),
                         ("free policy_meta.json", meta), ("free motion NPZ", npz)):
        expected = expected_by_path.get(path.resolve())
        if not expected:
            raise ShowBundleError(member)
        _verify_member(member, str(path), expected)
    return ResolvedBundle(policy, meta, npz, bundle_manifest=manifest_path)


def _cli_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(shows.PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _bundle_args(bundle: ResolvedBundle) -> list[str]:
    return [
        "--policy", _cli_path(bundle.policy),
        "--meta", _cli_path(bundle.meta),
        "--motion-npz", _cli_path(bundle.npz),
    ]

# PC2 (Jetson Orin) on the robot control net. A single 1 s ping is the reachability
# probe the run guard uses; it is the only "does the robot answer" check we can make
# without contacting the robot's control interface.
ROBOT_HOST = ROBOT_PC2_IP

# How many trailing run.log lines the status endpoint surfaces (~15 per the API).
TAIL_LINES = 15

# Serializes the check-and-spawn of a run and guards access to _current below.
_lock = threading.Lock()
# The one live-or-most-recent run, or None. Shape:
#   {"show_id", "dance_id", "mode", "proc", "log_path", "started_at"}
# We keep it after the process exits so the status endpoint can still report the
# final phase/log AND so the "record the outcome first" guard can see the open show.
_current: dict | None = None


class RunBusy(RuntimeError):
    """A run cannot start because one is already active or an outcome is pending."""


def robot_reachable(host: str = ROBOT_HOST) -> bool:
    """True iff PC2 answers a single 1 s ping (`ping -c1 -W1 <host>` rc==0).

    Isolated so tests can fake it — they must NEVER touch the real robot net."""
    try:
        return subprocess.run(
            ["ping", "-c1", "-W1", host],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
    except OSError:
        return False


def spawn_show_process(cmd: list[str], env: dict, log_path: Path):
    """Launch the show script detached, streaming stdout+stderr to log_path.

    Returns a Popen-like handle exposing .poll(). Isolated so tests monkeypatch it
    and NEVER spawn the real tools/show_run.sh."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")
    try:
        # start_new_session: the show outlives this request/thread and must not be
        # torn down by signals aimed at the web server; the damping remote — not a
        # process signal — is the stop.
        return subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT), start_new_session=True)
    finally:
        log.close()  # the child holds its own dup of the fd


def _build_env(operator: str, mode: str, exit_stand: bool, audio_mode: str,
               dance_id: str, body: str | None, free: bool = False) -> dict:
    env = dict(os.environ)
    # The operator name doubles as the runtime's CONFIRMED_BY_HUMAN gate; combined
    # with the typed API phrase + the physical remote, this is the deploy human-
    # confirmation CLAUDE.md requires.
    env["CONFIRMED_BY_HUMAN"] = operator
    env["AUDIO_MODE"] = audio_mode or "laptop"     # show default; dance/body may override
    env["AUDIO_LATENCY_COMP"] = "0.0"
    # v3-family sim envelope max ~17.1; 2.2 avoids the benign wrist cap that tripped
    # on hardware at the old 1.6/cap16 (context note, 2026-07-07).
    env["ARM_ACTION_CAP_SCALE"] = "2.2"
    env["DANCE_ID"] = dance_id                      # cue the THIS dance's music track
    if body:
        env["BODY"] = str(body)
    # Side-by-side reference|sim video on the big screen — for EVERY show, not just
    # `free`. BUG (2026-07-08 live run): SHOW_VIDEO was set only inside the `free`
    # branch, so a normal tethered show launched no video (show_run.sh only starts the
    # player when SHOW_VIDEO is non-empty). show_display.py falls back to full-screen
    # on the primary monitor when no external display is connected, so this is safe on
    # a laptop-only setup too; set SHOW_DISPLAY to force a specific xrandr output.
    env.setdefault("SHOW_VIDEO", FREE_SHOW_VIDEO)
    env.setdefault("SHOW_DISPLAY", "")
    if free:
        # HARDWARE-VALIDATED UNTETHERED ("free") SHOW CONFIG (see the module note at
        # FREE_POLICY_DIR). This runs the validated free config for a TRIAL/LIVE show;
        # the sha-pinned signed policy the proven tethered path uses is UNCHANGED —
        # `free` only adds the standtail --policy args (via begin_run) + these knobs.
        env["GROUND_LEG_KP_SCALE"] = "1.5"     # sagittal leg boost that fixed the arm-accent lean
        env["EXIT_MODE"] = "stand"             # standtail motion ends standing + hands to onboard
        env["MAX_SECS"] = "57"                 # the standtail motion is 54.2s
        env["ARM_ACTION_CAP_SCALE"] = "2.2"    # (already default) arm-accent cap validated on hardware
        env["AUDIO_MODE"] = audio_mode or "laptop"   # aux; music auto-cued at tick0+4.0s
        # Env contract for Lane B's side-by-side video launch (set only, not launched here).
        env["SHOW_VIDEO"] = FREE_SHOW_VIDEO
        env.setdefault("SHOW_DISPLAY", "")     # default empty; an operator/env may set a display
    elif exit_stand:
        # Stand-hold exit (OPT-IN) — the "standing at every point" show: after the last
        # dance tick, keep commanding the motion's FINAL standing pose at the policy's
        # holding gains, then hand back to onboard 'ai' while STILL STANDING so the
        # remote/phone can resume (the default ramp-to-damping leaves a damped robot the
        # phone can't recover — 2026-07-08 live-run finding). Enabled for LIVE + rehearsal
        # on operator opt-in. SAFE: deploy_runtime's `--exit stand` GUARD refuses and
        # falls back to damp unless the motion's final frame is within tolerance of the
        # default standing pose, so a non-stand-ending motion can never topple. The train
        # stage builds a stand-ending motion (deploy_ramp stand_end=True); still, the
        # FIRST run on a new policy/hardware must be tethered with the operator present.
        env["EXIT_MODE"] = "stand"
    # Stand-hold exit handback window. The default 2.0s hold + 0.5s overlap is too short for
    # the operator to engage the onboard stand on the remote before deploy_runtime releases —
    # onboard 'ai' takes back over PASSIVE and the robot sags ("limp after standing", 2026-07-08
    # live run). Hold the standing pose firmly for HANDOFF_HOLD_S, then keep holding it for
    # HANDOFF_OVERLAP_S AFTER 'ai' is restored, giving the operator a real window to press the
    # onboard stand/AI on the remote so it is actively balancing before we let go. Same-pose
    # command only (safe); env-overridable.
    if env.get("EXIT_MODE") == "stand":
        env.setdefault("HANDOFF_HOLD_S", "3.0")
        env.setdefault("HANDOFF_OVERLAP_S", "5.0")
    return env


def _why_blocked_locked() -> str | None:
    """Reason a new run may NOT start, or None. Caller holds _lock."""
    run = _current
    if run is None:
        return None
    proc = run.get("proc")
    if proc is not None and proc.poll() is None:
        return "a show is already running"
    # Process has exited: the show must be resolved (outcome recorded) before the
    # next run. Reuses the existing show.closed state (set by shows.record_outcome).
    try:
        show = shows.load_show(run["show_id"])
    except (FileNotFoundError, ValueError):
        return None
    if not show.closed:
        return (f"the previous show ({run['show_id']}) has no recorded outcome yet "
                "— record its outcome before starting another run")
    return None


def why_blocked() -> str | None:
    """Public read of the single-run / open-show guard (None => a run may start)."""
    with _lock:
        return _why_blocked_locked()


def begin_run(dance: "shows.Dance", *, operator: str, mode: str,
              exit_stand: bool = False, audio_mode: str = "laptop",
              body: str | None = None, free: bool = False,
              confirmation: str = "", robot_ping=None,
              venue_active=None) -> "shows.Show":
    """Atomically re-check the lock, create the Show, and spawn show_run.sh.

    Creating the Show INSIDE the lock (after the re-check) means a lost race never
    leaves an orphan open show. Raises RunBusy if a run is active / outcome pending.

    Bundle resolution and machine checks happen while holding the same lock as spawn.
    Typed consent is checked only after those checks, so the operator consents to the
    exact bytes that will be placed in the command.
    """
    with _lock:
        reason = _why_blocked_locked()
        if reason:
            raise RunBusy(reason)
        # Never trust the caller's potentially stale in-memory dance record.
        dance = shows.load_dance(dance.id)
        if free:
            if mode == "live":
                raise FreeModeError(
                    "the unsigned free configuration is forbidden in live mode"
                )
            if os.environ.get("G1_ALLOW_UNSIGNED_FREE") != "1":
                raise FreeModeError(
                    "the unsigned free rehearsal is disabled; set "
                    "G1_ALLOW_UNSIGNED_FREE=1 explicitly to enable the trial path"
                )
            bundle = _resolve_unsigned_free_bundle()
        else:
            bundle = resolve_bundle(dance)

        machine = preshow.evaluate_machine_checks(
            dance, robot_ping=robot_ping, venue_active=venue_active
        )
        failed = [item for item in machine["items"]
                  if item["severity"] == "blocker" and not item["ok"]]
        if failed:
            first = failed[0]
            raise PreShowError(f"pre-show check '{first['key']}' failed: {first['detail']}")
        if confirmation != RUN_CONFIRMATION_PHRASE:
            raise RunConsentError(
                "confirmation phrase does not match — type it EXACTLY, with the "
                "damping remote in your hand"
            )

        show = shows.new_show(dance, operator, mode=mode)
        env = _build_env(operator, mode, exit_stand, audio_mode, dance.id, body, free)
        # Pick the runtime mode from the POLICY's own contract (2026-08-05: v12's
        # history-stacked estimator-free contract refused under the legacy default —
        # deploy_runtime: "mode 'ground-run-legodom' has no history stacker for
        # history_length 5; use ground-run"). history_length > 1 ⇒ the estimator-free
        # stacked family ⇒ ground-run; otherwise keep the proven legodom default.
        # RUNTIME_MODE in the environment still wins (operator override).
        if "RUNTIME_MODE" not in env:
            try:
                hist = json.loads(bundle.meta.read_text()).get("history_length")
            except Exception:  # unreadable meta: keep the proven default, runtime re-gates
                hist = None
            if isinstance(hist, int) and hist > 1:
                env["RUNTIME_MODE"] = "ground-run"
        # Cap the run to THIS dance's length (+3 s entry slack) instead of the
        # historical 52 s default — a 55.8 s motion was silently truncated
        # (2026-08-05, anchor re-promotion). Explicit MAX_SECS env still wins.
        if "MAX_SECS" not in env and dance.duration_s:
            env["MAX_SECS"] = str(int(dance.duration_s) + 3)
        log_path = show.dir / "run.log"
        cmd = [str(SHOW_RUN_SH)] + _bundle_args(bundle)
        proc = spawn_show_process(cmd, env, log_path)
        global _current
        _current = {"show_id": show.id, "dance_id": dance.id, "mode": mode,
                    "proc": proc, "log_path": str(log_path),
                    "started_at": time.time()}
        threading.Thread(target=_watch_run_death, args=(proc, log_path, show.id),
                         daemon=True).start()
        _start_show_camera(show.dir, proc)
        show.log(f"RUN SHOW spawned (mode={mode}, audio={env['AUDIO_MODE']}, "
                 f"exit_mode={env.get('EXIT_MODE', 'ramp-to-damping')}, "
                 f"config={'FREE/untethered (standtail)' if free else 'proven default'}) — "
                 "operator holds the damping remote")
        return show


def _tail(path: Path, n: int) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()[-n:]
    except (FileNotFoundError, OSError):
        return []


def _log_shows_fall(text: str) -> bool:
    """True iff the run log shows the runtime tripped its fall detector.

    deploy_runtime's _check_fall raises RuntimeError("FALL DETECTED ...") the moment
    torso uprightness drops below FALL_UPRIGHT_MIN; the mode's abort path then prints
    that as "STOP: FALL DETECTED ... -> damping" (which damps + hands back to onboard
    'ai'). Either marker means a fall. A cheap substring scan over the log text we have
    already read (no extra I/O); robust to where in the tail the marker landed."""
    return "FALL DETECTED" in text or ("STOP:" in text and "FALL" in text)


def _start_show_camera(show_dir: Path, run_proc) -> None:
    """Record the run on the external camera (RealSense RGB, /dev/video8) into the
    show dir: camera.mp4 (full run) + cam_NNN.jpg every 2 s (reviewable stills —
    added 2026-08-05 so the agent can SEE runs, not just read telemetry).
    Best-effort: no camera or no ffmpeg must never affect the run. SHOW_CAMERA=0
    disables; SHOW_CAMERA_DEV overrides the device."""
    if os.environ.get("SHOW_CAMERA", "1") != "1":
        return
    dev = os.environ.get("SHOW_CAMERA_DEV", "/dev/video8")
    if not Path(dev).exists():
        return
    import shutil
    ff = (shutil.which("ffmpeg")
          or str(Path.home() / "miniconda3/envs/g1dance/bin/ffmpeg"))
    if not Path(ff).exists():
        return
    show_dir.mkdir(parents=True, exist_ok=True)
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-f", "v4l2", "-i", dev,
           "-t", "180",
           "-map", "0", "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-y", str(show_dir / "camera.mp4"),
           "-map", "0", "-vf", "fps=1/2,scale=640:-2", "-q:v", "4",
           "-y", str(show_dir / "cam_%03d.jpg")]
    try:
        cam = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception:
        return

    def _stop_after_run():
        run_proc.wait()
        time.sleep(2)   # catch the damp/settle on camera
        try:
            cam.terminate()
        except Exception:
            pass

    threading.Thread(target=_stop_after_run, daemon=True).start()


def _extract_death_reason(text: str) -> str | None:
    """The exact line that explains why a run refused/aborted/crashed, for the UI
    and the desktop notification (2026-08-05: a day of runs died silently with the
    reason only in run.log — REFUSED lines, guard STOPs, a dead-SDK traceback)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in reversed(lines):
        if l.startswith("REFUSED") or "STOP:" in l:
            return l
    # tracebacks: the last exception line is the reason
    for l in reversed(lines):
        if ("Error:" in l or l.startswith("ModuleNotFoundError")
                or l.startswith("SystemExit")) and "File \"" not in l:
            return l
    return None


def _notify_desktop(title: str, body: str) -> None:
    """Best-effort OS pop-up (notify-send). Never raises — a missing notifier
    must not affect a run."""
    try:
        subprocess.run(["notify-send", "-u", "critical", "-a", "G1 Dance Studio",
                        title, body[:400]], timeout=5,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _watch_run_death(proc, log_path: Path, show_id: str) -> None:
    """Daemon thread: wait for the spawned show to exit; if it did not end
    cleanly, pop a desktop notification with the EXACT reason and append it to
    the show log. The operator must never again watch a robot stand still with
    the explanation hidden in a file."""
    rc = proc.wait()
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        text = ""
    reason = _extract_death_reason(text)
    clean = ("segment done" in text or "ramp to damping" in text
             or "telemetry saved" in text) and not reason
    if clean:
        return
    msg = reason or f"show process exited rc={rc} with no clean-end marker"
    _notify_desktop("Show run ended abnormally", msg)
    try:
        from . import shows
        shows.load_show(show_id).log(f"RUN DIED: {msg}")
    except Exception:
        pass


def _derive_phase(text: str, running: bool) -> str:
    """Map the run.log markers (from deploy_runtime / show_run.sh) to a coarse phase.

    Later stages win over earlier ones; a fall is the highest-priority terminal state,
    above a generic abort ("STOP:"). Markers: 'FALL DETECTED' = the fall detector tripped
    (damp + onboard handoff); 'starting leg-odometry policy' = the dance began;
    'ramp to damping' / 'segment done' = clean end; 'STOP:' = aborted."""
    if not text.strip():
        return "launching" if running else "ended"
    # A fall trips deploy_runtime's detector -> immediate damp + onboard handoff. It is
    # terminal and outranks a plain STOP abort, so the app can steer the operator to
    # record an Incident.
    if _log_shows_fall(text):
        return "fall"
    if "STOP:" in text:
        phase = "stopped"
    elif "REFUSED" in text:
        phase = "refused"
    elif "ramp to damping" in text or "segment done" in text:
        phase = "ramp-to-damping"
    elif ("starting leg-odometry policy" in text
          or "starting ground policy" in text
          or "starting odometry-fed policy" in text
          or "starting policy" in text):
        phase = "performing"
    elif "SHOW RUN" in text or "move-to-default" in text or "GROUND-RUN" in text:
        phase = "arming"
    else:
        phase = "launching"
    # Process gone but no clean-end/stop marker => it exited unexpectedly.
    if not running and phase in ("launching", "arming", "performing"):
        return "ended"
    return phase


def stop_run() -> dict:
    """App-side STOP for a running show. SIGTERMs the show's whole process GROUP
    (spawn_show_process starts a new session, so the pgid == show_run.sh + deploy_runtime
    + the video player). deploy_runtime's signal handler GUARANTEES it damps the robot
    (soft) on any exit path incl. an external SIGTERM (see its module docstring), and
    show_run.sh's trap tears down the video. This is a SECOND, software stop beside the
    operator's physical remote B-damp — the remote stays the primary hard stop. The robot
    goes SOFT (damps) and will sag into the tether, so keep tension on it.
    Returns {stopped, was_running, detail}."""
    with _lock:
        run = _current
        proc = run.get("proc") if run else None
        if proc is None or proc.poll() is not None:
            return {"stopped": False, "was_running": False,
                    "detail": "no show is currently running"}
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError) as e:
            return {"stopped": False, "was_running": True,
                    "detail": f"could not signal the show process: {e}"}
    return {"stopped": True, "was_running": True,
            "detail": "STOP sent — the robot is damping (going soft); catch it on the tether."}


def _pkill_deploy() -> list[str]:
    """SIGTERM any stray deploy_runtime / show_run.sh processes that this module is not
    tracking in ``_current`` (e.g. a run launched from a terminal, or a leftover after a
    crash). SIGTERM — NEVER SIGKILL — so deploy_runtime's signal handler still damps the
    robot on the way out (a SIGKILL would skip damping and leave the motors live). Returns
    the list of cmdline patterns that matched at least one process. Best-effort: a missing
    ``pkill`` binary is non-fatal (returns []). Isolated so tests monkeypatch it and never
    signal real processes."""
    signaled: list[str] = []
    for pat in ("deploy_runtime", "show_run.sh"):
        try:
            rc = subprocess.run(
                ["pkill", "-TERM", "-f", pat],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        except OSError:
            continue
        if rc == 0:          # 0 == at least one process matched and was signaled
            signaled.append(pat)
    return signaled


def emergency_kill() -> dict:
    """Always-available EMERGENCY software stop ("E-STOP"): damp any app-launched policy
    run NOW. Two passes: (1) SIGTERM the tracked show's whole process group, and (2) SIGTERM
    any stray deploy_runtime / show_run.sh process not tracked here. Both use SIGTERM so
    deploy_runtime's signal handler ALWAYS damps the robot soft — this module never SIGKILLs
    a live-control process (that would skip damping and leave the motors energised).

    HONEST SCOPE (CLAUDE.md deploy rule / 2026-07-03 safety review): this can only reach
    processes THIS APP launched. The operator's hand-held remote B-damp — not this button —
    remains the PRIMARY hard stop, and it is the ONLY stop for a robot being driven from the
    remote/onboard 'ai' (which this app has no channel to command). The robot goes SOFT and
    will sag; keep the remote in hand and tension on any tether.

    Returns {stopped, tracked_stopped, strays_signaled, was_running, detail}."""
    with _lock:
        run = _current
        proc = run.get("proc") if run else None
        tracked_running = proc is not None and proc.poll() is None
        tracked_stopped = False
        err: str | None = None
        if tracked_running:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                tracked_stopped = True
            except (ProcessLookupError, PermissionError, OSError) as e:
                err = str(e)
    # pkill outside the lock (a subprocess call): catches anything ``_current`` misses.
    strays = _pkill_deploy()
    hit_something = tracked_stopped or bool(strays)
    if hit_something:
        detail = ("E-STOP sent — the app-launched policy is damping (going soft). "
                  "If the robot is still moving it is being driven from the remote/onboard: "
                  "use the remote B-damp or the power switch.")
    elif err:
        detail = (f"could not signal the tracked run ({err}); tried to stop any stray "
                  "deploy process — if the robot is moving, use the remote B-damp NOW.")
    else:
        detail = ("No app-launched policy run is active. If the robot is moving it is being "
                  "driven from the remote/onboard — stop it with the remote B-damp or the "
                  "power switch.")
    return {"stopped": hit_something, "tracked_stopped": tracked_stopped,
            "strays_signaled": strays, "was_running": tracked_running, "detail": detail}


def current_status() -> dict:
    """Status for GET /api/shows/runs/current: liveness + phase + last log lines."""
    run = _current
    if run is None:
        return {"running": False, "show_id": None, "mode": None,
                "phase": "idle", "last_lines": [], "dance_id": None,
                "started_at": None}
    proc = run.get("proc")
    running = proc is not None and proc.poll() is None
    log_path = Path(run["log_path"])
    try:
        full_text = log_path.read_text(errors="replace")
    except (FileNotFoundError, OSError):
        full_text = ""
    return {
        "running": running,
        "show_id": run["show_id"],
        "dance_id": run.get("dance_id"),
        "mode": run.get("mode"),
        "phase": _derive_phase(full_text, running),
        # Surface a tripped fall detector so the app can flag it + steer the operator
        # to record an Incident (which demotes the dance via record_outcome).
        "fall_detected": _log_shows_fall(full_text),
        # The exact refusal/abort line, so the UI can SAY why instead of the
        # operator digging through run.log (None while running / clean).
        "death_reason": None if running else _extract_death_reason(full_text),
        "last_lines": full_text.splitlines()[-TAIL_LINES:],
        "started_at": run.get("started_at"),
    }
