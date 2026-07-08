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

import os
import subprocess
import threading
import time
from pathlib import Path

from . import shows
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
# box must re-exam it). Requesting `free` therefore deploys the validated free config
# for a TRIAL/LIVE show; it does NOT touch, replace, or re-sign the sha-pinned signed
# policy the proven tethered path uses. Free is OPT-IN only (payload {"free": true}) so
# the proven tethered path stays the default.
FREE_POLICY_DIR = "data/policies/thriller_standtail_candidate"   # project-relative
# Side-by-side reference video (Lane B's show_run.sh reads SHOW_VIDEO/SHOW_DISPLAY to
# launch it; here we only set the env contract — the launch itself is Lane B's).
FREE_SHOW_VIDEO = "data/previews/thriller_side_by_side_v3e.mp4"


def _free_show_args() -> list[str]:
    """Extra ``show_run.sh "$@"`` args that select the untethered standtail policy.

    These flow through show_run.sh's ``"$@"`` into ``pipeline.deploy_runtime``
    (--policy/--meta/--motion-npz). Project-relative paths: show_run.sh cds to the repo
    root and spawn_show_process runs with cwd=PROJECT_ROOT, so they resolve there."""
    d = FREE_POLICY_DIR
    return [
        "--policy", f"{d}/policy.onnx",
        "--meta", f"{d}/policy_meta.json",
        "--motion-npz", f"{d}/thriller_deploy.npz",
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
              body: str | None = None, free: bool = False) -> "shows.Show":
    """Atomically re-check the lock, create the Show, and spawn show_run.sh.

    Creating the Show INSIDE the lock (after the re-check) means a lost race never
    leaves an orphan open show. Raises RunBusy if a run is active / outcome pending.

    ``free=True`` runs the HARDWARE-VALIDATED untethered config (standtail policy +
    leg-gain boost + stand-at-end); see FREE_POLICY_DIR. It is a trial/live show and
    does not alter the sha-pinned signed policy of the proven default path.
    """
    with _lock:
        reason = _why_blocked_locked()
        if reason:
            raise RunBusy(reason)
        show = shows.new_show(dance, operator, mode=mode)
        env = _build_env(operator, mode, exit_stand, audio_mode, dance.id, body, free)
        log_path = show.dir / "run.log"
        # Free adds the standtail --policy/--meta/--motion-npz args through show_run.sh's
        # "$@"; the proven default spawns show_run.sh with no policy override.
        cmd = [str(SHOW_RUN_SH)] + (_free_show_args() if free else [])
        proc = spawn_show_process(cmd, env, log_path)
        global _current
        _current = {"show_id": show.id, "dance_id": dance.id, "mode": mode,
                    "proc": proc, "log_path": str(log_path),
                    "started_at": time.time()}
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
        "last_lines": full_text.splitlines()[-TAIL_LINES:],
        "started_at": run.get("started_at"),
    }
