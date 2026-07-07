"""Pre-show checklist + show-phase ownership model for the operator console.

This is the LOGIC layer the operator console needs right before a dance is
performed; the UI (ui/) renders it. Two things live here:

1. ``evaluate_checklist(dance, ...)`` — a formal pre-show checklist. Each item is a
   ``ChecklistItem`` (key/label/kind/severity + an evaluator). AUTO items are decided
   from artifacts + injected facts (dance record, robot ping, selected venue); CONFIRM
   items are physical things the operator must tick off (damping remote in hand, tether/
   area set, feet placement). ``ready`` is True only when every BLOCKER item is ok — the
   deploy gate the console guards its one-confirmation deploy with.

2. ``make_show_phases()`` — a pure-data description of WHO controls the robot WHEN across
   a performance, so the UI can show it and the operator never has to guess at a handoff.

Design notes:
  * Nothing here mutates a dance or a show — reads only (the checklist re-hashes the
    policy on disk to confirm it is the exact artifact the sim exam passed, but never
    writes). Persistence stays with Lane 1's shows.py.
  * ``robot_ping`` / ``venue_active`` / ``acks`` are INJECTED so the console (and tests)
    supply them without this module reaching across lanes or touching the network. In
    particular ``venue_active`` is Lane 1's selected-venue value passed straight in — we
    duck-type it (only a ``name`` for display), we do not import the venue module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import shows
from .exam_verdict import full_sha256

# CONFIRM-item keys the operator must physically acknowledge. Exposed so the UI /
# caller can build the acks set from the checkboxes the operator ticks.
CONFIRM_KEYS = ("damping_remote", "tether_area", "feet_placement")


@dataclass(frozen=True)
class _Ctx:
    """Everything an evaluator is allowed to look at (all injected — no network)."""
    dance: object
    robot_ping: object          # None | bool | callable() -> truthy
    venue_active: object        # None | Venue-like (duck-typed for .name only)
    acks: frozenset


@dataclass(frozen=True)
class ChecklistItem:
    key: str
    label: str
    kind: str        # "auto"  = decided from artifacts/injected facts
                     # "confirm" = operator must physically tick it
    severity: str    # "blocker" = must be ok for ready; "warn" = advisory only
    evaluate: Callable[[_Ctx], tuple[bool, str]]

    def spec(self) -> dict:
        """Static definition (no evaluation) — for the UI to render the raw list."""
        return {"key": self.key, "label": self.label,
                "kind": self.kind, "severity": self.severity}


# --------------------------------------------------------------------------- #
# Path resolution (read-only): resolve a dance's stored project-relative file
# path the same way shows.py does, honoring its (test-monkeypatched) PROJECT_ROOT.
# --------------------------------------------------------------------------- #
def _abs(project_rel: str) -> Path:
    p = Path(project_rel)
    return p if p.is_absolute() else shows.PROJECT_ROOT / p


def _venue_name(venue_active) -> str | None:
    if venue_active is None:
        return None
    name = getattr(venue_active, "name", None)
    if name is None and isinstance(venue_active, dict):
        name = venue_active.get("name")
    return name


# --------------------------------------------------------------------------- #
# AUTO evaluators
# --------------------------------------------------------------------------- #
def _eval_show_ready(ctx: _Ctx) -> tuple[bool, str]:
    status = getattr(ctx.dance, "status", None)
    if status == "show-ready":
        return True, "dance status is 'show-ready'"
    return False, (f"dance status is {status!r} — must be 'show-ready' "
                   "(sim-verified + enough clean runs + human promotion)")


def _eval_policy_pinned(ctx: _Ctx) -> tuple[bool, str]:
    """Policy attached AND its on-disk bytes still hash to the exam-pinned sha.

    Re-hashing on disk (via exam_verdict.full_sha256) is the same guard shows.promote
    uses: it catches a policy file that was swapped/edited after the passing exam, so a
    checklist can never green-light a robot with an unverified artifact."""
    dance = ctx.dance
    policy_path = getattr(dance, "policy_path", None)
    pinned = getattr(dance, "policy_sha256", None)
    if not policy_path:
        return False, "no policy attached — train + attach a policy first"
    if not pinned:
        return False, "policy has no exam-pinned sha — run the signed sim exam"
    abs_path = _abs(policy_path)
    if not abs_path.is_file():
        return False, f"policy file is missing: {policy_path}"
    current = full_sha256(abs_path)
    if current != pinned:
        return False, ("policy file changed since the passing exam (sha mismatch) — "
                       "re-run the sim exam on the current policy")
    return True, "policy on disk matches the exam-pinned sha (unchanged since it passed)"


def _eval_venue_selected(ctx: _Ctx) -> tuple[bool, str]:
    if ctx.venue_active is None:
        return False, "no venue selected — pick the performance venue"
    name = _venue_name(ctx.venue_active)
    return True, f"venue selected: {name}" if name else "venue selected"


def _eval_audio_attached(ctx: _Ctx) -> tuple[bool, str]:
    audio = getattr(ctx.dance, "audio", None)
    if audio:
        src = audio.get("source") if isinstance(audio, dict) else None
        return True, f"music attached ({src})" if src else "music attached"
    return False, "no music attached — the dance will run silent"


def _eval_robot_reachable(ctx: _Ctx) -> tuple[bool, str]:
    """robot_ping is injected so no network is touched here.

    None -> unknown, which is a NO-GO (never assume reachable). A callable is invoked
    (an exception reads as unreachable); a bool is used as-is."""
    rp = ctx.robot_ping
    if rp is None:
        return False, "robot reachability unknown — no ping provided (treat as NO-GO)"
    if callable(rp):
        try:
            reachable = bool(rp())
        except Exception as e:  # a failed ping is a NO-GO, not a crash
            return False, f"robot ping failed: {type(e).__name__}: {e}"
    else:
        reachable = bool(rp)
    return (reachable,
            "robot responded to ping" if reachable else "robot did NOT respond to ping")


# --------------------------------------------------------------------------- #
# CONFIRM evaluators — ok only once the operator has ticked the key into `acks`
# --------------------------------------------------------------------------- #
def _confirm(key: str, instruction: str) -> Callable[[_Ctx], tuple[bool, str]]:
    def _eval(ctx: _Ctx) -> tuple[bool, str]:
        if key in ctx.acks:
            return True, f"acknowledged — {instruction}"
        return False, instruction
    return _eval


# The checklist, in the order the operator works through it: AUTO artifact/health
# checks first, then the physical CONFIRM ticks. `ready` needs every BLOCKER ok.
CHECKLIST_ITEMS: list[ChecklistItem] = [
    ChecklistItem("show_ready", "Dance is show-ready", "auto", "blocker",
                  _eval_show_ready),
    ChecklistItem("policy_pinned", "Policy attached & unchanged since exam",
                  "auto", "blocker", _eval_policy_pinned),
    ChecklistItem("venue_selected", "Venue selected", "auto", "blocker",
                  _eval_venue_selected),
    ChecklistItem("robot_reachable", "Robot reachable", "auto", "blocker",
                  _eval_robot_reachable),
    # Music is presentation-only (a silent dance is valid), so a missing track WARNS
    # but never blocks the deploy gate.
    ChecklistItem("audio_attached", "Music attached", "auto", "warn",
                  _eval_audio_attached),
    ChecklistItem("damping_remote", "Damping remote in hand", "confirm", "blocker",
                  _confirm("damping_remote",
                           "Hold the damping remote and test it responds; keep it in "
                           "hand for the whole performance (it is the ONLY stop — this "
                           "G1 has no torque-cutting hardware e-stop).")),
    ChecklistItem("tether_area", "Tether / area set per venue", "confirm", "blocker",
                  _confirm("tether_area",
                           "Set the tether and clear the performance area to match the "
                           "selected venue (hard flat ground, footprint within the "
                           "venue radius, clear of people and objects).")),
    ChecklistItem("feet_placement", "Feet placement correct", "confirm", "blocker",
                  _confirm("feet_placement",
                           "Place the robot's feet flat and square at the start mark so "
                           "the entry ramp begins from the expected standing pose.")),
]


def checklist_items() -> list[dict]:
    """Static item specs (key/label/kind/severity), no evaluation — for the UI to
    render the checklist before (or without) a dance being present."""
    return [it.spec() for it in CHECKLIST_ITEMS]


def evaluate_checklist(dance, *, robot_ping=None, venue_active=None,
                       acks=None) -> dict:
    """Evaluate the full pre-show checklist for `dance`.

    Args:
        dance: a dance record (real ``shows.Dance`` or any object exposing
            ``status`` / ``policy_path`` / ``policy_sha256`` / ``audio``). Read only.
        robot_ping: injected robot reachability — ``None`` (unknown -> NO-GO), a
            ``bool``, or a zero-arg callable returning truthy. No network here.
        venue_active: Lane 1's selected-venue value (or None). Duck-typed for a
            display ``name`` only; not imported across lanes.
        acks: the set/iterable of CONFIRM item keys the operator has ticked.

    Returns:
        ``{"items": [{key, label, ok, detail, kind, severity}, ...], "ready": bool}``
        where ``ready`` is True iff every BLOCKER item is ok. WARN items never block.
    """
    ctx = _Ctx(dance=dance, robot_ping=robot_ping, venue_active=venue_active,
               acks=frozenset(acks or ()))
    items: list[dict] = []
    ready = True
    for item in CHECKLIST_ITEMS:
        ok, detail = item.evaluate(ctx)
        items.append({"key": item.key, "label": item.label, "ok": bool(ok),
                      "detail": detail, "kind": item.kind, "severity": item.severity})
        if item.severity == "blocker" and not ok:
            ready = False
    return {"items": items, "ready": ready}


# --------------------------------------------------------------------------- #
# Show-phase ownership model
# --------------------------------------------------------------------------- #
def make_show_phases() -> list[dict]:
    """Ordered [{phase, owner, note}] describing WHO controls the robot WHEN.

    A performance moves through five phases; control of the robot changes hands twice,
    and those handoffs are the hazardous moments, so the console names them explicitly:

      WALK_ON / WALK_OFF  — the operator walks the robot on/off using the robot's
          ONBOARD locomotion (deploy_runtime's RESTORE_MOTION_MODE='ai' service, driven
          from the remote). The dance policy is not running here.
      ARM  — the operator presses RUN (R1+A): the deploy_runtime ENTRY handoff, where
          onboard control is released to the trained policy. The entry ramp (ENTRY_CATCH_S
          + the motion's activation ramp) catches the robot into the dance's first pose.
      DANCE — the trained policy owns the robot for the full choreography.
      STAND — the deploy_runtime EXIT handoff (``--exit stand``): the policy holds the
          final standing pose, then hands the robot back to the onboard 'ai' service
          (SelectMode) with a brief overlap so it is never unheld.

    Pure data (owner is descriptive, not an enum consumed elsewhere); the UI renders it
    so the operator never guesses who is in control at a handoff."""
    return [
        {"phase": "WALK_ON", "owner": "remote/onboard",
         "note": "Operator walks the robot onto the start mark using onboard 'ai' "
                 "locomotion from the remote. Dance policy not yet running."},
        {"phase": "ARM", "owner": "operator",
         "note": "Operator presses RUN (R1+A): deploy_runtime entry handoff — onboard "
                 "control released to the policy; entry ramp catches into pose 0."},
        {"phase": "DANCE", "owner": "policy",
         "note": "Trained policy owns the robot for the full choreography."},
        {"phase": "STAND", "owner": "policy->onboard",
         "note": "Exit handoff: policy holds the final standing pose, then hands back "
                 "to the onboard 'ai' service (brief overlap so it is never unheld)."},
        {"phase": "WALK_OFF", "owner": "remote/onboard",
         "note": "Operator walks the robot off using onboard 'ai' locomotion from the "
                 "remote. Dance policy no longer running."},
    ]
