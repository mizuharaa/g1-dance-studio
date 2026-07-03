"""Authenticated, content-derived authorization for sim-exam verdicts.

Safety review (docs/safety_review_findings.md) CRITICAL #0: the deploy chain trusted
the top-level ``"verdict": "pass"`` string in an exam JSON as authorization. That
string is inert, unauthenticated data — a hand-edit of one word, or a fabricated file
with empty phase dicts, could push an UNTESTED policy to a 35 kg humanoid.

This module makes authorization (a) DERIVED from phase contents (never the self-declared
string) and (b) AUTHENTICATED with an HMAC the web-facing process is not meant to hold.

Every consumer (deploy/gen_config.py, deploy/02_push_bundle.sh, ui/server.py) must call
``authorize()`` — which requires BOTH a valid signature AND a genuine content-derived
pass — rather than reading ``verdict["verdict"]``.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

# consecutive-clean / clean-run floor a verdict must show to authorize a deploy.
REQUIRED_CLEAN_RUNS = 3
# Held-out survival floor for show-ready (user decision 2026-07-04): a dance is
# sim-verified at >=99% survival across held-out episodes, with gantry-first robot
# day as the compensating control. Was a strict 100% (clean == runs); loosened to
# 0.99 deliberately and per explicit authorization. Still needs >= REQUIRED_CLEAN_RUNS
# episodes so a 1-run "100%" cannot sneak through.
REQUIRED_CLEAN_RATE = 0.99
# a push suite below this force floor cannot authorize (findings #22): a 5 N love-tap
# proving nothing must not read as "push PASS".
MIN_PUSH_FORCE_N = 150.0

_SIGNING_KEY_PATH = Path(__file__).resolve().parent.parent / ".secrets" / "exam_signing.key"


def full_sha256(p: Path) -> str:
    """Full 64-hex digest — finding #32 (16-char prefixes are too weak for identity)."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _signing_key() -> bytes:
    """Load (or create) the exam signing key.

    Kept in .secrets/ (gitignored, chmod 600). sim_exam.py signs with it; consumers
    verify. It is NOT a perfect trust boundary on a single-user laptop, but it defeats
    the actual finding: a hand-edited or fabricated verdict no longer authorizes,
    because re-signing requires this key, which casual/accidental edits will not do.
    """
    if _SIGNING_KEY_PATH.exists():
        return _SIGNING_KEY_PATH.read_bytes()
    _SIGNING_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(32)
    _SIGNING_KEY_PATH.write_bytes(key)
    _SIGNING_KEY_PATH.chmod(0o600)
    return key


def _canonical(verdict: dict) -> bytes:
    """Stable serialization of everything except the signature itself."""
    body = {k: v for k, v in verdict.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def sign_verdict(verdict: dict, key: bytes | None = None) -> dict:
    """Return a copy of ``verdict`` with an HMAC-SHA256 ``signature`` field."""
    key = key if key is not None else _signing_key()
    signed = dict(verdict)
    signed.pop("signature", None)
    signed["signature"] = hmac.new(key, _canonical(signed), hashlib.sha256).hexdigest()
    return signed


def signature_valid(verdict: dict, key: bytes | None = None) -> bool:
    sig = verdict.get("signature")
    if not isinstance(sig, str):
        return False
    key = key if key is not None else _signing_key()
    expect = hmac.new(key, _canonical(verdict), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expect)


def _phase_passed(phase: object) -> bool:
    """A phase authorizes only if it is a dict that actually ran AND passed.

    Empty dict / None / missing ``pass`` all read as FAILURE (finding #0, #21, #26):
    ``{}`` used to satisfy an ``is not None`` presence check.
    """
    return isinstance(phase, dict) and phase.get("pass") is True


def derive_pass(verdict: dict) -> bool:
    """Re-derive the pass decision from phase CONTENTS, ignoring the ``verdict`` string.

    All three phases must be present, have actually run, and passed; repeatability must
    show >= REQUIRED_CLEAN_RUNS episodes with a survival rate >= REQUIRED_CLEAN_RATE
    (the user's >=99% show-ready standard); the push suite must meet the force floor.
    """
    if not isinstance(verdict, dict) or verdict.get("schema") != "sim_exam/v1":
        return False
    nominal, push, repeat = verdict.get("nominal"), verdict.get("push"), verdict.get("repeatability")
    if not (_phase_passed(nominal) and _phase_passed(push) and _phase_passed(repeat)):
        return False
    if not (isinstance(push.get("force_n"), (int, float)) and push["force_n"] >= MIN_PUSH_FORCE_N):
        return False
    runs, clean = repeat.get("runs"), repeat.get("clean")
    if not (isinstance(runs, int) and isinstance(clean, int)) or runs <= 0:
        return False
    return clean >= REQUIRED_CLEAN_RUNS and (clean / runs) >= REQUIRED_CLEAN_RATE


def authorize(
    verdict: dict,
    *,
    policy_sha: str | None = None,
    motion_sha: str | None = None,
    key: bytes | None = None,
) -> tuple[bool, str]:
    """The single gate every consumer must use. Returns (ok, reason).

    ok=True requires: valid signature AND content-derived pass AND (if provided) the
    verdict binds to exactly this policy+motion full sha256.
    """
    if not signature_valid(verdict, key):
        return False, "signature invalid or missing — verdict is not authentically signed"
    if not derive_pass(verdict):
        return False, "content-derived pass failed (a phase did not genuinely pass)"
    if policy_sha is not None and verdict.get("policy_sha256") != policy_sha:
        return False, "policy sha256 does not match the verdict"
    if motion_sha is not None and verdict.get("motion_sha256") != motion_sha:
        return False, "motion sha256 does not match the verdict"
    return True, "authorized"
