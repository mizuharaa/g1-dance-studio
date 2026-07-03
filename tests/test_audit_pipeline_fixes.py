"""Regression tests for the pipeline-lane production-audit fixes owned by this
worker (venue.py grounding, monitor.py cost-after-deletion, ui/desktop.py stale
port). The local_motion.py and library.py findings are owned by another agent and
were reverted here to avoid a merge collision.
"""
from datetime import datetime, timezone

import numpy as np
import pytest


# ---- Finding: fit_motion_to_venue ran the pelvis>=0.35 z test un-grounded (LOW) ----
def test_venue_fit_grounds_before_window(monkeypatch):
    import pipeline.grounding as grounding
    from pipeline import venue as venuemod

    called = {"ground": 0}

    def fake_ground(m):
        called["ground"] += 1
        return np.asarray(m), 0.0

    monkeypatch.setattr(grounding, "have_model", lambda: True)
    monkeypatch.setattr(grounding, "ground_motion", fake_ground)

    # A motion whose XY footprint (radius ~3 m) does NOT fit a 2 m venue, so the
    # code enters the `not fits` branch that calls longest_window (the z-dependent
    # path that must run on a grounded motion).
    n = 90
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    m = np.zeros((n, 36))
    m[:, 0] = 3.0 * np.cos(th)
    m[:, 1] = 3.0 * np.sin(th)
    m[:, 2] = 0.7  # pelvis height

    v = venuemod.default_venue()
    report = venuemod.fit_motion_to_venue(m, v)
    assert report["fits_whole"] is False
    assert called["ground"] >= 1  # grounding applied before the z-dependent window


def test_venue_fit_survives_without_model(monkeypatch):
    # No model available → still works (no grounding, prior behaviour).
    import pipeline.grounding as grounding
    from pipeline import venue as venuemod
    monkeypatch.setattr(grounding, "have_model", lambda: False)
    n = 90
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    m = np.zeros((n, 36))
    m[:, 0] = 3.0 * np.cos(th)
    m[:, 1] = 3.0 * np.sin(th)
    m[:, 2] = 0.7
    report = venuemod.fit_motion_to_venue(m, venuemod.default_venue())
    assert report["fits_whole"] is False
    assert "suggested_window" in report


# ---- Finding: cost accrues forever after box deletion (LOW) ----
def _iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def test_cost_stops_at_deletion():
    from pipeline.monitor import compute_cost
    created = 1_000_000.0
    now = created + 10 * 3600  # 10h later
    billing = {"created_at": _iso(created), "rate_vnd_per_hour": 1000.0,
               "cap_vnd": 1_000_000.0, "usd_per_vnd": 4e-5}
    live = compute_cost(billing, now=now)
    assert live["hours"] == 10.0
    billing["deleted_at"] = _iso(created + 4 * 3600)  # delete after 4h
    frozen = compute_cost(billing, now=now)
    assert frozen["hours"] == 4.0
    assert frozen["accrued_vnd"] == 4000.0
    assert compute_cost(billing, now=now + 99 * 3600)["hours"] == 4.0


# ---- Finding: desktop silently attaches to a stale server on 8735 (LOW) ----
def test_desktop_port_in_use_detection():
    import socket as _sock
    from ui.desktop import _port_in_use
    with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as s:
        s.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert _port_in_use("127.0.0.1", port) is True   # occupied → detected
    assert _port_in_use("127.0.0.1", port) is False       # released → free
