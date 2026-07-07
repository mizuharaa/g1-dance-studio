"""Tests for tools/wireless_preflight.py — the WIFI show preflight (link + ping + DDS
LowState staleness -> GO/NO-GO). No robot, no unitree SDK, no network: ping output is
canned, the DDS subscriber is mocked, and the staleness sampler is driven by an injected
read_fn + clock so the whole thing runs headless in the g1dance env.

Covers the spec'd surface:
  * parse_ping -> RTT/loss (typical, '+errors' variant, 100%-loss, malformed);
  * link parsing (ip -o addr) + same-subnet logic;
  * GO/NO-GO logic honors every threshold (loss, ping RTT, DDS p99/max, deadman, window,
    subnet, and the ping-only-is-never-GO safety rule), including CLI threshold overrides;
  * the DDS-staleness path works with a MOCKED subscriber (gap distribution, dedup by tick,
    read misses, deadman trips).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

import tools.wireless_preflight as wp


# ---- ping parsing --------------------------------------------------------------
PING_OK = """PING 192.168.1.164 (192.168.1.164) 56(84) bytes of data.
64 bytes from 192.168.1.164: icmp_seq=1 ttl=64 time=1.23 ms
64 bytes from 192.168.1.164: icmp_seq=2 ttl=64 time=2.01 ms

--- 192.168.1.164 ping statistics ---
100 packets transmitted, 100 received, 0% packet loss, time 24810ms
rtt min/avg/max/mdev = 0.812/1.740/4.230/0.510 ms
"""

PING_ERRORS = """--- 192.168.1.164 ping statistics ---
200 packets transmitted, 198 received, +2 errors, 1% packet loss, time 39880ms
rtt min/avg/max/mdev = 0.9/3.4/58.0/6.2 ms
"""

PING_ALL_LOST = """--- 192.168.1.164 ping statistics ---
100 packets transmitted, 0 received, 100% packet loss, time 99000ms
"""


def test_parse_ping_typical():
    s = wp.parse_ping(PING_OK)
    assert (s.transmitted, s.received, s.loss_pct) == (100, 100, 0.0)
    assert s.rtt_avg_ms == pytest.approx(1.74)
    assert s.rtt_max_ms == pytest.approx(4.23)
    assert s.rtt_mdev_ms == pytest.approx(0.51)


def test_parse_ping_errors_variant():
    s = wp.parse_ping(PING_ERRORS)
    assert s.transmitted == 200 and s.received == 198 and s.loss_pct == 1.0
    assert s.rtt_max_ms == pytest.approx(58.0)


def test_parse_ping_all_lost_has_no_rtt():
    s = wp.parse_ping(PING_ALL_LOST)
    assert s.loss_pct == 100.0 and s.rtt_avg_ms is None


def test_parse_ping_malformed_raises():
    with pytest.raises(ValueError):
        wp.parse_ping("no statistics here")


# ---- link / subnet -------------------------------------------------------------
def test_parse_ip_o_addr():
    line = ("3: wlp0s20f3    inet 192.168.1.10/24 brd 192.168.1.255 scope global "
            "dynamic noprefixroute wlp0s20f3\\       valid_lft 3000sec")
    assert wp.parse_ip_o_addr(line) == ("192.168.1.10", 24)
    assert wp.parse_ip_o_addr("no inet here") is None


def test_same_subnet():
    assert wp.same_subnet("192.168.1.10", 24, "192.168.1.164") is True
    # the WIRED robot IP over the wifi iface must NOT look same-subnet (a real footgun guard)
    assert wp.same_subnet("192.168.1.10", 24, "192.168.123.164") is False


# ---- percentile ----------------------------------------------------------------
def test_percentile_nearest_rank():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert wp.percentile(vals, 50) == 2.0
    assert wp.percentile(vals, 100) == 4.0
    assert wp.percentile(vals, 0) == 1.0
    assert wp.percentile([], 99) != wp.percentile([], 99)  # NaN


# ---- staleness sampler (mocked read_fn + clock) --------------------------------
def _msg(tick):
    return SimpleNamespace(tick=tick)


def _scripted(reads, clocks):
    """read_fn/clock closures backed by lists; read_fn returns None once exhausted, clock
    repeats its last value once exhausted (the loop breaks on n_samples first here)."""
    r = iter(reads)
    c = iter(clocks)
    last = [clocks[-1]]

    def read_fn(_timeout):
        return next(r, None)

    def clock():
        try:
            last[0] = next(c)
        except StopIteration:
            pass
        return last[0]
    return read_fn, clock


def test_measure_staleness_gaps_dedup_misses_deadman():
    # ticks:   1  , None, 1(dup), 2   , 3   , None, 3(dup), 4
    reads = [_msg(1), None, _msg(1), _msg(2), _msg(3), None, _msg(3), _msg(4)]
    # start + one clock per iteration (8 iters)
    clocks = [0.0, 0.000, 0.010, 0.020, 0.024, 0.030, 0.031, 0.032, 0.550]
    read_fn, clock = _scripted(reads, clocks)
    st = wp.measure_staleness(read_fn, clock=clock, poll_timeout_s=0.05,
                              deadman_s=0.5, n_samples=4, max_wall_s=100.0)
    assert st.n_samples == 4          # four DISTINCT ticks (dups skipped)
    assert st.misses == 2             # two None reads
    assert st.n_gaps == 3             # gaps between the 4 fresh samples
    assert st.deadman_trips == 1      # the 0.030 -> 0.550 gap = 520 ms >= 0.5 s
    assert st.max_ms == pytest.approx(520.0)
    assert min(st.gaps_ms) == pytest.approx(6.0)   # 0.030 - 0.024


def test_measure_staleness_with_mocked_subscriber():
    """The DDS path built from a unitree-style ChannelSubscriber whose .Read is mocked."""
    sub = mock.Mock()
    sub.Read.side_effect = [_msg(10), _msg(11), None, _msg(12)]
    read_fn = wp._read_fn_from_subscriber(sub)
    clocks = [0.0, 0.001, 0.004, 0.005, 0.009]  # start + 4 iters
    _, clock = _scripted([], clocks)
    st = wp.measure_staleness(read_fn, clock=clock, n_samples=3, max_wall_s=100.0)
    assert st.n_samples == 3 and st.misses == 1 and st.n_gaps == 2
    assert sub.Read.called


def test_measure_staleness_needs_a_bound():
    with pytest.raises(ValueError):
        wp.measure_staleness(lambda t: None, clock=lambda: 0.0)


# ---- GO / NO-GO logic ----------------------------------------------------------
def _good_ping():
    return wp.PingStats(100, 100, 0.0, 0.8, 1.7, 4.2, 0.5)


def _good_stale(**over):
    base = dict(n_samples=1500, n_gaps=1499, misses=0, deadman_trips=0,
                p50_ms=1.8, p95_ms=2.0, p99_ms=3.0, max_ms=8.0, mean_ms=1.9)
    base.update(over)
    return wp.StalenessStats(**base)


def test_evaluate_go_when_all_pass():
    v = wp.evaluate(_good_ping(), _good_stale(), wp.Thresholds(),
                    iface_up=True, subnet_ok=True, dds_requested=True)
    assert v.go is True


def test_evaluate_nogo_on_packet_loss():
    lossy = wp.PingStats(100, 99, 1.0, 0.8, 1.7, 4.2, 0.5)
    v = wp.evaluate(lossy, _good_stale(), wp.Thresholds(),
                    iface_up=True, subnet_ok=True, dds_requested=True)
    assert v.go is False


def test_evaluate_nogo_on_high_p99():
    v = wp.evaluate(_good_ping(), _good_stale(p99_ms=15.0), wp.Thresholds(),
                    iface_up=True, subnet_ok=True, dds_requested=True)
    assert v.go is False


def test_evaluate_nogo_on_deadman_trip_or_miss():
    assert wp.evaluate(_good_ping(), _good_stale(deadman_trips=1), wp.Thresholds(),
                       iface_up=True, subnet_ok=True, dds_requested=True).go is False
    assert wp.evaluate(_good_ping(), _good_stale(misses=3), wp.Thresholds(),
                       iface_up=True, subnet_ok=True, dds_requested=True).go is False


def test_evaluate_nogo_on_short_window():
    v = wp.evaluate(_good_ping(), _good_stale(n_gaps=50), wp.Thresholds(),
                    iface_up=True, subnet_ok=True, dds_requested=True)
    assert v.go is False


def test_evaluate_nogo_on_bad_subnet_or_iface():
    assert wp.evaluate(_good_ping(), _good_stale(), wp.Thresholds(),
                       iface_up=True, subnet_ok=False, dds_requested=True).go is False
    assert wp.evaluate(_good_ping(), _good_stale(), wp.Thresholds(),
                       iface_up=True, subnet_ok=None, dds_requested=True).go is False
    assert wp.evaluate(_good_ping(), _good_stale(), wp.Thresholds(),
                       iface_up=False, subnet_ok=True, dds_requested=True).go is False


def test_ping_only_is_never_go():
    """Even a perfect link+ping without --dds must be NO-GO — staleness is the real metric."""
    v = wp.evaluate(_good_ping(), None, wp.Thresholds(),
                    iface_up=True, subnet_ok=True, dds_requested=False)
    assert v.go is False and "insufficient" in v.note.lower()


def test_threshold_override_flips_verdict():
    borderline = _good_stale(p99_ms=12.0, max_ms=13.0)
    # default p99 threshold is 10 -> NO-GO
    assert wp.evaluate(_good_ping(), borderline, wp.Thresholds(),
                       iface_up=True, subnet_ok=True, dds_requested=True).go is False
    # relax it to 15 -> GO
    relaxed = wp.Thresholds()
    relaxed.max_p99_gap_ms = 15.0
    assert wp.evaluate(_good_ping(), borderline, relaxed,
                       iface_up=True, subnet_ok=True, dds_requested=True).go is True


# ---- main() wiring -------------------------------------------------------------
def test_main_requires_iface_and_host(monkeypatch, capsys):
    monkeypatch.delenv("ROBOT_IFACE", raising=False)
    monkeypatch.delenv("ROBOT_HOST", raising=False)
    assert wp.main(["192.168.1.164"]) == 2          # no iface
    assert wp.main(["--iface", "wlan0"]) == 2        # no host


def test_main_ping_only_is_nogo(monkeypatch):
    monkeypatch.setattr(wp, "iface_operstate", lambda i: "up")
    monkeypatch.setattr(wp, "iface_ipv4", lambda i: ("192.168.1.10", 24))
    monkeypatch.setattr(wp, "resolve_host", lambda h: "192.168.1.164")
    monkeypatch.setattr(wp, "run_ping", lambda *a, **k: PING_OK)
    rc = wp.main(["192.168.1.164", "--iface", "wlan0"])
    assert rc == 1          # link+ping fine, but no --dds -> deliberately NO-GO


def test_main_full_go_with_mocked_dds(monkeypatch):
    monkeypatch.setattr(wp, "iface_operstate", lambda i: "up")
    monkeypatch.setattr(wp, "iface_ipv4", lambda i: ("192.168.1.10", 24))
    monkeypatch.setattr(wp, "resolve_host", lambda h: "192.168.1.164")
    monkeypatch.setattr(wp, "run_ping", lambda *a, **k: PING_OK)
    # stub the DDS init + the (separately unit-tested, timing-based) sampler so main()'s
    # wiring/verdict/exit-code is what's under test here, deterministically.
    monkeypatch.setattr(wp, "make_lowstate_read_fn", lambda iface: (lambda t: _msg(1)))
    monkeypatch.setattr(wp, "measure_staleness", lambda *a, **k: _good_stale())

    rc = wp.main(["192.168.1.164", "--iface", "wlan0", "--dds", "--json"])
    assert rc == 0
