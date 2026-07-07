#!/usr/bin/env python3
"""Wireless show preflight — is the WIFI link good enough to fly the 50 Hz balance loop?

For an UNTETHERED show the laptop reaches the robot over WIFI instead of the wired
ethernet the teleop uses (see docs/WIRELESS_SHOW.md). The deploy runtime's balance loop
runs at 50 Hz and reads LowState EVERY tick (20 ms budget); wifi jitter/dropout -> stale
reads -> the runtime's comms-loss deadman (read_state timeout 0.5 s -> damp) fires, or a
late command destabilizes a 35 kg robot. So the wifi link MUST be latency/jitter-validated
BEFORE any untethered wireless run. This tool measures it and emits a GO / NO-GO.

Three checks, escalating in fidelity:
  1. LINK      the wifi interface (ROBOT_IFACE / --iface) is UP and shares the robot's subnet
               (else CycloneDDS peer discovery can't work and control never starts).
  2. PING      N ICMP pings -> RTT min/avg/max/mdev + packet loss. A coarse transport proxy.
  3. STALENESS (--dds, the REAL metric) init CycloneDDS on the wifi iface, subscribe to
               rt/lowstate like the runtime does, and measure the gap distribution between
               successive FRESH LowState samples over a sustained window. This is exactly
               the staleness the 50 Hz loop sees. Wired baseline (deploy_runtime, verified
               on hardware): staleness p95 ~1.75 ms.

The DDS part is READ-ONLY w.r.t. the robot (subscribe only, never publish) and is a
separate, guarded code path so the tests never need the robot or the unitree SDK (which
lives only in the `tv` conda env). Ping/link parsing is pure and unit-tested.

Usage (run in the `tv` env for --dds; ping-only works anywhere):

    conda activate tv
    # link + ping only (quick, but NOT sufficient to authorize a wireless run):
    ROBOT_IFACE=wlp0s20f3 PYTHONPATH=. python tools/wireless_preflight.py 192.168.1.164
    # full GO/NO-GO with the real staleness window (do this before any untethered run):
    ROBOT_IFACE=wlp0s20f3 PYTHONPATH=. python tools/wireless_preflight.py 192.168.1.164 \
        --dds --dds-secs 60

Exit code: 0 = GO, 1 = NO-GO (incl. ping-only, which is deliberately never a GO), 2 = error.

SAFETY: a passing preflight is necessary, not sufficient. Validate on the tether first,
keep the wired cable as a fallback, and never run wireless untethered until this passes
over a sustained window (see docs/WIRELESS_SHOW.md).
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

# ---- control-loop facts these thresholds are derived from ----------------------
CONTROL_HZ = 50.0
TICK_MS = 1000.0 / CONTROL_HZ                 # 20 ms — the per-tick budget
# The runtime's run loop reads LowState with a 0.5 s timeout (pipeline.deploy_runtime
# read_state timeout_s=0.5); a single read exceeding this raises -> the mode's
# except/finally damps. So a fresh-sample gap >= this IS a comms-loss deadman trip.
DEADMAN_S = 0.5
WIRED_STALENESS_P95_MS = 1.75                 # hardware baseline, for context in the report


@dataclass
class Thresholds:
    """GO/NO-GO thresholds. Justified in docs/WIRELESS_SHOW.md; overridable on the CLI.

    Rationale: the 50 Hz loop wants a FRESH LowState every 20 ms tick with margin, and must
    never approach the 0.5 s comms-loss deadman. We therefore require staleness p99 under
    half a tick (10 ms), max under one full tick (20 ms), zero deadman trips, zero read
    misses, and a window long enough (min_gaps) to trust the tail. Ping is a coarse proxy
    only — its thresholds are generous relative to the DDS ones."""
    max_loss_pct: float = 0.0                 # any loss = NO-GO (a dropped tick is a stale read)
    max_avg_rtt_ms: float = 10.0              # avg RTT (proxy) under half a tick
    max_rtt_ms: float = 50.0                  # worst-case ping RTT ceiling (proxy)
    max_mdev_ms: float = 10.0                 # ping jitter (proxy) under half a tick
    max_p99_gap_ms: float = 10.0              # DDS staleness p99 under half a tick  <-- the real gate
    max_gap_ms: float = 20.0                  # DDS staleness worst gap under one tick
    min_gaps: int = 200                       # a window shorter than this is not trustworthy


@dataclass
class PingStats:
    transmitted: int
    received: int
    loss_pct: float
    rtt_min_ms: Optional[float] = None
    rtt_avg_ms: Optional[float] = None
    rtt_max_ms: Optional[float] = None
    rtt_mdev_ms: Optional[float] = None


@dataclass
class StalenessStats:
    n_samples: int          # fresh (distinct-tick) LowState samples seen
    n_gaps: int             # inter-sample gaps measured (= n_samples - 1 when > 0)
    misses: int             # reads that returned no sample within the poll timeout
    deadman_trips: int      # gaps >= DEADMAN_S (a real 0.5 s comms-loss trip)
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float
    gaps_ms: List[float] = field(default_factory=list)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Verdict:
    go: bool
    note: str
    checks: List[Check]


# ---- pure parsers (unit-tested; no robot, no network) --------------------------
_PING_COUNTS_RE = re.compile(
    r"(\d+)\s+packets transmitted,\s+(\d+)\s+(?:packets\s+)?received,"
    r"(?:\s*\+\d+\s+errors,)?\s*([\d.]+)%\s+packet loss")
_PING_RTT_RE = re.compile(
    r"(?:rtt|round-trip)\s+min/avg/max/(?:mdev|stddev)\s*=\s*"
    r"([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)\s*ms")


def parse_ping(text: str) -> PingStats:
    """Parse iputils `ping` summary output into a PingStats.

    Handles the '+N errors' variant and the 100%-loss case (no rtt line -> rtt fields None).
    Raises ValueError if the transmitted/received/loss summary line is absent."""
    m = _PING_COUNTS_RE.search(text)
    if not m:
        raise ValueError("could not parse ping statistics line from output:\n" + text[-500:])
    tx, rx, loss = int(m.group(1)), int(m.group(2)), float(m.group(3))
    rtt = _PING_RTT_RE.search(text)
    if rtt:
        return PingStats(tx, rx, loss,
                         float(rtt.group(1)), float(rtt.group(2)),
                         float(rtt.group(3)), float(rtt.group(4)))
    return PingStats(tx, rx, loss)


_IP_ADDR_RE = re.compile(r"\binet\s+(\d+\.\d+\.\d+\.\d+)/(\d+)\b")


def parse_ip_o_addr(text: str) -> Optional[tuple]:
    """Parse `ip -o -4 addr show dev IFACE` -> (address, prefixlen) or None if no IPv4."""
    m = _IP_ADDR_RE.search(text)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def same_subnet(iface_addr: str, prefixlen: int, host_ip: str) -> bool:
    """True iff host_ip is in the iface's IPv4 subnet (the L2 CycloneDDS discovery needs)."""
    net = ipaddress.ip_network(f"{iface_addr}/{prefixlen}", strict=False)
    return ipaddress.ip_address(host_ip) in net


def percentile(sorted_vals: List[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted list. NaN on empty."""
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return sorted_vals[0]
    if q >= 100:
        return sorted_vals[-1]
    rank = math.ceil(q / 100.0 * len(sorted_vals))
    idx = min(max(rank - 1, 0), len(sorted_vals) - 1)
    return sorted_vals[idx]


def summarize_gaps(gaps_ms: List[float], misses: int, deadman_trips: int,
                   n_samples: int) -> StalenessStats:
    s = sorted(gaps_ms)
    n = len(s)
    return StalenessStats(
        n_samples=n_samples, n_gaps=n, misses=misses, deadman_trips=deadman_trips,
        p50_ms=percentile(s, 50) if n else float("nan"),
        p95_ms=percentile(s, 95) if n else float("nan"),
        p99_ms=percentile(s, 99) if n else float("nan"),
        max_ms=s[-1] if n else float("nan"),
        mean_ms=(sum(s) / n) if n else float("nan"),
        gaps_ms=gaps_ms,
    )


def measure_staleness(read_fn: Callable[[float], object], *,
                      clock: Callable[[], float] = time.perf_counter,
                      poll_timeout_s: float = 0.05,
                      deadman_s: float = DEADMAN_S,
                      n_samples: Optional[int] = None,
                      duration_s: Optional[float] = None,
                      key_fn: Optional[Callable[[object], object]] = None,
                      max_wall_s: Optional[float] = None) -> StalenessStats:
    """Measure the gap distribution between successive FRESH LowState samples.

    `read_fn(timeout_s)` returns a LowState-like message or None (a read miss). Samples are
    de-duplicated by `key_fn` (default: the message's .tick, the robot's ms clock) so that a
    latched-but-unchanged read is not miscounted as a new arrival — the measured gaps are the
    true inter-sample staleness the 50 Hz loop would experience. Any gap >= deadman_s is
    counted as a comms-loss deadman trip. Bounded by n_samples and/or duration_s (at least one
    required); a hard max_wall_s guard keeps a dead stream from hanging the tool.

    Fully injectable (read_fn + clock) so tests drive it with a mocked subscriber — no robot."""
    if n_samples is None and duration_s is None:
        raise ValueError("measure_staleness needs n_samples or duration_s")
    if key_fn is None:
        key_fn = lambda m: getattr(m, "tick", None)
    if max_wall_s is None:
        max_wall_s = (duration_s + 5.0) if duration_s is not None \
            else (n_samples * poll_timeout_s * 4.0 + 5.0)

    gaps_ms: List[float] = []
    misses = 0
    deadman_trips = 0
    got = 0
    prev_key = None
    prev_t = None
    start = clock()
    while True:
        if n_samples is not None and got >= n_samples:
            break
        msg = read_fn(poll_timeout_s)
        now = clock()
        if duration_s is not None and now - start >= duration_s:
            break
        if now - start >= max_wall_s:
            break
        if msg is None:
            misses += 1
            continue
        k = key_fn(msg)
        if prev_key is not None and k == prev_key:
            continue                          # same latched sample — not a new arrival
        if prev_t is not None:
            dt = now - prev_t
            gaps_ms.append(dt * 1000.0)
            if dt >= deadman_s:
                deadman_trips += 1
        prev_key = k
        prev_t = now
        got += 1
    return summarize_gaps(gaps_ms, misses, deadman_trips, got)


def evaluate(ping: Optional[PingStats], staleness: Optional[StalenessStats],
             thr: Thresholds, *, iface_up: bool, subnet_ok: Optional[bool],
             dds_requested: bool) -> Verdict:
    """Combine the link / ping / staleness results into a GO / NO-GO verdict.

    GO requires ALL of: iface up, same subnet, ping clean, AND a passing DDS staleness
    window. Ping-only (no --dds) is deliberately NEVER a GO — staleness is the real metric
    for 50 Hz control, so authorizing an untethered run on ping alone is unsafe."""
    checks: List[Check] = []

    checks.append(Check("link.iface_up", bool(iface_up),
                        "wifi interface is up" if iface_up else "wifi interface is DOWN"))
    subnet_pass = subnet_ok is True
    checks.append(Check("link.subnet", subnet_pass,
                        "robot host shares the wifi subnet" if subnet_pass else
                        ("robot host NOT on the wifi subnet — CycloneDDS won't discover it"
                         if subnet_ok is False else
                         "could not confirm the robot host is on the wifi subnet")))

    ping_ok = False
    if ping is not None:
        loss_ok = ping.loss_pct <= thr.max_loss_pct
        checks.append(Check("ping.loss", loss_ok,
                            f"packet loss {ping.loss_pct:.1f}% (<= {thr.max_loss_pct:.1f}%)"))
        have_rtt = ping.rtt_avg_ms is not None
        rtt_ok = (have_rtt and ping.rtt_avg_ms <= thr.max_avg_rtt_ms
                  and ping.rtt_max_ms <= thr.max_rtt_ms
                  and ping.rtt_mdev_ms <= thr.max_mdev_ms)
        checks.append(Check(
            "ping.rtt", rtt_ok,
            (f"avg {ping.rtt_avg_ms:.2f} / max {ping.rtt_max_ms:.2f} / "
             f"mdev {ping.rtt_mdev_ms:.2f} ms "
             f"(<= {thr.max_avg_rtt_ms:.0f}/{thr.max_rtt_ms:.0f}/{thr.max_mdev_ms:.0f})")
            if have_rtt else "no RTT samples (all pings lost)"))
        ping_ok = loss_ok and rtt_ok
    else:
        checks.append(Check("ping.loss", False, "ping did not run"))

    dds_ok = False
    if dds_requested:
        if staleness is not None:
            window_ok = staleness.n_gaps >= thr.min_gaps
            no_miss = staleness.misses == 0
            no_trip = staleness.deadman_trips == 0
            p99_ok = staleness.p99_ms <= thr.max_p99_gap_ms
            max_ok = staleness.max_ms <= thr.max_gap_ms
            checks.append(Check("dds.window", window_ok,
                                f"{staleness.n_gaps} gaps measured (>= {thr.min_gaps})"))
            checks.append(Check("dds.no_deadman", no_trip and no_miss,
                                f"{staleness.deadman_trips} deadman trips, "
                                f"{staleness.misses} read misses (need 0/0)"))
            checks.append(Check("dds.staleness", p99_ok and max_ok,
                                f"p50 {staleness.p50_ms:.2f} / p99 {staleness.p99_ms:.2f} / "
                                f"max {staleness.max_ms:.2f} ms "
                                f"(p99 <= {thr.max_p99_gap_ms:.0f}, max <= {thr.max_gap_ms:.0f})"))
            dds_ok = window_ok and no_miss and no_trip and p99_ok and max_ok
        else:
            checks.append(Check("dds.staleness", False, "DDS staleness requested but not measured"))

    if dds_requested:
        go = iface_up and subnet_pass and ping_ok and dds_ok
        note = ("GO — link, ping and DDS staleness all within thresholds"
                if go else "NO-GO — one or more checks failed (see above)")
    else:
        go = False
        note = ("NO-GO for an untethered run: ping-only is insufficient. Re-run with --dds "
                "to measure LowState staleness (the real 50 Hz metric) over a sustained "
                "window." + ("" if (iface_up and subnet_pass and ping_ok)
                             else " Link/ping checks also did not fully pass."))
    return Verdict(go=go, note=note, checks=checks)


# ---- system probes (thin wrappers over subprocess/sysfs) -----------------------
def run_ping(host: str, count: int, interval: float, per_pkt_timeout: float = 1.0) -> str:
    """Run `ping` and return its stdout. Non-zero exit (e.g. loss) is fine — we parse output."""
    cmd = ["ping", "-n", "-c", str(count), "-i", str(interval),
           "-W", str(per_pkt_timeout), host]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=count * interval + per_pkt_timeout + 15.0)
    return proc.stdout + proc.stderr


def iface_operstate(iface: str) -> Optional[str]:
    """Read /sys/class/net/<iface>/operstate ('up','down','dormant',...) or None if absent."""
    p = f"/sys/class/net/{iface}/operstate"
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return None


def iface_ipv4(iface: str) -> Optional[tuple]:
    """(addr, prefixlen) of the iface's first IPv4, or None. Uses `ip -o -4 addr`."""
    try:
        out = subprocess.run(["ip", "-o", "-4", "addr", "show", "dev", iface],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_ip_o_addr(out)


def resolve_host(host: str) -> Optional[str]:
    """Resolve a hostname to an IPv4 string (returns host unchanged if already an IP);
    None if it cannot be resolved."""
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    try:
        return socket.gethostbyname(host)
    except OSError:
        return None


def _read_fn_from_subscriber(sub) -> Callable[[float], object]:
    """Wrap a unitree ChannelSubscriber into read_fn(timeout_s) -> msg|None."""
    return lambda timeout_s: sub.Read(timeout_s)


def make_lowstate_read_fn(iface: str) -> Callable[[float], object]:
    """Init CycloneDDS on `iface` and subscribe to rt/lowstate, reusing the SAME init the
    runtime uses (pipeline.deploy_runtime.make_dds / lowstate_subscriber). READ-ONLY.

    Imported lazily so the tests and the link/ping path never need the unitree SDK."""
    import pipeline.deploy_runtime as dr
    dr.make_dds(iface)
    sub = dr.lowstate_subscriber()
    return _read_fn_from_subscriber(sub)


# ---- reporting -----------------------------------------------------------------
def format_report(iface: str, host: str, ping: Optional[PingStats],
                  staleness: Optional[StalenessStats], verdict: Verdict,
                  thr: Thresholds) -> str:
    L = []
    L.append(f"WIRELESS PREFLIGHT  iface={iface}  robot={host}  "
             f"(50 Hz control, deadman {DEADMAN_S:.1f}s, wired p95 ~{WIRED_STALENESS_P95_MS:.2f} ms)")
    for c in verdict.checks:
        L.append(f"  [{'PASS' if c.ok else 'FAIL'}] {c.name:<18} {c.detail}")
    if staleness is not None:
        L.append(f"  staleness window: {staleness.n_samples} fresh samples, "
                 f"{staleness.n_gaps} gaps, mean {staleness.mean_ms:.2f} ms")
    L.append("")
    L.append(("GO" if verdict.go else "NO-GO") + " — " + verdict.note)
    return "\n".join(L)


def _verdict_dict(iface, host, ping, staleness, verdict) -> dict:
    def sd(s):
        if s is None:
            return None
        d = s.__dict__.copy()
        d.pop("gaps_ms", None)          # keep the JSON compact
        return d
    return {
        "iface": iface, "host": host, "go": verdict.go, "note": verdict.note,
        "checks": [c.__dict__ for c in verdict.checks],
        "ping": (ping.__dict__ if ping else None),
        "staleness": sd(staleness),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Wireless show preflight (link + ping + DDS LowState staleness) -> GO/NO-GO.")
    ap.add_argument("host", nargs="?", default=os.environ.get("ROBOT_HOST"),
                    help="robot address ON THE WIFI (its router-assigned IP/hostname), "
                         "or set ROBOT_HOST. NOT the wired 192.168.123.x address.")
    ap.add_argument("--iface", default=os.environ.get("ROBOT_IFACE"),
                    help="laptop WIFI interface (or set ROBOT_IFACE) — the same iface the "
                         "runtime will use for the wireless run.")
    ap.add_argument("--count", type=int, default=100, help="ping count (default 100)")
    ap.add_argument("--interval", type=float, default=0.25,
                    help="ping interval s (default 0.25; <0.2 needs root)")
    ap.add_argument("--dds", action="store_true",
                    help="also measure rt/lowstate staleness (the real 50 Hz metric; run in "
                         "the `tv` env). Without this the verdict can only be NO-GO.")
    ap.add_argument("--dds-secs", type=float, default=30.0,
                    help="staleness measurement window seconds (default 30)")
    ap.add_argument("--dds-poll", type=float, default=0.05,
                    help="per-read poll timeout s for the staleness sampler (default 0.05)")
    ap.add_argument("--deadman-s", type=float, default=DEADMAN_S,
                    help="comms-loss deadman seconds (default 0.5, matches the runtime)")
    ap.add_argument("--max-loss", type=float, default=None, help="override max packet loss %%")
    ap.add_argument("--max-avg-ms", type=float, default=None, help="override max avg ping RTT ms")
    ap.add_argument("--max-p99-ms", type=float, default=None, help="override max DDS p99 gap ms")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a text report")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.iface:
        print("ERROR: no wifi interface — pass --iface or set ROBOT_IFACE "
              "(e.g. `ip -o link` / `nmcli device` to find it).", file=sys.stderr)
        return 2
    if not args.host:
        print("ERROR: no robot host — pass the robot's WIFI address or set ROBOT_HOST.",
              file=sys.stderr)
        return 2

    thr = Thresholds()
    if args.max_loss is not None:
        thr.max_loss_pct = args.max_loss
    if args.max_avg_ms is not None:
        thr.max_avg_rtt_ms = args.max_avg_ms
    if args.max_p99_ms is not None:
        thr.max_p99_gap_ms = args.max_p99_ms

    # 1. LINK
    operstate = iface_operstate(args.iface)
    iface_up = operstate == "up"
    ipv4 = iface_ipv4(args.iface)
    host_ip = resolve_host(args.host)
    subnet_ok: Optional[bool] = None
    if ipv4 is not None and host_ip is not None:
        subnet_ok = same_subnet(ipv4[0], ipv4[1], host_ip)

    # 2. PING
    ping: Optional[PingStats] = None
    try:
        ping = parse_ping(run_ping(args.host, args.count, args.interval))
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        print(f"WARNING: ping failed: {e}", file=sys.stderr)

    # 3. STALENESS (opt-in; the real metric)
    staleness: Optional[StalenessStats] = None
    if args.dds:
        try:
            read_fn = make_lowstate_read_fn(args.iface)
            staleness = measure_staleness(
                read_fn, poll_timeout_s=args.dds_poll, deadman_s=args.deadman_s,
                duration_s=args.dds_secs)
        except Exception as e:  # noqa: BLE001 - SDK/DDS failure must not crash the report
            print(f"WARNING: DDS staleness measurement failed: {e}", file=sys.stderr)

    verdict = evaluate(ping, staleness, thr, iface_up=iface_up, subnet_ok=subnet_ok,
                       dds_requested=args.dds)

    if args.json:
        print(json.dumps(_verdict_dict(args.iface, args.host, ping, staleness, verdict), indent=2))
    else:
        print(format_report(args.iface, args.host, ping, staleness, verdict, thr))
    return 0 if verdict.go else 1


if __name__ == "__main__":
    raise SystemExit(main())
