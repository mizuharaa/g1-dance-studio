"""TickClock (Lane A): absolute-deadline pacing keeps schedule, overrun still damps."""
import time

import numpy as np
import pytest

from pipeline.deploy_runtime import TickClock


def test_paces_to_absolute_schedule():
    hz = 200.0
    clock = TickClock(hz=hz)
    start = time.perf_counter()
    for _ in range(40):
        t0 = time.perf_counter()
        clock.tick(t0)
    total = time.perf_counter() - start
    # 40 ticks @ 5 ms = 200 ms on an absolute schedule; relative pacing would drift longer.
    assert 0.18 <= total <= 0.30, total
    s = clock.stats()
    assert s["ticks"] == 40 and s["soft_overruns"] == 0
    assert s["work_ms"]["p99"] < s["dt_ms"]


def test_overrun_beyond_2dt_raises_for_damp():
    clock = TickClock(hz=50.0)
    t0 = time.perf_counter() - 0.05  # 50 ms of "work" > 2*dt (40 ms)
    with pytest.raises(RuntimeError, match="overrun"):
        clock.tick(t0)


def test_soft_overrun_counted_not_fatal():
    clock = TickClock(hz=50.0)
    clock.tick(time.perf_counter() - 0.025)  # 25 ms work: > dt, < 2*dt
    assert clock.soft_overruns == 1


def test_late_tick_does_not_burst_catch_up():
    clock = TickClock(hz=100.0)
    clock.tick(time.perf_counter())          # on time
    time.sleep(0.05)                          # miss several deadlines while "working"...
    clock.tick(time.perf_counter() - 0.015)   # ...but work itself only 15 ms (soft overrun)
    t0 = time.perf_counter()
    clock.tick(t0)                            # next tick must still get a full dt sleep
    assert time.perf_counter() - t0 >= 0.008
    assert np.asarray(clock.late_ms).max() > 0


def test_report_never_raises_and_returns_stats():
    clock = TickClock(hz=100.0)
    assert clock.report() == {}  # empty run
    clock.tick(time.perf_counter())
    s = clock.report()
    assert s["ticks"] == 1
