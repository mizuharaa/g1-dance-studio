"""Offline tests for the software remote e-stop (pipeline/remote_estop.py).

Byte layout under test comes from the official SDK example
(wireless_controller.py): byte2 bit5 = L2, byte3 bit1 = B.
"""
import pytest

from pipeline.remote_estop import (RemoteKill, RemoteKillRequested,
                                   chord_down, decode_buttons)


def _buf(b2=0, b3=0):
    buf = bytearray(40)
    buf[2], buf[3] = b2, b3
    return bytes(buf)


L2 = 1 << 5      # byte2
B = 1 << 1       # byte3


def test_decode_all_zero():
    assert all(v == 0 for v in decode_buttons(_buf()).values())


def test_decode_l2_and_b_bits():
    btn = decode_buttons(_buf(b2=L2, b3=B))
    assert btn["L2"] == 1 and btn["B"] == 1
    assert sum(btn.values()) == 2


def test_decode_every_button_isolated():
    names_b2 = ["R1", "L1", "Start", "Select", "R2", "L2", "F1", "F3"]
    names_b3 = ["A", "B", "X", "Y", "Up", "Right", "Down", "Left"]
    for i, name in enumerate(names_b2):
        btn = decode_buttons(_buf(b2=1 << i))
        assert btn[name] == 1 and sum(btn.values()) == 1
    for i, name in enumerate(names_b3):
        btn = decode_buttons(_buf(b3=1 << i))
        assert btn[name] == 1 and sum(btn.values()) == 1


def test_short_or_bad_buffer_is_all_zero_not_crash():
    assert all(v == 0 for v in decode_buttons(b"").values())
    assert all(v == 0 for v in decode_buttons(None).values())


def test_chord_requires_both():
    assert not chord_down(_buf(b2=L2))
    assert not chord_down(_buf(b3=B))
    assert chord_down(_buf(b2=L2, b3=B))


def test_debounce_fires_only_after_consecutive_ticks():
    kill = RemoteKill(ticks=3)
    chord = _buf(b2=L2, b3=B)
    assert not kill.update(chord)
    assert not kill.update(chord)
    assert kill.update(chord)          # 3rd consecutive -> fire
    assert kill.fired_at is not None


def test_release_resets_debounce():
    kill = RemoteKill(ticks=3)
    chord = _buf(b2=L2, b3=B)
    kill.update(chord); kill.update(chord)
    kill.update(_buf())                # released
    assert not kill.update(chord)
    assert not kill.update(chord)
    assert kill.update(chord)


def test_check_raises():
    kill = RemoteKill(ticks=1)
    with pytest.raises(RemoteKillRequested):
        kill.check(_buf(b2=L2, b3=B))


def test_other_buttons_do_not_fire():
    kill = RemoteKill(ticks=1)
    # every single button alone, plus a non-kill chord (L2+A)
    for b2, b3 in [(1 << i, 0) for i in range(8)] + [(0, 1 << i) for i in range(8)] \
                  + [(L2, 1 << 0)]:
        assert not kill.update(_buf(b2=b2, b3=b3))
