"""Decisive root-cause diagnosis of the v10 drift + jitter, so we fix the RIGHT
thing "once and for all" (user, 2026-07-20 — the real robot already did 70-80%
of the dance at full speed before the hardware fault, so sim drift/jitter is
suspected to be largely a SIM-fidelity artifact, not a policy defect).

Three experiments, all on the v10 policy, all laptop-side (no GPU):

  1. DRIFT vs FLOOR FRICTION. The BeyondMimic training env randomizes foot
     friction over static (0.3, 1.6) / dynamic (0.3, 1.2) about a nominal 1.0.
     The gate runs that DR, so its drift_MAX (3.5 m) is dominated by the
     slippery tail (mu->0.3) — floors far more slippery than any real show
     surface (rubber on hardwood ~ 0.8-1.1). We sweep the floor friction and
     roll out the FULL dance at each; if drift collapses as friction rises to a
     realistic ~1.0, the "drift problem" is a gate/eval-friction artifact and the
     honest fix is to gate drift at show-floor friction, not the DR mean.
  2. JITTER spectrum. Fraction of base-position and action-target variance ABOVE
     5 Hz (jitter band) vs below (real dance content is < ~3 Hz). Tells us if the
     visible shake is high-frequency chatter (fixable: contact softness / action
     smoothing) or just legitimate dance sway mislabeled as jitter.
  3. REAL-ROBOT baseline. The same jitter metric on the actual hardware
     telemetry (data/telemetry/*ground-run-legodom.npz, the 70-80% live runs):
     what does REAL command/joint smoothness look like? That is the target the
     sim visualization should match.

Writes drift_vs_friction.json + jitter.json + a short verdict to stdout, all
committed with raw numbers (measurement-discipline rule).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent
POLICY = ROOT / "exports/train-thriller_v10spd-0720"
FAITHFUL = ROOT / "tools/assets/g1_faithful/g1_mjlab_faithful.xml"
FRICTIONS = [0.3, 0.5, 0.7, 1.0, 1.3]
CONTROL_HZ = 50.0
STEPS = 2500                    # full ~49 s dance + a little


def _friction_variant(mu: float) -> Path:
    """Write a temp copy of the faithful model with the FLOOR friction pinned to
    mu at high priority (priority makes the floor's friction win the contact
    pair, overriding the feet's default 1.0)."""
    xml = FAITHFUL.read_text()
    old = '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>'
    new = (f'<geom name="floor" size="0 0 0.05" type="plane" material="groundplane" '
           f'friction="{mu} 0.005 0.0001" priority="1"/>')
    assert old in xml, "floor geom line not found — model changed"
    p = Path(tempfile.gettempdir()) / f"g1_faithful_mu{mu}.xml"
    p.write_text(xml.replace(old, new))
    return p


def drift_vs_friction() -> dict:
    from tools.sim_sandbox import run_sandbox
    rows = {}
    for mu in FRICTIONS:
        xml = _friction_variant(mu)
        out, _, _ = run_sandbox(POLICY, steps=STEPS, latency_ms=0, xml=xml, tether_kp=0)
        bp = np.asarray(out["base_pos"])
        fell = out["fell_at_tick"]
        n = len(bp) if fell is None else fell
        n = max(n, 2)
        disp = bp[:n] - bp[0]
        drift_xy = np.linalg.norm(disp[:, :2], axis=1)
        rows[f"mu_{mu}"] = {
            "friction": mu,
            "fell_at_s": (None if fell is None else round(fell / CONTROL_HZ, 1)),
            "drift_final_m": round(float(drift_xy[-1]), 3),
            "drift_max_m": round(float(drift_xy.max()), 3),
            "drift_mean_m": round(float(drift_xy.mean()), 3),
            "steps_alive": int(n),
        }
        print(f"  mu={mu}: drift final {rows[f'mu_{mu}']['drift_final_m']} m, "
              f"max {rows[f'mu_{mu}']['drift_max_m']} m, "
              f"fell_at {rows[f'mu_{mu}']['fell_at_s']}")
    return rows


def _hf_fraction(x: np.ndarray, fs: float, cut: float = 5.0) -> float:
    """Fraction of the signal's zero-mean variance above `cut` Hz (jitter band).
    x: (T,) or (T,D) — averaged over D."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    x = x - x.mean(0, keepdims=True)
    T = len(x)
    if T < 16:
        return float("nan")
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    P = np.abs(np.fft.rfft(x, axis=0)) ** 2
    tot = P.sum(0)
    hf = P[freqs > cut].sum(0)
    tot = np.where(tot < 1e-12, 1e-12, tot)
    return float(np.mean(hf / tot))


def jitter_spectrum() -> dict:
    from tools.sim_sandbox import run_sandbox
    xml = _friction_variant(1.0)                 # realistic show floor
    out, _, _ = run_sandbox(POLICY, steps=STEPS, latency_ms=0, xml=xml, tether_kp=0)
    bp = np.asarray(out["base_pos"])
    act = np.asarray(out["target"])              # commanded joint targets
    q = np.asarray(out["q"])
    sim = {
        "base_xy_hf_frac_above5hz": round(_hf_fraction(bp[:, :2], CONTROL_HZ), 4),
        "base_z_hf_frac_above5hz": round(_hf_fraction(bp[:, 2], CONTROL_HZ), 4),
        "target_hf_frac_above5hz": round(_hf_fraction(act, CONTROL_HZ), 4),
        "q_hf_frac_above5hz": round(_hf_fraction(q, CONTROL_HZ), 4),
        "base_xy_std_m": round(float(bp[:, :2].std()), 4),
    }
    print("  SIM jitter:", sim)

    # real hardware baseline — the 70-80% live runs
    reals = {}
    for name in ["20260710-145111", "20260708-192839"]:
        p = ROOT / f"data/telemetry/{name}_ground-run-legodom.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        tgt = np.asarray(d["target"], float)
        qh = np.asarray(d["q"], float)
        gyro = np.asarray(d["gyro"], float) if "gyro" in d else None
        reals[name] = {
            "target_hf_frac_above5hz": round(_hf_fraction(tgt, CONTROL_HZ), 4),
            "q_hf_frac_above5hz": round(_hf_fraction(qh, CONTROL_HZ), 4),
            "gyro_hf_frac_above5hz": (round(_hf_fraction(gyro, CONTROL_HZ), 4)
                                      if gyro is not None else None),
        }
        print(f"  REAL {name} jitter:", reals[name])
    return {"sim": sim, "real": reals}


def main() -> None:
    print("[1] drift vs floor friction (full dance, v10 policy):")
    dvf = drift_vs_friction()
    print("[2/3] jitter spectrum — sim vs real hardware:")
    jit = jitter_spectrum()
    (OUT / "drift_vs_friction.json").write_text(json.dumps(dvf, indent=2))
    (OUT / "jitter.json").write_text(json.dumps(jit, indent=2))

    real_mu = dvf.get("mu_1.0", {})
    slippery = dvf.get("mu_0.3", {})
    print("\n=== VERDICT ===")
    print(f"drift @ show-floor mu=1.0: {real_mu.get('drift_final_m')} m final "
          f"({real_mu.get('drift_max_m')} m max) | @ mu=0.3 (DR tail): "
          f"{slippery.get('drift_final_m')} m final")
    print("wrote", OUT / "drift_vs_friction.json", "and", OUT / "jitter.json")


if __name__ == "__main__":
    main()
