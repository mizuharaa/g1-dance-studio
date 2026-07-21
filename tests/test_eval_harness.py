"""Harness-v2 eval condition construction (audit F4) — CPU-only, no mjlab.

cloud/sim_gap_check.py's pure pieces (condition table, cfg builder,
realized-block extraction, horizon math) import without mjlab; these tests
prove, by construction:
  * `clean` has zero DR/RSI/delay/noise/push even when the play cfg is dirty,
  * one-factor rows touch exactly one knob (vs clean),
  * the horizon is exactly T = frames/fps (no +0.2 padding anywhere),
  * seeds are paired (same seed on every row, recorded),
  * harness_version: 2 is stamped and the §3.4 realized schema is exact,
  * heldout_eval and pick_checkpoint use the SAME builder / row names.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cloud"))

import heldout_eval as he  # noqa: E402
import pick_checkpoint as pc  # noqa: E402
import sim_gap_check as sgc  # noqa: E402

REALIZED_KEYS = {"seed", "play_mode", "startup_dr", "cmd_delay_steps",
                 "obs_delay_steps", "noise", "push"}  # CONVENTIONS §3.4, exact

ONE_FACTOR_ROWS = ("dr_nominal", "noise", "push", "cmd_delay20ms",
                   "cmd_delay40ms", "cmd_delay60ms", "cmd_delay80ms",
                   "obs_delay20ms", "obs_delay80ms")
COMPOSITE_ROWS = ("dr_delay20ms_push", "dr_delay40ms_push",
                  "dr_delay60ms_push", "dr_delay80ms_push")


# ---- fake mjlab-shaped cfgs (attribute-compatible; no mjlab needed) ---------

def _term(min_lag=0, max_lag=0):
    return SimpleNamespace(delay_min_lag=min_lag, delay_max_lag=max_lag,
                           delay_per_env=False)


def _actuator(min_lag=0, max_lag=0):
    return SimpleNamespace(delay_min_lag=min_lag, delay_max_lag=max_lag,
                           delay_hold_prob=0.8, delay_update_period=0,
                           delay_per_env_phase=True)


def _event(mode):
    return SimpleNamespace(mode=mode)


def make_play_cfg(dirty: bool = True):
    """A play cfg that (like the real stock one) still carries startup DR, and
    — when dirty — residual delays/noise/RSI a custom play cfg could leak.
    The builder must zero ALL of it explicitly."""
    terms = {name: _term() for name in
             sgc.DELAYED_OBS_TERMS + ("projected_gravity", "actions")}
    acts = [_actuator(), _actuator()]
    motion = SimpleNamespace(pose_range={}, velocity_range={},
                             sampling_mode="start", motion_file="")
    corruption = False
    if dirty:
        for t in terms.values():
            t.delay_min_lag, t.delay_max_lag = 1, 3
        acts = [_actuator(2, 6), _actuator(2, 6)]
        motion.pose_range = {"x": (-0.05, 0.05)}
        motion.velocity_range = {"x": (-0.5, 0.5)}
        motion.sampling_mode = "random"
        corruption = True
    return SimpleNamespace(
        seed=0,
        episode_length_s=int(1e9),
        commands={"motion": motion},
        observations={"actor": SimpleNamespace(terms=terms,
                                               enable_corruption=corruption)},
        # stock play cfg KEEPS these startup DR events (push_robot popped)
        events={"base_com": _event("startup"),
                "encoder_bias": _event("startup"),
                "foot_friction": _event("startup")},
        scene=SimpleNamespace(
            entities={"robot": SimpleNamespace(
                articulation=SimpleNamespace(actuators=acts))},
            num_envs=1),
    )


def make_donor_cfg():
    """A train (play=False) cfg: the DR/push event donor."""
    cfg = make_play_cfg(dirty=True)
    cfg.events = {
        "push_robot": _event("interval"),
        "base_com": _event("startup"),
        "encoder_bias": _event("startup"),
        "foot_friction": _event("startup"),
        "dr_pd_gains": _event("startup"),
        "dr_torso_mass": _event("startup"),
    }
    return cfg


def build(spec: sgc.ConditionSpec, seed: int = 91001, play=None, donor=None):
    return sgc.make_condition_cfg(
        play if play is not None else make_play_cfg(),
        seed=seed,
        cmd_delay_steps=spec.cmd_delay_steps,
        obs_delay_steps=spec.obs_delay_steps,
        noise=spec.noise,
        push=spec.push,
        startup_dr=spec.startup_dr,
        donor_train_cfg=donor if donor is not None else make_donor_cfg(),
    )


def realized(name: str, seed: int = 91001):
    return sgc.extract_realized(build(sgc.CONDITION_BY_NAME[name], seed=seed))


# ---- clean row: zero everything by construction -----------------------------

def test_clean_row_has_zero_dr_delay_noise_push_from_dirty_play_cfg():
    cfg = build(sgc.CONDITION_BY_NAME["clean"])
    r = sgc.extract_realized(cfg)
    assert r == {
        "seed": 91001,
        "play_mode": True,
        "startup_dr": False,
        "cmd_delay_steps": [0, 0],
        "obs_delay_steps": [0, 0],
        "noise": False,
        "push": False,
    }
    # mechanism, not just the summary: nothing survives in the cfg itself
    assert cfg.events == {}
    for act in cfg.scene.entities["robot"].articulation.actuators:
        assert (act.delay_min_lag, act.delay_max_lag) == (0, 0)
        assert act.delay_hold_prob == 0.0
    for term in cfg.observations["actor"].terms.values():
        assert (term.delay_min_lag, term.delay_max_lag) == (0, 0)
    assert cfg.observations["actor"].enable_corruption is False
    motion = cfg.commands["motion"]
    assert motion.pose_range == {} and motion.velocity_range == {}
    assert motion.sampling_mode == "start"


def test_builder_does_not_mutate_the_base_cfg():
    play = make_play_cfg()
    before = copy.deepcopy(play)
    build(sgc.CONDITION_BY_NAME["dr_delay40ms_push"], play=play)
    assert play.events.keys() == before.events.keys()
    assert play.observations["actor"].enable_corruption \
        == before.observations["actor"].enable_corruption
    assert [a.delay_max_lag for a in play.scene.entities["robot"].articulation.actuators] \
        == [a.delay_max_lag for a in before.scene.entities["robot"].articulation.actuators]


# ---- realized block schema (CONVENTIONS §3.4) -------------------------------

def test_realized_schema_is_exactly_conventions_3_4():
    r = realized("dr_delay40ms_push")
    assert set(r) == REALIZED_KEYS
    assert isinstance(r["seed"], int)
    for k in ("play_mode", "startup_dr", "noise", "push"):
        assert isinstance(r[k], bool)
    for k in ("cmd_delay_steps", "obs_delay_steps"):
        lo, hi = r[k]
        assert isinstance(lo, int) and isinstance(hi, int) and 0 <= lo <= hi


def test_realized_reads_actual_values_not_intent():
    # a cfg with leftover RSI + delays reports them (extraction is honest)
    cfg = make_play_cfg(dirty=True)
    r = sgc.extract_realized(cfg)
    assert r["play_mode"] is False
    assert r["cmd_delay_steps"] == [2, 6]
    assert r["obs_delay_steps"] == [1, 3]
    assert r["noise"] is True


# ---- one-factor and composite rows ------------------------------------------

def test_one_factor_rows_touch_exactly_one_knob():
    base = realized("clean")
    for name in ONE_FACTOR_ROWS:
        r = realized(name)
        diff = [k for k in REALIZED_KEYS - {"seed"} if r[k] != base[k]]
        assert diff and len(diff) == 1, f"{name} changed {diff}"


def test_cmd_delay_rows_set_constant_physics_step_lags():
    for name, steps in (("cmd_delay20ms", 4), ("cmd_delay40ms", 8),
                        ("cmd_delay60ms", 12), ("cmd_delay80ms", 16)):
        assert realized(name)["cmd_delay_steps"] == [steps, steps]


def test_obs_delay_rows_target_only_measured_terms():
    cfg = build(sgc.CONDITION_BY_NAME["obs_delay20ms"])
    terms = cfg.observations["actor"].terms
    for tname, term in terms.items():
        want = (1, 1) if tname in sgc.DELAYED_OBS_TERMS else (0, 0)
        assert (term.delay_min_lag, term.delay_max_lag) == want, tname


def test_composite_rows_carry_the_full_declared_stack():
    base = realized("clean")
    for name in COMPOSITE_ROWS:
        r = realized(name)
        assert r["startup_dr"] and r["noise"] and r["push"]
        assert r["obs_delay_steps"] == [0, 1]
        assert r["cmd_delay_steps"][0] == r["cmd_delay_steps"][1] > 0
        assert r["play_mode"] == base["play_mode"] is True  # RSI stays off


def test_dr_row_copies_only_startup_events_deeply():
    donor = make_donor_cfg()
    cfg = build(sgc.CONDITION_BY_NAME["dr_nominal"], donor=donor)
    expect = {k for k, ev in donor.events.items() if ev.mode == "startup"}
    assert set(cfg.events) == expect  # push_robot (interval) NOT included
    for k in expect:
        assert cfg.events[k] is not donor.events[k]  # deep copies


def test_push_row_uses_the_training_push_event():
    donor = make_donor_cfg()
    cfg = build(sgc.CONDITION_BY_NAME["push"], donor=donor)
    assert set(cfg.events) == {"push_robot"}
    assert cfg.events["push_robot"] is not donor.events["push_robot"]


def test_builder_refuses_dr_or_push_without_donor():
    play = make_play_cfg()
    with pytest.raises(ValueError):
        sgc.make_condition_cfg(play, seed=1, startup_dr=True, donor_train_cfg=None)
    with pytest.raises(ValueError):
        sgc.make_condition_cfg(play, seed=1, push=True, donor_train_cfg=None)
    donor_no_push = make_donor_cfg()
    donor_no_push.events.pop("push_robot")
    with pytest.raises(ValueError):
        sgc.make_condition_cfg(play, seed=1, push=True,
                               donor_train_cfg=donor_no_push)


# ---- paired seeds -----------------------------------------------------------

def test_seeds_are_paired_across_all_rows():
    for spec in sgc.CONDITIONS_V2:
        assert sgc.extract_realized(build(spec, seed=777))["seed"] == 777


# ---- exact horizon ----------------------------------------------------------

def test_horizon_is_exactly_frames_over_fps():
    eps, steps = sgc.exact_horizon(2464, 50.0)  # the v11 motion (audit F4)
    assert eps == 2464 / 50.0 and steps == 2464
    eps, steps = sgc.exact_horizon(300, 30.0)
    assert eps == 10.0 and steps == 500
    with pytest.raises(ValueError):
        sgc.exact_horizon(0, 50.0)


def test_no_padding_remains_in_either_evaluator():
    for mod in (sgc, he):
        src = Path(mod.__file__).read_text()
        assert "+ 0.2" not in src, f"{mod.__name__} still pads the horizon"
        assert "max_steps) + 100" not in src


# ---- harness_version stamp + payload ---------------------------------------

def test_gap_payload_is_stamped_harness_v2():
    assert sgc.HARNESS_VERSION == 2
    payload = sgc.gap_payload(
        task="T", checkpoint="c.pt", onnx="", motion_file="m.npz",
        episode_length_s=49.28, horizon_steps=2464, seeds=[91001],
        gate=None, conditions={"clean": {"realized": realized("clean")}},
    )
    assert payload["harness_version"] == 2
    assert payload["horizon_steps"] == 2464
    assert payload["seeds"] == [91001]
    assert "repeats" not in payload
    with_reps = sgc.gap_payload(
        task="T", checkpoint="c.pt", onnx="", motion_file="m.npz",
        episode_length_s=49.28, horizon_steps=2464, seeds=[91001, 91002],
        gate=None, conditions={}, repeats={"91002": {}},
    )
    assert with_reps["seeds"] == [91001, 91002] and "repeats" in with_reps


# ---- legacy --only selectors ------------------------------------------------

def test_legacy_only_selectors_map_to_v2_rows():
    assert sgc.resolve_only("nominal,delay40ms_push") == ["clean", "dr_delay40ms_push"]
    assert sgc.resolve_only("clean, noise") == ["clean", "noise"]
    with pytest.raises(SystemExit):
        sgc.resolve_only("bogus_row")
    for old, new in sgc.LEGACY_ROW_MAP.items():
        assert new in sgc.CONDITION_BY_NAME, (old, new)


# ---- cross-file consistency -------------------------------------------------

def test_heldout_eval_uses_the_same_builder():
    assert he.make_condition_cfg is sgc.make_condition_cfg
    assert he.extract_realized is sgc.extract_realized
    assert he.exact_horizon is sgc.exact_horizon
    assert he.HARNESS_VERSION == sgc.HARNESS_VERSION == 2


def test_pick_checkpoint_screens_v2_row_names():
    names = pc.SCREEN_ONLY.split(",")
    assert names == ["clean", "dr_delay40ms_push"]
    for n in names:
        assert n in sgc.CONDITION_BY_NAME
    # v2 gap.json scores...
    row = {"success_rate": 1.0, "drift": {"episode_max_p95_m": 0.4},
           "ankle_pitch": {"p95_abs": 10.0}}
    sc = pc._score({"harness_version": 2,
                    "conditions": {"clean": row, "dr_delay40ms_push": row}})
    assert sc["gate_passes"] == 5
    # ...and harness-v1 files still rescore via the fallback names
    sc_old = pc._score({"conditions": {"nominal": row, "delay40ms_push": row}})
    assert sc_old["gate_passes"] == 5


def test_condition_names_are_unique_and_table_is_honest():
    names = [c.name for c in sgc.CONDITIONS_V2]
    assert len(names) == len(set(names))
    clean = sgc.CONDITION_BY_NAME["clean"]
    assert clean.cmd_delay_steps == (0, 0) and clean.obs_delay_steps == (0, 0)
    assert not (clean.noise or clean.push or clean.startup_dr)
