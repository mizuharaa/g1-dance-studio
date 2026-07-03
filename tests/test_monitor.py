"""Tests for pipeline/monitor.py pure parsing + cost math (no live box needed)."""
from datetime import datetime, timezone

from pipeline import monitor


# ---- parse_gpu ---------------------------------------------------------------

def test_parse_gpu_busy():
    g = monitor.parse_gpu("80 %, 11 %, 4676 MiB, 24564 MiB, 213.12 W, 74")
    assert g["utilization_pct"] == 80
    assert g["memory_used_mib"] == 4676 and g["memory_total_mib"] == 24564
    assert g["power_w"] == 213.12 and g["temperature_c"] == 74
    assert g["busy"] is True


def test_parse_gpu_idle_not_busy():
    assert monitor.parse_gpu("3 %, 0 %, 12 MiB, 24564 MiB, 21 W, 35")["busy"] is False


def test_parse_gpu_no_gpu_or_garbage():
    assert monitor.parse_gpu("NO_GPU") is None
    assert monitor.parse_gpu("") is None
    assert monitor.parse_gpu("only, three, fields") is None


# ---- parse_job_log -----------------------------------------------------------

LOG = """
                         Learning iteration 1382/30000
                            Mean reward: 14.28
                    Mean episode length: 375.33
                         Learning iteration 1383/30000
                            Mean reward: 13.79
                    Mean episode length: 382.66
wandb: 🚀 View run at https://wandb.ai/luong-alois-vng-group/mjlab/runs/40g4byo3
"""


def test_parse_job_log_latest_values():
    j = monitor.parse_job_log("train-dance1-seg", LOG)
    assert j["iteration"] == 1383 and j["max_iteration"] == 30000
    assert j["mean_reward"] == 13.79            # last, not first
    assert j["mean_episode_length"] == 382.66
    assert j["wandb_url"].endswith("/runs/40g4byo3")
    assert abs(j["progress"] - 1383 / 30000) < 1e-9


def test_parse_job_log_empty():
    j = monitor.parse_job_log("x", "")
    assert j["iteration"] is None and j["mean_reward"] is None
    assert "progress" not in j


def test_parse_job_log_negative_reward():
    j = monitor.parse_job_log("x", "Mean reward: -2.22\n")
    assert j["mean_reward"] == -2.22


# ---- compute_cost ------------------------------------------------------------

def test_compute_cost_basic():
    created = "2026-07-03T15:00:00+00:00"
    now = datetime(2026, 7, 3, 20, 0, 0, tzinfo=timezone.utc).timestamp()  # +5h
    c = monitor.compute_cost({"created_at": created, "rate_vnd_per_hour": 18170,
                              "cap_vnd": 1_500_000, "usd_per_vnd": 1 / 25800}, now=now)
    assert abs(c["hours"] - 5.0) < 1e-6
    assert c["accrued_vnd"] == 90850              # 5 * 18170
    assert c["over_cap"] is False
    assert abs(c["cap_fraction"] - 90850 / 1_500_000) < 1e-4  # displayed to 4 dp


def test_compute_cost_over_cap():
    created = "2026-07-03T00:00:00+00:00"
    now = datetime(2026, 7, 7, 0, 0, 0, tzinfo=timezone.utc).timestamp()  # 96h
    c = monitor.compute_cost({"created_at": created, "rate_vnd_per_hour": 18170,
                              "cap_vnd": 1_500_000}, now=now)
    assert c["over_cap"] is True and c["cap_fraction"] > 1


def test_compute_cost_defaults_and_bad_date():
    c = monitor.compute_cost({"created_at": "not-a-date"})
    assert c["rate_vnd_per_hour"] > 0 and c["hours"] >= 0  # falls back, never crashes


# ---- parse_gather ------------------------------------------------------------

GATHER = """80 %, 11 %, 4676 MiB, 24564 MiB, 213.12 W, 74
@@TMUX@@
job-train-dance1-seg: 1 windows (created ...)
job-train-thriller-a1: 1 windows (created ...)
@@STATUS@@
@@FILE train-dance1-seg@@
{"name":"train-dance1-seg","state":"running","rc":null}
@@LOGS@@
@@FILE train-dance1-seg@@
                         Learning iteration 1382/30000
                            Mean reward: 14.28
                    Mean episode length: 375.33
"""


def test_parse_gather_full():
    g = monitor.parse_gather(GATHER)
    assert g["gpu"]["utilization_pct"] == 80
    assert "job-train-dance1-seg" in g["tmux_sessions"]
    assert len(g["jobs"]) == 1
    j = g["jobs"][0]
    assert j["name"] == "train-dance1-seg" and j["running"] is True
    assert j["iteration"] == 1382 and j["mean_reward"] == 14.28


def test_parse_gather_box_no_gpu_no_jobs():
    g = monitor.parse_gather("NO_GPU\n@@TMUX@@\nNONE\n@@STATUS@@\n@@LOGS@@\n")
    assert g["gpu"] is None and g["jobs"] == [] and g["tmux_sessions"] == []
