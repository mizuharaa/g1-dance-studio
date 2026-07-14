"""Read-only observability for the cloud GPU box, surfaced in the app's System panel.

Answers the questions the user kept having to ask Claude — "is the GPU running, how's
training going, how much has it cost" — by reading the box directly (nvidia-smi, tmux,
training logs) over the existing SSH transport and estimating accrued GreenNode cost.

Everything here is READ-ONLY: it never launches or kills training. Box calls use short
timeouts and the last good snapshot is cached, so a slow/unreachable box degrades the
panel to "stale" rather than hanging the UI.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone

from . import cloud

# ---- billing config ----------------------------------------------------------
# Overridable via a "billing" block in .secrets/cloud.json. Defaults encode the
# confirmed GreenNode economics (PROJECT_STATE decision log 2026-07-03):
#   base 16,080,632 VND/month ÷ 730 h × 0.75 (25% internal) × 1.10 (10% VAT).
DEFAULT_BILLING = {
    "created_at": "2026-07-03T15:00:00+00:00",  # instance creation (billing start)
    "rate_vnd_per_hour": round(16_080_632 / 730 * 0.75 * 1.10, 2),  # ≈ 18,170
    "cap_vnd": 1_500_000,
    "usd_per_vnd": 1 / 25_800,
}

# One combined shell command → one SSH round-trip. Sentinels delimit the sections.
_GATHER_CMD = r"""
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null || echo NO_GPU
echo '@@TMUX@@'
(export PATH=/workspace/notebook-data/bin:$PATH; tmux ls 2>/dev/null || echo NONE)
echo '@@STATUS@@'
for f in /workspace/notebook-data/jobs/*.status.json; do [ -e "$f" ] || continue; echo "@@FILE $(basename "$f" .status.json)@@"; cat "$f" 2>/dev/null; echo; done
echo '@@LOGS@@'
for f in /workspace/notebook-data/jobs/*.log; do [ -e "$f" ] || continue; case "$(basename "$f")" in train*|*train*) echo "@@FILE $(basename "$f" .log)@@"; grep -E 'Learning iteration|Mean reward:|Mean episode length:|Time elapsed|ETA:|Iteration time|wandb.ai' "$f" 2>/dev/null | tail -24;; esac; done
[ -e /workspace/notebook-data/attempt3.out ] && { echo "@@FILE train-attempt3@@"; grep -E 'Learning iteration|Mean reward:|Mean episode length:|Time elapsed|ETA:|Iteration time|wandb.ai' /workspace/notebook-data/attempt3.out 2>/dev/null | tail -24; }
echo '@@PROC@@'
ps -eo args= 2>/dev/null | grep -cE '[s]im2real_task_v[0-9]|[t]rain_sim2real'
""".strip()


# ---- pure parsers (unit-tested; no box needed) -------------------------------

def parse_gpu(line: str) -> dict | None:
    """Parse one nvidia-smi CSV line → dict, or None if no GPU / unparseable."""
    line = (line or "").strip()
    if not line or line == "NO_GPU":
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 6:
        return None

    def num(s: str) -> float:
        m = re.search(r"[-+]?\d*\.?\d+", s)
        return float(m.group()) if m else 0.0

    util = num(parts[0])
    return {
        "utilization_pct": util,
        "memory_util_pct": num(parts[1]),
        "memory_used_mib": num(parts[2]),
        "memory_total_mib": num(parts[3]),
        "power_w": num(parts[4]),
        "temperature_c": num(parts[5]),
        "busy": util >= 20,  # a fresh 4090 idles near 0%; >=20% = real work
    }


def _hms_to_s(h: str, m: str, s: str) -> int:
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_job_log(name: str, text: str) -> dict:
    """Pull latest iteration / max, reward, episode length, ETA, elapsed, per-iter time
    and W&B url from a log tail (rsl_rl on-policy runner format)."""
    info: dict = {"name": name, "iteration": None, "max_iteration": None,
                  "mean_reward": None, "mean_episode_length": None, "wandb_url": None,
                  "eta_s": None, "elapsed_s": None, "iteration_time_s": None}
    for m in re.finditer(r"Learning iteration\s+(\d+)\s*/\s*(\d+)", text):
        info["iteration"], info["max_iteration"] = int(m.group(1)), int(m.group(2))
    rewards = re.findall(r"Mean reward:\s*([-+]?\d*\.?\d+)", text)
    if rewards:
        info["mean_reward"] = float(rewards[-1])
    eps = re.findall(r"Mean episode length:\s*([-+]?\d*\.?\d+)", text)
    if eps:
        info["mean_episode_length"] = float(eps[-1])
    # rsl_rl prints "Time elapsed: H:MM:SS", "ETA: H:MM:SS" (current train() call — i.e.
    # this curriculum STAGE), and "Iteration time: N.NNs". Take the most recent of each.
    el = re.findall(r"Time elapsed:\s*(\d+):(\d+):(\d+)", text)
    if el:
        info["elapsed_s"] = _hms_to_s(*el[-1])
    eta = re.findall(r"ETA:\s*(\d+):(\d+):(\d+)", text)
    if eta:
        info["eta_s"] = _hms_to_s(*eta[-1])
    it = re.findall(r"Iteration time:\s*([\d.]+)\s*s", text)
    if it:
        info["iteration_time_s"] = float(it[-1])
    urls = re.findall(r"https?://(?:\w+\.)?wandb\.ai/\S+", text)
    if urls:
        info["wandb_url"] = urls[-1].rstrip(".,)")
    if info["iteration"] and info["max_iteration"]:
        info["progress"] = round(info["iteration"] / info["max_iteration"], 4)
    return info


def compute_cost(billing: dict, now: float | None = None) -> dict:
    """Accrued GreenNode cost from instance-creation to now (billing runs to deletion)."""
    now = time.time() if now is None else now
    b = {**DEFAULT_BILLING, **(billing or {})}
    try:
        created = datetime.fromisoformat(b["created_at"]).timestamp()
    except (ValueError, TypeError):
        created = datetime.fromisoformat(DEFAULT_BILLING["created_at"]).timestamp()
    # Billing runs creation→DELETION. Once the box is deleted, stamp
    # billing["deleted_at"] so cost stops accruing (audit: it accrued forever,
    # permanently over-reporting spend). We stop only on an EXPLICIT deletion time,
    # not on a transient unreachable uplink (which is not deletion).
    end = now
    deleted_at = b.get("deleted_at")
    if deleted_at:
        try:
            end = min(end, datetime.fromisoformat(deleted_at).timestamp())
        except (ValueError, TypeError):
            pass
    hours = max(0.0, (end - created) / 3600)
    rate = float(b["rate_vnd_per_hour"])
    cap = float(b["cap_vnd"])
    vnd = hours * rate
    return {
        "hours": round(hours, 2),
        "rate_vnd_per_hour": round(rate, 2),
        "accrued_vnd": round(vnd, 0),
        "accrued_usd": round(vnd * float(b["usd_per_vnd"]), 2),
        "cap_vnd": cap,
        "cap_fraction": round(vnd / cap, 4) if cap else None,
        "over_cap": vnd >= cap if cap else False,
    }


def parse_gather(raw: str) -> dict:
    """Split the combined gather output into gpu / tmux / jobs sections."""
    gpu_txt, _, rest = raw.partition("@@TMUX@@")
    tmux_txt, _, rest = rest.partition("@@STATUS@@")
    status_txt, _, rest2 = rest.partition("@@LOGS@@")
    logs_txt, _, proc_txt = rest2.partition("@@PROC@@")

    gpu = parse_gpu(gpu_txt.strip().splitlines()[0] if gpu_txt.strip() else "")
    sessions = [ln.split(":")[0] for ln in tmux_txt.strip().splitlines()
                if ln.strip() and ln.strip() != "NONE" and ":" in ln]
    # True process check — the runner uses nohup/setsid (no tmux on this box), so tmux-session
    # liveness alone always reads "finished". A live train_sim2real proc means training is running.
    try:
        training_active = int(proc_txt.strip().splitlines()[0]) > 0
    except (ValueError, IndexError):
        training_active = False

    statuses: dict[str, dict] = {}
    parts = re.split(r"@@FILE (\S+)@@", status_txt)
    for name, body in zip(parts[1::2], parts[2::2]):
        try:
            statuses[name] = json.loads(body.strip())
        except (json.JSONDecodeError, ValueError):
            continue

    def _live(name: str) -> bool:
        # tmux sessions on the box are named "job-<name>"; a job is genuinely
        # running iff its session exists. A status.json state=="running" is NOT
        # trusted on its own: a SIGKILL'd job never writes its terminal status,
        # so a stale log/"running" lingers with no session (this is exactly what
        # made retired jobs show as "Active Training" forever).
        return f"job-{name}" in sessions or name in sessions or training_active

    def _state(name: str, live: bool, st: dict) -> str:
        if live:
            return "running"
        s = st.get("state")
        if s in ("done", "failed"):
            return s
        if s == "running":       # claims running but no session → was killed
            return "stopped"
        return "finished"        # log exists, no session, no clean terminal status

    jobs: list[dict] = []
    lparts = re.split(r"@@FILE (\S+)@@", logs_txt)
    for name, body in zip(lparts[1::2], lparts[2::2]):
        job = parse_job_log(name, body)
        st = statuses.get(name) or {}
        live = _live(name)
        job["live"] = live
        job["running"] = live  # back-compat alias, now tied to true liveness
        job["state"] = _state(name, live, st)
        job["started_at"] = st.get("started_at")
        jobs.append(job)
    # A status-only job (no train log — e.g. a finished gvhmr/install) is worth
    # surfacing only while it's still genuinely running.
    for name, st in statuses.items():
        if name not in {j["name"] for j in jobs} and _live(name):
            jobs.append({"name": name, "state": "running", "live": True,
                         "running": True, "started_at": st.get("started_at")})

    return {"gpu": gpu, "tmux_sessions": sessions, "jobs": jobs,
            "training_active": training_active}


# ---- live gather (cached, degrades gracefully) -------------------------------

_last_good: dict = {}


def augment_job_plan(job: dict, plan: dict) -> dict:
    """Add whole-training fields from a `training_plan` config block:
      total_eta_s = seconds to the ENTIRE curriculum finishing = remaining iters to the
                    final target × current per-iter time + the fixed verify-chain seconds.
                    (The per-stage `eta_s` from rsl_rl only covers the CURRENT stage.)
      stage / total_stages = which curriculum stage we're in, from stage_boundaries.
    Returns the job dict (mutated). No-ops gracefully when the plan or live values are absent."""
    if not plan:
        return job
    it, itt = job.get("iteration"), job.get("iteration_time_s")
    final = plan.get("final_target_iter")
    if final and it and itt:
        remaining = max(0, int(final) - int(it))
        job["total_eta_s"] = round(remaining * float(itt) + float(plan.get("verify_seconds", 0)))
    bounds = plan.get("stage_boundaries") or []
    if bounds and it:
        job["stage"] = next((i + 1 for i, b in enumerate(bounds) if it <= b), len(bounds))
        job["total_stages"] = len(bounds)
    return job


def snapshot(timeout: int = 20) -> dict:
    """Read the box once and assemble the full System-panel payload.

    Never raises: on any failure returns the last good snapshot marked stale, or an
    unreachable placeholder. Cost is always computed locally (no box needed)."""
    global _last_good
    cfg = cloud.load_config()
    cost = compute_cost(cfg.get("billing", {}))
    out: dict = {"checked_at": time.time(), "reachable": False, "stale": False,
                 "gpu": None, "jobs": [], "tmux_sessions": [], "cost": cost,
                 "detail": ""}
    if not cfg.get("transport"):
        out["detail"] = "cloud box not configured (Studio → Cloud GPU)"
        return out
    try:
        rc, stdout, stderr = cloud.run(_GATHER_CMD, timeout=timeout, cfg=cfg)
        if rc != 0 and not stdout.strip():
            raise RuntimeError((stderr or "gather failed").strip()[-200:])
        parsed = parse_gather(stdout)
        out.update(parsed)
        plan = cfg.get("training_plan") or {}
        for job in out.get("jobs", []):
            augment_job_plan(job, plan)
        out["reachable"] = True
        out["detail"] = "ok"
        _last_good = {**out}
    except Exception as e:  # noqa: BLE001 — panel must never hang/crash the UI
        out["detail"] = f"box unreachable: {type(e).__name__}: {e}"[:200]
        if _last_good:
            out["gpu"] = _last_good.get("gpu")
            out["jobs"] = _last_good.get("jobs", [])
            out["tmux_sessions"] = _last_good.get("tmux_sessions", [])
            out["stale"] = True
            out["last_good_at"] = _last_good.get("checked_at")
    return out
