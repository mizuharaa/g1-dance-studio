"""Connection layer for the cloud GPU box (GreenNode notebook).

Two transports, matching what GreenNode actually offers (docs/GREENNODE_SETUP.md):

  ssh      — host/port/user/key from the console's Connect dialog (plan A).
  jupyter  — a Jupyter Server base URL + token; either GreenNode's own server or
             the cloudflared quick-tunnel one-liner from the setup guide (plan B).
             Commands run through the Jupyter kernel websocket API.

Config lives in .secrets/cloud.json (gitignored, chmod 600). This module only
ever talks to the GPU box — the robot is out of scope by design.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from .config import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / ".secrets" / "cloud.json"

# GreenNode notebooks regenerate host keys on every stop/start (verified in
# docs research), so strict host-key checking would break on every restart.
SSH_BASE_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=8",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
]

DEFAULT_CONFIG = {"transport": "", "ssh": {}, "jupyter": {}}
_SECRET_KEYS = {"password", "token"}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))
    return {**DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text())}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(mode=0o700, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    os.chmod(CONFIG_PATH, 0o600)


def masked_config() -> dict:
    """Config safe to show in the UI: secrets replaced by '•set•' markers."""
    cfg = load_config()
    for section in ("ssh", "jupyter"):
        for k in list(cfg.get(section, {})):
            if k in _SECRET_KEYS and cfg[section][k]:
                cfg[section][k] = "•set•"
    return cfg


def update_config(patch: dict) -> dict:
    """Merge a UI patch into the stored config. Empty strings mean 'keep the
    stored value' for secret fields so re-saving the form never wipes them."""
    cfg = load_config()
    if patch.get("transport") is not None:
        cfg["transport"] = patch["transport"]
    for section in ("ssh", "jupyter"):
        for k, v in (patch.get(section) or {}).items():
            if v == "" and k in _SECRET_KEYS:
                continue
            cfg.setdefault(section, {})[k] = v
    save_config(cfg)
    return cfg


# ---- ssh transport -----------------------------------------------------------

def _ssh_argv(cfg: dict, command: str) -> list[str]:
    s = cfg.get("ssh", {})
    if not s.get("host"):
        raise ValueError("ssh transport not configured (missing host)")
    # password auth goes through sshpass and must not use BatchMode
    opts = [o for pair in zip(SSH_BASE_OPTS[::2], SSH_BASE_OPTS[1::2])
            for o in pair if not (s.get("password") and pair[1] == "BatchMode=yes")]
    argv = ["ssh", *opts]
    if s.get("port"):
        argv += ["-p", str(s["port"])]
    if s.get("key_path"):
        argv += ["-i", str(Path(s["key_path"]).expanduser())]
    if s.get("password"):
        # sshpass -e reads the password from the SSHPASS env var, NOT argv, so it
        # never appears in the process table / `ps` output (audit MEDIUM security).
        argv = ["sshpass", "-e", *argv]
    return [*argv, f"{s.get('user', 'root')}@{s['host']}", command]


def _run_ssh(cfg: dict, command: str, timeout: int = 30) -> tuple[int, str, str]:
    argv = _ssh_argv(cfg, command)
    if argv[0] == "sshpass" and not _which("sshpass"):
        raise RuntimeError("password auth needs the 'sshpass' tool — either "
                           "install it (conda-forge) or use an SSH key file")
    # The desktop launcher exports LD_LIBRARY_PATH=$CONDA_PREFIX/lib for Qt; that
    # breaks the system ssh binary (OpenSSL version mismatch). Scrub it here.
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    pw = cfg.get("ssh", {}).get("password")
    if pw:
        env["SSHPASS"] = pw  # consumed by `sshpass -e`, kept out of argv/ps
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          env=env)
    return proc.returncode, proc.stdout, proc.stderr


def _which(exe: str) -> str | None:
    from shutil import which
    return which(exe)


# ---- jupyter transport ---------------------------------------------------------

def _jupyter_session(cfg: dict):
    import requests

    j = cfg.get("jupyter", {})
    url = (j.get("url") or "").rstrip("/")
    if not url:
        raise ValueError("jupyter transport not configured (missing url)")
    sess = requests.Session()
    if j.get("token"):
        sess.headers["Authorization"] = f"token {j['token']}"
    return sess, url


def _jupyter_status(cfg: dict, timeout: int = 10) -> dict:
    sess, url = _jupyter_session(cfg)
    r = sess.get(f"{url}/api/status", timeout=timeout)
    r.raise_for_status()
    return r.json()


def _run_jupyter(cfg: dict, command: str, timeout: int = 60) -> tuple[int, str, str]:
    """Execute a shell command on the box through a throwaway Jupyter kernel."""
    import websocket  # websocket-client

    sess, url = _jupyter_session(cfg)
    r = sess.post(f"{url}/api/kernels", timeout=15,
                  json={"name": "python3"})
    r.raise_for_status()
    kernel = r.json()["id"]
    try:
        ws_url = url.replace("https://", "wss://").replace("http://", "ws://")
        token = cfg.get("jupyter", {}).get("token", "")
        ws = websocket.create_connection(
            f"{ws_url}/api/kernels/{kernel}/channels"
            + (f"?token={token}" if token else ""),
            timeout=timeout,
            header=[f"Authorization: token {token}"] if token else [])
        msg_id = uuid.uuid4().hex
        code = ("import subprocess as _sp; _p=_sp.run(" + repr(command) +
                ", shell=True, capture_output=True, text=True, timeout=" +
                str(timeout - 5) + "); print(_p.stdout, end='');"
                "import sys; print(_p.stderr, end='', file=sys.stderr);"
                "print('\\n__RC__', _p.returncode)")
        ws.send(json.dumps({
            "header": {"msg_id": msg_id, "msg_type": "execute_request",
                       "username": "g1dance", "session": uuid.uuid4().hex,
                       "version": "5.3"},
            "parent_header": {}, "metadata": {},
            "content": {"code": code, "silent": False, "store_history": False,
                        "user_expressions": {}, "allow_stdin": False},
            "channel": "shell",
        }))
        out, err, rc = [], [], 1
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = json.loads(ws.recv())
            if frame.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            t = frame["msg_type"]
            if t == "stream":
                (out if frame["content"]["name"] == "stdout" else err
                 ).append(frame["content"]["text"])
            elif t == "error":
                err.append("\n".join(frame["content"]["traceback"]))
            elif t == "status" and frame["content"]["execution_state"] == "idle":
                break
        ws.close()
        text = "".join(out)
        if "__RC__" in text:
            text, _, rc_s = text.rpartition("__RC__")
            rc = int(rc_s.strip() or 1)
            text = text.rstrip("\n")
        return rc, text, "".join(err)
    finally:
        try:
            sess.delete(f"{url}/api/kernels/{kernel}", timeout=10)
        except Exception:
            pass


# ---- file transfer (ssh/scp only) -----------------------------------------------
# The GreenNode box is reached over SSH in practice; the jupyter transport was the
# plan-B command channel and has no sane bulk-file path, so transfers require ssh.

def _scp_argv(cfg: dict, sources: list[str], dest: str) -> list[str]:
    """scp argv mirroring _ssh_argv's option/auth handling. `sources`/`dest` are
    either local paths or 'remote:'-prefixed box paths (we add user@host)."""
    s = cfg.get("ssh", {})
    if not s.get("host"):
        raise ValueError("ssh transport not configured (missing host)")
    opts = [o for pair in zip(SSH_BASE_OPTS[::2], SSH_BASE_OPTS[1::2])
            for o in pair if not (s.get("password") and pair[1] == "BatchMode=yes")]
    argv = ["scp", "-q", *opts]
    if s.get("port"):
        argv += ["-P", str(s["port"])]           # scp uses -P, ssh uses -p
    if s.get("key_path"):
        argv += ["-i", str(Path(s["key_path"]).expanduser())]
    if s.get("password"):
        argv = ["sshpass", "-e", *argv]
    host = f"{s.get('user', 'root')}@{s['host']}"

    def resolve(p: str) -> str:
        return f"{host}:{p[len('remote:'):]}" if p.startswith("remote:") else p

    return [*argv, *[resolve(p) for p in sources], resolve(dest)]


def _run_scp(cfg: dict, sources: list[str], dest: str, timeout: int) -> None:
    if cfg.get("transport") != "ssh":
        raise RuntimeError("file transfer needs the ssh transport (scp); the "
                           "jupyter transport is command-only — configure SSH "
                           "in Studio → Cloud GPU")
    argv = _scp_argv(cfg, sources, dest)
    if argv[0] == "sshpass" and not _which("sshpass"):
        raise RuntimeError("password auth needs the 'sshpass' tool — either "
                           "install it (conda-forge) or use an SSH key file")
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    pw = cfg.get("ssh", {}).get("password")
    if pw:
        env["SSHPASS"] = pw
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                          env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"scp failed (rc={proc.returncode}): "
                           f"{(proc.stderr or proc.stdout).strip()[-300:]}")


def push_file(local: Path | str, remote: str, timeout: int = 1800,
              cfg: dict | None = None) -> None:
    """Copy a local file to `remote` (absolute path) on the box."""
    local = Path(local)
    if not local.is_file():
        raise FileNotFoundError(f"push_file: no such local file: {local}")
    _run_scp(cfg or load_config(), [str(local)], f"remote:{remote}", timeout)


def pull_file(remote: str, local: Path | str, timeout: int = 1800,
              cfg: dict | None = None) -> Path:
    """Copy a file from the box to `local`; returns the local path."""
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    _run_scp(cfg or load_config(), [f"remote:{remote}"], str(local), timeout)
    if not local.is_file():
        raise RuntimeError(f"pull_file: transfer reported ok but {local} is missing")
    return local


# ---- public API ----------------------------------------------------------------

def run(command: str, timeout: int = 60, cfg: dict | None = None) -> tuple[int, str, str]:
    """Run a shell command on the cloud box. Returns (rc, stdout, stderr)."""
    cfg = cfg or load_config()
    if cfg.get("transport") == "ssh":
        return _run_ssh(cfg, command, timeout)
    if cfg.get("transport") == "jupyter":
        return _run_jupyter(cfg, command, timeout)
    raise ValueError("cloud transport not configured")

GPU_QUERY = ("nvidia-smi --query-gpu=name,memory.total,memory.used,"
             "utilization.gpu --format=csv,noheader 2>/dev/null || echo NO_GPU")


def test_connection(cfg: dict | None = None) -> dict:
    """Full connectivity check: reachable? shell? GPU present/busy?"""
    cfg = cfg or load_config()
    result = {"connected": False, "transport": cfg.get("transport") or None,
              "detail": "", "gpu": None, "busy": None,
              "checked_at": time.time()}
    if not cfg.get("transport"):
        result["detail"] = "not configured"
        return result
    try:
        if cfg["transport"] == "jupyter":
            _jupyter_status(cfg)  # cheap reachability + auth check first
        rc, out, err = run(GPU_QUERY, timeout=45, cfg=cfg)
        if rc != 0:
            result["detail"] = (err or out or "command failed").strip()[-300:]
            return result
        result["connected"] = True
        line = out.strip().splitlines()[0] if out.strip() else "NO_GPU"
        if line == "NO_GPU":
            result["detail"] = "connected, but no NVIDIA GPU visible"
        else:
            name, mem_total, mem_used, util = [x.strip() for x in line.split(",")]
            util_pct = int(util.split()[0])
            result["gpu"] = {"name": name, "memory_total": mem_total,
                             "memory_used": mem_used, "utilization_pct": util_pct}
            result["busy"] = util_pct >= 50
            result["detail"] = f"{name}, util {util_pct}%"
    except Exception as e:
        result["detail"] = f"{type(e).__name__}: {e}"[:300]
    return result
