"""Cloud config handling, secret masking, ssh argv construction, GPU parsing.
No real network calls anywhere."""
import json
import stat

import pytest

from pipeline import cloud


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    p = tmp_path / "secrets" / "cloud.json"
    monkeypatch.setattr(cloud, "CONFIG_PATH", p)
    return p


def test_default_config_when_missing(cfg_path):
    cfg = cloud.load_config()
    assert cfg == {"transport": "", "ssh": {}, "jupyter": {}}


def test_save_sets_0600(cfg_path):
    cloud.save_config({"transport": "ssh", "ssh": {"host": "h"}, "jupyter": {}})
    mode = stat.S_IMODE(cfg_path.stat().st_mode)
    assert mode == 0o600


def test_masked_config_hides_secrets_but_shows_presence(cfg_path):
    cloud.save_config({"transport": "jupyter",
                       "ssh": {"host": "1.2.3.4", "password": "hunter2"},
                       "jupyter": {"url": "https://x", "token": "tok123"}})
    masked = cloud.masked_config()
    assert masked["ssh"]["password"] == "•set•"
    assert masked["jupyter"]["token"] == "•set•"
    assert masked["ssh"]["host"] == "1.2.3.4"      # non-secrets visible
    # and the file on disk still has the real values
    assert json.loads(cfg_path.read_text())["ssh"]["password"] == "hunter2"


def test_update_empty_secret_keeps_stored_value(cfg_path):
    cloud.save_config({"transport": "ssh",
                       "ssh": {"host": "a", "password": "real"},
                       "jupyter": {}})
    cloud.update_config({"ssh": {"host": "b", "password": ""}})
    cfg = cloud.load_config()
    assert cfg["ssh"]["host"] == "b"
    assert cfg["ssh"]["password"] == "real"        # not wiped by the empty form


def test_update_replaces_secret_when_given(cfg_path):
    cloud.save_config({"transport": "ssh", "ssh": {"password": "old"},
                       "jupyter": {}})
    cloud.update_config({"ssh": {"password": "new"}})
    assert cloud.load_config()["ssh"]["password"] == "new"


# ---- ssh argv construction -------------------------------------------------------

def test_ssh_argv_key_auth():
    argv = cloud._ssh_argv(
        {"ssh": {"host": "10.0.0.1", "port": 46936, "user": "root",
                 "key_path": "~/k.pem"}}, "echo hi")
    assert argv[0] == "ssh"
    assert "-p" in argv and "46936" in argv
    assert "-i" in argv
    i = argv.index("-i")
    assert argv[i + 1].startswith("/")              # ~ expanded
    assert argv[-2] == "root@10.0.0.1"
    assert argv[-1] == "echo hi"
    assert any("BatchMode=yes" in a for a in argv)  # key auth keeps BatchMode


def test_ssh_argv_password_uses_sshpass_env_not_argv():
    # audit MEDIUM security: the password must NOT appear in argv (process table).
    # sshpass -e reads it from the SSHPASS env var instead.
    argv = cloud._ssh_argv(
        {"ssh": {"host": "h", "password": "pw"}}, "true")
    assert argv[:2] == ["sshpass", "-e"]
    assert "pw" not in argv                          # secret never in the argument list
    assert not any("BatchMode" in a for a in argv)
    assert argv[-2] == "root@h"                     # default user root


def test_ssh_argv_requires_host():
    with pytest.raises(ValueError, match="host"):
        cloud._ssh_argv({"ssh": {}}, "true")


# ---- scp transfer argv (no network) ------------------------------------------------

def test_scp_argv_key_auth_remote_paths():
    argv = cloud._scp_argv(
        {"transport": "ssh",
         "ssh": {"host": "10.0.0.1", "port": 46936, "user": "root",
                 "key_path": "~/k.pem"}},
        ["/local/a.csv"], "remote:/workspace/notebook-data/motions/a.csv")
    assert argv[0] == "scp"
    assert "-P" in argv and "46936" in argv         # scp uses -P, not -p
    i = argv.index("-i")
    assert argv[i + 1].startswith("/")              # ~ expanded
    assert argv[-2] == "/local/a.csv"
    assert argv[-1] == "root@10.0.0.1:/workspace/notebook-data/motions/a.csv"


def test_scp_argv_password_uses_sshpass_env():
    argv = cloud._scp_argv({"ssh": {"host": "h", "password": "pw"}},
                           ["remote:/x/y"], "/tmp/y")
    assert argv[:2] == ["sshpass", "-e"]
    assert "pw" not in argv
    assert argv[-2] == "root@h:/x/y"


def test_push_file_requires_ssh_transport(tmp_path, cfg_path):
    cloud.save_config({"transport": "jupyter", "ssh": {}, "jupyter": {"url": "x"}})
    f = tmp_path / "a.txt"
    f.write_text("hi")
    with pytest.raises(RuntimeError, match="ssh transport"):
        cloud.push_file(f, "/remote/a.txt")


def test_push_file_missing_local_file(cfg_path):
    cloud.save_config({"transport": "ssh", "ssh": {"host": "h"}, "jupyter": {}})
    with pytest.raises(FileNotFoundError):
        cloud.push_file("/no/such/file", "/remote/a.txt")


# ---- transport selection / connection test ---------------------------------------

def test_run_unconfigured_raises(cfg_path):
    with pytest.raises(ValueError, match="transport"):
        cloud.run("true")


def test_test_connection_not_configured(cfg_path):
    got = cloud.test_connection()
    assert got["connected"] is False
    assert got["detail"] == "not configured"


def test_test_connection_parses_gpu(monkeypatch, cfg_path):
    cloud.save_config({"transport": "ssh", "ssh": {"host": "h"}, "jupyter": {}})
    monkeypatch.setattr(
        cloud, "run",
        lambda cmd, timeout=60, cfg=None:
            (0, "NVIDIA GeForce RTX 4090, 24564 MiB, 812 MiB, 97 %\n", ""))
    got = cloud.test_connection()
    assert got["connected"] is True
    assert got["gpu"]["name"] == "NVIDIA GeForce RTX 4090"
    assert got["gpu"]["utilization_pct"] == 97
    assert got["busy"] is True


def test_test_connection_idle_gpu_not_busy(monkeypatch, cfg_path):
    cloud.save_config({"transport": "ssh", "ssh": {"host": "h"}, "jupyter": {}})
    monkeypatch.setattr(
        cloud, "run",
        lambda cmd, timeout=60, cfg=None:
            (0, "NVIDIA GeForce RTX 4090, 24564 MiB, 3 MiB, 2 %\n", ""))
    got = cloud.test_connection()
    assert got["busy"] is False


def test_test_connection_no_gpu(monkeypatch, cfg_path):
    cloud.save_config({"transport": "ssh", "ssh": {"host": "h"}, "jupyter": {}})
    monkeypatch.setattr(cloud, "run",
                        lambda cmd, timeout=60, cfg=None: (0, "NO_GPU\n", ""))
    got = cloud.test_connection()
    assert got["connected"] is True
    assert got["gpu"] is None
    assert "no NVIDIA GPU" in got["detail"]


def test_test_connection_failure_captured(monkeypatch, cfg_path):
    cloud.save_config({"transport": "ssh", "ssh": {"host": "h"}, "jupyter": {}})
    monkeypatch.setattr(
        cloud, "run",
        lambda cmd, timeout=60, cfg=None: (255, "", "Connection refused"))
    got = cloud.test_connection()
    assert got["connected"] is False
    assert "refused" in got["detail"]


def test_test_connection_swallows_exceptions(monkeypatch, cfg_path):
    cloud.save_config({"transport": "ssh", "ssh": {"host": "h"}, "jupyter": {}})

    def boom(cmd, timeout=60, cfg=None):
        raise TimeoutError("dead box")
    monkeypatch.setattr(cloud, "run", boom)
    got = cloud.test_connection()
    assert got["connected"] is False
    assert "TimeoutError" in got["detail"]
