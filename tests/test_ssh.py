import asyncio

import pytest

from jms_bridge.config import BastionConfig, ConfigError
from jms_bridge.inventory import Host
from jms_bridge.ssh import ExecResult, SSHRunner


def make_runner(**kw):
    cfg = BastionConfig(url="https://bastion.example.com", ssh_port=2224, login_user="me", **kw)
    return SSHRunner(cfg)


def test_target_is_jumpserver_login_string():
    r = make_runner()
    h = Host(name="a", login="alice", account="root", asset_id="ASSET")
    assert r.target(h) == "alice#root#ASSET@bastion.example.com"


def test_target_strips_scheme_and_path():
    r = SSHRunner(BastionConfig(url="https://bastion.example.com/some/path", ssh_port=22))
    h = Host(name="a", login="u", asset_id="A")
    assert r.target(h) == "u#A@bastion.example.com"


def test_ssh_argv_is_argv_list_not_shell_string():
    r = make_runner()
    h = Host(name="weird name; rm -rf /", login="u", asset_id="A")
    argv = r._ssh_argv(h)
    assert isinstance(argv, list)
    assert argv[0] == "ssh"
    assert "-p" in argv and argv[argv.index("-p") + 1] == "2224"
    # the host string is a single argv element — the local shell never sees it
    assert "u#A@bastion.example.com" in argv
    # remote command is piped via `bash -s`, not embedded in the target
    assert argv[-2:] == ["u#A@bastion.example.com", "bash -s"]


def test_ssh_key_adds_identities_only():
    r = make_runner(ssh_key="/tmp/k")
    argv = r._ssh_argv(Host(name="a", login="u", asset_id="A"))
    assert "-i" in argv and "/tmp/k" in argv
    assert "IdentitiesOnly=yes" in argv


def test_controlmaster_is_enabled_for_reuse():
    r = make_runner()
    argv = r._ssh_argv(Host(name="a", login="u", asset_id="A"))
    joined = " ".join(argv)
    assert "ControlMaster=auto" in joined
    assert "ControlPersist=300" in joined


def test_bastion_password_uses_sshpass_and_forces_password_auth():
    r = make_runner(ssh_password="s3cret")
    h = Host(name="a", login="u", asset_id="A")
    argv = r._ssh_argv(h)
    assert argv[0] == "sshpass" and argv[1] == "-e"
    assert "PubkeyAuthentication=no" in argv
    assert "PreferredAuthentications=password" in argv
    # password rides in the env, never in argv (argv is visible in `ps`)
    assert "s3cret" not in argv
    assert r._subproc_env(h)["SSHPASS"] == "s3cret"


def test_direct_host_env_carries_no_bastion_password():
    """The bastion password must reach only the child that will actually use it.

    A direct host is key-based, so `sshpass` is not in its argv — handing it SSHPASS
    anyway is the password where it has no business being. `close()`'s `ssh -O exit`
    talks to the control socket and needs none either, so the hostless default omits it.
    """
    r = make_runner(ssh_password="s3cret")
    direct = Host(name="d", ssh_host="h", ssh_user="ubuntu")
    assert "SSHPASS" not in r._subproc_env(direct)
    assert "SSHPASS" not in r._subproc_env()


def test_direct_host_ignores_bastion_password():
    """A direct host stays key-based even when a bastion password is configured."""
    r = make_runner(ssh_password="s3cret")
    argv = r._ssh_argv(Host(name="d", ssh_host="h", ssh_user="ubuntu"))
    assert "sshpass" not in argv
    assert "PreferredAuthentications=password" not in argv


def test_exec_on_env_setup_false_omits_source_line():
    """``env_setup=False`` must not source the host toolchain in the remote script.

    Workspace commands run inside a container, where the host env script is absent;
    leaking it in would abort the command under ``set -e``. Capture the script the
    runner would pipe, without touching SSH.
    """
    import asyncio

    r = make_runner()
    h = Host(
        name="a",
        login="u",
        asset_id="A",
        env_setup="source /opt/cann/set_env.sh",
        workdir="/work",
    )
    captured = {}

    async def fake_run_script(host, script, timeout, *, tty=False):
        captured["script"] = script
        return ExecResult(host.name, 0, "", "")

    r._run_script = fake_run_script

    asyncio.run(r.exec_on(h, "echo hi", env_setup=False, workdir=False))
    assert "set_env.sh" not in captured["script"]
    assert "source" not in captured["script"]
    assert captured["script"].rstrip().endswith("echo hi")

    asyncio.run(r.exec_on(h, "echo hi", env_setup=True, workdir=False))
    assert "source /opt/cann/set_env.sh" in captured["script"]


# --- a missing local tool must say what to install ----------------------------------


def _hide_tool(monkeypatch, *binaries):
    """Pretend ``binaries`` are absent, leaving every other tool alone."""
    import shutil

    real = shutil.which

    def which(name, *a, **kw):
        return None if name in binaries else real(name, *a, **kw)

    monkeypatch.setattr("jms_bridge.ssh.shutil.which", which)


def test_missing_ssh_says_what_to_install(monkeypatch):
    # `create_subprocess_exec` would raise a bare FileNotFoundError, which the MCP SDK
    # shows the agent as "Error executing tool <name>" — useless on a fresh clone.
    _hide_tool(monkeypatch, "ssh")
    r = make_runner()
    with pytest.raises(ConfigError) as e:
        asyncio.run(r.exec_on(Host(name="a", login="u", asset_id="A"), "echo hi"))
    assert "`ssh` is not installed" in str(e.value)


def test_missing_rsync_says_what_to_install(monkeypatch):
    _hide_tool(monkeypatch, "rsync")
    r = make_runner()
    host = Host(name="a", login="u", asset_id="A")
    with pytest.raises(ConfigError) as e:
        asyncio.run(r.rsync(host, "/local", "/remote"))
    assert "`rsync` is not installed" in str(e.value)


def test_missing_sshpass_is_a_config_error(monkeypatch):
    _hide_tool(monkeypatch, "sshpass")
    r = make_runner(ssh_password="s3cret")
    with pytest.raises(ConfigError) as e:
        r._ssh_argv(Host(name="a", login="u", asset_id="A"))
    assert "sshpass" in str(e.value) and "install" in str(e.value)


# --- the fan-out cap is configurable, and can never be a hang -----------------------


def test_max_parallel_defaults_to_eight():
    assert make_runner().max_parallel == 8


def test_max_parallel_is_configurable():
    cfg = BastionConfig(url="https://bastion.example.com", max_parallel=32)
    assert SSHRunner(cfg, max_parallel=cfg.max_parallel).max_parallel == 32


def test_max_parallel_below_one_cannot_deadlock_the_semaphore():
    # Semaphore(0) is a hang, not an error, and the loader rejects 0 — this is the last
    # line of defence for a directly-constructed runner (tests, embedders).
    cfg = BastionConfig(url="https://bastion.example.com")
    assert SSHRunner(cfg, max_parallel=0).max_parallel == 1
    assert SSHRunner(cfg, max_parallel=-4).max_parallel == 1