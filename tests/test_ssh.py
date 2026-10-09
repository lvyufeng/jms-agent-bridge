from jms_bridge.config import BastionConfig
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
    argv = r._ssh_argv(Host(name="a", login="u", asset_id="A"))
    assert argv[0] == "sshpass" and argv[1] == "-e"
    assert "PubkeyAuthentication=no" in argv
    assert "PreferredAuthentications=password" in argv
    # password rides in the env, never in argv (argv is visible in `ps`)
    assert "s3cret" not in argv
    assert r._subproc_env()["SSHPASS"] == "s3cret"


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