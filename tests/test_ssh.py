from jms_bridge.config import BastionConfig
from jms_bridge.inventory import Host
from jms_bridge.ssh import SSHRunner


def make_runner(**kw):
    cfg = BastionConfig(url="https://bastion.example.com", ssh_port=2224, login_user="me", **kw)
    return SSHRunner(cfg)


def test_target_is_jumpserver_login_string():
    r = make_runner()
    h = Host(name="a", login="yflv", account="root", asset_id="ASSET")
    assert r.target(h) == "yflv#root#ASSET@bastion.example.com"


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