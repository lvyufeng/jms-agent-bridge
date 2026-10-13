"""The first-run path: what `jms-bridge init` writes and whether it is usable.

The rest of the CLI is covered only against a real bastion (see CONTRIBUTING), but the
templates are pure data and they are what a freshly-cloned checkout hands a new user.
The regression this guards: `init` used to write an inventory with no `workspaces:`
block, so the documented onboarding sequence produced a file on which every
`jms_dev_*` tool fails with "matched no workspace" — the dev loop, which is the point
of the tool, was unreachable from the documented path.
"""

import stat
import textwrap

from jms_bridge.cli import _CONFIG_TEMPLATE, _INVENTORY_TEMPLATE, cmd_init
from jms_bridge.config import read_toml
from jms_bridge.inventory import Inventory


def _load_template(tmp_path, text):
    p = tmp_path / "hosts.yaml"
    p.write_text(text)
    return Inventory.load(p)


def test_init_inventory_template_loads(tmp_path):
    inv = _load_template(tmp_path, _INVENTORY_TEMPLATE)
    assert set(inv.hosts) == {"ascend-910c", "node-4090-1"}
    assert inv.groups["cluster"] == ["ascend-910c", "node-4090-1"]


def test_init_inventory_template_has_a_workspace(tmp_path):
    # Without this, `jms_dev_run(["all"], ...)` resolves to no workspaces and the
    # documented quickstart cannot deliver code to any chip.
    inv = _load_template(tmp_path, _INVENTORY_TEMPLATE)
    assert inv.workspaces, "init template must ship at least one workspace"
    ws = next(iter(inv.workspaces.values()))
    assert ws.host in inv.hosts
    assert ws.repo and ws.container


def test_init_inventory_template_only_uses_git_source(tmp_path):
    # `git` is the only source that needs no extra field, so it is the only one safe to
    # put in a template a user might run before reading the comments. proxy-git and push
    # would need `proxy:` / `mount:` + `local:` to load at all.
    inv = _load_template(tmp_path, _INVENTORY_TEMPLATE)
    assert {ws.source for ws in inv.workspaces.values()} == {"git"}


def test_init_config_template_has_a_bastion_url(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(_CONFIG_TEMPLATE)
    data = read_toml(p)
    assert data["bastion"]["url"]
    # The terse `NestedKeyError`-style failure a missing url would cause is exactly what
    # this template exists to prevent.
    assert data["bastion"]["login_user"] == "your-login"


def test_init_writes_both_files_and_tightens_the_config(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    inv = tmp_path / "hosts.yaml"
    monkeypatch.setenv("JMS_BRIDGE_CONFIG", str(cfg))
    monkeypatch.setenv("JMS_BRIDGE_INVENTORY", str(inv))

    class Args:
        force = False

    assert cmd_init(Args()) == 0
    assert cfg.exists() and inv.exists()
    # config.toml will hold a session token after `login`, and hosts.yaml holds asset
    # ids; both should be owner-only from the moment init writes them.
    assert stat.S_IMODE(cfg.stat().st_mode) == 0o600


def test_init_does_not_clobber_without_force(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "config.toml"
    monkeypatch.setenv("JMS_BRIDGE_CONFIG", str(cfg))
    monkeypatch.setenv("JMS_BRIDGE_INVENTORY", str(tmp_path / "hosts.yaml"))
    cfg.write_text(textwrap.dedent("""\
        [bastion]
        url = "https://mine.example.com"
    """))

    class Args:
        force = False

    assert cmd_init(Args()) == 0
    assert "skipping" in capsys.readouterr().out
    assert "mine.example.com" in cfg.read_text()


def test_init_chmods_hosts_yaml_too(tmp_path, monkeypatch):
    # It names every host and its asset id, so it gets the same treatment as the config.
    cfg = tmp_path / "config.toml"
    inv = tmp_path / "hosts.yaml"
    monkeypatch.setenv("JMS_BRIDGE_CONFIG", str(cfg))
    monkeypatch.setenv("JMS_BRIDGE_INVENTORY", str(inv))

    class Args:
        force = False

    cmd_init(Args())
    assert stat.S_IMODE(inv.stat().st_mode) == 0o600
    # ...and the template comments survive the round trip (they are the documentation).
    assert "workspaces:" in inv.read_text()
