import textwrap

import pytest

from jms_bridge.inventory import (
    Host,
    Inventory,
    InventoryError,
    shell_quote,
    wrap_command,
)


def write_inventory(tmp_path, body):
    p = tmp_path / "hosts.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_load_and_resolve(tmp_path):
    p = write_inventory(
        tmp_path,
        """
        hosts:
          a:
            login: u
            account: root
            asset_id: "A1"
            platform: ascend-910c
            tags: [gpu, ascend]
          b:
            login: u
            account: ubuntu
            asset_id: "B1"
            platform: cuda
            tags: [gpu, cuda]
          board:
            login: u
            asset_id: "C1"
            platform: aarch64
            tags: [arm]
        groups:
          cluster: [a, b]
        """,
    )
    inv = Inventory.load(p)
    assert set(inv.hosts) == {"a", "b", "board"}
    assert {h.name for h in inv.resolve(["all"])} == {"a", "b", "board"}
    assert {h.name for h in inv.resolve(["gpu"])} == {"a", "b"}
    assert {h.name for h in inv.resolve(["cluster"])} == {"a", "b"}
    assert [h.name for h in inv.resolve(["a"])] == ["a"]
    # dedup across selectors
    assert {h.name for h in inv.resolve(["gpu", "cluster", "a"])} == {"a", "b"}


def test_unknown_selector_raises(tmp_path):
    p = write_inventory(tmp_path, "hosts:\n  a:\n    login: u\n")
    inv = Inventory.load(p)
    with pytest.raises(InventoryError):
        inv.resolve(["nope"])


def test_group_referencing_unknown_host_raises(tmp_path):
    p = write_inventory(tmp_path, "hosts:\n  a:\n    login: u\ngroups:\n  g: [a, ghost]\n")
    inv = Inventory.load(p)
    with pytest.raises(InventoryError):
        inv.resolve(["g"])


def test_login_string_full_and_partial():
    h = Host(name="a", login="yflv", account="root", asset_id="ASSET")
    assert h.login_string() == "yflv#root#ASSET"
    # no asset id -> user#account (asset resolved by name elsewhere)
    assert Host(name="b", login="yflv", account="root").login_string() == "yflv#root"
    # login falls back to the bastion default
    fallback = Host(name="c", account="root", asset_id="ASSET").login_string("fallback")
    assert fallback == "fallback#root#ASSET"


def test_login_string_without_login_raises():
    with pytest.raises(InventoryError):
        Host(name="x", asset_id="A").login_string()


def test_shell_quote_escapes_single_quotes():
    assert shell_quote("plain") == "'plain'"
    assert shell_quote("a'b") == "'a'\"'\"'b'"


def test_wrap_command_sources_env_and_cds():
    script = wrap_command("source /x/env.sh", "/work/dir", "echo hi")
    assert script.startswith("set -euo pipefail\n")
    assert "source /x/env.sh" in script
    # workdir is created if missing, then entered, so a fresh host does not fail
    assert "mkdir -p '/work/dir' && cd '/work/dir'" in script
    assert script.rstrip().endswith("echo hi")


def test_wrap_command_minimal():
    script = wrap_command(None, None, "true")
    assert script == "set -euo pipefail\ntrue\n"


def test_missing_inventory_raises(tmp_path):
    with pytest.raises(InventoryError):
        Inventory.load(tmp_path / "nope.yaml")


def test_platform_def_falls_back_to_generic():
    h = Host(name="x", platform="nonexistent")
    assert h.platform_def.name == "generic"