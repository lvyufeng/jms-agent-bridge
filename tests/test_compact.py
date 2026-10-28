"""Compact protocol and REST inventory tests use only synthetic assets and mock APIs."""

import asyncio
import json
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from jms_bridge import server as legacy
from jms_bridge.api import JumpServerError
from jms_bridge.compact import BridgeService, unwrap
from jms_bridge.config import BastionConfig, ConfigError
from jms_bridge.inventory import InventoryError
from jms_bridge.rest_inventory import AssetCatalog
from jms_bridge.results import ResultStore
from jms_bridge.runtime_config import BridgeSettings, default_state_dir, load_settings


@pytest.fixture(autouse=True)
def isolated_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("JMS_BRIDGE_INVENTORY", str(tmp_path / "optional-hosts.yaml"))


@pytest.fixture
def settings(tmp_path):
    return BridgeSettings(state_dir=tmp_path, preferred_account="root")


@pytest.fixture
def cfg():
    return BastionConfig(
        url="https://bastion.example.com",
        login_user="example-user",
        private_token="test-only-token",
    )


def asset(name="node", **overrides):
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "address": "192.0.2.10",
        "active": True,
        "protocols": ["ssh"],
        "accounts": ["root"],
        "account": "root",
        "login": "example-user",
        **overrides,
    }


@pytest.fixture
async def service(cfg, settings):
    service = BridgeService(cfg, settings)
    service.catalog._fetch = AsyncMock(return_value=[asset()])
    await service.catalog.refresh()
    yield service
    await service.catalog.close()
    await service.runner.close()


def test_result_store_complete_utf8_and_search(tmp_path):
    store = ResultStore(tmp_path)
    rid = store.create(kind="run")
    raw = ("a中文🙂z" * 10000).encode()
    store.path(rid, "stdout").write_bytes(raw)
    store.path(rid, "stderr").write_bytes(b"error-at-end")
    meta = store.finish(rid, ok=False)
    assert meta["bytes"]["stdout"] == len(raw)
    assert store.directory(rid).stat().st_mode & 0o777 == 0o700
    assert (store.directory(rid) / "meta.json").stat().st_mode & 0o777 == 0o600
    preview = store.preview(rid, 100)
    assert preview["stderr"] == "error-at-end" and preview["stdout_truncated"]
    chunks, offset = [], 0
    while offset < len(raw):
        page = store.read(rid, offset=offset, limit=1001)
        chunks.append(page["text"])
        offset = page["next_offset"]
    assert "".join(chunks).encode() == raw
    assert store.read(rid, tail=True, limit=17)["eof"]
    matches = store.read(rid, pattern="中文", limit=250)
    assert matches["matches"][0]["offset"] == 1
    assert len(matches["matches"]) <= 20
    assert store.read(rid, pattern="absent")["matches"] == []
    saved = store.save_json({"a": [1, 2], "ok": False})
    assert json.loads(store.read(saved, "result")["text"]) == {"a": [1, 2], "ok": False}


@pytest.mark.parametrize(
    "arguments",
    [
        {"result_id": "../escape"},
        {"result_id": "f" * 32},
        {"limit": 0},
        {"offset": -1},
        {"stream": "invalid"},
        {"pattern": ""},
        {"pattern": "a", "tail": True},
        {"pattern": "a" * 1025},
        {"stream": "stderr"},
    ],
)
def test_result_read_rejects_invalid_requests(tmp_path, arguments):
    store = ResultStore(tmp_path)
    rid = store.save_json({"test": True})
    with pytest.raises(ValueError):
        store.read(**({"result_id": rid, "stream": "result"} | arguments))


def test_settings_are_generic_and_validate(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("JMS_BRIDGE_STATE_DIR", raising=False)
    assert default_state_dir() == tmp_path / "state/jms-bridge"
    path = tmp_path / "config.toml"
    monkeypatch.setenv("JMS_BRIDGE_CONFIG", str(path))
    assert load_settings().refresh_interval == 3600
    path.write_text(
        '[bridge]\nssh_proxy_command=["nc", "-x", "proxy.example:3128"]\n'
        'known_hosts="~/known_hosts"\nssh_host="gateway.example.com"\n'
    )
    parsed = load_settings()
    assert parsed.ssh_proxy_command[-1] == "proxy.example:3128"
    assert parsed.known_hosts.is_absolute()
    assert parsed.gateway_host(None) == "gateway.example.com"
    assert (
        BridgeSettings().gateway_host(BastionConfig(url="https://bastion.example.com"))
        == "bastion.example.com"
    )
    with pytest.raises(ConfigError):
        BridgeSettings().gateway_host(BastionConfig(url="invalid"))
    monkeypatch.setenv("JMS_BRIDGE_STATE_DIR", str(tmp_path / "override"))
    assert load_settings().state_dir == tmp_path / "override"


@pytest.mark.parametrize(
    "body",
    [
        "bridge=[]",
        "[bridge]\nrefresh_interval=0",
        "[bridge]\nrefresh_interval=true",
        '[bridge]\nssh_proxy_command="shell command"',
        "[bridge]\nssh_proxy_command=[1]",
        "[bridge]\nknown_hosts=false",
    ],
)
def test_bad_settings(body, tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(body)
    monkeypatch.setenv("JMS_BRIDGE_CONFIG", str(path))
    with pytest.raises(ConfigError):
        load_settings()


async def test_catalog_atomic_refresh_removal_identity(cfg, settings, monkeypatch):
    catalog = AssetCatalog(cfg, settings)
    first = asset()
    catalog._fetch = AsyncMock(return_value=[first])
    assert (await catalog.refresh())["ok"]
    assert catalog.inventory.resolve(["node"])[0].asset_id == first["id"]
    assert catalog.inventory.resolve([first["id"]])[0].asset_id == first["id"]
    assert AssetCatalog(cfg, settings).inventory.hosts
    different = BastionConfig(url="https://other.example.com", login_user=cfg.login_user)
    assert not AssetCatalog(different, settings).inventory.hosts
    previous = catalog.inventory
    catalog._fetch = AsyncMock(return_value=[])
    with monkeypatch.context() as m:

        def fail(*args):
            raise OSError("disk full")

        m.setattr("jms_bridge.rest_inventory.atomic_json", fail)
        assert not (await catalog.refresh())["ok"]
    assert catalog.inventory is previous
    assert (await catalog.refresh())["ok"] and not catalog.inventory.hosts
    with pytest.raises(InventoryError, match="REST snapshot"):
        catalog.inventory.resolve(["node"])
    catalog.cache.write_text("invalid")
    assert AssetCatalog(cfg, settings).error


async def test_catalog_rest_details_accounts_and_failures(cfg, settings, monkeypatch):
    catalog = AssetCatalog(cfg, settings)
    ids = [str(uuid.uuid4()) for _ in range(4)]
    details = {
        i: {
            "id": i,
            "name": "duplicate",
            "address": "192.0.2.2",
            "permed_protocols": [{"name": "ssh"}],
            "permed_accounts": [
                {"username": "root", "has_secret": True},
                {"username": "service", "has_secret": True},
            ],
        }
        for i in ids
    }

    async def get_asset(i):
        if i == ids[-1]:
            raise JumpServerError("GET -> 403: permission changed")
        return details[i]

    api = SimpleNamespace(
        list_assets=AsyncMock(return_value=([{"id": i} for i in ids], False)),
        get_asset=get_asset,
        aclose=AsyncMock(),
    )
    monkeypatch.setattr(catalog, "make_api", lambda: api)
    assert (await catalog.refresh())["assets"] == 3
    assert len(catalog.inventory.hosts) == 3
    assert not catalog.inventory.groups  # Ambiguous names/addresses cannot fan out.
    settings.preferred_account = None
    assert (await catalog.refresh())["ssh_targets"] == 0  # No arbitrary account selection.
    settings.preferred_account = "root"
    api.list_assets.return_value = ([{"id": ids[0]}], True)
    previous = catalog.snapshot
    assert not (await catalog.refresh())["ok"] and catalog.snapshot is previous
    api.list_assets.side_effect = JumpServerError("private response must not escape")
    status = await catalog.refresh()
    assert "private response" not in status["last_error"]
    assert api.aclose.await_count == 4


async def test_catalog_excludes_inactive_and_non_ssh(cfg, settings):
    catalog = AssetCatalog(cfg, settings)
    catalog._fetch = AsyncMock(
        return_value=[
            asset(active=False),
            asset(protocols=["rdp"]),
            asset(login=""),
            asset(account="invalid#account"),
        ]
    )
    assert (await catalog.refresh())["ssh_targets"] == 0
    assert catalog.search("node", limit=2)["next_offset"] == 2


async def test_catalog_hourly_background_and_startup(cfg, settings, monkeypatch):
    catalog = AssetCatalog(cfg, settings)
    catalog._fetch = AsyncMock(return_value=[asset()])
    await catalog.refresh()
    catalog.snapshot["refreshed_at"] = time.time() - 3601
    catalog._last_attempt = 0
    await catalog.ensure()
    assert catalog._fetch.await_count == 2
    sleeps = 0

    async def sleep(_):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            raise asyncio.CancelledError
        catalog.snapshot["refreshed_at"] = time.time() - 3601
        catalog._last_attempt = 0

    monkeypatch.setattr("jms_bridge.rest_inventory.asyncio.sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        await catalog._background()
    assert catalog._fetch.await_count == 4


async def test_catalog_listener_does_not_wait_for_network(cfg, settings):
    catalog = AssetCatalog(cfg, settings)
    started, release = asyncio.Event(), asyncio.Event()

    async def slow():
        started.set()
        await release.wait()
        return []

    catalog._fetch = slow
    await asyncio.wait_for(catalog.start(), 0.5)
    await started.wait()
    request = asyncio.create_task(catalog.ensure())
    await asyncio.sleep(0.01)
    assert not request.done()
    release.set()
    await request
    await catalog.close()
    catalog.snapshot["refreshed_at"] = 0
    catalog._fetch = AsyncMock(side_effect=OSError())
    catalog._last_attempt = 0
    with pytest.raises(ValueError, match="REST refresh failed"):
        await catalog.ensure()


async def test_compact_discovery_and_runtime_isolation(service):
    tools = await service.server.list_tools()
    assert {t.name for t in tools} == {"run", "read_result", "discover", "call", "fs"}
    assert len(json.dumps([t.model_dump(mode="json") for t in tools if t.name != "fs"])) < 6000
    assert len(json.dumps(next(t for t in tools if t.name == "fs").model_dump(mode="json"))) < 6000
    assert (await service.discover())["categories"]["jobs"]
    assert (await service.discover(query="jobs", limit=2))["next_offset"] == 2
    assert (await service.discover(query="sync"))["operations"]
    assert (await service.discover(operation="jms_sync_pull"))["inputSchema"]
    assert (await service.discover(topic="assets"))["matched"] == 1
    assert (await service.discover(topic="status"))["inventory"]["refresh_interval_seconds"] == 3600
    result = await service.call("session_list", {})
    assert result["data"]["sessions"] == []
    detail = await service.call(
        "jms_asset_detail", {"asset_id": next(iter(service.catalog.inventory.hosts))}
    )
    assert "login" not in detail["data"]["asset"]
    assert (await service.call("jms_list_assets", {}))["data"]["matched"] == 1
    assert (await service.call("jms_inventory_reload", {}))["data"]["ok"]
    # Scope dependencies to the hidden operation; never monkeypatch the legacy singletons.
    service.runner.exec_on = AsyncMock()
    from jms_bridge.ssh import ExecResult

    service.runner.exec_on.return_value = ExecResult("node", 0, "jms-bridge-ok\nLinux\n", "")
    result = await service.call("jms_probe", {"targets": ["node"]})
    assert result["data"]["ok"]
    assert legacy._runtime.get() is None
    old = legacy._state.copy()
    with pytest.raises(ToolError):
        await service.call("jms_dev_sync", {"targets": ["missing-workspace"], "ref": "main"})
    assert legacy._state == old and legacy._runtime.get() is None


@pytest.mark.parametrize(
    "tool,args",
    [
        ("run", {"targets": [], "script": "true"}),
        ("run", {"targets": ["node"], "script": "true", "script_path": "/missing"}),
        ("run", {"targets": ["node"], "script_path": "relative.sh"}),
        ("run", {"targets": ["node"], "script": "true", "timeout": 1000}),
        ("run", {"targets": ["node"], "script": "rm -rf /"}),
        ("run", {"targets": ["missing"], "script": "true"}),
        ("discover", {"operation": "missing"}),
        ("discover", {"limit": 0}),
        ("call", {"operation": "missing", "arguments": {}}),
        ("call", {"operation": "session_close", "arguments": {}}),
        ("call", {"operation": "jms_asset_detail", "arguments": {"asset_id": "missing"}}),
        ("read_result", {"result_id": "../private"}),
    ],
)
async def test_tool_validation_before_side_effects(service, tool, args):
    with pytest.raises(ToolError):
        await service.server.call_tool(tool, args)
    assert service.runner.connection_status()["connected"] == 0


async def test_large_call_and_response_unwrap(service):
    service.catalog._fetch.return_value = [asset(name=f"node-{i}") for i in range(50)]
    await service.catalog.refresh()
    result = await service.call("jms_list_assets", {})
    assert result["truncated"] and len(result["preview"].encode()) <= 4099
    assert service.store.metadata(result["result_id"])["status"] == "complete"
    assert unwrap(
        SimpleNamespace(
            structured_content=None,
            is_error=False,
            content=[SimpleNamespace(type="text", text='{"ok":true}')],
        )
    ) == {"ok": True}
    assert (
        unwrap(
            SimpleNamespace(
                structured_content=None,
                is_error=True,
                content=[SimpleNamespace(type="text", text="failure")],
            )
        )["ok"]
        is False
    )


def test_explicit_rest_proxy(cfg, settings):
    settings.rest_proxy = "http://proxy.example:3128"
    catalog = AssetCatalog(cfg, settings)
    api = catalog.make_api()
    assert isinstance(api._client, httpx.AsyncClient)
    asyncio.run(api.aclose())


def test_state_directory_has_a_single_owner(tmp_path):
    from jms_bridge.runtime_config import StateLock

    with StateLock(tmp_path):
        with pytest.raises(ConfigError, match="already in use"), StateLock(tmp_path):
            pass
    with StateLock(tmp_path):
        pass


def test_search_across_scan_boundary(tmp_path):
    store = ResultStore(tmp_path)
    rid = store.create()
    store.path(rid, "stdout").write_bytes(b"a" * (8 * 1024 * 1024 - 2) + b"needle")
    store.finish(rid)
    assert store.read(rid, pattern="needle")["matches"][0]["offset"] == 8 * 1024 * 1024 - 2


async def test_compact_stdio_protocol(tmp_path):
    import os
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    path = tmp_path / "config.toml"
    path.write_text(
        '[bastion]\nurl="http://127.0.0.1:1"\nlogin_user="example-user"\n'
        'private_token="test-only-token"\napi_timeout=0.1\n'
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "jms_bridge.cli", "serve", "--compact"],
        env={
            **os.environ,
            "JMS_BRIDGE_CONFIG": str(path),
            "JMS_BRIDGE_STATE_DIR": str(tmp_path / "state"),
        },
    )
    async with stdio_client(params) as streams:
        async with ClientSession(streams[0], streams[1], read_timeout_seconds=10) as session:
            await session.initialize()
            assert {t.name for t in (await session.list_tools()).tools} == {
                "fs",
                "run",
                "read_result",
                "discover",
                "call",
            }
            status = unwrap(await session.call_tool("discover", {"topic": "status"}))
            assert status["inventory"]["source"] == "JumpServer REST"
            result = await session.call_tool("run", {"targets": [], "script": "true"})
            assert result.is_error


def test_compact_cli_dispatch(monkeypatch):
    from jms_bridge import cli, compact

    calls = []
    monkeypatch.setattr(compact, "main", lambda **kw: calls.append(kw))
    assert cli.main(["serve", "--compact", "--transport", "streamable-http", "--port", "8767"]) == 0
    assert calls == [{"transport": "streamable-http", "port": 8767}]
    assert cli.main(["serve", "--transport", "streamable-http"]) == 2
    assert cli.main(["serve", "--compact", "--port", "0"]) == 2


async def test_no_targets_names_an_unresolved_account(cfg, tmp_path):
    """Every asset offering several accounts and no `preferred_account` yields zero targets.

    The assets are all present and every selector matches nothing, so the reply has to
    name the real cause instead of leaving the reader hunting for a bad selector.
    """
    manifest = [asset(accounts=["root", "secure"], account=None) for _ in range(3)]
    maybe_ambiguous = AssetCatalog(cfg, BridgeSettings(state_dir=tmp_path))
    maybe_ambiguous._fetch = AsyncMock(return_value=manifest)
    assert (await maybe_ambiguous.refresh())["ok"]
    assert maybe_ambiguous.status()["assets"] == 3
    assert maybe_ambiguous.status()["ssh_targets"] == 0
    why = maybe_ambiguous.explain_no_targets()
    assert "bridge.preferred_account" in why and "(typically 2)" in why

    service = BridgeService(cfg, BridgeSettings(state_dir=tmp_path / "svc"))
    service.catalog._fetch = AsyncMock(return_value=manifest)
    await service.catalog.refresh()
    try:
        with pytest.raises(ValueError, match="no permitted SSH assets.*preferred_account"):
            await service.run(["all"], script="true")
    finally:
        await service.runner.close()


async def test_no_targets_distinguishes_the_other_causes(cfg, tmp_path):
    configured = AssetCatalog(cfg, BridgeSettings(state_dir=tmp_path, preferred_account="root"))
    configured._fetch = AsyncMock(
        return_value=[asset(accounts=["ubuntu"], account=None)]  # `root` is not permitted
    )
    await configured.refresh()
    assert "which none of these assets permits" in configured.explain_no_targets()

    no_ssh = AssetCatalog(cfg, BridgeSettings(state_dir=tmp_path / "b", preferred_account="root"))
    no_ssh._fetch = AsyncMock(return_value=[asset(protocols=["sftp"])])
    await no_ssh.refresh()
    assert "none is active with an SSH protocol" in no_ssh.explain_no_targets()


async def test_no_targets_is_silent_when_targets_exist(service):
    """A bad selector on a healthy snapshot gets the plain message, with no diagnosis."""
    assert service.catalog.explain_no_targets() is None
    with pytest.raises(InventoryError, match="not in the REST snapshot"):
        await service.run(["no-such-host"], script="true")


async def test_listed_but_unusable_asset_says_why(cfg, tmp_path):
    """An asset `discover` lists can still be unselectable — and the reply must say so.

    Reporting "not in the REST snapshot" for an entry the listing just showed sends the
    caller back to `discover` for a name it already has. Every exclusion reason is
    carried through to the error, keyed by both the display name and the asset id.
    """
    catalog = AssetCatalog(cfg, BridgeSettings(state_dir=tmp_path, preferred_account="root"))
    manifest = [
        asset(name="good", address="192.0.2.1"),
        asset(name="switched-off", address="192.0.2.2", active=False),
        asset(name="sftp-only", address="192.0.2.3", protocols=["sftp"]),
        asset(name="ambiguous", address="192.0.2.4", accounts=["root", "secure"], account=None),
    ]
    catalog._fetch = AsyncMock(return_value=manifest)
    await catalog.refresh()
    inventory = catalog.inventory
    assert len(inventory.hosts) == 1

    for name, reason in [
        ("switched-off", "is inactive in JumpServer"),
        ("sftp-only", "permits no ssh protocol"),
        ("ambiguous", "no unambiguous account selected"),
    ]:
        with pytest.raises(InventoryError, match=reason) as excinfo:
            inventory.resolve([name])
        assert "in the REST snapshot but is not a usable target" in str(excinfo.value)
        assert reason in inventory.excluded[name]

    # The id resolves to the same explanation: a caller holding a UUID is not stranded.
    inactive_id = manifest[1]["id"]
    with pytest.raises(InventoryError, match="is inactive"):
        inventory.resolve([inactive_id])

    # A name that is genuinely absent keeps the original, different message.
    with pytest.raises(InventoryError, match="^target is not in the REST snapshot"):
        inventory.resolve(["never-existed"])


async def test_search_marks_usability(cfg, tmp_path):
    catalog = AssetCatalog(cfg, BridgeSettings(state_dir=tmp_path, preferred_account="root"))
    catalog._fetch = AsyncMock(
        return_value=[
            asset(name="good", address="192.0.2.1"),
            asset(name="switched-off", address="192.0.2.2", active=False),
        ]
    )
    await catalog.refresh()
    rows = {row["name"]: row for row in catalog.search("", 0, 10)["assets"]}
    assert rows["good"]["usable"] is True and "unusable_because" not in rows["good"]
    assert rows["switched-off"]["usable"] is False
    assert "is inactive" in rows["switched-off"]["unusable_because"]


async def test_rest_targets_inherit_hosts_yaml_metadata(cfg, tmp_path, monkeypatch):
    """REST decides reachability; hosts.yaml supplies the machine's own settings.

    JumpServer exposes neither `env_setup` nor `workdir`, so a REST target otherwise
    runs with no environment and no working directory. Both files share `asset_id`, so
    the per-asset settings are keyed off that — without hosts.yaml ever granting access
    to an asset the REST snapshot did not already permit.
    """
    manifest = [
        asset(id="aaaaaaaa-0000-0000-0000-000000000001", name="known"),
        asset(id="bbbbbbbb-0000-0000-0000-000000000002", name="not-listed"),
    ]
    hosts_yaml = tmp_path / "hosts.yaml"
    hosts_yaml.write_text(
        "hosts:\n"
        "  local-name:\n"
        "    login: someone-else\n"  # must NOT override the REST login
        "    account: root\n"
        "    asset_id: aaaaaaaa-0000-0000-0000-000000000001\n"
        "    platform: ascend-910c\n"
        '    env_setup: "source /opt/ascend/set_env.sh"\n'
        "    workdir: /workspace/repo\n"
    )
    monkeypatch.setenv("JMS_BRIDGE_INVENTORY", str(hosts_yaml))

    catalog = AssetCatalog(cfg, BridgeSettings(state_dir=tmp_path / "s", preferred_account="root"))
    catalog._fetch = AsyncMock(return_value=manifest)
    await catalog.refresh()
    hosts = catalog.inventory.hosts
    known = hosts["aaaaaaaa-0000-0000-0000-000000000001"]
    assert known.workdir == "/workspace/repo"
    assert known.env_setup == "source /opt/ascend/set_env.sh"
    assert known.platform == "ascend-910c"
    assert known.login == cfg.login_user, "hosts.yaml must not change who we log in as"

    unlisted = hosts["bbbbbbbb-0000-0000-0000-000000000002"]
    assert unlisted.workdir is None and unlisted.env_setup is None
    assert unlisted.platform == "generic"


def test_asset_metadata_is_optional_and_tolerant(tmp_path, monkeypatch):
    """A REST-only deployment has no hosts.yaml, and a broken one must not break refresh."""
    from jms_bridge.inventory import Inventory

    monkeypatch.setenv("JMS_BRIDGE_INVENTORY", str(tmp_path / "absent.yaml"))
    assert Inventory.asset_metadata() == {}

    broken = tmp_path / "broken.yaml"
    broken.write_text("hosts: [this is not, a mapping\n")
    monkeypatch.setenv("JMS_BRIDGE_INVENTORY", str(broken))
    assert Inventory.asset_metadata() == {}
