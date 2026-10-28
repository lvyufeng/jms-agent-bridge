"""Optional overlays never grant targets or block the resident event loop."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from jms_bridge import rest_inventory
from jms_bridge.compact import BridgeService
from jms_bridge.config import BastionConfig
from jms_bridge.inventory import Inventory, InventoryError
from jms_bridge.rest_inventory import AssetCatalog
from jms_bridge.runtime_config import BridgeSettings

ASSET_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
OTHER_ID = "bbbbbbbb-0000-0000-0000-000000000002"


@pytest.fixture
def context(tmp_path, monkeypatch):
    path = tmp_path / "hosts.yaml"
    monkeypatch.setenv("JMS_BRIDGE_INVENTORY", str(path))
    cfg = BastionConfig(url="https://bastion.example.com", login_user="rest-user")
    settings = BridgeSettings(state_dir=tmp_path / "state", preferred_account="root")
    return SimpleNamespace(path=path, cfg=cfg, settings=settings)


def manifest(asset_id=ASSET_ID, name="node", **overrides):
    return {
        "id": asset_id,
        "name": name,
        "address": "192.0.2.10",
        "active": True,
        "protocols": ["ssh"],
        "accounts": ["root"],
        "account": "root",
        "login": "rest-user",
        **overrides,
    }


def write_hosts(context, hosts):
    context.path.write_text(yaml.safe_dump({"hosts": hosts}))


@pytest.mark.parametrize("body", [None, "", "null\n", "{}\n", "hosts: null\n", "hosts: {}\n"])
def test_missing_or_empty_metadata_is_normal(context, body):
    if body is not None:
        context.path.write_text(body)
    warnings = []
    assert Inventory.asset_metadata(warnings=warnings) == {}
    assert warnings == []


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("hosts: [first, second]\n", "hosts must be a mapping"),
        ("- PRIVATE_VALUE\n", "must be a mapping"),
        ("false\n", "must be a mapping"),
        ("hosts:\n  PRIVATE_VALUE: [broken\n", "could not be read"),
        ("hosts:\n  PRIVATE_VALUE: [one, two]\n", "entry #1 must be a mapping"),
    ],
)
def test_invalid_metadata_structure_has_safe_diagnostics(context, body, message):
    context.path.write_text(body)
    warnings = []
    assert Inventory.asset_metadata(warnings=warnings) == {}
    assert len(warnings) == 1 and message in warnings[0]
    assert "PRIVATE_VALUE" not in warnings[0]
    assert str(context.path) not in warnings[0]


def test_invalid_encoding_and_read_error_are_tolerated(context, monkeypatch):
    context.path.write_bytes(b"hosts: \xff\n")
    warnings = []
    assert Inventory.asset_metadata(warnings=warnings) == {}
    assert "could not be read" in warnings[0]

    def denied(*args, **kwargs):
        raise PermissionError("PRIVATE_ERROR_DETAIL")

    monkeypatch.setattr("pathlib.Path.open", denied)
    warnings = []
    assert Inventory.asset_metadata(warnings=warnings) == {}
    assert "PermissionError" in warnings[0]
    assert "PRIVATE_ERROR_DETAIL" not in warnings[0]


@pytest.mark.parametrize("value", [123, False, ["PRIVATE_VALUE"], {"PRIVATE_VALUE": 1}, "x\x00y"])
@pytest.mark.parametrize("field", ["env_setup", "workdir", "platform"])
def test_invalid_field_is_ignored_without_losing_other_valid_fields(context, field, value):
    spec = {
        "asset_id": ASSET_ID,
        "env_setup": "source /opt/env.sh",
        "workdir": "/work",
        "platform": "cuda",
        field: value,
    }
    write_hosts(context, {"PRIVATE_ENTRY_NAME": spec})
    warnings = []
    metadata = Inventory.asset_metadata(warnings=warnings)
    expected = {k: v for k, v in spec.items() if k not in ("asset_id", field)}
    assert metadata == {ASSET_ID: expected}
    assert len(warnings) == 1 and f"entry #1.{field}" in warnings[0]
    assert "PRIVATE" not in warnings[0] and "source /opt" not in warnings[0]


def test_unknown_platform_is_warned_and_empty_shell_fields_are_preserved(context):
    write_hosts(
        context,
        {
            "node": {
                "asset_id": ASSET_ID.upper(),
                "env_setup": "",
                "workdir": "",
                "platform": "PRIVATE_UNSUPPORTED_PLATFORM",
            }
        },
    )
    warnings = []
    assert Inventory.asset_metadata(warnings=warnings) == {
        ASSET_ID: {"env_setup": "", "workdir": ""}
    }
    assert len(warnings) == 1 and "supported platform key" in warnings[0]
    assert "PRIVATE" not in warnings[0]


@pytest.mark.parametrize("asset_id", [None, 123, False, "PRIVATE_INVALID_ID", ["PRIVATE_ID"]])
def test_metadata_needs_a_uuid_join_without_echoing_invalid_ids(context, asset_id):
    write_hosts(context, {"node": {"asset_id": asset_id, "workdir": "/work"}})
    warnings = []
    assert Inventory.asset_metadata(warnings=warnings) == {}
    assert len(warnings) == 1 and "valid asset_id UUID" in warnings[0]
    assert "PRIVATE" not in warnings[0]


def test_duplicate_uuid_overlay_is_not_order_dependent(context):
    write_hosts(
        context,
        {
            "one": {"asset_id": ASSET_ID, "workdir": "/one"},
            "two": {"asset_id": ASSET_ID.upper(), "workdir": "/two"},
            "three": {"asset_id": ASSET_ID, "env_setup": "PRIVATE_SCRIPT"},
        },
    )
    warnings = []
    assert Inventory.asset_metadata(warnings=warnings) == {}
    assert len(warnings) == 1 and "duplicate asset_id" in warnings[0]
    assert "PRIVATE_SCRIPT" not in warnings[0]


async def test_warning_is_visible_after_refresh_cache_restore_and_reload(context):
    context.path.write_text("hosts: [first, second]\n")
    service = BridgeService(context.cfg, context.settings)
    service.catalog._fetch = AsyncMock(return_value=[manifest()])
    try:
        status = await service.catalog.refresh()
        assert status["ok"] and status["last_error"] is None
        assert status["ssh_targets"] == 1 and status["metadata_warning_count"] == 1
        restored = AssetCatalog(context.cfg, context.settings)
        assert restored.status()["metadata_warnings"] == status["metadata_warnings"]
        assert restored.error is None
        visible = await service.discover(topic="status")
        assert visible["inventory"]["metadata_warnings"] == status["metadata_warnings"]
        reload = await service.call("jms_inventory_reload", {})
        assert reload["data"]["metadata_warnings"] == status["metadata_warnings"]
        service.catalog._fetch = AsyncMock(side_effect=RuntimeError("PRIVATE_API_ERROR"))
        failed = await service.catalog.refresh()
        assert not failed["ok"]
        assert failed["metadata_warnings"] == status["metadata_warnings"]
        assert "PRIVATE_API_ERROR" not in failed["last_error"]
        context.path.write_text("hosts: {}\n")
        service.catalog._fetch = AsyncMock(return_value=[manifest()])
        fixed = await service.catalog.refresh()
        assert fixed["ok"] and "metadata_warnings" not in fixed
    finally:
        await service.runner.close()


async def test_overlay_never_grants_targets_or_overrides_authorized_accounts(context):
    write_hosts(
        context,
        {
            "permitted": {
                "asset_id": ASSET_ID,
                "login": "other-user",
                "account": "admin",
                "workdir": "/work",
                "env_setup": 123,
            },
            "not-permitted": {"asset_id": OTHER_ID, "account": "root", "workdir": "/other"},
        },
    )
    catalog = AssetCatalog(context.cfg, context.settings)
    catalog._fetch = AsyncMock(return_value=[manifest(account="ubuntu", accounts=["ubuntu"])])
    assert (await catalog.refresh())["ok"]
    host = catalog.inventory.resolve([ASSET_ID])[0]
    assert host.login == "rest-user" and host.account == "ubuntu"
    assert host.workdir == "/work" and host.env_setup is None
    assert catalog.status()["metadata_warning_count"] == 1
    with pytest.raises(InventoryError):
        catalog.inventory.resolve([OTHER_ID])


async def test_many_warnings_remain_bounded_and_reload_keeps_structured_data(context):
    write_hosts(context, {f"PRIVATE_KEY_{i}": ["PRIVATE_VALUE"] for i in range(20)})
    service = BridgeService(context.cfg, context.settings)
    service.catalog._fetch = AsyncMock(return_value=[manifest()])
    try:
        result = await service.call("jms_inventory_reload", {})
        status = result["data"]
        assert status["metadata_warning_count"] == 20
        assert len(status["metadata_warnings"]) == 5 and status["metadata_warnings_truncated"]
        assert len(json.dumps(status).encode()) < 4096
        assert "PRIVATE" not in json.dumps(status)
        search = await service.discover(topic="assets")
        assert search["inventory"]["metadata_warnings"] == status["metadata_warnings"]
        assert len(json.dumps(search).encode()) < 4096
        assert "PRIVATE" not in json.dumps(search)
    finally:
        await service.runner.close()


@pytest.mark.parametrize("stage", ["read", "build", "persist"])
async def test_slow_refresh_io_runs_off_loop_and_publishes_atomically(context, monkeypatch, stage):
    write_hosts(context, {"node": {"asset_id": ASSET_ID, "workdir": "/old"}})
    catalog = AssetCatalog(context.cfg, context.settings)
    catalog._fetch = AsyncMock(return_value=[manifest()])
    await catalog.refresh()
    old_snapshot, old_inventory = catalog.snapshot, catalog.inventory
    write_hosts(context, {"node": {"asset_id": OTHER_ID, "workdir": "/new", "env_setup": 123}})
    catalog._fetch = AsyncMock(return_value=[manifest(OTHER_ID, name="replacement")])
    entered, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    if stage == "read":
        target, name = Inventory, "asset_metadata"
    elif stage == "build":
        target, name = catalog, "_build"
    else:
        target, name = rest_inventory, "atomic_json"
    original = getattr(target, name)

    def slow(*args, **kwargs):
        assert threading.get_ident() != loop_thread
        entered.set()
        assert release.wait(5), "test did not release storage worker"
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, slow)
    task = asyncio.create_task(catalog.refresh())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert not task.done()
        assert catalog.status()["refresh_in_progress"]
        assert catalog.snapshot is old_snapshot and catalog.inventory is old_inventory
        assert "metadata_warnings" not in catalog.status()
        assert catalog.inventory.hosts[ASSET_ID].workdir == "/old"
        assert catalog.search("node")["matched"] == 1
        # A stale view and elapsed retry pacing must not queue another refresh.
        catalog.snapshot["refreshed_at"] = 1
        catalog._last_attempt = 0
        assert await asyncio.wait_for(catalog.ensure(), 0.5) is old_inventory
    finally:
        release.set()
        status = await task
    assert status["ok"]
    assert catalog.snapshot is not old_snapshot and catalog.inventory is not old_inventory
    assert set(catalog.inventory.hosts) == {OTHER_ID}
    assert catalog.inventory.hosts[OTHER_ID].workdir == "/new"
    assert status["metadata_warning_count"] == 1
    assert not catalog.status()["refresh_in_progress"]


async def test_initial_ensure_waits_for_prepared_snapshot(context, monkeypatch):
    catalog = AssetCatalog(context.cfg, context.settings)
    catalog._fetch = AsyncMock(return_value=[manifest()])
    entered, release = threading.Event(), threading.Event()
    original = catalog._prepare

    def slow(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(catalog, "_prepare", slow)
    task = asyncio.create_task(catalog.refresh())
    waiter = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        waiter = asyncio.create_task(catalog.ensure())
        await asyncio.sleep(0)
        assert not waiter.done()
        assert catalog.status()["ssh_targets"] == 0
    finally:
        release.set()
        await task
        if waiter is not None:
            inventory = await waiter
    assert set(inventory.hosts) == {ASSET_ID}


def test_legacy_entries_without_overlay_are_ignored_without_warning(context):
    write_hosts(
        context,
        {
            "empty": None,
            "direct": {"ssh_host": "example.invalid"},
            "null-fields": {"asset_id": ASSET_ID, "env_setup": None},
        },
    )
    warnings = []
    assert Inventory.asset_metadata(warnings=warnings) == {}
    assert warnings == []


@pytest.mark.parametrize("disk_error", [False, True])
async def test_cancelled_persistence_keeps_lock_and_old_view_until_worker_finishes(
    context, monkeypatch, disk_error
):
    catalog = AssetCatalog(context.cfg, context.settings)
    catalog._fetch = AsyncMock(return_value=[manifest()])
    await catalog.refresh()
    catalog.error = "previous refresh error"
    old_snapshot, old_inventory = catalog.snapshot, catalog.inventory
    catalog._fetch = AsyncMock(return_value=[manifest(OTHER_ID, name="replacement")])
    entered, release = threading.Event(), threading.Event()
    original = rest_inventory.atomic_json

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        if disk_error:
            raise OSError("PRIVATE_DISK_DETAIL")
        return original(*args, **kwargs)

    monkeypatch.setattr(rest_inventory, "atomic_json", slow)
    task = asyncio.create_task(catalog.refresh())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and catalog._lock.locked()
        assert catalog.snapshot is old_snapshot and catalog.inventory is old_inventory
        assert catalog.error == "previous refresh error"
    finally:
        release.set()
        results = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)
    assert not catalog._lock.locked()
    assert catalog.snapshot is old_snapshot and catalog.inventory is old_inventory
    assert catalog.error == "previous refresh error"
    disk_asset = json.loads(catalog.cache.read_text())["assets"][0]["id"]
    assert disk_asset == (ASSET_ID if disk_error else OTHER_ID)
    monkeypatch.setattr(rest_inventory, "atomic_json", original)
    assert (await catalog.refresh())["ok"]
    assert set(catalog.inventory.hosts) == {OTHER_ID}


async def test_persistence_failure_keeps_metadata_diagnostics_and_complete_old_view(
    context, monkeypatch
):
    context.path.write_text("hosts: [broken, mapping]\n")
    catalog = AssetCatalog(context.cfg, context.settings)
    catalog._fetch = AsyncMock(return_value=[manifest()])
    await catalog.refresh()
    old_snapshot, old_inventory = catalog.snapshot, catalog.inventory
    old_disk = catalog.cache.read_bytes()
    context.path.write_text("hosts: {}\n")
    catalog._fetch = AsyncMock(return_value=[manifest(OTHER_ID, name="replacement")])

    def fail(*args):
        raise OSError("PRIVATE_DISK_DETAIL")

    monkeypatch.setattr(rest_inventory, "atomic_json", fail)
    status = await catalog.refresh()
    assert not status["ok"] and "OSError" in status["last_error"]
    assert "snapshot preparation" in status["last_error"]
    assert "local state storage" in status["last_error"]
    assert "API access" not in status["last_error"]
    assert "PRIVATE_DISK_DETAIL" not in status["last_error"]
    assert status["metadata_warning_count"] == 1
    assert catalog.snapshot is old_snapshot and catalog.inventory is old_inventory
    assert catalog.cache.read_bytes() == old_disk
