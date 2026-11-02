"""Backpressure outcomes through compact execution, recovery and wrappers."""

import asyncio
import uuid
from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from jms_bridge.cluster import preflight
from jms_bridge.compact import BridgeService
from jms_bridge.config import BastionConfig
from jms_bridge.inventory import Host
from jms_bridge.runtime_config import BridgeSettings, load_settings
from jms_bridge.scheduling import Timings


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), 2)


@pytest.fixture
async def service(tmp_path):
    value = BridgeService(
        BastionConfig(url="https://benchmark.invalid", login_user="test", max_parallel=2),
        BridgeSettings(
            state_dir=tmp_path,
            execution_limit=2,
            inspection_reserved=1,
            queue_limit=0,
            inspection_queue_limit=1,
            queue_timeout=1,
        ),
    )
    hosts = [
        Host(name=f"node-{i}", asset_id=str(uuid.uuid4()), login="test", account="root")
        for i in range(2)
    ]
    inventory = SimpleNamespace(resolve=lambda targets: [h for h in hosts if h.name in targets])
    value.catalog.ensure = AsyncMock(return_value=inventory)
    value.catalog.status = lambda: {"stale": False}
    connection = SimpleNamespace(
        is_closed=lambda: False,
        create_process=AsyncMock(),
        close=lambda: None,
        wait_closed=AsyncMock(),
    )
    for host in hosts:
        value.runner._connections[value.runner.key(host)] = connection
    try:
        yield value, hosts, connection
    finally:
        for task in tuple(value.requests.tasks.values()):
            task.cancel()
        await value.requests.drain()
        await value.filesystem.workers.close()
        await value.catalog.close()
        await value.runner.close()


def assert_rejected(row, lane="general", limit=0, depth=0):
    assert row["reason"] == "queue_full" and row["retryable"] is True
    assert row["queue"] == {"lane": lane, "limit": limit, "depth": depth}
    assert row["executed"] is False and row["execution_state"] == "not_started"
    assert row["completion_unknown"] is False and row["queue_timed_out"] is False
    assert row["failure_stage"] == "execution_queue"
    if "exit_code" in row:
        assert row["exit_code"] == 75
    if "timed_out" in row:
        assert row["timed_out"] is False


async def test_run_rejection_survives_duplicate_and_restart_without_transmission(service):
    svc, hosts, connection = service
    host = hosts[0]
    async with svc.runner.scheduler.slot("general", 1, Timings()):
        first = await svc.run([host.name], script="printf NEVER", request_id="overload")
        assert not first["ok"] and first["request_state"] == "complete"
        assert not first["completion_unknown"] and "retryable" not in first
        row = first["results"][0]
        assert_rejected(row)
        assert svc.store.metadata(row["result_id"])["status"] == "complete"
    # Capacity is now idle; reusing the rejected request ID must not replay it.
    duplicate = await svc.run([host.name], script="printf NEVER", request_id="overload")
    assert duplicate["duplicate"] and not duplicate["execution_replayed"]
    assert duplicate["results"][0]["result_id"] == row["result_id"]
    assert svc.runner.scheduler.status()["queue_full_rejections"]["general"] == 1
    connection.create_process.assert_not_called()
    restarted = BridgeService(svc.cfg, svc.settings)
    try:
        saved = await restarted.request_status("overload")
        assert_rejected(saved["results"][0])
        assert_rejected(saved["execution_results"][0]["result_status"])
        assert not saved["completion_unknown"]
    finally:
        await restarted.runner.close()
    assert svc.store.retention_status()["active_results"] == 0
    assert not svc.requests.tasks


async def test_legacy_exec_and_rsync_preserve_machine_readable_rejection(service):
    svc, hosts, connection = service
    async with svc.runner.scheduler.slot("general", 1, Timings()):
        reply = await svc.legacy_exec([hosts[0].name], "printf NEVER")
        assert_rejected(reply["results"][0])
        with patch(
            "jms_bridge.native_ssh.asyncio.create_subprocess_exec", new=AsyncMock()
        ) as spawn:
            transfer = await svc.runner.rsync(hosts[0], "/unused/local", "/unused/remote")
            spawn.assert_not_called()
        assert_rejected(transfer.to_dict())
        assert_rejected(svc.store.metadata(transfer.result_id))
    connection.create_process.assert_not_called()
    assert not svc.runner.scheduler.waiters


async def test_filesystem_and_job_receipts_preserve_rejection(service, tmp_path):
    svc, hosts, connection = service
    local = tmp_path / "source.txt"
    local.write_text("must not be transferred")
    async with svc.runner.scheduler.slot("general", 1, Timings()):
        file_reply = await svc.fs({"target": hosts[0].name, "action": "stat", "path": "/tmp"})
        assert_rejected(file_reply)
        with patch(
            "jms_bridge.native_ssh.asyncio.create_subprocess_exec", new=AsyncMock()
        ) as spawn:
            transfer = await svc.fs(
                {
                    "target": hosts[0].name,
                    "action": "upload",
                    "local_path": str(local),
                    "path": "/unused/remote",
                }
            )
            spawn.assert_not_called()
        assert_rejected(transfer)
        receipts = []
        from jms_bridge.recoverable_jobs import RecoverableJobManager

        handle, result = await RecoverableJobManager(svc.runner, receipts.append).submit(
            hosts[0], "printf NEVER"
        )
        assert handle is None
        assert_rejected(result.to_dict())
        assert_rejected(receipts[-1])
        assert receipts[-1]["submission_state"] == "not_started"
        health = (await preflight(svc.runner, [hosts[0]]))[0]
        assert not health.ok and not health.reachable
        assert_rejected(health.to_dict())
    connection.create_process.assert_not_called()


async def test_inspection_has_separate_queue_and_reports_local_overload(service):
    svc, hosts, connection = service
    svc.runner.scheduler.queue_limits["inspection"] = 0
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(svc.runner.scheduler.slot("general", 1, Timings()))
        await stack.enter_async_context(svc.runner.scheduler.slot("inspection", 1, Timings()))
        reply = await svc.inspector.inspect([hosts[0].name], timeout=15)
    assert not reply["ok"] and reply["hosts"][0]["state"] == "unknown"
    assert not reply["hosts"][0]["whole_node_available"]
    assert_rejected(reply["hosts"][0]["backpressure"][0], lane="inspection")
    connection.create_process.assert_not_called()


async def test_call_wrapper_preserves_rejections_in_saved_execution_receipts(service):
    svc, hosts, connection = service
    svc.catalog.make_api = lambda: SimpleNamespace(aclose=AsyncMock())
    async with svc.runner.scheduler.slot("general", 1, Timings()):
        reply = await svc.call("jms_exec", {"targets": [hosts[0].name], "command": "printf NEVER"})
        submitted = await svc.call(
            "jms_submit_job",
            {"targets": [hosts[0].name], "command": "printf NEVER"},
            request_id="job-overload",
        )
    assert not reply["ok"]
    assert_rejected(reply["data"]["results"][0])
    data = submitted["data"]
    assert not submitted["ok"] and not submitted["completion_unknown"]
    assert_rejected(data["execution_results"][0])
    assert_rejected(data["submission_receipts"][0])
    connection.create_process.assert_not_called()


async def test_inspection_enrichment_rejection_preserves_successful_sample(service):
    svc, hosts, connection = service
    svc.runner.scheduler.queue_limits["inspection"] = 0
    original = svc.inspector.execute_json
    calls = 0

    async def execute(host, options, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            rid = svc.store.create(host=host.name)
            result = svc.store.finish(
                rid,
                ok=True,
                executed=True,
                completion_unknown=False,
                execution_state="completed",
                timing={},
            )
            return {"finished_at": 1}, result
        return await original(host, options, timeout)

    svc.inspector.execute_json = execute
    parsed = {
        "state": "busy",
        "whole_node_available": False,
        "errors": [],
        "warnings": [],
        "devices": [],
        "processes": [{"pid": 17, "user": "test", "name": "work", "devices": []}],
    }
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(svc.runner.scheduler.slot("general", 1, Timings()))
        await stack.enter_async_context(svc.runner.scheduler.slot("inspection", 1, Timings()))
        with patch("jms_bridge.accelerator_inspection.parse_snapshot", return_value=parsed):
            reply = await svc.inspector.inspect([hosts[0].name], timeout=15)
    row = reply["hosts"][0]
    assert reply["ok"] and row["state"] == "busy" and row["process_count"] == 1
    assert "execution_state" not in row and "retryable" not in row
    assert_rejected(row["backpressure"][0], lane="inspection")
    assert calls == 2
    connection.create_process.assert_not_called()


async def test_mixed_fanout_does_not_mark_successful_targets_retryable(service):
    svc, hosts, connection = service
    original = svc.runner.execute_to_store

    async def one(host, *args, **kwargs):
        if host is hosts[1]:
            rid = await svc.store.acreate(host=host.name, session_id=None)
            svc.runner.record_result(rid, host)
            return svc.store.finish(
                rid,
                ok=True,
                exit_code=0,
                executed=True,
                timed_out=False,
                elapsed_seconds=0,
                execution_state="completed",
                completion_unknown=False,
                connection_state="connected",
                session_state=None,
                reconnected=False,
                timing={},
                queue_timed_out=False,
                failure_stage=None,
            )
        return await original(host, *args, **kwargs)

    svc.runner.execute_to_store = one
    async with svc.runner.scheduler.slot("general", 1, Timings()):
        reply = await svc.run([h.name for h in hosts], script="printf MIXED", request_id="mixed")
    assert not reply["ok"] and not reply["completion_unknown"]
    assert "retryable" not in reply
    assert_rejected(reply["results"][0])
    assert reply["results"][1]["ok"] and "retryable" not in reply["results"][1]
    connection.create_process.assert_not_called()


async def test_cancel_at_full_queue_frees_waiting_space_without_counting_as_rejection(service):
    svc, hosts, connection = service
    svc.runner.scheduler.queue_limits["general"] = 1
    tasks = []
    async with svc.runner.scheduler.slot("general", 1, Timings()):
        try:
            waiting = asyncio.create_task(svc.runner.execute_to_store(hosts[0], "printf NEVER", 1))
            tasks.append(waiting)
            await until(lambda: svc.runner.scheduler.status()["queued"]["general"] == 1)
            refused = await svc.runner.execute_to_store(hosts[0], "printf REFUSED", 1)
            assert_rejected(refused, limit=1, depth=1)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            successor = asyncio.create_task(
                svc.runner.execute_to_store(hosts[0], "printf SUCCESSOR", 1, queue_timeout=0.1)
            )
            tasks.append(successor)
            await until(lambda: svc.runner.scheduler.status()["queued"]["general"] == 1)
            timed_out = await successor
            assert timed_out["queue_timed_out"] and timed_out["execution_state"] == "not_started"
            assert not timed_out["completion_unknown"] and "reason" not in timed_out
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    metas = [svc.store.metadata(p.name) for p in svc.store.root.iterdir() if p.is_dir()]
    assert len(metas) == 3 and all(m["status"] == "complete" for m in metas)
    assert all(m["executed"] is False and not m["completion_unknown"] for m in metas)
    assert svc.runner.scheduler.status()["queue_full_rejections"]["general"] == 1
    assert not svc.runner.scheduler.waiters and not sum(svc.runner.scheduler.active.values())
    connection.create_process.assert_not_called()


async def test_status_reports_actual_limits_and_zero_rejections(service):
    svc, hosts, _ = service
    status = (await svc.discover(topic="status"))["execution"]
    assert status["queue_limits"] == {"general": 0, "inspection": 1}
    assert status["queue_full_rejections"] == {"general": 0, "inspection": 0}
    assert "excludes login and session" in status["queue_scope"]


@pytest.mark.parametrize("name", ["queue_limit", "inspection_queue_limit"])
@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_queue_config_rejects_invalid_values(tmp_path, monkeypatch, name, value):
    path = tmp_path / "config.toml"
    literal = str(value).lower() if type(value) is bool else repr(value).replace("'", '"')
    path.write_text(f"[bridge]\n{name}={literal}\n")
    monkeypatch.setenv("JMS_BRIDGE_CONFIG", str(path))
    from jms_bridge.config import ConfigError

    with pytest.raises(ConfigError, match=name):
        load_settings()


def test_old_configuration_gets_finite_defaults_and_zero_is_valid(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text("[bridge]\n")
    monkeypatch.setenv("JMS_BRIDGE_CONFIG", str(path))
    assert load_settings().queue_limit == 64
    assert load_settings().inspection_queue_limit == 16
    path.write_text("[bridge]\nqueue_limit=0\ninspection_queue_limit=0\n")
    assert load_settings().queue_limit == load_settings().inspection_queue_limit == 0
