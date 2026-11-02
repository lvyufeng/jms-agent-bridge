import asyncio
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from test_execution_scheduling import LocalConnection

from jms_bridge.config import BastionConfig
from jms_bridge.filesystem import Filesystem
from jms_bridge.inventory import Host
from jms_bridge.native_ssh import NativeSSHRunner
from jms_bridge.results import ResultStore
from jms_bridge.runtime_config import BridgeSettings


class Workers(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = ResultStore(self.root / "results")
        self.runner = NativeSSHRunner(
            BastionConfig(url="https://bastion.example.com"),
            BridgeSettings(state_dir=self.root),
            store=self.store,
        )
        self.host = Host(
            name="local-test", asset_id=str(uuid.uuid4()), login="test", account="root"
        )
        self.connection = LocalConnection()
        self.runner._connections[self.runner.key(self.host)] = self.connection
        catalog = SimpleNamespace(
            ensure=AsyncMock(return_value=SimpleNamespace(resolve=lambda _: [self.host]))
        )
        self.fs = Filesystem(catalog, self.runner, self.store)
        self.path = self.root / "a file; literal"
        self.path.write_text("first\n中文🙂\n")

    async def asyncTearDown(self):
        await self.fs.workers.close()
        await self.runner.close()
        self.temp.cleanup()

    async def test_reuses_channel_but_reads_fresh_files_and_preserves_error_pages(self):
        args = dict(target="local-test", action="read", path=str(self.path))
        first = await self.fs.execute(**args)
        self.path.write_text("changed\n")
        second = await self.fs.execute(**args)
        self.assertEqual(first["data"]["text"], "first\n中文🙂\n")
        self.assertEqual(second["data"]["text"], "changed\n")
        self.assertEqual(len(self.connection.processes), 1)
        self.assertEqual(self.runner.sessions(), [])
        self.assertTrue(self.store.path(first["execution_result_id"], "stdout").exists())
        missing = await self.fs.execute(**(args | {"path": str(self.root / "missing")}))
        self.assertFalse(missing["ok"])
        self.assertEqual(missing["data"]["error"], "FileNotFoundError")
        self.assertEqual(len(self.connection.processes), 1)

    async def test_closed_idle_channel_rebuilds_without_changing_user_sessions(self):
        await self.fs.execute(target="x", action="stat", path=str(self.path))
        self.connection.processes[0].close()
        await self.connection.processes[0].wait_closed()
        result = await self.fs.execute(target="x", action="stat", path=str(self.path))
        self.assertTrue(result["ok"])
        self.assertEqual(len(self.connection.processes), 2)
        sid = self.fs.workers.entries[0]["session_id"]
        with self.assertRaises(ValueError):
            await self.runner.execute_to_store(self.host, "true", 1, session_id=sid)
        with self.assertRaises(ValueError):
            await self.runner.close_session(self.host, sid)

    async def test_timeout_retires_worker_and_next_call_starts_clean(self):
        result = await self.fs.workers.execute(self.host, "sleep 0.15", 0.02)
        self.assertTrue(result["timed_out"])
        self.assertTrue(result["completion_unknown"])
        self.assertEqual(self.fs.workers.status()["total"], 0)
        self.assertEqual(self.runner._shells, {})
        self.assertTrue(
            (await self.fs.execute(target="x", action="read", path=str(self.path)))["ok"]
        )

    async def test_saturation_falls_back_and_idle_workers_are_bounded(self):
        self.fs.workers.per_target = self.fs.workers.limit = 1
        task = asyncio.create_task(self.fs.workers.execute(self.host, "sleep 0.1; printf first", 1))
        while not self.fs.workers.entries:
            await asyncio.sleep(0.001)
        second = await self.fs.workers.execute(self.host, "printf second", 1)
        self.assertIsNone(second["session_id"])
        self.assertTrue((await task)["ok"])
        self.assertEqual(self.fs.workers.status()["isolated_fallbacks"], 1)
        self.assertEqual(len(self.fs.workers.entries), 1)
        self.fs.workers.entries[0]["last_used"] -= 400
        await self.fs.workers.sweep()
        self.assertEqual(self.runner._shells, {})
        self.assertEqual(self.fs.workers.status()["total"], 0)

    async def test_cancellation_does_not_leak_busy_worker(self):
        task = asyncio.create_task(self.fs.workers.execute(self.host, "sleep 0.2", 1))
        while not self.connection.processes:
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.02)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.fs.workers.status()["total"], 0)
        self.assertEqual(self.store.retention_status()["active_results"], 0)

    async def test_default_run_stays_isolated(self):
        await self.runner.execute_to_store(self.host, "export JMS_ISOLATION_TEST=one", 1)
        result = await self.runner.execute_to_store(
            self.host, 'printf "%s" "${JMS_ISOLATION_TEST-unset}"', 1
        )
        self.assertEqual(self.store.path(result["result_id"], "stdout").read_text(), "unset")

    async def install_startup_failure(self, count, *, stall=False):
        original = self.connection.create_process
        writes = []

        async def create(command, **kwargs):
            nonlocal count
            if count > 0:
                count -= 1
                process = await original("sleep 2" if stall else "sleep 0.01", **kwargs)
                writer = process.stdin

                class RecordingWriter:
                    def write(self, data):
                        writes.append(data)
                        writer.write(data)

                    async def drain(self):
                        await writer.drain()

                    def close(self):
                        writer.close()

                process.stdin = RecordingWriter()
                return process
            return await original(command, **kwargs)

        self.connection.create_process = create
        return writes

    async def test_startup_eof_rebuilds_once_without_sending_query_to_dead_channel(self):
        writes = await self.install_startup_failure(1)
        result = await self.fs.execute(target="x", action="read", path=str(self.path))
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["data"]["text"], self.path.read_text())
        self.assertEqual(len(self.connection.processes), 2)
        self.assertTrue(writes)
        self.assertTrue(all(b"READY" in p and b"__JMS_FS_BASH__" not in p for p in writes))
        failed = self.store.metadata(result["startup_retry_result_id"])
        self.assertEqual(failed["execution_state"], "not_started")
        self.assertFalse(failed["completion_unknown"])
        self.assertEqual(failed["failure_stage"], "shell_startup")
        self.assertEqual(self.fs.workers.status()["startup_retries"], 1)

    async def test_repeated_startup_eof_is_bounded_and_never_completion_unknown(self):
        self.runner.recover_connection = AsyncMock(return_value=dict(status="failed"))
        await self.install_startup_failure(2)
        result = await self.fs.execute(target="x", action="list", path=str(self.root))
        self.assertFalse(result["ok"])
        self.assertEqual(result["execution_state"], "not_started")
        self.assertFalse(result["completion_unknown"])
        self.assertEqual(result["failure_stage"], "shell_startup")
        self.assertEqual(len(self.connection.processes), 2)
        self.assertEqual(self.fs.workers.status()["total"], 0)
        self.assertEqual(self.runner._shells, {})

    async def test_after_transmission_eof_is_not_replayed(self):
        result = await self.fs.workers.execute(self.host, "printf partial; exit", 1)
        self.assertFalse(result["ok"])
        self.assertTrue(result["completion_unknown"])
        self.assertEqual(result["failure_stage"], "execution")
        self.assertEqual(len(self.connection.processes), 1)
        self.assertEqual(self.fs.workers.status()["startup_retries"], 0)

    async def test_startup_timeout_is_not_retried_or_marked_as_transmitted(self):
        await self.install_startup_failure(1, stall=True)
        result = await self.fs.workers.execute(self.host, "printf query", 0.03)
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["completion_unknown"])
        self.assertEqual(result["execution_state"], "not_started")
        self.assertEqual(len(self.connection.processes), 1)
        self.assertEqual(self.fs.workers.status()["startup_retries"], 0)

    async def test_saturated_pool_fallback_also_confirms_startup_before_script(self):
        self.fs.workers.limit = 0
        self.runner.recover_connection = AsyncMock(return_value=dict(status="failed"))
        writes = await self.install_startup_failure(2)
        result = await self.fs.workers.execute(self.host, "printf secret_query", 1)
        self.assertFalse(result["ok"])
        self.assertFalse(result["completion_unknown"])
        self.assertEqual(result["failure_stage"], "shell_startup")
        self.assertTrue(all(b"secret_query" not in p for p in writes))

    async def test_user_named_shell_is_not_automatically_rebuilt(self):
        result = await self.runner.execute_to_store(
            self.host, "printf partial; exit", 1, session_id="user-shell"
        )
        self.assertTrue(result["completion_unknown"])
        self.assertEqual(len(self.connection.processes), 1)
        self.assertEqual(self.fs.workers.status()["startup_retries"], 0)
