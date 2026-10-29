import asyncio
import json
import shlex
import tempfile
import unittest
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from jms_bridge.inventory import Host
from jms_bridge.native_ssh import NativeSSHRunner
from jms_bridge.results import ResultStore
from jms_bridge.runtime_config import BridgeSettings
from jms_bridge.scheduling import ExecutionSlots, QueueTimeout, Timings


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_reserve_and_global_eight_limit(self):
        scheduler = ExecutionSlots(8, 2)
        async with AsyncExitStack() as stack:
            for _ in range(6):
                await stack.enter_async_context(scheduler.slot("general", 1, Timings()))
            with self.assertRaises(QueueTimeout):
                async with scheduler.slot("general", 0.02, Timings()):
                    self.fail("general work borrowed reserved capacity")
            for _ in range(2):
                await stack.enter_async_context(scheduler.slot("inspection", 1, Timings()))
            with self.assertRaises(QueueTimeout):
                async with scheduler.slot("inspection", 0.02, Timings()):
                    self.fail("exceeded total limit")
            self.assertEqual(scheduler.status()["active"], {"general": 6, "inspection": 2})
        self.assertEqual(sum(scheduler.active.values()), 0)
        self.assertEqual(scheduler.peak_active, 8)

    async def test_inspection_can_use_all_idle_capacity(self):
        scheduler = ExecutionSlots(8, 2)
        async with AsyncExitStack() as stack:
            for _ in range(8):
                await stack.enter_async_context(scheduler.slot("inspection", 1, Timings()))
            self.assertEqual(scheduler.active["inspection"], 8)

    async def test_cancel_waiter_and_granted_task_release_capacity(self):
        scheduler = ExecutionSlots(1, 0)
        async with scheduler.slot("general", 1, Timings()):

            async def waiting():
                async with scheduler.slot("general", 1, Timings()):
                    await asyncio.sleep(10)

            task = asyncio.create_task(waiting())
            await asyncio.sleep(0.01)
            self.assertEqual(scheduler.status()["queued"]["general"], 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(scheduler.status()["queued"]["general"], 0)
        task = asyncio.create_task(waiting())
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(sum(scheduler.active.values()), 0)
        async with scheduler.slot("general", 0.1, Timings()):
            pass


class LocalProcess:
    def __init__(self, process):
        self.process = process
        self.stdin, self.stdout, self.stderr = process.stdin, process.stdout, process.stderr

    @property
    def exit_status(self):
        return self.process.returncode

    async def wait_closed(self):
        await self.process.wait()
        self.process._transport.close()
        await asyncio.sleep(0)

    def is_closing(self):
        return self.process.returncode is not None

    def close(self):
        self.stdin.close()
        if self.process.returncode is None:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass


class LocalConnection:
    def __init__(self):
        self.processes = []

    def is_closed(self):
        return False

    async def create_process(self, command, **kwargs):
        proc = await asyncio.create_subprocess_exec(
            *shlex.split(command),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        wrapped = LocalProcess(proc)
        self.processes.append(wrapped)
        return wrapped

    def close(self):
        for p in self.processes:
            p.close()

    async def wait_closed(self):
        await asyncio.gather(*(p.wait_closed() for p in self.processes))


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ResultStore(Path(self.tmp.name) / "results")
        self.runner = NativeSSHRunner(
            SimpleNamespace(max_parallel=8),
            BridgeSettings(state_dir=Path(self.tmp.name), inspection_reserved=2, queue_timeout=0.1),
            store=self.store,
        )
        self.host = Host(name="test", asset_id=str(uuid.uuid4()), login="test", account="root")
        self.connection = LocalConnection()
        self.runner._connections[self.runner.key(self.host)] = self.connection

    async def asyncTearDown(self):
        await self.runner.close()
        self.tmp.cleanup()

    async def test_reads_configured_limit(self):
        self.assertEqual(self.runner.scheduler.limit, 8)
        self.assertEqual(self.runner.scheduler.inspection_reserved, 2)

    async def test_inspection_runs_while_six_general_jobs_hold_capacity(self):
        async with AsyncExitStack() as stack:
            for _ in range(6):
                await stack.enter_async_context(self.runner.scheduler.slot("general", 1, Timings()))
            general = asyncio.create_task(
                self.runner.execute_to_store(self.host, "echo general", 1)
            )
            inspected = await self.runner.execute_to_store(
                self.host, "echo inspected", 1, lane="inspection"
            )
            self.assertTrue(inspected["ok"])
            self.assertLess(inspected["timing"]["queue_seconds"], 0.05)
            blocked = await general
            self.assertTrue(blocked["queue_timed_out"])
            self.assertEqual(blocked["execution_state"], "not_started")
            self.assertFalse(blocked["executed"])
            self.assertFalse(blocked["completion_unknown"])
            self.assertEqual(blocked["failure_stage"], "execution_queue")
            self.assertEqual(len(self.connection.processes), 1)

    async def test_cancel_queued_run_finishes_metadata_without_transmission(self):
        async with AsyncExitStack() as stack:
            for _ in range(6):
                await stack.enter_async_context(self.runner.scheduler.slot("general", 1, Timings()))
            task = asyncio.create_task(
                self.runner.execute_to_store(self.host, "echo must-not-run", 1)
            )
            await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            meta = json.loads(next(self.store.root.glob("*/meta.json")).read_text())
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["execution_state"], "not_started")
            self.assertFalse(meta["completion_unknown"])
            self.assertEqual(self.runner.scheduler.status()["queued"]["general"], 0)
            self.assertEqual(len(self.connection.processes), 0)

    async def test_cold_login_does_not_occupy_execution_slot(self):
        cold = Host(name="cold", asset_id=str(uuid.uuid4()), login="test", account="root")
        started, release = asyncio.Event(), asyncio.Event()
        original = self.runner._connection

        async def connect(host):
            if host is cold:
                started.set()
                await release.wait()
                self.runner._connections[self.runner.key(host)] = self.connection
            return await original(host)

        self.runner._connection = connect
        task = asyncio.create_task(self.runner.execute_to_store(cold, "echo cold", 1))
        await started.wait()
        self.assertEqual(sum(self.runner.scheduler.active.values()), 0)
        warm = await self.runner.execute_to_store(self.host, "echo warm", 1)
        self.assertTrue(warm["ok"])
        self.assertFalse(task.done())
        release.set()
        self.assertTrue((await task)["ok"])

    async def test_execution_timeout_is_distinct_from_queue_timeout(self):
        result = await self.runner.execute_to_store(self.host, "sleep 1", 0.02)
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["queue_timed_out"])
        self.assertTrue(result["completion_unknown"])
        self.assertEqual(result["failure_stage"], "execution")
        self.assertEqual(sum(self.runner.scheduler.active.values()), 0)

    async def test_named_session_state_and_wait_do_not_consume_extra_slots(self):
        first = await self.runner.execute_to_store(
            self.host, "export JMS_TEST_VALUE=42", 1, session_id="s"
        )
        self.assertTrue(first["ok"])
        key = (self.runner.key(self.host), "s")
        gate = self.runner._session_gates[key]
        await gate.acquire()
        try:
            blocked = await self.runner.execute_to_store(
                self.host, "echo unsafe", 1, session_id="s"
            )
            self.assertTrue(blocked["queue_timed_out"])
            self.assertEqual(blocked["failure_stage"], "session_queue")
            self.assertEqual(blocked["session_state"], "active")
            self.assertEqual(sum(self.runner.scheduler.active.values()), 0)
            self.assertFalse(self.runner._shells[key].lost)
        finally:
            gate.release()
        second = await self.runner.execute_to_store(
            self.host, 'echo "$JMS_TEST_VALUE"', 1, session_id="s"
        )
        self.assertTrue(second["ok"])
        self.assertEqual(self.store.path(second["result_id"], "stdout").read_text(), "42\n")
        await self.runner.close_session(self.host, "s")

    async def test_rsync_queue_timeout_has_durable_not_started_result(self):
        async with AsyncExitStack() as stack:
            for _ in range(6):
                await stack.enter_async_context(self.runner.scheduler.slot("general", 1, Timings()))
            with patch(
                "jms_bridge.native_ssh.asyncio.create_subprocess_exec", new=AsyncMock()
            ) as spawn:
                result = await self.runner.rsync(self.host, "/unused/local", "/unused/remote")
                spawn.assert_not_called()
            meta = self.store.metadata(result.result_id)
            self.assertTrue(meta["queue_timed_out"])
            self.assertEqual(meta["execution_state"], "not_started")
            self.assertFalse(meta["completion_unknown"])

    async def test_rsync_success_still_transfers_and_records_timings(self):
        source = Path(self.tmp.name) / "source.txt"
        destination = Path(self.tmp.name) / "copied.txt"
        source.write_text("bounded transfer test\n")
        result = await self.runner.rsync(self.host, str(source), str(destination), timeout=3)
        self.assertTrue(result.ok, result.stderr)
        self.assertEqual(destination.read_text(), source.read_text())
        meta = self.store.metadata(result.result_id)
        self.assertEqual(meta["execution_state"], "completed")
        self.assertTrue(meta["executed"])
        self.assertFalse(meta["completion_unknown"])
        self.assertGreater(meta["timing"]["execution_seconds"], 0)
        self.assertEqual(sum(self.runner.scheduler.active.values()), 0)

    async def test_rsync_cancel_queued_transfer_finalizes_result(self):
        async with AsyncExitStack() as stack:
            for _ in range(6):
                await stack.enter_async_context(self.runner.scheduler.slot("general", 1, Timings()))
            task = asyncio.create_task(
                self.runner.rsync(self.host, "/unused/local", "/unused/remote")
            )
            for _ in range(200):
                if self.runner.scheduler.status()["queued"]["general"]:
                    break
                await asyncio.sleep(0.01)
            else:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                self.fail("transfer never reached the execution queue")
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            meta = json.loads(next(self.store.root.glob("*/meta.json")).read_text())
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["exit_code"], 130)
            self.assertFalse(meta["completion_unknown"])

    async def test_ten_requests_have_six_active_two_queued_two_rejected_and_inspection_runs(self):
        self.runner.scheduler = ExecutionSlots(8, 2, queue_limit=2, inspection_queue_limit=1)
        self.runner.queue_timeout = 3
        release = Path(self.tmp.name) / "release"
        script = (
            "printf STARTED; while [ ! -e "
            + shlex.quote(str(release))
            + " ]; do sleep 0.01; done; printf DONE"
        )
        tasks = [
            asyncio.create_task(self.runner.execute_to_store(self.host, script, 3))
            for _ in range(10)
        ]

        async def saturated():
            while True:
                status = self.runner.scheduler.status()
                if (
                    status["active"]["general"] == 6
                    and status["queued"]["general"] == 2
                    and sum(t.done() for t in tasks) == 2
                    and len(self.connection.processes) == 6
                ):
                    break
                await asyncio.sleep(0.005)

        try:
            await asyncio.wait_for(saturated(), 2)
            self.assertEqual(self.runner.scheduler.status()["queue_full_rejections"]["general"], 2)
            rejected = [t.result() for t in tasks if t.done()]
            self.assertTrue(all(r["reason"] == "queue_full" for r in rejected))
            self.assertTrue(all(r["executed"] is False for r in rejected))
            inspector = await self.runner.execute_to_store(
                self.host, "printf inspected", 1, lane="inspection"
            )
            self.assertTrue(inspector["ok"])
            self.assertEqual(self.runner.scheduler.status()["queued"]["general"], 2)
            release.touch()
            rows = await asyncio.wait_for(asyncio.gather(*tasks), 3)
            self.assertEqual(sum(r["ok"] for r in rows), 8)
            self.assertEqual(sum(r.get("reason") == "queue_full" for r in rows), 2)
            self.assertEqual(len(self.connection.processes), 9)
            self.assertEqual(self.store.retention_status()["active_results"], 0)
        finally:
            release.touch()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(sum(self.runner.scheduler.active.values()), 0)
        self.assertEqual(len(self.runner.scheduler.waiters), 0)

    async def test_named_session_queue_rejection_releases_gate_without_losing_shell(self):
        first = await self.runner.execute_to_store(
            self.host, "export JMS_TEST_VALUE=42", 1, session_id="s"
        )
        self.assertTrue(first["ok"])
        self.runner.scheduler.queue_limits["general"] = 0
        async with AsyncExitStack() as stack:
            for _ in range(6):
                await stack.enter_async_context(self.runner.scheduler.slot("general", 1, Timings()))
            rejected = await self.runner.execute_to_store(
                self.host, "echo unsafe", 1, session_id="s"
            )
            self.assertEqual(rejected["reason"], "queue_full")
            self.assertFalse(rejected["executed"])
            self.assertEqual(rejected["session_state"], "active")
            key = (self.runner.key(self.host), "s")
            self.assertFalse(self.runner._session_gates[key].locked())
            self.assertFalse(self.runner._shells[key].lost)
        second = await self.runner.execute_to_store(
            self.host, 'echo "$JMS_TEST_VALUE"', 1, session_id="s"
        )
        self.assertTrue(second["ok"])
        self.assertEqual(self.store.path(second["result_id"], "stdout").read_text(), "42\n")
        self.assertEqual(len(self.connection.processes), 1)


if __name__ == "__main__":
    unittest.main()
