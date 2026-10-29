import asyncio
import unittest

from jms_bridge.connection_recovery import ConnectionRecovery
from jms_bridge.scheduling import QueueTimeout

A, B = ("user", "root", "asset-a"), ("user", "root", "asset-b")


class Recovery(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = ConnectionRecovery()
        self.addAsyncCleanup(self.manager.close)
        self.manager.bind(A, object())
        self.manager.bind(B, object())
        self.replaced = 0

    async def replace(self):
        self.assertEqual(self.manager.state(A).active, 0)
        self.replaced += 1
        self.manager.bind(A, object())

    async def entered_operation(self, key, entered, release):
        async with self.manager.use(key):
            await self.manager.enter(key, 1)
            entered.set()
            await release.wait()

    async def test_drains_only_target_and_coalesces_parallel_recovery(self):
        entered, release, new_entered, new_release = [asyncio.Event() for _ in range(4)]
        active = asyncio.create_task(self.entered_operation(A, entered, release))
        await entered.wait()
        calls = [
            asyncio.create_task(self.manager.request(A, 1, self.replace, lambda: False))
            for _ in range(3)
        ]
        while self.manager.state(A).phase != "draining":
            await asyncio.sleep(0)
        waiting = asyncio.create_task(self.entered_operation(A, new_entered, new_release))
        async with self.manager.use(B):
            await self.manager.enter(B, 1)
            self.assertEqual(self.manager.state(B).active, 1)
            self.assertEqual(self.manager.state(A).active, 1)
            self.assertFalse(new_entered.is_set())
            self.assertEqual(self.replaced, 0)
        release.set()
        await active
        values = await asyncio.gather(*calls)
        await new_entered.wait()
        self.assertTrue(all(v["status"] in ("reset", "already_replaced") for v in values))
        self.assertEqual(self.replaced, 1)
        self.assertEqual(self.manager.state(B).generation, 1)
        new_release.set()
        await waiting

    async def test_named_shell_defers_without_blocking_existing_task(self):
        protected = True
        value = await self.manager.request(A, 1, self.replace, lambda: protected)
        self.assertEqual(value, dict(status="deferred", reason="persistent_shells"))
        async with self.manager.use(A):
            await self.manager.enter(A, 1)
            self.assertEqual(self.replaced, 0)
        protected = False
        self.manager.notify(A)
        await self.manager.state(A).task
        self.assertEqual(self.replaced, 1)

    async def test_shell_created_by_inflight_operation_is_rechecked_before_reset(self):
        protected = False
        async with self.manager.use(A):
            await self.manager.enter(A, 1)
            call = asyncio.create_task(self.manager.request(A, 1, self.replace, lambda: protected))
            while self.manager.state(A).phase != "draining":
                await asyncio.sleep(0)
            protected = True
        value = await asyncio.wait_for(call, 0.02) if call.done() else None
        self.assertIsNone(value)
        while self.manager.state(A).phase != "pending_sessions":
            await asyncio.sleep(0)
        self.assertEqual(self.replaced, 0)
        protected = False
        self.manager.notify(A)
        self.assertEqual((await call)["status"], "reset")

    async def test_cancel_waiter_does_not_cancel_shared_recovery(self):
        async with self.manager.use(A):
            await self.manager.enter(A, 1)
            task = asyncio.create_task(self.manager.request(A, 1, self.replace, lambda: False))
            while self.manager.state(A).task is None:
                await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(self.manager.state(A).task.cancelled())
        await self.manager.state(A).task
        self.assertEqual(self.replaced, 1)

    async def test_busy_wait_has_bounded_response_but_recovery_continues(self):
        async with self.manager.use(A):
            await self.manager.enter(A, 1)

            async def request():
                return await self.manager.request(
                    A, 1, self.replace, lambda: False, wait_seconds=0.01
                )

            value = await asyncio.create_task(request())
            self.assertEqual(value["status"], "deferred")
            self.assertEqual(value["active_uses"], 1)
            self.assertEqual(self.replaced, 0)
        await self.manager.state(A).task
        self.assertEqual(self.replaced, 1)

    async def test_new_waiter_timeout_or_cancel_never_leaks_a_lease(self):
        self.manager.state(A).phase = "draining"
        with self.assertRaises(QueueTimeout):
            async with self.manager.use(A):
                await self.manager.enter(A, 0.01)
        self.assertEqual(self.manager.state(A).active, 0)
        entered, release = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(self.entered_operation(A, entered, release))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.manager.state(A).active, 0)

    async def test_cooldown_and_stale_generation_do_not_reset_again(self):
        first = await self.manager.request(A, 1, self.replace, lambda: False)
        self.assertEqual(first["status"], "reset")
        stale = await self.manager.request(A, 1, self.replace, lambda: False)
        self.assertEqual(stale["status"], "already_replaced")
        current = await self.manager.request(A, 2, self.replace, lambda: False)
        self.assertEqual(current["status"], "cooldown")
        self.assertEqual(self.replaced, 1)

    async def test_failed_authentication_releases_gate_and_applies_cooldown(self):
        async def failure():
            raise ConnectionError("secret must not appear in the response")

        value = await self.manager.request(A, 1, failure, lambda: False)
        self.assertEqual(value, dict(status="failed", error_type="ConnectionError"))
        async with self.manager.use(A):
            await self.manager.enter(A, 0.01)
        self.assertEqual(
            (await self.manager.request(A, 1, failure, lambda: False))["status"], "cooldown"
        )

    async def test_generation_is_owned_by_task_not_inherited_child_context(self):
        async with self.manager.use(A):
            await self.manager.enter(A, 1)
            connection = object()
            self.manager.bind(A, connection)
            original = self.manager.generation(A)

            async def child():
                self.manager.bind(A, object())

            await asyncio.create_task(child())
            self.assertEqual(self.manager.generation(A), original)
            self.assertGreater(self.manager.state(A).generation, original)

    async def test_close_cancels_deferred_recovery_without_disconnect(self):
        await self.manager.request(A, 1, self.replace, lambda: True)
        await self.manager.close()
        self.assertEqual(self.replaced, 0)
        self.assertEqual(self.manager.status()["pending"], [])
