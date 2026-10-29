"""Private reusable shell channels; each filesystem probe is a fresh Bash process."""

import asyncio
import time
import uuid

from .connection_recovery import recover_startup


class FilesystemWorkers:
    def __init__(self, runner, *, limit=16, per_target=2, idle_seconds=300):
        self.runner, self.limit, self.per_target, self.idle_seconds = (
            runner,
            limit,
            per_target,
            idle_seconds,
        )
        self.entries = []
        self.lock = asyncio.Lock()
        self.sweeper = None
        self.closing = False
        self.hits, self.misses, self.fallbacks = 0, 0, 0
        self.startup_retries = 0

    def status(self):
        return dict(
            total=len(self.entries),
            busy=sum(e["busy"] for e in self.entries),
            limit=self.limit,
            per_target=self.per_target,
            idle_seconds=self.idle_seconds,
            reused=self.hits,
            created=self.misses,
            isolated_fallbacks=self.fallbacks,
            startup_retries=self.startup_retries,
        )

    async def retire(self, entry):
        await self.runner.close_session(entry["host"], entry["session_id"], _internal=True)

    async def sweep(self):
        async with self.lock:
            expired = [
                e
                for e in self.entries
                if not e["busy"] and time.monotonic() - e["last_used"] >= self.idle_seconds
            ]
            for entry in expired:
                self.entries.remove(entry)
                await self.retire(entry)

    async def loop(self):
        while True:
            await asyncio.sleep(min(30, self.idle_seconds))
            await self.sweep()

    async def execute(self, host, script, timeout, *, container=None):
        async def attempt():
            return await self._execute_once(host, script, timeout, container=container)

        def retried():
            self.startup_retries += 1

        return await recover_startup(self.runner, host, attempt, on_retry=retried)

    async def _execute_once(self, host, script, timeout, *, container=None):
        if self.sweeper is None:
            self.sweeper = asyncio.create_task(self.loop(), name="jms-fs-worker-cleanup")
        key = (self.runner.key(host), container)
        async with self.lock:
            if self.closing:
                raise ValueError("filesystem workers are closing")
            available = [e for e in self.entries if e["key"] == key and not e["busy"]]
            entry = available[0] if available else None
            if entry:
                self.hits += 1
            elif sum(e["key"] == key for e in self.entries) < self.per_target:
                if len(self.entries) >= self.limit:
                    idle = [e for e in self.entries if not e["busy"]]
                    if idle:
                        oldest = min(idle, key=lambda e: e["last_used"])
                        self.entries.remove(oldest)
                        await self.retire(oldest)
                if len(self.entries) < self.limit:
                    entry = dict(
                        key=key,
                        host=host,
                        session_id="_fs:" + uuid.uuid4().hex,
                        busy=False,
                        last_used=time.monotonic(),
                    )
                    self.entries.append(entry)
                    self.misses += 1
            if entry:
                entry["busy"] = True
            else:
                self.fallbacks += 1
        # Saturation uses the existing isolated execution path; global execution
        # slots still bound concurrency. No caller-owned shell state is involved.
        if entry is None:
            result = await self.runner.execute_to_store(
                host, script, timeout, container=container, _startup_probe=True
            )
        else:
            result = None
            try:
                result = await self.runner.execute_to_store(
                    host,
                    script,
                    timeout,
                    container=container,
                    session_id=entry["session_id"],
                    _internal_session=True,
                )
            finally:
                async with self.lock:
                    if result is None or not result.get("ok"):
                        try:
                            await self.retire(entry)
                        finally:
                            self.entries.remove(entry)
                    else:
                        entry.update(busy=False, last_used=time.monotonic())
        return result

    async def close(self):
        self.closing = True
        if self.sweeper:
            self.sweeper.cancel()
            await asyncio.gather(self.sweeper, return_exceptions=True)
            self.sweeper = None
        async with self.lock:
            for entry in list(self.entries):
                if entry["busy"]:
                    raise RuntimeError("filesystem work must drain before shutdown")
                await self.retire(entry)
                self.entries.remove(entry)
