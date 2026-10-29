"""Drain one SSH identity before replacement; never close an in-use connection."""

import asyncio
import contextvars
import functools
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from .scheduling import QueueTimeout


@dataclass
class Target:
    connection: object = None
    generation: int = 0
    active: int = 0
    phase: str = "idle"
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    task: object = None
    cooldown_until: float = 0


def connection_operation(method):
    """Keep a lease for the whole command/transfer, including queue and cleanup."""

    @functools.wraps(method)
    async def wrapped(self, host, *args, **kwargs):
        async with self.connection_use(host):
            return await method(self, host, *args, **kwargs)

    return wrapped


def startup_failed(result):
    """Only a pre-transmission readiness failure is safe to retry."""
    return (
        result.get("failure_stage") == "shell_startup"
        and result.get("execution_state") == "not_started"
        and not result.get("completion_unknown")
        and not result.get("timed_out")
        and result.get("exit_code") == 75
    )


async def recover_startup(runner, host, attempt, *, on_retry=None):
    first = await attempt()
    if not startup_failed(first):
        return first
    if on_retry:
        on_retry()
    result = await attempt()
    if startup_failed(result):
        recovery = await runner.recover_connection(host, result.get("connection_generation"))
        if recovery["status"] in ("reset", "already_replaced"):
            previous = result["result_id"]
            result = await attempt()
            result = dict(result, connection_recovery=dict(recovery, previous_result_id=previous))
        else:
            result = dict(result, connection_recovery=recovery)
    return dict(result, startup_retry_result_id=first["result_id"])


def recover_isolated_startup(method):
    """Run outside the per-attempt lease so recovery can safely drain it."""

    @functools.wraps(method)
    async def wrapped(self, host, *args, **kwargs):
        if (
            kwargs.get("session_id")
            or kwargs.get("image")
            or kwargs.get("_startup_probe")
            or kwargs.get("_internal_session")
        ):
            return await method(self, host, *args, **kwargs)

        async def attempt():
            return await method(self, host, *args, **kwargs)

        return await recover_startup(self, host, attempt)

    return wrapped


class ConnectionRecovery:
    def __init__(self, *, cooldown_seconds=60):
        self.targets = {}
        self.cooldown_seconds = cooldown_seconds
        self.current = contextvars.ContextVar("ssh_connection_lease", default=None)
        self.closing = False
        self.resets = self.failed = self.coalesced = 0

    def state(self, key):
        return self.targets.setdefault(key, Target())

    def notify(self, key):
        state = self.state(key)
        event, state.changed = state.changed, asyncio.Event()
        event.set()

    def bind(self, key, connection):
        state = self.state(key)
        if connection is not state.connection:
            state.connection = connection
            state.generation += 1
            self.notify(key)
        lease = self.current.get()
        if lease and lease["owner"] is asyncio.current_task() and lease["key"] == key:
            lease["generation"] = state.generation
        return state.generation

    def generation(self, key):
        lease = self.current.get()
        if lease and lease["owner"] is asyncio.current_task() and lease["key"] == key:
            return lease["generation"]
        return self.state(key).generation

    @asynccontextmanager
    async def use(self, key):
        existing = self.current.get()
        owner = asyncio.current_task()
        if existing and existing["owner"] is owner and existing["key"] == key:
            yield
            return
        state = self.state(key)
        if self.closing:
            raise ValueError("SSH runner is closing")
        lease = dict(owner=owner, key=key, generation=state.generation, active=False)
        token = self.current.set(lease)
        try:
            yield
        finally:
            self.current.reset(token)
            if lease["active"]:
                state.active -= 1
                self.notify(key)

    async def enter(self, key, timeout):
        """Wait at connection acquisition, inside the operation's result handling."""
        lease = self.current.get()
        if not lease or lease["owner"] is not asyncio.current_task() or lease["key"] != key:
            return  # Internal reconnect callback; no user operation is executing.
        if lease["active"]:
            return
        state = self.state(key)
        deadline = time.monotonic() + timeout
        while state.phase in ("draining", "reconnecting"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise QueueTimeout("connection_recovery")
            try:
                await asyncio.wait_for(state.changed.wait(), remaining)
            except asyncio.TimeoutError:
                raise QueueTimeout("connection_recovery") from None
        if self.closing:
            raise ValueError("SSH runner is closing")
        # No await between the gate check and increment: replacement cannot race
        # a new operation, and cancellation cannot leak a lease.
        state.active += 1
        lease["active"] = True

    async def request(self, key, expected, replace, protected, *, wait_seconds=60):
        state = self.state(key)
        if self.closing:
            return dict(status="unavailable", reason="runner_closing")
        if not isinstance(expected, int) or expected < 1:
            return dict(status="unavailable", reason="no_connection_generation")
        lease = self.current.get()
        if lease and lease["owner"] is asyncio.current_task() and lease["key"] == key:
            return dict(status="deferred", reason="current_operation")
        if state.task and not state.task.done():
            self.coalesced += 1
        else:
            if state.generation != expected:
                return dict(status="already_replaced", generation=state.generation)
            remaining = state.cooldown_until - time.monotonic()
            if remaining > 0:
                return dict(status="cooldown", retry_after_seconds=max(1, round(remaining)))
            state.phase = "pending_sessions" if protected() else "draining"
            self.notify(key)
            state.task = asyncio.create_task(
                self._replace(key, expected, replace, protected), name="jms-ssh-target-recovery"
            )
        if protected():
            return dict(status="deferred", reason="persistent_shells")
        try:
            # Cancellation of a caller must not cancel a recovery already needed
            # by other callers. This task only reconnects; it never runs a query.
            return await asyncio.wait_for(asyncio.shield(state.task), wait_seconds)
        except asyncio.TimeoutError:
            return dict(status="deferred", reason=state.phase, active_uses=state.active)

    async def _replace(self, key, expected, replace, protected):
        state = self.state(key)
        try:
            while True:
                if state.generation != expected:
                    return dict(status="already_replaced", generation=state.generation)
                if protected():
                    state.phase = "pending_sessions"
                else:
                    state.phase = "draining"
                    if not state.active:
                        break
                self.notify(key)
                await state.changed.wait()
                # A public shell can be created by an operation which was in
                # flight before draining. Recheck it after every lease release.
            state.phase = "reconnecting"
            state.cooldown_until = time.monotonic() + self.cooldown_seconds
            started = time.monotonic()
            logging.getLogger("jms_bridge.performance").info(
                "%s",
                json.dumps(
                    dict(event="ssh_target_recovery_started", asset_id=key[-1], generation=expected)
                ),
            )
            await replace()
            self.resets += 1
            logging.getLogger("jms_bridge.performance").info(
                "%s",
                json.dumps(
                    dict(
                        event="ssh_target_recovery_finished",
                        asset_id=key[-1],
                        generation=state.generation,
                        status="reset",
                        elapsed_seconds=round(time.monotonic() - started, 3),
                    )
                ),
            )
            return dict(status="reset", generation=state.generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failed += 1
            logging.getLogger("jms_bridge.performance").info(
                "%s",
                json.dumps(
                    dict(
                        event="ssh_target_recovery_finished",
                        asset_id=key[-1],
                        status="failed",
                        error_type=type(exc).__name__,
                    )
                ),
            )
            return dict(status="failed", error_type=type(exc).__name__)
        finally:
            state.phase = "idle"
            self.notify(key)

    def status(self):
        return dict(
            resets=self.resets,
            failed=self.failed,
            coalesced=self.coalesced,
            cooldown_seconds=self.cooldown_seconds,
            pending=[
                dict(asset_id=k[-1], phase=s.phase, active_uses=s.active)
                for k, s in self.targets.items()
                if s.phase != "idle"
            ],
        )

    async def close(self):
        self.closing = True
        tasks = [s.task for s in self.targets.values() if s.task and not s.task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for key, state in self.targets.items():
            state.phase = "idle"
            self.notify(key)
