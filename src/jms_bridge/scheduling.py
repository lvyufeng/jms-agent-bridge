"""Bound active commands independently of authentication and idle connections."""

import asyncio
import contextlib
import time
from collections import deque


class QueueTimeout(Exception):
    def __init__(self, stage):
        self.stage = stage
        super().__init__(f"{stage} timed out; command was not sent")


class QueueFull(Exception):
    """Execution wait capacity is exhausted; the new command was not sent."""

    def __init__(self, lane, limit, depth):
        self.lane, self.limit, self.depth = lane, limit, depth
        self.stage = "execution_queue"
        super().__init__(f"{lane} execution queue is full ({depth}/{limit}); command was not sent")

    def response(self):
        return {
            "reason": "queue_full",
            "retryable": True,
            "queue": {"lane": self.lane, "limit": self.limit, "depth": self.depth},
        }


def queue_rejection(metadata):
    """Expose per-execution rejection evidence, never whole-fanout retry advice."""
    if metadata.get("reason") != "queue_full":
        return {}
    return {
        key: metadata[key]
        for key in (
            "reason",
            "retryable",
            "queue",
            "executed",
            "execution_state",
            "completion_unknown",
            "queue_timed_out",
            "failure_stage",
        )
        if key in metadata
    }


class Timings:
    def __init__(self):
        self.values = dict(
            queue_seconds=0.0, connection_seconds=0.0, channel_seconds=0.0, execution_seconds=0.0
        )
        self.connection_details = {}

    @contextlib.contextmanager
    def connection_phase(self, name):
        started = time.monotonic()
        try:
            yield
        finally:
            self.connection_details[name] = (
                self.connection_details.get(name, 0) + time.monotonic() - started
            )

    def connection_snapshot(self):
        return {key: round(value, 6) for key, value in self.connection_details.items()}

    @contextlib.contextmanager
    def phase(self, name):
        started = time.monotonic()
        try:
            yield
        finally:
            self.values[name] += time.monotonic() - started

    def snapshot(self):
        return {k: round(v, 3) for k, v in self.values.items()}


class ExecutionSlots:
    """FIFO among eligible requests; inspection may use otherwise idle slots.

    General work cannot consume the inspection reserve. Inspection may use all
    slots when general work is absent. Authentication never holds a slot.
    Each lane has its own finite waiting allowance; immediate grants do not count
    toward it, and a full waiting lane rejects new arrivals before creating a waiter.
    Queue limits do not bound authentication, session or request-level waits.
    """

    def __init__(self, limit, inspection_reserved=2, *, queue_limit=64, inspection_queue_limit=16):
        if type(limit) is not int or limit < 1:
            raise ValueError("execution limit must be a positive integer")
        if type(inspection_reserved) is not int or not 0 <= inspection_reserved < limit:
            raise ValueError("inspection_reserved must be in 0..limit-1")
        for name, value in (
            ("queue_limit", queue_limit),
            ("inspection_queue_limit", inspection_queue_limit),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.limit = limit
        self.inspection_reserved = inspection_reserved
        self.queue_limits = {"general": queue_limit, "inspection": inspection_queue_limit}
        self.queue_full_rejections = dict(general=0, inspection=0)
        self.active = dict(general=0, inspection=0)
        self.waiters = deque()
        self.peak_active = 0

    def _eligible(self, lane):
        return sum(self.active.values()) < self.limit and (
            lane == "inspection" or self.active["general"] < self.limit - self.inspection_reserved
        )

    def _wake(self):
        for waiter in list(self.waiters):
            if waiter["future"].cancelled():
                self.waiters.remove(waiter)
            elif self._eligible(waiter["lane"]):
                self.waiters.remove(waiter)
                self.active[waiter["lane"]] += 1
                waiter["granted"] = True
                self.peak_active = max(self.peak_active, sum(self.active.values()))
                waiter["future"].set_result(None)

    def _release(self, lane):
        self.active[lane] -= 1
        self._wake()

    @contextlib.asynccontextmanager
    async def slot(self, lane, timeout, timings):
        if lane not in self.active:
            raise ValueError("unknown execution lane")
        if timeout <= 0:
            raise ValueError("queue timeout must be positive")
        self._wake()
        depth = sum(w["lane"] == lane for w in self.waiters)
        if not self._eligible(lane) and depth >= self.queue_limits[lane]:
            self.queue_full_rejections[lane] += 1
            raise QueueFull(lane, self.queue_limits[lane], depth)
        # No await between admission and append/grant: concurrent arrivals cannot
        # overbook waiting capacity. Immediate grants do not consume queue space.
        waiter = dict(lane=lane, future=asyncio.get_running_loop().create_future(), granted=False)
        self.waiters.append(waiter)
        self._wake()
        try:
            with timings.phase("queue_seconds"):
                # wait_for on older Python can swallow caller cancellation when
                # its inner future has just completed. wait leaves that future
                # untouched and lets cancellation propagate to our slot cleanup.
                done, _ = await asyncio.wait((waiter["future"],), timeout=timeout)
                if not done:
                    raise QueueTimeout("execution_queue")
            yield
        finally:
            if waiter["granted"]:
                self._release(lane)
            else:
                waiter["future"].cancel()
                self.waiters.remove(waiter)
                self._wake()

    def status(self):
        return {
            "limit": self.limit,
            "inspection_reserved": self.inspection_reserved,
            "general_limit": self.limit - self.inspection_reserved,
            "active": dict(self.active),
            "queued": {lane: sum(w["lane"] == lane for w in self.waiters) for lane in self.active},
            "queue_limits": dict(self.queue_limits),
            "queue_full_rejections": dict(self.queue_full_rejections),
            "queue_scope": "execution-slot waiters only; excludes login and session waits",
            "peak_active": self.peak_active,
        }
