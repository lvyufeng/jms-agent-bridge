"""Scheduler-only regression tests for ExecutionSlots admission and backpressure.

No SSH, runner or store: just the slot scheduler. Waits are bounded - an ``Event``
through ``wait_for``, a scheduler transition through ``until`` - so a failure is a
scheduler verdict, not a hang. Each test's tasks live in its ``q``; that fixture
cancels and gathers them on the way out and then asserts the scheduler drained.
"""

import asyncio

import pytest

from jms_bridge.scheduling import ExecutionSlots, QueueFull, QueueTimeout, Timings

POLL, TIMEOUT, STEP = 0.005, 2.0, 0.05
EMPTY = {"general": 0, "inspection": 0}


async def until(predicate, timeout=TIMEOUT):
    """Poll a scheduler predicate in bounded steps; only ever used to observe."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            return False
        await asyncio.sleep(POLL)
    return True


class Queue:
    """One scheduler for one event loop, plus the tasks its requests run in."""

    def __init__(self, limit=1, reserved=0, cap=0):
        self.scheduler = ExecutionSlots(
            limit, reserved, queue_limit=cap, inspection_queue_limit=cap
        )
        self.tasks, self.entered, self.errors = [], [], []

    def spawn(self, lane="general", timeout=5, body=None):
        """Admit ``lane``, record it in ``entered``/``errors``, then run ``body``."""
        scheduler = self.scheduler

        async def request():
            try:
                async with scheduler.slot(lane, timeout, Timings()):
                    self.entered.append(lane)
                    if body is not None:
                        await body()
            except (QueueFull, QueueTimeout) as error:
                self.errors.append(error)

        task = asyncio.create_task(request())
        self.tasks.append(task)
        return task

    def depth(self, lane):
        """``lane`` slots held plus requests waiting in its queue."""
        return self.scheduler.active[lane] + self.scheduler.status()["queued"][lane]

    async def hold(self, lanes, release):
        """Start one blocking task per lane, returning once each is inside its slot."""
        before = {lane: self.depth(lane) for lane in lanes}
        added = {lane: lanes.count(lane) for lane in lanes}
        tasks = [self.spawn(lane, 9, body=release.wait) for lane in lanes]

        def ready():
            return all(self.depth(lane) >= before[lane] + need for lane, need in added.items())

        assert await until(ready), "holders never entered their slots"
        return tasks

    async def drain(self):
        """Cancel every task and gather it on the loop it was created on."""
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()


@pytest.fixture
async def q(request):
    """The Queue under test, drained and checked drained after the test's own body."""
    queue = getattr(request, "param", None) or Queue()
    yield queue
    await queue.drain()
    assert queue.scheduler.active == EMPTY
    assert queue.scheduler.status()["queued"] == EMPTY and not queue.scheduler.waiters


async def enter(scheduler, lane, timeout=1):
    async with scheduler.slot(lane, timeout, Timings()):
        return True


def test_defaults_status_and_keyword_only():
    scheduler = ExecutionSlots(4)
    assert (scheduler.limit, scheduler.inspection_reserved) == (4, 2)
    assert scheduler.queue_limits == {"general": 64, "inspection": 16}
    assert scheduler.status() == {
        "limit": 4,
        "inspection_reserved": 2,
        "general_limit": 2,
        "active": EMPTY,
        "queued": EMPTY,
        "queue_limits": {"general": 64, "inspection": 16},
        "queue_full_rejections": EMPTY,
        "queue_scope": "execution-slot waiters only; excludes login and session waits",
        "peak_active": 0,
    }
    assert ExecutionSlots(4, 2, queue_limit=5, inspection_queue_limit=1).queue_limits == {
        "general": 5,
        "inspection": 1,
    }
    with pytest.raises(TypeError):
        ExecutionSlots(4, 2, 64)  # queue limits are keyword-only


@pytest.mark.parametrize(
    "bad",
    [
        {"limit": 0},
        {"limit": -1},
        {"limit": True},
        {"limit": 4.0},
        {"limit": None},
        {"limit": 4, "inspection_reserved": True},
        {"limit": 4, "inspection_reserved": 2.0},
        {"limit": 4, "inspection_reserved": None},
        {"limit": 4, "inspection_reserved": 4},
        {"limit": 4, "queue_limit": -1},
        {"limit": 4, "queue_limit": True},
        {"limit": 4, "inspection_queue_limit": "4"},
        {"limit": 4, "inspection_queue_limit": None},
    ],
)
def test_invalid_capacity_is_rejected(bad):
    with pytest.raises(ValueError):
        ExecutionSlots(**bad)


def test_queue_full_is_a_separate_retryable_exception():
    assert not issubclass(QueueFull, QueueTimeout) and not issubclass(QueueTimeout, QueueFull)
    error = QueueFull("general", 0, 0)
    assert error.stage == "execution_queue"
    assert error.response() == {
        "reason": "queue_full",
        "retryable": True,
        "queue": {"lane": "general", "limit": 0, "depth": 0},
    }


@pytest.mark.parametrize("lane, timeout", [("bulk", 1), ("general", 0)])
async def test_unknown_lane_and_nonpositive_timeout_rejected(q, lane, timeout):
    with pytest.raises(ValueError):
        async with q.scheduler.slot(lane, timeout, Timings()):
            raise AssertionError("invalid request was admitted")
    assert q.scheduler.active == EMPTY and not q.scheduler.waiters


@pytest.mark.parametrize("q", [Queue(1, 0, cap=0)], indirect=True)
async def test_zero_wait_idle_grant_and_blocked_rejection(q):
    """At cap 0 an idle slot is still granted at once, and a blocked request can only reject."""
    scheduler = q.scheduler
    assert await enter(scheduler, "general")
    assert not scheduler.waiters and scheduler.queue_full_rejections == EMPTY
    await q.hold(["general"], asyncio.Event())
    assert scheduler.active == {"general": 1, "inspection": 0}
    with pytest.raises(QueueFull) as caught:
        async with scheduler.slot("general", 1, Timings()):
            raise AssertionError("second general request was admitted past the limit")
    assert (caught.value.lane, caught.value.limit, caught.value.depth) == ("general", 0, 0)
    assert not scheduler.waiters, "a rejection must not leave a waiter behind"
    assert scheduler.status()["queued"] == EMPTY
    assert scheduler.queue_full_rejections == {"general": 1, "inspection": 0}


@pytest.mark.parametrize("q", [Queue(3, 1, cap=2)], indirect=True)
async def test_cap_counts_waiting_requests_only(q):
    scheduler, release = q.scheduler, asyncio.Event()
    await q.hold(["general", "general"], release)  # every general slot taken at once
    assert not scheduler.waiters, "immediate grants must not consume queue space"
    q.spawn()  # lane full, queue still empty: it waits instead of being rejected
    assert await until(lambda: scheduler.status()["queued"]["general"] == 1)
    assert scheduler.queue_full_rejections["general"] == 0
    release.set()


@pytest.mark.parametrize("q", [Queue(1, 0, cap=3)], indirect=True)
async def test_burst_of_arrivals_respects_cap_exactly(q):
    scheduler, release = q.scheduler, asyncio.Event()
    tasks = [q.spawn(body=release.wait) for _ in range(10)]
    await asyncio.sleep(0)  # every arrival gets its say in one tick
    assert await until(lambda: len(q.entered) + len(q.errors) == 7)
    assert q.entered == ["general"] and scheduler.active == {"general": 1, "inspection": 0}
    assert scheduler.status()["queued"] == {"general": 3, "inspection": 0}
    assert scheduler.queue_full_rejections == {"general": 6, "inspection": 0}
    assert all(isinstance(error, QueueFull) for error in q.errors)
    assert {(e.lane, e.limit, e.depth) for e in q.errors} == {("general", 3, 3)}
    release.set()
    await asyncio.wait_for(asyncio.gather(*tasks), TIMEOUT)  # the three queued run
    assert q.entered == ["general"] * 4 and scheduler.peak_active == 1


@pytest.mark.parametrize("q", [Queue(1, 0, cap=1)], indirect=True)
async def test_timeout_releases_queue_without_counting_as_rejection(q):
    scheduler, release = q.scheduler, asyncio.Event()
    await q.hold(["general"], release)
    q.spawn(timeout=STEP)
    assert await until(lambda: len(q.errors) == 1) and isinstance(q.errors[0], QueueTimeout)
    assert scheduler.status()["queued"] == EMPTY and not scheduler.waiters
    assert scheduler.queue_full_rejections == EMPTY, "a timeout is not a rejection"
    q.spawn()  # the freed queue entry admits the successor
    assert await until(lambda: scheduler.status()["queued"]["general"] == 1)
    release.set()


@pytest.mark.parametrize("q", [Queue(1, 0, cap=1)], indirect=True)
async def test_cancel_queued_request_frees_space(q):
    scheduler, release = q.scheduler, asyncio.Event()
    await q.hold(["general"], release)
    loser = q.spawn()
    assert await until(lambda: scheduler.status()["queued"]["general"] == 1)
    loser.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loser
    assert scheduler.status()["queued"] == EMPTY and not scheduler.waiters
    q.spawn()  # the freed queue entry admits the successor
    assert await until(lambda: scheduler.status()["queued"]["general"] == 1)
    assert scheduler.queue_full_rejections == EMPTY, "a cancel is not a rejection"
    release.set()


@pytest.mark.parametrize("q", [Queue(1, 0, cap=1)], indirect=True)
async def test_cancel_after_grant_before_resume_releases_slot(q):
    scheduler, release = q.scheduler, asyncio.Event()
    await q.hold(["general"], release)
    waiter = q.spawn()
    assert await until(lambda: scheduler.status()["queued"]["general"] == 1)
    release.set()
    await asyncio.sleep(0)  # the release wakes the waiter; its body has not run yet
    assert not scheduler.waiters and not waiter.done() and q.entered == ["general"]
    assert scheduler.active == {"general": 1, "inspection": 0}, "the grant took the slot"
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert q.entered == ["general"]
    assert scheduler.active == EMPTY, "a granted-then-cancelled request leaked a slot"
    assert await enter(scheduler, "general")


@pytest.mark.parametrize("q", [Queue(1, 0, cap=3)], indirect=True)
async def test_fifo_order_among_eligible_waiters(q):
    scheduler, release, order = q.scheduler, asyncio.Event(), []
    await q.hold(["general"], release)
    for depth, name in enumerate(["first", "second", "third"], start=1):
        q.spawn(body=lambda name=name: _record(order, name))
        assert await until(lambda depth=depth: scheduler.status()["queued"]["general"] == depth)
    release.set()
    await asyncio.wait_for(asyncio.gather(*q.tasks), TIMEOUT)
    assert order == ["first", "second", "third"] and scheduler.peak_active == 1


async def _record(order, name):
    order.append(name)


@pytest.mark.parametrize("q", [Queue(3, 1, cap=2)], indirect=True)
async def test_lanes_queue_independently_and_respect_global_limit(q):
    """Both queues fill to their own cap, and each lane books only its own overflow."""
    scheduler, release = q.scheduler, asyncio.Event()
    await q.hold(["general", "general"], release)  # general saturated, one slot free
    for _ in range(2):
        q.spawn(timeout=9)
    assert await until(lambda: scheduler.status()["queued"]["general"] == 2)
    with pytest.raises(QueueFull) as caught:
        async with scheduler.slot("general", 1, Timings()):
            raise AssertionError("general queue accepted a request past its cap")
    assert caught.value.response()["queue"] == {"lane": "general", "limit": 2, "depth": 2}
    # The reserved slot still admits inspection at once, though general is full.
    q.spawn("inspection", timeout=9, body=release.wait)
    assert await until(lambda: scheduler.active["inspection"] == 1)
    assert scheduler.active == {"general": 2, "inspection": 1}
    for _ in range(2):
        q.spawn("inspection", timeout=9)
    assert await until(lambda: scheduler.status()["queued"]["inspection"] == 2)
    with pytest.raises(QueueFull) as inspected:
        async with scheduler.slot("inspection", STEP, Timings()):
            raise AssertionError("inspection queue accepted a request past its cap")
    assert inspected.value.response()["queue"] == {"lane": "inspection", "limit": 2, "depth": 2}
    assert scheduler.status()["queued"] == {"general": 2, "inspection": 2}
    assert scheduler.queue_full_rejections == {"general": 1, "inspection": 1}
    release.set()


@pytest.mark.parametrize("q", [Queue(3, 1, cap=2)], indirect=True)
async def test_queued_inspector_granted_despite_general_waiters_ahead(q):
    """An inspector queued behind general waiters still gets the freed reserve slot.

    ``_wake`` scans the deque in FIFO order but never stops at a waiter it cannot
    grant. Two general waiters sit ahead of the inspector and can never be admitted
    (general is at its reserve floor), yet the inspector at the tail is eligible the
    moment the held inspection slot is released.
    """
    scheduler, release = q.scheduler, asyncio.Event()
    holder = q.spawn("inspection", timeout=9, body=asyncio.Event().wait)
    assert await until(lambda: scheduler.active["inspection"] == 1)
    await q.hold(["general", "general"], release)  # general saturated, deque general-first
    for _ in range(2):
        q.spawn(timeout=9)
    assert await until(lambda: scheduler.status()["queued"]["general"] == 2)
    inspector = q.spawn("inspection", timeout=9, body=release.wait)
    assert await until(lambda: scheduler.status()["queued"]["inspection"] == 1)
    assert [w["lane"] for w in scheduler.waiters] == ["general", "general", "inspection"]

    holder.cancel()  # releasing the reserve slot must reach the inspector at the tail
    with pytest.raises(asyncio.CancelledError):
        await holder
    assert await until(lambda: q.entered.count("inspection") == 2), "inspector was starved"
    assert scheduler.active == {"general": 2, "inspection": 1}
    assert scheduler.status()["queued"] == {"general": 2, "inspection": 0}
    assert not inspector.done() and scheduler.queue_full_rejections == EMPTY
    release.set()
