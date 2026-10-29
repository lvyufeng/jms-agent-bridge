"""Preserve a submission's candidate job ID before transport can become uncertain.

No commands, safety checks, permissions, authentication, or launch behavior are
changed. An ID is a reconciliation candidate, never proof of a successful launch.
"""

import asyncio
import inspect

from jms_bridge.jobs import (
    JobHandle,
    JobManager,
    _job_root,
    _launch_script,
    _launcher_of,
    _new_job_id,
)
from jms_bridge.ssh import ExecResult

from .async_io import offload
from .scheduling import queue_rejection


class RecoverableJobManager(JobManager):
    def __init__(self, runner, record):
        super().__init__(runner)
        self.record = record

    async def record_receipt(self, receipt):
        result = self.record(receipt)
        if inspect.isawaitable(result):
            await result

    async def submit(self, host, command, *, name=None):
        job_id = _new_job_id()
        session = f"jms-{job_id}"
        launch = _launch_script(
            _job_root(host, job_id), session, job_id, command, host.env_setup, host.workdir
        )
        receipt = dict(
            host=host.name,
            job_id=job_id,
            session=session,
            launch_acknowledged=False,
            submission_state="prepared",
        )
        # This write must succeed before any remote bytes may be transmitted.
        await self.record_receipt(receipt)
        try:
            result = await self.runner.exec_on(host, launch, timeout=90, workdir=False)
        except asyncio.CancelledError:
            await self.record_receipt(
                {**receipt, "submission_state": "unknown", "completion_unknown": True}
            )
            raise
        except Exception as exc:
            # Match collab's per-host isolation: one local failure must not discard
            # the other hosts' handles after they have already launched.
            result = ExecResult(host.name, 1, "", f"launch failed ({type(exc).__name__})")
        acknowledged = result.ok and "LAUNCHER=" in result.stdout
        result_id = getattr(result, "result_id", None)
        metadata = await offload(self.runner.store.metadata, result_id) if result_id else {}
        await self.record_receipt(
            {
                **receipt,
                "result_id": result_id,
                "launch_acknowledged": acknowledged,
                "submission_state": "acknowledged"
                if acknowledged
                else metadata.get("execution_state", "unknown"),
                "completion_unknown": metadata.get("completion_unknown", not acknowledged),
                **queue_rejection(metadata),
            }
        )
        if acknowledged:
            return JobHandle(job_id, host.name, session, _launcher_of(result.stdout)), result
        return None, result
