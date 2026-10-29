#!/usr/bin/env python3
"""Repeatable local ten-worker load test; no deployment credentials or LLMs.

Uses disposable loopback SSH, synthetic REST assets and a shared BridgeService.
Production login pacing is unchanged. JSON reports measure bridge/SSH behavior,
not client orchestration, MCP transport latency or a real heterogeneous fleet.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import platform
import shlex
import signal
import statistics
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import asyncssh  # noqa: E402

from jms_bridge import __version__  # noqa: E402
from jms_bridge.async_io import offload  # noqa: E402
from jms_bridge.compact import BridgeService  # noqa: E402
from jms_bridge.config import BastionConfig  # noqa: E402
from jms_bridge.scheduling import ExecutionSlots  # noqa: E402


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "p50": round(statistics.median(values), 6),
        "p95": round(values[max(0, math.ceil(len(values) * 0.95) - 1)], 6),
        "max": round(values[-1], 6),
    }


def summarize(rows):
    keys = ("queue_seconds", "connection_seconds", "channel_seconds", "execution_seconds")
    phases = sorted({k for row in rows for k in row.get("connection_timing", {})})
    return {
        "requests": len(rows),
        "successful": sum(row["ok"] for row in rows),
        "wall_seconds": distribution([row["wall_seconds"] for row in rows]),
        "timing_seconds": {key: distribution([r["timing"][key] for r in rows]) for key in keys},
        "connection_phases_seconds": {
            key: distribution([r.get("connection_timing", {}).get(key, 0) for r in rows])
            for key in phases
        },
    }


async def until(predicate, timeout=5):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), timeout)


async def drain(tasks):
    for task in tasks:
        if not task.done():
            task.cancel()
    return await asyncio.gather(*tasks, return_exceptions=True)


class LocalSSH:
    """Validate disposable credentials and run local Bash children in own groups."""

    def __init__(self, root):
        self.root = root
        self.server = None
        self.children = set()
        self.peak_children = 0
        self.handlers = set()
        self.logins = []
        self.commands = []
        self.allowed_users = set()
        self.client_key = asyncssh.generate_private_key("ssh-ed25519")

    async def start(self):
        owner = self

        class Server(asyncssh.SSHServer):
            def begin_auth(self, username):
                owner.logins.append({"username": username, "at": time.monotonic()})
                return True

            def public_key_auth_supported(self):
                return True

            def validate_public_key(self, username, key):
                return (
                    username in owner.allowed_users
                    and key.export_public_key() == owner.client_key.export_public_key()
                )

        key = asyncssh.generate_private_key("ssh-ed25519")
        self.server = await asyncssh.create_server(
            Server,
            "127.0.0.1",
            0,
            server_host_keys=[key],
            process_factory=self.serve,
            encoding=None,
            rdns_lookup=False,
            gss_host="localhost",
            config=None,
        )
        port = self.server.get_port()
        key_path = self.root / "disposable-key"
        key_path.write_bytes(self.client_key.export_private_key())
        key_path.chmod(0o600)
        known = self.root / "known_hosts"
        known.write_bytes(f"[127.0.0.1]:{port} ".encode() + key.export_public_key())
        return port, key_path, known

    @staticmethod
    async def kill(proc):
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
        await proc.wait()

    async def serve(self, process):
        handler = asyncio.current_task()
        self.handlers.add(handler)
        proc, streams = None, []
        try:
            argv = shlex.split(process.command)
            if argv not in (["bash", "-s"], ["bash", "--noprofile", "--norc"]):
                process.exit(126)
                return
            self.commands.append(process.command)
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.root,
                start_new_session=True,
            )
            self.children.add(proc)
            self.peak_children = max(
                self.peak_children, sum(p.returncode is None for p in self.children)
            )

            async def incoming():
                try:
                    while data := await process.stdin.read(65536):
                        proc.stdin.write(data)
                        await proc.stdin.drain()
                finally:
                    proc.stdin.close()

            async def outgoing(reader, writer):
                while data := await reader.read(65536):
                    writer.write(data)
                    await writer.drain()

            streams = [
                asyncio.create_task(incoming()),
                asyncio.create_task(outgoing(proc.stdout, process.stdout)),
                asyncio.create_task(outgoing(proc.stderr, process.stderr)),
            ]
            finished = asyncio.create_task(proc.wait())
            closed = asyncio.create_task(process.wait_closed())
            streams += [finished, closed]
            await asyncio.wait([finished, closed], return_when=asyncio.FIRST_COMPLETED)
            await self.kill(proc)
            await asyncio.gather(*streams[1:3], return_exceptions=True)
            process.exit(proc.returncode)
        finally:
            if proc is not None:
                await self.kill(proc)
                self.children.discard(proc)
            await drain(streams)
            self.handlers.discard(handler)

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        await asyncio.gather(*(self.kill(proc) for proc in tuple(self.children)))
        if self.handlers:
            await asyncio.wait_for(drain(tuple(self.handlers)), 5)


@asynccontextmanager
async def fleet(workers=10):
    """All configuration/state is synthetic; ambient user inventory is isolated."""
    with tempfile.TemporaryDirectory(prefix="jms-concurrency-") as directory:
        root = Path(directory)
        ssh = LocalSSH(root)
        service = None
        try:
            port, key, known = await ssh.start()
            cfg = BastionConfig(
                url="https://benchmark.invalid",
                login_user="benchmark",
                ssh_port=port,
                ssh_key=key,
                max_parallel=8,
            )
            from jms_bridge.runtime_config import BridgeSettings

            settings = BridgeSettings(
                state_dir=root / "state",
                ssh_host="127.0.0.1",
                known_hosts=known,
                execution_limit=8,
                queue_timeout=5,
            )
            with patch.dict(os.environ, {"JMS_BRIDGE_INVENTORY": str(root / "absent.yaml")}):
                service = BridgeService(cfg, settings)
                assets = [
                    {
                        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"benchmark/worker-{i}")),
                        "name": f"worker-{i}",
                        "address": f"192.0.2.{i + 1}",
                        "active": True,
                        "protocols": ["ssh"],
                        "accounts": ["root"],
                        "account": "root",
                        "login": "benchmark",
                    }
                    for i in range(workers)
                ]
                ssh.allowed_users = {f"benchmark#root#{a['id']}" for a in assets}

                async def fetch():
                    return assets

                service.catalog._fetch = fetch
                status = await service.catalog.refresh()
                if not status["ok"]:
                    raise RuntimeError(status["last_error"])
                hosts = service.catalog.inventory.resolve([a["name"] for a in assets])
                yield service, hosts, ssh
        finally:
            try:
                if service:
                    for task in tuple(service.requests.tasks.values()):
                        task.cancel()
                    await service.requests.drain()
                    await service.filesystem.workers.close()
                    await service.catalog.close()
                    await service.runner.close()
            finally:
                await ssh.close()


def reset_scheduler(service, limit, reserve):
    old = service.runner.scheduler.status()
    if sum(old["active"].values()) or sum(old["queued"].values()):
        raise RuntimeError("cannot start a case while previous work is active")
    service.runner.scheduler = ExecutionSlots(limit, reserve)


async def run_worker(service, host, script, timeout=5):
    started = time.monotonic()
    reply = await service.run([host.name], script=script, timeout=timeout)
    completed = time.monotonic()
    row = reply["results"][0]
    meta = await offload(service.store.metadata, row["result_id"])
    return {
        **row,
        "wall_seconds": completed - started,
        "report_metadata_seconds": time.monotonic() - completed,
        "connection_timing": meta["connection_timing"],
    }


async def batch(service, hosts, script, ssh):
    await until(lambda: not ssh.children)
    ssh.peak_children = 0
    stop = asyncio.Event()
    peak = {"general": 0, "inspection": 0}

    async def sample():
        # Passive internal counter sampling, not an MCP status call; peaks are approximate.
        while not stop.is_set():
            queued = service.runner.scheduler.status()["queued"]
            for lane in peak:
                peak[lane] = max(peak[lane], queued[lane])
            await asyncio.sleep(0.005)

    sampler = asyncio.create_task(sample())
    tasks = [asyncio.create_task(run_worker(service, h, script)) for h in hosts]
    started = time.monotonic()
    try:
        rows = await asyncio.gather(*tasks)
        return {
            "wall_seconds": time.monotonic() - started,
            "summary": summarize(rows),
            "queue_sample_peak": peak,
            "queue_sample_interval_seconds": 0.005,
            "max_concurrent_children": ssh.peak_children,
            "scheduler": service.runner.scheduler.status(),
            "rows": rows,
        }
    finally:
        stop.set()
        await drain([sampler, *tasks])


async def reserve_case(service, hosts, reserve, ssh, seconds=0.3):
    await until(lambda: not ssh.children)
    ssh.peak_children = 0
    reset_scheduler(service, 8, reserve)
    tasks = [
        asyncio.create_task(run_worker(service, h, f"sleep {seconds}; printf worker"))
        for h in hosts
    ]
    started = time.monotonic()
    try:
        await until(lambda: service.runner.scheduler.active["general"] == 8 - reserve)
        saturated = service.runner.scheduler.status()
        probe_started = time.monotonic()
        probe = await service.runner.execute_to_store(
            hosts[0], "printf inspection", 5, lane="inspection"
        )
        probe_wall = time.monotonic() - probe_started
        rows = await asyncio.gather(*tasks)
        return {
            "reserve": reserve,
            "limit": 8,
            "max_concurrent_children": ssh.peak_children,
            "saturated": saturated,
            "wall_seconds": time.monotonic() - started,
            "workers": summarize(rows),
            "inspection": {
                "wall_seconds": probe_wall,
                "timing": probe["timing"],
                "ok": probe["ok"],
            },
            "scheduler": service.runner.scheduler.status(),
        }
    finally:
        await drain(tasks)


async def cancel_case(service, hosts):
    reset_scheduler(service, 8, 2)
    before = {p.name for p in service.store.root.iterdir()}
    tasks = [asyncio.create_task(run_worker(service, h, "printf STARTED; sleep 30")) for h in hosts]
    try:
        await until(
            lambda: (
                service.runner.scheduler.active["general"] == 6
                and service.runner.scheduler.status()["queued"]["general"] == 4
            )
        )

        def started_count():
            count = 0
            for path in service.store.root.glob("*/stdout.log"):
                if path.parent.name not in before:
                    with path.open("rb") as file:
                        count += file.read(7) == b"STARTED"
            return count

        await until(lambda: started_count() == 6)
        saturated = service.runner.scheduler.status()
    finally:
        outcomes = await drain(tasks)
    metas = [
        await offload(service.store.metadata, p.name)
        for p in service.store.root.iterdir()
        if p.name not in before
    ]
    return {
        "cancelled": sum(isinstance(r, asyncio.CancelledError) for r in outcomes),
        "before_cancel": saturated,
        "results": len(metas),
        "states": {
            s: sum(r["execution_state"] == s for r in metas)
            for s in ("not_started", "unknown", "completed")
        },
        "scheduler": service.runner.scheduler.status(),
        "all_results_complete": all(r["status"] == "complete" for r in metas),
    }


async def queue_timeout_case(service, hosts):
    reset_scheduler(service, 8, 2)
    from jms_bridge.scheduling import Timings

    tasks = []
    async with contextlib.AsyncExitStack() as stack:
        for _ in range(6):
            await stack.enter_async_context(service.runner.scheduler.slot("general", 1, Timings()))
        try:
            tasks = [
                asyncio.create_task(
                    service.runner.execute_to_store(
                        host, "printf MUST-NOT-RUN", 5, queue_timeout=0.05
                    )
                )
                for host in hosts
            ]
            rows = await asyncio.gather(*tasks)
            blocked = service.runner.scheduler.status()
        finally:
            await drain(tasks)
    return {
        "requests": len(rows),
        "queue_timeout_seconds": 0.05,
        "all_not_started": all(
            r["execution_state"] == "not_started" and r["executed"] is False for r in rows
        ),
        "all_queue_timed_out": all(r["queue_timed_out"] for r in rows),
        "all_results_complete": all(r["status"] == "complete" for r in rows),
        "queue_seconds": distribution([r["timing"]["queue_seconds"] for r in rows]),
        "while_blocked": blocked,
        "scheduler": service.runner.scheduler.status(),
    }


async def slow_refresh_case(service, hosts):
    entered, release = threading.Event(), threading.Event()
    old = service.catalog.inventory
    original = service.catalog._prepare

    def slow(*args):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("storage test was not released")
        return original(*args)

    service.catalog._prepare = slow
    task = asyncio.create_task(service.catalog.refresh())
    try:
        if not await asyncio.to_thread(entered.wait, 2):
            raise RuntimeError("refresh did not reach storage preparation")
        # Simulate stale data and elapsed retry pacing without a timing-dependent sleep.
        service.catalog.snapshot["refreshed_at"] = 1
        service.catalog._last_attempt = 0
        started = time.monotonic()
        status = await service.discover(topic="status")
        status_wall = time.monotonic() - started
        row = await asyncio.wait_for(run_worker(service, hosts[0], "printf responsive"), 2)
        return {
            "status_wall_seconds": status_wall,
            "request": row,
            "old_view_used": service.catalog.inventory is old,
            "refresh_pending": not task.done(),
            "status_stale": status["inventory"]["stale"],
        }
    finally:
        release.set()
        try:
            await task
        finally:
            service.catalog._prepare = original


async def disconnect_case(service, host, ssh):
    before = len(service.runner._connections)
    channels_before = len(ssh.commands)
    old_results = {p.name for p in service.store.root.iterdir()}
    task = asyncio.create_task(run_worker(service, host, "printf PARTIAL; sleep 30; printf FINAL"))
    try:

        def started():
            for path in service.store.root.glob("*/stdout.log"):
                if path.parent.name not in old_results:
                    with path.open("rb") as file:
                        if file.read(7) == b"PARTIAL":
                            return True
            return False

        await until(started)
        connection = service.runner._connections[service.runner.key(host)]
        connection.abort()
        await connection.wait_closed()
        row = await asyncio.wait_for(task, 5)
        return {
            "result": row,
            "connection_status": service.runner.connection_status(),
            "connection_entries_before": before,
            "connection_entries_after": len(service.runner._connections),
            "channels_started": len(ssh.commands) - channels_before,
            "partial_without_final": row.get("stdout") == "PARTIAL",
        }
    finally:
        await drain([task])


async def benchmark(workers=10, rounds=3, seconds=0.3, burst=40):
    async with fleet(workers) as (service, hosts, ssh):
        report = {
            "version": __version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "workers": workers,
            "rounds": rounds,
            "work_seconds": seconds,
            "scope": "in-process BridgeService + real loopback SSH; no LLM/MCP transport/fleet",
            "production_login_interval_seconds": 5,
            "measurement": {
                "request_wall": "service.run return; excludes subsequent report metadata read",
                "batch_wall": "all workers and report metadata collection",
                "queue_peak": "approximate internal counter sample every 5ms",
                "inspection": "direct runner inspection lane, not automatic main-agent priority",
                "excluded": [
                    "LLM orchestration",
                    "MCP HTTP/stdio transport",
                    "real bastion",
                    "remote network",
                    "accelerators",
                    "distributed training",
                ],
            },
        }
        report["cold"] = await batch(service, hosts, "printf cold", ssh)
        report["cold"]["login_start_intervals_seconds"] = [
            b["at"] - a["at"] for a, b in zip(ssh.logins, ssh.logins[1:], strict=False)
        ]
        if report["cold"]["summary"]["successful"] != workers:
            raise RuntimeError("cold authentication failed; no warm benchmark data is valid")
        cold_logins = len(ssh.logins)
        report["warm"] = []
        for _ in range(rounds):
            reset_scheduler(service, 8, 0)
            report["warm"].append(await batch(service, hosts, f"sleep {seconds}; printf warm", ssh))
        report["reserve"] = [
            await reserve_case(service, hosts, reserve, ssh, seconds)
            for _ in range(rounds)
            for reserve in (0, 1, 2)
        ]
        reset_scheduler(service, 8, 2)
        report["burst"] = await batch(
            service,
            [hosts[i % workers] for i in range(burst)],
            f"sleep {seconds}; printf burst",
            ssh,
        )
        report["cancel"] = await cancel_case(service, hosts)
        await until(lambda: not ssh.children)
        report["queue_timeout"] = await queue_timeout_case(service, hosts)
        report["slow_refresh"] = await slow_refresh_case(service, hosts)
        await until(lambda: not ssh.children)
        report["warm_additional_logins"] = len(ssh.logins) - cold_logins
        report["disconnect"] = await disconnect_case(service, hosts[-1], ssh)
        await until(lambda: not ssh.children)
        report["before_close"] = {
            "scheduler": service.runner.scheduler.status(),
            "requests_active": len(service.requests.tasks),
            "children_active": len(ssh.children),
            "result_retention": service.store.retention_status(),
        }
    report["after_close"] = {
        "children_active": len(ssh.children),
        "handlers_active": len(ssh.handlers),
    }
    report["ok"] = (
        report["cold"]["summary"]["successful"] == workers
        and cold_logins == workers
        and report["cold"]["max_concurrent_children"] <= 8
        and all(
            r["summary"]["successful"] == workers and r["max_concurrent_children"] <= 8
            for r in report["warm"]
        )
        and all(
            r["workers"]["successful"] == workers
            and r["inspection"]["ok"]
            and r["max_concurrent_children"] <= 8
            for r in report["reserve"]
        )
        and report["burst"]["summary"]["successful"] == burst
        and report["burst"]["max_concurrent_children"] <= 6
        and report["cancel"]["cancelled"] == workers
        and report["cancel"]["results"] == workers
        and report["cancel"]["all_results_complete"]
        and report["cancel"]["states"] == {"not_started": workers - 6, "unknown": 6, "completed": 0}
        and report["queue_timeout"]["requests"] == workers
        and report["queue_timeout"]["all_not_started"]
        and report["queue_timeout"]["all_queue_timed_out"]
        and report["queue_timeout"]["all_results_complete"]
        and report["slow_refresh"]["old_view_used"]
        and report["slow_refresh"]["refresh_pending"]
        and report["slow_refresh"]["request"]["ok"]
        and report["slow_refresh"]["status_stale"]
        and report["disconnect"]["partial_without_final"]
        and report["disconnect"]["channels_started"] == 1
        and report["disconnect"]["result"]["completion_unknown"]
        and report["warm_additional_logins"] == 0
        and report["after_close"] == {"children_active": 0, "handlers_active": 0}
        and not sum(report["before_close"]["scheduler"]["active"].values())
        and not sum(report["before_close"]["scheduler"]["queued"].values())
        and report["before_close"]["requests_active"] == 0
        and report["before_close"]["result_retention"]["active_results"] == 0
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--work-seconds", type=float, default=0.3)
    parser.add_argument("--burst", type=int, default=40)
    parser.add_argument("--output", type=Path, help="write a NEW JSON report (never overwrite)")
    args = parser.parse_args(argv)
    if (
        not 1 <= args.rounds <= 20
        or not 0.1 <= args.work_seconds <= 2
        or not 10 <= args.burst <= 200
        or math.ceil(args.burst / 6) * args.work_seconds > 4
    ):
        parser.error("rounds=1..20, work-seconds=0.1..2, burst=10..200; burst workload <=4s")
    # Reserve the output before starting a ~45-second cold-login experiment.
    with contextlib.ExitStack() as stack:
        output = (
            stack.enter_context(
                os.fdopen(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w")
            )
            if args.output
            else sys.stdout
        )
        report = asyncio.run(
            benchmark(rounds=args.rounds, seconds=args.work_seconds, burst=args.burst)
        )
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
