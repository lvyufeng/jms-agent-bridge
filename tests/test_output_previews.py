"""Bounded presentation snapshots keep complete output only on disk."""

import asyncio
import hashlib
import threading
import tracemalloc
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from jms_bridge.compact import BridgeService
from jms_bridge.config import BastionConfig
from jms_bridge.inventory import Host
from jms_bridge.jobs import JobManager
from jms_bridge.native_ssh import NativeSSHRunner
from jms_bridge.results import OutputPreview, ResultStore
from jms_bridge.runtime_config import BridgeSettings
from jms_bridge.ssh import ExecResult


@pytest.fixture
def runtime(tmp_path):
    cfg = BastionConfig(url="https://bastion.example.com", login_user="test")
    settings = BridgeSettings(state_dir=tmp_path)
    store = ResultStore(tmp_path / "results")
    runner = NativeSSHRunner(cfg, settings, store=store)
    host = Host(name="node", asset_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", account="root")
    return SimpleNamespace(runner=runner, store=store, host=host, cfg=cfg, settings=settings)


def recorded(runtime, stdout=b"", stderr=b""):
    rid = runtime.store.create(kind="run")
    runtime.store.path(rid, "stdout").write_bytes(stdout)
    runtime.store.path(rid, "stderr").write_bytes(stderr)
    return runtime.store.finish(rid, exit_code=0, timed_out=False, ok=True)


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    [
        (b"", b""),
        (b"small", b"error"),
        (b"x" * 50000, b""),
        (b"", b"HEAD" + b"e" * 50000 + b"TAIL"),
        (b"x" * 50000, b"HEAD" + b"e" * 50000 + b"TAIL"),
        (("中文🙂" * 5000).encode(), b"\xff" * 50000 + b"TAIL"),
    ],
)
def test_cached_preview_matches_direct_file_policy(runtime, stdout, stderr):
    meta = recorded(runtime, stdout, stderr)
    preview = runtime.store.preview_snapshot(meta["result_id"])
    assert preview.stdout_head == stdout[:16384]
    assert preview.stderr_head == stderr[:16384] and preview.stderr_tail == stderr[-16384:]
    for budget in (0, 1, 17, 256, 4096, 16384):
        assert preview.compact(budget) == runtime.store.preview(meta["result_id"], budget)


@pytest.mark.parametrize("budget", [0, 1, 256, 4096, 16384])
def test_small_complete_snapshot_reports_budget_truncation(runtime, budget):
    meta = recorded(runtime, b"o" * 6000, b"HEAD" + b"e" * 6000 + b"TAIL")
    preview = runtime.store.preview_snapshot(meta["result_id"])
    assert not preview.stdout_truncated and not preview.stderr_truncated
    result = ExecResult("node", 0, "", "", output_preview=preview)
    result.result_id = meta["result_id"]
    row = result.to_dict(limit=budget)
    assert row["stdout_truncated"] == (6000 > min(6000, budget))
    remaining = max(0, budget - min(6000, budget))
    assert row["stderr_truncated"] == (6008 > min(6008, remaining))
    assert row["stdout_bytes"] == 6000 and row["stderr_bytes"] == 6008
    assert preview.compact(budget) == runtime.store.preview(meta["result_id"], budget)


def test_explicit_empty_preview_can_serialize_without_dynamic_attributes():
    preview = OutputPreview(b"", b"", b"", {})
    result = ExecResult("node", 0, "", "", output_preview=preview)
    result.result_id = "a" * 32
    row = result.to_dict(limit=0)
    assert row["stdout_bytes"] == row["stderr_bytes"] == 0
    assert not row["stdout_truncated"] and not row["stderr_truncated"]


@pytest.mark.parametrize("limit", [-1, 16385, True, 1.5])
def test_preview_snapshot_rejects_unbounded_limits(runtime, limit):
    meta = recorded(runtime, b"payload")
    with pytest.raises(ValueError, match="0..16384"):
        runtime.store.preview_snapshot(meta["result_id"], limit)


def test_preview_snapshot_never_uses_unbounded_file_reads(runtime, monkeypatch):
    meta = recorded(runtime, b"o" * 100000, b"e" * 100000)
    original = Path.open
    reads = []

    class Guarded:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file.close()

        def seek(self, *args):
            return self.file.seek(*args)

        def read(self, count=-1):
            assert 0 <= count <= 16384
            reads.append(count)
            return self.file.read(count)

    def guarded(path, *args, **kwargs):
        file = original(path, *args, **kwargs)
        return Guarded(file) if path.suffix == ".log" else file

    monkeypatch.setattr(Path, "open", guarded)
    runtime.store.preview_snapshot(meta["result_id"])
    assert reads == [16384, 16384, 16384]


async def test_default_full_parser_input_keeps_tail_without_duplicate_raw_copy(
    runtime, monkeypatch
):
    raw = b"x" * 100000 + b"\nLAUNCHER=tmux\nFINAL=complete\n"
    meta = recorded(runtime, raw, b"error detail\n")
    monkeypatch.setattr(runtime.runner, "execute_to_store", AsyncMock(return_value=meta))
    result = await runtime.runner.exec_on(runtime.host, "test parser command")
    assert result.stdout.encode() == raw and result.stderr == "error detail\n"
    assert "LAUNCHER=tmux" in result.stdout and result.stdout.endswith("FINAL=complete\n")
    assert not hasattr(result, "_bridge_output") and result.output_preview is None
    results = await runtime.runner.exec_many([runtime.host], "test parser command")
    assert results[0].stdout.encode() == raw
    await runtime.runner.close()


async def test_job_log_parser_receives_the_entire_64kib_page(runtime, monkeypatch):
    body = b"x" * (65536 - 10) + b"FINAL-BODY"
    meta = recorded(runtime, b"size=100000\n" + body)
    monkeypatch.setattr(runtime.runner, "execute_to_store", AsyncMock(return_value=meta))
    reply = await JobManager(runtime.runner).logs(runtime.host, "12345678", max_bytes=65536)
    assert reply["log"].encode() == body
    assert reply["next_offset"] == 65536 and reply["size"] == 100000
    await runtime.runner.close()


@pytest.mark.parametrize("limit", [0, 16, 16384])
def test_snapshot_limit_has_exact_raw_windows(runtime, limit):
    meta = recorded(runtime, b"s" * 50, b"HEAD" + b"e" * 50 + b"TAIL")
    preview = runtime.store.preview_snapshot(meta["result_id"], limit)
    assert len(preview.stdout_head) == min(limit, 50)
    assert len(preview.stderr_head) == min(limit, 58)
    assert len(preview.stderr_tail) == min(limit, 58)
    assert preview.compact(4096).get("stdout_truncated", False) == (50 > limit)
    assert preview.compact(4096).get("stderr_truncated", False) == (58 > limit)


async def test_preview_result_is_explicit_and_reports_true_non_utf8_byte_sizes(
    runtime, monkeypatch
):
    meta = recorded(runtime, b"\xff" * 100000, b"HEAD" + b"e" * 100000 + b"TAIL")
    monkeypatch.setattr(runtime.runner, "execute_to_store", AsyncMock(return_value=meta))

    def forbid_full_read(*args, **kwargs):
        raise AssertionError("presentation path read the whole file")

    monkeypatch.setattr(Path, "read_bytes", forbid_full_read)
    result = await runtime.runner.exec_on(runtime.host, "preview", preview_only=True)
    assert result.stdout_truncated and result.stderr_truncated
    row = result.to_dict(limit=256)
    assert row["stdout_bytes"] == 100000 and row["stderr_bytes"] == 100008
    assert row["stdout_truncated"] and row["stderr_truncated"]
    assert row["result_id"] == meta["result_id"]
    service = BridgeService(runtime.cfg, runtime.settings)
    rows = service.compact_exec_results([result])
    assert rows[0]["stderr"].endswith("TAIL")
    assert rows[0]["stdout_truncated"] and rows[0]["stderr_truncated"]
    preview = result.output_preview
    assert len(preview.stdout_head) + len(preview.stderr_head) + len(preview.stderr_tail) <= 49152
    await runtime.runner.close()
    await service.runner.close()


async def test_64mib_logs_have_constant_preview_allocation_and_lossless_pages(runtime):
    rid = runtime.store.create(kind="transfer")
    chunk = b"0123456789abcdef" * 4096
    digest = hashlib.sha256()
    for stream in ("stdout", "stderr"):
        with runtime.store.path(rid, stream).open("wb") as file:
            for _ in range(512):
                file.write(chunk)
                if stream == "stdout":
                    digest.update(chunk)
    meta = runtime.store.finish(rid, exit_code=0, timed_out=False, ok=True)
    tracemalloc.start()
    try:
        result = await runtime.runner._stored_result(runtime.host, meta, preview_only=True)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024 * 1024, f"preview allocated {peak} bytes for disk logs"
    assert result.output_preview.sizes["stdout"] == 32 * 1024 * 1024
    assert len(result.stdout) <= 16384 and len(result.stderr) <= 16384
    rebuilt = hashlib.sha256()
    offset = 0
    while True:
        page = runtime.store.read(rid, offset=offset, limit=16384)
        rebuilt.update(page["text"].encode())
        assert page["next_offset"] > offset
        offset = page["next_offset"]
        if page["eof"]:
            break
    assert offset == meta["bytes"]["stdout"] and rebuilt.digest() == digest.digest()
    await runtime.runner.close()


@pytest.mark.parametrize("cancel", [False, True])
async def test_slow_preview_storage_does_not_block_status_or_other_event_loop_work(
    runtime, monkeypatch, cancel
):
    meta = recorded(runtime, b"x" * 50000)
    monkeypatch.setattr(runtime.runner, "execute_to_store", AsyncMock(return_value=meta))
    entered, release = threading.Event(), threading.Event()
    original = runtime.store.preview_snapshot
    loop_thread = threading.get_ident()

    def slow(*args, **kwargs):
        assert threading.get_ident() != loop_thread
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime.store, "preview_snapshot", slow)
    task = asyncio.create_task(runtime.runner.exec_on(runtime.host, "preview", preview_only=True))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert not task.done()
        assert runtime.runner.scheduler.status()["active"] == {"general": 0, "inspection": 0}
        assert runtime.store.retention_status()["active_results"] == 0
        if cancel:
            task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        results = await asyncio.gather(task, return_exceptions=True)
    if cancel:
        assert isinstance(results[0], asyncio.CancelledError)
    else:
        assert results[0].ok
    assert runtime.store.metadata(meta["result_id"])["status"] == "complete"
    assert runtime.store.read(meta["result_id"])["total_bytes"] == 50000
    await runtime.runner.close()
