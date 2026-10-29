"""Private, durable results with bounded previews and byte-offset reads."""

import asyncio
import contextlib
import functools
import json
import logging
import math
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .runtime_config import default_state_dir


def locked(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, ensure_ascii=False, separators=(",", ":"))
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def preview_buffers(stdout, stderr, budget, *, sizes=None):
    """Preview head/tail buffers using true stream sizes, not cached lengths."""
    sizes = sizes or {"stdout": len(stdout), "stderr": len(stderr)}
    out_size, err_size = sizes.get("stdout", 0), sizes.get("stderr", 0)
    err_budget = min(err_size, max(0, budget // 2))
    out_budget = min(out_size, max(0, budget - err_budget))
    err_budget = min(err_size, max(0, budget - out_budget))
    result = {}
    for name, data, limit, size in [
        ("stdout", stdout, out_budget, out_size),
        ("stderr", stderr, err_budget, err_size),
    ]:
        limit = min(limit, len(data))
        if data and limit:
            result[name] = (data[-limit:] if name == "stderr" else data[:limit]).decode(
                "utf-8", errors="replace"
            )
        if size > limit:
            result[name + "_truncated"] = True
    return result


@dataclass(frozen=True)
class OutputPreview:
    """Bounded raw windows; full stream sizes refer to the unchanged disk logs."""

    stdout_head: bytes
    stderr_head: bytes
    stderr_tail: bytes
    sizes: dict[str, int]

    @property
    def stdout_truncated(self):
        return self.sizes.get("stdout", 0) > len(self.stdout_head)

    @property
    def stderr_truncated(self):
        return self.sizes.get("stderr", 0) > len(self.stderr_head)

    def compact(self, budget):
        return preview_buffers(self.stdout_head, self.stderr_tail, budget, sizes=self.sizes)


class ResultStore:
    def __init__(self, root=None, *, ttl_seconds=604800, cleanup_interval_seconds=3600):
        if not all(
            isinstance(v, (int, float)) and math.isfinite(v) and v > 0
            for v in (ttl_seconds, cleanup_interval_seconds)
        ):
            raise ValueError("result TTL and cleanup interval must be positive finite seconds")
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._lock = threading.RLock()
        self._status_lock = threading.Lock()
        self._active = set()
        self._cleanup_task = None
        self._sweeper = None
        self._last_cleanup = None
        self.root = Path(root) if root is not None else default_state_dir() / "results"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)

    def directory(self, result_id):
        if not re.fullmatch(r"[0-9a-f]{32}", result_id):
            raise ValueError("invalid result_id")
        return self.root / result_id

    @locked
    def create(self, **metadata):
        result_id = uuid.uuid4().hex
        directory = self.directory(result_id)
        directory.mkdir(mode=0o700)
        metadata.update(result_id=result_id, created_at=time.time(), status="running")
        atomic_json(directory / "meta.json", metadata)
        with self._status_lock:
            self._active.add(result_id)
        return result_id

    def path(self, result_id, stream):
        if stream not in ("stdout", "stderr", "result"):
            raise ValueError("stream must be stdout, stderr, or result")
        return self.directory(result_id) / (stream + (".json" if stream == "result" else ".log"))

    async def acreate(self, **metadata):
        from .async_io import offload

        def abandoned(result_id):
            self.finish(
                result_id,
                ok=False,
                executed=False,
                execution_state="not_started",
                completion_unknown=False,
                exit_code=130,
            )

        return await offload(self.create, on_cancel=abandoned, **metadata)

    @locked
    def metadata(self, result_id):
        try:
            return json.loads((self.directory(result_id) / "meta.json").read_text())
        except FileNotFoundError:
            raise ValueError(
                "unknown or expired result_id; result outputs are retained for "
                + str(self.ttl_seconds)
                + " seconds"
            ) from None

    @locked
    def finish(self, result_id, **values):
        meta = self.metadata(result_id)
        now = time.time()
        meta.update(values, status="complete", finished_at=now, expires_at=now + self.ttl_seconds)
        meta["bytes"] = {
            s: self.path(result_id, s).stat().st_size
            for s in ("stdout", "stderr", "result")
            if self.path(result_id, s).exists()
        }
        atomic_json(self.directory(result_id) / "meta.json", meta)
        with self._status_lock:
            self._active.discard(result_id)
        return meta

    @locked
    def save_json(self, value, **meta):
        result_id = self.create(**meta)
        atomic_json(self.path(result_id, "result"), value)
        self.finish(result_id, ok=not (isinstance(value, dict) and value.get("ok") is False))
        return result_id

    @locked
    def preview_snapshot(self, result_id, limit=16384):
        """Cache fixed-size raw prefixes and an error suffix for sync consumers.

        Keep both ends of stderr: legacy short summaries use its prefix, whereas
        compact diagnostic previews use its suffix. Preserve raw bytes so decoding
        and smaller per-host budgets have the same policy as direct file previews.
        """
        if type(limit) is not int or not 0 <= limit <= 16384:
            raise ValueError("preview snapshot limit must be 0..16384 bytes")
        sizes = self.metadata(result_id).get("bytes", {})
        prefixes = []
        stderr_tail = b""
        for stream in ("stdout", "stderr"):
            size = sizes.get(stream, 0)
            prefix = b""
            if size and limit:
                with self.path(result_id, stream).open("rb") as fh:
                    prefix = fh.read(min(size, limit))
                    if stream == "stderr":
                        if size <= limit:
                            stderr_tail = prefix
                        else:
                            fh.seek(size - limit)
                            stderr_tail = fh.read(limit)
            prefixes.append(prefix)
        return OutputPreview(prefixes[0], prefixes[1], stderr_tail, sizes)

    @locked
    def preview(self, result_id, budget=4096):
        meta = self.metadata(result_id)
        sizes = meta.get("bytes", {})
        # stderr gets a reserved share, so noisy stdout cannot hide the error.
        err_budget = min(sizes.get("stderr", 0), max(0, budget // 2))
        out_budget = min(sizes.get("stdout", 0), max(0, budget - err_budget))
        err_budget = min(sizes.get("stderr", 0), max(0, budget - out_budget))
        result = {}
        for stream, limit in [("stdout", out_budget), ("stderr", err_budget)]:
            size = sizes.get(stream, 0)
            path = self.path(result_id, stream)
            if size and limit:
                with path.open("rb") as fh:
                    # Failure diagnostics usually live at the end of stderr.
                    if stream == "stderr":
                        fh.seek(max(0, size - limit))
                    result[stream] = fh.read(limit).decode("utf-8", errors="replace")
            if size > limit:
                result[stream + "_truncated"] = True
        return result

    @locked
    def read(self, result_id, stream="stdout", offset=0, limit=4096, tail=False, pattern=None):
        meta = self.metadata(result_id)
        if not 1 <= limit <= 16384 or offset < 0:
            raise ValueError("limit must be 1..16384 bytes; offset must be nonnegative")
        path = self.path(result_id, stream)
        if not path.exists():
            raise ValueError(f"no {stream} stream; available: {list(meta.get('bytes', {}))}")
        size = path.stat().st_size
        if pattern is not None:
            if tail or not pattern or len(pattern.encode()) > 1024:
                raise ValueError(
                    "search needs a nonempty pattern of at most 1024 bytes and tail=false"
                )
            return self._search(path, result_id, stream, pattern, offset, limit, size)
        start = max(0, size - limit) if tail else min(offset, size)
        with path.open("rb") as fh:
            fh.seek(start)
            data = fh.read(limit + 3)
        if tail:
            while data and 0x80 <= data[0] < 0xC0:
                data = data[1:]
                start += 1
        count = min(limit, len(data))
        # Avoid splitting valid UTF-8 between pages. A tiny limit can grow by <=3 bytes.
        while count < len(data) and 0x80 <= data[count] < 0xC0:
            count += 1
        data = data[:count]
        end = start + len(data)
        return {
            "result_id": result_id,
            "stream": stream,
            "offset": start,
            "next_offset": end,
            "total_bytes": size,
            "eof": end >= size,
            "text": data.decode("utf-8", errors="replace"),
        }

    def _search(self, path, result_id, stream, pattern, offset, limit, size):
        needle = pattern.encode()
        start = min(offset, size)
        # Bounded scan and bounded result. Cursor can continue even with zero matches.
        end = min(size, start + 8 * 1024 * 1024)
        matches, spent, cursor = [], 0, start
        with path.open("rb") as fh:
            while cursor < end and spent < limit and len(matches) < 20:
                fh.seek(cursor)
                advance = min(65536, end - cursor)
                data = fh.read(advance + len(needle) - 1)
                found = data.find(needle)
                if found < 0:
                    cursor += advance
                    continue
                position = cursor + found
                lo = max(0, position - 120)
                fh.seek(lo)
                snippet = fh.read(min(360, limit - spent))
                matches.append(
                    {"offset": position, "text": snippet.decode("utf-8", errors="replace")}
                )
                spent += len(snippet)
                cursor = position + len(needle)
        return {
            "result_id": result_id,
            "stream": stream,
            "matches": matches,
            "next_offset": min(cursor, size),
            "total_bytes": size,
            "eof": cursor >= size,
        }

    def retention_status(self):
        # Pure in-memory status must not queue behind fsync or directory scans.
        # Mutations still hold the I/O lock; this short lock only protects status.
        with self._status_lock:
            return dict(
                ttl_seconds=self.ttl_seconds,
                cleanup_interval_seconds=self.cleanup_interval_seconds,
                active_results=len(self._active),
                last_cleanup=self._last_cleanup,
                request_receipts="retained separately; TTL never enables request replay",
            )

    def cleanup(self, now=None):
        """Expire completed outputs and abandoned old results; never live writers.

        Only this store's UUID directories are eligible. Rename under the same
        lock as reads to prevent partial reads, then unlink outside the lock.
        Never follow a symlink or touch the separate request journal.
        """
        now = time.time() if now is None else now
        report = dict(
            at=now,
            removed=0,
            removed_bytes=0,
            active_skipped=0,
            invalid_skipped=0,
            abandoned_removed=0,
            errors=0,
        )
        with os.scandir(self.root) as entries:
            for entry in entries:
                rid = entry.name
                garbage = re.fullmatch(r"\.gc-[0-9a-f]{32}", rid)
                if not (re.fullmatch(r"[0-9a-f]{32}", rid) or garbage):
                    continue
                if not entry.is_dir(follow_symlinks=False):
                    continue
                try:
                    with self._lock:
                        directory = Path(entry.path)
                        if rid in self._active:
                            report["active_skipped"] += 1
                            continue
                        if not garbage:
                            meta_path = directory / "meta.json"
                            if meta_path.is_symlink():
                                report["invalid_skipped"] += 1
                                continue
                            try:
                                meta = json.loads(meta_path.read_text())
                                stamp = (
                                    meta.get("finished_at")
                                    if meta.get("status") == "complete"
                                    else meta.get("created_at")
                                )
                                if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                                    raise ValueError("invalid timestamp")
                            except (ValueError, TypeError, AttributeError, FileNotFoundError):
                                report["invalid_skipped"] += 1
                                continue
                            if now - stamp < self.ttl_seconds:
                                continue
                            if meta.get("status") != "complete":
                                report["abandoned_removed"] += 1
                            trash = self.root / (".gc-" + rid)
                            directory.rename(trash)
                        else:
                            trash = directory
                    # Result directories contain only store-owned output files.
                    size = sum(
                        p.stat(follow_symlinks=False).st_size
                        for p in trash.iterdir()
                        if not p.is_symlink() and p.is_file()
                    )
                    shutil.rmtree(trash)
                    report["removed"] += 1
                    report["removed_bytes"] += size
                except OSError:
                    report["errors"] += 1
        with self._status_lock:
            self._last_cleanup = report
        logging.getLogger(__name__).info("result_retention %s", json.dumps(report))
        return report

    async def start_retention(self):
        if self._cleanup_task is not None:
            return

        async def loop():
            while True:
                try:
                    self._sweeper = asyncio.create_task(asyncio.to_thread(self.cleanup))
                    await asyncio.shield(self._sweeper)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logging.getLogger(__name__).exception(
                        "result retention failed; retrying next interval"
                    )
                await asyncio.sleep(self.cleanup_interval_seconds)

        self._cleanup_task = asyncio.create_task(loop(), name="jms-result-retention")

    async def close_retention(self):
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None
        if self._sweeper is not None:
            await asyncio.gather(self._sweeper, return_exceptions=True)
