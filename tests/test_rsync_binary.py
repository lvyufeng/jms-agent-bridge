"""Behavior tests for local rsync preflight; no deployment or user files are touched."""

import asyncio
import os
import subprocess
import threading
from unittest.mock import Mock

import pytest

from jms_bridge import rsync_binary
from jms_bridge.config import ConfigError
from jms_bridge.rsync_binary import RsyncBinary


@pytest.fixture
def binary(tmp_path, monkeypatch):
    directory = tmp_path / "bin with spaces"
    directory.mkdir()
    path = directory / "rsync"
    path.write_text("test binary\n")
    path.chmod(0o700)
    monkeypatch.setattr(rsync_binary.shutil, "which", lambda name: str(path))
    return path


def success(*args, **kwargs):
    return subprocess.CompletedProcess(
        args[0] if args else [], 0, "rsync  version 3.5.1  protocol version 33\n", ""
    )


async def test_capability_uses_real_options_and_caches_only_unchanged_success(binary, monkeypatch):
    probe = Mock(side_effect=success)
    monkeypatch.setattr(rsync_binary.subprocess, "run", probe)
    resolver = RsyncBinary()
    assert await resolver.resolve() == str(binary.resolve())
    assert await resolver.resolve() == str(binary.resolve())
    probe.assert_called_once_with(
        [str(binary.resolve()), "-s", "--version"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=3,
        check=False,
    )


async def test_missing_binary_is_not_probed_or_cached(monkeypatch):
    monkeypatch.setattr(rsync_binary.shutil, "which", lambda name: None)
    probe = Mock()
    monkeypatch.setattr(rsync_binary.subprocess, "run", probe)
    resolver = RsyncBinary()
    with pytest.raises(ConfigError, match="not installed or not executable"):
        await resolver.resolve()
    probe.assert_not_called()
    assert resolver._validated is None


@pytest.mark.parametrize(
    ("code", "stdout", "stderr", "message"),
    [
        (1, "", "rsync: invalid option -- s\n", "does not support --protect-args"),
        (1, "", "rsync: unknown option '-s'\n", "does not support --protect-args"),
        (0, "", "openrsync: protocol version 29\n", "does not support --protect-args"),
        (0, "rsync version 2.6.9 protocol version 29\n", "", "older than 3.0"),
        (7, "rsync version 3.5.1\n", "probe failed", "exit code 7"),
        (1, "", "", "exit code 1"),
        (0, "usage: rsync [options]\n", "", "unrecognized version output"),
    ],
)
async def test_failed_or_incompatible_probe_is_actionable_and_retryable(
    binary, monkeypatch, code, stdout, stderr, message
):
    probe = Mock(side_effect=[subprocess.CompletedProcess([], code, stdout, stderr), success()])
    monkeypatch.setattr(rsync_binary.subprocess, "run", probe)
    resolver = RsyncBinary()
    with pytest.raises(ConfigError, match=message) as exc:
        await resolver.resolve()
    assert str(binary.resolve()) in str(exc.value)
    assert "Transfer not started" in str(exc.value)
    assert resolver._validated is None
    assert await resolver.resolve() == str(binary.resolve())
    assert probe.call_count == 2


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (subprocess.TimeoutExpired(["rsync"], 3), "timed out after 3s"),
        (PermissionError("PRIVATE DETAIL"), "PermissionError"),
        (FileNotFoundError("PRIVATE DETAIL"), "FileNotFoundError"),
    ],
)
async def test_probe_errors_are_not_mislabeled_as_openrsync_or_cached(
    binary, monkeypatch, failure, message
):
    probe = Mock(side_effect=[failure, success()])
    monkeypatch.setattr(rsync_binary.subprocess, "run", probe)
    resolver = RsyncBinary()
    with pytest.raises(ConfigError, match=message) as exc:
        await resolver.resolve()
    assert "openrsync" not in str(exc.value)
    assert "PRIVATE DETAIL" not in str(exc.value)
    assert resolver._validated is None
    assert await resolver.resolve() == str(binary.resolve())
    assert probe.call_count == 2


async def test_version_on_stderr_is_recognized(binary, monkeypatch):
    monkeypatch.setattr(
        rsync_binary.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], 0, "", "rsync version 3.2.7\n")),
    )
    assert await RsyncBinary().resolve() == str(binary.resolve())


async def test_path_change_and_same_path_replacement_invalidate_cache(binary, monkeypatch):
    probe = Mock(side_effect=success)
    monkeypatch.setattr(rsync_binary.subprocess, "run", probe)
    resolver = RsyncBinary()
    await resolver.resolve()
    binary.write_text("upgraded binary with new contents\n")
    await resolver.resolve()
    assert probe.call_count == 2

    replacement = binary.with_name("replacement")
    replacement.write_text("replacement\n")
    replacement.chmod(0o700)
    os.replace(replacement, binary)
    await resolver.resolve()
    assert probe.call_count == 3

    alternate = binary.with_name("another-rsync")
    alternate.write_text("alternate installation\n")
    alternate.chmod(0o700)
    monkeypatch.setattr(rsync_binary.shutil, "which", lambda name: str(alternate))
    assert await resolver.resolve() == str(alternate.resolve())
    assert probe.call_count == 4


async def test_binary_disappearing_before_or_during_probe_is_not_cached(binary, monkeypatch):
    probe = Mock(side_effect=success)
    monkeypatch.setattr(rsync_binary.subprocess, "run", probe)
    resolver = RsyncBinary()
    monkeypatch.setattr(rsync_binary, "_identity", Mock(side_effect=FileNotFoundError()))
    with pytest.raises(ConfigError, match="Cannot inspect local rsync"):
        await resolver.resolve()
    probe.assert_not_called()

    monkeypatch.setattr(rsync_binary, "_identity", Mock(side_effect=[("before",), ("after",)]))
    with pytest.raises(ConfigError, match="changed during its capability check"):
        await resolver.resolve()
    assert resolver._validated is None

    monkeypatch.setattr(
        rsync_binary, "_identity", Mock(side_effect=[("before",), FileNotFoundError()])
    )
    with pytest.raises(ConfigError, match="changed during its capability check"):
        await resolver.resolve()
    assert resolver._validated is None


async def test_concurrent_cold_probes_are_coalesced_without_blocking_event_loop(
    binary, monkeypatch
):
    started, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()

    def slow_probe(*args, **kwargs):
        assert threading.get_ident() != loop_thread
        started.set()
        assert release.wait(3), "test did not release the probe"
        return success(*args, **kwargs)

    probe = Mock(side_effect=slow_probe)
    monkeypatch.setattr(rsync_binary.subprocess, "run", probe)
    resolver = RsyncBinary()
    tasks = [asyncio.create_task(resolver.resolve()) for _ in range(10)]
    try:
        assert await asyncio.to_thread(started.wait, 1)
        # The probe is still waiting while the event loop services other work.
        await asyncio.sleep(0)
        assert not any(task.done() for task in tasks)
        assert probe.call_count == 1
    finally:
        release.set()
        values = await asyncio.gather(*tasks, return_exceptions=True)
    assert values == [str(binary.resolve())] * 10
    assert probe.call_count == 1


async def test_cancelled_caller_does_not_release_probe_lock_early(binary, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def slow_probe(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return success(*args, **kwargs)

    probe = Mock(side_effect=slow_probe)
    monkeypatch.setattr(rsync_binary.subprocess, "run", probe)
    resolver = RsyncBinary()
    first = asyncio.create_task(resolver.resolve())
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 1)
        first.cancel()
        second = asyncio.create_task(resolver.resolve())
        await asyncio.sleep(0)
        assert not second.done()
        assert probe.call_count == 1
    finally:
        release.set()
        results = await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)
    assert results[1] == str(binary.resolve())
    assert probe.call_count == 1


async def test_real_executable_receives_literal_options(binary, monkeypatch):
    binary.write_text(
        "#!/bin/sh\n"
        '[ "$#" -eq 2 ] && [ "$1" = -s ] && [ "$2" = --version ] || exit 9\n'
        "printf 'rsync version 3.5.1 protocol version 33\\n'\n"
    )
    assert await RsyncBinary().resolve() == str(binary.resolve())


@pytest.mark.parametrize(
    ("output", "code", "message"),
    [
        ("rsync: invalid option -- s", 1, "does not support --protect-args"),
        ("unexpected installation failure", 7, "exit code 7"),
    ],
)
async def test_real_rejected_probe_is_not_cached(binary, output, code, message):
    binary.write_text(f"#!/bin/sh\nprintf '%s\\n' '{output}' >&2\nexit {code}\n")
    resolver = RsyncBinary()
    with pytest.raises(ConfigError, match=message):
        await resolver.resolve()
    assert resolver._validated is None
