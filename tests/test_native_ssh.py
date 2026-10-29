"""Real loopback SSH with disposable keys; never connects to a deployment."""

import asyncio
import contextlib
import os
import shlex
import shutil
import signal
import sys
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncssh
import pytest

from jms_bridge.compact import BridgeService, unwrap
from jms_bridge.config import BastionConfig, ConfigError
from jms_bridge.inventory import Host
from jms_bridge.native_ssh import NativeSSHRunner, pump
from jms_bridge.results import ResultStore
from jms_bridge.runtime_config import BridgeSettings


class KeyServer(asyncssh.SSHServer):
    def begin_auth(self, username):
        return True

    def public_key_auth_supported(self):
        return True

    def validate_public_key(self, username, key):
        return True


@pytest.fixture
async def loopback(tmp_path):
    server_key = asyncssh.generate_private_key("ssh-ed25519")
    client_key = asyncssh.generate_private_key("ssh-ed25519")
    key_path = tmp_path / "client-key"
    key_path.write_bytes(client_key.export_private_key())
    key_path.chmod(0o600)
    processes, commands = set(), []

    async def serve(process):
        commands.append(process.command)
        argv = shlex.split(process.command)
        # Test the Docker argv contract without requiring a Docker daemon on CI.
        if argv[0] == "docker":
            argv = ["bash", "-s"]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tmp_path,
            start_new_session=True,
        )
        processes.add(proc)

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

        streams = [
            asyncio.create_task(incoming()),
            asyncio.create_task(outgoing(proc.stdout, process.stdout)),
            asyncio.create_task(outgoing(proc.stderr, process.stderr)),
        ]
        finished = asyncio.create_task(proc.wait())
        closed = asyncio.create_task(process.wait_closed())
        try:
            await asyncio.wait([finished, closed], return_when=asyncio.FIRST_COMPLETED)
            if proc.returncode is None:
                os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
            await asyncio.gather(*streams[1:], return_exceptions=True)
            process.exit(proc.returncode)
        finally:
            for task in [*streams, finished, closed]:
                task.cancel()
            await asyncio.gather(*streams, finished, closed, return_exceptions=True)
            processes.discard(proc)

    server = await asyncssh.create_server(
        KeyServer,
        "127.0.0.1",
        0,
        server_host_keys=[server_key],
        process_factory=serve,
        encoding=None,
        rdns_lookup=False,
        gss_host="localhost",
        config=None,
    )
    port = server.get_port()
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_bytes(f"[127.0.0.1]:{port} ".encode() + server_key.export_public_key())
    cfg = BastionConfig(
        url="https://bastion.example.com",
        login_user="example-user",
        ssh_port=port,
        ssh_key=key_path,
        max_parallel=2,
    )
    settings = BridgeSettings(
        state_dir=tmp_path / "state", ssh_host="127.0.0.1", known_hosts=known_hosts
    )
    runner = NativeSSHRunner(cfg, settings, store=ResultStore(settings.state_dir / "results"))
    host = Host(name="node", asset_id=str(uuid.uuid4()), account="root", login="example-user")
    yield SimpleNamespace(
        runner=runner, host=host, cfg=cfg, settings=settings, commands=commands, root=tmp_path
    )
    await runner.close()
    server.close()
    await server.wait_closed()
    for proc in list(processes):
        if proc.returncode is None:
            # The server's `process_factory` children are spawned with
            # `start_new_session=True`; on macOS signalling a group whose leader has
            # already exited but not yet been reaped raises PermissionError (EPERM)
            # rather than ProcessLookupError, and the teardown reported that as an
            # error even though the test itself had passed.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()


async def run(test, script, **kwargs):
    return await test.runner.execute_to_store(test.host, script, kwargs.pop("timeout", 5), **kwargs)


def stdout(test, result):
    return test.runner.store.path(result["result_id"], "stdout").read_text()


async def test_isolation_literal_stdin_and_lossless_output(loopback):
    t = loopback
    literal = "quotes ' \" $PATH `false` 中文"
    result = await run(
        t, 'printf "%s" "$VALUE"; export SHARED=no', env={"VALUE": literal}, cwd=str(t.root)
    )
    assert result["ok"] and stdout(t, result) == literal
    result = await run(t, 'printf "%s" "${SHARED-unset}"')
    assert stdout(t, result) == "unset"
    script = f"{shlex.quote(sys.executable)} -c \"print('x' * 100000)\"\nprintf ERR >&2"
    result = await run(t, script)
    assert result["ok"] and len(stdout(t, result)) == 100001
    assert t.runner.store.preview(result["result_id"], 100)["stderr"] == "ERR"
    assert len(t.runner._connections) == 1
    assert result["execution_state"] == "completed" and not result["completion_unknown"]
    bad_cwd = await run(t, "printf SHOULD-NOT-RUN", cwd=str(t.root / "missing"))
    assert not bad_cwd["ok"] and "SHOULD-NOT-RUN" not in stdout(t, bad_cwd)


async def test_named_shell_serializes_and_persists(loopback):
    t = loopback
    assert (await run(t, "export VALUE=retained; COUNT=0; cd /", session_id="task"))["ok"]
    first, second = await asyncio.gather(
        run(t, "before=$COUNT; sleep .05; COUNT=$((before+1))", session_id="task"),
        run(t, "COUNT=$((COUNT+1))", session_id="task"),
    )
    assert first["ok"] and second["ok"]
    result = await run(t, 'printf "%s|%s|%s" "$PWD" "$VALUE" "$COUNT"', session_id="task")
    assert stdout(t, result) == "/|retained|2"
    result = await run(t, "true", session_id="task", container="another-container")
    assert not result["ok"] and result["execution_state"] == "not_started"
    assert t.runner.sessions()[0]["lost"] is False
    await t.runner.close_session(t.host, "task")
    assert not t.runner.sessions()


async def test_session_exit_and_restart_are_not_silent_resets(loopback):
    t = loopback
    result = await run(t, "exit 2", session_id="exited")
    assert not result["ok"] and result["session_state"] == "lost"
    assert (await run(t, "echo forbidden", session_id="exited"))["execution_state"] == "not_started"
    t.runner._shells.clear()  # Equivalent to a fresh process loading the durable registry.
    restarted = NativeSSHRunner(t.cfg, t.settings, store=t.runner.store)
    result = await restarted.execute_to_store(t.host, "echo forbidden", 5, session_id="exited")
    assert result["session_state"] == "lost" and result["execution_state"] == "not_started"
    await restarted.close_session(t.host, "exited")
    await restarted.close()


async def test_idle_disconnect_recovers_without_replaying_session(loopback):
    t = loopback
    await run(t, "export VALUE=old", session_id="task")
    connection = t.runner._connections[t.runner.key(t.host)]
    connection.abort()
    await connection.wait_closed()
    t.runner._last_login = 0  # Avoid the production login grace period in this test.
    result = await run(t, "printf recovered")
    assert result["ok"] and result["reconnected"] and stdout(t, result) == "recovered"
    lost = await run(t, "echo forbidden", session_id="task")
    assert lost["session_state"] == "lost" and lost["execution_state"] == "not_started"
    assert t.runner.connection_status() == {"connected": 1, "disconnected": 0}


async def test_mid_execution_disconnect_keeps_partial_output_and_never_replays(loopback):
    t = loopback
    before = len(t.commands)
    task = asyncio.create_task(run(t, "printf STARTED; sleep 30; printf FINISHED"))
    for _ in range(200):
        await asyncio.sleep(0.01)
        logs = list(t.runner.store.root.glob("*/stdout.log"))
        if logs and logs[0].read_bytes() == b"STARTED":
            break
    else:
        pytest.fail("execution never started")
    connection = t.runner._connections[t.runner.key(t.host)]
    connection.abort()
    await connection.wait_closed()
    result = await task
    assert not result["ok"] and result["completion_unknown"] and result["executed"] is True
    assert stdout(t, result) == "STARTED"
    assert len(t.commands) == before + 1


@pytest.mark.parametrize("session_id", [None, "timeout"])
async def test_timeouts_preserve_results_and_do_not_poison_pool(loopback, session_id):
    t = loopback
    result = await run(t, "printf STARTED; sleep 30", timeout=0.1, session_id=session_id)
    assert result["timed_out"] and result["exit_code"] == 124 and result["completion_unknown"]
    assert "STARTED" in stdout(t, result)
    if session_id:
        assert result["session_state"] == "lost"
    assert (await run(t, "true"))["ok"]


async def test_cancellation_retains_metadata(loopback):
    t = loopback
    task = asyncio.create_task(run(t, "printf STARTED; sleep 30"))
    for _ in range(100):
        await asyncio.sleep(0.01)
        logs = list(t.runner.store.root.glob("*/stdout.log"))
        if logs and logs[0].read_bytes():
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = t.runner.store.metadata(logs[0].parent.name)
    assert result["exit_code"] == 130 and result["completion_unknown"]


async def test_only_pre_transmission_transport_failure_retries(loopback, monkeypatch):
    t = loopback
    old = SimpleNamespace(
        create_process=AsyncMock(side_effect=asyncssh.ConnectionLost("lost")),
        close=lambda: None,
        wait_closed=AsyncMock(),
    )
    new = SimpleNamespace(create_process=AsyncMock(return_value="channel"))
    monkeypatch.setattr(t.runner, "connection", AsyncMock(side_effect=[old, new]))
    channel, recovered = await t.runner.open_process(t.host, "bash -s")
    assert recovered and channel == "channel" and t.runner.connection.await_count == 2
    old.create_process.side_effect = asyncssh.ChannelOpenError(1, "policy denial")
    monkeypatch.setattr(t.runner, "connection", AsyncMock(return_value=old))
    with pytest.raises(asyncssh.ChannelOpenError):
        await t.runner.open_process(t.host, "bash -s")
    assert t.runner.connection.await_count == 1


async def test_host_key_validation_and_login_cooldown(loopback):
    t = loopback
    t.settings.known_hosts.write_text("")
    result = await run(t, "echo forbidden")
    assert not result["ok"] and result["execution_state"] == "not_started"
    assert t.runner.connection_status()["connected"] == 0
    second = await run(t, "echo forbidden")
    assert "waiting before another attempt" in second["warning"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"env": {"BAD-NAME": "value"}},
        {"script": "nul\x00"},
        {"session_id": "bad/id"},
        {"container": "a", "image": "b"},
        {"container": "-invalid"},
        {"docker_args": ["--rm"]},
    ],
)
async def test_invalid_execution_options(loopback, kwargs):
    script = kwargs.pop("script", "true")
    with pytest.raises(ValueError):
        await run(loopback, script, **kwargs)
    assert not loopback.commands


async def test_remote_docker_argv_and_legacy_adapter(loopback):
    t = loopback
    result = await run(
        t, "printf container", container="container-name", docker_args=["--user", "root"]
    )
    assert result["ok"] and stdout(t, result) == "container"
    assert shlex.split(t.commands[-1]) == [
        "docker",
        "exec",
        "-i",
        "--user",
        "root",
        "container-name",
        "bash",
        "-s",
    ]
    result = await run(t, "printf image", image="example/image:tag")
    assert result["ok"] and stdout(t, result) == "image"
    assert "--rm" in shlex.split(t.commands[-1])
    result = (await t.runner.exec_many([t.host], "printf legacy"))[0]
    assert result.ok and result.stdout == "legacy" and result.result_id


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
async def test_rsync_uses_same_connection_and_preserves_space_paths(loopback):
    t = loopback
    source = t.root / "source file.txt"
    source.write_text("literal $PATH `command` 中文")
    remote = t.root / "remote file.txt"
    pushed = await t.runner.rsync(t.host, str(source), str(remote), timeout=5)
    assert pushed.ok, pushed.stderr
    pulled = t.root / "pulled file.txt"
    result = await t.runner.rsync(t.host, str(pulled), str(remote), direction="pull", timeout=5)
    assert result.ok and source.read_bytes() == pulled.read_bytes()
    assert len(t.runner._connections) == 1


async def test_facade_script_file_and_hidden_execution_results(loopback):
    t = loopback
    service = BridgeService(t.cfg, t.settings)
    service.runner = t.runner
    service.manager.runner = t.runner
    row = {
        "id": t.host.asset_id,
        "name": "node",
        "address": "192.0.2.20",
        "active": True,
        "protocols": ["ssh"],
        "accounts": ["root"],
        "account": "root",
        "login": t.host.login,
    }
    service.catalog._fetch = AsyncMock(return_value=[row])
    await service.catalog.refresh()
    script = t.root / "literal file.sh"
    script.write_text('printf "%s" "$VALUE"')
    response = await service.server.call_tool(
        "run", {"targets": ["node"], "script_path": str(script), "env": {"VALUE": "literal $PATH"}}
    )
    assert not response.is_error and unwrap(response)["results"][0]["stdout"] == "literal $PATH"
    legacy = await service.call("jms_exec", {"targets": ["node"], "command": "printf legacy"})
    assert legacy["ok"]
    probe = await service.call("jms_probe", {"targets": ["node"]})
    assert probe["data"]["execution_results"]
    await service.run(["node"], script="true", session_id="task")
    result = await service.call("session_close", {"target": "node", "session_id": "task"})
    assert result["data"]["closed"]


async def test_marker_boundary_pump(tmp_path):
    reader = asyncio.StreamReader()
    reader.feed_data(b"prefixMAR")
    reader.feed_data(b"KERbody")
    reader.feed_eof()
    progress = {}
    path = tmp_path / "output"
    assert await pump(reader, path, b"MARKER", progress)
    assert path.read_bytes() == b"prefixbody" and progress["started"]


def test_target_requires_rest_identity(loopback=None):
    # No SSH attempt should occur for an incomplete inventory entry.
    runner = object.__new__(NativeSSHRunner)
    with pytest.raises(ConfigError):
        runner.key(Host("unconfigured"))


async def test_jms_exec_keeps_the_legacy_contract(loopback):
    """`jms_exec` must keep the default mode's contract, not silently become `run`.

    The two differ in control flow even when the snapshot has no `env_setup`/`workdir`
    (REST never provides them): the legacy path goes through `wrap_command` and reports
    a dangerous command as `{"ok": false, "error": ...}` instead of raising. Mapping the
    name onto `run` made one operation mean two things depending on the mode serving it.
    """
    t = loopback
    service = BridgeService(t.cfg, t.settings)
    service.runner = t.runner
    service.manager.runner = t.runner
    service.catalog._fetch = AsyncMock(
        return_value=[
            {
                "id": t.host.asset_id,
                "name": "node",
                "address": "192.0.2.21",
                "active": True,
                "protocols": ["ssh"],
                "accounts": ["root"],
                "account": "root",
                "login": t.host.login,
            }
        ]
    )
    await service.catalog.refresh()

    calls = []
    original = t.runner.exec_on

    async def spy(host, command, *, timeout=60, workdir=True, env_setup=True, preview_only=False):
        calls.append(command)
        assert preview_only, "jms_exec is a presentation-only consumer"
        return await original(
            host,
            command,
            timeout=timeout,
            workdir=workdir,
            env_setup=env_setup,
            preview_only=preview_only,
        )

    t.runner.exec_on = spy
    result = await service.call("jms_exec", {"targets": ["node"], "command": "printf legacy-path"})
    assert result["ok"] and "exec_on" not in str(result)
    assert calls == ["printf legacy-path"], "jms_exec must go through exec_on, not run"
    assert result["data"]["results"][0]["result_id"]

    # A dangerous command is refused, not raised: the legacy reply shape.
    refused = await service.call(
        "jms_exec", {"targets": ["node"], "command": "rm -rf / --no-preserve-root"}
    )
    assert refused["data"]["ok"] is False
    assert "denylist" in refused["data"]["error"]


async def test_rsync_executes_the_validated_binary_even_if_path_changes(loopback, monkeypatch):
    from jms_bridge import native_ssh

    t = loopback
    binary = str(t.root / "validated rsync")
    monkeypatch.setattr(t.runner._rsync_binary, "resolve", AsyncMock(return_value=binary))

    async def connect(host):
        monkeypatch.setenv("PATH", "/changed/after/preflight")
        return SimpleNamespace()

    monkeypatch.setattr(t.runner, "connection", connect)
    spawn = AsyncMock(side_effect=OSError("stop before transfer"))
    monkeypatch.setattr(native_ssh.asyncio, "create_subprocess_exec", spawn)
    result = await t.runner.rsync(t.host, "/local source", "/remote destination")
    assert not result.ok
    argv = spawn.call_args.args
    assert argv[0] == binary
    assert argv[1] == "-azs"
    assert "--itemize-changes" in argv and "--" in argv
    assert argv[-2:] == ("/local source", f"{t.host.asset_id}:/remote destination")
    assert spawn.call_args.kwargs["start_new_session"] is True


async def test_rsync_preflight_failure_creates_no_result_or_ssh_connection(loopback, monkeypatch):
    t = loopback
    monkeypatch.setattr(
        t.runner._rsync_binary,
        "resolve",
        AsyncMock(side_effect=ConfigError("local rsync capability check failed")),
    )
    connection = AsyncMock()
    monkeypatch.setattr(t.runner, "connection", connection)
    with pytest.raises(ConfigError, match="capability check failed"):
        await t.runner.rsync(t.host, "/local", "/remote")
    connection.assert_not_awaited()
    assert not list(t.runner.store.root.iterdir())
    assert sum(t.runner.scheduler.active.values()) == 0


async def test_rsync_cancel_during_preflight_does_not_start_transfer(loopback, monkeypatch):
    t = loopback
    started = asyncio.Event()

    async def pending_probe():
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(t.runner._rsync_binary, "resolve", pending_probe)
    connection = AsyncMock()
    monkeypatch.setattr(t.runner, "connection", connection)
    task = asyncio.create_task(t.runner.rsync(t.host, "/local", "/remote"))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    connection.assert_not_awaited()
    assert not list(t.runner.store.root.iterdir())


async def test_large_legacy_exec_previews_preserve_full_logs_and_error_tail(loopback, monkeypatch):
    import json

    t = loopback
    service = BridgeService(t.cfg, t.settings)
    service.runner = t.runner
    service.manager.runner = t.runner
    service.catalog._fetch = AsyncMock(
        return_value=[
            {
                "id": t.host.asset_id,
                "name": "node",
                "address": "192.0.2.20",
                "active": True,
                "protocols": ["ssh"],
                "accounts": ["root"],
                "account": "root",
                "login": t.host.login,
            }
        ]
    )
    await service.catalog.refresh()
    code = "import sys; sys.stdout.write('x'*2000000); sys.stderr.write('y'*2000000+'ERROR-TAIL')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"

    def forbid_full_read(*args, **kwargs):
        raise AssertionError("preview consumer read a complete output file")

    monkeypatch.setattr("pathlib.Path.read_bytes", forbid_full_read)
    reply = await service.call("jms_exec", {"targets": ["node"], "command": command})
    rows = json.loads(service.store.path(reply["result_id"], "result").read_text())["results"]
    row = rows[0]
    assert row["ok"] and row["stdout_truncated"] and row["stderr_truncated"]
    assert row["stderr"].endswith("ERROR-TAIL")
    assert len(row["stdout"].encode()) + len(row["stderr"].encode()) <= 4096
    execution_id = row["result_id"]
    meta = service.store.metadata(execution_id)
    assert meta["bytes"]["stdout"] == 2000000 and meta["bytes"]["stderr"] == 2000010
    offset, total = 0, 0
    while True:
        page = service.store.read(execution_id, offset=offset, limit=16384)
        assert page["text"] == "x" * len(page["text"])
        total += len(page["text"])
        offset = page["next_offset"]
        if page["eof"]:
            break
    assert total == 2000000
    assert service.store.read(execution_id, "stderr", tail=True)["text"].endswith("ERROR-TAIL")


async def test_rsync_finalization_does_not_materialize_large_logs(loopback, monkeypatch):
    from jms_bridge import native_ssh

    t = loopback
    monkeypatch.setattr(
        t.runner._rsync_binary, "resolve", AsyncMock(return_value="/validated/rsync")
    )

    async def spawn(*args, **kwargs):
        kwargs["stdout"].write(b"i" * 2000000)
        kwargs["stderr"].write(b"e" * 2000000 + b"FINAL-RSYNC-ERROR")
        return SimpleNamespace(wait=AsyncMock(return_value=1), returncode=1)

    monkeypatch.setattr(native_ssh.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(t.runner, "connection", AsyncMock(return_value=SimpleNamespace()))

    def forbid_full_read(*args, **kwargs):
        raise AssertionError("transfer finalization read a complete output file")

    original_read = Path.read_bytes
    monkeypatch.setattr("pathlib.Path.read_bytes", forbid_full_read)
    result = await t.runner.rsync(t.host, "/local", "/remote")
    assert not result.ok and result.exit_code == 1
    assert result.stdout_truncated and result.stderr_truncated
    assert len(result.stdout) == 16384 and len(result.stderr) == 16384
    service = BridgeService(t.cfg, t.settings)
    rows = service.compact_exec_results([result])
    assert rows[0]["stderr"].endswith("FINAL-RSYNC-ERROR")
    assert rows[0]["stdout_truncated"] and rows[0]["stderr_truncated"]
    assert result.output_preview.sizes == {"stdout": 2000000, "stderr": 2000017}
    monkeypatch.setattr(Path, "read_bytes", original_read)
    full = await t.runner.rsync(t.host, "/local", "/remote", preview_only=False)
    assert full.stdout == "i" * 2000000
    assert full.stderr == "e" * 2000000 + "FINAL-RSYNC-ERROR"
    assert full.output_preview is None and not hasattr(full, "_bridge_output")
    await service.runner.close()


async def test_other_ssh_work_remains_responsive_during_large_preview_io(loopback, monkeypatch):
    t = loopback
    await run(t, "true")  # Warm the one shared connection before measuring storage behavior.
    entered, release = threading.Event(), threading.Event()
    original = t.runner.store.preview_snapshot
    loop_thread = threading.get_ident()

    def slow(*args, **kwargs):
        assert threading.get_ident() != loop_thread
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(t.runner.store, "preview_snapshot", slow)
    code = "import sys; sys.stdout.write('x'*2000000)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    task = asyncio.create_task(t.runner.exec_on(t.host, command, preview_only=True))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert not task.done()
        assert sum(t.runner.scheduler.active.values()) == 0
        small = await asyncio.wait_for(run(t, "printf STILL-RESPONSIVE"), 2)
        assert small["ok"] and stdout(t, small) == "STILL-RESPONSIVE"
        assert not task.done()
    finally:
        release.set()
        result = await task
    assert result.ok and result.stdout_truncated
