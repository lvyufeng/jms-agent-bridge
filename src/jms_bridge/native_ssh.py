"""Process-owned SSH connections, durable results and optional persistent shells."""

import asyncio
import contextlib
import contextvars
import json
import logging
import os
import re
import shlex
import sys
import tempfile
import time
import uuid
from pathlib import Path

import asyncssh

from .async_io import offload
from .config import ConfigError
from .results import atomic_json
from .rsync_binary import RsyncBinary
from .scheduling import ExecutionSlots, QueueTimeout, Timings
from .ssh import ExecResult


class ConnectionTimings(asyncssh.SSHClient):
    """Observe protocol phases without changing credential selection or authentication."""

    def __init__(self, timing, started):
        self.timing, self.started, self.auth_started = timing, started, None

    def begin_auth(self, username):
        self.auth_started = time.monotonic()
        if self.timing:
            self.timing.connection_details["proxy_and_handshake_seconds"] = (
                self.auth_started - self.started
            )

    def auth_completed(self):
        if self.timing and self.auth_started is not None:
            self.timing.connection_details["authentication_seconds"] = (
                time.monotonic() - self.auth_started
            )


def open_output(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


async def pump(reader, path, marker=None, progress=None):
    found, buffer = False, b""
    with open_output(path) as out:
        while True:
            try:
                chunk = await reader.read(65536)
            except BaseException:
                out.write(buffer)
                raise
            if not chunk:
                out.write(buffer)
                return found
            if marker is None or found:
                out.write(chunk)
                out.flush()
                continue
            buffer += chunk
            position = buffer.find(marker)
            if position >= 0:
                out.write(buffer[:position])
                out.write(buffer[position + len(marker) :])
                buffer, found = b"", True
                if progress is not None:
                    progress["started"] = True
                out.flush()
            elif len(buffer) >= len(marker):
                keep = len(marker) - 1
                out.write(buffer[:-keep])
                buffer = buffer[-keep:]


class ShellSession:
    def __init__(self, process, key, lock=None):
        self.process, self.key = process, key
        self.lock = lock or asyncio.Lock()
        self.lost = False
        self.last_used = time.monotonic()
        self.started = False
        self.buffers = {"stdout": b"", "stderr": b""}

    async def capture(self, stream, path, start, end):
        reader = getattr(self.process, stream)
        buffer = self.buffers[stream]
        active = False
        with open_output(path) as out:
            while True:
                if not active:
                    position = buffer.find(start)
                    if position >= 0:
                        buffer, active = buffer[position + len(start) :], True
                        self.started = True
                    elif len(buffer) >= len(start):
                        buffer = buffer[-(len(start) - 1) :]
                if active:
                    position = buffer.find(end)
                    if position >= 0:
                        suffix = buffer[position + len(end) :]
                        terminal = suffix.find(b"\x1f")
                        if terminal >= 0:
                            out.write(buffer[:position])
                            self.buffers[stream] = suffix[terminal + 1 :]
                            return int(suffix[:terminal])
                    keep = len(end) + 16
                    if len(buffer) > keep:
                        out.write(buffer[:-keep])
                        buffer = buffer[-keep:]
                try:
                    chunk = await reader.read(65536)
                except BaseException:
                    if active:
                        out.write(buffer)
                    raise
                if not chunk:
                    if active:
                        out.write(buffer)
                    self.lost = True
                    raise ConnectionError(
                        "persistent shell exited before completion; session state was lost"
                    )
                buffer += chunk


class NativeSSHRunner:
    def __init__(self, cfg, settings, *, store):
        self.cfg, self.settings, self.store = cfg, settings, store
        self._rsync_binary = RsyncBinary()
        limit = settings.execution_limit or cfg.max_parallel
        reserved = settings.inspection_reserved
        self.scheduler = ExecutionSlots(limit, reserved)
        self.queue_timeout = settings.queue_timeout
        self._timings_context = contextvars.ContextVar("execution_timings", default=None)
        self._session_gates = {}
        self._connect_lock = asyncio.Lock()
        self._connections = {}
        self._shells = {}
        self._shells_lock = asyncio.Lock()
        self._last_login = 0.0
        self._failed_until = {}
        self._registry = self.store.root.parent / "shell_sessions.json" if self.store else None
        self._known_sessions = set()
        self.result_context = contextvars.ContextVar("execution_results", default=None)
        if self._registry and self._registry.exists():
            self._known_sessions = {
                (tuple(row["target"]), row["session_id"])
                for row in json.loads(self._registry.read_text())
            }

    def save_sessions(self):
        if self._registry:
            atomic_json(
                self._registry,
                [
                    {"target": list(key[0]), "session_id": key[1]}
                    for key in sorted(self._known_sessions)
                ],
            )

    def record_result(self, result_id, host):
        records = self.result_context.get()
        if records is not None:
            records.append({"host": host.name, "result_id": result_id})

    def phase(self, name):
        timing = self._timings_context.get()
        return timing.phase(name) if timing else contextlib.nullcontext()

    def performance_log(self, meta):
        fields = (
            "result_id",
            "kind",
            "lane",
            "ok",
            "exit_code",
            "elapsed_seconds",
            "timing",
            "connection_timing",
            "queue_timed_out",
            "failure_stage",
            "completion_unknown",
        )
        logging.getLogger("jms_bridge.performance").info(
            "%s",
            json.dumps(
                {
                    "event": "execution_finished",
                    "at": time.time(),
                    **{k: meta.get(k) for k in fields},
                },
                separators=(",", ":"),
            ),
        )

    async def open_process(self, host, command):
        """Retry a transport failure once, only before any user script is sent."""
        previous = self._connections.get(self.key(host))
        connection = await self.connection(host)
        reconnected = previous is not None and previous is not connection
        try:
            with self.phase("channel_seconds"):
                return await connection.create_process(command, encoding=None), reconnected
        except (
            asyncssh.ConnectionLost,
            asyncssh.DisconnectError,
            BrokenPipeError,
            ConnectionResetError,
        ):
            connection.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(connection.wait_closed(), 5)
            connection = await self.connection(host)
            with self.phase("channel_seconds"):
                return await connection.create_process(command, encoding=None), True

    def connection_status(self):
        return {
            "connected": sum(not c.is_closed() for c in self._connections.values()),
            "disconnected": sum(c.is_closed() for c in self._connections.values()),
        }

    def key(self, host):
        if not host.asset_id or not host.account or not host.login:
            raise ConfigError("target needs a permitted REST asset ID and account")
        return (host.login, host.account, str(uuid.UUID(host.asset_id)))

    async def connection(self, host):
        with self.phase("connection_seconds"):
            return await self._connection(host)

    def connection_phase(self, name):
        timing = self._timings_context.get()
        return timing.connection_phase(name) if timing else contextlib.nullcontext()

    @contextlib.asynccontextmanager
    async def login_lock(self):
        with self.connection_phase("login_queue_seconds"):
            await self._connect_lock.acquire()
        try:
            yield
        finally:
            self._connect_lock.release()

    async def _connection(self, host):
        key = self.key(host)
        connection = self._connections.get(key)
        if connection and not connection.is_closed():
            return connection
        async with self.login_lock():
            connection = self._connections.get(key)
            if connection and not connection.is_closed():
                return connection
            if time.monotonic() < self._failed_until.get(key, 0):
                raise ConfigError(
                    "previous SSH authentication failed; waiting before another attempt"
                )
            with self.connection_phase("login_pacing_seconds"):
                await asyncio.sleep(max(0, 5 - (time.monotonic() - self._last_login)))
            cfg = self.cfg
            connect_started = time.monotonic()
            timing = self._timings_context.get()

            def client_factory():
                return ConnectionTimings(timing, connect_started)

            try:
                options = {
                    "port": cfg.ssh_port,
                    "username": host.login_string(cfg.login_user),
                    "preferred_auth": ["publickey", "password"],
                    "config": None,
                    "keepalive_interval": 30,
                    "keepalive_count_max": 3,
                    "connect_timeout": 30,
                    "login_timeout": 60,
                    "encoding": None,
                }
                if cfg.ssh_key:
                    options["client_keys"] = [str(cfg.ssh_key)]
                if cfg.ssh_password:
                    options["password"] = cfg.ssh_password
                if self.settings.known_hosts:
                    options["known_hosts"] = str(self.settings.known_hosts)
                if self.settings.host_key_alias:
                    options["host_key_alias"] = self.settings.host_key_alias
                if self.settings.ssh_proxy_command:
                    options["proxy_command"] = self.settings.ssh_proxy_command
                options["client_factory"] = client_factory
                connection = await asyncssh.connect(self.settings.gateway_host(cfg), **options)
            except Exception as exc:
                self._failed_until[key] = time.monotonic() + 60
                raise ConfigError(
                    f"SSH connection failed ({type(exc).__name__}); no automatic login retry"
                ) from None
            finally:
                if timing:
                    timing.connection_details["connect_total_seconds"] = (
                        time.monotonic() - connect_started
                    )
                self._last_login = time.monotonic()
            self._connections[key] = connection
            return connection

    def command(self, container=None, image=None, docker_args=None, persistent=False):
        if container and image:
            raise ValueError("choose container or image, not both")
        shell = ["bash", "--noprofile", "--norc"] if persistent else ["bash", "-s"]
        if container or image:
            target = container or image
            if not target or target.startswith("-") or "\x00" in target:
                raise ValueError("invalid Docker target")
            if container:
                args = ["docker", "exec", "-i", *(docker_args or []), target, *shell]
            else:
                args = [
                    "docker",
                    "run",
                    "--rm",
                    "-i",
                    *(docker_args or []),
                    "--entrypoint",
                    "bash",
                    target,
                    *(["--noprofile", "--norc"] if persistent else ["-s"]),
                ]
            return shlex.join(args)
        if docker_args:
            raise ValueError("docker_args requires container or image")
        return shlex.join(shell)

    async def execute_to_store(
        self,
        host,
        script,
        timeout,
        *,
        cwd=None,
        env=None,
        container=None,
        image=None,
        docker_args=None,
        session_id=None,
        wrapped=False,
        lane="general",
        queue_timeout=None,
        _internal_session=False,
    ):
        if env and any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in env):
            raise ValueError("invalid environment variable name")
        if any(
            "\x00" in x for x in [script, cwd or "", *(env or {}).values(), *(docker_args or [])]
        ):
            raise ValueError("NUL bytes are not valid shell arguments")
        if _internal_session and (
            not session_id or not re.fullmatch(r"_fs:[0-9a-f]{32}", session_id)
        ):
            raise ValueError("invalid private filesystem worker ID")
        if (
            session_id
            and not _internal_session
            and not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", session_id)
        ):
            raise ValueError(
                "session_id must contain 1..64 letters, digits, dots, underscores, or hyphens"
            )
        if lane not in ("general", "inspection"):
            raise ValueError("invalid execution lane")
        queue_timeout = self.queue_timeout if queue_timeout is None else queue_timeout
        if queue_timeout <= 0:
            raise ValueError("queue_timeout must be positive")
        command = self.command(container, image, docker_args, bool(session_id))
        result_id = await self.store.acreate(
            kind="fs_worker" if _internal_session else "run",
            host=host.name,
            asset_id=host.asset_id,
            container=container,
            image=image,
            session_id=session_id,
            lane=lane,
        )
        try:
            await offload(self.record_result, result_id, host)
        except BaseException:
            await offload(
                self.store.finish,
                result_id,
                ok=False,
                executed=False,
                execution_state="not_started",
                completion_unknown=False,
                exit_code=130,
            )
            raise
        paths = {s: self.store.path(result_id, s) for s in ("stdout", "stderr")}
        started_at = time.monotonic()
        prefix = [] if session_id or wrapped else ["set -euo pipefail"]
        prefix += ["export " + k + "=" + shlex.quote(v) for k, v in (env or {}).items()]
        body = "\n".join([*prefix, script]) + "\n"
        if cwd:
            body = (
                "if builtin cd -- " + shlex.quote(cwd) + "; then\n" + body + "\nelse\n false\nfi\n"
            )
        code, executed, timed_out, warning, process = 75, False, False, None, None
        sent, completed, reconnected, shell = False, False, False, None
        progress = {"started": False}
        timing = Timings()
        timing_token = self._timings_context.set(timing)
        session_gate, session_held = None, False
        execution_started, failure_stage, queue_timed_out = None, None, False
        try:
            if session_id:
                key = (self.key(host), session_id)
                session_gate = self._session_gates.setdefault(key, asyncio.Lock())
                with timing.phase("queue_seconds"):
                    try:
                        await asyncio.wait_for(session_gate.acquire(), queue_timeout)
                    except asyncio.TimeoutError:
                        raise QueueTimeout("session_queue") from None
                session_held = True
                existing = self._shells.get(key)
                if (
                    _internal_session
                    and existing
                    and (existing.lost or existing.process.is_closing())
                ):
                    existing.process.close()
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(existing.process.wait_closed(), 5)
                    self._shells.pop(key, None)
                    existing = None
                if existing is None and key in self._known_sessions:
                    raise ValueError(
                        "session state was lost after service restart; "
                        "use a new session_id or call session_close to reset it"
                    )
                if existing and (existing.lost or existing.process.is_closing()):
                    raise ValueError(
                        "session state was lost; inspect prior result, then use a new session_id"
                    )
            # Authenticate before reserving execution capacity: a cold login must
            # not block already-authenticated targets waiting for command slots.
            failure_stage = "connection"
            prior_connection = self._connections.get(self.key(host))
            prepared_connection = await self.connection(host)
            reconnected = (
                prior_connection is not None and prior_connection is not prepared_connection
            )
            failure_stage = "execution_queue"
            async with self.scheduler.slot(
                lane, max(0.001, queue_timeout - timing.values["queue_seconds"]), timing
            ):
                failure_stage = "channel"
                if session_id:
                    async with self._shells_lock:
                        key = (self.key(host), session_id)
                        shell = self._shells.get(key)
                        if shell is None and key in self._known_sessions:
                            raise ValueError(
                                "session state was lost after service restart; "
                                "use a new session_id or call session_close to reset it"
                            )
                        options = (container, image, tuple(docker_args or []))
                        if shell and shell.key != options:
                            raise ValueError(
                                "session_id already uses a different container/image; use a new ID"
                            )
                        if shell and (shell.lost or shell.process.is_closing()):
                            raise ValueError(
                                "session state was lost; "
                                "inspect prior result, then use a new session_id"
                            )
                        if shell is None:
                            process, channel_reconnected = await self.open_process(host, command)
                            reconnected = reconnected or channel_reconnected
                            if not _internal_session:
                                self._known_sessions.add(key)
                                try:
                                    await offload(self.save_sessions)
                                except BaseException:
                                    process.close()
                                    with contextlib.suppress(Exception):
                                        await asyncio.wait_for(process.wait_closed(), 5)
                                    raise
                            shell = ShellSession(process, options, lock=session_gate)
                            self._shells[key] = shell
                    if shell.lost or shell.process.is_closing():
                        raise ValueError(
                            "session state was lost; "
                            "use a new session_id or call session_close to reset it"
                        )
                    shell.started = False
                    nonce = uuid.uuid4().hex.encode()
                    start, end = b"\x1eSTART" + nonce + b"\x1f", b"\x1eEND" + nonce + b":"
                    # Quoting is generated once by the transport; the caller supplies literal Bash.
                    packet = (
                        f"printf '\\036START{nonce.decode()}\\037'; "
                        f"printf '\\036START{nonce.decode()}\\037' >&2\n"
                        + "builtin eval -- "
                        + shlex.quote(body)
                        + "\n"
                        + f"__jms_rc=$?; printf '\\036END{nonce.decode()}:%s\\037' \"$__jms_rc\"; "
                        f"printf '\\036END{nonce.decode()}:%s\\037' \"$__jms_rc\" >&2\n"
                    ).encode()
                    try:
                        execution_started = time.monotonic()
                        failure_stage = "execution"
                        sent = True
                        shell.process.stdin.write(packet)
                        await shell.process.stdin.drain()
                        captures = [
                            asyncio.create_task(shell.capture(s, paths[s], start, end))
                            for s in ("stdout", "stderr")
                        ]
                        try:
                            codes = await asyncio.wait_for(asyncio.gather(*captures), timeout)
                        finally:
                            for task in captures:
                                if not task.done():
                                    task.cancel()
                            await asyncio.gather(*captures, return_exceptions=True)
                        code, executed = codes[0], codes[0] == codes[1]
                        completed = executed
                        shell.last_used = time.monotonic()
                    except BaseException:
                        executed = shell.started
                        shell.lost = True
                        shell.process.close()
                        raise
                else:
                    marker = ("__JMS_EXEC_" + uuid.uuid4().hex + "__\n").encode()
                    process, channel_reconnected = await self.open_process(host, command)
                    reconnected = reconnected or channel_reconnected
                    tasks = [
                        asyncio.create_task(
                            pump(process.stdout, paths["stdout"], marker, progress)
                        ),
                        asyncio.create_task(pump(process.stderr, paths["stderr"])),
                    ]
                    try:
                        execution_started = time.monotonic()
                        failure_stage = "execution"
                        sent = True
                        process.stdin.write(
                            b"printf '%s\\n' "
                            + shlex.quote(marker.decode().strip()).encode()
                            + b"\n"
                            + body.encode()
                        )
                        process.stdin.write_eof()
                        await asyncio.wait_for(process.wait_closed(), timeout)
                        await asyncio.gather(*tasks)
                        executed = progress["started"]
                        completed = process.exit_status is not None and process.exit_status >= 0
                    except BaseException:
                        process.close()
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(process.wait_closed(), 5)
                        raise
                    finally:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        executed = progress["started"]
                    code = process.exit_status if process.exit_status is not None else 255
                    if not executed and code == 0:
                        code = 75
                    if not completed:
                        warning = (
                            "SSH channel ended without a completion status; "
                            "remote completion is unknown."
                        )
        except QueueTimeout as exc:
            code, timed_out, queue_timed_out = 124, True, True
            failure_stage, warning = exc.stage, str(exc)
        except asyncio.TimeoutError:
            code, timed_out = 124, True
            warning = (
                "Execution timed out; remote completion is unknown. Inspect before rerunning."
                if sent
                else "Connection setup timed out; command was not sent."
            )
        except asyncio.CancelledError:
            code = 130
            warning = (
                "Call cancelled; remote completion is unknown."
                if sent
                else "Call cancelled before transmission; command was not sent."
            )
            raise
        except Exception as exc:
            warning = (
                str(exc)
                if isinstance(exc, (ValueError, ConfigError, ConnectionError))
                else f"execution failed ({type(exc).__name__})"
            )
        finally:
            if execution_started is not None:
                timing.values["execution_seconds"] += time.monotonic() - execution_started
            if session_held:
                session_gate.release()
            self._timings_context.reset(timing_token)
            if session_id:
                shell = self._shells.get((self.key(host), session_id))
            unknown = sent and not completed
            if unknown and warning and "unknown" not in warning:
                warning += "; remote completion is unknown; do not replay automatically"
            for path in paths.values():
                if not path.exists():
                    with open_output(path):
                        pass
            if warning:
                with paths["stderr"].open("ab") as out:
                    out.write(("\n" + warning + "\n").encode())
            meta = await offload(
                self.store.finish,
                result_id,
                ok=code == 0 and executed,
                exit_code=code,
                timed_out=timed_out,
                executed=True if executed else (None if unknown else False),
                completion_unknown=unknown,
                execution_state="completed"
                if completed
                else ("unknown" if unknown else "not_started"),
                reconnected=reconnected,
                connection_state="connected"
                if (
                    self._connections.get(self.key(host))
                    and not self._connections[self.key(host)].is_closed()
                )
                else "disconnected",
                session_state=(
                    ("active" if not shell.lost and not shell.process.is_closing() else "lost")
                    if shell
                    else (
                        "lost"
                        if (self.key(host), session_id) in self._known_sessions
                        else "not_started"
                    )
                )
                if session_id
                else None,
                elapsed_seconds=round(time.monotonic() - started_at, 3),
                warning=warning,
                timing=timing.snapshot(),
                queue_timed_out=queue_timed_out,
                connection_timing=timing.connection_snapshot(),
                failure_stage=failure_stage if warning else None,
            )
            self.performance_log(meta)
        return meta

    async def exec_on(self, host, command, *, timeout=60, workdir=True, env_setup=True):
        from .inventory import wrap_command

        script = wrap_command(
            host.env_setup if env_setup else None, host.workdir if workdir else None, command
        )
        meta = await self.execute_to_store(host, script, timeout, wrapped=True)
        result_id = meta["result_id"]
        out, err = await offload(
            lambda: (
                self.store.path(result_id, "stdout").read_bytes(),
                self.store.path(result_id, "stderr").read_bytes(),
            )
        )
        result = ExecResult(
            host.name,
            meta["exit_code"],
            out.decode(errors="replace"),
            err.decode(errors="replace"),
            meta["timed_out"],
        )
        result._bridge_output = (out, err)
        result.result_id = result_id
        return result

    async def exec_many(self, hosts, command, *, timeout=60):
        return list(
            await asyncio.gather(*(self.exec_on(host, command, timeout=timeout) for host in hosts))
        )

    async def close_session(self, host, session_id, *, _internal=False):
        if session_id.startswith("_fs:") and not _internal:
            raise ValueError("private filesystem worker is not a user session")
        key = (self.key(host), session_id)
        async with self._shells_lock:
            shell = self._shells.get(key)
            gate = self._session_gates.get(key)
            if (gate and gate.locked()) or (shell and shell.lock.locked()):
                raise ValueError("session is executing; wait for its result before closing")
            if shell:
                shell.process.close()
                if _internal:
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(shell.process.wait_closed(), 5)
                else:
                    await shell.process.wait_closed()
            existed = key in self._known_sessions
            self._shells.pop(key, None)
            self._known_sessions.discard(key)
            if not _internal:
                await offload(self.save_sessions)
            if _internal:
                self._session_gates.pop(key, None)
        return {"ok": True, "closed": existed}

    def sessions(self):
        return [
            {
                "asset_id": key[0][2],
                "session_id": key[1],
                "lost": key not in self._shells
                or self._shells[key].lost
                or self._shells[key].process.is_closing(),
            }
            for key in sorted(self._known_sessions)
        ]

    async def close(self):
        for shell in self._shells.values():
            shell.process.close()
        for connection in self._connections.values():
            connection.close()
        await asyncio.gather(
            *(c.wait_closed() for c in self._connections.values()), return_exceptions=True
        )
        self._shells.clear()
        self._connections.clear()

    async def rsync(
        self,
        host,
        local,
        remote,
        *,
        direction="push",
        delete=False,
        excludes=None,
        timeout=900,
        container=None,
        overwrite=True,
    ):
        """Keep rsync semantics while its SSH stream uses our authenticated connection."""
        if container and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", container):
            raise ValueError("invalid existing container name/ID")
        binary = await self._rsync_binary.resolve()
        result_id = await self.store.acreate(
            kind="transfer", host=host.name, asset_id=host.asset_id, lane="general"
        )
        try:
            await offload(self.record_result, result_id, host)
        except BaseException:
            await offload(
                self.store.finish,
                result_id,
                ok=False,
                executed=False,
                execution_state="not_started",
                completion_unknown=False,
                exit_code=130,
            )
            raise
        out_path, err_path = (
            self.store.path(result_id, "stdout"),
            self.store.path(result_id, "stderr"),
        )
        started = time.monotonic()
        tasks, channels = set(), []
        code, timed_out = 75, False
        timing = Timings()
        token = self._timings_context.set(timing)
        execution_started, transfer_sent, complete = None, False, False
        failure_stage, warning, queue_timed_out = "connection", None, False
        try:
            connection = await self.connection(host)
            failure_stage = "execution_queue"
            async with self.scheduler.slot("general", self.queue_timeout, timing):
                failure_stage = "execution"
                execution_started = time.monotonic()
                with tempfile.TemporaryDirectory(prefix="jms-rsync-", dir="/tmp") as directory:
                    socket_path = str(Path(directory) / "stream.sock")

                    async def handle(reader, writer):
                        nonlocal transfer_sent
                        task = asyncio.current_task()
                        tasks.add(task)
                        try:
                            line = await asyncio.wait_for(reader.readline(), 10)
                            args = json.loads(line)
                            if len(args) < 2 or args[0] != host.asset_id or args[1] != "rsync":
                                raise ValueError("invalid rsync transport request")
                            remote_args = (
                                ["docker", "exec", "-i", container] if container else []
                            ) + args[1:]
                            channel = await connection.create_process(
                                shlex.join(remote_args), encoding=None
                            )
                            transfer_sent = True
                            channels.append(channel)

                            async def incoming():
                                while data := await reader.read(65536):
                                    channel.stdin.write(data)
                                    await channel.stdin.drain()
                                channel.stdin.write_eof()

                            async def outgoing():
                                while data := await channel.stdout.read(65536):
                                    writer.write(data)
                                    await writer.drain()

                            async def errors():
                                with err_path.open("ab") as err:
                                    while data := await channel.stderr.read(65536):
                                        err.write(data)

                            inbound = asyncio.create_task(incoming())
                            try:
                                await asyncio.gather(outgoing(), errors(), channel.wait_closed())
                            finally:
                                inbound.cancel()
                                await asyncio.gather(inbound, return_exceptions=True)
                        except Exception as exc:
                            with err_path.open("ab") as err:
                                err.write(
                                    f"Transport stream failed ({type(exc).__name__})\n".encode()
                                )
                        finally:
                            writer.close()
                            with contextlib.suppress(Exception):
                                await writer.wait_closed()
                            tasks.discard(task)

                    with (
                        open_output(out_path) as out,
                        os.fdopen(
                            os.open(
                                err_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600
                            ),
                            "ab",
                        ) as err,
                    ):
                        server = await asyncio.start_unix_server(handle, path=socket_path)
                        os.chmod(socket_path, 0o600)
                        async with server:
                            bridge = [
                                sys.executable,
                                str(Path(__file__).with_name("rsync_relay.py")),
                                socket_path,
                            ]
                            # Use the exact executable which passed local preflight.
                            # `-s` sends remote paths through rsync's protocol, avoiding
                            # rsync's legacy shell escaping being quoted again by the
                            # relay. Keep it for spaces and other literal path bytes.
                            argv = [binary, "-azs", "-e", shlex.join(bridge)]
                            if not overwrite:
                                argv.append("--ignore-existing")
                            argv.append("--itemize-changes")
                            if delete:
                                argv.append("--delete")
                            for pattern in excludes or []:
                                argv += ["--exclude", pattern]
                            spec = f"{host.asset_id}:{remote}"
                            argv += (
                                ["--", local, spec] if direction == "push" else ["--", spec, local]
                            )
                            proc = await asyncio.create_subprocess_exec(
                                *argv, stdout=out, stderr=err, start_new_session=True
                            )
                            try:
                                await asyncio.wait_for(proc.wait(), timeout)
                                code = proc.returncode
                                complete = True
                            except asyncio.TimeoutError:
                                timed_out, code = True, 124
                                from .ssh import _kill

                                await _kill(proc)
                            except BaseException:
                                from .ssh import _kill

                                await _kill(proc)
                                raise
                            finally:
                                for channel in channels:
                                    channel.close()
                                for task in list(tasks):
                                    task.cancel()
                                await asyncio.gather(*list(tasks), return_exceptions=True)
        except QueueTimeout as exc:
            timed_out, queue_timed_out, code = True, True, 124
            failure_stage, warning = exc.stage, str(exc)
        except asyncio.CancelledError:
            code, warning = 130, "Transfer cancelled"
            raise
        except Exception as exc:
            warning = f"Transfer failed ({type(exc).__name__})"
        finally:
            if execution_started is not None:
                timing.values["execution_seconds"] += time.monotonic() - execution_started
            self._timings_context.reset(token)
            for path in (out_path, err_path):
                if not path.exists():
                    with open_output(path):
                        pass
            if warning:
                with err_path.open("ab") as err:
                    err.write((warning + "\n").encode())
            meta = await offload(
                self.store.finish,
                result_id,
                ok=code == 0,
                exit_code=code,
                timed_out=timed_out,
                elapsed_seconds=round(time.monotonic() - started, 3),
                timing=timing.snapshot(),
                connection_timing=timing.connection_snapshot(),
                queue_timed_out=queue_timed_out,
                failure_stage=failure_stage if code else None,
                executed=transfer_sent,
                completion_unknown=transfer_sent and not complete,
                execution_state="completed"
                if complete
                else ("unknown" if transfer_sent else "not_started"),
            )
            self.performance_log(meta)
        out, err = await offload(lambda: (out_path.read_bytes(), err_path.read_bytes()))
        result = ExecResult(
            host.name, code, out.decode(errors="replace"), err.decode(errors="replace"), timed_out
        )
        result._bridge_output = (out, err)
        result.result_id = result_id
        return result
