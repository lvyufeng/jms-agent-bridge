"""Bound, resumable tool interactions over the resident SSH connection.

The MCP connection can disappear while a command runs. The command task, private
logs and explicit target binding belong to the resident service, not that HTTP
request. A lost SSH shell is never recreated or a command replayed implicitly.
"""

import asyncio
import contextlib
import re
import secrets
import shlex
import uuid
from dataclasses import dataclass, field

from .async_io import offload
from .native_ssh import ShellSession
from .results import atomic_json
from .scheduling import QueueFull, QueueTimeout, Timings


def integer(low, high, default):
    return dict(type="integer", minimum=low, maximum=high, default=default)


COMMON = {"request_id": {"type": "string", "pattern": r"^[A-Za-z0-9_.:-]{1,128}$"}}
PAGES = {
    "wait_ms": integer(0, 30000, 1000),
    "max_bytes": integer(1, 16384, 4096),
    "cursor": {
        "type": "object",
        "additionalProperties": False,
        "properties": {s: {"type": "integer", "minimum": 0} for s in ("stdout", "stderr")},
    },
}
SPECS = {
    "open": (
        {
            "target": {"type": "string", "minLength": 1},
            "container": {"type": "string"},
            "image": {"type": "string"},
            "docker_args": {"type": "array", "items": {"type": "string"}},
            "cwd": {"type": "string"},
            "env": {"type": "object", "additionalProperties": {"type": "string"}},
        },
        ["target"],
    ),
    "exec": (
        {
            "session": {"type": "string"},
            "script": {"type": "string", "maxLength": 1048576},
            "allow_dangerous": {"type": "boolean", "default": False},
            **PAGES,
        },
        ["session", "script"],
    ),
    "read": ({"session": {"type": "string"}, **PAGES}, ["session"]),
    "write": (
        {
            "session": {"type": "string"},
            "input": {"type": "string", "maxLength": 1048576},
            "signal": {"type": "string", "enum": ["INT", "TERM"]},
            "eof": {"type": "boolean", "default": False},
            "allow_dangerous": {"type": "boolean", "default": False},
            **PAGES,
        },
        ["session"],
    ),
    "close": ({"session": {"type": "string"}}, ["session"]),
}


def schemas():
    descriptions = {
        "open": "Bind one exact target/container/cwd/env; returns a session handle.",
        "exec": "Start a command. Running is not completion; logs survive MCP disconnection.",
        "read": "Read incremental output; a cursor re-reads saved bytes. Never runs work.",
        "write": "Send input, EOF or a signal; input acknowledgement is not completion.",
        "close": "Close this shell, possibly interrupting its command. Saved outputs remain.",
    }
    return {
        f"terminal_{action}": {
            "name": f"terminal_{action}",
            "description": descriptions[action],
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {**props, **(COMMON if action != "read" else {})},
                "required": required,
            },
            "annotations": {
                "readOnlyHint": action == "read",
                "destructiveHint": action != "read",
                "openWorldHint": True,
            },
        }
        for action, (props, required) in SPECS.items()
    }


@dataclass
class Command:
    result_id: str
    state: str = "queued"
    sent: bool = False
    started: bool = False
    task: object = None
    exit_code: object = None
    error: object = None
    rejection: dict = field(default_factory=dict)
    cursor: dict = field(default_factory=lambda: dict(stdout=0, stderr=0))
    changed: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class Binding:
    session: str
    host: object
    shell: object
    command: object = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    read_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closed: bool = False


class Interactive:
    def __init__(self, catalog, runner, store, requests, guard):
        self.catalog, self.runner, self.store = catalog, runner, store
        self.requests, self.guard = requests, guard
        self.bindings = {}
        self.registry = store.root.parent / "terminal_sessions.json"
        # Records are for recovery, not shell restoration. Never persist env/stdin.
        self.lost = {}
        self.registry_lock = asyncio.Lock()

    async def start(self):
        import json

        if self.registry.exists():
            records = await offload(lambda: json.loads(self.registry.read_text()))
            self.lost = {r["session"]: r for r in records}
            for row in records:
                if row.get("result_id"):
                    with contextlib.suppress(ValueError, OSError):
                        meta = await offload(self.store.metadata, row["result_id"])
                        if meta.get("status") == "running":
                            await offload(
                                self.store.finish,
                                row["result_id"],
                                ok=False,
                                execution_state="unknown",
                                completion_unknown=True,
                                exit_code=None,
                                error="service restarted; inspect saved output before further work",
                            )

    async def save(self):
        async with self.registry_lock:
            rows = list(self.lost.values()) + [
                dict(
                    session=b.session,
                    asset_id=b.host.asset_id,
                    host=b.host.name,
                    result_id=b.command.result_id if b.command else None,
                )
                for b in self.bindings.values()
                if not b.closed
            ]
            await offload(atomic_json, self.registry, rows)

    def check(self, text, allowed=False):
        if "\x00" in text:
            raise ValueError("NUL is invalid")
        if self.guard(text) and not allowed:
            raise ValueError(
                "JMS_SCRIPT_PATTERN_GUARD: already-authorized action needs allow_dangerous=true; "
                "this acknowledges only the local guard, never client or JumpServer policy"
            )

    def binding(self, session):
        if session in self.lost:
            row = self.lost[session]
            raise ValueError(
                "terminal state lost after restart; never replay automatically; "
                "saved result_id=" + str(row.get("result_id"))
            )
        if session not in self.bindings:
            raise ValueError("unknown terminal session")
        return self.bindings[session]

    async def permitted(self, b):
        inv = await self.catalog.ensure()
        hosts = inv.resolve([b.host.asset_id])
        if len(hosts) != 1 or self.runner.key(hosts[0]) != self.runner.key(b.host):
            raise ValueError("bound asset/account is no longer permitted")

    async def open(self, target, container=None, image=None, docker_args=None, cwd=None, env=None):
        if target.lower() == "all" or re.search(r"[*?\[\]]", target):
            raise ValueError("terminal needs one exact target")
        inv = await self.catalog.ensure()
        hosts = inv.resolve([target])
        if len(hosts) != 1:
            raise ValueError("terminal needs exactly one permitted asset")
        if env and any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) for k in env):
            raise ValueError("invalid environment variable name")
        if any("\x00" in s for s in [cwd or "", *(env or {}).values(), *(docker_args or [])]):
            raise ValueError("NUL is invalid")
        host = hosts[0]
        command = self.runner.command(container, image, docker_args, persistent=True)
        timing = Timings()
        async with self.runner.connection_use(host):
            await self.runner.connection(host)
            async with self.runner.scheduler.slot("general", self.runner.queue_timeout, timing):
                process, _ = await self.runner.open_process(host, command)
                shell = ShellSession(process, (container, image, tuple(docker_args or [])))
                try:
                    await shell.prime(10)
                except BaseException:
                    process.close()
                    raise
        session = secrets.token_urlsafe(6)
        while session in self.bindings or session in self.lost:
            session = secrets.token_urlsafe(6)
        b = Binding(session, host, shell)
        self.bindings[session] = b
        # Let existing connection-recovery logic recognize this protected shell.
        self.runner._shells[(self.runner.key(host), "term-" + session)] = shell
        setup = []
        if cwd:
            setup.append("cd -- " + shlex.quote(cwd))
        setup.extend("export " + k + "=" + shlex.quote(v) for k, v in (env or {}).items())
        try:
            await self.save()
            value = await self.execute(session, " && ".join(setup) or ":", wait_ms=10000)
            if value["state"] != "completed" or value.get("exit_code") != 0:
                raise ValueError("terminal setup failed; result_id=" + value["result_id"])
        except BaseException:
            await self.close(session)
            raise
        return {"ok": True, "session": session, "host": target, "state": "idle"}

    async def execute(self, session, script, allow_dangerous=False, **page):
        self.check(script, allow_dangerous)
        b = self.binding(session)
        await self.permitted(b)
        async with b.lock:
            if b.closed or b.shell.lost or b.shell.process.is_closing():
                raise ValueError("terminal shell was closed/lost; open a new session explicitly")
            if b.command and b.command.task and not b.command.task.done():
                raise ValueError("terminal is busy; read/write its active command first")
            rid = await self.store.acreate(
                kind="terminal", host=b.host.name, asset_id=b.host.asset_id, session_id=session
            )
            c = Command(rid)
            b.command = c
            # Create empty streams before polling; appenders never replace these files.
            for stream in ("stdout", "stderr"):
                await offload(self.store.path(rid, stream).touch, mode=0o600)
            try:
                await self.requests.aprogress(
                    "execution_results", {"host": b.host.name, "result_id": rid}
                )
                await self.save()
            except BaseException:
                c.state = "not_started"
                await offload(
                    self.store.finish,
                    rid,
                    ok=False,
                    executed=False,
                    execution_state="not_started",
                    completion_unknown=False,
                    exit_code=None,
                )
                raise
            c.task = asyncio.create_task(self.work(b, c, script), name="jms-terminal-" + session)
        wait_ms = page.pop("wait_ms", 1000)
        await asyncio.wait({c.task}, timeout=wait_ms / 1000)
        return await self.read(session, wait_ms=0, **page)

    async def capture(self, b, c, stream, start, end):
        buffer = b.shell.buffers[stream]
        active = False
        reader = getattr(b.shell.process, stream)
        path = self.store.path(c.result_id, stream)
        with path.open("ab", buffering=0) as out:
            while True:
                if not active:
                    pos = buffer.find(start)
                    if pos >= 0:
                        buffer, active = buffer[pos + len(start) :], True
                        b.shell.started = c.started = True
                        c.changed.set()
                    else:
                        buffer = buffer[-(len(start) - 1) :]
                if active:
                    pos = buffer.find(end)
                    if pos >= 0:
                        suffix = buffer[pos + len(end) :]
                        stop = suffix.find(b"\x1f")
                        if stop >= 0:
                            out.write(buffer[:pos])
                            b.shell.buffers[stream] = suffix[stop + 1 :]
                            c.changed.set()
                            return int(suffix[:stop])
                        out.write(buffer[:pos])
                        buffer = buffer[pos:]
                    else:
                        # Publish short prompts immediately; retain only a possible
                        # marker prefix, including markers split across SSH packets.
                        keep = next(
                            (
                                n
                                for n in range(min(len(buffer), len(end) - 1), 0, -1)
                                if buffer.endswith(end[:n])
                            ),
                            0,
                        )
                        chunk = buffer[:-keep] if keep else buffer
                        out.write(chunk)
                        buffer = buffer[-keep:] if keep else b""
                        if chunk:
                            c.changed.set()
                try:
                    chunk = await reader.read(65536)
                except BaseException:
                    if active:
                        out.write(buffer)
                        c.changed.set()
                    raise
                if not chunk:
                    if active:
                        out.write(buffer)
                    raise ConnectionError("terminal shell closed before completion marker")
                buffer += chunk

    async def work(self, b, c, script):
        timing, codes = Timings(), None
        try:
            async with self.runner.connection_use(b.host):
                async with self.runner.scheduler.slot("general", self.runner.queue_timeout, timing):
                    if b.closed or b.shell.lost or b.shell.process.is_closing():
                        raise ConnectionError("terminal shell lost before transmission")
                    nonce = uuid.uuid4().hex
                    start = b"\x1eSTART" + nonce.encode() + b"\x1f"
                    end = b"\x1eEND" + nonce.encode() + b":"
                    # One physical line is parsed before eval starts. A foreground
                    # read/REPL consumes later user input, never our end-marker code.
                    quoted = (
                        "$'"
                        + script.replace("\\", "\\\\")
                        .replace("'", "\\'")
                        .replace("\n", "\\n")
                        .replace("\r", "\\r")
                        + "'"
                    )
                    packet = (
                        f"printf '\\036START{nonce}\\037'; printf '\\036START{nonce}\\037' >&2; "
                        + "builtin eval -- "
                        + quoted
                        + f"; __jms_terminal_rc=$?; printf '\\036END{nonce}:%s\\037' "
                        + '"$__jms_terminal_rc"; '
                        + f"printf '\\036END{nonce}:%s\\037' \"$__jms_terminal_rc\" >&2\n"
                    ).encode()
                    b.shell.started = False
                    c.state, c.sent = "running", True
                    b.shell.process.stdin.write(packet)
                    await b.shell.process.stdin.drain()
                    readers = [
                        asyncio.create_task(self.capture(b, c, s, start, end))
                        for s in ("stdout", "stderr")
                    ]
                    try:
                        codes = await asyncio.gather(*readers)
                    finally:
                        for task in readers:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*readers, return_exceptions=True)
                    if codes[0] != codes[1]:
                        raise ConnectionError("terminal completion markers disagree")
                    c.state, c.exit_code = "completed", codes[0]
        except BaseException as exc:
            c.state = "unknown" if c.sent else "not_started"
            c.error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, QueueFull):
                c.rejection = {
                    **exc.response(),
                    "failure_stage": exc.stage,
                    "queue_timed_out": False,
                }
            elif isinstance(exc, QueueTimeout):
                c.rejection = {"failure_stage": exc.stage, "queue_timed_out": True}
            if c.sent:
                b.shell.lost = True
                b.shell.process.close()
        finally:
            await offload(
                self.store.finish,
                c.result_id,
                ok=c.state == "completed" and c.exit_code == 0,
                exit_code=c.exit_code,
                executed=True if c.started else (None if c.sent else False),
                execution_state=c.state,
                completion_unknown=c.state == "unknown",
                error=c.error,
                timing=timing.snapshot(),
                **c.rejection,
            )
            c.changed.set()

    async def read(self, session, wait_ms=1000, max_bytes=4096, cursor=None):
        b = self.binding(session)
        if b.shell.process.is_closing():
            b.shell.lost = True
        c = b.command
        if not c:
            return {"ok": True, "session": session, "state": "closed" if b.closed else "idle"}
        async with b.read_lock:
            positions = {**c.cursor, **(cursor or {})}

            def available():
                return any(
                    self.store.path(c.result_id, s).stat().st_size > positions[s] for s in positions
                )

            if c.task and not c.task.done() and not await offload(available) and wait_ms:
                c.changed.clear()
                if not await offload(available) and not c.task.done():
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(c.changed.wait(), wait_ms / 1000)
            out = {"ok": True, "session": session, "result_id": c.result_id, "state": c.state}
            # A preview budget covers both streams; unread bytes stay on disk.
            budget = max_bytes
            for stream in ("stderr", "stdout"):
                size = await offload(
                    lambda s=stream: self.store.path(c.result_id, s).stat().st_size
                )
                n = min(budget, max(0, size - positions[stream]))
                if n:
                    page = await offload(self.store.read, c.result_id, stream, positions[stream], n)
                    out[stream] = page["text"]
                    positions[stream] = page["next_offset"]
                    budget = max(0, budget - len(page["text"].encode()))
                if positions[stream] < size:
                    out["more"] = True
            c.cursor = positions
            out["cursor"] = positions.copy()
            if c.state == "completed":
                out.update(ok=c.exit_code == 0, exit_code=c.exit_code)
            elif c.state in ("unknown", "not_started"):
                out.update(
                    ok=False,
                    executed=True if c.started else (None if c.sent else False),
                    completion_unknown=c.state == "unknown",
                    error=c.error,
                    **c.rejection,
                )
            if b.closed or b.shell.lost:
                out["session_state"] = "closed" if b.closed else "lost"
            return out

    async def write(
        self, session, input=None, signal=None, eof=False, allow_dangerous=False, **page
    ):
        b = self.binding(session)
        await self.permitted(b)
        if sum((input is not None, signal is not None, eof)) != 1:
            raise ValueError("provide exactly one of input, signal or eof=true")
        async with b.lock:
            c = b.command
            if b.closed or b.shell.lost or not c or c.state != "running" or c.task.done():
                raise ValueError(
                    "write needs an active foreground command; use exec for shell commands"
                )
            if input is not None:
                self.check(input, allow_dangerous)
                b.shell.process.stdin.write(input.encode())
                await b.shell.process.stdin.drain()
            elif signal is not None:
                b.shell.process.send_signal(signal)
            else:
                b.shell.process.stdin.write_eof()
        out = await self.read(session, **page)
        out["input_sent"] = True
        return out

    async def close(self, session):
        if session in self.lost:
            del self.lost[session]
            await self.save()
            return {"ok": True, "session": session, "state": "closed"}
        b = self.binding(session)
        async with b.lock:
            b.closed = True
            b.shell.process.close()
            if b.command and b.command.task and not b.command.task.done():
                b.command.task.cancel()
                await asyncio.gather(b.command.task, return_exceptions=True)
                if b.command.state == "queued":
                    b.command.state = "not_started"
                    await offload(
                        self.store.finish,
                        b.command.result_id,
                        ok=False,
                        executed=False,
                        execution_state="not_started",
                        completion_unknown=False,
                        exit_code=None,
                    )
            with contextlib.suppress(Exception):
                await asyncio.wait_for(b.shell.process.wait_closed(), 5)
            self.runner._shells.pop((self.runner.key(b.host), "term-" + session), None)
            await self.save()
        return {"ok": True, "session": session, "state": "closed"}

    async def close_all(self):
        records = list(self.lost.values()) + [
            dict(
                session=b.session,
                asset_id=b.host.asset_id,
                host=b.host.name,
                result_id=b.command.result_id if b.command else None,
            )
            for b in self.bindings.values()
            if not b.closed
        ]
        for b in list(self.bindings.values()):
            if not b.closed:
                await self.close(b.session)
        await offload(atomic_json, self.registry, records)
