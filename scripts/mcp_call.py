#!/usr/bin/env python3
"""Call one tool on the jms-bridge MCP server, over the protocol it really serves.

The acceptance pass has to exercise the tools a Claude Code session would call —
``jms_list_assets``, ``jms_dev_discover``, ``jms_dev_run`` — and not some private API
path that happens to live in the same process. There is no CLI subcommand for them, so
this speaks the wire protocol. MCP over stdio is newline-delimited JSON-RPC, so a client
needs no SDK import and does not drift when the SDK's client classes move.

    mcp_call.py <tool> ['{"json":"arguments"}'] [--timeout SECONDS]

The tool's result is printed as JSON, and the exit status says what happened so the
caller branches on it rather than parsing prose:

    0   the tool returned a result
    1   the tool refused (``ok: false``) or errored
    2   the call never completed (bad arguments, a dead server, a timeout)

``JMS_BRIDGE_SERVE_CMD`` overrides how the server is launched (default: this interpreter,
``-c 'from jms_bridge.server import main; main()'``). The child inherits the environment,
so ``JMS_BRIDGE_CONFIG`` and the rest of the configuration reaches it.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import tempfile
import threading
import time

try:
    # The one thing worth borrowing from the SDK: the protocol version it will accept.
    # A hard-coded date would rot; an old version risks being rejected outright.
    from mcp.types import LATEST_PROTOCOL_VERSION as _LATEST
except Exception:  # pragma: no cover - mcp is a hard dependency; this is a safety net
    _LATEST = "2025-06-18"

_PROTOCOL = os.environ.get("MCP_PROTOCOL_VERSION") or _LATEST


def _serve_argv() -> list[str]:
    override = os.environ.get("JMS_BRIDGE_SERVE_CMD")
    if override:
        return shlex.split(override)
    return [sys.executable, "-c", "from jms_bridge.server import main; main()"]


class _Client:
    """A minimal stdio MCP client: one initialize handshake, then one tool call."""

    def __init__(self, argv: list[str], *, timeout: float):
        self.timeout = timeout
        self._stderr = tempfile.TemporaryFile(mode="w+")
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            # Unbuffered both ways: a request must reach the child before we wait on it,
            # and reading a pipe must return what has arrived rather than block for a
            # buffer's worth.
            bufsize=0,
        )
        self._inbox: queue.Queue = queue.Queue()
        threading.Thread(
            target=self._pump, args=(self.proc.stdout, self._inbox), daemon=True
        ).start()
        self._next_id = 0

    def _pump(self, stream, inbox: queue.Queue) -> None:
        """Split the child's stdout into JSON messages, off the calling thread."""
        buf = b""
        while True:
            chunk = stream.read(65536)
            if not chunk:
                inbox.put(None)  # EOF
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    inbox.put(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a stray non-JSON line is not a protocol message

    def _send(self, message: dict) -> None:
        self.proc.stdin.write((json.dumps(message) + "\n").encode())
        self.proc.stdin.flush()

    def _await(self, request_id: int) -> dict:
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"no reply to request {request_id} in {self.timeout:g}s")
            try:
                message = self._inbox.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(
                    f"no reply to request {request_id} in {self.timeout:g}s"
                ) from None
            if message is None:
                raise EOFError("the server closed its output stream")
            if message.get("id") == request_id:
                return message
            # Anything else (a notification, a late reply) is not what we are waiting on.

    def request(self, method: str, params: dict) -> dict:
        self._next_id += 1
        self._send(
            {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params}
        )
        reply = self._await(self._next_id)
        if "error" in reply:
            error = reply["error"]
            raise RuntimeError(f"{method}: {error.get('message', error)}")
        return reply["result"]

    def notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def close(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - teardown must not mask the real failure
            self.proc.kill()
        finally:
            for stream in (self.proc.stdin, self.proc.stdout):
                try:
                    stream.close()
                except Exception:  # noqa: BLE001
                    pass

    def diagnostics(self) -> str:
        """Whatever the server logged to stderr — the real cause when a call dies."""
        self._stderr.seek(0)
        return self._stderr.read()


def _payload(result: dict) -> dict:
    """The tool's return value, however the server chose to serialise it."""
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    texts = [
        part.get("text", "")
        for part in (result.get("content") or [])
        if part.get("type") == "text"
    ]
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except json.JSONDecodeError:
        return {"text": joined}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mcp_call.py", description="Call one tool on the jms-bridge MCP server."
    )
    parser.add_argument("tool")
    parser.add_argument(
        "arguments", nargs="?", default="{}", help="JSON object of tool arguments"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="seconds to wait for the tool call (jms_dev_run is the slow one)",
    )
    parser.add_argument(
        "--raw", action="store_true", help="print the raw MCP result, not the payload"
    )
    args = parser.parse_args(argv)

    try:
        tool_args = json.loads(args.arguments)
    except json.JSONDecodeError as exc:
        print(f"error: arguments is not JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(tool_args, dict):
        print("error: arguments must be a JSON object", file=sys.stderr)
        return 2

    client = _Client(_serve_argv(), timeout=args.timeout)
    try:
        client.request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL,
                "capabilities": {},
                "clientInfo": {"name": "verify-onboarding", "version": "0"},
            },
        )
        client.notify("notifications/initialized", {})
        raw = client.request("tools/call", {"name": args.tool, "arguments": tool_args})
    except Exception as exc:  # noqa: BLE001 - report, never traceback
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        logged = client.diagnostics().strip()
        if logged:
            print("--- server stderr ---", file=sys.stderr)
            print(logged, file=sys.stderr)
        return 2
    finally:
        client.close()

    if args.raw:
        print(json.dumps(raw, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    payload = _payload(raw)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    if raw.get("isError"):
        return 1
    if isinstance(payload, dict) and payload.get("ok") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
