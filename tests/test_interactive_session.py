import asyncio
import contextlib
import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import jsonschema

from jms_bridge.compact import BridgeService, CompactMCPServer
from jms_bridge.filesystem import SCHEMA as FS_SCHEMA
from jms_bridge.filesystem import validate as validate_fs
from jms_bridge.interactive import Interactive
from jms_bridge.request_journal import RequestJournal
from jms_bridge.response_views import compact
from jms_bridge.results import ResultStore
from jms_bridge.scheduling import ExecutionSlots
from jms_bridge.schema_views import fs_action_schema, merge_fs_options, strip_titles


class LocalProcess:
    def __init__(self, process):
        self.p = process
        self.stdin, self.stdout, self.stderr = process.stdin, process.stdout, process.stderr

    def is_closing(self):
        return self.p.returncode is not None

    def send_signal(self, name):
        if not self.is_closing():
            os.killpg(self.p.pid, getattr(signal, "SIG" + name))

    def close(self):
        if not self.is_closing():
            self.send_signal("TERM")

    async def wait_closed(self):
        await self.p.wait()


class LocalRunner:
    def __init__(self):
        self.scheduler = ExecutionSlots(8, 2)
        self.queue_timeout = 1
        self._shells = {}

    def key(self, host):
        return host.asset_id

    @contextlib.asynccontextmanager
    async def connection_use(self, host):
        yield

    async def connection(self, host):
        return self

    def command(self, *args, **kwargs):
        return "local-test-bash"

    async def open_process(self, host, command):
        p = await asyncio.create_subprocess_exec(
            "bash",
            "--noprofile",
            "--norc",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        return LocalProcess(p), False


class TerminalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ResultStore(Path(self.tmp.name) / "results")
        self.requests = RequestJournal(Path(self.tmp.name) / "requests")
        self.runner = LocalRunner()
        host = SimpleNamespace(asset_id="11111111-1111-1111-1111-111111111111", name="test-node")
        self.host = host
        self.allowed = True

        class Inventory:
            def resolve(inner, targets):
                return [host] if self.allowed and targets in (["10.0.0.1"], [host.asset_id]) else []

        async def ensure():
            return Inventory()

        self.catalog = SimpleNamespace(ensure=ensure)
        self.manager = Interactive(
            self.catalog, self.runner, self.store, self.requests, lambda s: "forbidden" in s
        )

    async def asyncTearDown(self):
        await self.manager.close_all()
        self.tmp.cleanup()

    async def opened(self, **kw):
        value = await self.manager.open("10.0.0.1", **kw)
        self.assertEqual(value["state"], "idle")
        return value["session"]

    async def complete(self, session, script):
        return await self.manager.execute(session, script, wait_ms=5000)

    async def test_persistent_cwd_env_functions_and_unicode(self):
        session = await self.opened(cwd=self.tmp.name, env={"EXAMPLE": "中文 value"})
        first = await self.complete(
            session, "export PERSIST=yes\nfunction fn(){ printf 'function-ok'; }\ncd /tmp"
        )
        self.assertEqual(first["exit_code"], 0)
        out = await self.complete(session, 'printf \'%s|%s|%s|\' "$PWD" "$EXAMPLE" "$PERSIST"; fn')
        self.assertEqual(out["stdout"], "/tmp|中文 value|yes|function-ok")
        self.assertEqual(self.runner.scheduler.status()["active"]["general"], 0)

    async def test_stdin_prompt_is_published_and_end_marker_not_consumed(self):
        session = await self.opened()
        value = await self.manager.execute(
            session, "printf 'Name?'; read -r name; printf 'Hello <%s>' \"$name\"", wait_ms=100
        )
        self.assertEqual(value["state"], "running")
        self.assertEqual(value["stdout"], "Name?")
        out = await self.manager.write(session, input="world\n", wait_ms=1000)
        for _ in range(5):
            if out["state"] == "completed":
                break
            await asyncio.sleep(0.02)
            out2 = await self.manager.read(session, wait_ms=1000)
            out = {**out2, "stdout": out.get("stdout", "") + out2.get("stdout", "")}
        self.assertEqual(out["state"], "completed")
        self.assertEqual(out["stdout"], "Hello <world>")
        again = await self.complete(session, "printf second")
        self.assertEqual(again["stdout"], "second")

    async def test_multiline_quotes_backslashes_and_literals(self):
        session = await self.opened()
        value = await self.complete(
            session, "cat <<'END'\n'quoted' \\n $literal `literal`\nEND\nprintf '\\nfin'\n"
        )
        self.assertEqual(value["stdout"], "'quoted' \\n $literal `literal`\n\nfin")

    async def test_explicit_cursor_recovers_complete_output(self):
        session = await self.opened()
        value = await self.manager.execute(
            session, "printf '中文abcdef'", wait_ms=1000, max_bytes=1
        )
        self.assertEqual(value["stdout"], "中")
        self.assertTrue(value["more"])
        rest = await self.manager.read(session, wait_ms=0)
        self.assertEqual(rest["stdout"], "文abcdef")
        again = await self.manager.read(session, wait_ms=0, cursor={"stdout": 0, "stderr": 0})
        self.assertEqual(again["stdout"], "中文abcdef")
        saved = self.store.read(value["result_id"])
        self.assertEqual(saved["text"], "中文abcdef")

    async def test_two_bindings_are_isolated(self):
        one, two = await asyncio.gather(
            self.opened(env={"BOUND": "one"}), self.opened(env={"BOUND": "two"})
        )
        a, b = await asyncio.gather(
            self.complete(one, 'printf "$BOUND"'), self.complete(two, 'printf "$BOUND"')
        )
        self.assertEqual((a["stdout"], b["stdout"]), ("one", "two"))

    async def test_client_cancel_keeps_running_command(self):
        session = await self.opened()
        caller = asyncio.create_task(
            self.manager.execute(session, "sleep 0.15; printf survived", wait_ms=5000)
        )
        await asyncio.sleep(0.05)
        caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await caller
        command = self.manager.binding(session).command
        await command.task
        value = await self.manager.read(session, wait_ms=0)
        self.assertEqual(value["stdout"], "survived")
        self.assertEqual(value["state"], "completed")

    async def test_busy_cannot_replace_command(self):
        session = await self.opened()
        await self.manager.execute(session, "read -r text", wait_ms=50)
        with self.assertRaisesRegex(ValueError, "busy"):
            await self.manager.execute(session, "printf wrong")
        await self.manager.write(session, input="done\n", wait_ms=100)

    async def test_nonzero_exit_shell_remains_usable(self):
        session = await self.opened()
        out = await self.complete(session, "printf problem >&2; false")
        self.assertFalse(out["ok"])
        self.assertEqual(out["exit_code"], 1)
        self.assertEqual(out["stderr"], "problem")
        self.assertEqual((await self.complete(session, "printf usable"))["stdout"], "usable")

    async def test_exit_without_marker_is_unknown_not_replayed(self):
        session = await self.opened()
        out = await self.complete(session, "exit 0")
        self.assertEqual(out["state"], "unknown")
        self.assertTrue(out["completion_unknown"])
        with self.assertRaisesRegex(ValueError, "closed/lost"):
            await self.complete(session, "printf never")

    async def test_guard_still_applies_to_exec_and_stdin(self):
        session = await self.opened()
        with self.assertRaisesRegex(ValueError, "JMS_SCRIPT_PATTERN_GUARD"):
            await self.complete(session, "printf forbidden")
        value = await self.manager.execute(
            session, "printf forbidden", allow_dangerous=True, wait_ms=1000
        )
        self.assertEqual(value["stdout"], "forbidden")
        await self.manager.execute(session, "read -r line", wait_ms=30)
        with self.assertRaisesRegex(ValueError, "JMS_SCRIPT_PATTERN_GUARD"):
            await self.manager.write(session, input="forbidden\n")
        await self.manager.write(session, input="safe\n", wait_ms=100)

    async def test_revoked_binding_cannot_execute(self):
        session = await self.opened()
        self.allowed = False
        with self.assertRaisesRegex(ValueError, "no longer permitted"):
            await self.complete(session, "printf no")

    async def test_setup_missing_directory_fails(self):
        with self.assertRaisesRegex(ValueError, "setup failed"):
            await self.opened(cwd=self.tmp.name + "/does-not-exist")
        self.assertFalse(any(not b.closed for b in self.manager.bindings.values()))

    async def test_restart_reports_lost_and_keeps_result_link(self):
        session = await self.opened()
        value = await self.complete(session, "printf evidence")
        await self.manager.close_all()
        after = Interactive(self.catalog, self.runner, self.store, self.requests, lambda s: False)
        await after.start()
        with self.assertRaisesRegex(ValueError, value["result_id"]):
            after.binding(session)
        self.assertEqual(self.store.read(value["result_id"])["text"], "evidence")
        await after.close(session)

    async def test_request_id_prevents_duplicate_execution_despite_view_changes(self):
        session = await self.opened()
        service = BridgeService.__new__(BridgeService)
        service.schemas, service.validators = {}, {}
        service.terminals, service.requests = self.manager, self.requests
        args = {
            "session": session,
            "script": 'COUNTER=$((${COUNTER:-0}+1)); printf "$COUNTER"',
            "request_id": "same-command",
            "wait_ms": 1000,
        }
        first = await service.terminal("exec", args)
        repeated = await service.terminal("exec", {**args, "wait_ms": 0, "max_bytes": 10})
        self.assertEqual(first["stdout"], "1")
        self.assertEqual(first["result_id"], repeated["result_id"])
        self.assertTrue(repeated["duplicate"])
        self.assertFalse(repeated["execution_replayed"])
        last = await self.complete(session, 'printf "$COUNTER"')
        self.assertEqual(last["stdout"], "1")

    async def test_open_and_running_receipts_are_successful_control_requests(self):
        service = BridgeService.__new__(BridgeService)
        service.schemas, service.validators = {}, {}
        service.terminals, service.requests = self.manager, self.requests
        opened = await service.terminal(
            "open", {"target": "10.0.0.1", "request_id": "open-receipt"}
        )
        self.assertTrue(opened["ok"])
        value = await service.terminal(
            "exec",
            {
                "session": opened["session"],
                "script": "read -r text",
                "wait_ms": 10,
                "request_id": "running-receipt",
            },
        )
        self.assertTrue(value["ok"])
        self.assertEqual(value["state"], "running")

    async def test_foreground_signal_has_explicit_outcome(self):
        session = await self.opened()
        await self.manager.execute(session, "sleep 30", wait_ms=20)
        value = await self.manager.write(session, signal="INT", wait_ms=50)
        command = self.manager.binding(session).command
        await command.task
        final = await self.manager.read(session, wait_ms=0)
        self.assertTrue(value["input_sent"])
        self.assertIn(final["state"], ("completed", "unknown"))
        if final["state"] == "unknown":
            self.assertTrue(final["completion_unknown"])


class SchemaTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.s = BridgeService.__new__(BridgeService)
        self.s.server = CompactMCPServer("offline-schema-test")
        self.s.schemas, self.s.validators = {}, {}
        self.s.register()

    async def test_advertised_fs_eight_parameters_and_no_titles(self):
        tools = {
            t.name: t.model_dump(mode="json", by_alias=True)
            for t in await self.s.server.list_tools()
        }
        self.assertEqual(len(tools), 7)
        self.assertEqual(len(tools["fs"]["inputSchema"]["properties"]), 8)
        for tool in tools.values():
            self.assertNotIn('"title":', json.dumps(tool["inputSchema"]))
        self.assertEqual(len(tools["terminal"]["inputSchema"]["properties"]), 5)

    async def test_flat_fs_compatibility_and_options_use_same_handler(self):
        seen = []

        async def fs(arguments):
            seen.append(arguments)
            return {"ok": True}

        self.s.fs = fs
        for extra in ({"max_bytes": 16}, {"options": {"max_bytes": 16}}):
            result = await self.s.server.call_tool(
                "fs", dict(target="10.0.0.1", action="read", path="/etc/hostname", **extra)
            )
            self.assertFalse(result.is_error, result.content)
        self.assertEqual(seen[0], seen[1])

    async def test_action_specific_discovery_matches_validation(self):
        await self.s.load_schemas()
        schema = await self.s.discover(query="read", operation="jms_fs")
        self.assertNotIn("overwrite", schema["inputSchema"]["properties"])
        for action in ("read", "search", "upload", "read_many"):
            s = fs_action_schema(FS_SCHEMA, action)["inputSchema"]
            jsonschema.Draft202012Validator.check_schema(s)
            self.assertEqual(s["properties"]["action"]["const"], action)
        self.assertNotIn(
            "path", fs_action_schema(FS_SCHEMA, "read_many")["inputSchema"]["properties"]
        )

    async def test_terminal_hidden_operations_and_validation(self):
        await self.s.load_schemas()
        self.assertIn("terminal_open", self.s.schemas)
        with self.assertRaises(jsonschema.ValidationError):
            await self.s.call("terminal_open", {"target": "10.0.0.1", "unknown": True})

    async def test_fs_identity_override_and_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            merge_fs_options({"target": "one"}, {"target": "two"})
        with self.assertRaises(ValueError):
            merge_fs_options({"max_bytes": 16}, {"max_bytes": 32})
        args = merge_fs_options(
            {"target": "one", "action": "read", "path": "/x"}, {"overwrite": True}
        )
        with self.assertRaises(ValueError):
            validate_fs(args)

    async def test_strip_schema_titles_preserves_user_property_named_title(self):
        value = {
            "title": "Generated",
            "type": "object",
            "properties": {"title": {"title": "Title", "type": "string"}},
        }
        self.assertEqual(strip_titles(value)["properties"], {"title": {"type": "string"}})

    async def test_terminal_view_hides_receipt_noise_but_preserves_unknown_state(self):
        value = {
            "session": "abc",
            "result_id": "test",
            "state": "completed",
            "exit_code": 0,
            "stdout": "answer",
            "cursor": {"stdout": 6, "stderr": 0},
            "ok": True,
            "request_id": "req",
            "request_state": "complete",
            "accepted_at": 1,
            "finished_at": 2,
            "operation": "terminal_exec",
            "duplicate": False,
            "execution_replayed": False,
        }
        view = compact("terminal", value, {})
        self.assertNotIn("accepted_at", view)
        self.assertEqual(view["cursor"], value["cursor"])
        unknown = {
            **value,
            "state": "unknown",
            "completion_unknown": True,
            "execution_results": [{"host": "node", "result_id": "test"}],
        }
        self.assertTrue(compact("terminal", unknown, {})["completion_unknown"])
        self.assertIn("execution_results", compact("terminal", unknown, {}))


if __name__ == "__main__":
    unittest.main()
