import json
import logging
import unittest
from unittest.mock import AsyncMock, patch

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent

from jms_bridge.compact import INSTRUCTIONS, BridgeService, CompactMCPServer
from jms_bridge.config import BastionConfig
from jms_bridge.runtime_config import BridgeSettings


class OutputCompatibility(unittest.IsolatedAsyncioTestCase):
    async def test_synthetic_samples_and_failure_states(self):
        samples = [
            {
                "ok": False,
                "execution_state": "unknown",
                "completion_unknown": True,
                "exit_code": 124,
                "timed_out": True,
                "result_id": "abcdef",
                "request_id": "original",
            },
            {
                "ok": False,
                "execution_state": "not_started",
                "queue_timed_out": True,
                "completion_unknown": False,
                "failure_stage": "execution_queue",
            },
            {
                "duplicate": True,
                "execution_replayed": False,
                "request_state": "interrupted",
                "submission_receipts": [{"host": "one", "job_id": "original-job"}],
            },
            {
                "ok": False,
                "results": [{"host": "one", "ok": True}, {"host": "two", "error": "denied"}],
            },
            {
                "text": ' \t中文🙂\r\n    indent\n"quote"\\slash\x00\n',
                "empty": "",
                "false": False,
                "none": None,
                "large": 2**60,
                "zero": -0.0,
                "fraction": 0.1,
                "next_offset": 16387,
                "truncated": True,
                "eof": False,
            },
        ]
        old, new = MCPServer("old"), CompactMCPServer("new")
        current = None

        async def probe() -> dict:
            return current

        old.tool(name="run")(probe)
        new.tool(name="run")(probe)
        for current in samples:
            with self.subTest(keys=list(current)):
                before = await old.call_tool("run", {})
                after = await new.call_tool("run", {})
                self.assertEqual(
                    json.loads(before.content[0].text), json.loads(after.content[0].text)
                )
                self.assertEqual(after.structured_content, before.structured_content)
                expected = before.model_dump()
                expected["content"][0]["text"] = json.dumps(
                    json.loads(before.content[0].text), ensure_ascii=False, separators=(",", ":")
                )
                self.assertEqual(after.model_dump(), expected)
                self.assertLessEqual(
                    len(after.content[0].text.encode()), len(before.content[0].text.encode())
                )

    async def test_non_json_text_and_metadata_are_not_reinterpreted(self):
        server = CompactMCPServer("test")
        values = [
            CallToolResult(
                content=[TextContent(type="text", text="  plain error\n")], is_error=True
            ),
            CallToolResult(content=[TextContent(type="text", text='  {"file":"text"}  ')]),
            CallToolResult(
                content=[TextContent(type="text", text="not JSON")], structured_content={"x": 1}
            ),
            CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text='{\n "x": 1\n}',
                        annotations={"priority": 0.5},
                        _meta={"source": "test"},
                    )
                ],
                structured_content={"different": True},
                _meta={"receipt": "keep"},
            ),
        ]
        for i, value in enumerate(values):
            with patch.object(MCPServer, "call_tool", new=AsyncMock(return_value=value)):
                result = await server.call_tool("other_text_tool" if i == 1 else "run", {})
            if i < 3:
                self.assertEqual(result.model_dump(), value.model_dump())
            else:
                expected = value.model_dump()
                expected["content"][0]["text"] = '{"x":1}'
                self.assertEqual(result.model_dump(), expected)

    async def test_unknown_tool_errors_unchanged(self):
        messages = []
        for server in (MCPServer("old"), CompactMCPServer("new")):
            with self.assertRaises(ToolError) as caught:
                await server.call_tool("missing", {})
            messages.append(str(caught.exception))
        self.assertEqual(*messages)


async def test_saved_bytes_cursors_and_native_tool_surface_are_preserved(tmp_path):
    service = BridgeService(
        BastionConfig(url="https://bastion.example.com"), BridgeSettings(state_dir=tmp_path)
    )
    raw = " \t中文🙂\r\n  line 2\n".encode()
    rid = service.store.create()
    path = service.store.path(rid, "stdout")
    path.write_bytes(raw)
    service.store.finish(rid)
    response = await service.server.call_tool(
        "read_result", {"result_id": rid, "limit": 7, "detail": "full"}
    )
    assert json.loads(response.content[0].text) == service.store.read(rid, limit=7)
    assert response.structured_content is None
    assert path.read_bytes() == raw
    assert {t.name for t in await service.server.list_tools()} == {
        "run",
        "read_result",
        "discover",
        "call",
        "fs",
        "meta",
        "terminal",
    }
    assert len(INSTRUCTIONS.encode()) < 1024
    await service.runner.close()


async def test_performance_log_contains_no_tool_argument_or_response_text(caplog):
    server = CompactMCPServer("test")

    @server.tool(name="run")
    async def probe(value: str) -> dict:
        return {"text": value}

    with caplog.at_level(logging.INFO, logger="jms_bridge.performance"):
        await server.call_tool("run", {"value": "private-script-or-file-content"})
    events = [json.loads(r.message) for r in caplog.records if r.name == "jms_bridge.performance"]
    assert len(events) == 1
    assert events[0]["tool"] == "run" and events[0]["response_bytes"] > 0
    assert "private-script-or-file-content" not in json.dumps(events)
