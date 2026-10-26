"""Four-tool MCP facade with progressive disclosure and REST-backed targets."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import __version__
from . import server as upstream
from .config import ConfigError, load_config
from .inventory import InventoryError
from .jobs import JobManager
from .native_ssh import NativeSSHRunner
from .rest_inventory import AssetCatalog
from .results import ResultStore
from .runtime_config import StateLock, load_settings

CATEGORIES = {
    "assets": ["jms_list_assets", "jms_asset_detail", "jms_inventory_reload"],
    "inspect": ["jms_probe", "jms_fleet_summary", "jms_exec"],
    "files": ["jms_sync_push", "jms_sync_pull", "jms_pull_artifacts"],
    "jobs": [
        "jms_submit_job",
        "jms_job_status",
        "jms_job_logs",
        "jms_job_cancel",
        "jms_job_wait",
        "jms_job_list",
    ],
    "cluster": [
        "jms_cluster_preflight",
        "jms_cluster_plan",
        "jms_cluster_submit",
        "jms_cluster_cancel",
    ],
    "development": ["jms_dev_sync", "jms_dev_run", "jms_dev_logs", "jms_dev_discover"],
    "audit": ["jms_audit_query", "jms_create_ticket"],
    "sessions": ["session_list", "session_close"],
}
INSTRUCTIONS = """JumpServer access through four tools.
Use run for literal Bash or a local script_path;
container/image selects remote Docker without nested shell quoting. SSH connections belong to this
service. Each run is isolated unless session_id is explicitly supplied. The same session_id on the
same asset retains cd/export/functions; use distinct IDs for unrelated tasks. Never replay a
timed-out
command automatically. Idle connections reconnect automatically; before script transmission a
transport failure can retry once. After transmission a failure returns completion_unknown=true and
must not be replayed blindly. Lost named shells require a new ID or explicit session_close reset.
Long/background work belongs to the job operations discovered below.
Output is saved completely to a private result store; use read_result(result_id, stream,
offset/limit,
tail or pattern) for more. Default previews share a 4096-byte budget across targets.
Use discover(topic="assets", query="...") for REST-authorized target names/IDs, refreshed hourly.
Use discover(query="jobs") for matching operation names, then discover(operation="jms_submit_job")
for that operation's schema. Invoke low-frequency operations with call(operation, arguments).
An inventory snapshot is not a hardware health check; unknown accelerator platform stays generic.
"""


def unwrap(result):
    if result.structured_content is not None:
        return result.structured_content
    blocks = [c.text for c in result.content if c.type == "text"]
    if len(blocks) == 1:
        try:
            return json.loads(blocks[0])
        except ValueError:
            pass
    return {"content": blocks, "ok": not result.is_error}


class BridgeService:
    def __init__(self, cfg=None, settings=None):
        self.cfg = cfg or load_config()
        self.settings = settings or load_settings()
        self.store = ResultStore(self.settings.state_dir / "results")
        self.catalog = AssetCatalog(self.cfg, self.settings)
        self.runner = NativeSSHRunner(self.cfg, self.settings, store=self.store)
        self.manager = JobManager(self.runner)
        self.schemas = {}

        @asynccontextmanager
        async def lifespan(server):
            with StateLock(self.settings.state_dir):
                await self.catalog.start()
                try:
                    yield {}
                finally:
                    await self.catalog.close()
                    await self.runner.close()

        self.server = MCPServer(
            "jms-agent-bridge", version=__version__, instructions=INSTRUCTIONS, lifespan=lifespan
        )
        self.register()

    def compact_exec_results(self, results):
        budget = 4096 // max(1, len(results))
        return [
            (
                {
                    "host": r.host,
                    "ok": r.ok,
                    "exit_code": r.exit_code,
                    "result_id": r.result_id,
                    **self.store.preview(r.result_id, budget),
                }
                if hasattr(r, "result_id")
                else r.to_dict(limit=budget)
            )
            for r in results
        ]

    async def load_schemas(self):
        if self.schemas:
            return
        self.schemas = {
            tool.name: tool.model_dump(mode="json", by_alias=True, exclude_none=True)
            for tool in await upstream.server.list_tools()
        }
        for name, desc in {
            "jms_exec": (
                "Compatibility operation; prefer run for scripts, Docker, and "
                "complete output storage."
            ),
            "jms_list_assets": "Search the hourly REST asset snapshot; no YAML inventory is used.",
            "jms_asset_detail": (
                "Read a permitted asset and available accounts from the REST snapshot."
            ),
            "jms_inventory_reload": (
                "Refresh the REST asset snapshot now; this does not close SSH "
                "connections or shells."
            ),
        }.items():
            self.schemas[name]["description"] = desc
        for name, description, props, required in [
            ("session_list", "List explicit persistent shells owned by this MCP service.", {}, []),
            (
                "session_close",
                "Close one named shell on one permitted asset; its shell state is discarded.",
                {"target": {"type": "string"}, "session_id": {"type": "string"}},
                ["target", "session_id"],
            ),
        ]:
            self.schemas[name] = {
                "name": name,
                "description": description,
                "inputSchema": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                    "additionalProperties": False,
                },
                "annotations": {
                    "readOnlyHint": name == "session_list",
                    "destructiveHint": name == "session_close",
                },
            }

    async def run(
        self,
        targets,
        script=None,
        script_path=None,
        cwd=None,
        env=None,
        container=None,
        image=None,
        docker_args=None,
        session_id=None,
        timeout=60,
        max_output_bytes=4096,
        allow_dangerous=False,
    ):
        if (script is None) == (script_path is None):
            raise ValueError("provide exactly one of script or script_path")
        if not 1 <= timeout <= 600 or not 256 <= max_output_bytes <= 16384:
            raise ValueError("timeout must be 1..600 seconds; max_output_bytes must be 256..16384")
        if not targets:
            raise ValueError("targets must not be empty")
        if script_path:
            path = Path(script_path).expanduser()
            if not path.is_absolute() or not path.is_file():
                raise ValueError("script_path must name an existing absolute local file")
            script = path.read_text()
        if upstream._looks_dangerous(script) and not allow_dangerous:
            raise ValueError(
                "script matched the existing destructive-pattern guard; authorized "
                "operations require allow_dangerous=true"
            )
        inventory = await self.catalog.ensure()
        hosts = inventory.resolve(targets)
        if not hosts:
            raise ValueError(
                "no permitted SSH assets matched the targets"
                + (f" — {why}" if (why := self.catalog.explain_no_targets()) else "")
            )

        async def one(host):
            return await self.runner.execute_to_store(
                host,
                script,
                timeout,
                cwd=cwd,
                env=env,
                container=container,
                image=image,
                docker_args=docker_args,
                session_id=session_id,
            )

        results = await asyncio.gather(*(one(host) for host in hosts))
        budget = max_output_bytes // max(1, len(results))
        rows = [
            {
                k: r[k]
                for k in (
                    "host",
                    "result_id",
                    "ok",
                    "exit_code",
                    "executed",
                    "timed_out",
                    "elapsed_seconds",
                    "session_id",
                )
            }
            | {
                k: r[k]
                for k in (
                    "bytes",
                    "execution_state",
                    "completion_unknown",
                    "connection_state",
                    "session_state",
                    "reconnected",
                )
            }
            | self.store.preview(r["result_id"], budget)
            | ({"warning": r["warning"]} if r.get("warning") else {})
            for r in results
        ]
        return {
            "ok": all(r["ok"] for r in results),
            "results": rows,
            **({"inventory_stale": True} if self.catalog.status()["stale"] else {}),
        }

    async def discover(self, query="", operation=None, topic="operations", limit=10, offset=0):
        if not 1 <= limit <= 30 or offset < 0:
            raise ValueError("limit must be 1..30; offset must be nonnegative")
        if topic == "status":
            return {
                "inventory": self.catalog.status(),
                "ssh_connections": self.runner.connection_status(),
                "persistent_shells": self.runner.sessions(),
                "result_directory": str(self.store.root),
            }
        if topic == "assets":
            await self.catalog.ensure()
            return self.catalog.search(query, offset, limit)
        await self.load_schemas()
        if operation:
            if operation not in self.schemas:
                raise ValueError("unknown operation; use discover(query=...)")
            return self.schemas[operation]
        if not query:
            return {
                "categories": {k: len(v) for k, v in CATEGORIES.items()},
                "hint": "Pass a category/search query, then operation for one full schema.",
            }
        names = CATEGORIES.get(query.lower()) or [
            n
            for n, d in self.schemas.items()
            if query.lower() in (n + " " + d["description"]).lower()
        ]
        selected = names[offset : offset + limit]
        return {
            "operations": [
                {"name": n, "description": self.schemas[n]["description"].split("\n")[0]}
                for n in selected
            ],
            "matched": len(names),
            "next_offset": offset + len(selected) if offset + len(selected) < len(names) else None,
        }

    async def call(self, operation, arguments):
        await self.load_schemas()
        if operation not in self.schemas:
            raise ValueError("unknown operation; discover its schema first")
        # Validate the hidden operation with its real schema before any side effect.
        import jsonschema

        jsonschema.validate(arguments, self.schemas[operation]["inputSchema"])
        if operation == "jms_inventory_reload":
            value = await self.catalog.refresh()
        elif operation == "session_list":
            value = {"sessions": self.runner.sessions()}
        else:
            inventory = await self.catalog.ensure()
            if operation == "jms_exec":
                return await self.run(
                    arguments["targets"],
                    script=arguments["command"],
                    timeout=arguments.get("timeout", 60),
                    allow_dangerous=arguments.get("allow_dangerous", False),
                )
            if operation == "session_close":
                hosts = inventory.resolve([arguments["target"]])
                if len(hosts) != 1:
                    raise ValueError("session_close requires exactly one asset")
                value = await self.runner.close_session(hosts[0], arguments["session_id"])
            elif operation == "jms_list_assets":
                value = self.catalog.search(
                    arguments.get("search") or "", 0, len(self.catalog.snapshot["assets"])
                )
                value.update(ok=True, count=value["matched"])
            elif operation == "jms_asset_detail":
                value = next(
                    (
                        dict(a)
                        for a in self.catalog.snapshot["assets"]
                        if a["id"] == arguments["asset_id"]
                    ),
                    None,
                )
                if value is None:
                    raise ValueError("asset is not in the permitted REST snapshot")
                value.pop("login", None)
                value = {"ok": True, "asset": value}
            else:
                api = self.catalog.make_api()
                runtime_token = upstream._runtime.set(
                    {
                        "cfg": self.cfg,
                        "inv": inventory,
                        "ssh": self.runner,
                        "api": api,
                        "jobs": self.manager,
                        "fanout_reply": self.compact_exec_results,
                    }
                )
                executions = []
                result_token = self.runner.result_context.set(executions)
                try:
                    value = unwrap(await upstream.server.call_tool(operation, arguments))
                    if executions:
                        value = (
                            {**value, "execution_results": executions}
                            if isinstance(value, dict)
                            else {"data": value, "execution_results": executions}
                        )
                finally:
                    self.runner.result_context.reset(result_token)
                    upstream._runtime.reset(runtime_token)
                    await api.aclose()
        result_id = self.store.save_json(value, kind="call", operation=operation)
        raw_size = self.store.path(result_id, "result").stat().st_size
        result = {
            "ok": not (isinstance(value, dict) and value.get("ok") is False),
            "result_id": result_id,
        }
        if raw_size <= 4096:
            return {**result, "data": value}
        summary = {
            k: value[k]
            for k in ("ok", "count", "matched", "job_id", "finished")
            if isinstance(value, dict) and k in value
        }
        return {
            **result,
            "summary": summary,
            "total_bytes": raw_size,
            "truncated": True,
            "preview": self.store.read(result_id, "result", limit=4096)["text"],
        }

    def register(self):
        server = self.server

        @server.tool(
            name="run", annotations=ToolAnnotations(destructiveHint=True, openWorldHint=True)
        )
        async def run(
            targets: list[str],
            script: str | None = None,
            script_path: str | None = None,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            container: str | None = None,
            image: str | None = None,
            docker_args: list[str] | None = None,
            session_id: str | None = None,
            timeout: float = 60,
            max_output_bytes: int = 4096,
            allow_dangerous: bool = False,
        ) -> dict:
            """Run literal Bash or a local script file; optional remote Docker.
            Complete stdout/stderr are saved; read_result fetches more. Default runs are isolated.
            Explicit session_id retains shell state on each target. Timeout 1..600s; longer work
            uses job operations.
            """
            try:
                return await self.run(
                    targets,
                    script,
                    script_path,
                    cwd,
                    env,
                    container,
                    image,
                    docker_args,
                    session_id,
                    timeout,
                    max_output_bytes,
                    allow_dangerous,
                )
            except (ValueError, OSError, ConfigError, InventoryError) as exc:
                raise ToolError(str(exc)) from None

        @server.tool(
            name="read_result", annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False)
        )
        async def read_result(
            result_id: str,
            stream: Literal["stdout", "stderr", "result"] = "stdout",
            offset: int = 0,
            limit: int = 4096,
            tail: bool = False,
            pattern: str | None = None,
        ) -> dict:
            """Read saved output by byte offset, tail, or literal pattern.
            Limit <=16384; continue with next_offset.
            """
            try:
                return self.store.read(result_id, stream, offset, limit, tail, pattern)
            except (ValueError, OSError) as exc:
                raise ToolError(str(exc)) from None

        @server.tool(
            name="discover", annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True)
        )
        async def discover(
            query: str = "",
            operation: str | None = None,
            topic: Literal["operations", "assets", "status"] = "operations",
            limit: int = 10,
            offset: int = 0,
        ) -> dict:
            """Find assets or operations; an exact operation returns its schema.
            No query lists categories.
            """
            try:
                return await self.discover(query, operation, topic, limit, offset)
            except ValueError as exc:
                raise ToolError(str(exc)) from None

        @server.tool(
            name="call", annotations=ToolAnnotations(destructiveHint=True, openWorldHint=True)
        )
        async def call(operation: str, arguments: dict) -> dict:
            """Invoke an operation after discovering its schema.
            Read the full saved response with read_result(stream='result').
            """
            import jsonschema

            try:
                return await self.call(operation, arguments)
            except jsonschema.ValidationError as exc:
                raise ToolError(
                    "Invalid operation arguments at "
                    + ".".join(map(str, exc.path))
                    + "; discover this operation for its schema"
                ) from None
            except (ValueError, ConfigError, InventoryError) as exc:
                raise ToolError(str(exc)) from None


def main(transport="stdio", port=8766):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncssh").setLevel(logging.WARNING)
    service = BridgeService()
    if transport == "stdio":
        service.server.run("stdio")
    else:
        import uvicorn

        app = service.server.streamable_http_app(
            host="127.0.0.1", stateless_http=True, json_response=True
        )
        uvicorn.run(app, host="127.0.0.1", port=port, timeout_graceful_shutdown=5, access_log=False)
