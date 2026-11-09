"""Compact MCP facade with progressive disclosure and REST-backed targets."""

import asyncio
import contextvars
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import jsonschema
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, ToolAnnotations
from pydantic import Field

from . import __version__
from . import server as upstream
from .accelerator_inspection import OPERATION as INSPECT_OPERATION
from .accelerator_inspection import SCHEMA as INSPECT_SCHEMA
from .accelerator_inspection import AcceleratorInspector
from .async_io import io_metrics, offload
from .comment_guard import check as check_script_guard
from .config import ConfigError, load_config
from .filesystem import OPERATION as FS_OPERATION
from .filesystem import SCHEMA as FS_SCHEMA
from .filesystem import Filesystem
from .filesystem import validate as validate_fs
from .interactive import Interactive
from .interactive import schemas as terminal_schemas
from .inventory import InventoryError
from .native_ssh import NativeSSHRunner
from .recoverable_jobs import RecoverableJobManager
from .request_journal import RequestJournal
from .response_views import compact as compact_response
from .rest_inventory import AssetCatalog
from .results import ResultStore
from .runtime_config import StateLock, load_settings
from .scheduling import QueueFull, QueueTimeout, queue_rejection
from .schema_views import advertised, fs_action_schema, merge_fs_options

CATEGORIES = {
    "assets": ["jms_list_assets", "jms_asset_detail", "jms_inventory_reload"],
    "inspect": [INSPECT_OPERATION, "jms_probe", "jms_fleet_summary", "jms_exec"],
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
    "requests": ["request_status"],
    "fs": [FS_OPERATION],
    "guard": ["script_guard_check"],
    "metadata": ["result_metadata"],
    "terminal": [
        "terminal_open",
        "terminal_exec",
        "terminal_read",
        "terminal_write",
        "terminal_close",
    ],
}
DANGEROUS_FLAG_DESCRIPTION = (
    "Acknowledge the local guard for this authorized command; "
    "does not grant authorization or override client or server policy."
)
INSTRUCTIONS = """JumpServer: run, fs, terminal, read_result, discover, call, meta.
Use REST-authorized targets. Compact responses omit routine metadata; detail=full restores it.
After a lost reply, query request_status with the original request_id; never blindly replay
uncertain work. allow_dangerous acknowledges only the local guard for an already-authorized
action; it grants no approval and cannot override client/JumpServer denial.
Saved outputs expire by TTL (default 7 days, hourly cleanup); reads do not extend it.
Active results are protected; deduplication receipts are retained separately."""


class CompactMCPServer(MCPServer):
    """Project compact views while retaining canonical saved responses."""

    async def list_tools(self, *args, **kwargs):
        tools = await super().list_tools(*args, **kwargs)
        return [
            t.model_copy(update={"input_schema": advertised(t.name, t.input_schema)}) for t in tools
        ]

    async def call_tool(self, name, arguments, context=None):
        started = time.monotonic()
        metrics = {}
        token = io_metrics.set(metrics)
        error, result = None, None
        try:
            result = await self.compact_call(name, arguments, context)
            return result
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            io_metrics.reset(token)
            logging.getLogger("jms_bridge.performance").info(
                "%s",
                json.dumps(
                    dict(
                        event="tool_finished",
                        at=time.time(),
                        tool=name,
                        operation=str(
                            (arguments or {}).get("operation")
                            or (arguments or {}).get("action")
                            or ""
                        )[:80],
                        tool_handler_seconds=round(time.monotonic() - started, 6),
                        response_bytes=sum(
                            len(c.text.encode()) for c in result.content if c.type == "text"
                        )
                        if isinstance(result, CallToolResult)
                        else 0,
                        error_type=error,
                        **{k: round(v, 6) for k, v in metrics.items()},
                    ),
                    separators=(",", ":"),
                ),
            )

    async def compact_call(self, name, arguments, context=None):
        result = await super().call_tool(name, arguments, context)
        if (
            name in {"run", "fs", "terminal", "meta", "read_result", "discover", "call"}
            and isinstance(result, CallToolResult)
            and not result.is_error
            and len(result.content) == 1
            and result.content[0].type == "text"
        ):
            block = result.content[0]
            serialization_started = time.monotonic()
            try:
                # These five tools return dicts. This SDK also emits JSON text when
                # structured_content is absent. Preserve either structured view as-is.
                value = original = json.loads(block.text)
                if (arguments or {}).get("detail", "compact") == "compact" and hasattr(
                    self, "response_view"
                ):
                    value = self.response_view(name, value, arguments or {})
                text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            except (ValueError, TypeError):
                return result
            finally:
                metrics = io_metrics.get()
                if metrics is not None:
                    metrics["compaction_seconds"] = time.monotonic() - serialization_started
            update = {"content": [block.model_copy(update={"text": text})]}
            if result.structured_content == original and value != original:
                update["structured_content"] = value
            result = result.model_copy(update=update)
        return result


def validate_cached(validator, arguments):
    # Keep jsonschema.validate's best-match error choice, without rebuilding the
    # validator and checking its trusted static schema for every request.
    error = jsonschema.exceptions.best_match(validator.iter_errors(arguments))
    if error is not None:
        raise error


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


def script_is_dangerous(script):
    return check_script_guard(script, upstream._DANGEROUS_PATTERNS)["blocked"]


class BridgeService:
    def __init__(self, cfg=None, settings=None):
        self.cfg = cfg or load_config()
        self.settings = settings or load_settings()
        self.store = ResultStore(
            self.settings.state_dir / "results",
            ttl_seconds=self.settings.result_ttl_seconds,
            cleanup_interval_seconds=self.settings.result_cleanup_interval_seconds,
        )
        self.requests = RequestJournal(self.settings.state_dir / "requests")
        self.submissions_context = contextvars.ContextVar("submission_receipts", default=None)
        self.catalog = AssetCatalog(self.cfg, self.settings)
        self.runner = NativeSSHRunner(self.cfg, self.settings, store=self.store)
        self.filesystem = Filesystem(self.catalog, self.runner, self.store)
        self.inspector = AcceleratorInspector(self.catalog, self.runner, self.store)
        self.manager = RecoverableJobManager(self.runner, self.record_submission)
        self.terminals = Interactive(
            self.catalog, self.runner, self.store, self.requests, script_is_dangerous
        )
        self.schemas = {}
        self.validators = {}
        self.fs_validator = jsonschema.validators.validator_for(FS_SCHEMA["inputSchema"])(
            FS_SCHEMA["inputSchema"]
        )

        @asynccontextmanager
        async def lifespan(server):
            with StateLock(self.settings.state_dir):
                await self.catalog.start()
                await self.store.start_retention()
                await self.terminals.start()
                try:
                    yield {}
                finally:
                    await self.requests.drain()
                    await self.terminals.close_all()
                    await self.filesystem.workers.close()
                    await self.store.close_retention()
                    await self.catalog.close()
                    await self.runner.close()

        self.server = CompactMCPServer(
            "jms-agent-bridge", version=__version__, instructions=INSTRUCTIONS, lifespan=lifespan
        )
        self.server.response_view = self.response_view
        self.register()

    def response_view(self, tool, value, arguments):
        # IPs are used only when the REST inventory resolves them uniquely.
        assets = self.catalog.snapshot.get("assets", [])
        counts = {}
        for asset in assets:
            address = asset.get("address")
            counts[address] = counts.get(address, 0) + 1
        addresses = {
            a["name"]: a["address"]
            for a in assets
            if a.get("address") and counts[a["address"]] == 1
        }
        return compact_response(tool, value, arguments, addresses)

    async def meta(self, ref="service", section="summary"):
        if section not in ("summary", "all", "timing", "transport", "request", "policy"):
            raise ValueError("unknown metadata section")
        if ref == "service":
            if section == "summary":
                value = {
                    "version": __version__,
                    "response_detail": "compact",
                    "active_requests": len(self.requests.tasks),
                }
                if self.catalog.status()["stale"]:
                    value["inventory_stale"] = True
                return value
            value = await self.discover(topic="status")
            if section == "transport":
                return {k: value[k] for k in ("ssh_connections", "persistent_shells", "execution")}
            if section == "policy":
                return {
                    k: value[k] for k in ("result_retention", "inventory", "execution", "login")
                }
            return value
        if ref.startswith("request:"):
            value = await self.request_status(ref[len("request:") :])
            # Do not re-emit command output or arbitrary operation data here.
            keys = (
                "request_id",
                "operation",
                "request_state",
                "completion_unknown",
                "accepted_at",
                "finished_at",
                "duplicate",
                "execution_replayed",
                "execution_results",
                "submission_receipts",
                "error",
            )
            return {k: value[k] for k in keys if k in value}
        if ref.startswith("result:"):
            ref = ref[len("result:") :]
        value = await offload(self.store.metadata, ref)
        if section == "summary":
            keys = (
                "result_id",
                "kind",
                "operation",
                "host",
                "status",
                "ok",
                "exit_code",
                "execution_state",
                "completion_unknown",
                "producer",
                "expires_at",
            )
            return {k: value[k] for k in keys if k in value}
        if section == "timing":
            keys = (
                "result_id",
                "host",
                "created_at",
                "finished_at",
                "elapsed_seconds",
                "timing",
                "connection_timing",
            )
        elif section == "transport":
            keys = (
                "result_id",
                "host",
                "asset_id",
                "container",
                "image",
                "lane",
                "session_id",
                "connection_state",
                "session_state",
                "reconnected",
                "executed",
                "execution_state",
                "completion_unknown",
                "timed_out",
                "queue_timed_out",
                "failure_stage",
                "reason",
                "retryable",
                "queue",
                "startup_retry_result_id",
                "connection_recovery",
                "warning",
            )
        else:
            keys = tuple(value)
        result = {k: value[k] for k in keys if k in value}
        if value.get("execution_result_id"):
            parent = await offload(self.store.metadata, value["execution_result_id"])
            result["execution_result_id"] = value["execution_result_id"]
            if section in ("timing", "transport"):
                result["execution"] = {k: parent[k] for k in keys if k in parent}
        if value.get("kind") in ("call", "filesystem") and section in ("all", "request", "policy"):
            # Canonical operation results are already saved. Only expose links
            # and metadata here, not log contents or arbitrary result bodies.
            raw = await offload(self.store.path(ref, "result").read_text)
            data = json.loads(raw)
            if isinstance(data, dict):
                for key in ("execution_results", "submission_receipts", "policy"):
                    if key in data:
                        result[key] = data[key]
                if value.get("kind") == "filesystem":
                    result["file"] = {
                        k: data[k]
                        for k in (
                            "path",
                            "encoding",
                            "decoding_errors",
                            "fingerprint",
                            "start_line",
                            "next_line",
                            "byte_offset",
                            "next_byte_offset",
                            "total_bytes",
                            "eof",
                            "truncated",
                        )
                        if k in data
                    }
        return result

    async def record_submission(self, receipt):
        rows = self.submissions_context.get()
        if rows is not None:
            for i, row in enumerate(rows):
                if row["host"] == receipt["host"] and row["job_id"] == receipt["job_id"]:
                    rows[i] = dict(receipt)
                    break
            else:
                rows.append(dict(receipt))
        await self.requests.aprogress("submission_receipts", receipt)

    async def request_status(self, request_id):
        receipt = await self.requests.astatus(request_id)
        # The request response may have been interrupted after transport already
        # saved a terminal result. Expose that evidence without executing anything
        # or silently upgrading the whole request to successful/completed.
        for row in receipt.get("execution_results", []):
            try:
                meta = await offload(self.store.metadata, row["result_id"])
                row["result_status"] = {
                    key: meta[key]
                    for key in (
                        "status",
                        "ok",
                        "exit_code",
                        "executed",
                        "execution_state",
                        "completion_unknown",
                        "timed_out",
                        "queue_timed_out",
                        "reason",
                        "retryable",
                        "queue",
                        "failure_stage",
                        "startup_retry_result_id",
                        "connection_recovery",
                        "finished_at",
                    )
                    if key in meta
                }
            except (ValueError, OSError):
                row["result_status"] = {"status": "unavailable"}
        return receipt

    def compact_exec_results(self, results):
        budget = 4096 // max(1, len(results))
        return [
            (
                {
                    "host": r.host,
                    "ok": r.ok,
                    "exit_code": r.exit_code,
                    "result_id": r.result_id,
                    **getattr(r, "rejection", {}),
                    **getattr(r, "recovery", {}),
                    **(
                        r.output_preview.compact(budget)
                        if getattr(r, "output_preview", None) is not None
                        else self.store.preview(r.result_id, budget)
                    ),
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
        self.schemas[INSPECT_OPERATION] = INSPECT_SCHEMA
        self.schemas[FS_OPERATION] = FS_SCHEMA
        self.schemas.update(terminal_schemas())
        for schema in self.schemas.values():
            flag = schema.get("inputSchema", {}).get("properties", {}).get("allow_dangerous")
            if flag is not None:
                flag["description"] = DANGEROUS_FLAG_DESCRIPTION
        for name, desc in {
            "jms_exec": (
                "Run a short command with the legacy environment: sources the asset's "
                "env_setup and enters its workdir, as jms_exec does in the default mode. "
                "Prefer run for scripts, Docker, or a clean environment, and for "
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
            (
                "result_metadata",
                "Read saved metadata without running a command.",
                {
                    "ref": {"type": "string"},
                    "section": {
                        "type": "string",
                        "enum": ["summary", "all", "timing", "transport", "request", "policy"],
                    },
                },
                ["ref"],
            ),
            (
                "request_status",
                "Read saved request/job/result receipts; never executes work. "
                "Reconcile the original ID after reply loss; reusing it never replays.",
                {"request_id": {"type": "string"}},
                ["request_id"],
            ),
            (
                "script_guard_check",
                "Read-only diagnosis of the exact script guard; never executes supplied text.",
                {"script": {"type": "string", "maxLength": 1048576}},
                ["script"],
            ),
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
                    "readOnlyHint": name
                    in ("session_list", "request_status", "script_guard_check", "result_metadata"),
                    "destructiveHint": name == "session_close",
                },
            }

        self.validators = {
            name: jsonschema.validators.validator_for(schema["inputSchema"])(schema["inputSchema"])
            for name, schema in self.schemas.items()
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
        request_id=None,
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
            script = await offload(path.read_text)
        if script_is_dangerous(script) and not allow_dangerous:
            raise ValueError(
                "script matched the existing destructive-pattern guard; authorized "
                "operations require allow_dangerous=true"
            )
        if request_id is not None:
            arguments = dict(
                targets=targets,
                script=script,
                cwd=cwd,
                env=env,
                container=container,
                image=image,
                docker_args=docker_args,
                session_id=session_id,
                timeout=timeout,
                max_output_bytes=max_output_bytes,
                allow_dangerous=allow_dangerous,
            )

            async def perform():
                token = self.runner.result_context.set(self.requests.receipts("execution_results"))
                try:
                    return await self.run(**arguments)
                finally:
                    self.runner.result_context.reset(token)

            return await self.requests.execute(request_id, "run", arguments, perform)
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
        previews = await asyncio.gather(
            *(offload(self.store.preview, r["result_id"], budget) for r in results)
        )
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
                    "timing",
                    "queue_timed_out",
                    "failure_stage",
                )
            }
            | preview
            | queue_rejection(r)
            | ({"warning": r["warning"]} if r.get("warning") else {})
            | {k: r[k] for k in ("startup_retry_result_id", "connection_recovery") if k in r}
            for r, preview in zip(results, previews, strict=True)
        ]
        return {
            "ok": all(r["ok"] for r in results),
            "results": rows,
            **({"inventory_stale": True} if self.catalog.status()["stale"] else {}),
        }

    async def legacy_exec(self, targets, command, *, timeout=60.0, allow_dangerous=False):
        """`jms_exec` with the contract the default mode gives it.

        That mode runs the command through the asset's ``env_setup`` and ``workdir``,
        so a probe like ``nvidia-smi`` finds a toolchain that a bare ``bash -s`` would
        not. ``run`` is deliberately clean instead, which is why this does not delegate
        to it. The safety guard returns a refusal, matching the legacy reply rather than
        raising, so a client migrating between modes sees the same shape.
        """
        if script_is_dangerous(command) and not allow_dangerous:
            return {
                "ok": False,
                "error": (
                    "command matched the safety denylist; pass allow_dangerous=true to override"
                ),
                "command": command,
            }
        inventory = await self.catalog.ensure()
        hosts = inventory.resolve(targets)
        if not hosts:
            raise ValueError(
                "no permitted SSH assets matched the targets"
                + (f" — {why}" if (why := self.catalog.explain_no_targets()) else "")
            )
        results = await asyncio.gather(
            *(
                self.runner.exec_on(host, command, timeout=timeout, preview_only=True)
                for host in hosts
            )
        )
        rows = await offload(self.compact_exec_results, results)
        return {
            "ok": all(row.get("ok") for row in rows),
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
                "result_retention": self.store.retention_status(),
                "filesystem_workers": self.filesystem.workers.status(),
                "execution": {
                    **self.runner.scheduler.status(),
                    "queue_timeout_seconds": self.runner.queue_timeout,
                },
                "login": {
                    "serialized": True,
                    "minimum_interval_seconds": 5,
                    "in_progress": self.runner._connect_lock.locked(),
                },
                "requests": {"active": len(self.requests.tasks)},
            }
        if topic == "assets":
            await self.catalog.ensure()
            return self.catalog.search(query, offset, limit)
        await self.load_schemas()
        if operation:
            if operation not in self.schemas:
                raise ValueError("unknown operation; use discover(query=...)")
            if operation == FS_OPERATION and query:
                return fs_action_schema(self.schemas[operation], query)
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

    async def call(self, operation, arguments, request_id=None):
        await self.load_schemas()
        if operation not in self.schemas:
            raise ValueError("unknown operation; discover its schema first")
        # Validate the hidden operation with its real schema before any side effect.

        validate_cached(self.validators[operation], arguments)
        if request_id is not None:
            if operation != "jms_submit_job":
                raise ValueError("call request_id currently supports only jms_submit_job")
            return await self.requests.execute(
                request_id, operation, arguments, lambda: self.call(operation, arguments)
            )
        if operation == "result_metadata":
            return await self.meta(**arguments)
        if operation.startswith("terminal_"):
            return await self.terminal(operation[len("terminal_") :], arguments)
        if operation == FS_OPERATION:
            return await self.fs(arguments, _validated=True)
        if operation == INSPECT_OPERATION:
            return await self.inspector.inspect(**arguments)
        if operation == "request_status":
            return await self.request_status(arguments["request_id"])
        if operation == "script_guard_check":
            return check_script_guard(arguments["script"], upstream._DANGEROUS_PATTERNS)
        if operation == "jms_inventory_reload":
            value = await self.catalog.refresh()
        elif operation == "session_list":
            value = {"sessions": self.runner.sessions()}
        else:
            inventory = await self.catalog.ensure()
            if operation == "jms_exec":
                # Keep the legacy contract: `jms_exec` sources the asset's `env_setup`
                # and enters its workdir, which `run` deliberately does not. Mapping
                # this name onto `run` made the same operation mean two different
                # things depending on which mode was serving it.
                value = await self.legacy_exec(
                    arguments["targets"],
                    arguments["command"],
                    timeout=float(arguments.get("timeout", 60)),
                    allow_dangerous=arguments.get("allow_dangerous", False),
                )
            elif operation == "session_close":
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
                executions = self.requests.receipts("execution_results")
                submissions = []
                submission_token = self.submissions_context.set(submissions)
                result_token = self.runner.result_context.set(executions)
                try:
                    value = unwrap(await upstream.server.call_tool(operation, arguments))
                    if executions:
                        for execution in executions:
                            try:
                                meta = await offload(self.store.metadata, execution["result_id"])
                            except (ValueError, OSError):
                                continue
                            execution.update(queue_rejection(meta))
                        value = (
                            {**value, "execution_results": executions}
                            if isinstance(value, dict)
                            else {"data": value, "execution_results": executions}
                        )
                    if submissions:
                        value = {**value, "submission_receipts": submissions}
                finally:
                    self.submissions_context.reset(submission_token)
                    self.runner.result_context.reset(result_token)
                    upstream._runtime.reset(runtime_token)
                    await api.aclose()
        result_id = await offload(self.store.save_json, value, kind="call", operation=operation)
        raw_size = (await offload(self.store.path(result_id, "result").stat)).st_size
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
            "preview": (await offload(self.store.read, result_id, "result", limit=4096))["text"],
        }

    async def fs(self, arguments, *, _validated=False):
        if not _validated:
            validate_cached(self.fs_validator, arguments)
        validate_fs(arguments)
        args = dict(arguments)
        request_id = args.pop("request_id", None)
        if request_id is not None:

            async def perform():
                token = self.runner.result_context.set(self.requests.receipts("execution_results"))
                try:
                    return await self.filesystem.execute(**args)
                finally:
                    self.runner.result_context.reset(token)

            return await self.requests.execute(request_id, "fs", args, perform)
        return await self.filesystem.execute(**args)

    async def terminal(self, action, arguments):
        await self.load_schemas()
        validate_cached(self.validators["terminal_" + action], arguments)
        args = dict(arguments)
        request_id = args.pop("request_id", None)

        async def perform():
            method = self.terminals.execute if action == "exec" else getattr(self.terminals, action)
            try:
                return await method(**args)
            except (QueueFull, QueueTimeout) as exc:
                return {
                    "ok": False,
                    "execution_state": "not_started",
                    "executed": False,
                    "completion_unknown": False,
                    "failure_stage": exc.stage,
                    "queue_timed_out": isinstance(exc, QueueTimeout),
                    "error": str(exc),
                    **(exc.response() if isinstance(exc, QueueFull) else {}),
                }

        if request_id is not None:
            # Preview/wait settings do not affect execution identity.
            identity = {
                k: v for k, v in args.items() if k not in ("wait_ms", "max_bytes", "cursor")
            }
            return await self.requests.execute(request_id, "terminal_" + action, identity, perform)
        return await perform()

    def register(self):
        server = self.server

        @server.tool(
            name="meta", annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False)
        )
        async def meta(
            ref: str = "service",
            section: Literal[
                "summary", "all", "timing", "transport", "request", "policy"
            ] = "summary",
        ) -> dict:
            """Read service, result or request:<ID> metadata; use read_result for output."""
            try:
                return await self.meta(ref, section)
            except (ValueError, OSError) as exc:
                raise ToolError(str(exc)) from None

        @server.tool(
            name="fs",
            annotations=ToolAnnotations(
                readOnlyHint=False, destructiveHint=True, openWorldHint=True
            ),
        )
        async def fs(
            target: str,
            action: Literal[
                "list",
                "stat",
                "read",
                "read_many",
                "glob",
                "search",
                "tree",
                "checksum",
                "upload",
                "download",
            ],
            path: str | None = None,
            paths: list[str] | None = None,
            container: str | None = None,
            pattern: str | None = None,
            local_path: str | None = None,
            overwrite: bool | None = None,
            request_id: str | None = None,
            start_line: int | None = None,
            line_count: int | None = None,
            byte_offset: int | None = None,
            tail: bool | None = None,
            offset: int | None = None,
            limit: int | None = None,
            max_bytes: int | None = None,
            depth: int | None = None,
            literal: bool | None = None,
            case_sensitive: bool | None = None,
            include: list[str] | None = None,
            exclude: list[str] | None = None,
            hidden: bool | None = None,
            respect_ignore: bool | None = None,
            context_lines: int | None = None,
            max_file_bytes: int | None = None,
            timeout: int | None = None,
            detail: Literal["compact", "full"] = "compact",
            options: dict | None = None,
        ) -> dict:
            """Explore files or transfer them on one exact REST asset; optional existing container.
            Absolute paths/patterns are data, not shell.
            read/list/stat/read_many/glob/search/tree/checksum are read-only; upload/download
            write destinations. local_path is on this MCP machine; path is remote.
            overwrite=false skips existing files (directory metadata may sync); extras are never
            deleted. Set request_id for transfers; recover via request_status. Bounded pages:
            read uses byte_offset; list/glob/search/tree use offset/limit. Search defaults
            literal, case-sensitive. result_id saves this page, not unseen pages.
            Put advanced fields in options; discover(operation='jms_fs', query=action) lists them.
            """
            import jsonschema

            arguments = dict(
                target=target,
                action=action,
                path=path,
                paths=paths,
                container=container,
                pattern=pattern,
                local_path=local_path,
                overwrite=overwrite,
                request_id=request_id,
                start_line=start_line,
                line_count=line_count,
                byte_offset=byte_offset,
                tail=tail,
                offset=offset,
                limit=limit,
                max_bytes=max_bytes,
                depth=depth,
                literal=literal,
                case_sensitive=case_sensitive,
                include=include,
                exclude=exclude,
                hidden=hidden,
                respect_ignore=respect_ignore,
                context_lines=context_lines,
                max_file_bytes=max_file_bytes,
                timeout=timeout,
            )
            try:
                return await self.fs(
                    merge_fs_options({k: v for k, v in arguments.items() if v is not None}, options)
                )
            except jsonschema.ValidationError as exc:
                raise ToolError(
                    "Invalid fs arguments at "
                    + ".".join(map(str, exc.path))
                    + "; discover jms_fs for details"
                ) from None
            except (ValueError, OSError, ConfigError, InventoryError) as exc:
                raise ToolError(str(exc)) from None

        @server.tool(
            name="terminal", annotations=ToolAnnotations(destructiveHint=True, openWorldHint=True)
        )
        async def terminal(
            action: Literal["open", "exec", "read", "write", "close"] = "exec",
            session: str | None = None,
            script: str | None = None,
            input: str | None = None,
            options: dict | None = None,
        ) -> dict:
            """Open a bound shell using options.target; use its handle for exec/read/write/close.
            Running is not completion. Discover terminal_<action> for advanced options.
            """
            args = dict(options or {})
            for key, value in (("session", session), ("script", script), ("input", input)):
                if value is not None:
                    if key in args:
                        raise ToolError("duplicate terminal argument: " + key)
                    args[key] = value
            try:
                return await self.terminal(action, args)
            except (
                ValueError,
                OSError,
                ConfigError,
                InventoryError,
                jsonschema.ValidationError,
            ) as exc:
                raise ToolError(str(exc)) from None

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
            allow_dangerous: Annotated[bool, Field(description=DANGEROUS_FLAG_DESCRIPTION)] = False,
            request_id: str | None = None,
            detail: Literal["compact", "full"] = "compact",
        ) -> dict:
            """Run Bash on REST targets; session_id retains shell state. Use request_id for
            writes, then reconcile lost replies. queue_full is an unsent per-target refusal.
            Full output is saved; read_result fetches it. allow_dangerous is not authorization.
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
                    request_id,
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
            detail: Literal["compact", "full"] = "compact",
        ) -> dict:
            """Read saved output by byte offset (limit <=16384), tail, or literal pattern; continue
            with next_offset.
            """
            try:
                return await offload(
                    self.store.read, result_id, stream, offset, limit, tail, pattern
                )
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
            detail: Literal["compact", "full"] = "compact",
        ) -> dict:
            """Search assets or operation names; operation returns one schema. Inventory does
            not show accelerator occupancy; use jms_accelerator_status for live checks.
            """
            try:
                return await self.discover(query, operation, topic, limit, offset)
            except ValueError as exc:
                raise ToolError(str(exc)) from None

        @server.tool(
            name="call", annotations=ToolAnnotations(destructiveHint=True, openWorldHint=True)
        )
        async def call(
            operation: str,
            arguments: dict,
            request_id: str | None = None,
            detail: Literal["compact", "full"] = "compact",
        ) -> dict:
            """Call a discovered operation. request_id applies to jms_submit_job.
            Full responses: read_result(stream='result').
            """
            import jsonschema

            try:
                return await self.call(operation, arguments, request_id)
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
