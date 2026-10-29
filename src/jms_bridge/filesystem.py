"""Structured, read-only remote exploration over the resident SSH runner."""

import re
from pathlib import Path

from .async_io import offload
from .bash_fs import WIRE_LIMIT, decode, remote_script
from .fs_workers import FilesystemWorkers

OPERATION = "jms_fs"
ACTIONS = [
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
]


def diagnostic_prefix(path):
    with path.open("rb") as stream:
        return stream.read(2048)


def integer(low, high, default, description):
    return dict(type="integer", minimum=low, maximum=high, default=default, description=description)


SCHEMA = {
    "name": OPERATION,
    "description": (
        "Remote filesystem operations without shell commands. "
        "list/stat/read/read_many/glob/search/tree/checksum are "
        "read-only; upload/download write the destination via "
        "existing SSH/rsync. Prefer the top-level fs tool. One exact "
        "REST asset; optional existing Docker container. Bounded "
        "pages/cursors and explicit partial/error reporting."
    ),
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["target", "action"],
        "properties": {
            "target": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "One exact REST asset name, ID, IP or unique alias; no groups/wildcards."
                ),
            },
            "action": {"type": "string", "enum": ACTIONS},
            "path": {
                "type": "string",
                "minLength": 1,
                "maxLength": 4096,
                "description": (
                    "Absolute remote file/directory path. Required except "
                    "read_many. Search accepts a file or directory. No shell "
                    "expansion."
                ),
            },
            "paths": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "items": {"type": "string", "minLength": 1},
                "description": "Absolute paths for read_many; per-file errors are preserved.",
            },
            "container": {
                "type": "string",
                "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
                "description": "Existing Docker container name/ID; never creates a container.",
            },
            "local_path": {
                "type": "string",
                "minLength": 1,
                "maxLength": 4096,
                "description": (
                    "upload/download only: absolute path on the MCP server "
                    "machine. path is always the remote path. Trailing slash "
                    "uses normal rsync directory semantics."
                ),
            },
            "overwrite": {
                "type": "boolean",
                "default": False,
                "description": (
                    "upload/download: replace existing destination files only "
                    "when true. False skips existing files; directory metadata "
                    "may still be synchronized. Never deletes extra destination "
                    "files."
                ),
            },
            "request_id": {
                "type": "string",
                "pattern": "^[A-Za-z0-9_.:-]{1,128}$",
                "description": (
                    "upload/download: caller-generated unique ID for receipt "
                    "recovery and no duplicate execution. Query request_status "
                    "after a lost response."
                ),
            },
            "pattern": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1024,
                "description": (
                    "Required for glob/search. Glob matches basename or relative "
                    "path with *, ?, [], **/ (no brace expansion). Search is "
                    "literal by default."
                ),
            },
            "literal": {
                "type": "boolean",
                "default": True,
                "description": (
                    "search: false enables regex (ripgrep syntax, or GNU grep POSIX ERE "
                    "when rg is absent)."
                ),
            },
            "case_sensitive": {"type": "boolean", "default": True},
            "include": {
                "type": "array",
                "maxItems": 20,
                "items": {"type": "string", "minLength": 1, "maxLength": 256},
                "description": "glob/search: include file globs, e.g. *.py.",
            },
            "exclude": {
                "type": "array",
                "maxItems": 20,
                "items": {"type": "string", "minLength": 1, "maxLength": 256},
                "description": "Traversal exclusions, e.g. .git, node_modules, build/**.",
            },
            "hidden": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Include dotfiles in directory traversal. Explicit read/stat "
                    "paths are unaffected."
                ),
            },
            "respect_ignore": {
                "type": "boolean",
                "default": True,
                "description": (
                    "glob/search: rg respects .gitignore/.ignore. Bash "
                    "fallback reports ignore rules unavailable."
                ),
            },
            "offset": integer(
                0,
                100000,
                0,
                (
                    "list/tree/glob/search: number of result rows to skip; "
                    "repeat identical options to continue. Not a snapshot."
                ),
            ),
            "limit": integer(1, 1000, 100, "Maximum result rows per directory/search page."),
            "depth": integer(1, 20, 3, "tree: maximum relative depth."),
            "start_line": integer(
                1,
                10000000,
                1,
                (
                    "read/read_many: 1-based starting line; prefer byte_offset "
                    "for continuing long files."
                ),
            ),
            "line_count": integer(1, 2000, 200, "read/read_many: maximum lines per file."),
            "byte_offset": integer(
                0,
                1099511627776,
                0,
                (
                    "read: optional byte cursor returned by the previous page; "
                    "exclusive with start_line/tail."
                ),
            ),
            "tail": {
                "type": "boolean",
                "default": False,
                "description": (
                    "read/read_many: last line_count lines within max_bytes; no "
                    "whole-file download."
                ),
            },
            "max_bytes": integer(
                256,
                65536,
                4096,
                (
                    "Content budget per response (shared across read_many); "
                    "metadata is additional. One indivisible directory/search "
                    "row may exceed it. UTF-8 alignment may add <=3 bytes per "
                    "file."
                ),
            ),
            "context_lines": integer(
                0,
                5,
                0,
                "search: include nearby context lines; context rows also count toward limit.",
            ),
            "max_file_bytes": integer(
                1024, 67108864, 8388608, "search/checksum: maximum individual file size."
            ),
            "timeout": integer(
                1,
                900,
                30,
                (
                    "Exploration: 1..120s, default 30. Transfers: 1..900s, "
                    "default 900. Separate from SSH setup/queue wait."
                ),
            ),
        },
    },
    "annotations": {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": True},
}


def validate(arguments):
    action = arguments["action"]
    paths = arguments.get("paths", []) if action == "read_many" else [arguments.get("path", "")]
    if not paths or any(not p.startswith("/") or "\x00" in p for p in paths):
        raise ValueError("provide absolute remote path (or paths for read_many); NUL is invalid")
    if action in ("glob", "search") and not arguments.get("pattern"):
        raise ValueError("glob/search requires pattern")
    if "byte_offset" in arguments and ("start_line" in arguments or arguments.get("tail")):
        raise ValueError("byte_offset cannot be combined with start_line or tail")
    if action == "read_many" and "byte_offset" in arguments:
        raise ValueError("use read for individual byte cursors")
    if action == "read_many" and arguments.get("max_bytes", 4096) < 256 * len(paths):
        raise ValueError("read_many max_bytes must allow at least 256 bytes per file")
    if action in ("upload", "download"):
        local = arguments.get("local_path", "")
        if not local.startswith("/") or "\x00" in local:
            raise ValueError(
                "upload/download requires absolute local_path on the MCP server machine"
            )
        if action == "upload" and not Path(local).exists():
            raise ValueError("upload local_path does not exist on the MCP server machine")
    elif arguments.get("timeout", 30) > 120:
        raise ValueError("exploration timeout must be <=120 seconds")
    allowed = {"target", "action", "path", "container", "timeout", "max_bytes"}
    allowed |= {
        "list": {"offset", "limit", "hidden", "exclude"},
        "tree": {"offset", "limit", "hidden", "exclude", "depth"},
        "stat": set(),
        "checksum": {"max_file_bytes"},
        "upload": {"local_path", "overwrite", "exclude", "request_id"},
        "download": {"local_path", "overwrite", "exclude", "request_id"},
        "read": {"start_line", "line_count", "byte_offset", "tail"},
        "read_many": {"paths", "start_line", "line_count", "tail"},
        "glob": {"pattern", "include", "exclude", "hidden", "respect_ignore", "offset", "limit"},
        "search": {
            "pattern",
            "literal",
            "case_sensitive",
            "include",
            "exclude",
            "hidden",
            "respect_ignore",
            "offset",
            "limit",
            "context_lines",
            "max_file_bytes",
        },
    }[action]
    unused = set(arguments) - allowed
    if unused or (action == "read_many" and "path" in arguments):
        raise ValueError(
            "parameters not applicable to action: " + ", ".join(sorted(unused or {"path"}))
        )


class Filesystem:
    def __init__(self, catalog, runner, store):
        self.catalog, self.runner, self.store = catalog, runner, store
        self.workers = FilesystemWorkers(runner)

    async def execute(self, **arguments):
        validate(arguments)
        target = arguments["target"]
        if target.lower() == "all" or re.search(r"[*?\[\]]", target):
            raise ValueError("jms_fs requires one explicit REST target")
        inventory = await self.catalog.ensure()
        hosts = inventory.resolve([target])
        if len(hosts) != 1:
            raise ValueError("jms_fs requires exactly one permitted SSH asset")
        if arguments["action"] in ("upload", "download"):
            transfer = await self.runner.rsync(
                hosts[0],
                arguments["local_path"],
                arguments["path"],
                direction="push" if arguments["action"] == "upload" else "pull",
                excludes=arguments.get("exclude", []),
                delete=False,
                timeout=arguments.get("timeout", 900),
                container=arguments.get("container"),
                overwrite=arguments.get("overwrite", False),
            )
            meta = await offload(self.store.metadata, transfer.result_id)
            return (
                {
                    k: meta[k]
                    for k in (
                        "ok",
                        "host",
                        "result_id",
                        "exit_code",
                        "timing",
                        "completion_unknown",
                        "execution_state",
                        "queue_timed_out",
                        "failure_stage",
                    )
                }
                | await offload(self.store.preview, transfer.result_id, 4096)
                | {
                    "action": arguments["action"],
                    "local_path": arguments["local_path"],
                    "path": arguments["path"],
                    "overwrite": arguments.get("overwrite", False),
                    "existing_files": "replace" if arguments.get("overwrite") else "skip",
                }
            )
        options = {k: v for k, v in arguments.items() if k not in ("target", "container")}
        meta = await self.workers.execute(
            hosts[0],
            remote_script(options),
            arguments.get("timeout", 30) + 5,
            container=arguments.get("container"),
        )
        rid = meta["result_id"]
        if not meta["ok"]:
            return {
                k: v
                for k, v in meta.items()
                if k
                in (
                    "ok",
                    "result_id",
                    "host",
                    "exit_code",
                    "execution_state",
                    "completion_unknown",
                    "timing",
                    "warning",
                    "failure_stage",
                    "startup_retry_result_id",
                    "connection_recovery",
                )
            } | await offload(self.store.preview, rid, 2048)
        path = self.store.path(rid, "stdout")
        if (await offload(path.stat)).st_size > WIRE_LIMIT * 2 + 65536:
            return dict(ok=False, result_id=rid, error="filesystem probe exceeded wire limit")
        raw = await offload(path.read_bytes)
        try:
            stderr = await offload(diagnostic_prefix, self.store.path(rid, "stderr"))
            value = decode(raw, options, stderr)
        except (ValueError, UnicodeError, KeyError, TypeError, OverflowError):
            return dict(
                ok=False,
                result_id=rid,
                error="invalid filesystem probe response; inspect stdout/stderr",
            )
        result_id = await offload(
            self.store.save_json,
            value,
            kind="filesystem",
            operation=arguments["action"],
            host=hosts[0].name,
        )
        result = dict(
            ok=value.get("ok", False),
            result_id=result_id,
            execution_result_id=rid,
            host=hosts[0].name,
            timing=meta["timing"],
        )
        if "startup_retry_result_id" in meta:
            result["startup_retry_result_id"] = meta["startup_retry_result_id"]
        if "connection_recovery" in meta:
            result["connection_recovery"] = meta["connection_recovery"]
        # Keep the native response usable JSON even when content is large.
        # max_bytes bounds content; rows/metadata can add modest overhead.
        return result | {"data": value}
