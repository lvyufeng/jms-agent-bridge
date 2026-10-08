"""The MCP server: the tool surface Claude Code talks to.

Runs as a local stdio process launched by Claude Code. Tools are grouped by plane:

* control plane (:mod:`jms_bridge.api`) — ``jms_list_assets``, ``jms_audit_query``,
  ``jms_create_ticket``
* data plane (:mod:`jms_bridge.ssh`, :mod:`jms_bridge.jobs`) — ``jms_probe``,
  ``jms_exec``, ``jms_sync_push`` / ``jms_sync_pull``, and the job lifecycle

State (config, inventory, the SSH runner, the API client) is built lazily on first
use so that ``import`` never fails just because the inventory is not written yet.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp.server.mcpserver import MCPServer

from .api import JumpServerAPI, JumpServerError
from .config import BastionConfig, load_config
from .inventory import Host, Inventory
from .jobs import JobManager
from .ssh import SSHRunner

log = logging.getLogger("jms_bridge.server")

server = MCPServer("jms-agent-bridge")

# --- lazily-initialised singletons ------------------------------------------------

_state: dict[str, Any] = {}


def _cfg() -> BastionConfig:
    if "cfg" not in _state:
        _state["cfg"] = load_config()
    return _state["cfg"]


def _inv() -> Inventory:
    if "inv" not in _state:
        _state["inv"] = Inventory.load()
    return _state["inv"]


def _ssh() -> SSHRunner:
    if "ssh" not in _state:
        _state["ssh"] = SSHRunner(_cfg())
    return _state["ssh"]


def _jobs() -> JobManager:
    if "jobs" not in _state:
        _state["jobs"] = JobManager(_ssh())
    return _state["jobs"]


def _api() -> JumpServerAPI:
    if "api" not in _state:
        _state["api"] = JumpServerAPI(_cfg())
    return _state["api"]


def _resolve(selectors: list[str]) -> list[Host]:
    return _inv().resolve(selectors)


# --- control plane: discovery / audit / tickets -----------------------------------


@server.tool()
async def jms_list_assets(search: str | None = None) -> dict:
    """List assets this account can reach in JumpServer, with their bound accounts.

    Use this to discover asset ids for the inventory. Returns assets (id, name, ip,
    platform) and, for the first page, each asset's system-user accounts.
    """
    api = _api()
    try:
        assets = await api.list_assets(search=search)
    except JumpServerError as exc:
        return {"ok": False, "error": str(exc)}
    out = []
    for a in assets:
        out.append(
            {
                "id": a.get("id"),
                "name": a.get("name"),
                "address": a.get("address") or a.get("ip"),
                "platform": (a.get("platform") or {}).get("name")
                if isinstance(a.get("platform"), dict)
                else a.get("platform"),
            }
        )
    return {"ok": True, "count": len(out), "assets": out}


@server.tool()
async def jms_audit_query(
    asset: str | None = None, user: str | None = None, limit: int = 30
) -> dict:
    """Recent command records from JumpServer's audit log.

    This is how the agent (and you) answer 'what has actually been executed?' —
    every data-plane command is attributed here because it goes through KoKo.
    """
    api = _api()
    try:
        records = await api.command_records(asset=asset, user=user, limit=limit)
    except JumpServerError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "count": len(records),
        "records": [
            {
                "user": r.get("user"),
                "asset": r.get("asset"),
                "input": r.get("input"),
                "timestamp": r.get("timestamp"),
                "session": r.get("session"),
            }
            for r in records
        ],
    }


@server.tool()
async def jms_create_ticket(
    title: str, ticket_type: str, body: str = "", assignees: list[str] | None = None
) -> dict:
    """Raise a JumpServer approval ticket (e.g. for a command or login that policy gates)."""
    api = _api()
    try:
        return {"ok": True, "ticket": await api.create_ticket(
            title=title, ticket_type=ticket_type, body=body, assignees=assignees
        )}
    except JumpServerError as exc:
        return {"ok": False, "error": str(exc)}


# --- data plane: probe / exec -----------------------------------------------------


@server.tool()
async def jms_probe(targets: list[str]) -> dict:
    """Probe hosts for liveness, accelerator type and driver/runtime version.

    ``targets`` are inventory selectors: host names, group names, or tags. Runs the
    platform-appropriate command (``npu-smi`` / ``nvidia-smi`` / ``uname``) on every
    host in parallel and parses the result.
    """
    hosts = _resolve(targets)
    results = []
    for host in hosts:
        platform = host.platform_def
        res = await _ssh().exec_on(host, platform.probe_cmd, timeout=45)
        parsed = platform.parse_probe(res.stdout) if res.ok else {}
        results.append(
            {
                "host": host.name,
                "platform": platform.name,
                "ok": res.ok,
                "exit_code": res.exit_code,
                **parsed,
                "stderr": res.stderr.strip()[:400] if not res.ok else "",
            }
        )
    return {"ok": all(r["ok"] for r in results), "hosts": results}


@server.tool()
async def jms_exec(
    targets: list[str], command: str, timeout: float = 60.0, allow_dangerous: bool = False
) -> dict:
    """Run a SHORT command on one or many hosts, in parallel, and return its output.

    Use this for inspection (`nvidia-smi`, `ls`, `git status`, `df -h`), not for
    training — anything long-running must go through `jms_submit_job`. ``timeout``
    is per host; a host that exceeds it is reported as timed out, not an error.
    """
    if _looks_dangerous(command) and not allow_dangerous:
        return {
            "ok": False,
            "error": "command matched the safety denylist; pass allow_dangerous=true to override",
            "command": command,
        }
    hosts = _resolve(targets)
    results = await _ssh().exec_many(hosts, command, timeout=timeout)
    return {"ok": all(r.ok for r in results), "results": [r.to_dict() for r in results]}


# --- data plane: file sync --------------------------------------------------------


@server.tool()
async def jms_sync_push(
    targets: list[str],
    local: str,
    remote: str,
    excludes: list[str] | None = None,
    delete: bool = False,
) -> dict:
    """rsync a local path to every targeted host (through the bastion).

    Defaults exclude common noise (`.git`, virtualenvs, caches) so a push does not
    drag a workstation's junk onto a cluster node.
    """
    hosts = _resolve(targets)
    excludes = excludes if excludes is not None else _DEFAULT_EXCLUDES
    results = await _gather_rsync(hosts, local, remote, "push", excludes, delete)
    return {"ok": all(r.ok for r in results), "results": [r.to_dict() for r in results]}


@server.tool()
async def jms_sync_pull(
    targets: list[str], remote: str, local: str, excludes: list[str] | None = None
) -> dict:
    """rsync a remote path from every targeted host back to the local machine."""
    hosts = _resolve(targets)
    results = await _gather_rsync(hosts, local, remote, "pull", excludes or [], False)
    return {"ok": all(r.ok for r in results), "results": [r.to_dict() for r in results]}


async def _gather_rsync(hosts, local, remote, direction, excludes, delete):  # noqa: ANN001
    return list(
        await asyncio.gather(
            *(
                _ssh().rsync(
                    h,
                    local,
                    remote,
                    direction=direction,
                    delete=delete,
                    excludes=excludes,
                )
                for h in hosts
            )
        )
    )


# --- data plane: jobs -------------------------------------------------------------


@server.tool()
async def jms_submit_job(targets: list[str], command: str) -> dict:
    """Launch a long-running command (e.g. Torch-FL training) detached on each host.

    Returns immediately with a job id per host. The command runs inside a tmux
    session with its env activated, writing to a log you poll with `jms_job_logs`.
    Use this — never `jms_exec` — for anything that outlives a tool call.
    """
    if _looks_dangerous(command):
        return {"ok": False, "error": "command matched the safety denylist", "command": command}
    hosts = _resolve(targets)
    jobs = _jobs()
    out = []
    submitted = await asyncio.gather(*(jobs.submit(h, command) for h in hosts))
    for handle, res in submitted:
        out.append(
            {
                "host": handle.host if handle else "?",
                "job_id": handle.job_id if handle else None,
                "launched": handle is not None,
                "stderr": res.stderr.strip()[:400] if handle is None else "",
            }
        )
    return {"ok": all(o["launched"] for o in out), "jobs": out}


@server.tool()
async def jms_job_status(targets: list[str], job_id: str) -> dict:
    """Check a job's status on each targeted host (status, exit code, log size, alive)."""
    hosts = _resolve(targets)
    results = await asyncio.gather(*(_jobs().status(h, job_id) for h in hosts))
    return {"ok": True, "status": list(results)}


@server.tool()
async def jms_job_logs(
    host: str, job_id: str, offset: int = 0, max_bytes: int = 65536
) -> dict:
    """Read a job's log from `offset` onward. Poll with the returned `size` as the next offset."""
    h = _inv().resolve([host])[0]
    return {"ok": True, **(await _jobs().logs(h, job_id, offset=offset, max_bytes=max_bytes))}


@server.tool()
async def jms_job_cancel(host: str, job_id: str) -> dict:
    """Kill a running job on one host."""
    h = _inv().resolve([host])[0]
    return {"ok": True, **(await _jobs().cancel(h, job_id))}


# --- helpers ----------------------------------------------------------------------

_DEFAULT_EXCLUDES = [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "*.pyc",
    ".mypy_cache",
    ".pytest_cache",
    "*.egg-info",
    ".DS_Store",
]

#: Patterns that are almost certainly a mistake when issued by an autonomous agent.
#: Not a security boundary — JumpServer's own command blacklist is the backstop.
_DANGEROUS_PATTERNS = [
    "rm -rf /",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",
    "shutdown",
    "reboot",
    "> /dev/sda",
    "chmod -R 777 /",
]


def _looks_dangerous(command: str) -> bool:
    flat = " ".join(command.split())
    return any(pat in flat for pat in _DANGEROUS_PATTERNS)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server.run("stdio")


if __name__ == "__main__":
    main()