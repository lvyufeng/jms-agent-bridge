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
from .cluster import (
    DEFAULT_RDZV_PORT,
    is_heterogeneous,
    plan_islands,
    preflight,
    resolve_rendezvous,
)
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
    asset_id: str | None = None, user: str | None = None, days: int = 5, limit: int = 30
) -> dict:
    """Recent command records from JumpServer's audit log.

    This is how the agent (and you) answer 'what has actually been executed?' —
    every data-plane command is attributed here because it goes through KoKo.
    `days` is the look-back window (JumpServer defaults to the last 5 days).
    """
    api = _api()
    try:
        records = await api.command_records(
            asset_id=asset_id, user=user, days=days, limit=limit
        )
    except JumpServerError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "count": len(records),
        "records": [
            {
                "user": r.get("user"),
                "asset": r.get("asset"),
                "account": r.get("account"),
                "input": r.get("input"),
                "timestamp": r.get("timestamp"),
                "session": r.get("session"),
            }
            for r in records
        ],
    }


@server.tool()
async def jms_create_ticket(
    title: str,
    ticket_type: str = "login_asset_confirm",
    comment: str = "",
    org_id: str | None = None,
    fields: dict | None = None,
) -> dict:
    """Raise a JumpServer approval ticket via the subtype's ``open`` action.

    `ticket_type` is one of: `login_asset_confirm`, `command_confirm`, `apply_asset`,
    `login_confirm`. `fields` carries the type-specific payload, e.g. for a
    login-asset ticket ``{"apply_login_user": "<uuid>", "apply_login_asset":
    "<uuid>", "apply_login_account": "root"}``.
    """
    api = _api()
    try:
        return {
            "ok": True,
            "ticket": await api.create_ticket(
                title=title,
                ticket_type=ticket_type,
                comment=comment,
                org_id=org_id,
                **(fields or {}),
            ),
        }
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


# --- cluster: preflight + coordinated launch ---------------------------------------


@server.tool()
async def jms_cluster_preflight(targets: list[str], port: int = DEFAULT_RDZV_PORT) -> dict:
    """Check a cluster is ready for a distributed run, host by host.

    For every host: reachability, accelerator visibility and version (via the
    platform probe), and whether the rendezvous `port` is already bound (a stale
    previous run). Returns an estimated `world_size` and a suggested rendezvous.
    Run this before `jms_cluster_submit`.
    """
    hosts = _resolve(targets)
    health = await preflight(_ssh(), hosts, port=port)
    ranks = sum(h.device_count or 1 for h in health if h.reachable and h.device_count)
    rdv = resolve_rendezvous(hosts, port=port)
    return {
        "ok": all(h.ok for h in health),
        "world_size_estimate": ranks,
        "rendezvous": rdv.to_dict(),
        "heterogeneous": is_heterogeneous(hosts),
        "islands": [i.to_dict() for i in plan_islands(hosts)],
        "hosts": [h.to_dict() for h in health],
        "platforms": sorted({h.platform for h in health}),
    }


@server.tool()
async def jms_cluster_plan(targets: list[str], port: int = DEFAULT_RDZV_PORT) -> dict:
    """Plan a federated layout for a target set, grouped by communication backend.

    Ascend (HCCL) and NVIDIA (NCCL) nodes cannot form a single collective. This
    partitions the targets into **islands** — one per backend — each of which runs
    its own coordinated job, and returns the federation plan: per-island rendezvous,
    world size and job wiring. Run `jms_cluster_submit` once per island, then let the
    islands exchange model updates between rounds (the FL exchange is the agent's
    application logic; the bridge provides per-island launch + rendezvous).
    """
    hosts = _resolve(targets)
    plan = []
    for island in plan_islands(hosts):
        rdv = resolve_rendezvous(island.hosts, port=port)
        plan.append(
            {
                **island.to_dict(),
                "rendezvous": rdv.to_dict(),
                "world_size": len(island.hosts),
                "submit_with_targets": [h.name for h in island.hosts],
            }
        )
    return {
        "ok": True,
        "heterogeneous": is_heterogeneous(hosts),
        "islands": plan,
        "note": (
            "Same-backend hosts in one island CAN form a single collective. Cross-island "
            "training must federate (exchange updates between rounds), not share a "
            "collective — HCCL and NCCL do not interoperate."
        ),
    }


@server.tool()
async def jms_cluster_submit(
    targets: list[str],
    command: str,
    master: str | None = None,
    port: int = DEFAULT_RDZV_PORT,
) -> dict:
    """Launch ONE coordinated distributed job across a heterogeneous cluster.

    Unlike `jms_submit_job` (which runs the command independently on each host),
    this assigns every host a rank and gives them a shared rendezvous, exporting
    `JMS_RANK`, `JMS_LOCAL_RANK`, `JMS_NPROC`, `JMS_WORLD_SIZE`, `JMS_MASTER_ADDR`
    and `JMS_MASTER_PORT` before `command` runs. All ranks launch concurrently and
    the call returns immediately with a job id per host.

    `command` is your per-node launcher; wire it with the exported env. For torchrun:

        torchrun --nnodes=$JMS_WORLD_SIZE --node_rank=$JMS_RANK \\
                 --nproc_per_node=$JMS_NPROC \\
                 --rdzv_endpoint=$JMS_MASTER_ADDR:$JMS_MASTER_PORT \\
                 train.py --rounds 100

    `targets` order defines rank order; each host contributes one rank node whose
    `JMS_NPROC` is its accelerator count. Poll with `jms_job_status` / `jms_job_logs`.
    """
    if _looks_dangerous(command):
        return {"ok": False, "error": "command matched the safety denylist", "command": command}
    hosts = _resolve(targets)
    if len(hosts) < 2:
        return {
            "ok": False,
            "error": "cluster submit needs at least 2 hosts; use jms_submit_job for one",
        }
    if is_heterogeneous(hosts):
        islands = plan_islands(hosts)
        return {
            "ok": False,
            "error": (
                "targets span multiple communication backends "
                f"({[i.backend for i in islands]}); a single collective is impossible "
                "(e.g. HCCL and NCCL do not interoperate). Submit one island at a time "
                "and federate across them."
            ),
            "islands": [i.to_dict() for i in islands],
            "hint": "call jms_cluster_plan, then jms_cluster_submit once per island",
        }
    rdv = resolve_rendezvous(hosts, master=master, port=port)

    health = {h.host: h for h in await preflight(_ssh(), hosts, port=port)}
    assignments = [(h, max(1, health[h.name].device_count or 1)) for h in hosts]
    world_size = sum(n for _, n in assignments)

    handles, results = await _jobs().submit_collab(
        assignments, command, master_addr=rdv.master_addr, port=port, world_size=world_size
    )
    by_host = {h.host: h.job_id for h in handles}
    return {
        "ok": len(handles) == len(hosts),
        "rendezvous": rdv.to_dict(),
        "world_size": world_size,
        "ranks": [
            {
                "host": h.name,
                "job_id": by_host.get(h.name),
                "nproc": n,
                "launched": h.name in by_host,
            }
            for h, n in assignments
        ],
    }


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