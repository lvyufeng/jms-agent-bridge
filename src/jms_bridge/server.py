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
import functools
import logging
import os
import re
import subprocess
from collections.abc import Callable
from typing import Any, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .api import JumpServerAPI, JumpServerError
from .cluster import (
    DEFAULT_RDZV_PORT,
    is_heterogeneous,
    plan_islands,
    preflight,
    resolve_rendezvous,
)
from .config import BastionConfig, ConfigError, load_config
from .discover import DEFAULT_REPO_MATCH
from .inventory import Host, Inventory, InventoryError, shell_quote
from .jobs import JobManager, validate_job_id
from .ssh import ExecResult, SSHRunner
from .workspace import CONTAINER_RE, DEFAULT_CONTAINER, SOURCES

log = logging.getLogger("jms_bridge.server")

server = MCPServer("jms-agent-bridge")

_F = TypeVar("_F", bound=Callable[..., Any])


def _tool() -> Callable[[_F], _F]:
    """``@server.tool()``, with setup failures turned into a readable ``ToolError``.

    ``ConfigError`` (no bastion URL, unreadable config) and ``InventoryError``
    (missing/malformed ``hosts.yaml``, or a selector that matched nothing) are raised
    from the lazy state helpers and from ``resolve()`` inside almost every tool. The
    SDK forwards the *text* of a ``ToolError`` to the client; anything else reaches the
    agent as the opaque ``Error executing tool <name>`` — which is exactly, and
    uselessly, what a freshly-cloned-but-unconfigured checkout would see. Translating
    keeps the actionable message ("No inventory at …; run `jms-bridge init` or set
    JMS_BRIDGE_INVENTORY") in front of the agent.
    """

    def decorate(fn: _F) -> _F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except (ConfigError, InventoryError) as exc:
                raise ToolError(str(exc)) from exc

        return server.tool()(wrapper)  # type: ignore[return-value]

    return decorate


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


@_tool()
async def jms_list_assets(search: str | None = None) -> dict:
    """List the assets this account can reach in JumpServer.

    Use this to discover the ``asset_id`` values the inventory needs. Returns each
    asset's id, name, address and platform. Bound accounts are *not* included — this
    call is one REST request, and enumerating the ``#account`` part of every login
    would be one request per asset. For a single asset, either read it from the
    JumpServer UI or use the asset's detail endpoint.
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


@_tool()
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


@_tool()
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


@_tool()
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
        res = await _ssh().exec_on(host, platform.probe_cmd, timeout=45, workdir=False)
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


@_tool()
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


@_tool()
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


@_tool()
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


#: Where `jms_pull_artifacts` puts what it pulls, one directory per host.
_ARTIFACT_ROOT = "~/.jms-bridge/artifacts"

#: Artifacts are checkpoints and can be gigabytes, so the fan-out is bounded far below
#: the SSHRunner's own cap: a handful of concurrent transfers keeps a shared uplink
#: usable for everyone else on the cluster.
_ARTIFACT_PARALLELISM = 4

#: Machine-generated noise, not artifacts. Deliberately *not* `_DEFAULT_EXCLUDES` — that
#: list drops whole directories, and pulling a run's output directory must not silently
#: skip a `.git`- or cache-named subdirectory the run actually wrote.
_ARTIFACT_EXCLUDES = ["*.pyc", "__pycache__", ".pytest_cache"]


def _artifact_remote_error(remote: str) -> str | None:
    """Reject a remote path that could reach outside the host workdir.

    The path is agent-supplied and the pull runs as the asset's account, so an absolute
    path or a `..` segment would rsync something like `/etc` back to the workstation.
    Artifacts belong under the run's workdir, so require a relative, `..`-free path; if
    a run wrote elsewhere, this is the point to say so rather than to guess.
    """
    if not remote.strip():
        return "remote is empty"
    if remote.startswith(("/", "~")):
        return "remote must be relative to the host workdir (no leading '/' or '~')"
    if ".." in remote.replace("\\", "/").split("/"):
        return "remote must not contain a '..' segment"
    return None


def _artifact_subdir(name: str) -> str:
    """A filesystem-safe directory name for one host.

    Host names are operator-supplied and may contain characters that mean something to
    a path (or be `.`/`..`). One directory per host is what stops same-named files from
    different hosts overwriting each other on the way back.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip(".")
    return safe or "host"


@_tool()
async def jms_pull_artifacts(
    targets: list[str],
    remote: str,
    local: str | None = None,
    excludes: list[str] | None = None,
) -> dict:
    """rsync one artifact path from every targeted host into its own local directory.

    `jms_sync_pull` sends every host to the *same* local path, so pulling the same
    relative path off ten nodes means ten transfers into one directory and nine silent
    overwrites. This gives each host its own subdirectory instead —
    `~/.jms-bridge/artifacts/<host>/` by default, or `<local>/<host>/` — so a
    per-node checkpoint, log or metric dump survives the trip back.

    `remote` must be relative to the host workdir (no leading `/` or `~`, no `..`): the
    copy runs as the asset's account, and an absolute path would pull arbitrary system
    files off the cluster. `excludes` defaults to machine-generated noise only — a
    checkpoint pull should not also drag back a run's caches. Transfers are capped at
    four at a time and a host that fails is reported on its own, not at the expense of
    the others.
    """
    hosts = _resolve(targets)
    bad = _artifact_remote_error(remote)
    if bad:
        return {"ok": False, "error": bad, "remote": remote}
    root = os.path.expanduser(local or _ARTIFACT_ROOT)
    excludes = _ARTIFACT_EXCLUDES if excludes is None else excludes
    sem = asyncio.Semaphore(_ARTIFACT_PARALLELISM)

    async def one(host: Host) -> dict:
        dest = os.path.join(root, _artifact_subdir(host.name))
        async with sem:
            try:
                res = await _ssh().rsync(host, dest, remote, direction="pull", excludes=excludes)
            except Exception as exc:  # noqa: BLE001 - one host must not hide the rest
                return {"host": host.name, "local": dest, "ok": False, "error": str(exc)}
        return {
            "host": host.name,
            "local": dest,
            "ok": res.ok,
            "stderr": res.stderr.strip()[:400] if not res.ok else "",
        }

    results = list(await asyncio.gather(*(one(h) for h in hosts)))
    return {
        "ok": bool(results) and all(r["ok"] for r in results),
        "remote": remote,
        "root": root,
        "results": results,
    }


async def _per_host(hosts: list[Host], fn: Callable[..., Any], *, key: str) -> list[dict]:
    """Run ``fn(host)`` on every host; one host's failure becomes that host's entry.

    A plain ``gather`` lets a single unreachable host raise and erase every other host's
    result — on a 40-node cluster one bad node hides the other 39. ``CancelledError`` is
    a ``BaseException``, so a real cancellation still propagates.
    """

    async def one(host: Host) -> dict:
        try:
            return {"host": host.name, key: await fn(host)}
        except Exception as exc:  # noqa: BLE001 - isolating one host is the whole point
            return {"host": host.name, key: None, "error": f"{type(exc).__name__}: {exc}"}

    return list(await asyncio.gather(*(one(h) for h in hosts)))


# --- data plane: jobs -------------------------------------------------------------


@_tool()
async def jms_submit_job(targets: list[str], command: str, allow_dangerous: bool = False) -> dict:
    """Launch a long-running command (e.g. Torch-FL training) detached on each host.

    Returns immediately with a job id per host. The command runs inside a tmux
    session with its env activated, writing to a log you poll with `jms_job_logs`.
    Use this — never `jms_exec` — for anything that outlives a tool call.
    """
    if _looks_dangerous(command) and not allow_dangerous:
        return {
            "ok": False,
            "error": "command matched the safety denylist; pass allow_dangerous=true to override",
            "command": command,
        }
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


@_tool()
async def jms_job_status(targets: list[str], job_id: str) -> dict:
    """Check a job's status on each targeted host (status, exit code, log size, alive)."""
    hosts = _resolve(targets)
    results = await asyncio.gather(*(_jobs().status(h, job_id) for h in hosts))
    return {"ok": True, "status": list(results)}


@_tool()
async def jms_job_logs(
    host: str, job_id: str, offset: int = 0, max_bytes: int = 65536
) -> dict:
    """Read a job's log from `offset` onward. Poll with the returned `next_offset`."""
    h = _inv().resolve([host])[0]
    return {"ok": True, **(await _jobs().logs(h, job_id, offset=offset, max_bytes=max_bytes))}


@_tool()
async def jms_job_cancel(host: str, job_id: str) -> dict:
    """Kill a running job on one host."""
    h = _inv().resolve([host])[0]
    return {"ok": True, **(await _jobs().cancel(h, job_id))}


@_tool()
async def jms_job_wait(
    targets: list[str], job_id: str, timeout: float = 150.0, interval: float = 2.0
) -> dict:
    """Block until a job has stopped on every targeted host, or `timeout` elapses.

    Polls `jms_job_status` for you and returns the moment every targeted host reports a
    terminal status (`done` / `failed` / `missing`) — the same job id on each host, which
    is what a coordinated `jms_cluster_submit` hands out. `finished` says whether they
    all stopped; a timeout is not a failure and **kills nothing**, so on `finished=false`
    the jobs are still running and you can call this again or keep polling manually.

    Keep `timeout` under ~200s: the gateway buffers a non-streaming response, so a tool
    call that runs longer is cut off by the transport rather than by this timeout.
    """
    hosts = _resolve(targets)
    bad = validate_job_id(job_id)
    if bad:
        return {"ok": False, "error": bad}
    latest, finished = await _jobs().wait(
        [(h, job_id) for h in hosts], timeout=timeout, interval=interval
    )
    return {"ok": finished, "finished": finished, "job_id": job_id, "status": latest}


@_tool()
async def jms_job_list(targets: list[str]) -> dict:
    """List the job ids currently running on each targeted host.

    Answers "what did I leave running on this cluster?" — useful after a lost session,
    when the job ids you were handed are gone but the tmux sessions are not.

    Reads `tmux ls`, so it sees jobs launched by `tmux` only: a host that fell back to
    `setsid` (no tmux installed) or a job nested inside a container is invisible here,
    and a job is only listed while its tmux session lives — a finished job is gone from
    this list even though its `status` file and logs remain readable. Pass the ids you
    find to `jms_job_status` / `jms_job_wait` / `jms_job_cancel`.
    """
    hosts = _resolve(targets)
    per_host = await _per_host(hosts, _jobs().list_jobs, key="job_ids")
    return {
        # A host that could not be asked reports its own error; that must not read as
        # "this host has no jobs". An empty selection is likewise not a green listing.
        "ok": bool(per_host) and all("error" not in entry for entry in per_host),
        "hosts": per_host,
        "note": (
            "tmux-launched jobs only; a host with no tmux falls back to setsid and lists nothing"
        ),
    }


@_tool()
async def jms_cluster_cancel(targets: list[str], job_id: str) -> dict:
    """Stop every rank of one coordinated launch.

    A `jms_cluster_submit` hands every rank the *same* job id (on a different host each),
    so one cancel per host is what actually stops a collective — and doing that by hand
    from the per-rank reply is easy to get half-right, leaving a rank alive still holding
    the rendezvous port and the cards. Pass the same `targets` you submitted with and the
    one job id. Hosts are cancelled in parallel and independently; a host where nothing
    was killed reports `killed=false` rather than failing the batch.
    """
    hosts = _resolve(targets)
    bad = validate_job_id(job_id)
    if bad:
        return {"ok": False, "error": bad}
    if not hosts:
        # `resolve([])` is empty without raising, and `all()` over nothing is True — a
        # cancel that reached no host must not report success.
        return {
            "ok": False,
            "job_id": job_id,
            "error": "no hosts matched the selection; nothing was cancelled",
            "cancelled": [],
            "results": [],
        }
    results = await _jobs().cancel_many([(h, job_id) for h in hosts])
    return {
        "ok": all(r.get("ok") for r in results),
        "job_id": job_id,
        "cancelled": [r for r in results if r.get("killed")],
        "results": results,
    }


# --- dev loop: workspace sync / run / logs -----------------------------------------


@_tool()
async def jms_inventory_reload() -> dict:
    """Re-read `hosts.yaml` (and the config) and drop everything cached from them.

    The advertised dev loop is *emit YAML → paste into `hosts.yaml` → run*, but every
    tool reads the inventory through a cache populated on first use. Without this the
    paste step silently needs an MCP restart before the new hosts/workspaces appear.
    Call this after editing the inventory; the next tool call re-reads from disk.

    The config is dropped too, and for the same reason: `jms-bridge login` caches a
    session token *into the config file*, so a server started before logging in would
    otherwise hold a credential-less config until it was restarted.

    Returns the freshly loaded summary so the caller can confirm the paste landed.
    """
    stale_ssh = _state.pop("ssh", None)
    if stale_ssh is not None:
        await stale_ssh.close()  # else each reload leaks a ControlMaster socket dir
    for key in ("inv", "cfg", "jobs", "api"):
        _state.pop(key, None)
    try:
        inv = _inv()
    except Exception as exc:  # noqa: BLE001 - surface any load error to the agent
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "hosts": sorted(inv.hosts),
        "workspaces": sorted(inv.workspaces),
        "groups": {k: list(v) for k, v in inv.groups.items()},
    }


def _workspaces(selectors: list[str]):  # noqa: ANN202
    return _inv().resolve_workspaces(selectors)


def _host_of(ws):  # noqa: ANN001, ANN202
    return _inv().hosts[ws.host]


#: A git object name as `git rev-parse` emits it: 7..40 lowercase hex chars.
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _local_sha(path: str) -> str | None:
    """Best-effort HEAD sha of a *local* git tree (for ``push`` workspaces).

    The container has no git for a ``push`` workspace, but the developer's tree
    does, and a fork and the local tree share commit ids — so this is what lets a
    mixed git+push run detect that the pushed tree is not at ``ref``. Validated
    against :data:`_SHA_RE` so a banner or a short/full mismatch cannot slip
    through as a divergence (or false agreement).
    """
    try:
        out = subprocess.run(
            # expanduser: `local:` in the inventory is often `~/...`, and this path
            # goes straight to exec (no shell), so `~` would never be expanded.
            ["git", "-C", os.path.expanduser(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip() if out.returncode == 0 else ""
    return sha if _SHA_RE.match(sha) else None


def _last_sha(stdout: str) -> str | None:
    """The last sha-shaped line of an SSH stdout, or None.

    Scanning for a *sha-shaped* line (bottom-up) rather than the last non-empty
    line means a container logout banner or login notice printed to stdout after
    ``git rev-parse`` is ignored instead of being mistaken for the commit.
    """
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if _SHA_RE.match(candidate):
            return candidate
    return None


async def _deliver(ws, ref: str) -> dict:  # noqa: ANN001
    """Deliver ``ref`` to one workspace, never raising.

    The fan-out over workspaces is a ``gather``: an uncaught raise from one workspace
    (a workspace naming a host the inventory no longer has, a transport error) would
    discard every other workspace's result and reach the agent as an opaque tool error.
    Converting it into this workspace's failure record keeps the per-chip picture
    intact and names the workspace that broke.
    """
    try:
        return await _deliver_one(ws, ref)
    except Exception as exc:  # noqa: BLE001 - isolating one workspace is the point
        return {
            "workspace": ws.name,
            "source": getattr(ws, "source", "?"),
            "ok": False,
            "sha": None,
            "ref_pinned": False,
            "stderr": f"delivery raised: {exc}",
        }


async def _deliver_one(ws, ref: str) -> dict:  # noqa: ANN001
    """Put the workspace's repo at ``ref`` using its configured source.

    Returns the delivered commit under ``sha`` (best-effort) so a caller can spot
    revision divergence across chips. For ``push`` the code arrives out-of-band and
    the container has no git, so ``ref_pinned`` is False and the sha — when the
    local tree could be read — is the *local* HEAD, not the container's.
    """
    host = _host_of(ws)
    if ws.source == "push":
        # `local:` is commonly `~/...` (the shipped example uses it), and rsync runs
        # via exec with no shell, so `~` would stay literal and the whole delivery
        # would silently push a path named `~`. `_local_sha` expands it for the same
        # reason; the path that actually reaches rsync must match.
        res = await _ssh().rsync(
            host,
            os.path.expanduser(ws.local),
            ws.mount,
            direction="push",
            excludes=_DEFAULT_EXCLUDES,
        )
        sha = await asyncio.to_thread(_local_sha, ws.local)
        return {
            "workspace": ws.name,
            "source": "push",
            "ok": res.ok,
            # Push never applies `ref` in-container; `sha` (below) is the *local*
            # tree's HEAD, which is why ref_pinned is always False here.
            "ref_pinned": False,
            "sha": sha,
            "stderr": res.stderr.strip()[:400],
        }
    cmd = ws.delivery_cmd(ref)
    # One round trip: fetch + checkout write to stderr, rev-parse to stdout.
    res = await _ssh().exec_on(
        host,
        ws.container_cmd(f"{cmd} && git rev-parse HEAD"),
        timeout=300,
        workdir=False,
        env_setup=False,
    )
    sha = _last_sha(res.stdout)
    return {
        "workspace": ws.name,
        "source": ws.source,
        # A delivery that cannot report which commit it landed on is a failure:
        # a run over unverified code must not report ok.
        "ok": res.ok and sha is not None,
        "sha": sha,
        "ref_pinned": True,
        "stderr": res.stderr.strip()[:400],
    }


@_tool()
async def jms_dev_sync(targets: list[str], ref: str) -> dict:
    """Deliver a git `ref` to every selected workspace's container.

    Each workspace uses its configured source (git / proxy-git / push), so hosts with
    direct GitHub access, hosts behind a proxy, and offline hosts all work from this
    one call. `ref` is a branch on the workspace's `origin` (the developer's fork).
    """
    workspaces = _workspaces(targets)
    results = await asyncio.gather(*(_deliver(w, ref) for w in workspaces))
    # `bool(results)`: delivering nothing (empty selector) is not success —
    # all([]) is True, the same trap the consistency table guards against.
    return {
        "ok": bool(results) and all(r["ok"] for r in results),
        "ref": ref,
        "results": list(results),
    }


@_tool()
async def jms_dev_run(
    targets: list[str], ref: str, timeout: float = 600.0, ut_cmd: str | None = None
) -> dict:
    """Deliver `ref` to every workspace, run the UT suite in each container, and
    return ONE cross-chip consistency table.

    This is the development loop: write once, verify on every chip. Each workspace
    runs its configured `ut_cmd` (override with `ut_cmd`) and reports per-test
    outcomes; results are aligned by test id. A FAIL carries a one-line summary
    inline — fetch full output with `jms_dev_logs`.

    An `ut_cmd` override MUST emit pytest `-rA` short-summary lines
    (`PASSED/FAILED/ERROR/SKIPPED ... <id> - <msg>`): the consistency table is
    built from them, so a command without `-rA` yields no rows and the run is
    reported as not-ok even on a clean pass.

    Long training runs do NOT go here; use `jms_submit_job` / `jms_cluster_submit`.
    """
    from .attest import (
        build_attest_script,
        check_attestation_parity,
        parse_attestation,
    )
    from .workspace import DEFAULT_UT_CMD, build_consistency_table, parse_pytest_summary

    workspaces = _workspaces(targets)
    delivered = await asyncio.gather(*(_deliver(w, ref) for w in workspaces))
    # The early returns carry the *same* keys as the success path (`attestation`, the
    # `attestation_*` trio, `hosts` included) so a caller that reads any of them
    # uniformly never KeyErrors on a delivery failure.
    empty_attest = {
        "attestation": {},
        "attestation_consistent": False,
        "attestation_diverged": [],
        "attestation_unverified": [],
        "hosts": {},
    }
    if not delivered:
        # No workspace matched: report it the same shape as a normal run (empty
        # table) rather than a bare success, so the caller can always read `rows`.
        return {
            "ok": False,
            "stage": "deliver",
            "error": "no workspace matched the selector",
            "ref": ref,
            "columns": [],
            "rows": [],
            "revisions": {},
            "revisions_consistent": False,
            "revisions_unverified": [],
            "failed_hosts": [],
            "results": [],
            **empty_attest,
        }
    if not all(d["ok"] for d in delivered):
        return {
            "ok": False,
            "stage": "deliver",
            "ref": ref,
            "columns": [],
            "rows": [],
            "revisions": {},
            "revisions_consistent": False,
            "revisions_unverified": [d["workspace"] for d in delivered if not d.get("sha")],
            "failed_hosts": [d["workspace"] for d in delivered if not d["ok"]],
            "results": list(delivered),
            **empty_attest,
        }

    async def _run_one(ws):  # noqa: ANN001, ANN202
        host = _host_of(ws)
        # Run the suite from inside the repo: the container's default cwd is not
        # the checkout, so `python -m pytest tests/unit` would not find the tests.
        # `cd --` so a repo path beginning with `-` is not read as a cd option.
        # `set -o pipefail` *inside the subshell* is essential: the inner runs under
        # `docker exec ... bash -lc`, a fresh shell that inherits neither `-e` nor
        # pipefail from the host wrapper, so a piped `ut_cmd` (`... | tee log`) would
        # return tee's 0 and mask a fleet-wide failure.
        # The attestation probe rides the same round trip, *after* the suite. It must
        # never `exit` (it would clobber pytest's status), so the UT run's exit code is
        # captured and re-raised explicitly — otherwise exit_code would always be the
        # probe's and `failed` could never fire. An empty `ut_cmd` (a YAML null override)
        # falls back to the default rather than yielding a `()` syntax error.
        # Activate the container's conda env *before* the repo `cd`, so the UT
        # command and the attestation probe both use the interpreter the built
        # `_C` extension was compiled for (the container default is the wrong ABI
        # on most chips). `env_setup` is caller-controlled spec text, not argv.
        cmd = ut_cmd or ws.ut_cmd or DEFAULT_UT_CMD
        env_prefix = f"{ws.env_setup}; " if ws.env_setup else ""
        inner = (
            env_prefix
            + f"cd -- {shell_quote(ws.repo)} && {{ (set -o pipefail; {cmd}); rc=$?; "
            + build_attest_script(ws.repo)
            + "exit $rc; }"
        )
        res = await _ssh().exec_on(
            host,
            # The UT run must not inherit the delivery proxy.
            ws.container_cmd(inner, with_proxy=False),
            timeout=timeout,
            workdir=False,
            env_setup=False,
        )
        return ws.name, res, parse_pytest_summary(res.stdout)

    async def _one(ws):  # noqa: ANN001, ANN202
        """Run one chip, turning a raise into *that chip's* failure.

        A raise before a result exists (a workspace naming a host the inventory no
        longer has, a transport error) would otherwise escape the fan-out `gather` and
        replace the whole cross-chip table — this tool's entire product — with an
        opaque tool error. Here the chip is reported as failed and the table survives.
        """
        try:
            return await _run_one(ws)
        except Exception as exc:  # noqa: BLE001 - isolating one chip is the point
            return ws.name, ExecResult(ws.name, 1, "", f"run failed: {exc}"), {}

    ran = await asyncio.gather(*(_one(w) for w in workspaces))
    per_ws = {name: parsed for name, _, parsed in ran}
    table = build_consistency_table(per_ws)
    # A chip that timed out or exited nonzero (often with empty stdout, hence no
    # parsed rows) must fail the whole run, not just be reported as informational.
    failed = [name for name, res, _ in ran if not res.ok]
    shas = {d["workspace"]: d["sha"] for d in delivered if d.get("sha")}
    # A workspace whose revision could not be read must not be silently dropped:
    # comparing only the survivors could report agreement while a chip ran
    # unverified code.
    unverified = [d["workspace"] for d in delivered if not d.get("sha")]
    revisions_consistent = len(set(shas.values())) <= 1 and len(shas) == len(delivered)

    # Attestation: the same sha is not the same *code path*. Parse each chip's record
    # out of the UT stdout and require the op->backend map (and plugin version, where
    # readable) to agree. A chip whose map could not be read is `unverified`, never
    # coerced into agreement — the same rule `revisions` follows.
    attestation = {name: parse_attestation(res.stdout) for name, res, _ in ran}
    parity = check_attestation_parity(attestation)

    table.update(
        {
            "ok": (
                table["ok"]
                and not failed
                and revisions_consistent
                and parity["consistent"]
            ),
            "ref": ref,
            "failed_hosts": failed,
            "revisions": shas,
            "revisions_consistent": revisions_consistent,
            "revisions_unverified": unverified,
            "attestation": attestation,
            "attestation_consistent": parity["consistent"],
            "attestation_diverged": parity["diverged"],
            "attestation_unverified": parity["unverified"],
            "hosts": {
                name: {
                    "ok": res.ok,
                    "exit_code": res.exit_code,
                    "stderr": res.stderr.strip()[:400],
                }
                for name, res, _ in ran
            },
        }
    )
    return table


@_tool()
async def jms_dev_logs(workspace: str, test_id: str, timeout: float = 300.0) -> dict:
    """Full output for one `test_id` on one `workspace` (drill into a FAIL).

    Re-runs just that test in the container with a full traceback, so it works for
    any suite without depending on a saved report file.
    """
    ws = _inv().resolve_workspaces([workspace])[0]
    host = _host_of(ws)
    # Activate the container env *before* the repo `cd`, exactly as jms_dev_run does.
    # The whole point of this tool is to say why a test failed on a chip; re-running
    # it under the container's default interpreter (the wrong ABI on most chips)
    # would report the cause of a failure that is not the one the fleet run hit.
    env_prefix = f"{ws.env_setup}; " if ws.env_setup else ""
    inner = (
        env_prefix
        + f"cd -- {shell_quote(ws.repo)} && python -m pytest "
        + f"{shell_quote(test_id)} -q --tb=long -rA"
    )
    res = await _ssh().exec_on(
        host,
        ws.container_cmd(inner, with_proxy=False),
        timeout=timeout,
        workdir=False,
        env_setup=False,
    )
    return {
        "ok": res.ok,
        "workspace": ws.name,
        "test_id": test_id,
        "exit_code": res.exit_code,
        "output": res.stdout[-20000:],
        "stderr": res.stderr.strip()[:2000],
    }


@_tool()
async def jms_dev_discover(
    targets: list[str],
    container: str = DEFAULT_CONTAINER,
    source: str = "git",
    repo_match: str = DEFAULT_REPO_MATCH,
    max_depth: int = 5,
    proxy: str | None = None,
    mount: str | None = None,
    local: str | None = None,
    timeout: float = 120.0,
) -> dict:
    """Inspect each host's container and emit a paste-ready `workspaces:` block.

    This is the front door to the dev loop. `jms_dev_run` needs to know, per chip,
    which container holds the repo, where the checkout lives inside it, and whether
    the container can reach the git remote — and that is exactly the per-host
    busywork worth automating. One call fans out `docker exec` discovery to every
    host and returns the inventory YAML to paste into `hosts.yaml`.

    `container` is normally the same name on every host (`lvyufeng-torch-fl`);
    `source` seeds the `source:` field (git / proxy-git / push) and `repo_match` is
    a case-insensitive substring a checkout path must contain. Because the emitted
    block must be *loadable*, `source='proxy-git'` requires `proxy` and
    `source='push'` requires `mount` + `local` — without them the pasted entry would
    make `Inventory.load` reject the whole inventory, so this call refuses rather
    than emit a block that cannot be pasted.

    Every host's raw findings are in `hosts`; anything the probe could not confirm
    is a `warnings` entry (a timed-out or unreachable origin, a missing pytest, a
    missing docker, a container that is not there) and a host that could not be
    inspected is listed in `skipped` — `ok` is true only when every requested host
    produced at least one workspace.

    The emitted `source` is a *guess* from reachability alone; confirm it before
    relying on it.
    """
    from .discover import (
        build_discovery_script,
        parse_discovery,
        render_workspaces_yaml,
    )

    if not CONTAINER_RE.fullmatch(container):
        return {"ok": False, "error": f"invalid container name {container!r}"}
    if source not in SOURCES:
        return {"ok": False, "error": f"source must be one of {SOURCES}, got {source!r}"}
    # The emitted block is advertised as paste-ready, so it must survive
    # Inventory.load: Workspace rejects proxy-git without a proxy and push without
    # both mount and local, and a single bad entry aborts the whole inventory.
    if source == "proxy-git" and not proxy:
        return {"ok": False, "error": "source 'proxy-git' requires a 'proxy' argument"}
    if source == "push" and not (mount and local):
        return {"ok": False, "error": "source 'push' requires both 'mount' and 'local'"}

    hosts = _resolve(targets)
    script = build_discovery_script(container, repo_match=repo_match, max_depth=max_depth)
    results = await asyncio.gather(
        *(
            # Discovery must not depend on the host toolchain (env_setup) nor the
            # host workdir — it inspects a container, which is host-toolchain-free.
            _ssh().exec_on(
                h, script, timeout=timeout, workdir=False, env_setup=False
            )
            for h in hosts
        )
    )
    reports = [
        {
            "host": res.host,
            "exit_code": res.exit_code,
            "timed_out": res.timed_out,
            "timeout": timeout,
            "report": parse_discovery(res.stdout),
            "stderr": res.stderr.strip()[:400],
        }
        for res in results
    ]
    rendered = render_workspaces_yaml(
        reports,
        container=container,
        source=source,
        proxy=proxy,
        mount=mount,
        local=local,
        host_tags={h.name: h.tags for h in hosts},
    )
    return {
        "ok": rendered["ok"],
        "container": container,
        "source": source,
        "hosts": reports,
        "skipped": rendered["skipped"],
        "warnings": rendered["warnings"],
        "workspaces": rendered["workspaces"],
        "yaml": rendered["yaml"],
    }


# --- cluster: preflight + coordinated launch ---------------------------------------


@_tool()
async def jms_cluster_preflight(targets: list[str], port: int = DEFAULT_RDZV_PORT) -> dict:
    """Check a cluster is ready for a distributed run, host by host.

    For every host: reachability, accelerator visibility and version (via the
    platform probe), whether the rendezvous `port` is already bound (a stale previous
    run), and how many accelerators already hold a compute process. Returns an
    estimated `world_size` and a suggested rendezvous. Run this before
    `jms_cluster_submit`.

    `ok` is false when a host is unreachable, holds the rendezvous port, **or** already
    has accelerators in use — i.e. exactly when a default `jms_cluster_submit` would
    refuse. `busy_devices` is `null` when occupancy is unknown: a platform with no
    occupancy probe, or a host whose driver tool did not answer. Unknown is never
    reported as busy, so a probe gap degrades to the old behaviour instead of blocking
    a launch. Set `allow_busy=true` on the submit to override the busy gate.
    """
    hosts = _resolve(targets)
    health = await preflight(_ssh(), hosts, port=port)
    # Match `jms_cluster_submit` exactly: a reachable host with no device count still gets
    # `nproc=1` there, so it must count 1 here. Counting 0 under-reported the world size
    # the submit would really launch, which is the number this estimate exists to predict.
    ranks = sum(max(1, h.device_count or 1) for h in health if h.reachable)
    rdv = resolve_rendezvous(hosts, port=port)
    busy = [h for h in health if h.busy]
    return {
        "ok": all(h.ok for h in health) and not busy,
        "world_size_estimate": ranks,
        "rendezvous": rdv.to_dict(),
        "heterogeneous": is_heterogeneous(hosts),
        "islands": [i.to_dict() for i in plan_islands(hosts)],
        "hosts": [h.to_dict() for h in health],
        "platforms": sorted({h.platform for h in health}),
        "busy_hosts": [h.host for h in busy],
        "note": (
            f"{len(busy)} host(s) already have accelerators in use; jms_cluster_submit "
            "refuses a set containing them unless you pass allow_busy=true"
        )
        if busy
        else "",
    }


@_tool()
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


@_tool()
async def jms_cluster_submit(
    targets: list[str],
    command: str,
    master: str | None = None,
    port: int = DEFAULT_RDZV_PORT,
    allow_busy: bool = False,
) -> dict:
    """Launch ONE coordinated distributed job across a heterogeneous cluster.

    Unlike `jms_submit_job` (which runs the command independently on each host),
    this assigns every host a rank and gives them a shared rendezvous, exporting
    `JMS_RANK`, `JMS_LOCAL_RANK`, `JMS_NPROC`, `JMS_WORLD_SIZE`, `JMS_MASTER_ADDR`
    and `JMS_MASTER_PORT` before `command` runs. All ranks launch concurrently and
    the call returns immediately with **one** job id, shared by every rank — pass it
    (and the same `targets`) to `jms_job_status` / `jms_job_wait` / `jms_job_logs` /
    `jms_cluster_cancel`.

    `command` is your per-node launcher; wire it with the exported env. For torchrun:

        torchrun --nnodes=$JMS_WORLD_SIZE --node_rank=$JMS_RANK \\
                 --nproc_per_node=$JMS_NPROC \\
                 --rdzv_endpoint=$JMS_MASTER_ADDR:$JMS_MASTER_PORT \\
                 train.py --rounds 100

    `targets` order defines rank order; each host contributes one rank node whose
    `JMS_NPROC` is its accelerator count. Every target is preflighted first, and a host
    that is unreachable, holding a stale rendezvous `port`, or already running work on
    its accelerators fails the call rather than being handed a rank. The busy gate is
    what stops a second training being co-scheduled onto cards that are already in use;
    pass `allow_busy=true` to launch anyway — e.g. to resume a run whose previous job is
    still winding down. Occupancy the probe could not determine is not treated as busy,
    so a missing driver tool never blocks a launch on its own. `jms_cluster_preflight`
    reports the same busy hosts before you get here.
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
    # Refuse before assigning ranks, rather than quietly shrinking the collective. An
    # unreachable host reports no `device_count` yet still contributed `nproc=1` to the
    # world size below, so every launched rank was told a world size larger than the set
    # that could join it — the job hangs in rendezvous, or the survivors silently train a
    # different-width collective than the caller asked for. A host with a stale rendezvous
    # port is equally unlaunchable: its rank cannot bind. Same readiness test as
    # `jms_cluster_preflight`, so the two tools never disagree about a target set.
    unready = [h for h in hosts if not health[h.name].ok]
    if unready:
        return {
            "ok": False,
            "error": (
                f"{len(unready)} of {len(hosts)} targets are not ready to join a collective "
                f"({', '.join(h.name for h in unready)}); assigning a rank to a host that "
                "cannot join leaves every other rank waiting on a rendezvous that never "
                "completes."
            ),
            "unready": [health[h.name].to_dict() for h in unready],
            "ready_targets": [h.name for h in hosts if health[h.name].ok],
            "hint": (
                "fix the hosts above, or submit the ready subset by name — "
                "jms_cluster_submit needs at least 2 hosts"
            ),
        }
    # Busy is a *separate* gate from `ok`, and it is the one that keeps two trainings off
    # the same cards. It stays out of `HostHealth.ok` because the two failures need
    # different answers, and it is overridable because the most common honest reason a
    # card is busy is this caller's own job: refusing outright would make a legitimate
    # relaunch impossible. Unknown occupancy (`busy_devices` is None) is not busy.
    busy = [h for h in hosts if health[h.name].busy]
    if busy and not allow_busy:
        return {
            "ok": False,
            "error": (
                f"{len(busy)} of {len(hosts)} targets already have accelerators in use "
                f"({', '.join(h.name for h in busy)}); launching now would co-schedule a "
                "second training onto occupied cards."
            ),
            "busy": [health[h.name].to_dict() for h in busy],
            "ready_targets": [h.name for h in hosts if not health[h.name].busy],
            "hint": "pass allow_busy=true to launch anyway (e.g. resuming an unfinished job)",
        }
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
        # Every rank shares one job id (`submit_collab`), so it is hoisted out of the
        # per-rank entries and repeated here for convenience.
        "job_id": handles[0].job_id if handles else None,
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