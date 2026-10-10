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
import os
import re
import subprocess
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
from .discover import DEFAULT_REPO_MATCH
from .inventory import Host, Inventory, shell_quote
from .jobs import JobManager
from .ssh import SSHRunner
from .workspace import CONTAINER_RE, DEFAULT_CONTAINER, SOURCES

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


# --- dev loop: workspace sync / run / logs -----------------------------------------


@server.tool()
async def jms_inventory_reload() -> dict:
    """Re-read `hosts.yaml` and drop the cached inventory.

    The advertised dev loop is *emit YAML → paste into `hosts.yaml` → run*, but every
    tool reads the inventory through a cache populated on first use. Without this the
    paste step silently needs an MCP restart before the new hosts/workspaces appear.
    Call this after editing the inventory; the next tool call re-reads from disk.

    Returns the freshly loaded summary so the caller can confirm the paste landed.
    """
    _state.pop("inv", None)
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
    """Put the workspace's repo at ``ref`` using its configured source.

    Returns the delivered commit under ``sha`` (best-effort) so a caller can spot
    revision divergence across chips. For ``push`` the code arrives out-of-band and
    the container has no git, so ``ref_pinned`` is False and the sha — when the
    local tree could be read — is the *local* HEAD, not the container's.
    """
    host = _host_of(ws)
    if ws.source == "push":
        res = await _ssh().rsync(
            host, ws.local, ws.mount, direction="push", excludes=_DEFAULT_EXCLUDES
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


@server.tool()
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


@server.tool()
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

    async def one(ws):  # noqa: ANN001, ANN202
        host = _host_of(ws)
        # Run the suite from inside the repo: the container's default cwd is not
        # the checkout, so `python -m pytest tests/` would not find the tests.
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

    ran = await asyncio.gather(*(one(w) for w in workspaces))
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


@server.tool()
async def jms_dev_logs(workspace: str, test_id: str, timeout: float = 300.0) -> dict:
    """Full output for one `test_id` on one `workspace` (drill into a FAIL).

    Re-runs just that test in the container with a full traceback, so it works for
    any suite without depending on a saved report file.
    """
    ws = _inv().resolve_workspaces([workspace])[0]
    host = _host_of(ws)
    inner = (
        f"cd -- {shell_quote(ws.repo)} && python -m pytest "
        f"{shell_quote(test_id)} -q --tb=long -rA"
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


@server.tool()
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