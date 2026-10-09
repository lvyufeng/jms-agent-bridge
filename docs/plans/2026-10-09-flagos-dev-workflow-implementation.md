# flagos dev workflow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a single-action "write once, verify on every chip" development loop to jms-agent-bridge: deliver a git ref to a uniform container on every host, run the Torch-FL UT suite there, and return one cross-chip consistency table.

**Architecture:** A new `Workspace` model (host + container + repo + delivery source + UT command) parsed from the existing `hosts.yaml` under a new `workspaces:` key. Every workspace command runs as `docker exec <container> bash -lc '<cmd>'` over the existing SSH data plane. Three delivery sources (`git`, `proxy-git`, `push`) collapse to one verb. Three new MCP tools: `jms_dev_sync`, `jms_dev_run`, `jms_dev_logs`.

**Tech Stack:** Python 3.10+, asyncio, `mcp>=2.0` (MCPServer), PyYAML, pytest.

**Design doc:** `docs/plans/2026-10-09-flagos-dev-workflow-design.md`

---

## Notes for the implementer

- The repo already has a live SSH data plane (`SSHRunner.exec_on`) and inventory
  (`Inventory.load`). Read `src/jms_bridge/inventory.py` and `src/jms_bridge/ssh.py`
  before starting; match their style (dataclasses, `shell_quote`, `wrap_command`).
- Tests are plain pytest, no network. Follow `tests/test_inventory.py` for the
  `write_inventory(tmp_path, body)` + `textwrap.dedent` pattern.
- Run everything with the project venv: `.venv/bin/python -m pytest -q`.
  Lint with `.venv/bin/ruff check src tests`.
- Commit after each task. Small commits, conventional-commit subjects.

---

### Task 1: Workspace model

**Files:**
- Create: `src/jms_bridge/workspace.py`
- Test: `tests/test_workspace.py`

**Step 1: Write the failing test**

```python
import pytest

from jms_bridge.workspace import Workspace, WorkspaceError


def test_workspace_defaults():
    w = Workspace(name="w1", host="a", container="torchfl", repo="/work/Torch-FL")
    assert w.source == "git"
    assert w.container_cmd("echo hi") == "docker exec torchfl bash -lc 'echo hi'"


def test_workspace_proxy_git_wraps_proxy():
    w = Workspace(
        name="w1", host="a", container="torchfl", repo="/work/Torch-FL",
        source="proxy-git", proxy="http://proxy:8080",
    )
    cmd = w.container_cmd("git fetch")
    assert "http_proxy='http://proxy:8080'" in cmd
    assert "https_proxy='http://proxy:8080'" in cmd


def test_workspace_bad_source_raises():
    with pytest.raises(WorkspaceError):
        Workspace(name="w", host="a", container="c", repo="/r", source="ftp")
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_workspace.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'jms_bridge.workspace'`

**Step 3: Write minimal implementation**

```python
"""Workspaces: the flagos development unit.

A Workspace binds a host (reached through the bastion) to a *uniform* container on
that host, the flagos repo inside it, how code is delivered, and how the UT suite is
run. Commands in a workspace are wrapped as ``docker exec <container> bash -lc ...`` so
the host is only a carrier — development happens in the container, the same way on
every chip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .inventory import shell_quote

SOURCES = ("git", "proxy-git", "push")


class WorkspaceError(RuntimeError):
    """Raised when a workspace is missing required fields or misconfigured."""


@dataclass
class Workspace:
    name: str
    host: str
    container: str
    repo: str
    source: str = "git"
    proxy: str | None = None
    ut_cmd: str = "python -m pytest tests/ -q --tb=line -rA"
    #: Host directory bind-mounted to ``repo`` (needed for source="push").
    mount: str | None = None
    #: Local checkout to rsync from (needed for source="push").
    local: str | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise WorkspaceError(
                f"workspace {self.name!r}: source must be one of {SOURCES}, got {self.source!r}"
            )
        if self.source == "proxy-git" and not self.proxy:
            raise WorkspaceError(f"workspace {self.name!r}: proxy-git requires 'proxy'")
        if self.source == "push" and not (self.mount and self.local):
            raise WorkspaceError(f"workspace {self.name!r}: push requires 'mount' and 'local'")

    def container_cmd(self, command: str) -> str:
        """Wrap a shell command to run inside the workspace container.

        For ``proxy-git`` the proxy env is injected inside the container so the same
        verb works whether or not the host reaches GitHub directly.
        """
        inner = command
        if self.source == "proxy-git" and self.proxy:
            inner = (
                f"export http_proxy={shell_quote(self.proxy)} "
                f"https_proxy={shell_quote(self.proxy)}; {command}"
            )
        return f"docker exec {shell_quote(self.container)} bash -lc {shell_quote(inner)}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "host": self.host,
            "container": self.container,
            "repo": self.repo,
            "source": self.source,
            "ut_cmd": self.ut_cmd,
            "tags": list(self.tags),
        }
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_workspace.py -q`
Expected: PASS (3 passed)

**Step 5: Commit**

```bash
git add src/jms_bridge/workspace.py tests/test_workspace.py
git commit -m "feat(workspace): add the Workspace model with container command wrapping"
```

---

### Task 2: Parse `workspaces:` from the inventory

**Files:**
- Modify: `src/jms_bridge/inventory.py` (dataclass `Inventory`, classmethod `load`)
- Test: `tests/test_inventory.py` (append)

**Step 1: Write the failing test**

```python
def test_load_workspaces(tmp_path):
    p = write_inventory(
        tmp_path,
        """
        hosts:
          a: {login: u, account: root, asset_id: A1, platform: ascend-910c}
        workspaces:
          fl-ascend:
            host: a
            container: torchfl
            repo: /work/Torch-FL
            source: git
            tags: [ascend]
        """,
    )
    inv = Inventory.load(p)
    assert set(inv.workspaces) == {"fl-ascend"}
    w = inv.workspaces["fl-ascend"]
    assert w.host == "a" and w.container == "torchfl"


def test_workspace_referencing_unknown_host_raises(tmp_path):
    p = write_inventory(
        tmp_path,
        "hosts:\n  a: {login: u}\nworkspaces:\n  w: {host: ghost, container: c, repo: /r}\n",
    )
    with pytest.raises(InventoryError):
        Inventory.load(p)


def test_resolve_workspaces_by_name_and_tag(tmp_path):
    p = write_inventory(
        tmp_path,
        """
        hosts:
          a: {login: u}
          b: {login: u}
        workspaces:
          wa: {host: a, container: c, repo: /r, tags: [ascend]}
          wb: {host: b, container: c, repo: /r, tags: [cuda]}
        """,
    )
    inv = Inventory.load(p)
    assert [w.name for w in inv.resolve_workspaces(["ascend"])] == ["wa"]
    assert {w.name for w in inv.resolve_workspaces(["all"])} == {"wa", "wb"}
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_inventory.py -q -k workspace`
Expected: FAIL — `Inventory` has no `workspaces`

**Step 3: Write minimal implementation**

In `inventory.py`: add `from .workspace import Workspace` (guard against circular import by
importing inside `load`, since `workspace.py` imports `shell_quote` from this module — move
the import to the function body to break the cycle).

Add to the `Inventory` dataclass:

```python
    workspaces: dict[str, Workspace] = field(default_factory=dict)

    def resolve_workspaces(self, selectors: list[str]) -> list[Workspace]:
        """Resolve names / tags / 'all' into a unique, order-stable workspace list."""
        picked: dict[str, Workspace] = {}
        for sel in selectors:
            if sel in ("all", "*"):
                picked.update(self.workspaces)
            elif sel in self.workspaces:
                picked[sel] = self.workspaces[sel]
            else:
                tagged = [w for w in self.workspaces.values() if sel in w.tags]
                if not tagged:
                    raise InventoryError(
                        f"selector {sel!r} matched no workspace or tag "
                        f"(known: {sorted(self.workspaces)})"
                    )
                for w in tagged:
                    picked[w.name] = w
        return list(picked.values())
```

In `load()`, after building `hosts` and before `return`:

```python
        workspaces: dict[str, Workspace] = {}
        for name, spec in (data.get("workspaces") or {}).items():
            spec = spec or {}
            if spec.get("host") not in hosts:
                raise InventoryError(
                    f"workspace {name!r} references unknown host {spec.get('host')!r}"
                )
            workspaces[name] = Workspace(
                name=name,
                host=spec["host"],
                container=spec.get("container", "torchfl"),
                repo=spec["repo"],
                source=spec.get("source", "git"),
                proxy=spec.get("proxy"),
                ut_cmd=spec.get("ut_cmd", "python -m pytest tests/ -q --tb=line -rA"),
                mount=spec.get("mount"),
                local=spec.get("local"),
                tags=list(spec.get("tags") or []),
            )
```

and `return cls(hosts=hosts, groups=groups, workspaces=workspaces)`. Wrap `Workspace(...)`
in a `try/except WorkspaceError` that re-raises as `InventoryError`.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_inventory.py -q`
Expected: PASS (all, old + new)

**Step 5: Commit**

```bash
git add src/jms_bridge/inventory.py tests/test_inventory.py
git commit -m "feat(inventory): parse and resolve workspaces"
```

---

### Task 3: Parse pytest output into per-test outcomes

**Files:**
- Modify: `src/jms_bridge/workspace.py`
- Test: `tests/test_workspace.py` (append)

**Step 1: Write the failing test**

```python
from jms_bridge.workspace import parse_pytest_summary

SAMPLE = """\
=================== short test summary info ===================
PASSED tests/test_ops.py::test_softmax
FAILED tests/test_ops.py::test_layer_norm - AssertionError: max diff 1e-3
SKIPPED [1] tests/test_ops.py::test_gemm_dtype: not supported on this backend
ERROR tests/test_ops.py::test_bad_setup - RuntimeError: boom
1 failed, 1 passed, 1 skipped, 1 error
"""


def test_parse_pytest_summary():
    got = parse_pytest_summary(SAMPLE)
    assert got["tests/test_ops.py::test_softmax"]["outcome"] == "PASS"
    assert got["tests/test_ops.py::test_layer_norm"]["outcome"] == "FAIL"
    assert "max diff" in got["tests/test_ops.py::test_layer_norm"]["message"]
    assert got["tests/test_ops.py::test_gemm_dtype"]["outcome"] == "SKIP"
    assert got["tests/test_ops.py::test_bad_setup"]["outcome"] == "ERROR"


def test_parse_pytest_summary_empty():
    assert parse_pytest_summary("no summary here") == {}
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_workspace.py -q -k parse`
Expected: FAIL — no `parse_pytest_summary`

**Step 3: Write minimal implementation**

```python
import re

_PYTEST_LINE = re.compile(
    r"^(?P<status>PASSED|FAILED|ERROR|SKIPPED)\s+"
    r"(?:\[\d+\]\s+)?"              # SKIPPED carries a "[n]" count
    r"(?P<test>\S+?)"
    r"(?:\s+-\s+(?P<dash>.*)|\s*:\s*(?P<colon>.*))?$"
)
_STATUS = {"PASSED": "PASS", "FAILED": "FAIL", "ERROR": "ERROR", "SKIPPED": "SKIP"}


def parse_pytest_summary(output: str) -> dict[str, dict]:
    """Map ``test_id -> {outcome, message}`` from pytest ``-rA`` short summary lines."""
    results: dict[str, dict] = {}
    for line in output.splitlines():
        m = _PYTEST_LINE.match(line.strip())
        if not m:
            continue
        message = (m.group("dash") or m.group("colon") or "").strip()
        results[m.group("test")] = {"outcome": _STATUS[m.group("status")], "message": message}
    return results
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_workspace.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/jms_bridge/workspace.py tests/test_workspace.py
git commit -m "feat(workspace): parse pytest -rA output into per-test outcomes"
```

---

### Task 4: The consistency table

**Files:**
- Modify: `src/jms_bridge/workspace.py`
- Test: `tests/test_workspace.py` (append)

**Step 1: Write the failing test**

```python
from jms_bridge.workspace import build_consistency_table


def test_table_aligns_by_test_id_and_flags_divergence():
    per_ws = {
        "ascend": {"t_a": {"outcome": "PASS"}, "t_b": {"outcome": "FAIL", "message": "x"}},
        "cuda": {"t_a": {"outcome": "PASS"}, "t_b": {"outcome": "PASS"}},
        "maca": {"t_a": {"outcome": "SKIP"}},
    }
    table = build_consistency_table(per_ws)
    rows = {r["test_id"]: r for r in table["rows"]}
    assert rows["t_a"]["consistent"] is False          # PASS/PASS/SKIP
    assert rows["t_b"]["consistent"] is False          # FAIL vs PASS
    assert rows["t_b"]["outcomes"]["ascend"]["message"] == "x"
    assert table["columns"] == ["ascend", "cuda", "maca"]
    assert table["ok"] is False
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_workspace.py -q -k table`
Expected: FAIL — no `build_consistency_table`

**Step 3: Write minimal implementation**

```python
def build_consistency_table(per_workspace: dict[str, dict[str, dict]]) -> dict:
    """Align per-workspace outcomes by test id into one cross-chip table.

    A test is ``consistent`` when every workspace that reported it agrees on the
    outcome. Workspaces that never reported a test show ``outcome=None`` — a test a
    chip did not run is visible, not silently dropped.
    """
    columns = list(per_workspace)
    test_ids: list[str] = []
    for results in per_workspace.values():
        for tid in results:
            if tid not in test_ids:
                test_ids.append(tid)

    rows = []
    for tid in test_ids:
        cells = {ws: per_workspace[ws].get(tid) for ws in columns}
        outcomes = {c["outcome"] for c in cells.values() if c}
        rows.append(
            {
                "test_id": tid,
                "outcomes": cells,
                "consistent": len(outcomes) <= 1 and all(c is not None for c in cells.values()),
            }
        )
    return {
        "ok": all(r["consistent"] for r in rows),
        "columns": columns,
        "rows": rows,
    }
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_workspace.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/jms_bridge/workspace.py tests/test_workspace.py
git commit -m "feat(workspace): build the cross-chip consistency table"
```

---

### Task 5: Delivery — resolve a workspace to a host and deliver a ref

**Files:**
- Modify: `src/jms_bridge/workspace.py`
- Test: `tests/test_workspace.py` (append)

**Step 1: Write the failing test** (pure command construction, no SSH)

```python
def test_delivery_commands_per_source():
    g = Workspace(name="g", host="a", container="c", repo="/r", source="git")
    assert "git fetch origin" in g.delivery_cmd("feat/x")
    assert "git checkout 'feat/x'" in g.delivery_cmd("feat/x")

    p = Workspace(name="p", host="a", container="c", repo="/r",
                  source="proxy-git", proxy="http://prx:80")
    assert "http_proxy='http://prx:80'" in p.delivery_cmd("feat/x")

    # push mode delivers nothing inside the container (host bind-mount already updated)
    u = Workspace(name="u", host="a", container="c", repo="/r",
                  source="push", mount="/srv/fl", local="/Users/me/Torch-FL")
    assert u.delivery_cmd("feat/x") is None
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_workspace.py -q -k delivery`
Expected: FAIL — no `delivery_cmd`

**Step 3: Write minimal implementation**

```python
    def delivery_cmd(self, ref: str) -> str | None:
        """The in-container command that puts ``repo`` at ``ref``, or None for push.

        ``push`` works by rsync into the host ``mount`` dir (which is bind-mounted to
        ``repo``), so nothing runs inside the container to deliver code.
        """
        if self.source == "push":
            return None
        r = shell_quote(self.repo)
        return (
            f"cd {r} && git fetch --depth=1 origin {shell_quote(ref)} && "
            f"git checkout {shell_quote(ref)}"
        )
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_workspace.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/jms_bridge/workspace.py tests/test_workspace.py
git commit -m "feat(workspace): resolve delivery per source"
```

---

### Task 6: `jms_dev_sync` MCP tool

**Files:**
- Modify: `src/jms_bridge/server.py` (add after the job tools, ~line 300)
- Test: `tests/test_server.py` (append to the expected-tools set)

**Step 1: Write the failing test**

Add the three new names to `expected` in `tests/test_server.py::test_mcp_server_lists_all_tools`:

```python
        "jms_dev_sync", "jms_dev_run", "jms_dev_logs",
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_server.py -q`
Expected: FAIL — missing tools

**Step 3: Write minimal implementation**

Add helpers + tool to `server.py`:

```python
def _workspaces(selectors: list[str]) -> list:
    return _inv().resolve_workspaces(selectors)


def _host_of(ws):  # noqa: ANN001
    return _inv().hosts[ws.host]


async def _deliver(ws, ref: str) -> dict:  # noqa: ANN001
    """Put the workspace's repo at ``ref`` using its configured source."""
    host = _host_of(ws)
    if ws.source == "push":
        res = await _ssh().rsync(host, ws.local, ws.mount, direction="push",
                                 excludes=_DEFAULT_EXCLUDES)
        return {"workspace": ws.name, "source": "push", "ok": res.ok,
                "stderr": res.stderr.strip()[:400]}
    cmd = ws.delivery_cmd(ref)
    res = await _ssh().exec_on(host, ws.container_cmd(cmd), timeout=300, workdir=False)
    return {"workspace": ws.name, "source": ws.source, "ok": res.ok,
            "stderr": res.stderr.strip()[:400]}


@server.tool()
async def jms_dev_sync(targets: list[str], ref: str) -> dict:
    """Deliver a git `ref` to every selected workspace's container.

    Each workspace uses its configured source (git / proxy-git / push), so hosts with
    direct GitHub access, hosts behind a proxy, and offline hosts all work from this
    one call. `ref` is a branch on the workspace's `origin` (the developer's fork).
    """
    workspaces = _workspaces(targets)
    results = await asyncio.gather(*(_deliver(w, ref) for w in workspaces))
    return {"ok": all(r["ok"] for r in results), "ref": ref, "results": list(results)}
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_server.py -q`
Expected: `jms_dev_sync` present (test still fails on `jms_dev_run`/`jms_dev_logs` — add
stubs in the next tasks, or implement Task 7/8 before re-running)

**Step 5: Commit**

```bash
git add src/jms_bridge/server.py tests/test_server.py
git commit -m "feat(server): add jms_dev_sync to deliver a ref to workspaces"
```

---

### Task 7: `jms_dev_run` MCP tool — sync + run UT + consistency table

**Files:**
- Modify: `src/jms_bridge/server.py`
- Test: `tests/test_workspace.py` (already covers the table; server test covers registration)

**Step 1: Write the failing test**

Covered by the tool-listing test from Task 6 (`jms_dev_run` must be present). Run:

`.venv/bin/python -m pytest tests/test_server.py -q`
Expected: FAIL — `jms_dev_run` missing

**Step 2: Implement**

```python
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

    Long training runs do NOT go here; use `jms_submit_job` / `jms_cluster_submit`.
    """
    from .workspace import build_consistency_table, parse_pytest_summary

    workspaces = _workspaces(targets)
    delivered = await asyncio.gather(*(_deliver(w, ref) for w in workspaces))
    if not all(d["ok"] for d in delivered):
        return {"ok": False, "stage": "deliver", "ref": ref, "results": list(delivered)}

    async def one(ws):  # noqa: ANN001
        host = _host_of(ws)
        res = await _ssh().exec_on(
            host, ws.container_cmd(ut_cmd or ws.ut_cmd), timeout=timeout, workdir=False
        )
        return ws.name, res, parse_pytest_summary(res.stdout)

    ran = await asyncio.gather(*(one(w) for w in workspaces))
    per_ws = {name: parsed for name, _, parsed in ran}
    table = build_consistency_table(per_ws)
    table.update({
        "ref": ref,
        "hosts": {
            name: {"ok": res.ok, "exit_code": res.exit_code,
                   "stderr": res.stderr.strip()[:400]}
            for name, res, _ in ran
        },
    })
    return table
```

**Step 3: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_server.py -q`
Expected: still FAIL on `jms_dev_logs` only

**Step 4: Commit**

```bash
git add src/jms_bridge/server.py
git commit -m "feat(server): add jms_dev_run for the cross-chip UT loop"
```

---

### Task 8: `jms_dev_logs` MCP tool — drill into one failing test

**Files:**
- Modify: `src/jms_bridge/server.py`

**Step 1 & 2: Failing test** — the tool-listing test still fails on `jms_dev_logs`.

**Step 3: Implement**

The UT command must leave full output retrievable. Standardize by re-running the single
test with `-q --tb=long` on demand (simple, no report files to manage):

```python
@server.tool()
async def jms_dev_logs(workspace: str, test_id: str, timeout: float = 300.0) -> dict:
    """Full output for one `test_id` on one `workspace` (drill into a FAIL).

    Re-runs just that test in the container with a full traceback, so it works for any
    suite without depending on a saved report file.
    """
    ws = _inv().resolve_workspaces([workspace])[0]
    host = _host_of(ws)
    cmd = ws.container_cmd(f"cd {shell_quote(ws.repo)} && python -m pytest "
                           f"{shell_quote(test_id)} -q --tb=long -rA")
    res = await _ssh().exec_on(host, cmd, timeout=timeout, workdir=False)
    return {"ok": res.ok, "workspace": ws.name, "test_id": test_id,
            "exit_code": res.exit_code,
            "output": res.stdout[-20000:], "stderr": res.stderr.strip()[:2000]}
```

Add `from .inventory import shell_quote` if not already imported in `server.py`.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_server.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/jms_bridge/server.py
git commit -m "feat(server): add jms_dev_logs to drill into a failing test"
```

---

### Task 9: Document the workflow

**Files:**
- Modify: `README.md` (Tools table + a "flagos development loop" section)
- Modify: `examples/hosts.yaml` (add a `workspaces:` example)

**Step 1: Add to the README Tools table**

Rows for `jms_dev_sync`, `jms_dev_run`, `jms_dev_logs` under a "Dev loop" heading, plus a
short section showing the loop:

```
jms_dev_sync(["flagos-cluster"], ref="feat/unify-softmax")
jms_dev_run(["flagos-cluster"], ref="feat/unify-softmax")   # → consistency table
jms_dev_logs("fl-ascend", "tests/test_ops.py::test_layer_norm")
```

**Step 2: Add a `workspaces:` block to `examples/hosts.yaml`** (generic values only, no
real hosts/ids).

**Step 3: Lint + full suite**

Run: `.venv/bin/ruff check src tests && .venv/bin/python -m pytest -q`
Expected: all pass

**Step 4: Commit**

```bash
git add README.md examples/hosts.yaml
git commit -m "docs: document the flagos development loop"
```

---

### Task 10: Live smoke test on one 910C node

**Not TDD** — this validates against real hardware.

**Step 1:** Confirm the container name on `ascend-910c-108` (assumed `torchfl`):

```
jms_exec(["ascend-910c-108"], "docker ps --format '{{.Names}}'")
```

**Step 2:** Write a real `workspaces:` entry into `~/.config/jms-bridge/hosts.yaml` with
the confirmed container name and the repo path inside it.

**Step 3:** Run `jms_dev_run(["fl-ascend"], ref="<an-existing-branch>")` and confirm the
table comes back with per-test rows, not an error.

**Step 4:** Record what was learned (container name, repo path, UT duration vs the 200s
TTFB) back into the design doc's "Open questions" section, and commit.