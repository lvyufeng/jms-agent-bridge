import asyncio

import jms_bridge.server as S
from jms_bridge.inventory import Host, Inventory
from jms_bridge.server import _looks_dangerous
from jms_bridge.ssh import ExecResult
from jms_bridge.workspace import Workspace


def test_denylist_catches_obvious_destruction():
    assert _looks_dangerous("rm -rf /")
    assert _looks_dangerous("mkfs.ext4 /dev/sda1")
    assert _looks_dangerous("dd if=/dev/zero of=/dev/sda")
    assert _looks_dangerous("shutdown -h now")
    assert _looks_dangerous(":(){:|:&};:")


def test_denylist_normalises_whitespace():
    assert _looks_dangerous("rm    -rf    /")


def test_denylist_allows_normal_work():
    assert not _looks_dangerous("nvidia-smi")
    assert not _looks_dangerous("python train.py --epochs 3")
    assert not _looks_dangerous("npu-smi info")
    assert not _looks_dangerous("rm -rf ./build")
    assert not _looks_dangerous("git status")

def test_mcp_server_lists_all_tools():
    """The MCP surface is stable: a client sees every documented tool."""
    import asyncio
    import os
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    expected = {
        "jms_list_assets", "jms_audit_query", "jms_create_ticket",
        "jms_probe", "jms_exec", "jms_sync_push", "jms_sync_pull",
        "jms_submit_job", "jms_job_status", "jms_job_logs", "jms_job_cancel",
        "jms_cluster_preflight", "jms_cluster_plan", "jms_cluster_submit",
        "jms_dev_sync", "jms_dev_run", "jms_dev_logs",
    }

    async def run():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "jms_bridge.cli", "serve"],
            env={**os.environ},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return {t.name for t in tools.tools}

    names = asyncio.run(run())
    assert expected <= names, f"missing tools: {expected - names}"


# --- jms_dev_run unit coverage (fakes; no SSH, no hardware) -------------------------


class _FakeSSH:
    """A tiny SSHRunner stand-in driven by a per-workspace script of responses.

    ``plan`` maps a substring of the command to an :class:`ExecResult`-shaped
    ``(exit_code, stdout, stderr)`` tuple, so a test declares what each chip's
    delivery and UT run produce.
    """

    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    async def exec_on(self, host, command, *, timeout=60.0, workdir=True, env_setup=True):
        self.calls.append({"host": host.name, "command": command,
                           "workdir": workdir, "env_setup": env_setup})
        code, out, err = 0, "", ""
        for key, value in self.plan.items():
            if key in command:
                code, out, err = value
                break
        return ExecResult(host.name, code, out, err)

    async def rsync(self, host, local, remote, **kw):
        self.calls.append({"host": host.name, "rsync": (local, remote)})
        return ExecResult(host.name, 0, "", "")


def _install(monkeypatch, workspaces, plan):
    """Point the server's lazy singletons at fakes and return the fake runner."""
    inv = Inventory(
        hosts={w.host: Host(name=w.host, login="u", asset_id="A") for w in workspaces},
        groups={},
        workspaces={w.name: w for w in workspaces},
    )
    fake = _FakeSSH(plan)
    monkeypatch.setitem(S._state, "inv", inv)
    monkeypatch.setitem(S._state, "ssh", fake)
    return fake


def _ws(name="w1", host="h1"):
    return Workspace(name=name, host=host, container="c", repo="/work/Torch-FL", source="git")


def test_dev_run_all_pass_is_ok(monkeypatch):
    fake = _install(monkeypatch, [_ws("a", "h1"), _ws("b", "h2")], {
        "git fetch": (0, "abc1234\n", ""),
        "pytest": (0, "PASSED tests/t.py::test_x\n1 passed\n", ""),
    })
    r = asyncio.run(S.jms_dev_run(["all"], "main"))
    assert r["ok"] is True
    assert r["revisions_consistent"] is True
    assert r["failed_hosts"] == []
    # the UT run must not source the host env, and must be proxyless
    run_calls = [c for c in fake.calls if "pytest" in c["command"]]
    assert run_calls and all(c["env_setup"] is False for c in run_calls)


def test_dev_run_nonzero_exit_is_not_ok(monkeypatch):
    # Chip ``b`` times out / exits nonzero with empty stdout -> no parsed rows, but
    # the run is still not ok.
    _install(monkeypatch, [_ws("a", "h1"), _ws("b", "h2")], {
        "git fetch": (0, "abc1234\n", ""),
        "pytest": (124, "", "timed out"),
    })
    r = asyncio.run(S.jms_dev_run(["all"], "main"))
    assert r["ok"] is False
    assert r["failed_hosts"] == ["a", "b"]


def test_dev_run_git_chip_without_sha_fails_delivery(monkeypatch):
    # A git chip whose rev-parse stdout carries no sha-shaped line (banner, empty)
    # cannot prove which commit landed, so its delivery is a failure.
    class _NoShaSSH(_FakeSSH):
        async def exec_on(self, host, command, *, timeout=60.0, workdir=True, env_setup=True):
            self.calls.append({"host": host.name, "command": command})
            if "git fetch" in command:
                return ExecResult(host.name, 0, "Welcome to the container\n", "")
            return ExecResult(host.name, 0, "PASSED tests/t.py::test_x\n", "")

    inv = Inventory(
        hosts={"h1": Host(name="h1", login="u", asset_id="A"),
               "h2": Host(name="h2", login="u", asset_id="A")},
        groups={},
        workspaces={"a": _ws("a", "h1"), "b": _ws("b", "h2")},
    )
    monkeypatch.setitem(S._state, "inv", inv)
    monkeypatch.setitem(S._state, "ssh", _NoShaSSH({}))
    r = asyncio.run(S.jms_dev_run(["all"], "main"))
    assert r["ok"] is False
    assert r["stage"] == "deliver"
    # The failure shape stays readable by an agent that keys on `rows`.
    for key in ("rows", "columns", "revisions", "failed_hosts"):
        assert key in r
    assert r["failed_hosts"] == ["a", "b"]


def test_dev_sync_empty_selector_is_not_ok(monkeypatch):
    # Delivering to zero workspaces is not success (all([]) is True).
    monkeypatch.setitem(S._state, "inv", Inventory(hosts={}, groups={}, workspaces={}))
    monkeypatch.setitem(S._state, "ssh", _FakeSSH({}))
    r = asyncio.run(S.jms_dev_sync(["all"], "main"))
    assert r["ok"] is False
    assert r["results"] == []


def test_local_sha_expands_user(monkeypatch, tmp_path):
    # `local:` is commonly `~/...` and `_local_sha` execs git directly (no shell),
    # so it must expanduser or every push chip lands in revisions_unverified.
    import subprocess

    tree = tmp_path / "tree"
    tree.mkdir()
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    subprocess.run(["git", "-C", str(tree), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tree), "config", "user.name", "t"], check=True)
    (tree / "f").write_text("x")
    subprocess.run(["git", "-C", str(tree), "add", "f"], check=True)
    subprocess.run(["git", "-C", str(tree), "commit", "-qm", "init"], check=True)

    monkeypatch.setenv("HOME", str(tmp_path))
    sha = S._local_sha("~/tree")
    assert sha is not None
    assert S._SHA_RE.match(sha)


def test_dev_run_push_without_local_sha_is_unverified(monkeypatch):
    # A push chip delivers the local tree and cannot report an in-container sha;
    # when the local tree's sha cannot be read either, that chip must not be
    # silently dropped from the revision comparison.
    push = Workspace(
        name="p", host="h2", container="c", repo="/work/Torch-FL", source="push",
        mount="/srv/fl", local="/nonexistent/local/tree",
    )
    inv = Inventory(
        hosts={"h1": Host(name="h1", login="u", asset_id="A"),
               "h2": Host(name="h2", login="u", asset_id="A")},
        groups={},
        workspaces={"g": _ws("g", "h1"), "p": push},
    )
    fake = _FakeSSH({
        "git fetch": (0, "abc1234\n", ""),
        "pytest": (0, "PASSED tests/t.py::test_x\n", ""),
    })
    monkeypatch.setitem(S._state, "inv", inv)
    monkeypatch.setitem(S._state, "ssh", fake)
    r = asyncio.run(S.jms_dev_run(["all"], "main"))
    assert r["revisions_unverified"] == ["p"]
    assert r["revisions_consistent"] is False
    assert r["ok"] is False


def test_dev_run_banner_after_sha_is_ignored(monkeypatch):
    # A banner printed *after* the sha must not become a fake sha: we take the last
    # sha-shaped line, so both chips agree on the real commit.
    class _BannerSSH(_FakeSSH):
        async def exec_on(self, host, command, *, timeout=60.0, workdir=True, env_setup=True):
            self.calls.append({"host": host.name, "command": command})
            if "git fetch" in command:
                return ExecResult(host.name, 0, "abc1234\nGoodbye, see you soon!\n", "")
            return ExecResult(host.name, 0, "PASSED tests/t.py::test_x\n", "")

    inv = Inventory(
        hosts={"h1": Host(name="h1", login="u", asset_id="A"),
               "h2": Host(name="h2", login="u", asset_id="A")},
        groups={},
        workspaces={"a": _ws("a", "h1"), "b": _ws("b", "h2")},
    )
    monkeypatch.setitem(S._state, "inv", inv)
    monkeypatch.setitem(S._state, "ssh", _BannerSSH({}))
    r = asyncio.run(S.jms_dev_run(["all"], "main"))
    assert r["revisions"] == {"a": "abc1234", "b": "abc1234"}
    assert r["ok"] is True


def test_dev_run_mixed_revisions_is_not_ok(monkeypatch):
    # Two chips, one delivery pins a different commit -> not consistent even though
    # every parsed outcome agrees and both exited zero.
    def plan_for(cmd):
        return (0, "PASSED tests/t.py::test_x\n1 passed\n", "")

    class _RevSSH(_FakeSSH):
        async def exec_on(self, host, command, *, timeout=60.0, workdir=True, env_setup=True):
            self.calls.append({"host": host.name, "command": command})
            if "git fetch" in command:
                sha = "abc1234" if host.name == "h1" else "def4567"
                return ExecResult(host.name, 0, sha + "\n", "")
            return ExecResult(host.name, 0, plan_for(command)[1], "")

    inv = Inventory(
        hosts={"h1": Host(name="h1", login="u", asset_id="A"),
               "h2": Host(name="h2", login="u", asset_id="A")},
        groups={},
        workspaces={"a": _ws("a", "h1"), "b": _ws("b", "h2")},
    )
    monkeypatch.setitem(S._state, "inv", inv)
    monkeypatch.setitem(S._state, "ssh", _RevSSH({}))
    r = asyncio.run(S.jms_dev_run(["all"], "main"))
    assert r["revisions"] == {"a": "abc1234", "b": "def4567"}
    assert r["revisions_consistent"] is False
    assert r["ok"] is False
