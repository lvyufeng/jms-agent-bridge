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
        "jms_dev_sync", "jms_dev_run", "jms_dev_logs", "jms_dev_discover",
        "jms_inventory_reload",
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
        # Mirror the real runner: a timeout surfaces as exit 124 + timed_out=True.
        return ExecResult(host.name, code, out, err, timed_out=(code == 124))

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


#: A UT stdout that also carries a *consistent* attestation record, so the default
#: fixture behaves like a fleet of chips actually running the same code path.
_UT_OK = (
    "PASSED tests/t.py::test_x\n"
    "1 passed\n"
    "JMSA backends_read=1\n"
    "JMSA backend=test_x = flaggems\n"
)


def test_dev_run_all_pass_is_ok(monkeypatch):
    fake = _install(monkeypatch, [_ws("a", "h1"), _ws("b", "h2")], {
        "git fetch": (0, "abc1234\n", ""),
        "pytest": (0, _UT_OK, ""),
    })
    r = asyncio.run(S.jms_dev_run(["all"], "main"))
    assert r["ok"] is True
    assert r["revisions_consistent"] is True
    assert r["failed_hosts"] == []
    # the UT run must not source the host env, and must be proxyless
    run_calls = [c for c in fake.calls if "pytest" in c["command"]]
    assert run_calls and all(c["env_setup"] is False for c in run_calls)


def test_dev_run_command_carries_the_probe_and_preserves_exit(monkeypatch):
    # The core wiring: the composed container command must (a) actually include the
    # attestation probe, (b) capture the UT exit status and re-raise it, and (c) not
    # let a repo path that begins with '-' be read as a `cd` option.
    fake = _install(monkeypatch, [_ws("a", "h1")], {
        "git fetch": (0, "abc1234\n", ""),
        "pytest": (0, _UT_OK, ""),
    })
    asyncio.run(S.jms_dev_run(["all"], "main"))
    ut = next(c["command"] for c in fake.calls if "pytest" in c["command"])
    assert "set +eu +o pipefail" in ut           # the probe is present
    assert "rc=$?" in ut and "exit $rc" in ut    # exit status is preserved
    assert "set -o pipefail" in ut               # piped ut_cmd cannot mask a failure
    assert "cd -- " in ut and "/work/Torch-FL" in ut  # end-of-options marker + repo
    # the probe reads the build identity / code-path map, i.e. it is the real script
    assert "torch_fl/_build_config.py" in ut and "torch_fl/backends.conf" in ut


def test_dev_run_activates_the_container_env_first(monkeypatch):
    # The fleet convention activates a shared conda env *inside* the container;
    # the UT command and the attest probe must both run under it (before the repo
    # `cd`), else they use the wrong-ABI default interpreter.
    ws = Workspace(
        name="a", host="h1", container="c", repo="/work/Torch-FL", source="git",
        env_setup="source /shared/env.sh && conda activate fl-env",
    )
    fake = _install(monkeypatch, [ws], {
        "git fetch": (0, "abc1234\n", ""),
        "pytest": (0, _UT_OK, ""),
    })
    asyncio.run(S.jms_dev_run(["all"], "main"))
    ut = next(c["command"] for c in fake.calls if "pytest" in c["command"])
    # activation precedes the repo cd, so both the UT cmd and the probe inherit it
    assert "source /shared/env.sh && conda activate fl-env; " in ut
    assert ut.index("conda activate fl-env") < ut.index("cd -- ")
    assert ut.index("conda activate fl-env") < ut.index("torch_fl/_build_config.py")


def test_dev_run_empty_ut_cmd_falls_back_to_default(monkeypatch):
    # A `ut_cmd:` YAML null override must not build a `();` syntax error.
    fake = _install(monkeypatch, [_ws("a", "h1")], {
        "git fetch": (0, "abc1234\n", ""),
        "pytest": (0, _UT_OK, ""),
    })
    asyncio.run(S.jms_dev_run(["all"], "main", ut_cmd=""))
    ut = next(c["command"] for c in fake.calls if "pytest" in c["command"])
    assert "(set -o pipefail; python -m pytest" in ut  # default used, no empty subshell


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
            return ExecResult(host.name, 0, _UT_OK, "")

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
    # The failure shape stays readable by an agent that keys on `rows`, and carries
    # the same attestation keys as the success path so a uniform reader doesn't KeyError.
    for key in ("rows", "columns", "revisions", "failed_hosts",
                "attestation", "attestation_consistent", "attestation_diverged",
                "attestation_unverified", "hosts"):
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
        "pytest": (0, _UT_OK, ""),
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
            return ExecResult(host.name, 0, _UT_OK, "")

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
    class _RevSSH(_FakeSSH):
        async def exec_on(self, host, command, *, timeout=60.0, workdir=True, env_setup=True):
            self.calls.append({"host": host.name, "command": command})
            if "git fetch" in command:
                sha = "abc1234" if host.name == "h1" else "def4567"
                return ExecResult(host.name, 0, sha + "\n", "")
            return ExecResult(host.name, 0, _UT_OK, "")

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


def test_dev_run_same_sha_but_different_code_path_is_not_ok(monkeypatch):
    # The whole point of attestation: two chips at the SAME commit, every test
    # PASSing, but dispatching the op to a different backend. A sha-only check
    # would read green; the run must be not-ok and name the divergence.
    class _PathSSH(_FakeSSH):
        async def exec_on(self, host, command, *, timeout=60.0, workdir=True, env_setup=True):
            self.calls.append({"host": host.name, "command": command})
            if "git fetch" in command:
                return ExecResult(host.name, 0, "abc1234\n", "")
            backend = "flaggems" if host.name == "h1" else "flagos_python"
            return ExecResult(host.name, 0, f"PASSED tests/t.py::test_x\n"
                                            f"JMSA backends_read=1\n"
                                            f"JMSA backend=softmax = {backend}\n", "")

    inv = Inventory(
        hosts={"h1": Host(name="h1", login="u", asset_id="A"),
               "h2": Host(name="h2", login="u", asset_id="A")},
        groups={},
        workspaces={"a": _ws("a", "h1"), "b": _ws("b", "h2")},
    )
    monkeypatch.setitem(S._state, "inv", inv)
    monkeypatch.setitem(S._state, "ssh", _PathSSH({}))
    r = asyncio.run(S.jms_dev_run(["all"], "main"))
    assert r["revisions_consistent"] is True   # same commit...
    assert r["attestation_consistent"] is False
    assert "backend_map_hash" in r["attestation_diverged"]
    assert r["ok"] is False


def test_dev_run_unreadable_attestation_is_unverified_not_ok(monkeypatch):
    # No attestation lines at all (an env where the probe could not run): the run
    # must not claim the code paths agree.
    _install(monkeypatch, [_ws("a", "h1"), _ws("b", "h2")], {
        "git fetch": (0, "abc1234\n", ""),
        "pytest": (0, "PASSED tests/t.py::test_x\n1 passed\n", ""),
    })
    r = asyncio.run(S.jms_dev_run(["all"], "main"))
    assert r["attestation_consistent"] is False
    assert sorted(r["attestation_unverified"]) == ["a", "b"]
    assert r["ok"] is False


# --- jms_dev_discover ---------------------------------------------------------------

_DISCOVERED = (
    "JMSD container=lvyufeng-torch-fl\n"
    "JMSD running=1\n"
    "JMSD inner=1\n"
    "JMSD python=/usr/local/python3.11.15/bin/python3|3.11.15|8.4.2\n"
    "JMSD repo=/public-flash/lvyufeng/PyTorch-Plugin-FL\n"
    "JMSD remote=git@github.com:lvyufeng/PyTorch-Plugin-FL.git\n"
    "JMSD branch=flagos-plugin\n"
    "JMSD sha=abc1234def5678\n"
    "JMSD reachable=ok\n"
)


def _install_hosts(monkeypatch, hosts, plan):
    inv = Inventory(
        hosts={h.name: h for h in hosts}, groups={}, workspaces={}
    )
    fake = _FakeSSH(plan)
    monkeypatch.setitem(S._state, "inv", inv)
    monkeypatch.setitem(S._state, "ssh", fake)
    return fake


def test_dev_discover_emits_workspaces(monkeypatch):
    from jms_bridge.inventory import Host

    fake = _install_hosts(
        monkeypatch,
        [Host(name="ascend-910c-108", login="u", asset_id="A", tags=["ascend"])],
        {"docker exec": (0, _DISCOVERED, "")},
    )
    r = asyncio.run(S.jms_dev_discover(["all"]))
    assert r["ok"] is True
    assert r["workspaces"]["fl-ascend-910c-108"]["repo"] == \
        "/public-flash/lvyufeng/PyTorch-Plugin-FL"
    assert r["workspaces"]["fl-ascend-910c-108"]["tags"] == ["ascend"]
    assert "workspaces:" in r["yaml"]
    # The command actually sent must be the real discovery script, not a stub.
    sent = fake.calls[0]["command"]
    assert "docker exec" in sent and "-print0" in sent and "PyTorch" not in sent
    # discovery must not source the host env or cd into the host workdir
    assert all(c["env_setup"] is False and c["workdir"] is False for c in fake.calls)


def test_dev_discover_rejects_bad_container(monkeypatch):
    _install_hosts(monkeypatch, [], {})
    r = asyncio.run(S.jms_dev_discover(["all"], container="foo; id"))
    assert r["ok"] is False
    assert "invalid container" in r["error"]


def test_dev_discover_rejects_bad_source(monkeypatch):
    _install_hosts(monkeypatch, [], {})
    r = asyncio.run(S.jms_dev_discover(["all"], source="ftp"))
    assert r["ok"] is False
    assert "source must be" in r["error"]


def test_dev_discover_requires_proxy_for_proxy_git(monkeypatch):
    _install_hosts(monkeypatch, [], {})
    r = asyncio.run(S.jms_dev_discover(["all"], source="proxy-git"))
    assert r["ok"] is False
    assert "proxy" in r["error"]


def test_dev_discover_requires_mount_and_local_for_push(monkeypatch):
    _install_hosts(monkeypatch, [], {})
    r = asyncio.run(S.jms_dev_discover(["all"], source="push", mount="/srv/fl"))
    assert r["ok"] is False
    assert "mount" in r["error"] and "local" in r["error"]


def test_dev_discover_no_match_is_not_ok(monkeypatch):
    from jms_bridge.inventory import Host

    _install_hosts(
        monkeypatch,
        [Host(name="h1", login="u", asset_id="A")],
        {"docker exec": (0, "JMSD container_absent=1\n", "")},
    )
    r = asyncio.run(S.jms_dev_discover(["all"]))
    assert r["ok"] is False
    assert r["skipped"] == ["h1"]


def test_dev_discover_partial_is_not_ok(monkeypatch):
    from jms_bridge.inventory import Host

    # one healthy host + one with the container absent -> ok must be False even
    # though workspaces were emitted for the healthy one.
    class _TwoSSH(_FakeSSH):
        async def exec_on(self, host, command, *, timeout=60.0, workdir=True, env_setup=True):
            self.calls.append({"host": host.name, "command": command})
            if host.name == "h2":
                return ExecResult(host.name, 0, "JMSD container_absent=1\n", "")
            return ExecResult(host.name, 0, _DISCOVERED, "")

    inv = Inventory(
        hosts={"h1": Host(name="h1", login="u", asset_id="A"),
               "h2": Host(name="h2", login="u", asset_id="A")},
        groups={}, workspaces={},
    )
    monkeypatch.setitem(S._state, "inv", inv)
    monkeypatch.setitem(S._state, "ssh", _TwoSSH({}))
    r = asyncio.run(S.jms_dev_discover(["all"]))
    assert r["ok"] is False
    assert "fl-h1" in r["workspaces"]
    assert r["skipped"] == ["h2"]


def test_dev_discover_timeout_is_reported_as_timeout(monkeypatch):
    from jms_bridge.inventory import Host

    _install_hosts(
        monkeypatch,
        [Host(name="h1", login="u", asset_id="A")],
        {"docker exec": (124, "", "timed out")},
    )
    r = asyncio.run(S.jms_dev_discover(["all"]))
    assert r["ok"] is False
    assert any("timed out" in w for w in r["warnings"])


# --- jms_inventory_reload -----------------------------------------------------------


def test_inventory_reload_rereads_and_summarises(monkeypatch):
    # A cached inventory (from an earlier tool call) must be dropped and re-read, so a
    # freshly pasted workspaces block is visible without an MCP restart.
    monkeypatch.setitem(S._state, "inv", Inventory(hosts={}, groups={}, workspaces={}))

    fresh = Inventory(
        hosts={"h1": Host(name="h1", login="u", asset_id="A")},
        groups={"cluster": ["h1"]},
        workspaces={"fl-h1": _ws("fl-h1", "h1")},
    )
    monkeypatch.setattr(Inventory, "load", classmethod(lambda cls, path=None: fresh))

    r = asyncio.run(S.jms_inventory_reload())
    assert r["ok"] is True
    assert r["hosts"] == ["h1"]
    assert r["workspaces"] == ["fl-h1"]
    assert r["groups"] == {"cluster": ["h1"]}
    # the cache now holds the fresh inventory, not the stale empty one
    assert S._state["inv"] is fresh


def test_inventory_reload_surfaces_a_load_error(monkeypatch):
    monkeypatch.setitem(S._state, "inv", Inventory(hosts={}, groups={}, workspaces={}))

    def boom(cls, path=None):
        raise RuntimeError("inventory /x/hosts.yaml has no 'hosts:' section")

    monkeypatch.setattr(Inventory, "load", classmethod(boom))
    r = asyncio.run(S.jms_inventory_reload())
    assert r["ok"] is False
    assert "no 'hosts:'" in r["error"]
