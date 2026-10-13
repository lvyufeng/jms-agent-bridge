import asyncio

import jms_bridge.server as S
from jms_bridge.inventory import Host, Inventory
from jms_bridge.jobs import TERMINAL_STATUS, JobHandle
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
        "jms_pull_artifacts", "jms_fleet_summary",
        "jms_submit_job", "jms_job_status", "jms_job_logs", "jms_job_cancel",
        "jms_job_wait", "jms_job_list",
        "jms_cluster_preflight", "jms_cluster_plan", "jms_cluster_submit",
        "jms_cluster_cancel",
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


def test_dev_run_screens_the_ut_cmd_override(monkeypatch):
    # `ut_cmd` is the one caller-supplied string this tool interpolates into a container
    # shell, so it gets the same denylist as jms_exec's `command` — and the guard fires
    # before anything is delivered.
    fake = _install(monkeypatch, [_ws("a", "h1")], {"git fetch": (0, "abc1234\n", "")})
    r = asyncio.run(S.jms_dev_run(["all"], "main", ut_cmd="rm -rf / && pytest"))
    assert r["ok"] is False
    assert "denylist" in r["error"]
    assert fake.calls == []  # nothing ran, nothing was delivered
    # the same shape as every other early return, so callers never KeyError
    assert r["rows"] == [] and r["attestation"] == {} and r["hosts"] == {}


def test_dev_run_allow_dangerous_overrides_the_ut_cmd_guard(monkeypatch):
    fake = _install(monkeypatch, [_ws("a", "h1")], {
        "git fetch": (0, "abc1234\n", ""),
        "pytest": (0, _UT_OK, ""),
    })
    r = asyncio.run(
        S.jms_dev_run(["all"], "main", ut_cmd="rm -rf / && pytest", allow_dangerous=True)
    )
    assert r["ok"] is True
    assert any("pytest" in c["command"] for c in fake.calls)


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


def test_dev_sync_isolates_one_workspace_whose_delivery_raises(monkeypatch):
    # A workspace whose delivery *raises* (a transport error, a host the inventory no
    # longer has) must not discard the others through the fan-out `gather` — that
    # would surface the whole delivery as an opaque tool error and lose the per-chip
    # picture. The raise becomes that workspace's failure record.
    class _BoomSSH(_FakeSSH):
        async def exec_on(self, host, command, *, timeout=60.0, workdir=True, env_setup=True):
            if host.name == "h2":
                raise RuntimeError("ssh binary vanished")
            return await super().exec_on(
                host, command, timeout=timeout, workdir=workdir, env_setup=env_setup
            )

    inv = Inventory(
        hosts={"h1": Host(name="h1", login="u", asset_id="A"),
               "h2": Host(name="h2", login="u", asset_id="A")},
        groups={},
        workspaces={"a": _ws("a", "h1"), "b": _ws("b", "h2")},
    )
    monkeypatch.setitem(S._state, "inv", inv)
    monkeypatch.setitem(S._state, "ssh", _BoomSSH({"git fetch": (0, "abc1234\n", "")}))
    r = asyncio.run(S.jms_dev_sync(["all"], "main"))
    assert r["ok"] is False
    # both workspaces still report; only the raising one is marked failed
    assert [x["workspace"] for x in r["results"]] == ["a", "b"]
    broke = r["results"][1]
    assert broke["ok"] is False and "delivery raised" in broke["stderr"]


def test_dev_run_isolates_one_chip_whose_run_raises(monkeypatch):
    # Same isolation for the run stage: one chip's raise must leave the cross-chip
    # table (aligned rows from the healthy chips) intact.
    class _BoomSSH(_FakeSSH):
        async def exec_on(self, host, command, *, timeout=60.0, workdir=True, env_setup=True):
            if host.name == "h2" and "pytest" in command:
                raise RuntimeError("ssh binary vanished")
            return await super().exec_on(
                host, command, timeout=timeout, workdir=workdir, env_setup=env_setup
            )

    inv = Inventory(
        hosts={"h1": Host(name="h1", login="u", asset_id="A"),
               "h2": Host(name="h2", login="u", asset_id="A")},
        groups={},
        workspaces={"a": _ws("a", "h1"), "b": _ws("b", "h2")},
    )
    monkeypatch.setitem(S._state, "inv", inv)
    monkeypatch.setitem(S._state, "ssh", _BoomSSH({
        "git fetch": (0, "abc1234\n", ""),
        "pytest": (0, _UT_OK, ""),
    }))
    r = asyncio.run(S.jms_dev_run(["all"], "main"))
    assert r["ok"] is False
    assert r["failed_hosts"] == ["b"]
    # the healthy chip's rows survive the other chip's raise
    assert any(row["test_id"] == "tests/t.py::test_x" for row in r["rows"])


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


# --- jms_dev_logs and the delivery path (git / push) --------------------------------


def test_dev_logs_reruns_under_the_container_env(monkeypatch):
    # The point of this tool is to say why a test failed on a chip, so it must re-run
    # under the very interpreter the fleet run used — activation before the repo cd.
    ws = Workspace(
        name="a", host="h1", container="c", repo="/work/Torch-FL", source="git",
        env_setup="source /shared/env.sh && conda activate fl-env",
    )
    fake = _install(monkeypatch, [ws], {"pytest": (0, "1 failed", "")})
    r = asyncio.run(S.jms_dev_logs("a", "tests/unit/test_softmax.py::test_x"))
    assert r["ok"] is True
    sent = fake.calls[0]["command"]
    assert "conda activate fl-env; cd -- " in sent
    assert "tests/unit/test_softmax.py::test_x" in sent
    assert "-q --tb=long -rA" in sent          # drill-in needs the full traceback
    assert sent.index("conda activate fl-env") < sent.index("cd -- ")
    # env_setup is inlined in-container; the *host* env must stay untouched
    assert fake.calls[0]["env_setup"] is False


def test_dev_logs_without_env_setup_cds_straight_into_the_repo(monkeypatch):
    fake = _install(monkeypatch, [_ws("a", "h1")], {"pytest": (0, "1 passed", "")})
    asyncio.run(S.jms_dev_logs("a", "tests/unit/test_softmax.py::test_x"))
    inner = fake.calls[0]["command"].split("bash -lc ")[1]
    # no env_setup -> no `; ` (not even an empty one) leading into the cd
    assert inner.startswith("'cd -- ")
    assert "/work/Torch-FL" in inner


def test_dev_sync_git_reports_the_sha_the_container_landed_on(monkeypatch):
    fake = _install(monkeypatch, [_ws("a", "h1")], {"git fetch": (0, "abc1234\n", "")})
    r = asyncio.run(S.jms_dev_sync(["all"], "main"))
    assert r["ok"] is True
    assert r["results"][0]["sha"] == "abc1234"
    assert r["results"][0]["ref_pinned"] is True
    sent = fake.calls[0]["command"]
    assert "git fetch origin " in sent and "main" in sent and "--depth=1" in sent
    # the ref is pinned by explicit fetch + detached checkout, not a bare `checkout`
    assert "git checkout -f --detach FETCH_HEAD" in sent
    assert "git rev-parse HEAD" in sent                    # which commit actually landed
    assert fake.calls[0]["env_setup"] is False


def test_dev_sync_git_without_a_sha_is_a_failed_delivery(monkeypatch):
    # A chip whose rev-parse yields no sha-shaped line cannot prove which code it
    # holds; delivering "something" is not the same as delivering `ref`.
    _install(monkeypatch, [_ws("a", "h1")], {"git fetch": (0, "Welcome\n", "")})
    r = asyncio.run(S.jms_dev_sync(["all"], "main"))
    assert r["ok"] is False
    assert r["results"][0]["sha"] is None


def test_dev_sync_push_rsyncs_the_expanded_local_tree(monkeypatch, tmp_path):
    # `local:` is commonly `~/...` and rsync is exec'd with no shell, so a literal `~`
    # would push a directory *named* `~`: every chip would run stale code and the run
    # would still look green.
    tree = tmp_path / "tree"
    tree.mkdir()
    push = Workspace(
        name="p", host="h2", container="c", repo="/work/Torch-FL", source="push",
        mount="/srv/fl", local="~/tree",
    )
    inv = Inventory(
        hosts={"h2": Host(name="h2", login="u", asset_id="A")},
        groups={}, workspaces={"p": push},
    )
    fake = _FakeSSH({})
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setitem(S._state, "inv", inv)
    monkeypatch.setitem(S._state, "ssh", fake)
    r = asyncio.run(S.jms_dev_sync(["all"], "main"))
    assert r["ok"] is True
    puts = [c for c in fake.calls if "rsync" in c]
    assert puts, "push delivery must go through rsync"
    assert puts[0]["rsync"] == (str(tree), "/srv/fl")   # expanded, not "~/tree"
    # push never applies `ref` in-container
    assert r["results"][0]["ref_pinned"] is False


def test_job_logs_tool_surfaces_the_next_offset(monkeypatch):
    # The MCP tool is where the byte-offset contract reaches the agent: it must hand
    # back the resume point, not just the file size.
    from jms_bridge.jobs import JobManager

    class _LogRunner:
        async def exec_on(self, host, command, *, timeout=60, workdir=True):
            return ExecResult(host.name, 0, "size=100000\n" + "x" * 10, "")

    inv = Inventory(
        hosts={"h1": Host(name="h1", login="u", asset_id="A")}, groups={}, workspaces={}
    )
    monkeypatch.setitem(S._state, "inv", inv)
    monkeypatch.setitem(S._state, "jobs", JobManager(_LogRunner()))
    r = asyncio.run(S.jms_job_logs("h1", "deadbeef", offset=0, max_bytes=10))
    assert r["ok"] is True
    assert r["size"] == 100000
    assert r["next_offset"] == 10


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


# --- jms_cluster_submit: preflight gates the rank assignment ------------------------


#: A preflight reply for an Ascend host showing two healthy chips and a free port.
_NPU_READY = (
    "===PROBE===\n"
    "| 0     910C   | OK | 45   0 / 0 |\n"
    "| 0     0      | 0000:01:00.0 | 0 / 65536 |\n"
    "| 1     910C   | OK | 44   0 / 0 |\n"
    "| 1     0      | 0000:02:00.0 | 0 / 65536 |\n"
    "===PORT===\nPORT_FREE\n===END===\n"
)


class _PreflightSSH(_FakeSSH):
    """Replies to the preflight probe per host; anything else succeeds."""

    def __init__(self, by_host):
        super().__init__({})
        self.by_host = by_host

    async def exec_on(self, host, command, *, timeout=60.0, workdir=True, env_setup=True):
        self.calls.append({"host": host.name, "command": command})
        if "===PROBE===" in command:
            return self.by_host.get(host.name, ExecResult(host.name, 0, _NPU_READY, ""))
        return ExecResult(host.name, 0, "", "")


class _FakeJobs:
    """Records the collab assignments the tool hands to the job manager."""

    def __init__(self):
        self.collab = []

    async def submit_collab(self, assignments, command, *, master_addr, port, world_size):
        self.collab.append(
            {
                "assignments": [(h.name, n) for h, n in assignments],
                "command": command,
                "master_addr": master_addr,
                "port": port,
                "world_size": world_size,
            }
        )
        handles = [
            JobHandle(f"{i:08x}", h.name, f"jms-{i:08x}", "tmux")
            for i, (h, _) in enumerate(assignments)
        ]
        results = [ExecResult(h.name, 0, "LAUNCHER=tmux", "") for h, _ in assignments]
        return handles, results


def _cluster_install(monkeypatch, names, by_host):
    hosts = [
        Host(name=n, login="u", asset_id="A", platform="ascend-910c", address=f"10.0.0.{i + 1}")
        for i, n in enumerate(names)
    ]
    inv = Inventory(hosts={h.name: h for h in hosts}, groups={}, workspaces={})
    fake = _PreflightSSH(by_host)
    jobs = _FakeJobs()
    monkeypatch.setitem(S._state, "inv", inv)
    monkeypatch.setitem(S._state, "ssh", fake)
    monkeypatch.setitem(S._state, "jobs", jobs)
    return fake, jobs


def test_cluster_submit_ranks_every_ready_host(monkeypatch):
    _, jobs = _cluster_install(monkeypatch, ["a", "b"], {})
    r = asyncio.run(S.jms_cluster_submit(["a", "b"], "torchrun train.py"))
    assert r["ok"] is True
    assert r["world_size"] == 4  # 2 chips on each of 2 hosts
    assert [x["host"] for x in r["ranks"]] == ["a", "b"]
    assert all(x["launched"] and x["nproc"] == 2 for x in r["ranks"])
    assert jobs.collab[0]["assignments"] == [("a", 2), ("b", 2)]
    assert jobs.collab[0]["world_size"] == 4


def test_cluster_submit_refuses_to_rank_an_unreachable_host(monkeypatch):
    # An unreachable host reports no device_count, yet it used to be handed a rank with
    # nproc=1 — so every rank was told a world size larger than the set that could join
    # it. The job cannot rendezvous; the call must refuse instead of launching it.
    _, jobs = _cluster_install(
        monkeypatch,
        ["a", "b"],
        {"b": ExecResult("b", 255, "", "ssh: connect to host b port 2222: refused")},
    )
    r = asyncio.run(S.jms_cluster_submit(["a", "b"], "torchrun train.py"))
    assert r["ok"] is False
    assert "not ready" in r["error"] and "b" in r["error"]
    assert [u["host"] for u in r["unready"]] == ["b"]
    assert r["ready_targets"] == ["a"]
    assert jobs.collab == []  # nothing launched at the wrong world size


def test_cluster_submit_refuses_a_host_holding_the_rendezvous_port(monkeypatch):
    # A stale run still bound to the rendezvous port makes that rank unlaunchable, even
    # though the host is reachable and its accelerator probe is fine.
    busy = _NPU_READY.replace("PORT_FREE", "PORT_BUSY")
    _, jobs = _cluster_install(
        monkeypatch, ["a", "b"], {"b": ExecResult("b", 0, busy, "")}
    )
    r = asyncio.run(S.jms_cluster_submit(["a", "b"], "torchrun train.py"))
    assert r["ok"] is False
    assert r["unready"][0]["port_free"] is False
    assert jobs.collab == []


def test_cluster_submit_screens_the_command_like_the_other_launchers(monkeypatch):
    # It used to hard-fail with no escape hatch, unlike jms_exec / jms_submit_job, so a
    # caller who genuinely needed an overridable command was stuck. The override now
    # exists; without it the command is still refused, before any host is contacted.
    fake, jobs = _cluster_install(monkeypatch, ["a", "b"], {})
    r = asyncio.run(S.jms_cluster_submit(["a", "b"], "rm -rf / ; torchrun train.py"))
    assert r["ok"] is False
    assert "allow_dangerous=true" in r["error"]
    assert fake.calls == [] and jobs.collab == []

    r = asyncio.run(
        S.jms_cluster_submit(
            ["a", "b"], "rm -rf / ; torchrun train.py", allow_dangerous=True
        )
    )
    assert r["ok"] is True
    assert jobs.collab[0]["command"] == "rm -rf / ; torchrun train.py"


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


# --- job lifecycle tools: wait / list / cluster_cancel ------------------------------


class _FakeJobManager:
    """A JobManager stand-in for the lifecycle tools (no SSH, no tmux).

    ``statuses`` drives `wait` (a host with no scripted status reports `done`); ``jobs``
    drives `list_jobs`; ``killed`` names the hosts whose cancel reports a kill and
    ``raises`` the hosts whose cancel fails outright.
    """

    def __init__(self, *, statuses=None, jobs=None, killed=(), raises=(), list_raises=()):
        self.statuses = statuses or {}
        self.jobs = jobs or {}
        self.killed = set(killed)
        self.raises = set(raises)
        self.list_raises = set(list_raises)
        self.calls = []

    async def wait(self, pairs, *, timeout=150.0, interval=2.0):
        self.calls.append(("wait", [(h.name, j) for h, j in pairs], timeout))
        latest = [
            {"host": h.name, "job_id": j, "status": self.statuses.get(h.name, "done")}
            for h, j in pairs
        ]
        return latest, all(s["status"] in TERMINAL_STATUS for s in latest)

    async def list_jobs(self, host):
        if host.name in self.list_raises:
            raise RuntimeError("tmux: command not found")
        return list(self.jobs.get(host.name, []))

    async def cancel_many(self, pairs):
        self.calls.append(("cancel", [(h.name, j) for h, j in pairs]))
        out = []
        for h, j in pairs:
            if h.name in self.raises:
                out.append(
                    {"host": h.name, "job_id": j, "killed": False, "ok": False, "error": "boom"}
                )
            else:
                out.append(
                    {"host": h.name, "job_id": j, "ok": True, "killed": h.name in self.killed}
                )
        return out


def _lifecycle_install(monkeypatch, names, **kw):
    hosts = [Host(name=n, login="u", asset_id="A") for n in names]
    inv = Inventory(hosts={h.name: h for h in hosts}, groups={}, workspaces={})
    jobs = _FakeJobManager(**kw)
    monkeypatch.setitem(S._state, "inv", inv)
    monkeypatch.setitem(S._state, "jobs", jobs)
    return jobs


def test_job_wait_reports_finished_once_every_host_is_terminal(monkeypatch):
    jobs = _lifecycle_install(monkeypatch, ["a", "b"], statuses={"a": "done", "b": "failed"})
    r = asyncio.run(S.jms_job_wait(["a", "b"], "abcd1234", timeout=5))
    assert r["ok"] is True and r["finished"] is True
    assert [s["host"] for s in r["status"]] == ["a", "b"]
    # the one shared collab id reaches every target, and the caller's timeout is used
    assert jobs.calls == [("wait", [("a", "abcd1234"), ("b", "abcd1234")], 5)]


def test_job_wait_timeout_is_not_a_failure(monkeypatch):
    # A bounded wait that runs out must say so, not error: the job is still running, and
    # the caller is free to wait again or cancel.
    jobs = _lifecycle_install(monkeypatch, ["a"], statuses={"a": "running"})
    r = asyncio.run(S.jms_job_wait(["a"], "abcd1234", timeout=0.1))
    assert r["ok"] is False and r["finished"] is False
    assert r["status"][0]["status"] == "running"
    # ...and it cancels nothing.
    assert all(c[0] != "cancel" for c in jobs.calls)


def test_job_wait_rejects_a_malformed_id_without_polling(monkeypatch):
    jobs = _lifecycle_install(monkeypatch, ["a"])
    r = asyncio.run(S.jms_job_wait(["a"], "not-an-id"))
    assert r["ok"] is False and "invalid job id" in r["error"]
    assert jobs.calls == []


def test_job_list_is_per_host(monkeypatch):
    _lifecycle_install(monkeypatch, ["a", "b"], jobs={"a": ["deadbeef"], "b": []})
    r = asyncio.run(S.jms_job_list(["a", "b"]))
    assert r["ok"] is True
    assert r["hosts"] == [
        {"host": "a", "job_ids": ["deadbeef"]},
        {"host": "b", "job_ids": []},
    ]


def test_job_list_on_an_empty_selection_is_not_success(monkeypatch):
    # `resolve([])` returns [] without raising, so `all()` over nothing would be True.
    _lifecycle_install(monkeypatch, [])
    r = asyncio.run(S.jms_job_list([]))
    assert r["ok"] is False and r["hosts"] == []


def test_job_list_isolates_a_host_that_fails(monkeypatch):
    _lifecycle_install(monkeypatch, ["a", "b"], jobs={"a": ["deadbeef"]}, list_raises={"b"})
    r = asyncio.run(S.jms_job_list(["a", "b"]))
    assert r["ok"] is False  # a host that could not be asked is not "no jobs"
    # one host's tmux being missing must not discard the ids found on the others
    assert r["hosts"][0] == {"host": "a", "job_ids": ["deadbeef"]}
    assert r["hosts"][1]["job_ids"] is None and "tmux" in r["hosts"][1]["error"]


def test_cluster_cancel_stops_every_rank_with_one_id(monkeypatch):
    jobs = _lifecycle_install(monkeypatch, ["a", "b"], killed={"a", "b"})
    r = asyncio.run(S.jms_cluster_cancel(["a", "b"], "abcd1234"))
    assert r["ok"] is True
    assert r["job_id"] == "abcd1234"
    assert [c["host"] for c in r["cancelled"]] == ["a", "b"]
    assert jobs.calls == [("cancel", [("a", "abcd1234"), ("b", "abcd1234")])]


def test_cluster_cancel_reports_a_rank_that_did_not_stop(monkeypatch):
    # A host where nothing was killed is not an error (it may simply have finished), but
    # a host whose cancel failed must not be reported as a clean sweep.
    _lifecycle_install(monkeypatch, ["a", "b"], killed={"a"}, raises={"b"})
    r = asyncio.run(S.jms_cluster_cancel(["a", "b"], "abcd1234"))
    assert r["ok"] is False
    assert [c["host"] for c in r["cancelled"]] == ["a"]
    assert r["results"][1]["ok"] is False


def test_cluster_cancel_on_an_empty_selection_is_not_success(monkeypatch):
    # `resolve([])` is empty without raising and `all()` over nothing is True, so a
    # cancel that reached no host must not come back green.
    jobs = _lifecycle_install(monkeypatch, ["a"])
    r = asyncio.run(S.jms_cluster_cancel([], "abcd1234"))
    assert r["ok"] is False
    assert "nothing was cancelled" in r["error"]
    assert jobs.calls == []


def test_cluster_cancel_rejects_a_malformed_id(monkeypatch):
    jobs = _lifecycle_install(monkeypatch, ["a"])
    r = asyncio.run(S.jms_cluster_cancel(["a"], "nope"))
    assert r["ok"] is False and "invalid job id" in r["error"]
    assert jobs.calls == []


# --- jms_pull_artifacts: one directory per host, no path escape ----------------------


def test_pull_artifacts_gives_each_host_its_own_directory(monkeypatch):
    fake = _install_hosts(
        monkeypatch,
        [Host(name="a", login="u", asset_id="A"), Host(name="b", login="u", asset_id="A")],
        {},
    )
    r = asyncio.run(S.jms_pull_artifacts(["a", "b"], "runs/ckpt", local="/tmp/arts"))
    assert r["ok"] is True
    assert r["root"] == "/tmp/arts"
    assert [x["local"] for x in r["results"]] == ["/tmp/arts/a", "/tmp/arts/b"]
    # every transfer is a pull of the same relative artifact, each to its own directory
    assert [c["host"] for c in fake.calls] == ["a", "b"]
    assert all("rsync" in c for c in fake.calls)
    assert all(c["rsync"][1] == "runs/ckpt" for c in fake.calls)


def test_pull_artifacts_sanitises_a_host_name(monkeypatch):
    # A host name is operator-supplied and may carry path separators; one directory per
    # host is what keeps same-named files from different nodes from colliding.
    _install_hosts(monkeypatch, [Host(name="../evil", login="u", asset_id="A")], {})
    r = asyncio.run(S.jms_pull_artifacts(["../evil"], "log", local="/tmp/arts"))
    assert r["results"][0]["local"] == "/tmp/arts/_evil"


def test_pull_artifacts_rejects_paths_that_escape_the_workdir(monkeypatch):
    fake = _install_hosts(monkeypatch, [Host(name="a", login="u", asset_id="A")], {})
    for remote in ["/etc", "~/.ssh", "../../etc", "", "  ", "a/../../b"]:
        r = asyncio.run(S.jms_pull_artifacts(["a"], remote, local="/tmp/arts"))
        assert r["ok"] is False, remote
        assert "remote" in r["error"]
    assert fake.calls == []  # nothing was ever transferred


def test_pull_artifacts_isolates_one_host_failure(monkeypatch):
    class _RsyncBoom(_FakeSSH):
        async def rsync(self, host, local, remote, **kw):
            self.calls.append({"host": host.name, "rsync": (local, remote)})
            if host.name == "b":
                raise RuntimeError("ssh: connection closed")
            return ExecResult(host.name, 0, "", "")

    hosts = [Host(name="a", login="u", asset_id="A"), Host(name="b", login="u", asset_id="A")]
    inv = Inventory(hosts={h.name: h for h in hosts}, groups={}, workspaces={})
    monkeypatch.setitem(S._state, "inv", inv)
    monkeypatch.setitem(S._state, "ssh", _RsyncBoom({}))

    r = asyncio.run(S.jms_pull_artifacts(["a", "b"], "runs", local="/tmp/arts"))
    assert r["ok"] is False  # the batch is not green...
    assert r["results"][0]["ok"] is True  # ...but a's artifacts did come back
    assert r["results"][1]["ok"] is False and "connection closed" in r["results"][1]["error"]


def test_artifact_subdir_falls_back_for_a_name_with_nothing_usable():
    from jms_bridge.server import _artifact_subdir

    assert _artifact_subdir("ascend-910c-108") == "ascend-910c-108"
    assert _artifact_subdir("a/b c") == "a_b_c"
    assert _artifact_subdir("..") == "host"
    assert _artifact_subdir("") == "host"


# --- occupancy gates the cluster tools ----------------------------------------------


#: The ready reply with one card already holding a process spliced in before the port.
_NPU_BUSY_CARD = _NPU_READY.replace(
    "===PORT===\n",
    "===OCCUPANCY===\n"
    "NPU ID  Chip ID  Process id  Name  Mem\n"
    "0  0  777  python  1024\n"
    "===PORT===\n",
)


def test_cluster_preflight_flags_busy_hosts_and_leaves_unknown_alone(monkeypatch):
    _cluster_install(monkeypatch, ["a", "b"], {"a": ExecResult("a", 0, _NPU_BUSY_CARD, "")})
    r = asyncio.run(S.jms_cluster_preflight(["a", "b"]))
    by_host = {h["host"]: h for h in r["hosts"]}
    assert by_host["a"]["busy_devices"] == 1 and by_host["a"]["busy"] is True
    # b sent no occupancy section: unknown, which is *not* busy and does not block.
    assert by_host["b"]["busy_devices"] is None and by_host["b"]["busy"] is False
    assert by_host["b"]["reachable"] is True and by_host["b"]["port_free"] is True
    assert r["ok"] is False  # the set as a whole is not launchable by default
    assert r["busy_hosts"] == ["a"]


def test_cluster_submit_refuses_to_share_occupied_cards(monkeypatch):
    _, jobs = _cluster_install(
        monkeypatch, ["a", "b"], {"a": ExecResult("a", 0, _NPU_BUSY_CARD, "")}
    )
    r = asyncio.run(S.jms_cluster_submit(["a", "b"], "torchrun train.py"))
    assert r["ok"] is False
    assert "already have accelerators in use" in r["error"]
    assert [b["host"] for b in r["busy"]] == ["a"]
    assert r["ready_targets"] == ["b"]
    assert jobs.collab == []  # nothing was launched onto the busy card


def test_cluster_submit_allow_busy_overrides_the_gate(monkeypatch):
    # The common honest reason a card is busy is this caller's own job, so a legitimate
    # relaunch must remain possible.
    _, jobs = _cluster_install(
        monkeypatch, ["a", "b"], {"a": ExecResult("a", 0, _NPU_BUSY_CARD, "")}
    )
    r = asyncio.run(S.jms_cluster_submit(["a", "b"], "torchrun train.py", allow_busy=True))
    assert r["ok"] is True
    assert r["world_size"] == 4
    assert jobs.collab[0]["assignments"] == [("a", 2), ("b", 2)]


def test_preflight_world_size_matches_what_submit_would_launch(monkeypatch):
    # The estimate exists to predict the submit; the two drifted apart once already.
    _cluster_install(monkeypatch, ["a", "b"], {})
    pf = asyncio.run(S.jms_cluster_preflight(["a", "b"]))
    sub = asyncio.run(S.jms_cluster_submit(["a", "b"], "torchrun train.py"))
    assert pf["world_size_estimate"] == sub["world_size"] == 4


# --- shutdown: the lazily-built clients are closed, and forgotten -------------------


def _record_closes():
    """Install fake ssh/api/jobs clients that record what happens to them."""
    calls = []

    class _FakeSSH:
        async def close(self):
            calls.append("ssh")

    class _FakeAPI:
        async def aclose(self):
            calls.append("api")

    return calls, _FakeSSH(), _FakeAPI()


def test_aclose_state_closes_and_forgets_the_clients(monkeypatch):
    # Nothing else closes the runner, so without this every session leaves one
    # ControlMaster socket per host it talked to (plus the /tmp/jmsb-* directory).
    calls, ssh, api = _record_closes()
    monkeypatch.setitem(S._state, "ssh", ssh)
    monkeypatch.setitem(S._state, "api", api)
    monkeypatch.setitem(S._state, "jobs", object())

    asyncio.run(S._aclose_state())

    assert calls == ["api", "ssh"]
    assert "ssh" not in S._state and "api" not in S._state and "jobs" not in S._state


def test_aclose_state_tolerates_a_half_built_state(monkeypatch):
    # A session that only ever used the control plane has no runner cached, and vice
    # versa; cleanup must not assume either exists.
    monkeypatch.delitem(S._state, "ssh", raising=False)
    monkeypatch.delitem(S._state, "api", raising=False)
    monkeypatch.delitem(S._state, "jobs", raising=False)
    asyncio.run(S._aclose_state())  # must not raise


def test_main_cleans_up_on_the_way_out(monkeypatch):
    # `server.run` returns on stdin EOF and on Ctrl-C; either way the process is leaving
    # and the sockets must go with it.
    calls, ssh, api = _record_closes()
    monkeypatch.setitem(S._state, "ssh", ssh)
    monkeypatch.setitem(S._state, "api", api)
    monkeypatch.setattr(S.server, "run", lambda transport: None)

    S.main()

    assert calls == ["api", "ssh"]


def test_main_survives_a_failing_cleanup(monkeypatch):
    # A cleanup failure at shutdown must not turn a clean exit into a nonzero one.
    class _Boom:
        async def close(self):
            raise RuntimeError("control socket already gone")

    monkeypatch.setitem(S._state, "ssh", _Boom())
    monkeypatch.setattr(S.server, "run", lambda transport: None)

    S.main()  # must not raise


# --- jms_fleet_summary: the rollup rules and the tool (fakes; no SSH) ---------------


def test_fleet_rollup_counts_reachable_and_devices_per_platform():
    rows = [
        {"host": "a", "platform": "cuda", "reachable": True, "device_count": 8, "reason": ""},
        {"host": "b", "platform": "cuda", "reachable": True, "device_count": 8, "reason": ""},
        {"host": "c", "platform": "metax", "reachable": False,
         "device_count": None, "reason": "403"},
    ]
    r = S._fleet_rollup(rows)
    assert r["by_platform"]["cuda"] == {"hosts": 2, "reachable": 2, "devices": 16}
    # An unreachable host is counted as a host, never as reachable or as devices.
    assert r["by_platform"]["metax"] == {"hosts": 1, "reachable": 0, "devices": 0}
    assert r["unreachable"] == [{"host": "c", "reason": "403"}]


def test_fleet_rollup_unknown_device_count_is_zero_not_guessed():
    # Reachable but the probe gave no count -> contributes 0 devices, not a guess.
    rows = [
        {"host": "a", "platform": "iluvatar", "reachable": True,
         "device_count": None, "reason": ""},
    ]
    r = S._fleet_rollup(rows)
    assert r["by_platform"]["iluvatar"]["devices"] == 0


class _ReachSSH:
    """Reachability + probe fake: hosts in ``unreachable`` answer a KoKo rejection."""

    def __init__(self, unreachable=()):
        self.unreachable = set(unreachable)

    async def exec_on(self, host, command, **kw):  # noqa: ANN001
        if "jms-bridge-ok" in command:
            if host.name in self.unreachable:
                # KoKo rejects with exit 0 and the error on stdout (no sentinel).
                return ExecResult(host.name, 0, "403 no permission\n", "")
            return ExecResult(host.name, 0, "host\njms-bridge-ok\n", "")
        cuda = "\n".join(f"| {i}  NVIDIA A100   00000000:1B:0{i}.0 Off |" for i in range(8))
        metax = "\n".join(f"| {i}     MetaX C550 | 0 Off | 0000:2b:0{i}.0 |" for i in range(8))
        return ExecResult(host.name, 0, metax if "mx-smi" in command else cuda, "")


def test_fleet_summary_splits_reachable_from_unreachable(monkeypatch):
    hosts = {
        "a": Host(name="a", login="u", asset_id="A", platform="cuda"),
        "b": Host(name="b", login="u", asset_id="B", platform="metax"),
        "c": Host(name="c", login="u", asset_id="C", platform="cuda"),
    }
    monkeypatch.setitem(S._state, "inv", Inventory(hosts=hosts, groups={}, workspaces={}))
    monkeypatch.setitem(S._state, "ssh", _ReachSSH(unreachable={"c"}))

    r = asyncio.run(S.jms_fleet_summary(["all"]))

    assert r["hosts_total"] == 3 and r["hosts_reachable"] == 2
    assert r["devices_total"] == 16
    assert r["by_platform"]["cuda"] == {"hosts": 2, "reachable": 1, "devices": 8}
    assert r["by_platform"]["metax"] == {"hosts": 1, "reachable": 1, "devices": 8}
    assert r["unreachable"] == [{"host": "c", "reason": "403 no permission"}]
    assert r["ok"] is False  # one host did not answer
