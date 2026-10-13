import asyncio

import pytest

from jms_bridge.inventory import Host
from jms_bridge.jobs import (
    JobManager,
    _validate_job_id,
    build_runner_script,
    validate_job_id,
)
from jms_bridge.ssh import ExecResult


def test_runner_exports_rank_wiring_for_collab():
    """Collaborative launch exports the shared rank/world env before the command."""
    from jms_bridge.jobs import _collab_exports

    exports = _collab_exports(rank=2, nproc=8, world_size=16, master_addr="10.0.0.1", port=29400)
    assert "export JMS_RANK='2'" in exports
    assert "export JMS_WORLD_SIZE='16'" in exports
    assert "export JMS_NPROC='8'" in exports
    assert "export JMS_MASTER_ADDR='10.0.0.1'" in exports
    assert "export JMS_MASTER_PORT='29400'" in exports
    # exports are injected before the user command
    script = build_runner_script("abcd1234", f"{exports}\ntorchrun train.py", None)
    assert script.index("JMS_WORLD_SIZE") < script.index("torchrun train.py")


def test_runner_records_pid_status_exit_code():
    script = build_runner_script("abcd1234", "echo hello", "source env.sh")
    # Self-locating: state is written with absolute paths under $R (the job dir),
    # so it lands there regardless of the cwd tmux handed the session.
    assert 'R="$(cd "$(dirname "$0")" && pwd)"' in script
    assert 'echo "$$" > "$R/pid"' in script
    assert 'echo running > "$R/status"' in script
    assert 'echo done > "$R/status"' in script
    assert 'echo failed > "$R/status"' in script
    assert 'echo "$code" > "$R/exit_code"' in script
    # env activation happens before the command
    assert script.index("source env.sh") < script.index("echo hello")


def test_runner_cds_into_workdir_tilde_safe():
    """A tilde workdir must be a shell expression, not a literal '~' in quotes."""
    script = build_runner_script("abcd1234", "python train.py", None, "~/torch-fl")
    assert 'cd "$HOME"/torch-fl 2>/dev/null || cd "$R"' in script
    # absolute workdir is single-quoted
    script = build_runner_script("abcd1234", "python train.py", None, "/data/fl")
    assert "cd '/data/fl' 2>/dev/null || cd \"$R\"" in script


def test_runner_wraps_command_snippet():
    script = build_runner_script("abcd1234", "python train.py --epochs 3", None)
    assert "python train.py --epochs 3" in script


def test_runner_isolates_command_exit_in_subshell():
    """A command ending in `exit N` must not kill the wrapper before status is written."""
    script = build_runner_script("abcd1234", "echo boom >&2; exit 7", None)
    lines = script.splitlines()
    cmd_i = lines.index("echo boom >&2; exit 7")
    # the command sits between a lone '(' and a lone ')'
    assert lines[cmd_i - 1].strip() == "("
    assert lines[cmd_i + 1].strip() == ")"
    # and the exit-code capture follows the subshell
    assert script.index("code=$?") > script.index("exit 7")


def test_runner_relaxes_nounset_around_env_setup_only():
    """Vendor env scripts (CANN's set_env.sh, conda hooks) read possibly-unset vars,
    and this script runs under `set -u` — without the guard one unset var aborts the
    job before the command ever starts. The relaxation must bracket only env_setup."""
    script = build_runner_script("abcd1234", "python train.py", "source /opt/cann/set_env.sh")
    assert "set +u\nsource /opt/cann/set_env.sh\nset -u" in script
    # `set -u` is back on by the time the command runs
    assert script.index("set -u\n") < script.index("python train.py")
    # no env_setup -> nothing to relax, and `-u` stays on throughout
    assert "set +u" not in build_runner_script("abcd1234", "echo hi", None)


def test_validate_job_id_accepts_8_hex():
    _validate_job_id("deadbeef")
    _validate_job_id("0123abcd")


# --- coordinated launch (ranks) -----------------------------------------------------


class _CollabRunner:
    """Records each launch; a named rank may raise instead of returning a result."""

    def __init__(self, explode="_none_"):
        self.explode = explode
        self.launched = []
        self.commands = {}

    async def exec_on(self, host, command, *, timeout=90, workdir=False):
        self.launched.append(host.name)
        self.commands[host.name] = command
        if host.name == self.explode:
            raise RuntimeError("ssh binary vanished")
        return ExecResult(host.name, 0, "LAUNCHER=tmux", "")


def test_collab_launch_isolates_one_rank_that_raises():
    """One rank raising must not discard every other rank's launch result.

    An unguarded gather lets a single rank's transport error surface the whole
    collective as an opaque tool error, hiding which nodes actually came up.
    """
    runner = _CollabRunner(explode="b")
    hosts = [Host(name=n, login="u", asset_id="A") for n in ("a", "b", "c")]
    handles, results = asyncio.run(
        JobManager(runner).submit_collab(
            [(h, 2) for h in hosts],
            "torchrun train.py",
            master_addr="10.0.0.1",
            port=29400,
            world_size=6,
        )
    )
    # every rank produced a result; only the healthy ones produced a job handle
    assert [r.host for r in results] == ["a", "b", "c"]
    assert [h.host for h in handles] == ["a", "c"]
    failed = results[1]
    assert failed.ok is False and "launch failed" in failed.stderr


def test_collab_assigns_ranks_in_order_over_nproc():
    runner = _CollabRunner()
    hosts = [Host(name=n, login="u", asset_id="A") for n in ("a", "b")]
    asyncio.run(
        JobManager(runner).submit_collab(
            [(hosts[0], 8), (hosts[1], 2)],
            "torchrun train.py",
            master_addr="10.0.0.1",
            port=29400,
            world_size=10,
        )
    )
    assert runner.launched == ["a", "b"]
    # ranks advance by nproc, and every rank is told the same world size
    assert "export JMS_RANK='0'" in runner.commands["a"]
    assert "export JMS_RANK='8'" in runner.commands["b"]
    assert "export JMS_NPROC='2'" in runner.commands["b"]
    assert "export JMS_WORLD_SIZE='10'" in runner.commands["b"]


@pytest.mark.parametrize("bad", ["", "xyz", "deadbeef00", "../etc", "ABCD1234", "a;b"])
def test_validate_job_id_rejects_bad(bad):
    with pytest.raises(ValueError):
        _validate_job_id(bad)


# --- incremental log polling --------------------------------------------------------


class _LogsRunner:
    """Stands in for the SSHRunner: replays one canned `size=<n>\\n<body>` response."""

    def __init__(self, stdout):
        self.stdout = stdout
        self.commands = []

    async def exec_on(self, host, command, *, timeout=60, workdir=True):
        self.commands.append(command)
        return ExecResult(host.name, 0, self.stdout, "")


def _job_logs(stdout, *, offset=0, max_bytes=65536):
    runner = _LogsRunner(stdout)
    host = Host(name="h1", login="u", asset_id="A")
    out = asyncio.run(JobManager(runner).logs(host, "deadbeef", offset=offset, max_bytes=max_bytes))
    return out, runner


def test_job_logs_clipped_read_does_not_skip_the_unread_middle():
    # A training log dwarfs the 64 KiB cap, so the read is clipped. `size` is the whole
    # file; resuming the next poll at `size` would silently drop everything between the
    # clip and EOF. `next_offset` is where this read actually stopped.
    out, _ = _job_logs("size=100000\n" + "x" * 10, offset=0, max_bytes=10)
    assert out["size"] == 100000
    assert out["next_offset"] == 10
    assert out["offset"] == 0
    assert len(out["log"]) == 10


def test_job_logs_clipped_read_advances_from_a_mid_file_offset():
    out, runner = _job_logs("size=100000\n" + "y" * 2048, offset=40, max_bytes=2048)
    assert out["next_offset"] == 40 + 2048
    # the remote must resume at the caller's offset, not at the start of the file
    assert "tail -c +41" in runner.commands[0]


def test_job_logs_next_offset_reaches_eof_when_the_rest_fits():
    out, _ = _job_logs("size=120\n" + "z" * 80, offset=40, max_bytes=65536)
    assert out["next_offset"] == 120 == out["size"]


def test_job_logs_missing_file_advances_nowhere():
    out, _ = _job_logs("size=0\n", offset=0, max_bytes=65536)
    assert out["log"] == ""
    assert out["size"] == 0
    assert out["next_offset"] == 0


# --- shared collab job id -----------------------------------------------------------


def test_collab_hands_every_rank_one_shared_job_id():
    """A collective answers to a single id, which is what the lifecycle tools take.

    The ranks live on different hosts, so one id cannot collide, and it is the only
    thing that lets `jms_cluster_cancel` stop every rank without the caller pairing
    per-rank ids to hosts first.
    """
    runner = _CollabRunner()
    hosts = [Host(name=n, login="u", asset_id="A") for n in ("a", "b")]
    handles, _ = asyncio.run(
        JobManager(runner).submit_collab(
            [(h, 2) for h in hosts],
            "torchrun train.py",
            master_addr="10.0.0.1",
            port=29400,
            world_size=4,
        )
    )
    assert len({h.job_id for h in handles}) == 1
    assert len({h.session for h in handles}) == 1
    assert [h.host for h in handles] == ["a", "b"]


# --- bounded wait / parallel cancel -------------------------------------------------


class _StatusRunner:
    """Answers status polls from a per-host script of statuses (the last one repeats)."""

    def __init__(self, script):
        self.script = {host: list(seq) for host, seq in script.items()}
        self.commands = []

    async def exec_on(self, host, command, *, timeout=60, workdir=False):
        self.commands.append((host.name, command))
        if "kill-session" in command:
            return ExecResult(host.name, 0, "killed=yes", "")
        seq = self.script.setdefault(host.name, ["unknown"])
        status = seq.pop(0) if len(seq) > 1 else seq[0]
        return ExecResult(
            host.name, 0, f"status={status}\nexit_code=\npid=1\nlog_bytes=0\nalive=yes\n", ""
        )


def _pair(name="a"):
    return Host(name=name, login="u", asset_id="A")


def test_wait_returns_as_soon_as_every_job_is_terminal():
    # `running` first, so the loop must actually poll more than once before it agrees.
    runner = _StatusRunner({"a": ["running", "done"], "b": ["running", "failed"]})
    latest, finished = asyncio.run(
        JobManager(runner).wait([(_pair("a"), "abcd1234"), (_pair("b"), "abcd1234")],
                                timeout=5, interval=0.01)
    )
    assert finished is True
    assert [s["status"] for s in latest] == ["done", "failed"]


def test_wait_treats_missing_as_terminal():
    # A job directory that does not exist never will, so waiting on it can only time out.
    runner = _StatusRunner({"a": ["missing"]})
    _, finished = asyncio.run(
        JobManager(runner).wait([(_pair("a"), "abcd1234")], timeout=5, interval=0.01)
    )
    assert finished is True


def test_wait_times_out_without_cancelling_anything():
    # A timeout stops *watching*, not the job: the caller may want to wait again, and an
    # unexpected kill here would be the worst possible surprise.
    runner = _StatusRunner({"a": ["running"]})
    latest, finished = asyncio.run(
        JobManager(runner).wait([(_pair("a"), "abcd1234")], timeout=0.05, interval=0.01)
    )
    assert finished is False
    assert latest[0]["status"] == "running"
    assert all("kill-session" not in cmd for _, cmd in runner.commands)


def test_cancel_many_isolates_a_host_that_raises():
    class _Boom(_StatusRunner):
        async def exec_on(self, host, command, *, timeout=60, workdir=False):
            if host.name == "b":
                raise RuntimeError("ssh died")
            return await super().exec_on(host, command, timeout=timeout, workdir=workdir)

    runner = _Boom({"a": ["running"], "b": ["running"]})
    results = asyncio.run(
        JobManager(runner).cancel_many([(_pair("a"), "abcd1234"), (_pair("b"), "abcd1234")])
    )
    assert results[0]["ok"] is True and results[0]["killed"] is True
    assert results[1]["ok"] is False and "ssh died" in results[1]["error"]


def test_validate_job_id_reports_instead_of_raising():
    # The server tools validate an id the caller typed; a raise would become an opaque
    # tool error, so the public helper answers with a reason instead.
    assert validate_job_id("deadbeef") is None
    assert "invalid job id" in validate_job_id("nope")
