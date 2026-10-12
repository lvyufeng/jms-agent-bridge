import asyncio

import pytest

from jms_bridge.inventory import Host
from jms_bridge.jobs import JobManager, _validate_job_id, build_runner_script
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
