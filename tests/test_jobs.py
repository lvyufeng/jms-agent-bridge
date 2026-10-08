import pytest

from jms_bridge.jobs import _validate_job_id, build_runner_script


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


def test_validate_job_id_accepts_8_hex():
    _validate_job_id("deadbeef")
    _validate_job_id("0123abcd")


@pytest.mark.parametrize("bad", ["", "xyz", "deadbeef00", "../etc", "ABCD1234", "a;b"])
def test_validate_job_id_rejects_bad(bad):
    with pytest.raises(ValueError):
        _validate_job_id(bad)