import pytest

from jms_bridge.jobs import _validate_job_id, build_runner_script


def test_runner_records_pid_status_exit_code():
    script = build_runner_script("abcd1234", "echo hello", "source env.sh")
    assert "echo \"$$\" > pid" in script
    assert "echo running > status" in script
    assert "echo done > status" in script
    assert "echo failed > status" in script
    assert 'echo "$code" > exit_code' in script
    # env activation happens before the command
    assert script.index("source env.sh") < script.index("echo hello")


def test_runner_wraps_command_snippet():
    script = build_runner_script("abcd1234", "python train.py --epochs 3", None)
    assert "python train.py --epochs 3" in script


def test_validate_job_id_accepts_8_hex():
    _validate_job_id("deadbeef")
    _validate_job_id("0123abcd")


@pytest.mark.parametrize("bad", ["", "xyz", "deadbeef00", "../etc", "ABCD1234", "a;b"])
def test_validate_job_id_rejects_bad(bad):
    with pytest.raises(ValueError):
        _validate_job_id(bad)