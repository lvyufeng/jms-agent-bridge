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