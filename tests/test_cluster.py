import pytest

from jms_bridge.cluster import (
    DEFAULT_RDZV_PORT,
    HostHealth,
    _parse_preflight,
    _preflight_command,
    resolve_rendezvous,
)
from jms_bridge.inventory import Host, InventoryError
from jms_bridge.platforms import PLATFORMS

NPU_WITH_FREE_PORT = """\
===PROBE===
| npu-smi 24.1.rc3                 Version: 24.1.rc3                                            |
| 0     910C                    | OK              | 92.3       45               0 / 0           |
| 0     0                       | 0000:01:00.0    | 0          0 / 65536                        |
| 1     910C                    | OK              | 88.1       44               0 / 0           |
| 1     0                       | 0000:02:00.0    | 0          0 / 65536                        |
===PORT===
PORT_FREE
===END===
"""

NPU_WITH_BUSY_PORT = NPU_WITH_FREE_PORT.replace("PORT_FREE", "PORT_BUSY")


def test_preflight_command_probes_and_checks_port():
    cmd = _preflight_command(PLATFORMS["ascend-910c"], 29400)
    assert "npu-smi info" in cmd
    assert "/dev/tcp/127.0.0.1/29400" in cmd
    assert "===PROBE===" in cmd and "===PORT===" in cmd


def test_parse_preflight_healthy_freepath():
    h = Host(name="a", platform="ascend-910c")
    health = _parse_preflight(h, PLATFORMS["ascend-910c"], 29400, NPU_WITH_FREE_PORT)
    assert health.reachable is True
    assert health.device_count == 2
    assert health.version == "24.1.rc3"
    assert health.port_free is True
    assert health.ok is True


def test_parse_preflight_busy_port_is_not_ok():
    h = Host(name="a", platform="ascend-910c")
    health = _parse_preflight(h, PLATFORMS["ascend-910c"], 29400, NPU_WITH_BUSY_PORT)
    assert health.port_free is False
    assert health.ok is False  # a bound rendezvous port means a stale run


def test_parse_preflight_probe_failure_is_flagged():
    h = Host(name="a", platform="cuda")
    body = "===PROBE===\n__PROBE_FAILED__\n===PORT===\nPORT_FREE\n===END===\n"
    health = _parse_preflight(h, PLATFORMS["cuda"], 29400, body)
    assert health.device_count is None
    assert "probe failed" in health.detail


def test_resolve_rendezvous_defaults_to_first_host():
    hosts = [Host(name="a", address="10.0.0.1"), Host(name="b", address="10.0.0.2")]
    rdv = resolve_rendezvous(hosts)
    assert rdv.master_host == "a"
    assert rdv.master_addr == "10.0.0.1"
    assert rdv.port == DEFAULT_RDZV_PORT


def test_resolve_rendezvous_explicit_master():
    hosts = [Host(name="a", address="10.0.0.1"), Host(name="b", address="10.0.0.2")]
    rdv = resolve_rendezvous(hosts, master="b", port=1234)
    assert rdv.master_host == "b"
    assert rdv.master_addr == "10.0.0.2"
    assert rdv.port == 1234


def test_resolve_rendezvous_falls_back_to_name_without_address():
    rdv = resolve_rendezvous([Host(name="ascend-910c")])
    assert rdv.master_addr == "ascend-910c"


def test_resolve_rendezvous_unknown_master_raises():
    with pytest.raises(InventoryError):
        resolve_rendezvous([Host(name="a")], master="ghost")


def test_resolve_rendezvous_empty_raises():
    with pytest.raises(InventoryError):
        resolve_rendezvous([])


def test_health_to_dict_roundtrip():
    h = HostHealth("a", "cuda", True, 8, "12.4", 29400, True)
    d = h.to_dict()
    assert d["device_count"] == 8 and d["port_free"] is True and d["host"] == "a"


# --- heterogeneous topology -------------------------------------------------------


def test_plan_islands_partitions_by_backend():
    from jms_bridge.cluster import plan_islands

    hosts = [
        Host(name="asc1", platform="ascend-910c"),
        Host(name="gpu1", platform="cuda"),
        Host(name="gpu2", platform="cuda"),
        Host(name="arm", platform="aarch64"),
    ]
    islands = plan_islands(hosts)
    assert [i.backend for i in islands] == ["hccl", "nccl", "gloo"]
    assert [h.name for h in islands[1].hosts] == ["gpu1", "gpu2"]
    assert islands[0].name == "island-0-hccl"
    assert islands[0].to_dict()["nodes"] == 1


def test_plan_islands_homogeneous_is_one_island():
    from jms_bridge.cluster import plan_islands

    hosts = [Host(name=f"g{i}", platform="cuda") for i in range(4)]
    islands = plan_islands(hosts)
    assert len(islands) == 1
    assert islands[0].backend == "nccl"
    assert islands[0].to_dict()["nodes"] == 4


def test_is_heterogeneous():
    from jms_bridge.cluster import is_heterogeneous

    assert is_heterogeneous([Host(name="a", platform="cuda")]) is False
    assert (
        is_heterogeneous(
            [Host(name="a", platform="cuda"), Host(name="b", platform="ascend-910c")]
        )
        is True
    )