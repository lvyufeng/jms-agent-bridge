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


def test_plan_islands_partitions_by_collective_engine():
    from jms_bridge.cluster import plan_islands

    hosts = [
        Host(name="asc1", platform="ascend-910c"),
        Host(name="gpu1", platform="cuda"),
        Host(name="gpu2", platform="cuda"),
        Host(name="arm", platform="aarch64"),
    ]
    islands = plan_islands(hosts)
    assert [i.collective for i in islands] == ["hccl", "nccl", "gloo"]
    assert [h.name for h in islands[1].hosts] == ["gpu1", "gpu2"]
    assert islands[0].name == "island-0-hccl"
    assert islands[0].to_dict()["nodes"] == 1


def test_plan_islands_homogeneous_is_one_island():
    from jms_bridge.cluster import plan_islands

    hosts = [Host(name=f"g{i}", platform="cuda") for i in range(4)]
    islands = plan_islands(hosts)
    assert len(islands) == 1
    assert islands[0].collective == "nccl"
    assert islands[0].to_dict()["nodes"] == 4


def test_two_vendors_that_both_say_nccl_do_not_share_an_island():
    # The bug this pins: partitioning on `dist_backend` grouped every NVIDIA-ABI vendor
    # under the string "nccl", so a mixed NVIDIA + Kunlunxin selection was planned as ONE
    # collective — a launch that cannot work, because each vendor ships its own
    # libnccl.so.2 and they do not rendezvous with each other or with NVIDIA's.
    from jms_bridge.cluster import is_heterogeneous, plan_islands
    from jms_bridge.platforms import CUDA, ILUVATAR, KUNLUNXIN

    assert CUDA.dist_backend == ILUVATAR.dist_backend == KUNLUNXIN.dist_backend == "nccl"
    assert len({CUDA.collective, ILUVATAR.collective, KUNLUNXIN.collective}) == 3

    hosts = [
        Host(name="gpu", platform="cuda"),
        Host(name="klx", platform="kunlunxin"),
        Host(name="ilu", platform="iluvatar"),
    ]
    islands = plan_islands(hosts)
    assert [len(i.hosts) for i in islands] == [1, 1, 1]
    assert is_heterogeneous(hosts) is True


def test_hosts_that_genuinely_share_an_engine_still_merge():
    # ...and the opposite failure: the key must not split hosts that *can* collect
    # together. Both CPU platforms run gloo over the same transport.
    from jms_bridge.cluster import is_heterogeneous, plan_islands

    hosts = [Host(name="arm", platform="aarch64"), Host(name="x86", platform="generic")]
    islands = plan_islands(hosts)
    assert len(islands) == 1
    assert islands[0].collective == "gloo"
    assert is_heterogeneous(hosts) is False


def test_is_heterogeneous():
    from jms_bridge.cluster import is_heterogeneous

    assert is_heterogeneous([Host(name="a", platform="cuda")]) is False
    assert (
        is_heterogeneous(
            [Host(name="a", platform="cuda"), Host(name="b", platform="ascend-910c")]
        )
        is True
    )


# --- device occupancy -------------------------------------------------------------


def _with_occupancy(body: str) -> str:
    """The two-chip Ascend reply with an occupancy section spliced in."""
    return NPU_WITH_FREE_PORT.replace(
        "===PORT===\n", f"===OCCUPANCY===\n{body}===PORT===\n"
    )


def test_preflight_command_adds_occupancy_only_where_there_is_a_probe():
    ascend = _preflight_command(PLATFORMS["ascend-910c"], 29400)
    cuda = _preflight_command(PLATFORMS["cuda"], 29400)
    generic = _preflight_command(PLATFORMS["generic"], 29400)
    # The accelerators get the section, fenced so a missing vendor tool cannot abort
    # the whole preflight and hide the port result.
    assert "===OCCUPANCY===" in ascend and "__OCCUPANCY_FAILED__" in ascend
    assert "npu-smi info -t proc-mem" in ascend
    assert "nvidia-smi --query-compute-apps=gpu_uuid" in cuda
    # A CPU-only board has nothing to query; an empty section would parse as unknown
    # anyway, so it is not sent at all.
    assert "===OCCUPANCY===" not in generic
    # Wire order: probe, then occupancy, then port.
    assert (
        ascend.index("===PROBE===")
        < ascend.index("===OCCUPANCY===")
        < ascend.index("===PORT===")
    )


def test_parse_preflight_reports_how_many_cards_are_busy():
    h = Host(name="a", platform="ascend-910c")
    body = _with_occupancy("NPU ID  Chip  pid  name  mem\n0  0  777  python  1024\n")
    health = _parse_preflight(h, PLATFORMS["ascend-910c"], 29400, body)
    assert health.busy_devices == 1
    assert health.busy is True
    # Occupancy is advisory: the helper's own previous run may hold the card, so a busy
    # host can still be ranked — the *caller* decides via `allow_busy`.
    assert health.ok is True


def test_parse_preflight_without_an_occupancy_section_is_unknown_not_free():
    # The fixtures above predate the section entirely (an older host mid-upgrade, or a
    # truncated reply). Unknown must not be reported as "the cards are free".
    h = Host(name="a", platform="ascend-910c")
    health = _parse_preflight(h, PLATFORMS["ascend-910c"], 29400, NPU_WITH_FREE_PORT)
    assert health.busy_devices is None
    assert health.busy is False
    assert health.ok is True


def test_parse_preflight_occupancy_probe_failure_is_unknown():
    # `|| echo` fired, so stderr text ("command not found") must not be parsed — the
    # absence of a table is not evidence that nothing is running.
    h = Host(name="a", platform="ascend-910c")
    body = _with_occupancy("__OCCUPANCY_FAILED__\nnpu-smi: command not found\n")
    health = _parse_preflight(h, PLATFORMS["ascend-910c"], 29400, body)
    assert health.busy_devices is None
    assert health.busy is False


def test_parse_preflight_discards_an_occupancy_parse_that_exceeds_the_card_count():
    # Nine occupied devices on a two-chip host is a regex matching something that is
    # not device rows; discard it rather than report numbers we know are wrong.
    h = Host(name="a", platform="ascend-910c")
    rows = "".join(f"{i}  0  {1000 + i}  python  1\n" for i in range(9))
    health = _parse_preflight(h, PLATFORMS["ascend-910c"], 29400, _with_occupancy(rows))
    assert health.device_count == 2
    assert health.busy_devices is None
    assert health.busy is False


def test_parse_preflight_header_only_occupancy_is_a_known_free():
    h = Host(name="a", platform="ascend-910c")
    body = _with_occupancy("NPU ID  Chip ID  Process id  Name  Mem\n")
    health = _parse_preflight(h, PLATFORMS["ascend-910c"], 29400, body)
    assert health.busy_devices == 0
    assert health.busy is False


def test_parse_preflight_unrecognized_occupancy_is_unknown_not_free():
    # A non-empty table in a layout the regex no longer matches must not read as "no
    # cards busy" — that is the one way this feature could turn a busy host green.
    h = Host(name="a", platform="ascend-910c")
    bordered = "| 0 | 0 | 777 | python | 1024 |\n"
    health = _parse_preflight(h, PLATFORMS["ascend-910c"], 29400, _with_occupancy(bordered))
    assert health.busy_devices is None
    assert health.busy is False
    assert health.ok is True  # advisory: unknown never blocks, it is just not "free"


def test_busy_is_only_true_on_a_positive_count():
    base = ("a", "cuda", True, 8, "12.4", 29400, True)
    assert HostHealth(*base, busy_devices=None).busy is False
    assert HostHealth(*base, busy_devices=0).busy is False
    assert HostHealth(*base, busy_devices=1).busy is True


def test_health_to_dict_carries_occupancy():
    d = HostHealth("a", "cuda", True, 8, "12.4", 29400, True, busy_devices=3).to_dict()
    assert d["busy_devices"] == 3 and d["busy"] is True
    d = HostHealth("a", "cuda", True, 8, "12.4", 29400, True).to_dict()
    assert d["busy_devices"] is None and d["busy"] is False


def test_sections_treats_a_missing_marker_as_empty():
    from jms_bridge.cluster import _sections

    sections = _sections(NPU_WITH_FREE_PORT)
    assert sections["OCCUPANCY"] == ""  # absent → unknown, never a healthy value
    assert "PORT_FREE" in sections["PORT"]
    assert "npu-smi" in sections["PROBE"]