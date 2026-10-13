"""Occupancy parsing: which accelerators already hold a compute process.

This decides whether `jms_cluster_submit` refuses to launch on a host, so the two
failure modes that matter are a *false busy* (a legitimate relaunch is blocked) and an
unnoticed *unknown* (a parse we did not trust being read as "the cards are free").
"""

from jms_bridge.platforms import (
    AARCH64,
    ASCEND,
    CUDA,
    ENFLAME,
    GENERIC,
    HYGON_DCU,
    ILUVATAR,
    KUNLUNXIN,
    METAX,
    MOORE_THREADS,
    PLATFORMS,
    T_HEAD,
    TSINGMICRO,
)

# --- which platforms probe at all ---------------------------------------------------

def test_accelerator_platforms_have_an_occupancy_probe():
    assert ASCEND.occupancy_cmd and ASCEND.occupancy_re
    assert CUDA.occupancy_cmd and CUDA.occupancy_re
    # Ascend answers a free host with its table header, so a zero needs something to
    # recognize. CUDA answers nothing at all, which is already unknown.
    assert ASCEND.occupancy_known_re
    assert CUDA.occupancy_known_re is None


def test_cpu_platforms_have_no_occupancy_probe():
    # No accelerator to query: occupancy must stay unknown, never a fake zero, so the
    # cluster tools apply no gate on these hosts.
    assert AARCH64.occupancy_cmd is None and AARCH64.occupancy_re is None
    assert GENERIC.occupancy_cmd is None and GENERIC.occupancy_re is None
    assert AARCH64.parse_occupancy("16\n", device_count=1) is None


# --- Ascend: one row per (npu, chip, process) ---------------------------------------

_NPU_PROC_MEM = """\
NPU ID    Chip ID    Process id    Process name    Process memory(MB)
0         0          12345         python          1024
0         0          12346         python          512
1         0          12347         python          2048
"""


def test_ascend_counts_devices_not_processes():
    # Two processes on NPU 0 must count as one occupied card; counting rows would report
    # three busy cards on a two-chip host and trip the device_count guard below.
    assert ASCEND.parse_occupancy(_NPU_PROC_MEM, device_count=2) == 2


def test_ascend_header_only_is_a_known_zero():
    # A host with nothing running answers with just the header — that is a real "no
    # accelerator is busy", not an unknown, and must not block a launch. It is *the
    # header* that licenses the zero (see `occupancy_known_re`).
    header = "NPU ID    Chip ID    Process id    Process name    Process memory(MB)\n"
    assert ASCEND.parse_occupancy(header, device_count=2) == 0


def test_empty_output_is_unknown_not_zero():
    # An empty reply can mean "nothing running" *or* "the probe printed nothing useful".
    # Reporting 0 would claim the cards are free on no evidence.
    assert ASCEND.parse_occupancy("", device_count=2) is None
    assert ASCEND.parse_occupancy("   \n", device_count=2) is None


def test_a_parse_claiming_more_devices_than_exist_is_discarded():
    # The device_count guard: if the regex matched a format change rather than device
    # rows, it can invent more busy devices than the host has cards. Discard the parse
    # rather than report a number we know is wrong.
    rows = "".join(f"{i}  0  {1000 + i}  python  1\n" for i in range(9))
    assert ASCEND.parse_occupancy(rows, device_count=2) is None
    # ...and without a known device count there is nothing to check it against.
    assert ASCEND.parse_occupancy(rows) == 9


# --- NVIDIA: one CSV row per process, keyed by GPU UUID ------------------------------

_CUDA_APPS = (
    "GPU-2f8ab3c1-0000-0000-0000-000000000001\n"
    "GPU-2f8ab3c1-0000-0000-0000-000000000001\n"
    "GPU-77de10aa-0000-0000-0000-000000000002\n"
)


def test_cuda_counts_distinct_gpu_uuids():
    assert CUDA.parse_occupancy(_CUDA_APPS, device_count=4) == 2


def test_cuda_empty_reply_stays_unknown():
    assert CUDA.parse_occupancy("\n", device_count=4) is None


def test_cuda_one_process_on_one_gpu():
    assert CUDA.parse_occupancy("GPU-2f8ab3c1-0000-0000-0000-000000000001\n", device_count=1) == 1


def test_an_unrecognized_body_is_unknown_not_free():
    # The failure this guards: a non-empty reply with zero matching rows used to parse as
    # a confident 0 ("the cards are free"). It is equally what a drifted vendor layout or
    # a stray warning looks like, so trusting it would report a busy host as free and let
    # a second launch land on it. The device_count guard cannot help here (there is no
    # count to check) and the empty-body check never fires (the body is not empty).
    bordered = (
        "| NPU  | Chip | Process | Name   | Mem  |\n"
        "| 0    | 0    | 777     | python | 1024 |\n"
    )
    assert ASCEND.parse_occupancy(bordered, device_count=2) is None
    warning = "WARNING: device 0 is in use by another process\n"
    assert ASCEND.parse_occupancy(warning, device_count=1) is None
    # ...and a body that *is* recognizable still yields the trusted zero.
    assert ASCEND.parse_occupancy("NPU ID  Chip ID  Process id  Name  Mem\n") == 0


def test_probe_failure_text_does_not_read_as_occupancy():
    # The remote probe captures stderr too, so a driver that refuses to answer has its
    # error text inside the section. Nothing in it matches the regex, and neither body is
    # recognizable as the tool's output, so both degrade to unknown rather than to "free".
    assert CUDA.parse_occupancy("NVIDIA-SMI has failed because it couldn't communicate\n") is None
    assert ASCEND.parse_occupancy("npu-smi: command not found\n") is None


# --- domestic accelerators -------------------------------------------------------
# Probe-only profiles: they report version + device count but carry no occupancy probe,
# so the cluster tools apply no busy gate on them. The samples below are trimmed from
# real SMI output on this fleet — the regexes are pinned to those layouts.

_DOMESTIC = [HYGON_DCU, METAX, ENFLAME, KUNLUNXIN, T_HEAD, ILUVATAR, MOORE_THREADS, TSINGMICRO]


def test_every_domestic_profile_has_a_probe_and_no_occupancy_gate():
    for plat in _DOMESTIC:
        assert plat.probe_cmd, plat.name
        assert plat.device_re, plat.name
        # No per-process query yet: occupancy must stay unknown, never a fake zero.
        assert plat.occupancy_cmd is None, plat.name
        assert plat.occupancy_re is None, plat.name
        assert plat.parse_occupancy("0  0  python\n", device_count=1) is None, plat.name


def test_domestic_profiles_are_registered_under_their_names_and_aliases():
    for alias in (
        "hygon-dcu", "hygon", "metax", "enflame", "kunlunxin", "kunlun",
        "thead", "t-head", "iluvatar", "moore-threads", "mthreads", "tsingmicro", "tsm",
    ):
        assert alias in PLATFORMS, alias


def test_hygon_dcu_probe():
    # `hy-smi` default table has no version line, so only the device count is extracted.
    out = (
        "HCU     Temp     AvgPwr     Perf     PwrCap     VRAM%      HCU%\n"
        "0       51.0C    95.0W      auto     1000.0W    0%         0.0%\n"
        "1       51.0C    82.0W      auto     1000.0W    0%         0.0%\n"
    )
    assert HYGON_DCU.parse_probe(out)["device_count"] == 2


def test_metax_probe():
    out = (
        "mx-smi  version: 2.3.1\n"
        "| MX-SMI 2.3.1                       Kernel Mode Driver Version: 3.8.1            |\n"
        "| MACA Version: 3.7.1.5              BIOS Version: 1.29.1.0                       |\n"
        "| 0     MetaX C550 | 0           Off | 0000:2b:00.0        | 0%          Disabled |\n"
        "| 1     MetaX C550 | 1           Off | 0000:3a:00.0        | 91%         Disabled |\n"
    )
    parsed = METAX.parse_probe(out)
    assert parsed["version"] == "3.8.1"
    assert parsed["device_count"] == 2


def test_enflame_probe():
    out = (
        "| EFSMI: 1.9.10                Driver Ver: 1.9.10                             |\n"
        "| 0      ZIXIAOC200           | 35.6.5            | 00:23:00.0  Enable        |\n"
        "| 53℃    LP0      132W / 300W | 42976MiB  Disable | 0%          A0B1J60510343 |\n"
        "| 1      ZIXIAOC200           | 35.6.5            | 00:33:00.0  Enable        |\n"
    )
    parsed = ENFLAME.parse_probe(out)
    assert parsed["version"] == "1.9.10"
    # The temperature line ("| 53℃ ...") must not count as a device row.
    assert parsed["device_count"] == 2


def test_kunlunxin_probe():
    out = (
        "| XPU-SMI               Driver Version: 5.0.21.43    XPU-RT Version: 5.0.21   |\n"
        "|   0  P800 OAM           N/A   | 00000000:03:00.0 N/A |                    0 |\n"
        "|   1  P800 OAM           N/A   | 00000000:16:00.0 N/A |                    0 |\n"
    )
    parsed = KUNLUNXIN.parse_probe(out)
    assert parsed["version"] == "5.0.21.43"
    assert parsed["device_count"] == 2


def test_thead_probe():
    out = (
        "| PPU-SMI 1.28          Driver Version: 1.3.2-d7f5a2  HGGC Version: 13.0        |\n"
        "| 0  PPU-ZW810E        N/A        | 00000000:3C:00.0     |                    0 |\n"
    )
    parsed = T_HEAD.parse_probe(out)
    assert parsed["version"] == "1.3.2-d7f5a2"
    assert parsed["device_count"] == 1


def test_iluvatar_probe():
    out = (
        "|  IX-ML: 4.4.0            Driver Version: 4.5.0          CUDA Version: 10.2     |\n"
        "| 0    Iluvatar BI-V150            | 00000000:13:00.0     | 1500MHz   1600MHz    |\n"
        "| 1    Iluvatar BI-V150            | 00000000:16:00.0     | 1600MHz   1600MHz    |\n"
    )
    parsed = ILUVATAR.parse_probe(out)
    assert parsed["version"] == "4.5.0"
    assert parsed["device_count"] == 2


def test_moore_threads_probe():
    out = (
        "    mthreads-gmi:2.3.2           Driver Version:3.3.5-server\n"
        "0    MTT S5000            |00000000:18:00.0    |0%    0MiB(81920MiB)\n"
        "1    MTT S5000            |00000000:3a:00.0    |0%    0MiB(81920MiB)\n"
    )
    parsed = MOORE_THREADS.parse_probe(out)
    assert parsed["version"] == "3.3.5-server"
    assert parsed["device_count"] == 2


def test_tsingmicro_counts_chips_on_both_row_shapes():
    # The first chip of each card shares its line with the card name; the rest stand
    # alone. Anchoring on the row start misses one chip per card (the bug this pins).
    out = (
        "| SMI V0.29842.5.01                          Chip count: 8                     |\n"
        "|  Card  Name              | Chip    Bus-Id      Temp   Pwr-Usage      Mem-Usage |\n"
        "|  0    TX8110-256-00      |  0    0000:d1:00.0   45C   68W / 200W    8230M     |\n"
        "|                          |  1    0000:cf:00.0   44C   64W / 200W    8224M     |\n"
        "|  1    TX8110-256-00      |  2    0000:4c:00.0   48C   67W / 200W    8224M     |\n"
        "|                          |  3    0000:5d:00.0   43C   64W / 200W    8224M     |\n"
        "|=============================================================================|\n"
        "|   0     1212884    python                                  2M               |\n"
    )
    parsed = TSINGMICRO.parse_probe(out)
    assert parsed["version"] == "0.29842.5.01"
    # 4 chips across 2 cards; the process row below the table is not a device.
    assert parsed["device_count"] == 4
