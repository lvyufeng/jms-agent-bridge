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
    guess_platform,
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
# Each profile probes version + device count *and* carries a per-vendor occupancy
# query, so the cluster tools apply a real busy gate. The samples below are trimmed
# from real SMI output on this fleet — the regexes are pinned to those layouts.

_DOMESTIC = [HYGON_DCU, METAX, ENFLAME, KUNLUNXIN, T_HEAD, ILUVATAR, MOORE_THREADS, TSINGMICRO]


def test_every_domestic_profile_has_a_probe_and_an_occupancy_query():
    for plat in _DOMESTIC:
        assert plat.probe_cmd, plat.name
        assert plat.device_re, plat.name
        assert plat.occupancy_cmd, plat.name
        assert plat.occupancy_re, plat.name


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


# --- domestic occupancy ----------------------------------------------------------
# Row carries the device id as its first field (hygon keys on HCU index, the rest on
# the SMI device/card id); dedupe means several processes on one card is one busy card.

_HYGON_PIDS = """\
PIDs for KFD processes:

PID: 1280324
\tPASID: 32832
\tHCU Index: ['2']
PID: 1280322
\tPASID: 32816
\tHCU Index: ['0']
"""

_METAX_BUSY = """\
| Process:
|  GPU                    PID         Process Name                 GPU Memory
|=================================================================================
|  0                  2548673         python                       0
|  0                  2572595         python                       0
|  1                  1234567         python                       0
"""

_METAX_IDLE = """\
| Process:
|  GPU                    PID         Process Name                 GPU Memory
|=================================================================================
|  no process found
+---------------------------------------------------------------------------------+
"""

_KUNLUN_BUSY = """\
Attached XPUs                             : 8
XPU 00000000:03:00.0
    Product Name                          : P800 OAM
    Processes
        XPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 1288826
            Type                          : C
XPU 00000000:16:00.0
    Product Name                          : P800 OAM
    Processes                             : None
"""

_KUNLUN_IDLE = """\
Attached XPUs                             : 8
XPU 00000000:03:00.0
    Processes                             : None
XPU 00000000:16:00.0
    Processes                             : None
"""

_THEAD_IDLE = """\
Attached PPUs                               : 16
PPU 00000000:3C:00.0
    Product Name                            : PPU-ZW810E
    Processes                               : None
PPU 00000000:36:00.0
    Processes                               : None
"""

_ILUVATAR_BUSY = """\
GPU 00000000:13:00.0
    Processes
        Process ID                              : 3274461
            Type                                : C
            Name                                : VLLM::Worker
        Process ID                              : 3271203
            Type                                : C
GPU 00000000:16:00.0
    Processes
        Process ID                              : 3274462
            Type                                : C
"""

_ILUVATAR_IDLE = """\
GPU 00000000:13:00.0
    Processes
GPU 00000000:16:00.0
    Processes
"""

_ENFLAME_IDLE = """\
*Dev Pid      DevMem  SysMem  Command
*Idx #        %       M       Name
0    N/A      N/A     N/A     no process running on
1    N/A      N/A     N/A     no process running on
"""

_ENFLAME_BUSY = """\
*Dev Pid      DevMem  SysMem  Command
*Idx #        %       M       Name
0    1280324  12      5       python
1    N/A      N/A     N/A     no process running on
"""

_MOORE_BUSY = """\
Processes:
ID   PID           Process name                           GPU Memory
                                                               Usage
+-------------------------------------------------------------------+
0    2091833       python                                      40MiB
0    2092171       ...rtualenvs/sglang-0.5.6/bin/python        40MiB
1    2093534       sglang::scheduler_DP0_TP0                78549MiB
"""

_MOORE_IDLE = """\
Processes:
ID   PID           Process name                           GPU Memory
                                                               Usage
+-------------------------------------------------------------------+
   No running processes found
---------------------------------------------------------------------
"""

_TSM_PROC = """\
|  Card   PID      Process name                            Memory Usage       |
|=============================================================================|
|   0     1212884    python                                  2M               |
|   0     1216475    python                                  2M               |
|  16     1954602    python                                  144M             |
|  17     1954601    python                                  38128M           |
"""


def test_hygon_counts_busy_hcus_not_processes():
    # Two processes on one HCU would be one busy card; here two distinct indices.
    assert HYGON_DCU.parse_occupancy(_HYGON_PIDS, device_count=8) == 2


def test_metax_dedupes_processes_per_device_and_licenses_the_zero():
    # Several processes on device 0 plus one on device 1 -> two busy cards.
    assert METAX.parse_occupancy(_METAX_BUSY, device_count=8) == 2
    assert METAX.parse_occupancy(_METAX_IDLE, device_count=8) == 0


def test_enflame_only_counts_a_held_card():
    # The idle rows answer "N/A" for the PID, which must not count as busy.
    assert ENFLAME.parse_occupancy(_ENFLAME_IDLE, device_count=8) == 0
    assert ENFLAME.parse_occupancy(_ENFLAME_BUSY, device_count=8) == 1


def test_kunlunxin_counts_busy_blocks_and_licenses_the_zero():
    # A "Processes : None" block must not count; the one with a Process ID must.
    assert KUNLUNXIN.parse_occupancy(_KUNLUN_BUSY, device_count=8) == 1
    assert KUNLUNXIN.parse_occupancy(_KUNLUN_IDLE, device_count=8) == 0


def test_thead_idle_reply_is_a_licensed_zero():
    assert T_HEAD.parse_occupancy(_THEAD_IDLE, device_count=16) == 0


def test_iluvatar_counts_busy_gpus_and_leaves_idle_unknown():
    assert ILUVATAR.parse_occupancy(_ILUVATAR_BUSY, device_count=16) == 2
    # No free-host reply was captured, so a GPU header with no process rows is
    # unknown (no gate) rather than a confident zero.
    assert ILUVATAR.parse_occupancy(_ILUVATAR_IDLE, device_count=16) is None


def test_moore_threads_dedupes_and_licenses_the_zero():
    assert MOORE_THREADS.parse_occupancy(_MOORE_BUSY, device_count=8) == 2
    assert MOORE_THREADS.parse_occupancy(_MOORE_IDLE, device_count=8) == 0


def test_tsingmicro_counts_busy_cards_not_processes():
    # Cards 0, 16, 17 are held (card 0 twice) -> three busy cards.
    assert TSINGMICRO.parse_occupancy(_TSM_PROC, device_count=32) == 3


def test_domestic_occupancy_respects_the_device_count_guard():
    # More "busy devices" than the probe saw means the match is not device rows.
    assert METAX.parse_occupancy(_METAX_BUSY, device_count=1) is None


def test_domestic_unrecognized_body_is_unknown_not_free():
    for plat in _DOMESTIC:
        assert plat.parse_occupancy("WARNING: driver busy\n", device_count=8) is None, plat.name
        assert plat.parse_occupancy("", device_count=8) is None, plat.name


# --- guessing a profile for an asset we were only told about --------------------------
# A wrong tag is quiet: a domestic accelerator tagged `cuda` probes `nvidia-smi`, finds
# nothing and reports a host with no accelerator at all. So the guess has two jobs —
# always answer with a profile that exists, and never answer silently when it is only a
# fallback.


def test_a_declared_alias_is_canonicalized_to_the_profiles_own_name():
    # JumpServer may say "NVIDIA", the profile is called `cuda` — and it is the profile's
    # name that must come out, because that is what an inventory stores.
    assert guess_platform(declared="NVIDIA") == ("cuda", None)
    assert guess_platform(declared="Ascend 910C") == ("ascend-910c", None)


def test_every_accepted_platform_alias_round_trips():
    # The answer is always a profile in the table, so the tag can never name a platform
    # the probes do not have.
    for alias in PLATFORMS:
        name, note = guess_platform(declared=alias)
        assert note is None and name in {p.name for p in PLATFORMS.values()}


def test_a_vendor_name_or_address_decides_when_the_declared_field_cannot():
    # JumpServer's platform list is free text and often unhelpful ("Linux" is its usual
    # default for any Unix host), so the asset's own name and address are the next best
    # evidence.
    assert guess_platform(declared="Linux", name="fl-moore-1") == ("moore-threads", None)
    assert guess_platform(declared="Linux", name="node-7", address="kunlun-8c") == (
        "kunlunxin",
        None,
    )


def test_a_declared_profile_wins_over_a_vendor_substring_in_the_name():
    # The declared field is scanned first, so a customer's free-text label that happens to
    # contain a vendor word cannot override a platform that really is known.
    assert guess_platform(declared="NVIDIA CUDA", name="摩尔线程-1") == ("cuda", None)


def test_no_evidence_falls_back_to_generic_with_a_note():
    name, note = guess_platform(declared="Plain Linux Board", name="node-7", address="10.0.0.9")
    assert name == "generic"
    assert note and "Plain Linux Board" in note and "name or address" in note


def test_an_absent_declared_field_is_still_a_noted_fallback():
    name, note = guess_platform()
    assert name == "generic"
    assert note and "declares no platform" in note
