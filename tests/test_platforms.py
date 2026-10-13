"""Occupancy parsing: which accelerators already hold a compute process.

This decides whether `jms_cluster_submit` refuses to launch on a host, so the two
failure modes that matter are a *false busy* (a legitimate relaunch is blocked) and an
unnoticed *unknown* (a parse we did not trust being read as "the cards are free").
"""

from jms_bridge.platforms import AARCH64, ASCEND, CUDA, GENERIC

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
