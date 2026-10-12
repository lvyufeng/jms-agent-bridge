"""Accelerator platform profiles.

One adapter per accelerator family. Each knows how to *probe* a host (liveness +
identity + driver/runtime version) and how to *launch* a distributed run there.
This is the layer that makes a heterogeneous Ascend + CUDA cluster look uniform to
the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Platform:
    name: str
    #: Short shell command that reports liveness + accelerator identity.
    probe_cmd: str
    #: Launcher for a distributed run (``torchrun`` on both, different backends).
    dist_backend: str
    #: Env vars that must be set for distributed comms on this platform.
    dist_env: dict[str, str] = field(default_factory=dict)
    #: Regexes applied to probe output to extract (driver/runtime version, device count).
    version_re: str | None = None
    device_re: str | None = None
    #: Command that lists which accelerators currently hold a compute process, and a
    #: regex whose first group is the *device* each line refers to. ``None`` means the
    #: platform has no occupancy probe (CPU-only boards) and no occupancy gate applies.
    occupancy_cmd: str | None = None
    occupancy_re: str | None = None
    #: Regex that must match the reply before a **zero** count is trusted. A non-empty
    #: body with no recognizable device row is *not* evidence that the cards are free —
    #: it is equally what a drifted vendor format or a stray warning looks like. Trusting
    #: it would report a genuinely busy host as free and let a second launch land on it,
    #: so such a body degrades to unknown instead. ``None`` means no zero is trusted for
    #: this platform; a positive count never needs this, because the matched rows *are*
    #: the evidence.
    occupancy_known_re: str | None = None

    def parse_probe(self, output: str) -> dict[str, str | int | None]:
        """Best-effort extraction of a version string and device count from probe output."""
        result: dict[str, str | int | None] = {"raw": output.strip()[:2000]}
        if self.version_re:
            m = re.search(self.version_re, output, re.MULTILINE)
            if m:
                result["version"] = m.group(1) if m.groups() else m.group(0)
        if self.device_re:
            count = len(re.findall(self.device_re, output, re.MULTILINE))
            result["device_count"] = count
        return result

    def parse_occupancy(self, output: str, *, device_count: int | None = None) -> int | None:
        """Count distinct accelerators currently holding a compute process.

        Returns ``None`` for "unknown" — no probe, empty output, or a parse we do not
        trust. That distinction is load-bearing: the cluster tools only raise the busy
        flag on a *positive* count, so an unknown (a missing driver tool, a vendor
        output format this regex does not match) degrades to today's behaviour rather
        than blocking a host on a bad guess. Two guards keep a bad parse from reading as
        "the cards are free": ``device_count`` (more occupied devices than the probe saw
        means the match is not device rows) and ``occupancy_known_re`` (a zero is only
        believed when the reply is recognizably the expected tool's output).
        """
        if not self.occupancy_re or not output.strip():
            return None
        # Dedupe: a query that lists one row per *process* repeats a device when it
        # holds several, and "how many cards are busy" must count cards, not processes.
        devices = {
            (m.group(1) if m.groups() else m.group(0)).strip()
            for m in re.finditer(self.occupancy_re, output, re.MULTILINE)
        }
        devices.discard("")
        if device_count is not None and len(devices) > device_count:
            return None
        if not devices and not (
            self.occupancy_known_re
            and re.search(self.occupancy_known_re, output, re.MULTILINE)
        ):
            # Nothing matched and this does not look like the tool's output at all —
            # the honest answer is "unknown", never "nothing is running".
            return None
        return len(devices)


# --- Ascend (CANN / torch_npu / HCCL) --------------------------------------------
# `npu-smi info` prints a table; the "Version" line carries the driver/CANN version
# and each accelerator contributes one line whose first field is the NPU id.
ASCEND = Platform(
    name="ascend-910c",
    probe_cmd="npu-smi info",
    dist_backend="hccl",
    dist_env={
        "ASCEND_RT_VISIBLE_DEVICES": "all",
    },
    version_re=r"(?:Version|version)\s*[:=]?\s*([0-9][0-9A-Za-z._\-]+)",
    # Per-chip row, anchored on the PCI bus-id so it survives npu-smi column changes
    # (26.x emits "| 0 | 0000:C1:00.0 |", 24.x emitted "| 0  0 | 0000:01:00.0 |").
    device_re=r"^\|\s*(\d+)\s+.*?[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:",
    # Per-process memory table: one row per process, first column is the NPU id. The
    # header row is non-numeric so it never matches; the device_count sanity check in
    # `parse_occupancy` throws the whole parse away as unknown if a format change makes
    # this match something that is not a device row. A host with nothing running answers
    # with the header alone, so the header is also what licenses the zero: without it
    # (a bordered 26.x layout, a warning in place of the table) the reply is unknown,
    # not "free".
    occupancy_cmd="npu-smi info -t proc-mem",
    occupancy_re=r"^\s*(\d+)\s+\d+\s+\d+\s+\S",
    occupancy_known_re=r"NPU\s+ID\b.*Chip\s+ID",
)

# --- NVIDIA (CUDA / NCCL) --------------------------------------------------------
CUDA = Platform(
    name="cuda",
    probe_cmd="nvidia-smi",
    dist_backend="nccl",
    dist_env={},
    version_re=r"CUDA Version:\s*([0-9][0-9A-Za-z._\-]*)",
    # GPU row: "|   0  NVIDIA GeForce RTX 4090  On  |   00000000:1B:00.0 Off |"
    device_re=r"^\|\s*(\d+)\s+.*?[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:",
    # One CSV row per running compute process carrying the GPU UUID it holds; several
    # processes on one GPU repeat the UUID, which `parse_occupancy` dedupes. With no
    # processes this query prints nothing, which already parses as unknown, so there is
    # no "recognizably free" reply to license a zero with — and a non-empty reply that
    # matches no UUID is a warning or a driver error, not an empty GPU list.
    occupancy_cmd="nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader",
    occupancy_re=r"^\s*(GPU-[0-9A-Fa-f-]+)\s*$",
)

# --- aarch64 board (CPU / small accelerator) -------------------------------------
AARCH64 = Platform(
    name="aarch64",
    probe_cmd="uname -a && nproc",
    dist_backend="gloo",
    dist_env={},
    version_re=r"Linux\s+\S+\s+([0-9][0-9A-Za-z._\-]*)",
    device_re=r"^\s*(\d+)\s*$",  # nproc prints a bare core count
)

GENERIC = Platform(
    name="generic",
    probe_cmd="uname -a && (nproc || true)",
    dist_backend="gloo",
    dist_env={},
    version_re=r"Linux\s+\S+\s+([0-9][0-9A-Za-z._\-]*)",
    device_re=None,
)

PLATFORMS: dict[str, Platform] = {
    "ascend-910c": ASCEND,
    "ascend": ASCEND,
    "cuda": CUDA,
    "nvidia": CUDA,
    "aarch64": AARCH64,
    "arm64": AARCH64,
    "generic": GENERIC,
}