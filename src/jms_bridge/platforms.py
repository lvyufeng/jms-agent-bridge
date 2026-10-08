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
    # per-chip row: "| 0  0  | 0000:01:00.0 | ..." — the PCI bus-id column is the anchor
    device_re=r"^\|\s*(\d+)\s+\d+\s+\|\s*[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:",
)

# --- NVIDIA (CUDA / NCCL) --------------------------------------------------------
CUDA = Platform(
    name="cuda",
    probe_cmd="nvidia-smi",
    dist_backend="nccl",
    dist_env={},
    version_re=r"CUDA Version:\s*([0-9][0-9A-Za-z._\-]*)",
    # GPU row: "|   0  NVIDIA GeForce ...  | 00000000:01:00.0 Off |" — anchor on the bus id
    device_re=r"^\|\s+(\d+)\s+.*?\|\s*[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:",
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