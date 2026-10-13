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

# --- domestic accelerators -------------------------------------------------------
# One profile per vendor SMI. Each knows how to report liveness, runtime version,
# device count *and* which cards already hold a compute process, so the cluster tools
# apply a real busy gate on these hosts instead of falling back to "unknown".
#
# Occupancy comes in two shapes and the regex reflects which the vendor emits:
#  * a per-process row that carries the device id as its first field (metax, enflame,
#    moore-threads, tsingmicro) — ``parse_occupancy`` dedupes repeated device ids, so
#    several processes on one card still count as one busy card;
#  * the nvidia-smi lineage (kunlunxin, thead, iluvatar), where processes are nested
#    under a device header (``XPU``/``PPU``/``GPU <bus-id>``) with no id on the row —
#    there the regex matches a header and its first nested ``Process ID`` together, so
#    each busy device contributes exactly one match. hygon keys on the ``HCU Index``
#    each KFD process reports.
#
# Where the vendor prints a recognizable "nothing running" reply it is quoted as
# ``occupancy_known_re`` and licenses a trusted **zero**; where it does not (a busy
# host is the only sample we have), the zero stays *unknown* — a positive count never
# needs licensing, because the matched rows are the evidence.
#
# The ``dist_backend`` / ``dist_env`` values below are the vendors' commonly cited
# torchrun backends but are **UNVERIFIED on this fleet** — confirm before relying on
# a multi-node launch. Each row sample the regexes were written against is quoted
# inline; the parsers are pinned by tests/test_platforms.py.

HYGON_DCU = Platform(  # 海光 DCU — `hy-smi` (not on PATH; installed under /opt/hyhal/bin)
    name="hygon-dcu",
    probe_cmd="hy-smi 2>/dev/null || /opt/hyhal/bin/hy-smi 2>/dev/null || true",
    dist_backend="nccl",  # RCCL-based; UNVERIFIED
    dist_env={},
    version_re=None,  # `hy-smi` prints no version line in its default table
    # "0       51.0C    95.0W      auto     1000.0W    0%..."
    device_re=r"^\s*(\d+)\s+\d+\.\d+C\s+\d+\.\d+W\b",
    # `--showpids` lists one block per KFD process; the HCU each holds is on its
    # "HCU Index: ['2']" line. Dedupe on that index: 8 processes on 2 HCUs is 2 busy.
    occupancy_cmd=(
        "hy-smi --showpids 2>/dev/null || /opt/hyhal/bin/hy-smi --showpids 2>/dev/null || true"
    ),
    occupancy_re=r"HCU\s+Index:\s*\['?(\d+)'?\]",
)

METAX = Platform(  # 沐曦
    name="metax",
    probe_cmd="mx-smi",
    dist_backend="nccl",  # MACA stack; UNVERIFIED
    dist_env={},
    version_re=r"Kernel Mode Driver Version:\s*([0-9][0-9A-Za-z._\-]*)",
    # "| 0     MetaX C550 | 0           Off | 0000:2b:00.0        | 0%"
    device_re=r"^\|\s*(\d+)\s+MetaX\b",
    # `--show-all-process` prints one row per process: "|  0   2548673  python  0 |".
    # The device id is the first field (deduped); the second is the PID, which rules
    # the "|  GPU  PID  Process Name" header and any device row out.
    occupancy_cmd="mx-smi --show-all-process",
    occupancy_re=r"^\|\s*(\d+)\s+\d{2,}\s+\S",
    occupancy_known_re=r"no process found",
)

ENFLAME = Platform(  # 燧原
    name="enflame",
    probe_cmd="efsmi",
    dist_backend="eccl",  # UNVERIFIED
    dist_env={},
    version_re=r"Driver Ver:\s*([0-9][0-9A-Za-z._\-]*)",
    # "| 0      ZIXIAOC200           | 35.6.5            | 00:23:00.0  Enable"
    device_re=r"^\|\s*(\d+)\s+[A-Za-z]\S*\s*\|",
    # `pmon -c 1` prints one row per GCU: "0    1280324  12  5  python". A free card
    # answers "0    N/A   N/A  N/A  no process running on", whose PID field is not
    # numeric, so the row regex only matches an actually-held card.
    occupancy_cmd="efsmi pmon -c 1",
    occupancy_re=r"^\s*(\d+)\s+\d{2,}\s+\S",
    occupancy_known_re=r"no process running on",
)

KUNLUNXIN = Platform(  # 昆仑芯
    name="kunlunxin",
    probe_cmd="xpu-smi",
    dist_backend="bkcl",  # UNVERIFIED
    dist_env={},
    version_re=r"Driver Version:\s*([0-9][0-9A-Za-z._\-]*)",
    # "|   0  P800 OAM           N/A   | 00000000:03:00.0 N/A |"
    device_re=r"^\|\s*(\d+)\s+P800\b",
    # nvidia-smi-shaped: `-q` nests a "Processes" section under each "XPU <bus-id>"
    # header, with no id on the process rows. Match a header and its first nested
    # "Process ID" together (the newline-anchored guard stops the scan at the next
    # header, so an indented "XPU instance ID" line inside the block is not one). One
    # match per busy card, keyed on the bus-id. Idle answers "Processes : None".
    occupancy_cmd="xpu-smi -q",
    occupancy_re=r"XPU\s+([0-9A-Fa-f:.]+)\s*\n(?:(?!\nXPU\s)[\s\S])*?Process ID",
    occupancy_known_re=r"Processes\s*:\s*None",
)

T_HEAD = Platform(  # 平头哥
    name="thead",
    probe_cmd="ppu-smi",
    dist_backend="nccl",  # UNVERIFIED
    dist_env={},
    version_re=r"Driver Version:\s*([0-9][0-9A-Za-z._\-]*)",
    # "| 0  PPU-ZW810E        N/A        | 00000000:3C:00.0     |"
    device_re=r"^\|\s*(\d+)\s+PPU-",
    # PPU-SMI mirrors the XPU-SMI shape (see kunlunxin): per-"PPU <bus-id>" nested
    # Processes, idle "Processes : None". The header sits before the scan guard below.
    occupancy_cmd="ppu-smi -q",
    occupancy_re=r"PPU\s+([0-9A-Fa-f:.]+)\s*\n(?:(?!\nPPU\s)[\s\S])*?Process ID",
    occupancy_known_re=r"Processes\s*:\s*None",
)

ILUVATAR = Platform(  # 天数
    name="iluvatar",
    # `ixsmi` ships under /usr/local/corex/bin and only some hosts put it on PATH;
    # there it also needs the IXML library dir, else it dies on libixml.so. Try the
    # plain name first, then the install path with LD_LIBRARY_PATH set.
    probe_cmd=(
        "export LD_LIBRARY_PATH=/usr/local/corex/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; "
        "ixsmi 2>/dev/null || /usr/local/corex/bin/ixsmi 2>/dev/null || true"
    ),
    dist_backend="cncl",  # UNVERIFIED
    dist_env={},
    version_re=r"Driver Version:\s*([0-9][0-9A-Za-z._\-]*)",
    # "| 0    Iluvatar BI-V150            | 00000000:13:00.0     | 1500MHz"
    device_re=r"^\|\s*(\d+)\s+Iluvatar\b",
    # ix-smi is nvidia-smi-shaped; `-d PIDS` nests a Processes section under each
    # "GPU <bus-id>" header. Same bus-id-keyed match as kunlunxin/thead, and the same
    # PATH/LD fallback as the probe. No free-host reply was captured, so a zero stays
    # *unknown* (a busy host is what the gate is for; not trusting an absent reply
    # costs nothing here).
    occupancy_cmd=(
        "export LD_LIBRARY_PATH=/usr/local/corex/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; "
        "ixsmi -q -d PIDS 2>/dev/null || /usr/local/corex/bin/ixsmi -q -d PIDS 2>/dev/null || true"
    ),
    occupancy_re=r"GPU\s+([0-9A-Fa-f:.]+)\s*\n(?:(?!\nGPU\s)[\s\S])*?Process ID",
)

MOORE_THREADS = Platform(  # 摩尔线程
    name="moore-threads",
    probe_cmd="mthreads-gmi",
    dist_backend="mccl",  # UNVERIFIED
    dist_env={},
    version_re=r"Driver Version:\s*([0-9][0-9A-Za-z._\-]*)",
    # "0    MTT S5000            |00000000:18:00.0    |0%    0MiB(81920MiB)"
    device_re=r"^\s*(\d+)\s+MTT\b",
    # The default summary carries a "Processes:" table: "0  2091833  python  40MiB".
    # Device id first field (deduped), PID the second — the MTT device rows start with
    # a name, not a 3+ digit pid, so they never match. Idle answers "No running
    # processes found".
    occupancy_cmd="mthreads-gmi",
    occupancy_re=r"^\s*(\d+)\s+\d{3,}\s+\S",
    occupancy_known_re=r"No running processes found",
)

TSINGMICRO = Platform(  # 清微 — `tsm_smi` reports chips, not cards; count chips (32 here)
    name="tsingmicro",
    probe_cmd="tsm_smi",
    dist_backend="nccl",  # UNVERIFIED — vendor backend unknown
    dist_env={},
    version_re=r"SMI V([0-9][0-9A-Za-z._\-]*)",
    # Chip rows come in two shapes: the first chip of each card shares its line with the
    # card name, the rest stand alone — so anchor on the bus-id, not the row start.
    #   "|  0    TX8110-256-00      |  0    0000:d1:00.0   45C ..."   (card + chip 0)
    #   "|                          |  1    0000:cf:00.0   44C ..."   (chip alone)
    device_re=r"\|\s*(?:\d+\s+\S+\s*\|\s*)?(\d+)\s+[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:",
    # The same default output carries a trailing "Processes:" table:
    # "|   0     1216475    python      2M |". The card id is the first field (deduped);
    # the second is the PID, which keeps the "| Card PID Process name" header and the
    # TX8110 card rows (name, not a pid, in the second field) from matching. No free
    # host was captured, so a zero stays unknown.
    occupancy_cmd="tsm_smi",
    occupancy_re=r"^\|\s*(\d+)\s+\d{2,}\s+\S",
)

PLATFORMS: dict[str, Platform] = {
    "ascend-910c": ASCEND,
    "ascend": ASCEND,
    "cuda": CUDA,
    "nvidia": CUDA,
    "aarch64": AARCH64,
    "arm64": AARCH64,
    "generic": GENERIC,
    # domestic accelerators introduced above
    "hygon-dcu": HYGON_DCU,
    "hygon": HYGON_DCU,
    "metax": METAX,
    "enflame": ENFLAME,
    "kunlunxin": KUNLUNXIN,
    "kunlun": KUNLUNXIN,
    "thead": T_HEAD,
    "t-head": T_HEAD,
    "iluvatar": ILUVATAR,
    "moore-threads": MOORE_THREADS,
    "mthreads": MOORE_THREADS,
    "tsingmicro": TSINGMICRO,
    "tsm": TSINGMICRO,
}