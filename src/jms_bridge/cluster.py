"""Cluster-level coordination: preflight and rendezvous.

A distributed run fails for boring reasons long before it fails for interesting
ones: a host is unreachable, the accelerator isn't visible, or the rendezvous port
is already taken by a previous run that never died. :func:`preflight` checks all of
that across the cluster *before* a launch, and :func:`resolve_rendezvous` picks the
master address and port so every rank agrees.
"""

from __future__ import annotations

from dataclasses import dataclass

from .inventory import Host, InventoryError, wrap_command
from .platforms import Platform
from .ssh import SSHRunner

#: Default rendezvous port. Uses a fixed, non-ephemeral port so a stale process is
#: easy to spot; override per run when running clusters side by side.
DEFAULT_RDZV_PORT = 29400


@dataclass
class HostHealth:
    host: str
    platform: str
    reachable: bool
    device_count: int | None
    version: str | None
    port: int
    port_free: bool | None
    detail: str = ""
    #: How many accelerators currently hold a compute process, or ``None`` when that is
    #: unknown (no occupancy probe for this platform, or an unparseable reply).
    busy_devices: int | None = None

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "platform": self.platform,
            "reachable": self.reachable,
            "device_count": self.device_count,
            "version": self.version,
            "port": self.port,
            "port_free": self.port_free,
            "detail": self.detail,
            "busy_devices": self.busy_devices,
            "busy": self.busy,
        }

    @property
    def ok(self) -> bool:
        """Hard gate for a launch: reachable, and the rendezvous port is not taken.

        Deliberately does *not* include :attr:`busy`. Occupancy is advisory — a host
        whose cards are busy may be running the very job being resumed, and refusing
        to rank it would make a legitimate relaunch impossible. ``busy`` is reported
        so the caller can decide, and ``jms_cluster_submit`` takes an explicit
        ``allow_busy`` override.
        """
        return self.reachable and self.port_free is not False

    @property
    def busy(self) -> bool:
        """True only on a *positive* occupancy count — unknown (``None``) is not busy."""
        return bool(self.busy_devices)


#: Section markers in the preflight reply, in wire order.
_MARKERS = ("PROBE", "OCCUPANCY", "PORT", "END")


def _preflight_command(platform: Platform, port: int) -> str:
    """Probe the accelerator, sample device occupancy, and check the rendezvous port.

    The occupancy probe is chained between the identity probe and the port test so one
    round trip covers all three; it is fenced with ``|| echo`` like the identity probe
    so a host whose driver tools are missing still reports a (failed) section rather
    than aborting the whole script. Platforms with no occupancy probe skip the section
    entirely, and the parser then reports occupancy as unknown.
    """
    lines = [
        'echo "===PROBE==="',
        f"{{ {platform.probe_cmd}; }} 2>&1 || echo __PROBE_FAILED__",
    ]
    if platform.occupancy_cmd:
        lines += [
            'echo "===OCCUPANCY==="',
            f"{{ {platform.occupancy_cmd}; }} 2>&1 || echo __OCCUPANCY_FAILED__",
        ]
    lines += [
        'echo "===PORT==="',
        f"if (exec 3<>/dev/tcp/127.0.0.1/{int(port)}) 2>/dev/null; then "
        f"echo PORT_BUSY; else echo PORT_FREE; fi",
        'echo "===END==="',
    ]
    return "\n".join(lines)


def _sections(stdout: str) -> dict[str, str]:
    """Split a marker-delimited preflight reply into ``{NAME: body}``.

    A section that never arrived (an older host mid-upgrade, a truncated reply) maps to
    ``""``, which the caller reads as "unknown" — never as a healthy value.
    """
    out: dict[str, str] = {}
    for i, name in enumerate(_MARKERS):
        head = f"==={name}==="
        body = stdout.split(head, 1)[1] if head in stdout else ""
        for tail in _MARKERS[i + 1 :]:
            marker = f"==={tail}==="
            if marker in body:
                body = body.split(marker, 1)[0]
                break
        out[name] = body
    return out


def _parse_preflight(host: Host, platform: Platform, port: int, stdout: str) -> HostHealth:
    sections = _sections(stdout)
    probe = sections["PROBE"]
    after = sections["PORT"]
    port_state = None
    if "PORT_BUSY" in after:
        port_state = False
    elif "PORT_FREE" in after:
        port_state = True

    probe_failed = "__PROBE_FAILED__" in probe
    parsed = platform.parse_probe(probe) if (probe.strip() and not probe_failed) else {}
    device_count = parsed.get("device_count")

    occupancy = sections["OCCUPANCY"]
    busy_devices = None
    if occupancy.strip() and "__OCCUPANCY_FAILED__" not in occupancy:
        busy_devices = platform.parse_occupancy(occupancy, device_count=device_count)

    return HostHealth(
        host=host.name,
        platform=platform.name,
        reachable=True,
        device_count=device_count,
        version=parsed.get("version"),
        port=port,
        port_free=port_state,
        detail="accelerator probe failed" if probe_failed else "",
        busy_devices=busy_devices,
    )


async def preflight(
    runner: SSHRunner, hosts: list[Host], *, port: int = DEFAULT_RDZV_PORT, timeout: float = 45.0
) -> list[HostHealth]:
    """Check reachability, accelerator visibility and rendezvous-port availability."""
    import asyncio

    async def one(host: Host) -> HostHealth:
        platform = host.platform_def
        res = await runner.exec_on(
            host, _preflight_command(platform, port), timeout=timeout, workdir=False
        )
        if not res.ok and not res.stdout:
            return HostHealth(
                host=host.name,
                platform=platform.name,
                reachable=False,
                device_count=None,
                version=None,
                port=port,
                port_free=None,
                detail=(res.stderr or "unreachable").strip()[:300],
            )
        return _parse_preflight(host, platform, port, res.stdout)

    return list(await asyncio.gather(*(one(h) for h in hosts)))


@dataclass
class Rendezvous:
    master_host: str
    master_addr: str
    port: int

    def env(self) -> dict[str, str]:
        return {"JMS_MASTER_ADDR": self.master_addr, "JMS_MASTER_PORT": str(self.port)}

    def to_dict(self) -> dict:
        return {
            "master_host": self.master_host,
            "master_addr": self.master_addr,
            "port": self.port,
        }


def resolve_rendezvous(
    hosts: list[Host], *, master: str | None = None, port: int = DEFAULT_RDZV_PORT
) -> Rendezvous:
    """Pick the master host and its address.

    Every rank must agree on one address reachable from all of them. Prefer an
    explicit ``master`` host name; else the first host. The address is the host's
    ``address`` field (its IP/DNS as JumpServer knows it) and falls back to the host
    name.
    """
    if not hosts:
        raise InventoryError("cannot resolve a rendezvous for an empty host list")
    by_name = {h.name: h for h in hosts}
    if master:
        if master not in by_name:
            raise InventoryError(
                f"master {master!r} is not in the target set {sorted(by_name)}"
            )
        chosen = by_name[master]
    else:
        chosen = hosts[0]
    addr = chosen.address or chosen.name
    return Rendezvous(master_host=chosen.name, master_addr=addr, port=port)


# --- heterogeneous topology: islands + federation ----------------------------------


@dataclass
class Island:
    """A set of hosts that share one collective engine and *can* form one collective.

    Ascend (HCCL) and NVIDIA (NCCL) nodes cannot rendezvous into a single
    ``world_size`` job — different collective libraries, different transports. So a
    mixed cluster is partitioned into islands, each island running its own
    distributed job, and the islands are *federated*: they exchange model updates
    between rounds rather than gradients within one collective.

    Grouping is by ``Platform.collective``, the *interoperability* key, and not by
    ``dist_backend`` — several NVIDIA-ABI vendors all report the backend name ``nccl``
    while shipping their own ``libnccl.so.2``, and those builds cannot rendezvous with
    each other or with NVIDIA's. Keying on the name would have planned one collective
    across engines that cannot talk.
    """

    id: int
    collective: str
    hosts: list[Host]

    @property
    def name(self) -> str:
        return f"island-{self.id}-{self.collective}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "collective": self.collective,
            "hosts": [h.name for h in self.hosts],
            "nodes": len(self.hosts),
        }


def plan_islands(hosts: list[Host]) -> list[Island]:
    """Partition hosts into islands by collective engine, preserving order.

    Hosts are grouped by their platform's ``collective`` (``hccl`` / ``nccl`` /
    ``rccl`` / ``mccl`` / ``flagcx`` / ``gloo``). Same-engine hosts form one island that
    runs a single collective; differing engines become separate islands that federate.
    A homogeneous cluster yields a single island, which the caller may run as a plain
    coordinated job.
    """
    order: list[str] = []
    buckets: dict[str, list[Host]] = {}
    for h in hosts:
        key = h.platform_def.collective
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(h)
    return [Island(id=i, collective=k, hosts=buckets[k]) for i, k in enumerate(order)]


def is_heterogeneous(hosts: list[Host]) -> bool:
    """True when a target set spans more than one collective engine."""
    return len({h.platform_def.collective for h in hosts}) > 1


# Re-export for callers that only import cluster.
__all__ = [
    "DEFAULT_RDZV_PORT",
    "HostHealth",
    "Island",
    "Rendezvous",
    "is_heterogeneous",
    "plan_islands",
    "preflight",
    "resolve_rendezvous",
    "wrap_command",  # re-exported for symmetry with inventory
]