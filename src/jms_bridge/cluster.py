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
        }

    @property
    def ok(self) -> bool:
        return self.reachable and self.port_free is not False


def _preflight_command(platform: Platform, port: int) -> str:
    """Probe the accelerator and check whether `port` is already bound on localhost."""
    return "\n".join(
        [
            'echo "===PROBE==="',
            f"{{ {platform.probe_cmd}; }} 2>&1 || echo __PROBE_FAILED__",
            'echo "===PORT==="',
            f"if (exec 3<>/dev/tcp/127.0.0.1/{int(port)}) 2>/dev/null; then "
            f"echo PORT_BUSY; else echo PORT_FREE; fi",
            'echo "===END==="',
        ]
    )


def _parse_preflight(host: Host, platform: Platform, port: int, stdout: str) -> HostHealth:
    probe = ""
    port_state = None
    if "===PROBE===" in stdout and "===PORT===" in stdout:
        probe = stdout.split("===PROBE===", 1)[1].split("===PORT===", 1)[0]
        after = stdout.split("===PORT===", 1)[1]
        if "PORT_BUSY" in after:
            port_state = False
        elif "PORT_FREE" in after:
            port_state = True

    probe_failed = "__PROBE_FAILED__" in probe
    parsed = platform.parse_probe(probe) if (probe.strip() and not probe_failed) else {}
    return HostHealth(
        host=host.name,
        platform=platform.name,
        reachable=True,
        device_count=parsed.get("device_count"),
        version=parsed.get("version"),
        port=port,
        port_free=port_state,
        detail="accelerator probe failed" if probe_failed else "",
    )


async def preflight(
    runner: SSHRunner, hosts: list[Host], *, port: int = DEFAULT_RDZV_PORT, timeout: float = 45.0
) -> list[HostHealth]:
    """Check reachability, accelerator visibility and rendezvous-port availability."""
    import asyncio

    async def one(host: Host) -> HostHealth:
        platform = host.platform_def
        res = await runner.exec_on(host, _preflight_command(platform, port), timeout=timeout)
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


# Re-export for callers that only import cluster.
__all__ = [
    "DEFAULT_RDZV_PORT",
    "HostHealth",
    "Rendezvous",
    "preflight",
    "resolve_rendezvous",
    "wrap_command",  # re-exported for symmetry with inventory
]