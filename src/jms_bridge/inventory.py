"""Host inventory: the logical view of the cluster.

The inventory is what lets a single tool call address "the Ascend box and the four
4090s" without the agent knowing per-host SSH incantations. Each entry maps a
friendly name to a JumpServer login target plus the platform metadata needed to
probe and to launch work correctly on that host.

Example ``~/.config/jms-bridge/hosts.yaml``::

    hosts:
      ascend-910c:
        login: your-login
        account: root
        asset_id: aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
        platform: ascend-910c
        env_setup: "source /path/to/ascend/env.sh"
        workdir: /path/to/torch-fl
        tags: [gpu, ascend, training]

      node-4090-1:
        login: your-login
        account: ubuntu
        asset_id: 11111111-2222-3333-4444-555555555555
        platform: cuda
        env_setup: "source ~/miniconda3/etc/profile.d/conda.sh && conda activate fl"
        workdir: ~/torch-fl
        tags: [gpu, cuda, training]

    groups:
      cluster: [ascend-910c, node-4090-1]
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .config import inventory_path
from .platforms import PLATFORMS, Platform

if TYPE_CHECKING:  # pragma: no cover - import cycle is broken at runtime in load()
    from .workspace import Workspace


class InventoryError(RuntimeError):
    """Raised when the inventory is missing or malformed."""


def wrap_command(env_setup: str | None, workdir: str | None, command: str) -> str:
    """Build the remote *bash* script (a string, piped to ``ssh ... bash -s``).

    ``command`` is a literal shell snippet the caller already controls; it is never
    assembled from untrusted argv here. Callers that build a command from user input
    MUST run it through :func:`shell_quote` first.

    The script is piped over stdin rather than passed as an argv, so ``source`` in
    ``env_setup`` and multi-line commands work without any quoting games.
    """
    lines = ["set -euo pipefail"]
    if env_setup:
        # Vendor env scripts (CANN's set_env.sh, many conda hooks) reference
        # variables that may be unset; relax `-u` just for them.
        lines += ["set +u", env_setup, "set -u"]
    if workdir:
        # Create it if missing, then enter it — a bare `cd` would make every exec
        # fail on a host whose workdir has not been created yet.
        q = shell_quote(workdir)
        lines.append(f"mkdir -p {q} && cd {q}")
    lines.append(command)
    return "\n".join(lines) + "\n"


def shell_quote(value: str) -> str:
    """Single-quote a value for safe interpolation into a POSIX shell snippet."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


@dataclass
class Host:
    """One addressable asset in the cluster."""

    name: str
    login: str | None = None
    account: str | None = None
    asset_id: str | None = None
    #: The asset's own address (IP/DNS as JumpServer knows it). Used as the
    #: rendezvous address for distributed runs, since ranks talk to each other
    #: directly rather than through the bastion. Falls back to the name.
    address: str | None = None
    #: Direct-SSH override: connect straight to this host (tailscale name, IP, or an
    #: ``~/.ssh/config`` alias) instead of through the JumpServer gateway. Lets the
    #: same tool address both bastion'd assets and directly-reachable machines.
    ssh_host: str | None = None
    ssh_user: str | None = None
    ssh_port: int | None = None
    #: Platform key; resolved through :data:`jms_bridge.platforms.PLATFORMS`.
    platform: str = "generic"
    env_setup: str | None = None
    workdir: str | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def platform_def(self) -> Platform:
        return PLATFORMS.get(self.platform, PLATFORMS["generic"])

    def login_string(self, default_login: str | None = None) -> str:
        """The JumpServer per-asset SSH login: ``login#account#asset_id``.

        Verified against koko (``pkg/auth/ssh.go``): the ``#``-separated form takes
        exactly three fields (``user#account#asset``) or four
        (``user#protocol#account#asset``); two fields do not parse, so a direct
        login needs an *account*. ``asset_id`` may be a UUID or an asset name/IP;
        it must resolve to exactly one permitted asset. ``@`` also works as the
        separator, but ``#`` is unambiguous here.
        """
        login = self.login or default_login
        if not login:
            raise InventoryError(
                f"host {self.name!r} has no 'login' and no default login user is configured"
            )
        pieces = [login]
        if self.account:
            pieces.append(self.account)
        if self.asset_id:
            pieces.append(self.asset_id)
        return "#".join(pieces)

    @property
    def is_direct(self) -> bool:
        """True when this host is reached by plain SSH, not through JumpServer."""
        return bool(self.ssh_host)


@dataclass
class Inventory:
    hosts: dict[str, Host]
    groups: dict[str, list[str]]
    workspaces: dict[str, Workspace] = field(default_factory=dict)

    def resolve(self, selectors: list[str]) -> list[Host]:
        """Resolve a list of names / tags / group names into a unique host list.

        A selector can be:
        - a host name (``ascend-910c``)
        - a group name (``cluster``) — expands to its members
        - a tag (``gpu``) — expands to every host carrying it
        - ``all`` / ``*`` — every host
        """
        picked: dict[str, Host] = {}
        for sel in selectors:
            if sel in ("all", "*"):
                picked.update(self.hosts)
            elif sel in self.groups:
                for name in self.groups[sel]:
                    if name not in self.hosts:
                        raise InventoryError(f"group {sel!r} references unknown host {name!r}")
                    picked[name] = self.hosts[name]
            elif sel in self.hosts:
                picked[sel] = self.hosts[sel]
            else:
                tagged = [h for h in self.hosts.values() if sel in h.tags]
                if not tagged:
                    raise InventoryError(
                        f"selector {sel!r} matched no host, group, or tag "
                        f"(known hosts: {sorted(self.hosts)})"
                    )
                for h in tagged:
                    picked[h.name] = h
        return list(picked.values())

    def resolve_workspaces(self, selectors: list[str]) -> list[Workspace]:
        """Resolve names / tags into a unique workspace list.

        Unlike hosts there are no groups; a selector is:
        - a workspace name (the key, e.g. ``fl-ascend``)
        - a tag (``ascend``) — expands to every workspace carrying it
        - ``all`` / ``*`` — every workspace
        """
        picked: dict[str, Workspace] = {}
        for sel in selectors:
            if sel in ("all", "*"):
                picked.update(self.workspaces)
            elif sel in self.workspaces:
                picked[sel] = self.workspaces[sel]
            else:
                tagged = [w for w in self.workspaces.values() if sel in w.tags]
                if not tagged:
                    raise InventoryError(
                        f"selector {sel!r} matched no workspace or tag "
                        f"(known workspaces: {sorted(self.workspaces)})"
                    )
                for w in tagged:
                    picked[w.name] = w
        return list(picked.values())

    @staticmethod
    def asset_metadata(
        path: str | Path | None = None, *, warnings: list[str] | None = None
    ) -> dict[str, dict]:
        """Validated optional machine settings, joined to REST assets by UUID.

        Missing files are normal for REST-only deployments. Invalid files/fields
        are ignored with optional diagnostics; valid fields remain usable. Never
        copy connection/authorization fields or echo configuration values in a
        warning: scripts and even invalid field values may contain credentials.
        """

        def warn(message):
            if warnings is not None:
                warnings.append(message)

        p = inventory_path(path)
        try:
            with p.open("rb") as fh:
                data = yaml.safe_load(fh)
        except FileNotFoundError:
            return {}
        except (OSError, yaml.YAMLError, UnicodeError) as exc:
            warn(f"hosts.yaml metadata could not be read ({type(exc).__name__}); overlay ignored")
            return {}
        if data is None:
            return {}
        if not isinstance(data, dict):
            warn("hosts.yaml must be a mapping; metadata overlay ignored")
            return {}
        hosts = data.get("hosts")
        if hosts is None:
            return {}
        if not isinstance(hosts, dict):
            warn("hosts.yaml hosts must be a mapping; metadata overlay ignored")
            return {}
        metadata: dict[str, dict] = {}
        seen, duplicates = set(), set()
        for index, spec in enumerate(hosts.values(), 1):
            context = f"hosts.yaml entry #{index}"
            if spec is None:
                continue
            if not isinstance(spec, dict):
                warn(f"{context} must be a mapping; entry ignored")
                continue
            # Entries without overlay fields can remain normal legacy/direct-SSH
            # entries. Only metadata intended for a REST asset needs a UUID join.
            fields = {
                k: spec[k] for k in ("env_setup", "workdir", "platform") if spec.get(k) is not None
            }
            if not fields:
                continue
            try:
                asset_id = str(uuid.UUID(spec["asset_id"]))
            except (KeyError, ValueError, TypeError, AttributeError):
                warn(f"{context} needs a valid asset_id UUID; metadata ignored")
                continue
            if asset_id in seen:
                metadata.pop(asset_id, None)
                if asset_id not in duplicates:
                    warn(f"{context} has a duplicate asset_id; all metadata for this asset ignored")
                    duplicates.add(asset_id)
                continue
            seen.add(asset_id)
            validated = {}
            for name, value in fields.items():
                if not isinstance(value, str) or "\x00" in value:
                    warn(f"{context}.{name} must be a string without NUL; field ignored")
                elif name == "platform" and value not in PLATFORMS:
                    warn(f"{context}.platform is not a supported platform key; field ignored")
                else:
                    validated[name] = value
            if validated:
                metadata[asset_id] = validated
        return metadata

    @classmethod
    def load(cls, path: str | Path | None = None) -> Inventory:
        p = inventory_path(path)
        if not p.exists():
            raise InventoryError(
                f"No inventory at {p}. Run `jms-bridge init` to write a starter one, "
                "edit it, or point JMS_BRIDGE_INVENTORY at an existing file. "
                "`examples/hosts.yaml` in the repo shows every supported field."
            )
        with p.open("rb") as fh:
            try:
                data = yaml.safe_load(fh) or {}
            except yaml.YAMLError as exc:
                # A hand-edited `hosts.yaml` with a bad indent is the single most likely
                # way this file is wrong, and `yaml` raises `ParserError`, not our
                # `InventoryError` — so the tool wrapper would hand the agent the opaque
                # "Error executing tool <name>" instead of the line to fix.
                raise InventoryError(f"{p} is not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            # `hosts:` as a list (or a bare scalar) parses fine but has no `.get`; say so
            # here rather than dying later with an AttributeError.
            raise InventoryError(
                f"inventory {p} must be a mapping with a 'hosts:' key, got {type(data).__name__}"
            )

        raw_hosts = data.get("hosts") or {}
        if not raw_hosts:
            raise InventoryError(f"inventory {p} has no 'hosts:' section")

        hosts: dict[str, Host] = {}
        for name, spec in raw_hosts.items():
            spec = spec or {}
            hosts[name] = Host(
                name=name,
                login=spec.get("login"),
                account=spec.get("account"),
                asset_id=str(spec["asset_id"]) if spec.get("asset_id") is not None else None,
                address=spec.get("address"),
                ssh_host=spec.get("ssh_host"),
                ssh_user=spec.get("ssh_user"),
                ssh_port=int(spec["ssh_port"]) if spec.get("ssh_port") else None,
                platform=spec.get("platform", "generic"),
                env_setup=spec.get("env_setup"),
                workdir=spec.get("workdir"),
                tags=list(spec.get("tags") or []),
            )

        groups = {k: list(v) for k, v in (data.get("groups") or {}).items()}

        # Imported here rather than at module scope: workspace.py needs
        # shell_quote from this module, so a top-level import would be circular.
        from .workspace import (
            DEFAULT_CONTAINER,
            DEFAULT_UT_CMD,
            Workspace,
            WorkspaceError,
        )

        raw_workspaces = data.get("workspaces") or {}
        workspaces: dict[str, Workspace] = {}
        for name, spec in raw_workspaces.items():
            spec = spec or {}
            host = spec.get("host")
            if host not in hosts:
                raise InventoryError(
                    f"workspace {name!r} references unknown host {host!r} "
                    f"(known hosts: {sorted(hosts)})"
                )
            try:
                workspaces[name] = Workspace(
                    name=name,
                    host=host,
                    container=spec.get("container", DEFAULT_CONTAINER),
                    repo=spec.get("repo", ""),
                    source=spec.get("source", "git"),
                    proxy=spec.get("proxy"),
                    ut_cmd=spec.get("ut_cmd", DEFAULT_UT_CMD),
                    env_setup=spec.get("env_setup"),
                    mount=spec.get("mount"),
                    local=spec.get("local"),
                    tags=list(spec.get("tags") or []),
                )
            except WorkspaceError as exc:
                raise InventoryError(str(exc)) from exc

        return cls(hosts=hosts, groups=groups, workspaces=workspaces)
