"""Workspaces: where the flagos dev workflow actually runs.

A *workspace* is the unit of reproducibility for the flagos development flow: one
host, one uniform container (``lvyufeng-torch-fl``), one checkout of the Torch-FL
repo, plus how code is delivered to it and how its unit tests are run. The same
Claude Code session drives the same workspace name across many chips, so the tools
can fan a command out to "every workspace tagged ascend" and compare results.

The three ``source`` modes describe how a commit gets in front of a test:

* ``git`` — the container can clone/fetch from a reachable git remote itself.
* ``proxy-git`` — same, but the container needs an HTTP proxy to reach the remote.
* ``push`` — no in-container git at all: the host bind-mounts a directory and the
  controller rsyncs the working tree into it from the developer's machine.

Example inventory entry::

    workspaces:
      fl-ascend:
        host: ascend-910c
        container: lvyufeng-torch-fl
        repo: /work/Torch-FL
        source: git
        tags: [ascend]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .inventory import shell_quote

#: How a workspace receives code.
SOURCES = ("git", "proxy-git", "push")

#: Default unit-test command; overridable per workspace.
DEFAULT_UT_CMD = "python -m pytest tests/ -q --tb=line -rA"


class WorkspaceError(RuntimeError):
    """Raised when a workspace spec is incomplete or inconsistent."""


@dataclass
class Workspace:
    """One host + container + repo combination the dev workflow targets."""

    name: str
    host: str
    container: str
    repo: str
    source: str = "git"
    #: HTTP(S) proxy the container must use to reach the git remote (``proxy-git``).
    proxy: str | None = None
    ut_cmd: str = DEFAULT_UT_CMD
    #: Host-side directory bind-mounted into the container (``push``).
    mount: str | None = None
    #: Developer-side directory rsynced into ``mount`` (``push``).
    local: str | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise WorkspaceError(
                f"workspace {self.name!r}: unknown source {self.source!r} "
                f"(expected one of {', '.join(SOURCES)})"
            )
        if self.source == "proxy-git" and not self.proxy:
            raise WorkspaceError(f"workspace {self.name!r}: source 'proxy-git' requires 'proxy'")
        if self.source == "push" and not (self.mount and self.local):
            raise WorkspaceError(
                f"workspace {self.name!r}: source 'push' requires both 'mount' and 'local'"
            )

    def _proxy_prefix(self) -> str:
        """The ``export http_proxy=... https_proxy=...;`` prefix, or ``''``."""
        if self.source != "proxy-git" or not self.proxy:
            return ""
        q = shell_quote(self.proxy)
        return f"export http_proxy={q} https_proxy={q}; "

    def container_cmd(self, command: str) -> str:
        """Wrap ``command`` to run inside this workspace's container.

        For ``proxy-git`` the proxy is exported to the container process via
        ``docker exec ... env http_proxy=... https_proxy=...`` so git (and anything
        it spawns) inherits it. The command itself is single-quoted, so a command
        containing quotes survives — untrusted values must still be passed through
        :func:`~jms_bridge.inventory.shell_quote` before being embedded.
        """
        q = shell_quote(self.proxy) if self.source == "proxy-git" and self.proxy else None
        env = f"env http_proxy={q} https_proxy={q} " if q else ""
        # The container name is an inventory-controlled identifier and docker only
        # accepts [a-zA-Z0-9][a-zA-Z0-9_.-]*, so it carries no shell metacharacters.
        return f"docker exec {self.container} {env}bash -lc {shell_quote(command)}"

    def delivery_cmd(self, ref: str) -> str | None:
        """The in-container command that puts ``ref`` in place, or None for ``push``.

        For ``push`` the code arrives out-of-band via rsync into the bind-mount, so
        there is nothing for the container to fetch.
        """
        if self.source == "push":
            return None
        q = shell_quote(self.repo)
        qref = shell_quote(ref)
        return (
            f"{self._proxy_prefix()}cd {q} && "
            f"git fetch --depth=1 origin {qref} && git checkout {qref}"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "host": self.host,
            "container": self.container,
            "repo": self.repo,
            "source": self.source,
            "ut_cmd": self.ut_cmd,
            "tags": list(self.tags),
        }


#: ``PASSED/FAILED/ERROR/SKIPPED [n]? <test id><separator><message>`` summary line.
#: The ``[n]`` group is pytest's count for whole-module SKIPPED reports.
_PYTEST_LINE = re.compile(
    r"^(PASSED|FAILED|ERROR|SKIPPED)\s+(?:\[\d+\]\s+)?(?P<rest>.*)$"
)

#: pytest's one-letter terminal statuses -> our normalized outcome.
_PYTEST_OUTCOMES = {
    "PASSED": "PASS",
    "FAILED": "FAIL",
    "ERROR": "ERROR",
    "SKIPPED": "SKIP",
}


def _split_test_id(rest: str) -> tuple[str, str]:
    """Split a summary line body into ``(test_id, message)``.

    pytest separates the two with `` - `` (assertion lines) or ``: `` (skip/error
    reasons). A test id may itself contain ``::``, so only a bare ``": "`` counts.
    """
    for sep in (" - ", ": "):
        idx = rest.find(sep)
        if idx != -1:
            return rest[:idx].strip(), rest[idx + len(sep) :].strip()
    return rest.strip(), ""


def parse_pytest_summary(output: str) -> dict[str, dict]:
    """Parse pytest's ``-rA`` short summary into ``{test_id: {outcome, message}}``.

    Only the per-test summary lines are meaningful — the trailing
    "N failed, M passed" counts are ignored. A test that appears more than once
    keeps its first report.
    """
    summary: dict[str, dict] = {}
    for line in output.splitlines():
        match = _PYTEST_LINE.match(line.strip())
        if not match:
            continue
        test_id, message = _split_test_id(match.group("rest"))
        if not test_id or test_id in summary:
            continue
        summary[test_id] = {
            "outcome": _PYTEST_OUTCOMES[match.group(1)],
            "message": message,
        }
    return summary