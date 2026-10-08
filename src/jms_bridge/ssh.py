"""SSH data plane through the JumpServer KoKo gateway.

Design notes that matter:

* **ControlMaster multiplexing.** A KoKo handshake is slow, and the deployment's
  gateway has been measured at ~200s TTFB for buffered responses. We open one
  persistent connection per host and reuse it via OpenSSH's control socket, so N
  commands cost one handshake, not N.
* **stdin-piped scripts, not argv.** The remote command is a bash script piped to
  ``bash -s``. That way ``env_setup`` can ``source`` files and multi-line commands
  need no quoting, and nothing user-supplied is ever interpreted by the *local*
  shell (we always pass ``argv`` lists to ``subprocess``).
* **Bounded fan-out.** Parallelism is capped so a 20-host call does not fork-bomb
  the local machine or trip the bastion's connection limits.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import BastionConfig
from .inventory import Host, InventoryError, wrap_command

#: ssh options shared by every invocation. BatchMode keeps a missing key an error
#: instead of an interactive prompt (nothing can answer one under an MCP server).
_BASE_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=6",
    "-o", "ConnectTimeout=30",
]


@dataclass
class ExecResult:
    host: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "ok": self.ok,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


class SSHRunner:
    """Runs commands and transfers files on inventory hosts through the bastion."""

    def __init__(self, cfg: BastionConfig, *, max_parallel: int = 8):
        self.cfg = cfg
        self.max_parallel = max_parallel
        self._sem = asyncio.Semaphore(max_parallel)
        self._cmi_dir = Path(tempfile.mkdtemp(prefix="jms-bridge-cm-"))

    # --- target construction ---------------------------------------------------

    def target(self, host: Host) -> str:
        """``login#account#asset_id@bastion`` — the JumpServer per-asset address."""
        return f"{host.login_string(self.cfg.login_user)}@{self.cfg.url.split('://')[-1].split('/')[0]}"

    def _ssh_argv(self, host: Host, *, tty: bool = False) -> list[str]:
        control = f"{self._cmi_dir}/{host.name}.sock"
        argv = ["ssh", *_BASE_OPTS]
        if self.cfg.ssh_key:
            argv += ["-i", str(self.cfg.ssh_key), "-o", "IdentitiesOnly=yes"]
        argv += [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={control}",
            "-o", "ControlPersist=300",
            "-p", str(self.cfg.ssh_port),
        ]
        if tty:
            argv.append("-tt")
        argv += [self.target(host), "bash -s"]
        return argv

    # --- execution -------------------------------------------------------------

    async def _run_script(
        self, host: Host, script: str, timeout: float, *, tty: bool = False
    ) -> ExecResult:
        async with self._sem:
            proc = await asyncio.create_subprocess_exec(
                *self._ssh_argv(host, tty=tty),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "TERM": os.environ.get("TERM", "dumb")},
            )
            try:
                out, err = await asyncio.wait_for(
                    proc.communicate(script.encode()), timeout=timeout
                )
                return ExecResult(
                    host=host.name,
                    exit_code=proc.returncode or 0,
                    stdout=out.decode(errors="replace"),
                    stderr=err.decode(errors="replace"),
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return ExecResult(
                    host=host.name,
                    exit_code=124,
                    stdout="",
                    stderr=f"timed out after {timeout}s",
                    timed_out=True,
                )

    async def exec_on(self, host: Host, command: str, *, timeout: float = 60.0) -> ExecResult:
        """Run a short command on one host, activating the host's env first."""
        script = wrap_command(host.env_setup, host.workdir, command)
        return await self._run_script(host, script, timeout)

    async def exec_many(
        self, hosts: list[Host], command: str, *, timeout: float = 60.0
    ) -> list[ExecResult]:
        """Fan a command out across hosts in parallel (bounded by ``max_parallel``)."""
        return list(
            await asyncio.gather(*(self.exec_on(h, command, timeout=timeout) for h in hosts))
        )

    # --- file transfer ---------------------------------------------------------

    async def rsync(
        self,
        host: Host,
        local: str,
        remote: str,
        *,
        direction: str = "push",
        delete: bool = False,
        excludes: list[str] | None = None,
        timeout: float = 900.0,
    ) -> ExecResult:
        """Push or pull a path via rsync over the multiplexed SSH connection."""
        ssh_cmd = " ".join(
            shlex.quote(a)
            for a in [
                "ssh",
                *_BASE_OPTS,
                *(("-i", str(self.cfg.ssh_key)) if self.cfg.ssh_key else ()),
                "-o", "ControlMaster=auto",
                "-o", f"ControlPath={self._cmi_dir}/{host.name}.sock",
                "-o", "ControlPersist=300",
                "-p", str(self.cfg.ssh_port),
            ]
        )
        argv = ["rsync", "-az", "--info=stats1", "-e", ssh_cmd]
        if delete:
            argv.append("--delete")
        for pat in excludes or []:
            argv += ["--exclude", pat]

        remote_spec = f"{self.target(host)}:{remote}"
        if direction == "push":
            argv += [local, remote_spec]
        else:
            argv += [remote_spec, local]

        async with self._sem:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return ExecResult(host.name, 124, "", f"rsync timed out after {timeout}s", True)
            return ExecResult(
                host.name,
                proc.returncode or 0,
                out.decode(errors="replace"),
                err.decode(errors="replace"),
            )

    async def close(self) -> None:
        """Tear down multiplexed connections (call on shutdown)."""
        for sock in self._cmi_dir.glob("*.sock"):
            host = sock.stem
            argv = ["ssh", "-o", f"ControlPath={sock}", "-O", "exit", host]
            try:
                p = await asyncio.create_subprocess_exec(
                    *argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
                )
                await p.wait()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        for sock in self._cmi_dir.glob("*.sock"):
            sock.unlink(missing_ok=True)


def require_login(host: Host, default_login: str | None) -> None:
    if not (host.login or default_login):
        raise InventoryError(
            f"host {host.name!r} has no 'login' and no bastion default login user is set"
        )