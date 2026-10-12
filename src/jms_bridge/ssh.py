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
import hashlib
import os
import shlex
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import BastionConfig
from .inventory import Host, InventoryError, wrap_command

#: ssh options shared by every invocation.
_BASE_OPTS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=6",
    "-o", "ConnectTimeout=30",
]

#: Extra options for key/agent auth: keep a missing key an error instead of an
#: interactive prompt that nothing can answer under an MCP server. Deliberately
#: NOT applied when authenticating with a password — ``BatchMode=yes`` disables
#: the password prompt *entirely*, so ``sshpass`` has nothing to feed and every
#: password-auth connection dies with "Permission denied (password,publickey)".
_BATCH_MODE = ["-o", "BatchMode=yes"]


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
        # A short socket dir: the full ControlPath must fit in a Unix-domain socket
        # path (104 bytes on macOS), so a long temp dir + long host name overflows.
        self._cmi_dir = Path(tempfile.mkdtemp(prefix="jmsb-", dir="/tmp"))

    def _sock(self, host: Host) -> Path:
        """A short, collision-free control-socket path for a host."""
        # Re-created on demand: close() removes the directory, and a runner that is
        # used again afterwards must still be able to open a control socket.
        self._cmi_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(host.name.encode()).hexdigest()[:12]
        return self._cmi_dir / f"{digest}.sock"

    # --- target construction ---------------------------------------------------

    def _sshpass_prefix(self, host: Host) -> list[str]:
        """``sshpass`` argv prefix for password auth, or ``[]``.

        Only used for the bastion gateway: direct hosts are key-based. Failing
        loudly here is better than silently falling back to an interactive prompt
        that nothing can answer under the MCP server.
        """
        if host.is_direct or not self.cfg.ssh_password:
            return []
        if shutil.which("sshpass") is None:
            raise RuntimeError(
                "bastion ssh_password is set but `sshpass` is not installed "
                "(apt/brew install sshpass), or register an SSH key instead"
            )
        return ["sshpass", "-e"]  # password read from the SSHPASS env var

    def _subproc_env(self) -> dict[str, str]:
        """Environment for a spawned ssh/rsync, carrying SSHPASS when in use."""
        env = {**os.environ, "TERM": os.environ.get("TERM", "dumb")}
        if self.cfg.ssh_password:
            env["SSHPASS"] = self.cfg.ssh_password
        return env

    def target(self, host: Host) -> str:
        """``login#account#asset_id@bastion`` — the JumpServer per-asset address."""
        return f"{host.login_string(self.cfg.login_user)}@{self.cfg.url.split('://')[-1].split('/')[0]}"

    def _host_port(self, host: Host) -> tuple[str, int]:
        """Resolve the ssh target and port for a host (direct override or bastion)."""
        if host.is_direct:
            return host.ssh_host, host.ssh_port or 22
        return self.target(host), self.cfg.ssh_port

    def _ssh_argv(self, host: Host, *, tty: bool = False) -> list[str]:
        control = str(self._sock(host))
        password_auth = bool(self.cfg.ssh_password) and not host.is_direct
        argv = self._sshpass_prefix(host) + ["ssh", *_BASE_OPTS]
        # Only suppress prompts when we are NOT relying on sshpass to answer one.
        if not password_auth:
            argv += _BATCH_MODE
        key = self.cfg.ssh_key
        if host.is_direct and host.ssh_user:
            argv += ["-l", host.ssh_user]
        if key:
            argv += ["-i", str(key), "-o", "IdentitiesOnly=yes"]
        if password_auth:
            # Password auth to the bastion; the key, if any, may still be tried first.
            argv += ["-o", "PubkeyAuthentication=no", "-o", "PreferredAuthentications=password"]
        argv += [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={control}",
            "-o", "ControlPersist=300",
        ]
        target, port = self._host_port(host)
        if port != 22:
            argv += ["-p", str(port)]
        if tty:
            argv.append("-tt")
        argv += [target, "bash -s"]
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
                env=self._subproc_env(),
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
            except (TimeoutError, asyncio.TimeoutError):
                proc.kill()
                await proc.wait()
                return ExecResult(
                    host=host.name,
                    exit_code=124,
                    stdout="",
                    stderr=f"timed out after {timeout}s",
                    timed_out=True,
                )

    async def exec_on(
        self,
        host: Host,
        command: str,
        *,
        timeout: float = 60.0,
        workdir: bool = True,
        env_setup: bool = True,
    ) -> ExecResult:
        """Run a short command on one host, activating the host's env first.

        ``workdir=False`` skips the ``cd`` into the host workdir — used by probes,
        which must succeed even before the workdir exists.

        ``env_setup=False`` skips sourcing the host's ``env_setup`` entirely. Used by
        commands that run *inside a container* (workspaces): the host toolchain is
        irrelevant there, and a missing or failing host env script would abort the
        whole command under ``set -e``.
        """
        script = wrap_command(
            host.env_setup if env_setup else None,
            host.workdir if workdir else None,
            command,
        )
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
        target, port = self._host_port(host)
        ssh_parts = [
            *self._sshpass_prefix(host),
            "ssh",
            *_BASE_OPTS,
            *(["-l", host.ssh_user] if host.is_direct and host.ssh_user else []),
            *(("-i", str(self.cfg.ssh_key)) if self.cfg.ssh_key else ()),
            *(
                ("-o", "PubkeyAuthentication=no", "-o", "PreferredAuthentications=password")
                if self.cfg.ssh_password and not host.is_direct
                else ()
            ),
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self._sock(host)}",
            "-o", "ControlPersist=300",
            *(["-p", str(port)] if port != 22 else []),
        ]
        ssh_cmd = " ".join(shlex.quote(a) for a in ssh_parts)
        # ``--stats``/``--info=*`` are GNU-rsync extensions; the macOS built-in is
        # openrsync, which rejects them. The exit code is the contract here, so the
        # transfer is run without them and works with either implementation.
        argv = ["rsync", "-az", "-e", ssh_cmd]
        if delete:
            argv.append("--delete")
        for pat in excludes or []:
            argv += ["--exclude", pat]

        remote_spec = f"{target}:{remote}"
        if direction == "push":
            argv += [local, remote_spec]
        else:
            argv += [remote_spec, local]

        async with self._sem:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._subproc_env(),
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except (TimeoutError, asyncio.TimeoutError):
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
                    *argv,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    env=self._subproc_env(),
                )
                await p.wait()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        for sock in self._cmi_dir.glob("*.sock"):
            sock.unlink(missing_ok=True)
        # Drop the (now empty) short socket dir too: the runner is rebuilt on every
        # inventory/config reload, so leaving it behind would leak one /tmp/jmsb-*
        # directory per reload.
        shutil.rmtree(self._cmi_dir, ignore_errors=True)


def require_login(host: Host, default_login: str | None) -> None:
    if not (host.login or default_login):
        raise InventoryError(
            f"host {host.name!r} has no 'login' and no bastion default login user is set"
        )