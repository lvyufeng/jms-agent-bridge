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
import signal
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import BastionConfig, ConfigError
from .inventory import Host, InventoryError, wrap_command

if TYPE_CHECKING:
    from .results import OutputPreview

#: Echoed by a data-plane reachability check and required back in the reply. KoKo
#: rejects a connection (no permission, asset offline, bad address) with ssh **exit
#: code 0** and the error text on stdout, so the exit status alone reads as success.
#: Only a reply that actually contains this sentinel proves the command ran on the
#: far side — it is what makes a reachability check trustworthy.
REACH_SENTINEL = "jms-bridge-ok"

#: Prefix a command so its reply is provably from the far side: if the command runs at
#: all, the sentinel comes back ahead of its output. Use with :func:`reached`.
REACH_PROBE = f"echo {REACH_SENTINEL}"


def _require_tool(binary: str, hint: str) -> None:
    """Fail with an actionable ``ConfigError`` when a local tool is missing.

    ``create_subprocess_exec`` raises a bare ``FileNotFoundError`` for an absent
    binary, which the SDK shows the agent as the opaque ``Error executing tool <name>``.
    A freshly-cloned machine is exactly where this happens, so it has to say what to
    install; ``ConfigError`` is what the tool wrapper turns into that text.
    """
    if shutil.which(binary) is None:
        raise ConfigError(f"`{binary}` is not installed — {hint}")


async def _kill(proc: asyncio.subprocess.Process) -> None:
    """Terminate a spawned ssh/rsync **and its process group**, then reap it.

    Called under ``except BaseException`` — which includes ``asyncio.CancelledError`` — so
    a cancelled tool call (the client disconnecting, the gateway cutting a long response
    off) does not leave the child running. Catching only ``TimeoutError`` leaked one ssh
    process per cancelled call: ``CancelledError`` is a ``BaseException``, so it skipped
    the kill entirely. ``wait()`` is itself shielded from a second cancellation by
    re-raising only after the child is reaped.

    The *group* matters, not just the child: password auth spawns ``sshpass`` which in
    turn spawns ``ssh``, so ``proc.kill()`` alone left the real ssh running — and a
    surviving ssh keeps the ControlMaster socket alive for ``ControlPersist`` seconds.
    The next call then reuses a half-dead multiplexed channel, which is how a 2s timeout
    turned into a 10s call returning **exit 0 with a DNS error on stdout** (see
    :meth:`SSHRunner._run_script`). Every child is spawned with ``start_new_session=True``
    precisely so this group kill is the whole tree.
    """
    if proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return  # already gone between the check and the lookup
    # Never signal our *own* group: that would take down the MCP server (and, in tests,
    # pytest itself). A child spawned without `start_new_session` shares our group, so
    # this is a real hazard, not just a test artifact — `killpg(getpgid(self))` is a
    # process-suicide button. Fall back to signalling the child alone in that case.
    if pgid == os.getpgid(0):
        try:
            proc.kill()
        except ProcessLookupError:
            return
    else:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except ProcessLookupError:
                return
    try:
        await proc.wait()
    except asyncio.CancelledError:  # pragma: no cover - second cancel during reaping
        pass


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


#: Character budget for the stdout/stderr of one host in a fan-out reply.
#:
#: A tool result is consumed by an agent, not a human at a terminal, and a plain
#: `cat train.log` across 40 hosts produced an 8 MB (≈2 M token) reply — enough to blow
#: any client's context. The cap is per host and per stream, so the *number* of hosts a
#: caller may fan out to is unaffected; only the inline preview is bounded.
OUTPUT_LIMIT = 16_000


def _clip(text: str, limit: int = OUTPUT_LIMIT) -> tuple[str, int]:
    """Return ``(preview, total_chars)``; the preview ends with an explicit marker.

    The marker is not decoration: an agent that cannot tell a clipped log from a whole one
    will reason about a partial result as if it were complete. The true length is returned
    alongside so the caller can also report it as a number.
    """
    if len(text) <= limit:
        return text, len(text)
    head = text[:limit]
    return f"{head}\n… [truncated: showing {limit} of {len(text)} chars]", len(text)


@dataclass
class ExecResult:
    host: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    output_preview: OutputPreview | None = None

    @property
    def stdout_truncated(self):
        return self.output_preview is not None and self.output_preview.stdout_truncated

    @property
    def stderr_truncated(self):
        return self.output_preview is not None and self.output_preview.stderr_truncated

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self, *, limit: int = OUTPUT_LIMIT) -> dict:
        """The per-host reply entry, with stdout/stderr bounded for an agent consumer.

        ``limit`` is the total for this host across **both** streams, not per stream: a
        host that wrote 1 MB to stdout and 1 MB to stderr must not spend double its share
        of a fan-out's budget. stdout is served first (it is the payload a caller asked
        for) and stderr gets whatever remains, so the pair never exceeds ``limit``.

        `stdout_bytes` / `stderr_bytes` report the true sizes and
        `stdout_truncated` / `stderr_truncated` say whether a preview is partial, so a
        clipped stream can never be mistaken for a complete one.
        """
        if self.output_preview is not None:
            # This legacy serializer intentionally retains stdout-first prefixes;
            # compact diagnostics use OutputPreview.compact's reserved error tail.
            preview = self.output_preview
            raw_out, raw_err = preview.stdout_head, preview.stderr_head
            shown_out = raw_out[:max(0, limit)]
            shown_err = raw_err[:max(0, limit - len(shown_out))]
            sizes = preview.sizes
            return {
                "host": self.host,
                "ok": self.ok,
                "exit_code": self.exit_code,
                "timed_out": self.timed_out,
                "stdout": shown_out.decode(errors="replace"),
                "stderr": shown_err.decode(errors="replace"),
                "stdout_bytes": sizes.get("stdout", 0),
                "stderr_bytes": sizes.get("stderr", 0),
                "stdout_truncated": sizes.get("stdout", 0) > len(shown_out),
                "stderr_truncated": sizes.get("stderr", 0) > len(shown_err),
                "result_id": self.result_id,
            }
        out, out_len = _clip(self.stdout, limit)
        # Give stderr what stdout did not use, so the two together stay within `limit`.
        remaining = max(0, limit - len(out))
        err, err_len = _clip(self.stderr, remaining) if remaining else ("", len(self.stderr))
        return {
            "host": self.host,
            "ok": self.ok,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "stdout": out,
            "stderr": err,
            "stdout_bytes": out_len,
            "stderr_bytes": err_len,
            "stdout_truncated": out_len > len(out),
            "stderr_truncated": err_len > len(err),
        }


def reached(res: ExecResult) -> bool:
    """Whether ``res`` proves the command actually ran on the far side.

    An exit code of 0 is *not* that proof: KoKo rejects a connection (no permission, asset
    offline, bad address) with exit 0 and its error on stdout, so a status-only check
    reports a dead asset green. Only a reply carrying :data:`REACH_SENTINEL` — which the
    command echoed via :data:`REACH_PROBE` — does. Every reachability decision in the
    codebase goes through here, so the rule cannot drift apart between tools.
    """
    return res.ok and REACH_SENTINEL in res.stdout


def failure_reason(res: ExecResult) -> str:
    """Why a command did not succeed — the text a caller should show the user.

    A rejection's explanation is on **stdout**, not stderr: KoKo reports "no permission",
    "asset offline" or "必须是唯一的账号 root" there, and exits **0** while doing it. Looking
    only at stderr (and only when ``res.ok`` is false) therefore discards the one sentence
    that says what to fix. So: prefer stderr when the command genuinely failed, else
    stdout — which is either the rejection text or the sentinel-prefixed output that a
    caller has already decided is not good enough. Trimmed, because a banner can precede
    the real message.
    """
    if res.timed_out:
        return res.stderr.strip()[:400] or "timed out"
    if not res.ok:
        return (res.stderr or res.stdout).strip()[:400]
    return res.stdout.strip()[:400]


class SSHRunner:
    """Runs commands and transfers files on inventory hosts through the bastion."""

    def __init__(self, cfg: BastionConfig, *, max_parallel: int = 8):
        self.cfg = cfg
        # A cap of 0 would build a Semaphore that no task can ever acquire — a hang,
        # which is the worst way for a fan-out to fail. The loader rejects it too; this
        # covers a directly-constructed runner (tests, embedders).
        self.max_parallel = max(1, max_parallel)
        self._sem = asyncio.Semaphore(self.max_parallel)
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
            raise ConfigError(
                "bastion ssh_password is set but `sshpass` is not installed "
                "(apt/brew install sshpass), or register an SSH key instead"
            )
        return ["sshpass", "-e"]  # password read from the SSHPASS env var

    def _subproc_env(self, host: Host | None = None) -> dict[str, str]:
        """Environment for a spawned ssh/rsync, carrying ``SSHPASS`` only when needed.

        The password is handed to the child that will actually run under ``sshpass`` —
        a bastion host with password auth — and to nothing else. A direct host is
        key-based, so giving it the bastion password serves no purpose, and ``close()``'s
        ``ssh -O exit`` talks to the existing control socket rather than authenticating,
        so it needs none either.
        """
        env = {**os.environ, "TERM": os.environ.get("TERM", "dumb")}
        if host is not None and not host.is_direct and self.cfg.ssh_password:
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
        _require_tool("ssh", "install an SSH client (apt/brew install openssh)")
        async with self._sem:
            proc = await asyncio.create_subprocess_exec(
                *self._ssh_argv(host, tty=tty),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._subproc_env(host),
                # Own session => own process group, so `_kill` can take down sshpass
                # *and* the ssh it spawned in one signal. Without this the `sshpass -e
                # ssh ...` chain outlives a `proc.kill()` on the group leader.
                start_new_session=True,
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
                await _kill(proc)
                # The ssh that owns this host's ControlMaster is gone, but the *master*
                # (a forked background process, not our child) outlives it for
                # ControlPersist=300s and keeps the socket open. Reusing that channel is
                # the documented way a short timeout produces a long, wrongly-successful
                # call, so tear it down before anyone can reuse it.
                await self._close_control_master(host)
                return ExecResult(
                    host=host.name,
                    exit_code=124,
                    stdout="",
                    stderr=f"timed out after {timeout}s",
                    timed_out=True,
                )
            except BaseException:
                # Cancellation (or anything else) must still reap the child — otherwise a
                # cancelled call leaves the ssh process running. Re-raised so the caller
                # sees the original failure, not a masked one.
                await _kill(proc)
                await self._close_control_master(host)
                raise

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
        _require_tool("rsync", "install rsync (apt/brew install rsync)")
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
                env=self._subproc_env(host),
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except (TimeoutError, asyncio.TimeoutError):
                await _kill(proc)
                return ExecResult(host.name, 124, "", f"rsync timed out after {timeout}s", True)
            except BaseException:
                # A cancelled transfer must not leave rsync running — a multi-gigabyte
                # checkpoint pull would otherwise keep going unseen. Note this kills
                # rsync itself, not the ssh child rsync spawned for the transfer: that
                # child is a *slave* connection, so it exits on its own once rsync's
                # pipes close (unlike `_run_script`, where ssh is the long-lived master
                # the kill has to reach).
                await _kill(proc)
                raise
            return ExecResult(
                host.name,
                proc.returncode or 0,
                out.decode(errors="replace"),
                err.decode(errors="replace"),
            )

    async def _close_control_master(self, host: Host) -> None:
        """Ask the ControlMaster for *host* to exit, then drop its socket.

        Best effort by design: it runs on the timeout and cancellation paths, where the
        ssh is already dead and a failure to clean up must not replace the real result
        with a cleanup error. ``-O exit`` talks to the existing socket, so it needs no
        authentication — hence no ``SSHPASS``.

        ``-o ControlMaster=no`` is essential: without it this invocation would try to
        *become* a master for the same ControlPath and could re-create the socket we are
        trying to remove.
        """
        sock = self._sock(host)
        if not sock.exists():
            return
        try:
            p = await asyncio.create_subprocess_exec(
                "ssh",
                "-o",
                f"ControlPath={sock}",
                "-o",
                "ControlMaster=no",
                "-o",
                "BatchMode=yes",
                "-O",
                "exit",
                self.target(host),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=self._subproc_env(),
                start_new_session=True,
            )
            # Bounded: a wedged master would otherwise hang the cleanup of the very call
            # that already timed out.
            await asyncio.wait_for(p.wait(), timeout=5)
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
        sock.unlink(missing_ok=True)

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