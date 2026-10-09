"""Remote job lifecycle: submit / status / logs / cancel.

The whole point of this module: an MCP tool call **must never block** on a long
command. A Torch-FL run lasts hours, and the gateway on these hosts buffers
non-streaming responses for ~200s, so any synchronous "run the training" tool
would time out and leave the agent blind.

Instead every long action becomes a *job*:

* a small runner script is written to ``<workdir>/.jms-bridge/jobs/<job_id>/``
* it is launched inside a **tmux** session (survives SSH disconnect)
* the runner writes ``status`` and ``exit_code`` files and streams to ``stdout.log``
* ``jms_submit_job`` returns the job id immediately

``status``/``logs`` are cheap reads the agent can poll. ``logs`` takes a byte
offset so polling only returns new output.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass

from .inventory import Host, shell_quote
from .ssh import ExecResult, SSHRunner

#: Where job state lives on each remote host, relative to the host workdir.
JOB_DIR = ".jms-bridge/jobs"

_JOB_ID_RE = re.compile(r"^[0-9a-f]{8}$")
_STATE_FILES = ("status", "exit_code", "pid")


def _new_job_id() -> str:
    return uuid.uuid4().hex[:8]


def _launcher_of(stdout: str) -> str:
    """Which detached launcher the remote picked: ``tmux`` or ``setsid``."""
    for line in stdout.splitlines():
        if line.startswith("LAUNCHER="):
            return line.split("=", 1)[1].strip() or "unknown"
    return "unknown"


def _job_root(host: Host, job_id: str) -> str:
    """A *shell expression* for the remote job directory — use it unquoted.

    A leading ``~`` becomes ``"$HOME"`` so the remote shell expands it; an absolute
    path is single-quoted. The result is embedded directly into command strings and
    must NOT be wrapped in :func:`shell_quote` again.
    """
    base = host.workdir or "~"
    if base.startswith("~"):
        rest = base[1:].lstrip("/")
        tail = f"{rest}/{JOB_DIR}/{job_id}" if rest else f"{JOB_DIR}/{job_id}"
        return f'"$HOME"/{tail}'
    return shell_quote(f"{base.rstrip('/')}/{JOB_DIR}/{job_id}")


def _workdir_expr(workdir: str | None) -> str | None:
    """A shell expression for a workdir, tilde-safe (``~`` → ``"$HOME"``)."""
    if not workdir:
        return None
    if workdir.startswith("~"):
        rest = workdir[1:].lstrip("/")
        return f'"$HOME"/{rest}' if rest else '"$HOME"'
    return shell_quote(workdir)


def build_runner_script(
    job_id: str, command: str, env_setup: str | None, workdir: str | None = None
) -> str:
    """The remote bash script that wraps one job.

    Self-locating: it derives its own directory (the job dir) into ``$R`` and writes
    ``pid``/``status``/``exit_code`` there with **absolute** paths, so the state lands
    in the job dir no matter what cwd tmux handed it. Written so the exit status, a
    terminal ``status`` marker and the PID are recorded even if the command fails, and
    the tmux session ends with the job rather than lingering.
    """
    lines = [
        "set -uo pipefail",
        'R="$(cd "$(dirname "$0")" && pwd)"',
    ]
    if env_setup:
        lines.append(env_setup)
    wd = _workdir_expr(workdir)
    if wd:
        # Run in the host workdir when it exists, else stay in the job dir.
        lines.append(f'cd {wd} 2>/dev/null || cd "$R"')
    lines += [
        'echo "$$" > "$R/pid"',
        'echo running > "$R/status"',
        "set +e",
        # Run the command in a subshell: a command that ends in `exit N` would
        # otherwise terminate this wrapper before the status/exit_code are written,
        # leaving the job forever 'running' with no exit code.
        "(",
        command,
        ")",
        "code=$?",
        'echo "$code" > "$R/exit_code"',
        'if [ "$code" -eq 0 ]; then echo done > "$R/status"; '
        'else echo failed > "$R/status"; fi',
        'exit "$code"',
    ]
    return "\n".join(lines) + "\n"


def _collab_exports(
    *, rank: int, nproc: int, world_size: int, master_addr: str, port: int
) -> str:
    """The ``export`` lines that wire one rank into a shared rendezvous.

    Every rank gets the same ``JMS_WORLD_SIZE`` / ``JMS_MASTER_*`` and its own
    ``JMS_RANK``; a distributed launcher on the host reads these to join the job.
    """
    env = {
        "JMS_RANK": str(rank),
        "JMS_LOCAL_RANK": "0",
        "JMS_NPROC": str(nproc),
        "JMS_WORLD_SIZE": str(world_size),
        "JMS_MASTER_ADDR": master_addr,
        "JMS_MASTER_PORT": str(port),
    }
    return "\n".join(f"export {k}={shell_quote(v)}" for k, v in env.items())


@dataclass
class JobHandle:
    job_id: str
    host: str
    session: str
    launcher: str = "tmux"  # "tmux" or "setsid"

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "host": self.host,
            "session": self.session,
            "launcher": self.launcher,
        }


def _launch_script(
    root: str,
    session: str,
    job_id: str,
    body: str,
    env_setup: str | None,
    workdir: str | None = None,
) -> str:
    """Build the remote script that stages and launches one detached job.

    ``root`` is a shell *expression* (from :func:`_job_root`), embedded unquoted.
    All the quoting is done here, once: the runner is written with a quoted heredoc
    (so nothing in it is expanded on the way in), and the tmux command is built by
    assigning ``R="$HOME"/...`` first so the doubly-quoted inner command sees a plain
    path with no stray single quotes from the absolute-path form.
    """
    runner_script = build_runner_script(job_id, body, env_setup, workdir)
    # Prefer tmux (nice `list_jobs`, clean kill); fall back to setsid on hosts that
    # lack it (e.g. minimal domestic-chip images) so detached jobs still survive the
    # SSH disconnect. Detection and launch happen in the same remote script.
    return "\n".join(
        [
            "set -eu",
            f"R={root}",
            'mkdir -p "$R"',
            f'cat > "$R/run.sh" <<\'JMS_RUNNER_EOF\'\n{runner_script}JMS_RUNNER_EOF',
            # Common detached runner body (records the launcher, then runs the job).
            'if command -v tmux >/dev/null 2>&1; then',
            f"  tmux new-session -d -s {shell_quote(session)} "
            '"bash \\"$R/run.sh\\" > \\"$R/stdout.log\\" 2>&1; echo \\$? > \\"$R/tmux_exit\\""',
            f"  tmux has-session -t {shell_quote(session)} && echo LAUNCHER=tmux",
            "else",
            "  command -v setsid >/dev/null 2>&1 || { echo 'no tmux and no setsid' >&2; exit 3; }",
            '  setsid bash "$R/run.sh" > "$R/stdout.log" 2>&1 < /dev/null &',
            '  echo "$!" > "$R/launcher_pid"',
            "  sleep 1",
            '  kill -0 "$(cat "$R/launcher_pid")" 2>/dev/null && echo LAUNCHER=setsid',
            "fi",
        ]
    )


class JobManager:
    def __init__(self, runner: SSHRunner):
        self.runner = runner

    async def submit(
        self, host: Host, command: str, *, name: str | None = None
    ) -> tuple[JobHandle | None, ExecResult]:
        """Launch a detached job; returns the handle and the launch result."""
        job_id = _new_job_id()
        session = f"jms-{job_id}"
        root = _job_root(host, job_id)
        launch = _launch_script(
            root, session, job_id, command, host.env_setup, host.workdir
        )
        res = await self.runner.exec_on(host, launch, timeout=90, workdir=False)
        if res.ok and "LAUNCHER=" in res.stdout:
            return JobHandle(job_id, host.name, session, _launcher_of(res.stdout)), res
        return None, res

    async def submit_collab(
        self,
        assignments: list[tuple[Host, int]],
        command: str,
        *,
        master_addr: str,
        port: int,
        world_size: int,
    ) -> tuple[list[JobHandle], list[ExecResult]]:
        """Launch one *cooperative* job across ranked ranks sharing a rendezvous.

        Unlike :meth:`submit` (which copies a command to each host to run
        independently), this launches on every rank in parallel with the same
        ``JMS_RANK`` / ``JMS_WORLD_SIZE`` / ``JMS_MASTER_*`` environment, so a
        distributed launcher on each host (``torchrun --rdzv-endpoint``) can join a
        single job. The agent supplies the launcher command; the bridge supplies the
        rank/world wiring, which is the part that must be consistent across nodes.

        ``assignments`` is ``[(host, nproc), ...]``; ranks are assigned in order, so
        the host list order defines rank order. All ranks launch concurrently and the
        call returns as soon as every launch is acknowledged — the job itself keeps
        running detached.
        """
        handles: list[JobHandle] = []
        results: list[ExecResult] = []
        rank = 0
        pending = []
        for host, nproc in assignments:
            job_id = _new_job_id()
            session = f"jms-{job_id}"
            root = _job_root(host, job_id)

            # Rank/world wiring, exported before the command; the launcher reads these.
            exports = _collab_exports(
                rank=rank, nproc=nproc, world_size=world_size, master_addr=master_addr, port=port
            )
            launch = _launch_script(
                root, session, job_id, f"{exports}\n{command}", host.env_setup, host.workdir
            )
            pending.append((host, job_id, session, launch))
            rank += nproc

        launched = await asyncio.gather(
            *(
                self.runner.exec_on(host, launch, timeout=90, workdir=False)
                for host, _, _, launch in pending
            )
        )
        for (host, job_id, session, _), res in zip(pending, launched, strict=True):
            results.append(res)
            if res.ok and "LAUNCHER=" in res.stdout:
                handles.append(
                    JobHandle(job_id, host.name, session, _launcher_of(res.stdout))
                )
        return handles, results

    async def status_many(self, pairs: list[tuple[Host, str]]) -> list[dict]:
        """Status for ``[(host, job_id), ...]`` in parallel."""
        return list(
            await asyncio.gather(*(self.status(h, jid) for h, jid in pairs))
        )

    async def status(self, host: Host, job_id: str) -> dict:
        _validate_job_id(job_id)
        root = _job_root(host, job_id)
        q = root
        alive_probe = (
            f"if tmux has-session -t jms-{job_id} 2>/dev/null; then echo yes; "
            f"elif [ -f {q}/pid ] && kill -0 \"$(cat {q}/pid)\" 2>/dev/null; then echo yes; "
            f"else echo no; fi"
        )
        cmd = (
            f"if [ -d {q} ]; then "
            f"echo \"status=$(cat {q}/status 2>/dev/null || echo unknown)\"; "
            f"echo \"exit_code=$(cat {q}/exit_code 2>/dev/null || echo '')\"; "
            f"echo \"pid=$(cat {q}/pid 2>/dev/null || echo '')\"; "
            f"echo \"log_bytes=$(stat -c %s {q}/stdout.log 2>/dev/null || echo 0)\"; "
            f"echo \"alive=$({alive_probe})\"; "
            f"else echo 'status=missing'; fi"
        )
        res = await self.runner.exec_on(host, cmd, timeout=60, workdir=False)
        parsed: dict[str, str] = {"host": host.name, "job_id": job_id}
        for line in res.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                parsed[k.strip()] = v.strip()
        parsed["ok"] = res.ok
        return parsed

    async def logs(
        self, host: Host, job_id: str, *, offset: int = 0, max_bytes: int = 65536
    ) -> dict:
        """Return log bytes from ``offset`` onward (for incremental polling)."""
        _validate_job_id(job_id)
        log_path = f"{_job_root(host, job_id)}/stdout.log"
        cmd = (
            f"if [ -f {log_path} ]; then "
            f"size=$(stat -c %s {log_path}); "
            f"echo \"size=$size\"; "
            f"tail -c +{int(offset) + 1} {log_path} | head -c {int(max_bytes)}; "
            f"else echo 'size=0'; fi"
        )
        res = await self.runner.exec_on(host, cmd, timeout=60, workdir=False)
        size = 0
        body = res.stdout
        if body.startswith("size="):
            first, _, rest = body.partition("\n")
            try:
                size = int(first.split("=", 1)[1])
            except ValueError:
                size = 0
            body = rest
        return {"host": host.name, "job_id": job_id, "offset": offset, "size": size, "log": body}

    async def cancel(self, host: Host, job_id: str) -> dict:
        _validate_job_id(job_id)
        root = _job_root(host, job_id)
        cmd = (
            f"killed=no; "
            f"if tmux has-session -t jms-{job_id} 2>/dev/null; then "
            f"tmux kill-session -t jms-{job_id} && killed=yes; fi; "
            f"if [ -f {root}/pid ]; then pid=$(cat {root}/pid); "
            f'kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null; killed=yes; fi; '
            f"echo cancelled > {root}/status 2>/dev/null || true; "
            f'echo "killed=$killed"'
        )
        res = await self.runner.exec_on(host, cmd, timeout=60, workdir=False)
        return {
            "host": host.name,
            "job_id": job_id,
            "killed": "killed=yes" in res.stdout,
            "ok": res.ok,
        }

    async def list_jobs(self, host: Host) -> list[str]:
        cmd = "tmux ls 2>/dev/null | grep -o '^jms-[0-9a-f]\\{8\\}' || true"  # tmux hosts only
        res = await self.runner.exec_on(host, cmd, timeout=60, workdir=False)
        return [
            line.strip().removeprefix("jms-")
            for line in res.stdout.splitlines()
            if line.strip()
        ]


def _validate_job_id(job_id: str) -> None:
    if not _JOB_ID_RE.match(job_id):
        raise ValueError(f"invalid job id {job_id!r} (expected 8 hex chars)")


async def wait_for(predicate, timeout: float, interval: float = 2.0):  # noqa: ANN001 - small helper
    """Poll ``predicate`` (an async callable) until true or ``timeout`` elapses."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        if await predicate():
            return True
        if asyncio.get_event_loop().time() >= deadline:
            return False
        await asyncio.sleep(interval)