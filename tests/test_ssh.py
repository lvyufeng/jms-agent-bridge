import asyncio
import contextlib
import os
import signal

import pytest

from jms_bridge.config import BastionConfig, ConfigError
from jms_bridge.inventory import Host
from jms_bridge.ssh import ExecResult, SSHRunner


def make_runner(**kw):
    cfg = BastionConfig(url="https://bastion.example.com", ssh_port=2224, login_user="me", **kw)
    return SSHRunner(cfg)


def test_target_is_jumpserver_login_string():
    r = make_runner()
    h = Host(name="a", login="alice", account="root", asset_id="ASSET")
    assert r.target(h) == "alice#root#ASSET@bastion.example.com"


def test_target_strips_scheme_and_path():
    r = SSHRunner(BastionConfig(url="https://bastion.example.com/some/path", ssh_port=22))
    h = Host(name="a", login="u", asset_id="A")
    assert r.target(h) == "u#A@bastion.example.com"


def test_ssh_argv_is_argv_list_not_shell_string():
    r = make_runner()
    h = Host(name="weird name; rm -rf /", login="u", asset_id="A")
    argv = r._ssh_argv(h)
    assert isinstance(argv, list)
    assert argv[0] == "ssh"
    assert "-p" in argv and argv[argv.index("-p") + 1] == "2224"
    # the host string is a single argv element — the local shell never sees it
    assert "u#A@bastion.example.com" in argv
    # remote command is piped via `bash -s`, not embedded in the target
    assert argv[-2:] == ["u#A@bastion.example.com", "bash -s"]


def test_ssh_key_adds_identities_only():
    r = make_runner(ssh_key="/tmp/k")
    argv = r._ssh_argv(Host(name="a", login="u", asset_id="A"))
    assert "-i" in argv and "/tmp/k" in argv
    assert "IdentitiesOnly=yes" in argv


def test_controlmaster_is_enabled_for_reuse():
    r = make_runner()
    argv = r._ssh_argv(Host(name="a", login="u", asset_id="A"))
    joined = " ".join(argv)
    assert "ControlMaster=auto" in joined
    assert "ControlPersist=300" in joined


def test_bastion_password_uses_sshpass_and_forces_password_auth(monkeypatch):
    # This tests argv construction, not the workstation's installed binaries.
    monkeypatch.setattr("jms_bridge.ssh.shutil.which", lambda name: f"/usr/bin/{name}")
    r = make_runner(ssh_password="s3cret")
    h = Host(name="a", login="u", asset_id="A")
    argv = r._ssh_argv(h)
    assert argv[0] == "sshpass" and argv[1] == "-e"
    assert "PubkeyAuthentication=no" in argv
    assert "PreferredAuthentications=password" in argv
    # password rides in the env, never in argv (argv is visible in `ps`)
    assert "s3cret" not in argv
    assert r._subproc_env(h)["SSHPASS"] == "s3cret"


def test_direct_host_env_carries_no_bastion_password():
    """The bastion password must reach only the child that will actually use it.

    A direct host is key-based, so `sshpass` is not in its argv — handing it SSHPASS
    anyway is the password where it has no business being. `close()`'s `ssh -O exit`
    talks to the control socket and needs none either, so the hostless default omits it.
    """
    r = make_runner(ssh_password="s3cret")
    direct = Host(name="d", ssh_host="h", ssh_user="ubuntu")
    assert "SSHPASS" not in r._subproc_env(direct)
    assert "SSHPASS" not in r._subproc_env()


def test_direct_host_ignores_bastion_password():
    """A direct host stays key-based even when a bastion password is configured."""
    r = make_runner(ssh_password="s3cret")
    argv = r._ssh_argv(Host(name="d", ssh_host="h", ssh_user="ubuntu"))
    assert "sshpass" not in argv
    assert "PreferredAuthentications=password" not in argv


def test_exec_on_env_setup_false_omits_source_line():
    """``env_setup=False`` must not source the host toolchain in the remote script.

    Workspace commands run inside a container, where the host env script is absent;
    leaking it in would abort the command under ``set -e``. Capture the script the
    runner would pipe, without touching SSH.
    """
    import asyncio

    r = make_runner()
    h = Host(
        name="a",
        login="u",
        asset_id="A",
        env_setup="source /opt/cann/set_env.sh",
        workdir="/work",
    )
    captured = {}

    async def fake_run_script(host, script, timeout, *, tty=False):
        captured["script"] = script
        return ExecResult(host.name, 0, "", "")

    r._run_script = fake_run_script

    asyncio.run(r.exec_on(h, "echo hi", env_setup=False, workdir=False))
    assert "set_env.sh" not in captured["script"]
    assert "source" not in captured["script"]
    assert captured["script"].rstrip().endswith("echo hi")

    asyncio.run(r.exec_on(h, "echo hi", env_setup=True, workdir=False))
    assert "source /opt/cann/set_env.sh" in captured["script"]


# --- a missing local tool must say what to install ----------------------------------


def _hide_tool(monkeypatch, *binaries):
    """Pretend ``binaries`` are absent, leaving every other tool alone."""
    import shutil

    real = shutil.which

    def which(name, *a, **kw):
        return None if name in binaries else real(name, *a, **kw)

    monkeypatch.setattr("jms_bridge.ssh.shutil.which", which)


def test_missing_ssh_says_what_to_install(monkeypatch):
    # `create_subprocess_exec` would raise a bare FileNotFoundError, which the MCP SDK
    # shows the agent as "Error executing tool <name>" — useless on a fresh clone.
    _hide_tool(monkeypatch, "ssh")
    r = make_runner()
    with pytest.raises(ConfigError) as e:
        asyncio.run(r.exec_on(Host(name="a", login="u", asset_id="A"), "echo hi"))
    assert "`ssh` is not installed" in str(e.value)


def test_missing_rsync_says_what_to_install(monkeypatch):
    _hide_tool(monkeypatch, "rsync")
    r = make_runner()
    host = Host(name="a", login="u", asset_id="A")
    with pytest.raises(ConfigError) as e:
        asyncio.run(r.rsync(host, "/local", "/remote"))
    assert "`rsync` is not installed" in str(e.value)


def test_missing_sshpass_is_a_config_error(monkeypatch):
    _hide_tool(monkeypatch, "sshpass")
    r = make_runner(ssh_password="s3cret")
    with pytest.raises(ConfigError) as e:
        r._ssh_argv(Host(name="a", login="u", asset_id="A"))
    assert "sshpass" in str(e.value) and "install" in str(e.value)


# --- the fan-out cap is configurable, and can never be a hang -----------------------


def test_max_parallel_defaults_to_eight():
    assert make_runner().max_parallel == 8


def test_max_parallel_is_configurable():
    cfg = BastionConfig(url="https://bastion.example.com", max_parallel=32)
    assert SSHRunner(cfg, max_parallel=cfg.max_parallel).max_parallel == 32


def test_max_parallel_below_one_cannot_deadlock_the_semaphore():
    # Semaphore(0) is a hang, not an error, and the loader rejects 0 — this is the last
    # line of defence for a directly-constructed runner (tests, embedders).
    cfg = BastionConfig(url="https://bastion.example.com")
    assert SSHRunner(cfg, max_parallel=0).max_parallel == 1
    assert SSHRunner(cfg, max_parallel=-4).max_parallel == 1


# --- a cancelled or timed-out call must reap its child process ----------------------


class _ChildTracker:
    """Stands in for the runner's subprocess spawn, handing back a real long-lived child.

    A subprocess is what actually leaks, so the test uses a real one: the tracked process
    is a ``sleep`` and the assertion is that none survives the call. ``_real_exec`` is
    captured before patching, so the tracker's own spawn does not recurse into the patch.
    """

    def __init__(self):
        self.procs: list[asyncio.subprocess.Process] = []

    async def spawn(self, real_exec):
        proc = await real_exec(
            "sleep",
            "300",
            # `stdin=PIPE` matters: `_run_script` calls `communicate(script)`, and on
            # Python 3.10/3.11 that raises AttributeError when stdin is None (3.12+
            # tolerates it). The fake must mirror the real spawn's shape, or the test
            # only passes on the newer interpreter.
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.procs.append(proc)
        return proc


_REAL_EXEC = asyncio.create_subprocess_exec


def _patch_spawn(monkeypatch, tracker):
    """Make the runner's spawn return a tracked real child."""

    async def fake_exec(*argv, **kw):
        return await tracker.spawn(_REAL_EXEC)

    monkeypatch.setattr("jms_bridge.ssh.asyncio.create_subprocess_exec", fake_exec)
    monkeypatch.setattr("jms_bridge.ssh.shutil.which", lambda name, *a, **k: "/usr/bin/" + name)


def test_cancelling_a_command_kills_the_child_process(monkeypatch):
    """A cancelled tool call must not leave ssh running.

    ``asyncio.CancelledError`` is a ``BaseException``; catching only ``TimeoutError`` (as
    the runner used to) skips the kill entirely, so every cancelled call — a client
    disconnect, a long response cut off by the gateway — leaked one ssh process holding
    the ControlMaster socket.
    """
    r = make_runner()
    tracker = _ChildTracker()
    _patch_spawn(monkeypatch, tracker)

    h = Host(name="a", login="u", asset_id="A")

    async def main():
        task = asyncio.create_task(r._run_script(h, "echo hi", 600))
        await asyncio.sleep(0.2)  # let the child start
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert tracker.procs, "no child was spawned"
    for proc in tracker.procs:
        assert proc.returncode is not None, "cancelling leaked the child process"


def test_a_timed_out_command_kills_the_child_process(monkeypatch):
    """The timeout path must still kill (and now also reap) the child."""
    r = make_runner()
    tracker = _ChildTracker()
    _patch_spawn(monkeypatch, tracker)

    h = Host(name="a", login="u", asset_id="A")
    res = asyncio.run(r._run_script(h, "echo hi", 0.2))
    assert res.timed_out is True and res.exit_code == 124
    assert tracker.procs and all(p.returncode is not None for p in tracker.procs)


def test_kill_helper_ignores_an_already_dead_process():
    """`_kill` must be safe on a process that already exited (race between check/kill)."""
    from jms_bridge.ssh import _kill

    async def main():
        proc = await asyncio.create_subprocess_exec(
            "true", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc.wait()
        await _kill(proc)  # must not raise ProcessLookupError

    asyncio.run(main())


def test_run_script_spawns_in_its_own_session(monkeypatch):
    """The child must lead its own process group, or `_kill` cannot take down the tree.

    `sshpass -e ssh ...` is two processes. Killing only the group leader leaves the real
    ssh running; a surviving ssh keeps the ControlMaster socket alive for ControlPersist
    seconds, and the next call then reuses a half-dead channel — observed as a 2s timeout
    becoming a 10s call that returned exit 0 with a DNS error on stdout.
    """
    r = make_runner()
    tracker = _ChildTracker()
    seen = {}

    async def fake_exec(*argv, **kw):
        seen.update(kw)
        return await tracker.spawn(_REAL_EXEC)

    monkeypatch.setattr("jms_bridge.ssh.asyncio.create_subprocess_exec", fake_exec)
    monkeypatch.setattr("jms_bridge.ssh.shutil.which", lambda name, *a, **k: "/usr/bin/" + name)

    h = Host(name="a", login="u", asset_id="A")

    async def main():
        # The fake child is `sleep 300`, so cancel rather than wait out a real timeout.
        task = asyncio.create_task(r._run_script(h, "echo hi", 600))
        await asyncio.sleep(0.2)  # let the spawn happen
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert seen.get("start_new_session") is True


def test_kill_takes_down_the_whole_process_group(monkeypatch):
    """A grandchild (sshpass -> ssh) must die with its leader.

    Signals the *real* group here, unlike the tracker-based tests, because the point is
    that the group is killed rather than only the leader.

    The probe is ``killpg(pgid, 0)`` rather than scraping ``ps``: an orphaned grandchild
    is reparented and its recorded pgid stops being a reliable handle (on macOS ``ps``
    then reports the group as already empty while the process is still alive), whereas
    signal 0 resolves the group in the kernel on every platform.

    The *verdict* must not be "the first probe raises ESRCH". ``proc.wait()`` reaps only
    the leader; the grandchildren are killed-but-unreaped at that moment, and the kernel
    drops a process group only when its last member is reaped. On CI (Linux, reaping
    under load) the group still answers for tens of milliseconds — measured at ~50 ms —
    while on macOS it answers ESRCH at once, so probing a single time raced and failed
    on three of four CI versions but never locally. Poll until the group is gone, and
    keep the budget short enough that a *surviving* grandchild can never pass.
    """
    from jms_bridge.ssh import _kill

    def group_gone(pgid: int) -> bool:
        """True once the kernel no longer resolves the group at all."""
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:  # pragma: no cover - not reachable for our own child
            return False
        return False

    async def main():
        proc = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            "sleep 300 & sleep 300",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        await asyncio.sleep(0.3)
        pgid = os.getpgid(proc.pid)
        try:
            assert pgid != os.getpgid(0), "child must not share the test runner's group"
            assert not group_gone(pgid), "the group must be alive before the kill"

            await _kill(proc)
            assert proc.returncode is not None

            # Reaping is asynchronous; give the kernel a bounded window to catch up. A
            # grandchild that truly survived stays resolvable and still fails the assert.
            for _ in range(50):  # ~1s, far above the ~50ms observed on a loaded CI runner
                if group_gone(pgid):
                    break
                await asyncio.sleep(0.02)
            assert group_gone(pgid), "a process in the group outlived the leader"
        finally:
            # A failing run may leave a survivor behind; never leak a 300s sleeper.
            if not group_gone(pgid):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(pgid, signal.SIGKILL)

    asyncio.run(main())


def test_kill_refuses_to_signal_its_own_process_group(monkeypatch):
    """`killpg` on our own group would take down the MCP server (or pytest)."""
    from jms_bridge.ssh import _kill

    killed = {}

    async def fake_exec(*argv, **kw):
        # Deliberately *without* start_new_session: the child shares our group.
        return await _REAL_EXEC(
            "sleep",
            "300",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def main():
        proc = await fake_exec()
        await asyncio.sleep(0.1)
        assert os.getpgid(proc.pid) == os.getpgid(0)

        real_killpg = os.killpg

        def spy_killpg(pgid, sig):
            killed["pgid"] = pgid
            return real_killpg(pgid, sig)

        monkeypatch.setattr("jms_bridge.ssh.os.killpg", spy_killpg)
        await _kill(proc)
        # The group was never signalled, and the child is still reaped.
        assert "pgid" not in killed
        assert proc.returncode is not None

    asyncio.run(main())


def test_a_timeout_closes_the_hosts_control_master(monkeypatch):
    """A timed-out call must not leave a reusable ControlMaster behind.

    Without this, the surviving master answers the next call on a half-dead channel,
    yielding exit 0 with a transport error on stdout — a false green.
    """
    r = make_runner()
    tracker = _ChildTracker()
    _patch_spawn(monkeypatch, tracker)

    closed = []

    async def fake_close(host):
        closed.append(host.name)

    monkeypatch.setattr(r, "_close_control_master", fake_close)

    h = Host(name="a", login="u", asset_id="A")
    res = asyncio.run(r._run_script(h, "echo hi", 0.2))
    assert res.timed_out is True
    assert closed == ["a"], "timeout must tear the ControlMaster down"


def test_close_control_master_is_a_noop_without_a_socket(monkeypatch, tmp_path):
    """No socket means no master to talk to; the helper must not spawn or raise."""
    from jms_bridge.ssh import SSHRunner

    r = make_runner()
    h = Host(name="never-connected", login="u", asset_id="A")
    spawned = []

    async def fake_exec(*argv, **kw):
        spawned.append(argv)
        raise AssertionError("must not spawn without a socket")

    monkeypatch.setattr("jms_bridge.ssh.asyncio.create_subprocess_exec", fake_exec)
    asyncio.run(r._close_control_master(h))
    assert spawned == []
    assert isinstance(r, SSHRunner)
