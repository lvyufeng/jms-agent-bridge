"""Fast local checks for the benchmark; CI does not repeat 45-second cold pacing."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_concurrency.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_concurrency", _SCRIPT)
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)


@pytest.fixture
async def warmed():
    async with bench.fleet(10) as (service, hosts, ssh):
        connection = await service.runner.connection(hosts[0])
        # Fast scheduler/channel regression only. The standalone experiment creates
        # ten separate real logins at unchanged production pacing instead.
        for host in hosts:
            service.runner._connections[service.runner.key(host)] = connection
        yield service, hosts, ssh


def test_distribution_nearest_rank_and_empty_input():
    assert bench.distribution([]) == {"count": 0}
    assert bench.distribution([0.4, 0.1, 0.3, 0.2]) == {
        "count": 4,
        "p50": 0.25,
        "p95": 0.4,
        "max": 0.4,
    }


async def test_ten_worker_reserve_and_burst_cleanup(warmed):
    service, hosts, ssh = warmed
    for reserve in (0, 1, 2):
        row = await bench.reserve_case(service, hosts, reserve, ssh, seconds=0.15)
        assert row["workers"]["successful"] == 10 and row["inspection"]["ok"]
        assert row["saturated"]["active"]["general"] == 8 - reserve
        assert 1 <= row["max_concurrent_children"] <= 8
        assert row["scheduler"]["peak_active"] <= 8
        assert sum(row["scheduler"]["active"].values()) == 0
        assert sum(row["scheduler"]["queued"].values()) == 0
    bench.reset_scheduler(service, 8, 2)
    burst = await bench.batch(service, hosts * 3, "sleep 0.03; printf burst", ssh)
    assert burst["summary"]["successful"] == 30
    assert 1 <= burst["max_concurrent_children"] <= 6
    assert burst["scheduler"]["peak_active"] <= 8
    assert burst["queue_sample_peak"]["general"] > 0
    await bench.until(lambda: not ssh.children)


async def test_cancelled_active_and_queued_workers_have_durable_results(warmed):
    service, hosts, ssh = warmed
    result = await bench.cancel_case(service, hosts)
    assert result["cancelled"] == 10 and result["results"] == 10
    assert result["states"] == {"not_started": 4, "unknown": 6, "completed": 0}
    assert result["all_results_complete"]
    assert sum(result["scheduler"]["active"].values()) == 0
    assert sum(result["scheduler"]["queued"].values()) == 0
    await bench.until(lambda: not ssh.children)
    assert not service.requests.tasks


async def test_slow_refresh_keeps_old_assets_and_disconnect_never_replays(warmed):
    service, hosts, ssh = warmed
    refreshed = await bench.slow_refresh_case(service, hosts)
    assert refreshed["old_view_used"] and refreshed["refresh_pending"] and refreshed["status_stale"]
    assert refreshed["request"]["ok"] and refreshed["request"]["stdout"] == "responsive"
    await bench.until(lambda: not ssh.children)
    disconnected = await bench.disconnect_case(service, hosts[-1], ssh)
    assert disconnected["partial_without_final"] and disconnected["channels_started"] == 1
    row = disconnected["result"]
    assert not row["ok"] and row["completion_unknown"] and row["execution_state"] == "unknown"
    assert disconnected["connection_entries_after"] == disconnected["connection_entries_before"]
    await bench.until(lambda: not ssh.children)


async def test_overload_times_out_without_remote_execution(warmed):
    service, hosts, ssh = warmed
    commands_before = len(ssh.commands)
    row = await bench.queue_timeout_case(service, hosts)
    assert row["requests"] == 10
    assert row["all_not_started"] and row["all_queue_timed_out"] and row["all_results_complete"]
    assert len(ssh.commands) == commands_before
    assert sum(row["scheduler"]["active"].values()) == 0
    assert sum(row["scheduler"]["queued"].values()) == 0


async def test_fleet_exit_cleans_children_handlers_and_restores_ambient_environment(monkeypatch):
    monkeypatch.setenv("JMS_BRIDGE_INVENTORY", "/must-not-be-read/user-hosts.yaml")
    async with bench.fleet(1) as (service, hosts, ssh):
        row = await bench.run_worker(service, hosts[0], "printf local")
        assert row["ok"] and row["stdout"] == "local"
        await bench.until(lambda: not ssh.children)
    import os

    assert os.environ["JMS_BRIDGE_INVENTORY"] == "/must-not-be-read/user-hosts.yaml"
    assert not ssh.children and not ssh.handlers


@pytest.mark.parametrize(
    "arguments",
    [["--rounds", "0"], ["--work-seconds", "nan"], ["--burst", "200", "--work-seconds", "2"]],
)
def test_cli_rejects_invalid_workload_before_starting(monkeypatch, arguments):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid input launched a benchmark")

    monkeypatch.setattr(bench, "benchmark", forbidden)
    with pytest.raises(SystemExit) as error:
        bench.main(arguments)
    assert error.value.code == 2


def test_output_report_never_overwrites_existing_file(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("KEEP")
    with pytest.raises(FileExistsError):
        bench.main(["--output", str(path)])
    assert path.read_text() == "KEEP"


@pytest.mark.parametrize("ok", [False, True])
def test_cli_writes_private_complete_report_without_running_cold_wave(tmp_path, monkeypatch, ok):
    async def result(**kwargs):
        return {"ok": ok, "scope": "synthetic test report"}

    monkeypatch.setattr(bench, "benchmark", result)
    path = tmp_path / "new-report.json"
    assert bench.main(["--output", str(path)]) == (0 if ok else 1)
    assert json.loads(path.read_text()) == {"ok": ok, "scope": "synthetic test report"}
    assert path.stat().st_mode & 0o777 == 0o600


async def test_exception_inside_fleet_still_closes_all_local_resources():
    with pytest.raises(RuntimeError, match="intentional failure"):
        async with bench.fleet(1) as (service, hosts, ssh):
            row = await bench.run_worker(service, hosts[0], "printf before-failure")
            assert row["ok"]
            raise RuntimeError("intentional failure")
    assert not ssh.children and not ssh.handlers
    assert service.runner.connection_status()["connected"] == 0


async def test_reset_scheduler_refuses_to_discard_active_work(warmed):
    service, hosts, ssh = warmed
    task = asyncio.create_task(bench.run_worker(service, hosts[0], "printf START; sleep 30"))
    try:
        await bench.until(lambda: service.runner.scheduler.active["general"] > 0)
        with pytest.raises(RuntimeError, match="previous work is active"):
            bench.reset_scheduler(service, 8, 2)
    finally:
        await bench.drain([task])
    await bench.until(lambda: not ssh.children)


async def test_local_server_rejects_non_shell_commands(warmed):
    service, hosts, ssh = warmed
    connection = service.runner._connections[service.runner.key(hosts[0])]
    process = await connection.create_process("touch forbidden")
    await process.wait_closed()
    assert process.exit_status == 126 and not (ssh.root / "forbidden").exists()


def test_script_help_runs_without_real_config(capsys):
    with pytest.raises(SystemExit) as error:
        bench.main(["--help"])
    assert error.value.code == 0
    assert "--output" in capsys.readouterr().out
