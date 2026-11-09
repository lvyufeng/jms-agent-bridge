import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from jms_bridge.accelerator_inspection import (
    OPERATION,
    AcceleratorInspector,
    device_state,
    parse_snapshot,
)
from jms_bridge.inspection_probe import command, safe_argv
from jms_bridge.inspection_vendor.families import Card
from jms_bridge.results import ResultStore

ROOT = Path(__file__).resolve().parents[1]


def nvidia(gpus=None, apps="", error=None):
    gpus = gpus or "0, NVIDIA A100, 0, 4, 40960, 30, 58.8, 00000000:03:00.0\n"
    return {
        "schema_version": 1,
        "finished_at": 1,
        "node": {},
        "vendors": {
            "nvidia": {
                "text": gpus + "\n" + apps,
                "errors": [error] if error else [],
                "queries": [
                    {"stdout": gpus, "rc": 0, "error": None},
                    {"stdout": apps, "rc": 1 if error else 0, "error": error},
                ],
            }
        },
    }


class ParsingTests(unittest.TestCase):
    def test_idle_requires_complete_metrics_and_process_query(self):
        self.assertEqual(parse_snapshot(nvidia())["state"], "free")
        broken = parse_snapshot(nvidia(error="process query failed"))
        self.assertEqual(broken["devices"][0]["state"], "unknown")
        self.assertFalse(broken["whole_node_available"])

    def test_idle_utilization_with_holder_is_busy_even_na_memory(self):
        result = parse_snapshot(nvidia(apps="0000:03:00.0, 4242, python, [N/A]\n"))
        self.assertEqual(result["state"], "busy")
        self.assertEqual(result["processes"][0]["pid"], 4242)
        self.assertEqual(result["processes"][0]["devices"], ["nvidia:0"])

    def test_unmapped_or_unparsed_process_never_means_free(self):
        for apps in ("00000000:04:00.0, 4242, python, 100\n", "invalid process line\n"):
            with self.subTest(apps=apps):
                result = parse_snapshot(nvidia(apps=apps))
                self.assertEqual(result["devices"][0]["state"], "unknown")

    def test_memory_held_without_visible_process(self):
        result = parse_snapshot(
            nvidia(gpus="0, NVIDIA A100, 0, 35000, 40960, 30, 58, 00000000:03:00.0\n")
        )
        self.assertEqual(result["state"], "memory_held")
        self.assertFalse(result["whole_node_available"])

    def test_missing_fields_and_invalid_numbers_are_unknown(self):
        for card in (
            Card(0),
            Card(0, util=0),
            Card(0, util=0, mem_used=0, mem_total=0),
            Card(0, util=float("nan"), mem_used=0, mem_total=100),
            Card(0, util=0, mem_used=-1, mem_total=100),
        ):
            self.assertEqual(device_state(card, True, True), "unknown")
        self.assertEqual(device_state(Card(0, health="ERROR"), True, True), "unhealthy")

    def test_recorded_vendor_fixtures(self):
        samples = {
            "ascend": "ascend_sample.txt",
            "mthreads": "mthreads_sample.txt",
            "kunlunxin": "kunlunxin_sample.txt",
            "metax": "metax_sample.txt",
            "enflame": "enflame_sample.txt",
            "tsingmicro": "tsingmicro_sample.txt",
            "thead": "thead_sample.txt",
            "iluvatar": "iluvatar_sample.txt",
            "hygon": "hygon_sample.txt",
        }
        for family, name in samples.items():
            with self.subTest(family=family):
                text = (ROOT / "tests/fixtures/accelerators" / name).read_text()
                result = parse_snapshot({"vendors": {family: {"text": text, "errors": []}}})
                self.assertGreater(len(result["devices"]), 0)
                self.assertEqual(len({d["id"] for d in result["devices"]}), len(result["devices"]))
                if family == "ascend":
                    self.assertEqual(len(result["devices"]), 16)
                    self.assertEqual(len({d["physical_card"] for d in result["devices"]}), 8)
                    self.assertEqual(result["devices"][1]["pids"], [41001])
                    self.assertEqual(result["devices"][2]["pids"], [41002])
                if family in ("mthreads", "ascend", "kunlunxin"):
                    self.assertGreater(len(result["processes"]), 0)
                if family == "tsingmicro":
                    self.assertFalse(result["whole_node_available"])

    def test_empty_driver_and_absent_tool_do_not_report_free(self):
        failed = parse_snapshot({"vendors": {"ascend": {"text": "", "errors": ["timeout"]}}})
        self.assertEqual(failed["state"], "unknown")
        self.assertEqual(parse_snapshot({"vendors": {}})["state"], "no_accelerator_detected")

    def test_secrets_redacted_before_serialization(self):
        args = [
            "python",
            "train.py",
            "--api-key",
            "secret1",
            "--password=secret2",
            "API_TOKEN=secret3",
            "https://user:secret4@host/a?token=secret5",
            "Authorization: Bearer secret6",
            "--model",
            "model-name",
        ]
        safe = " ".join(safe_argv(args))
        for i in range(1, 7):
            self.assertNotIn("secret" + str(i), safe)
        self.assertIn("train.py", safe)
        self.assertIn("model-name", safe)

    def test_vendor_process_timeout_and_output_limit(self):
        response = command([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.05)
        self.assertEqual(response["error"], "timeout")
        response = command([sys.executable, "-c", 'print("x" * 600000)'], timeout=3)
        self.assertEqual(response["error"], "output_limit")
        self.assertLessEqual(len(response["stdout"]), 512 * 1024)


class FakeCatalog:
    def __init__(self, hosts):
        self.hosts = hosts
        self.calls = 0

    async def ensure(self):
        self.calls += 1
        return self

    def resolve(self, names):
        if names == ["group"]:
            return list(self.hosts.values())
        return [self.hosts[names[0]]]

    def status(self):
        return {"stale": False}


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ResultStore(Path(self.tmp.name))
        self.hosts = {
            str(i): SimpleNamespace(name=str(i), address="example-" + str(i), asset_id=str(i))
            for i in range(9)
        }
        self.catalog = FakeCatalog(self.hosts)
        self.runner = SimpleNamespace(execute_to_store=AsyncMock(side_effect=self.execute))
        self.inspector = AcceleratorInspector(self.catalog, self.runner, self.store)

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def execute(self, host, script, timeout, **kwargs):
        rid = self.store.create(kind="run")
        self.store.path(rid, "stdout").write_text(json.dumps(nvidia()))
        self.store.path(rid, "stderr").write_text("")
        self.store.finish(rid, ok=True)
        return {"ok": True, "result_id": rid, "connection_state": "connected"}

    async def test_nine_targets_rejected_before_discovery_or_ssh(self):
        with self.assertRaises(ValueError):
            await self.inspector.inspect(list(self.hosts))
        self.assertEqual(self.catalog.calls, 0)
        self.runner.execute_to_store.assert_not_called()

    async def test_wildcard_and_group_expansion_rejected_before_ssh(self):
        for targets in (["all"], ["*"], ["example-*"], ["group"]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                await self.inspector.inspect(targets)
        self.runner.execute_to_store.assert_not_called()

    async def test_eight_host_summary_keeps_all_targets_and_details(self):
        result = await self.inspector.inspect(list(self.hosts)[:8])
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["hosts"]), 8)
        self.assertEqual(self.runner.execute_to_store.call_count, 8)
        for row in result["hosts"]:
            self.assertEqual(row["free_devices"], ["nvidia:0"])
            self.assertTrue(self.store.path(row["result_id"], "result").exists())
        self.assertEqual(
            len(json.loads(self.store.path(result["result_id"], "result").read_text())["hosts"]),
            8,
        )

    async def test_one_host_failure_does_not_erase_other_results(self):
        original = self.execute

        async def execute(host, script, timeout, **kwargs):
            if host.name == "1":
                raise OSError("simulated transport failure")
            return await original(host, script, timeout)

        self.runner.execute_to_store.side_effect = execute
        result = await self.inspector.inspect(["0", "1"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["hosts"][0]["state"], "free")
        self.assertEqual(result["hosts"][1]["state"], "unknown")

    async def test_tool_surface_and_hidden_operation_is_discoverable(self):
        from jms_bridge.compact import BridgeService
        from jms_bridge.config import BastionConfig
        from jms_bridge.runtime_config import BridgeSettings

        service = BridgeService(
            BastionConfig(url="https://bastion.example.com", login_user="example-user"),
            BridgeSettings(state_dir=Path(self.tmp.name) / "service"),
        )
        tools = await service.server.list_tools()
        self.assertEqual(
            {t.name for t in tools},
            {"run", "read_result", "discover", "call", "fs", "meta", "terminal"},
        )
        schema = await service.discover(operation=OPERATION)
        self.assertEqual(schema["inputSchema"]["properties"]["targets"]["maxItems"], 8)
        self.assertTrue(schema["annotations"]["readOnlyHint"])
        service.inspector = self.inspector
        result = await service.call(OPERATION, {"targets": ["0"]})
        self.assertEqual(result["hosts"][0]["state"], "free")
        import jsonschema

        with self.assertRaises(jsonschema.ValidationError):
            await service.call(OPERATION, {"targets": list(self.hosts)})


if __name__ == "__main__":
    unittest.main()
