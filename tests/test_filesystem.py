import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import jsonschema
from fs_harness import execute as execute_probe
from fs_harness import without_rg

from jms_bridge import bash_fs
from jms_bridge.config import BastionConfig
from jms_bridge.filesystem import SCHEMA, Filesystem, remote_script, validate
from jms_bridge.results import ResultStore
from jms_bridge.runtime_config import BridgeSettings


class FS(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "src").mkdir()
        (self.root / "src/a.py").write_text("first\nneedle α\nlast\n")
        (self.root / "src/b.py").write_text("second NEEDLE\n")
        (self.root / "config.json").write_text('{"a":1}\n')
        (self.root / ".secret").write_text("hidden\n")
        (self.root / "symlink").symlink_to(self.root / "src", target_is_directory=True)

    def run_fs(self, action, **kwargs):
        return execute_probe(dict(action=action, path=str(self.root), **kwargs))

    def test_list_pagination_and_tree_no_symlink_recursion(self):
        first = self.run_fs("list", limit=2)
        second = self.run_fs("list", limit=2, offset=first["next_offset"])
        self.assertEqual(
            [r["name"] for r in first["entries"] + second["entries"]],
            ["config.json", "src", "symlink"],
        )
        self.assertFalse(second["truncated"])
        tree = self.run_fs("tree", hidden=True)
        self.assertEqual(len(tree["entries"]), 6)
        self.assertFalse(any("symlink/" in r["path"] for r in tree["entries"]))

    def test_stat_symlink_and_checksum(self):
        value = execute_probe(dict(action="stat", path=str(self.root / "symlink")))
        self.assertEqual(value["type"], "symlink")
        value = execute_probe(dict(action="checksum", path=str(self.root / "config.json")))
        self.assertEqual(value["digest"], hashlib.sha256(b'{"a":1}\n').hexdigest())

    def test_read_lines_and_continuation(self):
        opts = dict(action="read", path=str(self.root / "src/a.py"), start_line=2, line_count=1)
        value = execute_probe(opts)
        self.assertEqual(value["text"], "needle α\n")
        value2 = execute_probe(
            dict(action="read", path=opts["path"], byte_offset=value["next_byte_offset"])
        )
        self.assertEqual(value2["text"], "last\n")

    def test_utf8_byte_pages_reassemble_long_line(self):
        path = self.root / "long"
        text = "汉🙂é" * 4000 + "\n"
        path.write_text(text)
        offset, chunks = 0, []
        while True:
            value = execute_probe(
                dict(action="read", path=str(path), byte_offset=offset, max_bytes=257)
            )
            chunks.append(value["text"])
            offset = value["next_byte_offset"]
            if value["eof"]:
                break
        self.assertEqual("".join(chunks), text)

    def test_tail_partial_batch_and_binary_fifo(self):
        value = execute_probe(
            dict(action="read", path=str(self.root / "src/a.py"), tail=True, line_count=2)
        )
        self.assertEqual(value["text"], "needle α\nlast\n")
        value = execute_probe(
            dict(
                action="read_many", paths=[str(self.root / "src/a.py"), str(self.root / "missing")]
            )
        )
        self.assertFalse(value["ok"])
        self.assertTrue(value["files"][0]["ok"])
        (self.root / "binary").write_bytes(b"\x00\xff")
        self.assertIn(
            "binary", execute_probe(dict(action="read", path=str(self.root / "binary")))["message"]
        )
        os.mkfifo(self.root / "fifo")
        self.assertIn(
            "regular", execute_probe(dict(action="read", path=str(self.root / "fifo")))["message"]
        )

    def test_bash_glob_search_regex_context_and_excludes(self):
        with without_rg():
            value = self.run_fs("glob", pattern="**/*.py")
            self.assertEqual(len(value["entries"]), 2)
            self.assertIn("warnings", value)
            value = self.run_fs(
                "search",
                pattern="NEED.*",
                literal=False,
                case_sensitive=False,
                context_lines=1,
                include=["*.py"],
                exclude=["b.py"],
            )
            self.assertEqual([r["line"] for r in value["entries"]], [1, 2, 3])
            self.assertEqual([r["kind"] for r in value["entries"]], ["context", "match", "context"])

    def test_search_single_file_both_engines_and_explicit_symlink(self):
        path = self.root / "a-link.py"
        path.symlink_to(self.root / "src/a.py")
        opts = dict(action="search", path=str(path), pattern="needle")
        with without_rg():
            value = execute_probe(opts)
            self.assertEqual(value["entries"][0]["line"], 2)
        if shutil.which("rg"):
            value = execute_probe(opts)
            self.assertEqual(value["entries"][0]["line"], 2)

    def test_rg_invalid_regex_reports_error(self):
        if not shutil.which("rg"):
            self.skipTest("rg unavailable")
        value = self.run_fs("search", pattern="[", literal=False)
        self.assertFalse(value["ok"])
        self.assertEqual(value["errors"][0]["error"], "ripgrep_failed")

    def test_read_tail_exact_boundary_and_size_limit(self):
        path = self.root / "tail"
        path.write_text("a\nb\nc\n")
        result = execute_probe(dict(action="read", path=str(path), tail=True, max_bytes=4))
        self.assertEqual(result["text"], "b\nc\n")
        self.assertFalse(
            execute_probe(dict(action="checksum", path=str(path), max_file_bytes=3))["ok"]
        )

    @unittest.skipUnless(shutil.which("rg"), "rg unavailable")
    def test_rg_glob_ignore_and_search_pages(self):
        (self.root / ".git").mkdir()
        (self.root / ".gitignore").write_text("src/b.py\n")
        value = self.run_fs("glob", pattern="**/*.py")
        self.assertEqual([r["relative_path"] for r in value["entries"]], ["src/a.py"])
        value = self.run_fs(
            "search", pattern="needle", case_sensitive=False, respect_ignore=False, limit=1
        )
        self.assertEqual(len(value["entries"]), 1)
        self.assertEqual(value["next_offset"], 1)
        next_page = self.run_fs(
            "search", pattern="needle", case_sensitive=False, respect_ignore=False, offset=1
        )
        self.assertEqual(next_page["entries"][0]["relative_path"], "src/b.py")

    def test_no_shell_interpolation_and_probe_protocol(self):
        name = "quotes ' \" $(touch BAD) `touch BAD`\n.json"
        path = self.root / name
        path.write_text("safe\n")
        script = remote_script(dict(action="read", path=str(path), timeout=2))
        response = subprocess.run(
            ["bash", "-s"], input=script, text=True, capture_output=True, cwd=self.root, timeout=5
        )
        self.assertEqual(response.returncode, 0, response.stderr)
        self.assertEqual(
            bash_fs.decode(
                response.stdout.encode(), dict(action="read", path=str(path), timeout=2)
            )["text"],
            "safe\n",
        )
        self.assertFalse((self.root / "BAD").exists())

    def test_budget_and_bounded_scan(self):
        for i in range(100):
            (self.root / f"file{i:03}").write_text("needle\n")
        value = self.run_fs("list", max_bytes=256)
        self.assertEqual(value["returned"], 1)
        self.assertTrue(value["truncated"])
        with patch("jms_bridge.bash_fs.SCAN_LIMIT", 10):
            value = self.run_fs("tree")
        self.assertTrue(value["scan_limited"])
        self.assertFalse(value["complete"])

    def test_utility_timeout_is_structured_not_hung(self):
        with without_rg() as commands:
            (commands / "grep").unlink()
            (commands / "grep").write_text("#!/bin/bash\nsleep 5\n")
            (commands / "grep").chmod(0o755)
            value = self.run_fs("search", pattern="needle", timeout=1)
        self.assertTrue(value["timed_out"])
        self.assertFalse(value["complete"])

    def test_validation_rejects_wrong_action_options_and_paths(self):
        for options in [
            dict(target="x", action="read", path="relative"),
            dict(target="x", action="read", path="/a", start_line=1, byte_offset=2),
            dict(target="x", action="stat", path="/a", tail=True),
        ]:
            with self.assertRaises(ValueError):
                validate(options)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(dict(target="x", action="write", path="/a"), SCHEMA["inputSchema"])


class Integration(unittest.IsolatedAsyncioTestCase):
    async def test_transfer_receipt_prevents_duplicate(self):
        from jms_bridge.compact import BridgeService

        with tempfile.TemporaryDirectory() as td:
            service = BridgeService(
                BastionConfig(url="https://bastion.example.com", login_user="example-user"),
                BridgeSettings(state_dir=Path(td)),
            )
            service.filesystem.execute = AsyncMock(return_value={"ok": True, "result_id": "a" * 32})
            args = dict(
                target="x",
                action="download",
                path="/a",
                local_path=td + "/a",
                request_id="fs-transfer-one",
            )
            first = await service.fs(args)
            second = await service.fs(args)
            self.assertTrue(first["ok"])
            self.assertTrue(second["duplicate"])
            self.assertEqual(service.filesystem.execute.await_count, 1)
            with self.assertRaises(ValueError):
                await service.fs(args | {"path": "/b"})
            await service.runner.close()

    async def test_rsync_transfer_overwrite_and_reverse_direction(self):
        import uuid

        from test_execution_scheduling import LocalConnection

        from jms_bridge.inventory import Host
        from jms_bridge.native_ssh import NativeSSHRunner

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = ResultStore(root / "results")
            runner = NativeSSHRunner(
                BastionConfig(url="https://bastion.example.com"),
                BridgeSettings(state_dir=root),
                store=store,
            )
            connection = LocalConnection()
            runner.connection = AsyncMock(return_value=connection)
            host = Host(name="test", asset_id=str(uuid.uuid4()), login="test", account="root")
            catalog = SimpleNamespace(
                ensure=AsyncMock(return_value=SimpleNamespace(resolve=lambda _: [host]))
            )
            fs = Filesystem(catalog, runner, store)
            src, dst, pulled = [root / name for name in ("source space", "dest space", "pulled")]
            src.write_text("payload-one")
            args = dict(target="test", action="upload", path=str(dst), local_path=str(src))
            self.assertTrue((await fs.execute(**args))["ok"])
            self.assertEqual(dst.read_text(), "payload-one")
            src.write_text("payload-longer-two")
            self.assertTrue((await fs.execute(**args))["ok"])
            self.assertEqual(dst.read_text(), "payload-one")
            self.assertTrue((await fs.execute(**args, overwrite=True))["ok"])
            self.assertEqual(dst.read_text(), "payload-longer-two")
            result = await fs.execute(
                target="test", action="download", path=str(dst), local_path=str(pulled)
            )
            self.assertTrue(result["ok"])
            self.assertEqual(pulled.read_text(), src.read_text())
            await runner.close()

    async def test_transfer_container_forwarding(self):
        with tempfile.TemporaryDirectory() as td:
            store = ResultStore(Path(td) / "results")
            rid = store.create(host="one")
            store.finish(
                rid,
                ok=True,
                exit_code=0,
                timing={},
                completion_unknown=False,
                execution_state="completed",
                queue_timed_out=False,
                failure_stage=None,
            )
            runner = SimpleNamespace(rsync=AsyncMock(return_value=SimpleNamespace(result_id=rid)))
            host = SimpleNamespace(name="one")
            catalog = SimpleNamespace(
                ensure=AsyncMock(return_value=SimpleNamespace(resolve=lambda _: [host]))
            )
            fs = Filesystem(catalog, runner, store)
            await fs.execute(
                target="one",
                action="download",
                path="/a",
                local_path=td + "/a",
                container="existing",
            )
            kwargs = runner.rsync.call_args.kwargs
            self.assertEqual(kwargs["container"], "existing")
            self.assertFalse(kwargs["delete"])
            self.assertFalse(kwargs["overwrite"])

    async def test_schema_tools_and_fs_dispatch(self):
        from jms_bridge.compact import BridgeService

        with tempfile.TemporaryDirectory() as td:
            service = BridgeService(
                BastionConfig(url="https://bastion.example.com", login_user="example-user"),
                BridgeSettings(state_dir=Path(td)),
            )
            tools = await service.server.list_tools()
            self.assertEqual(
                {t.name for t in tools},
                {"run", "read_result", "discover", "call", "fs", "meta", "terminal"},
            )
            schema = await service.discover(operation="jms_fs")
            self.assertFalse(schema["annotations"]["readOnlyHint"])
            service.filesystem.execute = AsyncMock(return_value={"ok": True})
            await service.call("jms_fs", dict(target="x", action="stat", path="/etc/os-release"))
            service.filesystem.execute.assert_awaited_once()
            await service.runner.close()

    async def test_fs_target_resolution_and_container_forwarding(self):
        with tempfile.TemporaryDirectory() as td:
            store = ResultStore(Path(td) / "results")
            rid = store.create()
            sample = Path(td) / "sample"
            sample.write_text("hello")
            wire = subprocess.run(
                ["bash", "-s"],
                input=remote_script(dict(action="read", path=str(sample))).encode(),
                capture_output=True,
                check=True,
            )
            store.path(rid, "stdout").write_bytes(wire.stdout)
            store.path(rid, "stderr").write_bytes(wire.stderr)
            store.finish(rid)
            host = SimpleNamespace(name="one")
            catalog = SimpleNamespace(
                ensure=AsyncMock(return_value=SimpleNamespace(resolve=lambda targets: [host]))
            )
            runner = SimpleNamespace(
                execute_to_store=AsyncMock(return_value=dict(ok=True, result_id=rid, timing={}))
            )
            fs = Filesystem(catalog, runner, store)
            fs.workers.execute = runner.execute_to_store
            result = await fs.execute(target="one", action="read", path="/a", container="existing")
            self.assertEqual(result["data"]["text"], "hello")
            self.assertEqual(runner.execute_to_store.call_args.kwargs["container"], "existing")
            with self.assertRaises(ValueError):
                await fs.execute(target="all", action="read", path="/a")
            self.assertEqual(runner.execute_to_store.await_count, 1)
