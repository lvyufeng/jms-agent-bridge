import pytest

from jms_bridge.workspace import (
    Workspace,
    WorkspaceError,
    build_consistency_table,
    parse_pytest_summary,
)


def test_workspace_defaults():
    w = Workspace(name="w1", host="a", container="lvyufeng-torch-fl", repo="/work/Torch-FL")
    assert w.source == "git"
    assert w.container_cmd("echo hi") == "docker exec lvyufeng-torch-fl bash -lc 'echo hi'"


def test_workspace_proxy_git_wraps_proxy():
    w = Workspace(
        name="w1", host="a", container="lvyufeng-torch-fl", repo="/work/Torch-FL",
        source="proxy-git", proxy="http://proxy:8080",
    )
    cmd = w.container_cmd("git fetch")
    assert "http_proxy='http://proxy:8080'" in cmd
    assert "https_proxy='http://proxy:8080'" in cmd


def test_workspace_bad_source_raises():
    with pytest.raises(WorkspaceError):
        Workspace(name="w", host="a", container="c", repo="/r", source="ftp")

SAMPLE = """\
=================== short test summary info ===================
PASSED tests/test_ops.py::test_softmax
FAILED tests/test_ops.py::test_layer_norm - AssertionError: max diff 1e-3
SKIPPED [1] tests/test_ops.py::test_gemm_dtype: not supported on this backend
ERROR tests/test_ops.py::test_bad_setup - RuntimeError: boom
1 failed, 1 passed, 1 skipped, 1 error
"""


def test_parse_pytest_summary():
    got = parse_pytest_summary(SAMPLE)
    assert got["tests/test_ops.py::test_softmax"]["outcome"] == "PASS"
    assert got["tests/test_ops.py::test_layer_norm"]["outcome"] == "FAIL"
    assert "max diff" in got["tests/test_ops.py::test_layer_norm"]["message"]
    assert got["tests/test_ops.py::test_gemm_dtype"]["outcome"] == "SKIP"
    assert got["tests/test_ops.py::test_bad_setup"]["outcome"] == "ERROR"


def test_parse_pytest_summary_empty():
    assert parse_pytest_summary("no summary here") == {}


def test_table_aligns_by_test_id_and_flags_divergence():
    per_ws = {
        "ascend": {"t_a": {"outcome": "PASS"}, "t_b": {"outcome": "FAIL", "message": "x"}},
        "cuda": {"t_a": {"outcome": "PASS"}, "t_b": {"outcome": "PASS"}},
        "maca": {"t_a": {"outcome": "SKIP"}},
    }
    table = build_consistency_table(per_ws)
    rows = {r["test_id"]: r for r in table["rows"]}
    assert rows["t_a"]["consistent"] is False          # PASS/PASS/SKIP
    assert rows["t_b"]["consistent"] is False          # FAIL vs PASS
    assert rows["t_b"]["outcomes"]["ascend"]["message"] == "x"
    assert table["columns"] == ["ascend", "cuda", "maca"]
    assert table["ok"] is False
