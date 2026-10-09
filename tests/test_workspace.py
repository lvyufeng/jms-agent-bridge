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


def test_parse_pytest_summary_parametrized_id_with_dash():
    got = parse_pytest_summary("FAILED tests/test_ops.py::test_x[lo - hi] - AssertionError: bad")
    assert "tests/test_ops.py::test_x[lo - hi]" in got
    entry = got["tests/test_ops.py::test_x[lo - hi]"]
    assert entry["outcome"] == "FAIL"
    assert entry["message"].startswith("AssertionError")


def test_parse_pytest_summary_duplicate_line_keeps_first():
    got = parse_pytest_summary("PASSED tests/a.py::t\nFAILED tests/a.py::t - boom")
    assert list(got) == ["tests/a.py::t"]
    assert got["tests/a.py::t"]["outcome"] == "PASS"


def test_workspace_proxy_git_requires_proxy():
    with pytest.raises(WorkspaceError):
        Workspace(name="w", host="a", container="c", repo="/r", source="proxy-git")


def test_workspace_push_requires_mount_and_local():
    with pytest.raises(WorkspaceError):
        Workspace(name="w", host="a", container="c", repo="/r", source="push")


def test_workspace_bad_container_raises():
    with pytest.raises(WorkspaceError):
        Workspace(name="w", host="a", container="foo; id", repo="/r")


def test_workspace_empty_repo_raises():
    with pytest.raises(WorkspaceError):
        Workspace(name="w", host="a", container="c", repo="")


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


def test_delivery_commands_per_source():
    g = Workspace(name="g", host="a", container="c", repo="/r", source="git")
    assert "git fetch origin" in g.delivery_cmd("feat/x")
    assert "git checkout 'feat/x'" in g.delivery_cmd("feat/x")

    p = Workspace(name="p", host="a", container="c", repo="/r",
                  source="proxy-git", proxy="http://prx:80")
    assert "http_proxy='http://prx:80'" in p.delivery_cmd("feat/x")

    u = Workspace(name="u", host="a", container="c", repo="/r",
                  source="push", mount="/srv/fl", local="/Users/me/Torch-FL")
    assert u.delivery_cmd("feat/x") is None


def test_table_flags_a_missing_test_even_when_others_agree():
    # Both workspaces agree on t_shared; A also reported t_x but B did not, so the
    # t_x row must be inconsistent purely because a workspace never ran it.
    per_ws = {
        "A": {"t_shared": {"outcome": "PASS"}, "t_x": {"outcome": "PASS"}},
        "B": {"t_shared": {"outcome": "PASS"}},
    }
    table = build_consistency_table(per_ws)
    rows = {r["test_id"]: r for r in table["rows"]}
    assert rows["t_shared"]["consistent"] is True
    assert rows["t_x"]["consistent"] is False
    assert rows["t_x"]["outcomes"]["B"] is None
    assert table["ok"] is False


def test_table_single_workspace_is_consistent():
    table = build_consistency_table({"A": {"t": {"outcome": "FAIL"}}})
    assert table["ok"] is True
    assert table["rows"][0]["consistent"] is True
