import pytest

from jms_bridge.workspace import Workspace, WorkspaceError


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