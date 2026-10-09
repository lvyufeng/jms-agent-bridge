from jms_bridge.discover import (
    build_discovery_script,
    parse_discovery,
    render_workspaces_yaml,
)

SAMPLE = """\
JMSD container=lvyufeng-torch-fl
JMSD running=1
JMSD inner=1
JMSD python=/usr/local/python3.11.15/bin/python3|3.11.15|8.4.2
JMSD python=/usr/bin/python3|3.10.12|
JMSD repo=/public-flash/lvyufeng/PyTorch-Plugin-FL
JMSD remote=git@github.com:lvyufeng/PyTorch-Plugin-FL.git
JMSD branch=flagos-plugin
JMSD sha=abc1234def5678
JMSD reachable=1
Goodbye, see you soon!
"""


def test_build_script_quotes_container_layer():
    script = build_discovery_script("lvyufeng-torch-fl")
    assert "docker exec" in script
    assert "docker inspect" in script
    # The container script is single-quoted for the host shell, so the host never
    # expands the container's own $vars (the bug that killed the hand-rolled version).
    assert "'" in script


def test_parse_discovery_full_report():
    rep = parse_discovery(SAMPLE)
    assert rep["container_present"] is True
    assert rep["running"] is True
    assert rep["inner_ok"] is True
    assert rep["python"][0] == {
        "path": "/usr/local/python3.11.15/bin/python3", "version": "3.11.15", "pytest": "8.4.2"
    }
    assert rep["python"][1]["pytest"] is None
    repo = rep["repos"][0]
    assert repo["path"] == "/public-flash/lvyufeng/PyTorch-Plugin-FL"
    assert repo["branch"] == "flagos-plugin"
    assert repo["reachable"] is True
    # The trailing logout banner carries no JMSD marker and must be ignored.
    assert len(rep["repos"]) == 1


def test_parse_discovery_absent_container():
    rep = parse_discovery("JMSD container_absent=1\n")
    assert rep["container_present"] is False
    assert rep["repos"] == []


def test_parse_discovery_orphan_fields_dropped():
    # remote/branch with no preceding repo= must not attach to anything.
    rep = parse_discovery("JMSD remote=git@x\nJMSD branch=main\n")
    assert rep["repos"] == []


def test_render_emits_workspace_and_flags_unreachable_origin():
    reports = [{
        "host": "ascend-910c-108",
        "exit_code": 0,
        "report": parse_discovery(SAMPLE),
    }]
    out = render_workspaces_yaml(reports, container="lvyufeng-torch-fl")
    assert out["skipped"] == []
    ws = out["workspaces"]["fl-ascend-910c-108"]
    assert ws["host"] == "ascend-910c-108"
    assert ws["container"] == "lvyufeng-torch-fl"
    assert ws["repo"] == "/public-flash/lvyufeng/PyTorch-Plugin-FL"
    assert ws["source"] == "git"
    assert "workspaces:" in out["yaml"]
    assert "fl-ascend-910c-108" in out["yaml"]


def test_render_warns_when_origin_unreachable():
    offline = SAMPLE.replace("JMSD reachable=1", "JMSD reachable=0")
    reports = [{"host": "h1", "exit_code": 0, "report": parse_discovery(offline)}]
    out = render_workspaces_yaml(reports, container="lvyufeng-torch-fl")
    assert out["workspaces"]["fl-h1"]["source"] == "git"  # guess stands until confirmed
    assert any("unreachable" in w for w in out["warnings"])


def test_render_skips_host_without_container():
    reports = [{
        "host": "h1",
        "exit_code": 0,
        "report": parse_discovery("JMSD container_absent=1\n"),
    }]
    out = render_workspaces_yaml(reports, container="lvyufeng-torch-fl")
    assert out["workspaces"] == {}
    assert out["skipped"] == ["h1"]
    assert any("not found" in w for w in out["warnings"])


def test_render_skips_container_without_matching_checkout():
    reports = [{
        "host": "h1",
        "exit_code": 0,
        "report": parse_discovery("JMSD container=c\nJMSD running=1\nJMSD inner=1\n"),
    }]
    out = render_workspaces_yaml(reports, container="c")
    assert out["skipped"] == ["h1"]
    assert any("no matching checkout" in w for w in out["warnings"])


def test_render_multiple_checkouts_get_two_workspaces():
    two = SAMPLE + (
        "JMSD repo=/work/PyTorch-Plugin-FL\n"
        "JMSD remote=git@x\nJMSD branch=main\nJMSD sha=deadbeef\nJMSD reachable=1\n"
    )
    reports = [{"host": "h1", "exit_code": 0, "report": parse_discovery(two)}]
    out = render_workspaces_yaml(reports, container="lvyufeng-torch-fl")
    assert set(out["workspaces"]) == {"fl-h1", "fl-h1-2"}
    assert any("2 checkouts" in w for w in out["warnings"])


def test_render_warns_when_no_pytest():
    nopytest = SAMPLE.replace("|8.4.2", "|").replace(
        "JMSD python=/usr/bin/python3|3.10.12|", "JMSD python=/usr/bin/python3|3.10.12|"
    )
    reports = [{"host": "h1", "exit_code": 0, "report": parse_discovery(nopytest)}]
    out = render_workspaces_yaml(reports, container="lvyufeng-torch-fl")
    assert any("pytest" in w for w in out["warnings"])


def test_render_carries_host_tags():
    reports = [{"host": "h1", "exit_code": 0, "report": parse_discovery(SAMPLE)}]
    out = render_workspaces_yaml(
        reports, container="lvyufeng-torch-fl", host_tags={"h1": ["ascend", "gpu"]}
    )
    assert out["workspaces"]["fl-h1"]["tags"] == ["ascend", "gpu"]