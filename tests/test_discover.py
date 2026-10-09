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
JMSD reachable=ok
Goodbye, see you soon!
"""


def test_build_script_single_quotes_container_layer():
    script = build_discovery_script("lvyufeng-torch-fl")
    # The container script is embedded as a single-quoted argument to `docker exec
    # ... bash -lc`, so the host shell never expands the container's own $vars (the
    # bug that killed the hand-rolled version). Its own single quotes must therefore
    # be escaped as '"'"' — if the quoting were dropped, the inner text would appear
    # raw (plain ') and the host would expand it.
    assert "bash -lc '" in script
    assert "'\"'\"'" in script
    # `find -print0` + `read -r -d` keeps a path with spaces from word-splitting.
    assert "-print0" in script
    assert "read -r -d" in script


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
    assert repo["reachable"] == "ok"
    # The trailing logout banner carries no JMSD marker and must be ignored.
    assert len(rep["repos"]) == 1


def test_parse_discovery_reachable_has_three_outcomes():
    assert parse_discovery("JMSD repo=/a\nJMSD reachable=denied\n")["repos"][0]["reachable"] \
        == "denied"
    assert parse_discovery("JMSD repo=/a\nJMSD reachable=timeout\n")["repos"][0]["reachable"] \
        == "timeout"


def test_parse_discovery_inner_exit():
    rep = parse_discovery("JMSD container=c\nJMSD running=1\nJMSD inner_exit=126\n")
    assert rep["inner_ok"] is False
    assert rep["inner_exit"] == 126


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
    offline = SAMPLE.replace("JMSD reachable=ok", "JMSD reachable=denied")
    reports = [{"host": "h1", "exit_code": 0, "report": parse_discovery(offline)}]
    out = render_workspaces_yaml(reports, container="lvyufeng-torch-fl")
    assert out["workspaces"]["fl-h1"]["source"] == "git"  # guess stands until confirmed
    assert any("did not answer" in w for w in out["warnings"])


def test_render_warns_when_reachability_timed_out():
    slow = SAMPLE.replace("JMSD reachable=ok", "JMSD reachable=timeout")
    reports = [{"host": "h1", "exit_code": 0, "report": parse_discovery(slow)}]
    out = render_workspaces_yaml(reports, container="lvyufeng-torch-fl")
    assert any("probe timed out" in w for w in out["warnings"])
    # a reachability *probe* timeout still yields a workspace (the checkout was found)
    assert set(out["workspaces"]) == {"fl-h1"}


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
    assert out["ok"] is False


def test_render_reports_docker_absent_distinctly():
    reports = [{"host": "h1", "exit_code": 0,
                "report": parse_discovery("JMSD error=docker_absent\n")}]
    out = render_workspaces_yaml(reports, container="c")
    assert out["skipped"] == ["h1"]
    # must NOT be misreported as a missing container
    assert any("no docker CLI" in w for w in out["warnings"])
    assert not any("not found" in w for w in out["warnings"])


def test_render_reports_exec_failure_not_missing_checkout():
    # container present, but `docker exec` failed (inner_exit) -> must be an inspect
    # failure, NOT the "no matching checkout" bucket.
    rep = parse_discovery("JMSD container=c\nJMSD running=1\nJMSD inner_exit=126\n")
    out = render_workspaces_yaml([{"host": "h1", "exit_code": 0, "report": rep}], container="c")
    assert out["skipped"] == ["h1"]
    assert any("could not inspect" in w and "126" in w for w in out["warnings"])
    assert not any("no matching checkout" in w for w in out["warnings"])


def test_render_reports_transport_timeout_distinctly():
    # exec_on timed out: empty stdout, timed_out flag set -> must say "timed out",
    # never "container not found".
    out = render_workspaces_yaml(
        [{"host": "h1", "exit_code": 124, "timed_out": True, "timeout": 120.0,
          "report": parse_discovery("")}],
        container="c",
    )
    assert out["skipped"] == ["h1"]
    assert any("timed out" in w for w in out["warnings"])
    assert not any("not found" in w for w in out["warnings"])


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
        "JMSD remote=git@x\nJMSD branch=main\nJMSD sha=deadbeef\nJMSD reachable=ok\n"
    )
    reports = [{"host": "h1", "exit_code": 0, "report": parse_discovery(two)}]
    out = render_workspaces_yaml(reports, container="lvyufeng-torch-fl")
    assert set(out["workspaces"]) == {"fl-h1", "fl-h1-2"}
    assert any("2 checkouts" in w for w in out["warnings"])


def test_render_workspace_key_collision_does_not_drop_a_workspace():
    # host `node` with two checkouts -> fl-node, fl-node-2; a DIFFERENT host named
    # `node-2` also wants fl-node-2. Both must survive.
    two = SAMPLE + (
        "JMSD repo=/work/PyTorch-Plugin-FL\n"
        "JMSD remote=git@x\nJMSD branch=main\nJMSD sha=deadbeef\nJMSD reachable=ok\n"
    )
    reports = [
        {"host": "node", "exit_code": 0, "report": parse_discovery(two)},
        {"host": "node-2", "exit_code": 0, "report": parse_discovery(SAMPLE)},
    ]
    out = render_workspaces_yaml(reports, container="lvyufeng-torch-fl")
    assert len(out["workspaces"]) == 3
    # the second checkout of `node` is preserved, not overwritten by `node-2`
    assert {w["host"] for w in out["workspaces"].values()} == {"node", "node-2"}
    assert sum(1 for w in out["workspaces"].values() if w["host"] == "node") == 2


def test_render_threads_proxy_mount_local():
    reports = [{"host": "h1", "exit_code": 0, "report": parse_discovery(SAMPLE)}]
    pg = render_workspaces_yaml(
        reports, container="c", source="proxy-git", proxy="http://prx:80"
    )["workspaces"]["fl-h1"]
    assert pg["proxy"] == "http://prx:80"
    pu = render_workspaces_yaml(
        reports, container="c", source="push", mount="/srv/fl", local="~/fl"
    )["workspaces"]["fl-h1"]
    assert pu["mount"] == "/srv/fl" and pu["local"] == "~/fl"


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


def test_render_ok_true_only_when_no_host_skipped():
    clean = [{"host": "h1", "exit_code": 0, "report": parse_discovery(SAMPLE)}]
    assert render_workspaces_yaml(clean, container="c")["ok"] is True
    mixed = clean + [{"host": "h2", "exit_code": 0,
                      "report": parse_discovery("JMSD container_absent=1\n")}]
    assert render_workspaces_yaml(mixed, container="c")["ok"] is False


def test_emitted_yaml_loads_for_every_source(tmp_path, monkeypatch):
    """The block is advertised paste-ready — it must survive Inventory.load for every
    source, including the proxy-git / push ones whose required fields are threaded in."""
    import yaml as _yaml

    from jms_bridge.inventory import Inventory

    reports = [{"host": "h1", "exit_code": 0, "report": parse_discovery(SAMPLE)}]
    if not any(w["host"] == "h1" for w in reports):  # pragma: no cover - guard
        raise AssertionError("sample must produce a workspace")

    hosts_block = "hosts:\n  h1:\n    login: u\n    asset_id: A\n"
    for source, extra in (
        ("git", {}),
        ("proxy-git", {"proxy": "http://prx:80"}),
        ("push", {"mount": "/srv/fl", "local": "~/fl"}),
    ):
        rendered = render_workspaces_yaml(reports, container="c", source=source, **extra)
        doc = _yaml.safe_load(rendered["yaml"])
        text = hosts_block + _yaml.safe_dump(doc)
        path = tmp_path / f"{source}.yaml"
        path.write_text(text)
        inv = Inventory.load(path)
        assert "fl-h1" in inv.workspaces, f"{source} block did not load"