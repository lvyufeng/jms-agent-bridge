"""The acceptance script's tagger: what it writes into the inventory, and what it says out loud.

`scripts/verify-onboarding.sh` turns `jms_list_assets` into `hosts.yaml` and then probes the
result. A platform tagged by guess is the one place that chain can go quietly wrong: a
domestic accelerator tagged `cuda` probes `nvidia-smi`, sees nothing, and reports a host with
no accelerator — which reads as a fine host. So the guess must always name a real profile,
and a fallback must be visible on stderr while the parseable table stays on stdout.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "verify_support.py"


def _load_module():
    # By path rather than by import: `scripts/` is not a package and not on sys.path, and
    # the acceptance script is run as a file, not imported.
    spec = importlib.util.spec_from_file_location("verify_support", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_support"] = mod
    spec.loader.exec_module(mod)
    return mod


vs = _load_module()


@pytest.fixture
def assets_file(tmp_path):
    def write(assets):
        p = tmp_path / "assets.json"
        p.write_text(json.dumps({"assets": assets}))
        return str(p)

    return write


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _mkhosts_args(assets, out, **kw):
    base = dict(
        assets=assets,
        out=str(out),
        login="me",
        account="root",
        pick="",
        hosts="",
    )
    base.update(kw)
    return Args(**base)


def test_a_guess_is_written_into_the_inventory(assets_file, tmp_path):
    # The tag matters more than the warning: the inventory is what gets probed.
    src = assets_file([{"id": "a1", "name": "node-1", "address": "10.0.0.1", "platform": "NVIDIA"}])
    out = tmp_path / "hosts.yaml"
    assert vs.cmd_mkhosts(_mkhosts_args(src, out)) == 0
    assert "platform: cuda" in out.read_text()


def test_a_fallback_is_warned_about_on_stderr_while_the_rows_stay_on_stdout(
    assets_file, tmp_path, capsys
):
    # `verify-onboarding.sh` merges stderr into the captured output, so a warning printed to
    # stdout would land in the middle of a table the caller parses as tab-separated rows.
    src = assets_file(
        [
            {"id": "a1", "name": "node-1", "address": "10.0.0.1", "platform": "NVIDIA"},
            {"id": "a2", "name": "mystery-board", "address": "10.0.0.2", "platform": ""},
        ]
    )
    out = tmp_path / "hosts.yaml"
    assert vs.cmd_mkhosts(_mkhosts_args(src, out)) == 0

    cap = capsys.readouterr()
    assert "warning: mystery-board" in cap.err
    assert "tag it by hand" in cap.err
    assert "warning" not in cap.out
    # It still gets tagged, and with a profile that exists — a warning is not a refusal.
    assert "platform: generic" in out.read_text()
    assert [line.split("\t")[0] for line in cap.out.splitlines()] == ["node-1", "mystery-board"]


def test_a_caller_supplied_platform_is_not_warned_about(assets_file, tmp_path, capsys):
    # `--hosts name=ref:platform` is a human deciding; there is nothing left to guess.
    src = assets_file([{"id": "a1", "name": "node-1", "address": "10.0.0.1", "platform": ""}])
    out = tmp_path / "hosts.yaml"
    args = _mkhosts_args(src, out, hosts="node-1=a1:tsingmicro")
    assert vs.cmd_mkhosts(args) == 0

    cap = capsys.readouterr()
    assert cap.err == ""
    assert "platform: tsingmicro" in out.read_text()


def test_a_vendor_name_rescues_an_unhelpful_declared_field(assets_file, tmp_path, capsys):
    # JumpServer says "Linux" for anything Unix; the asset's own name is the next best
    # evidence, and it is enough to skip the warning.
    src = assets_file(
        [{"id": "a1", "name": "fl-moore-1", "address": "10.0.0.1", "platform": "Linux"}]
    )
    out = tmp_path / "hosts.yaml"
    assert vs.cmd_mkhosts(_mkhosts_args(src, out)) == 0
    assert "platform: moore-threads" in out.read_text()
    assert "warning" not in capsys.readouterr().err


def test_the_assets_listing_warns_too(assets_file, capsys):
    # The tag is shown before an inventory is written, so the moment to notice a bad one is
    # while reading the asset list.
    src = assets_file(
        [{"id": "a2", "name": "mystery-board", "address": "10.0.0.2", "platform": ""}]
    )
    assert vs.cmd_assets(Args(file=src)) == 0
    cap = capsys.readouterr()
    assert "mystery-board" in cap.out and "generic" in cap.out
    assert "warning: mystery-board" in cap.err
