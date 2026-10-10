"""Attestation: parse/parity logic and the in-container probe's shell contract."""

import json

from jms_bridge.attest import (
    MARKER,
    build_attest_script,
    check_attestation_parity,
    parse_attestation,
)

# --- script contract -----------------------------------------------------------------


def test_script_quotes_the_repo_path():
    script = build_attest_script("/public-flash/lvy ufeng/PyTorch-Plugin-FL")
    assert "cd '/public-flash/lvy ufeng/PyTorch-Plugin-FL'" in script


def test_script_never_exits():
    # This snippet is appended after the UT command, so an `exit` here would clobber
    # the pytest exit status the run is gated on (and short-circuit the report).
    script = build_attest_script("/work/Torch-FL")
    assert "exit" not in script


def test_script_reads_both_static_sources_and_a_bounded_import():
    script = build_attest_script("/work/Torch-FL")
    assert "torch_fl/_build_config.py" in script
    assert "torch_fl/backends.conf" in script
    assert "_C.cpython-*.so" in script
    assert "TORCH_DEVICE_BACKEND_AUTOLOAD=0" in script
    assert "timeout 30" in script  # the import must be bounded


def test_script_survives_a_quote_in_the_repo():
    script = build_attest_script("/work/it's-here")
    assert "'\"'\"'" in script


def test_script_is_valid_bash_and_emits_a_real_abi_tag(tmp_path):
    # The contract tests above only *read* the string; the broken `${a%%-*}` ABI
    # derivation passed them. Execute the script against a real checkout so the shell
    # logic is actually exercised.
    import subprocess

    repo = tmp_path / "fl"
    (repo / "torch_fl").mkdir(parents=True)
    (repo / "torch_fl" / "_build_config.py").write_text('ACCELERATOR = "ascend"\n')
    (repo / "torch_fl" / "backends.conf").write_text("# comment\nsoftmax = flaggems\n")
    # A real wheel's extension name; the tag must come out as `cpython-310`, not `cpython`.
    (repo / "torch_fl" / "_C.cpython-310-x86_64-linux-gnu.so").write_bytes(b"")

    out = subprocess.run(
        ["bash", "-c", build_attest_script(str(repo))],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    rec = parse_attestation(out.stdout)
    assert rec["accelerator"] == "ascend"
    assert rec["so_abi"] == "cpython-310"
    assert rec["backend_map"] == {"softmax": "flaggems"}
    assert rec["backends_read"] is True


def test_script_marks_an_empty_backends_conf_as_read(tmp_path):
    # backends.conf present but defining nothing: the map is *read* (empty), which is
    # a fact two chips can agree on — not the same as "could not be read".
    import subprocess

    repo = tmp_path / "fl"
    (repo / "torch_fl").mkdir(parents=True)
    (repo / "torch_fl" / "backends.conf").write_text("# only a comment\n")

    out = subprocess.run(
        ["bash", "-c", build_attest_script(str(repo))], capture_output=True, text=True
    )
    rec = parse_attestation(out.stdout)
    assert rec["backends_read"] is True
    assert rec["backend_map"] == {}
    assert rec["backend_map_hash"]  # a read-empty map still hashes


def test_script_records_cd_failure_without_exiting(tmp_path):
    import subprocess

    out = subprocess.run(
        ["bash", "-c", build_attest_script("/nonexistent/repo") + "echo AFTER"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0          # never fails the enclosing UT run
    assert "AFTER" in out.stdout        # and does not short-circuit it
    assert parse_attestation(out.stdout)["missing"] == ["cd_failed"]


# --- parsing -------------------------------------------------------------------------


_RUNTIME = json.dumps(
    {"torch": "2.10.0+cpu", "import_error": "ModuleNotFoundError: No module named 'torch_fl._C'"}
)

_FULL = (
    "collected 3 items\n"
    "PASSED tests/t.py::test_x\n"
    f'{MARKER}build_config=ACCELERATOR = "ascend"\n'
    f"{MARKER}backends_read=1\n"
    f"{MARKER}backend=softmax = flaggems\n"
    f"{MARKER}backend=gemm = flagos_python\n"
    f"{MARKER}so_abi=cpython-310\n"
    f"{MARKER}python=/usr/local/python3.11.15/bin/python3\n"
    f"{MARKER}interp_abi=cpython-311\n"
    f"{MARKER}runtime={_RUNTIME}\n"
    "1 passed\n"
)


def test_parse_full_record():
    rec = parse_attestation(_FULL)
    assert rec["accelerator"] == "ascend"
    assert rec["backend_map"] == {"softmax": "flaggems", "gemm": "flagos_python"}
    assert rec["backend_map_hash"]
    assert rec["so_abi"] == "cpython-310"
    assert rec["interp_abi"] == "cpython-311"
    assert rec["abi_match"] is False
    assert rec["python"] == "/usr/local/python3.11.15/bin/python3"
    assert rec["torch"] == "2.10.0+cpu"
    assert "torch_fl._C" in rec["import_error"]
    assert rec["missing"] == []


def test_parse_ignores_unmarked_lines():
    rec = parse_attestation("PASSED tests/t.py::test_x\nplain noise\n")
    assert rec["accelerator"] is None
    assert rec["backend_map"] == {}

def test_backend_map_hash_is_order_independent_and_stable():
    r = f"{MARKER}backends_read=1\n"
    a = parse_attestation(r + f"{MARKER}backend=x = 1\n{MARKER}backend=y = 2\n")
    b = parse_attestation(r + f"{MARKER}backend=y = 2\n{MARKER}backend=x = 1\n")
    assert a["backend_map_hash"] == b["backend_map_hash"]
    c = parse_attestation(r + f"{MARKER}backend=x = 1\n{MARKER}backend=y = 3\n")
    assert c["backend_map_hash"] != a["backend_map_hash"]


def test_hash_none_when_map_not_read():
    # No backends_read marker -> the map is unreadable, so it has no hash at all.
    rec = parse_attestation("")
    assert rec["backend_map_hash"] is None
    assert rec["backends_read"] is False


def test_abi_match_true_only_when_both_known():
    assert parse_attestation(f"{MARKER}so_abi=cpython-310\n"
                             f"{MARKER}interp_abi=cpython-310\n")["abi_match"] is True
    # A missing interpreter ABI means we cannot tell -> None, not a false mismatch.
    assert parse_attestation(f"{MARKER}so_abi=cpython-310\n")["abi_match"] is None


def test_runtime_accelerator_fills_in_when_static_read_absent():
    rec = parse_attestation(
        f"{MARKER}build_config_absent=1\n"
        f'{MARKER}runtime={{"runtime_accelerator": "metax"}}\n'
    )
    assert rec["accelerator"] == "metax"
    assert "build_config" in rec["missing"]


def test_error_line_is_recorded_not_swallowed():
    rec = parse_attestation(f"{MARKER}error=cd_failed\n")
    assert rec["missing"] == ["cd_failed"]


def test_malformed_runtime_json_does_not_raise():
    rec = parse_attestation(f"{MARKER}runtime=not json\n")
    assert rec["torch"] is None
    assert rec["plugin_version"] is None


def test_backend_line_without_separator_is_dropped():
    # A comment-ish or malformed line must not become a phantom op->backend entry.
    rec = parse_attestation(f"{MARKER}backend=just_a_word\n")
    assert rec["backend_map"] == {}


# --- parity --------------------------------------------------------------------------

def _rec(backend_hash, version=None, **kw):
    return {
        "backend_map_hash": backend_hash,
        "backends_read": backend_hash is not None,
        "plugin_version": version,
        **kw,
    }


def test_parity_empty_is_never_consistent():
    r = check_attestation_parity({})
    assert r["consistent"] is False
    assert r["diverged"] == ["no workspaces"]


def test_parity_same_map_is_consistent():
    r = check_attestation_parity({"a": _rec("h1"), "b": _rec("h1")})
    assert r["consistent"] is True
    assert r["diverged"] == []
    assert r["unverified"] == []


def test_parity_different_map_diverges():
    r = check_attestation_parity({"a": _rec("h1"), "b": _rec("h2")})
    assert r["consistent"] is False
    assert "backend_map_hash" in r["diverged"]


def test_parity_unreadable_map_is_unverified_not_agreement():
    # A chip whose map could not be read must never be coerced into agreement.
    r = check_attestation_parity({"a": _rec("h1"), "b": _rec(None)})
    assert r["consistent"] is False
    assert r["unverified"] == ["b"]


def test_parity_empty_but_read_map_is_consistent_not_unverified():
    # Both chips read backends.conf and it defines nothing: that agrees, it is not a
    # chip we failed to read.
    empty = parse_attestation(f"{MARKER}backends_read=1\n")
    r = check_attestation_parity({"a": empty, "b": empty})
    assert r["unverified"] == []
    assert r["consistent"] is True


def test_parity_different_plugin_version_diverges():
    r = check_attestation_parity({"a": _rec("h1", "1.0"), "b": _rec("h1", "2.0")})
    assert r["consistent"] is False
    assert "plugin_version" in r["diverged"]


def test_parity_unknown_version_on_one_chip_is_not_a_divergence():
    # Versions legitimately unreadable on one chip (the import may fail) — surface but
    # do not gate, unlike the backend map.
    r = check_attestation_parity({"a": _rec("h1", "1.0"), "b": _rec("h1", None)})
    assert r["consistent"] is True
    assert r["diverged"] == []