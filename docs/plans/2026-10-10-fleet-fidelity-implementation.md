# v0.3 — Fleet Breadth + Run Fidelity: Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: use superpowers:executing-plans task-by-task.
> Design: [`2026-10-10-fleet-fidelity-design.md`](2026-10-10-fleet-fidelity-design.md).

**Goal:** make "write once, verify on every chip" true — onboard the fleet, and make a
cross-chip PASS mean the same *code path*, not merely the same git sha.

**Architecture:** Part B is one pure module (`attest.py`) whose in-container probe rides
the SSH round trip `jms_dev_run` already makes, plus a parity check gating `ok`. Part A is
a cache-drop tool (`jms_inventory_reload`) unblocking the paste-the-YAML loop.

**Tech Stack:** Python 3.10+, pytest, `mcp>=2.0`. Two-level shell quoting (host wrapper +
`docker exec … bash -lc`).

---

## Task 1: `attest.py` — script + parse + parity (TDD)

**Files:** Create `src/jms_bridge/attest.py`, `tests/test_attest.py`.

Steps: write the failing tests (script contract, full parse, 3-valued ABI, hash order-
independence, parity cases) → run (fail) → implement → run (pass) → commit.

**Key invariants (each is a real bug avoided):**
- The script must **never** `exit` — it is appended after the UT command, so an `exit`
  clobbers pytest's status. Verified by `test_script_never_exits`.
- The runtime import is bounded with `timeout 30` — an unbounded import can hang on a
  wedged driver and take the whole UT run down.
- `abi_match` is `None` (not `False`) when either side is unknown — a missing interpreter
  ABI is "cannot tell", not "mismatch".
- `backend_map_hash` hashes a *sorted* blob — the map's order is not meaningful.
- `check_attestation_parity`: an unreadable map is `unverified`, never coerced into
  agreement; `plugin_version` is gated only among chips that reported one.

## Task 2: compose the probe into `jms_dev_run`

**Files:** Modify `src/jms_bridge/server.py` (`jms_dev_run.one`).

Wrap the UT command so the exit code survives the appended probe:

```python
inner = (
    f"cd {shell_quote(ws.repo)} && {{ ({ut_cmd or ws.ut_cmd}); rc=$?; "
    + build_attest_script(ws.repo)
    + "exit $rc; }"
)
```

Without capturing `rc`, `res.exit_code` would always be the probe's, and `failed`
(`not res.ok`) could never fire — a silent regression of the nonzero-exit gate. Then parse
each chip's record from `res.stdout`, run `check_attestation_parity`, and add
`attestation` / `attestation_consistent` / `attestation_diverged` / `attestation_unverified`
to the table; `ok` additionally requires `attestation_consistent`.

**Tests:** `test_dev_run_same_sha_but_different_code_path_is_not_ok` (the core signal),
`test_dev_run_unreadable_attestation_is_unverified_not_ok`, and the existing suite updated
with an attested `_UT_OK` fixture.

## Task 3: `jms_inventory_reload`

**Files:** Modify `src/jms_bridge/server.py`; test in `tests/test_server.py`.

`_state.pop("inv")` then re-load, returning `{hosts, workspaces, groups}`; a load error is
surfaced, not raised.

## Task 4: docs + memory

README tool count 18→19, the `jms_inventory_reload` row, and the attestation paragraph in
the dev-loop section. Memory updated.

## Deferred (design non-goals)

Onboarding the 14 hosts is a *runtime* step needing control-plane access for the asset
list; `jms_dev_discover` already emits the block. No new `Platform` profiles, no island
change, no build stage.