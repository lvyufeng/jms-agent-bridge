# v0.3 — Fleet Breadth + Run Fidelity

> **Status:** design (2026-10-10). Supersedes nothing; extends the flagos dev loop.

## Why

The flagos dev loop exists to make one sentence true: **"write once, verify on every
chip."** Two things make it false today.

**Breadth — the fleet is not in the table.** `jms_dev_discover` finds the container,
checkout and origin-reachability on every host and emits a paste-ready `workspaces:`
block, but the inventory holds **one** host. A one-workspace consistency table is a
one-column table: nothing is "verified on every chip" when every chip but one is
invisible to `_resolve`, `plan_islands`, `preflight` and `jms_dev_run`.

**Fidelity — "verify" currently means "the same git sha exited the same way".**
`build_consistency_table` (`workspace.py`) compares *outcomes only*; `jms_dev_run` gates
`ok` on delivery, per-chip exit code and a shared `revisions` sha (`server.py`). Nothing
observes **what actually ran**:

- Two chips at the same sha but **different builds** (a wheel built for `ascend` vs one
  built for `dcu`) both report `PASS` and the table reads `ok=true` — a green row that
  proves only that two different things both "worked".
- The plugin dispatches per op (`torch_fl/backends.conf`: `flaggems` | `flagos_python` |
  `cuda`). A test that routes `_softmax` to a different backend on chip A than chip B
  **PASSes on both**, and the table asserts uniform behaviour it never observed. The
  design doc's acceptance #1 is *path uniformity* (`完全一样的写法`); it is unverified.
- A test the plugin **cannot route** on a chip is indistinguishable from one that ran.

## Scope

v0.3 delivers **Breadth** and **Fidelity**. It does **not** build the chip-fingerprint
layer, island re-partitioning, or cross-island federation — those serve the *training*
loop the design doc defers, and they are gated on real per-chip samples that Breadth is
what produces. (A useful slice of fingerprinting falls out for free — see below.)

## Part A — Breadth: onboard the fleet, and keep the inventory live

1. **`jms_inventory_reload`** (new tool). Every tool reads the inventory through `_inv()`,
   which caches it in `_state` on first use (`server.py`). The advertised loop is *emit
   YAML → paste into `hosts.yaml` → `jms_dev_run`, but the running MCP server keeps the
   stale inventory, so the paste step silently requires a restart. The tool drops the
   cached `Inventory` (and any derived state) so the next call re-reads from disk.
   Returns the new `{hosts, workspaces, groups}` summary.

2. **Onboard the 14 hosts.** `jms_dev_discover` already emits the `workspaces:` block;
   onboarding is *drive it across the fleet and paste the result*. Its output carries the
   raw per-host findings, so a host whose container is absent or whose origin is
   unreachable is a warning, not a silent omission. Runtime step — needs control-plane
   access for the asset list (or the asset names already recovered from the earlier scan).

3. **Asset-name addressing.** `Host.asset_id` accepts an asset **name** as well as a UUID
   (`login_string` docstring). Record the asset names discovered by the scan as the seed so
   onboarding does not require the REST asset list to be reachable first.

## Part B — Fidelity: attest what ran

A new module `attest.py` (pure, testable like `discover.py`) that runs **one** in-container
probe per workspace, folded into the round-trip `jms_dev_run` already makes, and a table
extension that gates on it.

### The probe — two layers

The container's interpreter frequently **cannot import the plugin** (measured on
`ascend-910c-108`: `_C` is built for cpython-310/314, the interpreter is 3.11 →
`ModuleNotFoundError: torch_fl._C`; `torch` is `2.10.0+cpu`). So the probe must not depend
on import, and "cannot import" is itself a first-class result.

- **Static (no import; always available).** Read, from the checkout:
  - `torch_fl/_build_config.py` → `ACCELERATOR` — the **build identity** the maintainers
    generate specifically to tell wheels apart ("the generated file is what makes a DCU
    wheel distinguishable").
  - `torch_fl/backends.conf` → `op -> backend` — the **code-path map**.
  - the `_C` extension's **ABI tag** (`cpython-3NN`) and the **default interpreter's**
    `sys.implementation.cache_tag` → an **ABI match/mismatch** bit (explains an import
    failure instead of leaving it mysterious).
  - the **git sha** of the checkout (already delivered; `_deliver` returns it).
  - `torch.__version__` if importable (best-effort).
- **Runtime (best-effort).** `python -c "import torch_fl, json; ..."` yielding
  `{version, accelerator, backend, ops}`; a failure is recorded as `import_error`, not
  swallowed.

Output: a per-workspace **attestation record**. The probe line is written to a sentinel so
it can be lifted out of the UT stdout without a second SSH round-trip.

### The table extension

`jms_dev_run` gains, alongside `rows`/`revisions`:
- **`attestation`** — per workspace `{accelerator, backend_map, abi, abi_match, torch,
  plugin, import_error}`.
- **`attestation_consistent`** — do the fields that *must* agree actually agree? The
  **build accelerator** and the **registered op→backend map** are gated; the interpreter
  ABI and installed torch are surfaced but **not** gated (chips legitimately differ there).
  A divergence is the interesting signal: "the same sha is running different code paths".
- `ok` additionally requires `attestation_consistent`. Like `revisions`, a workspace whose
  attestation could not be read is **unverified**, never silently dropped from the
  comparison.

### Freebie: a slice of fingerprinting

The static layer's `ACCELERATOR` is exactly the per-host adapter identity the deferred
fingerprint layer wanted (`ascend` / `metax` / `dcu` / …). v0.3 exposes it per host without
adding a single `Platform` profile or touching `plan_islands` — the identity is now
*recorded*, ready for the training loop to consume later.

## Non-goals (explicit)

- No new `Platform` profiles, no `plan_islands` / `dist_backend` change. The dev loop does
  not read `platform_def`; islands serve the deferred training path.
- No cross-island federation, no numerical-equivalence check (still deferred).
- No build/rebuild stage: the probe **reports** the build identity, it does not **make** it.
  Building per-chip wheels needs per-chip recipes that are still unknown.

## Open questions

- Does the backend map (`backends.conf`) ever *legitimately* differ across chips — i.e. is
  `attestation_consistent` too strict as an `ok` gate? Working assumption: the map is
  build-independent (a property of the source at `ref`, so it should be identical at a
  shared sha); if a real chip needs a different map, the gate must become a warning and a
  separate override.
- Concurrency: two overlapping `jms_dev_run`s, or a UT run on a host a training job already
  holds, are undetected. Out of scope for v0.3; noted.