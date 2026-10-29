# Changelog

## Unreleased

- Add opt-in compact MCP serving with four tools, durable bounded-output reads,
  REST inventory refresh, native SSH connection/session ownership, remote Docker,
  and rsync over the pooled connection. Existing serving behavior remains the default.

### Added

- **Repeatable local ten-worker concurrency evidence.** A disposable loopback SSH
  benchmark measures paced cold/warm connections, inspection reserve tradeoffs,
  burst queues, cancellation, not-started queue expiry, slow refresh and disconnect
  cleanup. It reports independent local child peaks and timing scope; no production
  defaults, real cluster credentials or agent-orchestration behavior are changed.

- **`jms_asset_detail` — one asset's accounts, and a paste-ready `hosts.yaml` entry.**
  `jms_list_assets` deliberately omits accounts (it is one REST request, and enumerating
  `#account` per asset would be one request each), but the account is exactly the field a
  new inventory is missing — and the obvious endpoint for it answers **403** to a normal
  account. The asset detail endpoint carries `permed_accounts` in the same single request,
  so this tool returns the accounts, the asset metadata, and the YAML block to paste under
  `hosts:`. `platform` is deliberately left a placeholder: JumpServer's platform name does
  not say which accelerator to probe, and a wrong guess silently reports a healthy host
  with no accelerator.
- **Long tools now report MCP progress.** `jms_dev_run` (per chip) and `jms_dev_discover`
  (per host) previously produced nothing until they finished, so a fleet-wide run was
  indistinguishable from a hang. Progress is client-opt-in (MCP delivers it only to a
  client that registered a progress callback) and best-effort by design — a client without
  support, or a direct call outside a live request, must not fail the work, so the
  notification is swallowed rather than propagated. Verified end to end over a real stdio
  session: three reports for a three-step tool.
- **Every tool now carries MCP annotations, and the server publishes instructions.** All
  24 tools previously shipped `annotations: null`, so a client had no way to tell an
  inspection call from one that spawns or kills work. The eleven read-only tools
  (`jms_list_assets`, `jms_audit_query`, `jms_probe`, `jms_fleet_summary`,
  `jms_job_status`, `jms_job_logs`, `jms_job_wait`, `jms_job_list`, `jms_dev_discover`,
  `jms_cluster_preflight`, `jms_cluster_plan`) now set `readOnlyHint`, which lets a client
  auto-approve them; the tools that mutate remote state set `destructiveHint`. A test pins
  the read-only set in both directions, because "this tool only reads" is a safety claim.
  A top-level `instructions` string (plus a `title`) now states the selector grammar and
  which tool to reach for; most clients inject it into the agent's context.
- **Five tools declare a structured `outputSchema`.** `jms_job_status`, `jms_job_logs`,
  `jms_cluster_preflight`, `jms_cluster_submit` and `jms_dev_run` return TypedDicts instead
  of an untyped `dict`, so a client can validate and render the payload instead of guessing
  at fields. The schemas are partial by design, and a test asserts each one still lists the
  keys its tool actually returns.
- **DeepSeek Harness (dsh) is now a documented, verified MCP client.** The bridge already
  spoke plain stdio MCP, so no code changed — what was missing was the registration path,
  which is unlike Claude Code's and Codex's: dsh has no `mcp add` and registers a server by
  inserting an `@deepseek-ai/dsh-mcp-client` entry into a profile's `cordis.patch.yml`, so
  the README now shows that entry and `examples/dsh-mcp.patch.yml` is a copy-and-edit
  version of it. Two dsh behaviours are called out because each fails quietly otherwise:
  its 60-second default tool-call timeout sits below `jms_job_wait`'s own 150s default
  (raise `toolCallTimeoutMs`), and it scrubs `KEY`/`PASSWORD`/`SECRET`/`TOKEN`-shaped names
  from the parent environment, so every `JMS_BRIDGE_*` secret must be listed under the
  entry's `env:` rather than exported. Verified end to end against dsh 0.2.0-rc.2: the
  server connects and all 24 tools register as `mcp__jms-bridge__*`.

### Changed

- **CI now covers Python 3.11 and 3.13, and lints `scripts/`.** The matrix was 3.10 and
  3.12 only, so the one branch that legitimately differs by version — `config.py` using
  stdlib `tomllib` from 3.11 and the `tomli` fallback on 3.10 — was the exact boundary
  left untested, while `requires-python = ">=3.10"` advertises both ends. `scripts/` was
  never linted despite being the acceptance pass's MCP client (`mcp_call.py`) and its
  inventory builder (`verify_support.py`): a typo there breaks onboarding, not a test.
- **Coverage is measured and gated.** `pytest` now reports coverage by default with a
  floor (`[tool.coverage.report]` in `pyproject.toml`, a few points under the current
  77% so ordinary churn does not trip it), and `pytest-cov` joined the `dev` extra. The
  point is to fail a change that lands a large untested block — not to demand a number.

### Fixed

- **Compact command/transfer previews no longer load complete disk logs into memory.**
  Explicit presentation reads cache fixed stdout/error windows, preserve true byte
  sizes and truncation, and keep stderr-tail diagnostics. Complete outputs remain
  readable by result ID. Parser inputs remain full by default and no longer keep a
  duplicate full raw-byte buffer; legacy OpenSSH and parser contracts are unchanged.

- **Optional metadata errors no longer break REST asset discovery or hide silently.**
  Validate host mappings, UUID joins, shell-field types and platform keys, retain
  valid fields, and expose bounded value-free warnings in inventory status. Duplicate
  overlay UUIDs are ignored rather than choosing an order-dependent configuration.
  REST permissions and login/account selection remain authoritative.
- **Resident inventory refresh no longer performs metadata/durable cache I/O on the
  event loop.** Build and fsync off-loop, then publish the complete snapshot and its
  diagnostics together. Failed/cancelled refreshes retain the previous in-memory
  view; cancellation waits for outstanding disk work before releasing the lock.

- **Compact transfers use the installed rsync instead of pinning `/usr/bin/rsync`.**
  Resolve `PATH`, verify `-s --version` without blocking other MCP calls, and execute
  that same binary. Missing/incompatible binaries and failed probes have distinct
  errors; only unchanged successful checks are cached, so an upgrade or retry does
  not require restarting the service. The required `-s` path protection is retained.

- **A self-review of the fixes above found three of them half-done.** Each was a case of
  the right intent applied inconsistently, so the corrections are recorded rather than
  quietly folded in:
  1. `list_assets` bounded its pagination walk at ten pages and then returned the result
     as though it were complete — reintroducing, in the fix for silent truncation, exactly
     the silent truncation it was fixing (a 5000-asset deployment returned 2000 with no
     signal). It now returns `(assets, truncated)`; `truncated` is True only when the walk
     stopped while the server still offered a `next` page, and `jms_list_assets` surfaces
     that as a `warning` — not an `error`, since the call succeeded and this surface
     reserves `error` for refusals.
  2. `jms_asset_detail` shipped its inventory entry twice — once as `inventory_entry` (a
     dict) and once as `inventory_yaml` — which is the same needless duplication the
     `jms_job_logs` fix above removed, committed in the same batch. Only the YAML remains;
     a caller wanting a dict can parse it.
  3. `jms_asset_detail` returned a 211-character constant `note` on every call, restating
     guidance already in its docstring. Removed; the explanation lives in the docstring,
     which is where a client reads it once rather than per response.
- **`jms_probe` reported an unreachable host with no reason, and could call a failed probe
  healthy.** The sentinel fix caught the KoKo-rejection-as-green bug, but the *explanation*
  was still dropped: KoKo puts it on **stdout** ("必须是唯一的账号 root"), while the tool
  consulted only stderr, and only when the exit code was nonzero — so `reached: false` came
  with an empty `error` and the caller could see that a host failed but not what to fix. One
  `ssh.failure_reason` helper now serves every reachability site (`jms_probe`,
  `jms_fleet_summary`, `doctor`, `ssh-key --verify`, `probe`), prefers stderr only when the
  command genuinely failed, and the probe entry's field is renamed `error` (it is not always
  stderr). Verified against a host the real deployment rejects.
- **A tool reply could be arbitrarily large — one call produced 8 MB (≈2 M tokens).**
  `ExecResult.to_dict` forwarded stdout/stderr whole, and `jms_exec` / `jms_sync_push` /
  `jms_sync_pull` fan it out across the fleet, so `jms_exec(["cluster"], "cat
  train.log")` over 40 hosts returned 8 MB into the agent's context. Every other reply in
  the codebase was already bounded (`[:400]`, `[:20000]`); these three were not. A fan-out
  now shares one aggregate budget split across hosts (with a per-host floor), each host's
  two streams share *its* slice, and a clipped stream reports its true size
  (`stdout_bytes`) plus an explicit `truncated` flag and marker — so a partial result can
  never read as a complete one. Measured: 8.0 MB → 0.41 MB at 40 hosts, and bounded at any
  fleet size.
- **`jms_job_logs` sent its payload twice.** Adding an `outputSchema` to a tool whose
  payload is a large text blob doubled the reply — the SDK sends a structured result
  *alongside* the text block, measured at 100 KB of log crossing the wire as 200 KB. A
  blob tool gains nothing from a schema (the fields are already in the text), so
  `jms_job_logs` is unstructured again and a test keeps it that way.
- **`jms_job_wait`'s default timeout exceeded the tightest client cap.** It defaulted to
  150 s while dsh caps a tool call at 60 s by default, so the bounded wait was killed by
  the harness before it could return its own partial answer — the user saw a transport
  error instead of `finished: false`. The default is now 50 s, inside the cap, and a test
  asserts it stays there. The README's dsh section also named only `jms_job_wait`, missing
  `jms_dev_run` (600 s) and `jms_dev_logs` (300 s) — the dev-loop tools a user raises the
  cap for; all three are now documented.
- **`jms_probe` (and `jms-bridge probe`) reported a KoKo-rejected asset as a healthy,
  accelerator-less host.** KoKo refuses a connection (no permission, asset offline, bad
  address) with ssh **exit 0** and the error on stdout, so a status-only check is green on
  a dead asset. `jms_fleet_summary` and `doctor` already guarded with the `jms-bridge-ok`
  sentinel; `jms_probe` did not. The sentinel rule now lives in one place
  (`ssh.reached` / `ssh.REACH_PROBE`), every reachability decision calls it, and the probe
  report separates "never reached" from "the platform command failed on the host".
  `jms-bridge probe` also now exits nonzero when a host is unreachable — it always
  returned 0 before.
- **A cancelled or timed-out command leaked its ssh/rsync process.** `_run_script` and
  `rsync` caught only `TimeoutError`, and `asyncio.CancelledError` is a `BaseException`, so
  a cancelled call (a client disconnecting, a long response the gateway cuts off) skipped
  the kill entirely and left the child holding the ControlMaster socket. Both now kill and
  reap under `except BaseException`, through one `ssh._kill` helper, and re-raise.
- **Every control-plane call turned a transport failure into an opaque tool error.**
  `jms_list_assets`, `jms_audit_query` and `jms_create_ticket` caught `JumpServerError`,
  but `httpx` raises its own family (ConnectError, ConnectTimeout, TLS failures) for a
  mistyped URL, DNS failure or timeout — the *first* thing a new user hits. `api.request`,
  `login_with_password` and `ping` now normalise those into `JumpServerError`, which every
  caller already renders as a message.
- **`jms_list_assets` silently truncated at 200 assets.** The first page was returned as
  though it were the whole fleet; a deployment with 250 permitted assets showed 200 with
  nothing to say so. `list_assets` now follows DRF `next`/`offset` pagination and still
  handles the older bare-list shape. (The runaway guard added here was itself silently
  truncating; see the self-review note below for the correction.)
- **`jms_job_logs` reported `ok: true` on a host it never read.** An unreachable host
  returns an empty body, indistinguishable from a job that has not logged yet, so the tool
  hard-coded a green that was really "we learned nothing". `JobManager.logs` now reports
  the read's own transport status (`ok`, `exit_code`, `error`), and the tool forwards it —
  matching `jms_job_cancel`, which already took its `ok` from the operation.
- **A malformed `config.toml` / `hosts.yaml` reached the agent as an opaque tool error.**
  The `_tool()` wrapper only translates `ConfigError` / `InventoryError`, but a typo made
  the raw `TOMLDecodeError` / `yaml.ParserError` escape — so *every* tool degraded to
  "Error executing tool <name>" at once, for the most ordinary mistake there is.
  `read_toml` and `Inventory.load` now translate parse failures (and a non-mapping
  inventory) into their domain errors, with the file path and the parser's line.
- **`jms_cluster_submit` refused the single-host island its own docs told you to submit.**
  The gate required two hosts, but `jms_cluster_plan` emits one-node islands for a mixed
  cluster and the README's federation walkthrough submits the lone Ascend box on its own.
  A single host is now a valid submit (one rank node); an empty selection is refused with
  the usual "matched no host".
- **The documented `torchrun` recipe conflated device rank with node ordinal.** Ranks were
  exported as `JMS_RANK` (global *device* rank) and `JMS_WORLD_SIZE` (total *devices*),
  while the README and tool docstring used them for `--node_rank` / `--nnodes`. A 2-node,
  2-chip run therefore asked for `--nnodes=4 --node_rank=2` — four nodes that do not exist
  and an ordinal that is not a node. `JMS_NODE_RANK` and `JMS_NNODES` are now exported
  alongside the device-level pair, and every example uses the node-level ones.
- **`jms_probe` probed its hosts sequentially.** Its docstring promised a parallel fan-out
  and every other fan-out in the file uses `gather`; a 40-host probe paid 40 SSH round
  trips in series. It is now concurrent and isolates a host that raises, so one bad node no
  longer erases the other hosts' results.
- **`jms_dev_logs` silently drilled into the first workspace a tag matched.**
  `resolve_workspaces` treats a selector as a union, so a tag could match several chips and
  the tool answered about an arbitrary one. An ambiguous selector is now refused by name.
- **`jms_create_ticket` turned a `fields` collision into a bare `TypeError`.** `fields` is
  spread beside `title` / `comment` / `org_id`, so a repeated key raised from inside the API
  call and surfaced as "Error executing tool jms_create_ticket". The collision is now a
  readable refusal naming the keys.
- **`jms_dev_discover` on an empty selection gave no reason.** It returned `ok: false` with
  no `error`, unlike every other fan-out; it now says the selectors matched no host.
- **`jms_fleet_summary` claimed an `all` default it does not have.** `targets` is required;
  the docstring said "the default of `all`". It now says pass `all`.
- **`doctor`'s data-plane handler caught `(InventoryError, Exception)`.** `InventoryError`
  derives from `Exception`, so the first element was dead weight that read as though a
  narrower case were handled separately. It is now `except Exception`, with the same
  behaviour and an accurate comment.
- **A README "24 tools" claim that nothing checked.** The count lived in prose while the
  real surface was the set of `@_tool()` decorators, so adding a tool silently left the
  doc wrong. A test now derives the count from the server's registry and compares it, and
  the claim is verified to fail when the two drift.
- **`api.request`'s `json` parameter shadowed the `json` module.** The workaround was an
  inline `__import__("json")`, and the shadowed module had leaked into `__all__` as though
  it were public API (nothing imported it that way). The parameter is now `json_body` and
  the export is dropped.
- **`discover.build_discovery_script` carried two adjacent docstrings.** The second was
  dead — a no-op expression statement — and the first was the real one; the stray is gone.

## 0.2.2 — 2026-10-14

### Fixed

- **Every MCP client was told the server had no version.** The server was built as
  `MCPServer("jms-agent-bridge")`, and the SDK defaults `version` to the empty string —
  so the version an agent shows for this server (and that a bug report would quote back)
  was blank, while the real one sat unused in `jms_bridge.__version__`. It is now passed
  through, and a test asserts the handshake reports it, so the two cannot drift apart
  unnoticed. Nothing else about the tool surface changes: every launch path imports the
  same server object, so this was the only construction site.

## 0.2.1 — 2026-10-14

### Fixed

- **Islands were partitioned on a backend *name*, so mixed-vendor clusters were planned
  as one collective.** `dist_backend` doubled as the island key, and five unrelated
  stacks all answer `nccl`: NVIDIA, Metax, Kunlunxin, T-Head and Iluvatar each ship their
  own `libnccl.so.2` and cannot rendezvous with each other or with NVIDIA's. Selecting a
  CUDA host alongside any of them produced a single island and a `jms_cluster_submit`
  that could only fail at launch. Platforms now carry a separate `collective` — the
  interoperability key islands actually partition on — so only hosts sharing one engine
  are grouped. `Island.backend` is `Island.collective` in the reply for the same reason
  (the value is not always a backend name).
- **Three domestic `dist_backend` guesses were wrong, and the field claimed to be
  verified.** They were labelled UNVERIFIED but the notes came from the chip names: it is
  NCCL, not CNCL, on Iluvatar and not BKCL on Kunlunxin, and Enflame's ECCL has no
  `ProcessGroup` at all — FlagCX is the only path to it, so `eccl` named a backend no
  torch build exposes. The values now come from the measured vendor table in the sibling
  PyTorch-Plugin-FL project (`torch_fl/comm/process_group.py:_VENDOR_PROFILES`), and
  TsingMicro — absent from that table — says so instead of asserting a backend. A test
  pins the whole table so it cannot drift back.

### Added

- **`jms-bridge ssh-key` — the key-registration hand-off, made safe to perform.**
  Key auth is the recommended data plane, but the two steps around it had no safe
  order: registering a key meant hand-rolling `ssh-keygen` and pasting the public half
  into the web UI (this deployment exposes no REST endpoint for it), and removing
  `ssh_password` before the key was proven could leave the data plane with no working
  credential. The subcommand generates a bastion-only key (0600, no passphrase — the
  MCP server is non-interactive, so a prompt nothing can answer is a failure mode, not
  a protection), prints the public half and the exact web steps, and `--verify` then
  proves the key *alone* authenticates by cloning the config with the password removed.
  Reachability is judged by the `jms-bridge-ok` sentinel, not the exit code, so a
  refused asset is not reported green.
- **`jms_fleet_summary` — the fleet in one call.** Reachability and accelerator
  inventory for a selector used to take one `jms_probe` round per host plus a manual
  tally of who answered. This tool fans the whole set out in parallel and returns
  `hosts_total` / `hosts_reachable` / `devices_total` plus a `by_platform` breakdown
  (`hosts` / `reachable` / `devices`), the hosts whose device count came back unknown,
  and each unreachable host with its reason. Reachability is proven by the
  `jms-bridge-ok` sentinel coming back rather than by exit status: KoKo rejects a
  connection with **exit 0** and the error on stdout, so an exit-code check would count
  a dead asset as green. A host that answers but reports no device count contributes 0
  devices and is named under `device_count_unknown` instead of being guessed at.
- **`python -m jms_bridge` — the CLI without an install.** `[project.scripts]` puts
  `jms-bridge` on `PATH` only after `pip install`, so a freshly-cloned checkout had to reach
  `init` / `login` / `doctor` through `python -c 'from jms_bridge.cli import main; …'`. The
  new `src/jms_bridge/__main__.py` is the same `main`, reached the ordinary way.
- **`platforms.guess_platform(declared, name, address)`** — the one place that decides which
  profile an asset JumpServer described should be probed with, returning the profile's
  canonical name plus a note when the answer is only a fallback. The declared field is
  scanned first, so a customer's free-text label containing a vendor word cannot override a
  platform that really is known; the asset's own name and address are the second source.

### Fixed

- **A fan-out whose selector matched no host reported success.** `resolve([])` returns `[]`
  without raising and `all()` over nothing is `True`, so `jms_probe`, `jms_exec`,
  `jms_sync_push`, `jms_sync_pull`, `jms_submit_job`, `jms_job_status`, `jms_job_logs`,
  `jms_job_cancel`, `jms_job_wait`, `jms_cluster_cancel`, `jms_cluster_plan` and
  `jms_cluster_preflight` could all come back green over zero hosts — and for
  `jms_job_wait` that green was the claim "every rank is terminal" about hosts nobody had
  asked. Each now returns `ok: false` with "matched no host", in its own reply shape, and
  without contacting anything.
- **`jms_cluster_preflight` raised on an empty selection** instead of answering.
  `resolve_rendezvous([])` raises `InventoryError("cannot resolve a rendezvous for an empty
  host list")`, which the `bool(health)` guard could not reach because the rendezvous is
  resolved first — so the caller got an opaque tool error rather than a preflight that says
  "not ready".
- **The job tools returned an opaque error for a malformed job id.** `jms_job_status`,
  `jms_job_logs` and `jms_job_cancel` passed the argument straight to the manager, where
  `ValueError` is not one of the two exceptions `@_tool()` translates, so the client saw
  `Error executing tool …` rather than what was wrong with the id. They now use the
  message-returning `validate_job_id` and return `{"ok": false, "error": …}`. The same
  lookup no longer raises `IndexError` for a host that resolved to nothing.
- **`scripts/verify_support.py` could tag a domestic accelerator `cuda` in silence.** Its
  platform list was a hand-maintained copy of four entries, so it drifted from the profiles
  the probes are built from — and an asset it did not recognize was tagged `generic`, which
  probes `uname`, finds no accelerator and reads as a healthy host with none. The tagger is
  now generated from `platforms.PLATFORMS` (via `guess_platform`, so the same name/address
  heuristics the rest of the bridge uses) and every fallback is reported as a
  `warning: <host>: <why>` line on stderr. `scripts/verify-onboarding.sh` captures stderr
  together with stdout, so its two asset-table loops now pass a `warning:` line through to
  the terminal instead of rendering it as a row whose columns are all empty.

## 0.2.0 — 2026-10-13

### Added

- **`scripts/verify-onboarding.sh` — the acceptance pass, runnable.** The standing
  criterion (clone on a second machine, configure an account and a password, drive the
  cluster) is exactly what reading the code cannot confirm, so the whole first-run path is
  now scripted: `init` → `login` → `jms_list_assets` → build `hosts.yaml` → `doctor` (the
  gate) → `jms_dev_discover` → one real `jms_dev_run`, stopping at the first step that does
  not work and printing that step's real output. It pins the config, inventory and
  credentials inside a scratch `mktemp -d` and forces keyring's null backend, so the real
  `~/.config/jms-bridge` and the OS keychain are never touched, and leaves `$HOME` alone
  because ssh still needs the real `~/.ssh`. `scripts/mcp_call.py` speaks the MCP stdio
  protocol directly, so the pass exercises the same tools a Claude Code session calls
  rather than a private in-process API.
- **Configurable SSH fan-out cap** (`max_parallel`, or `JMS_BRIDGE_MAX_PARALLEL`). The
  fan-out width was hard-coded at 8 and passed nowhere, so a wide selector on a large,
  healthy fleet could not be widened without editing the code — and the value that reached
  `SSHRunner` was always its default. The server and both CLI paths now build the runner
  from the setting, a sub-1 value is rejected where it was written (a `Semaphore(0)` is a
  hang, not an error), and directly-constructed runners clamp rather than deadlock.
- **Flagos development loop (4 tools).** `jms_dev_discover`, `jms_dev_sync`,
  `jms_dev_run`, `jms_dev_logs` implement *write once, verify on every chip*: deliver a
  git `ref` into every workspace's container and return one cross-chip consistency table
  aligned by test id. Delivery is per workspace (`git` / `proxy-git` / `push`), the suite
  runs under `set -o pipefail` inside the container, and `jms_dev_run` attests what
  actually ran (`torch_fl/_build_config.py`, `torch_fl/backends.conf`, the `_C` ABI tag)
  so "same sha" is never mistaken for "same code path".
- **Workspaces** (`workspaces:` in the inventory): host + container + repo + delivery
  source + `ut_cmd` + an in-container `env_setup` run before the suite, with tag/name
  selection.
- **Coordinated cluster launch** — `jms_cluster_preflight`, `jms_cluster_plan`,
  `jms_cluster_submit`: one job with a shared rendezvous and per-rank
  `JMS_RANK`/`JMS_NPROC`/`JMS_WORLD_SIZE`/`JMS_MASTER_ADDR`/`JMS_MASTER_PORT`, plus
  backend-homogeneous **islands** (HCCL/NCCL/gloo) for heterogeneous clusters.
- **Job lifecycle, without hand-rolling a poll loop (3 tools).** `jms_job_wait` blocks
  for a *bounded* timeout (default 150s, deliberately under the gateway's ~200s response
  buffer) until every job is terminal, then returns whether they had finished — a
  timeout stops *watching*, never the job. `jms_job_list` lists every job the bridge has
  launched on a host, and `jms_cluster_cancel` stops **every rank** of one coordinated
  launch from that launch's single job id.
- **Artifact collection (1 tool).** `jms_pull_artifacts` pulls one remote path from many
  hosts into a local directory, giving each host its **own** subdirectory so same-named
  checkpoints from different chips cannot overwrite each other; a host name is sanitised
  into a safe segment and a remote path that would escape the workdir is rejected.
- **Device-occupancy detection in preflight.** `jms_cluster_preflight` now asks each
  accelerator what already holds a compute process (`npu-smi info -t proc-mem`,
  `nvidia-smi --query-compute-apps`), so two trainings can no longer land on the same
  cards undetected. The count is advisory — only a *positive* parse raises the flag, and
  `jms_cluster_submit` refuses occupied hosts unless `allow_busy=true` (the usual honest
  reason a card is busy is your own previous run).
- **Control-plane bootstrap** — `jms-bridge login` exchanges username/password for a
  session token and caches it in the config (mode 600), so the server never handles a
  password; `jms_inventory_reload` now drops the cached config and credentials too, so a
  reload after logging in takes effect without a restart.
- **KoKo password auth** for the data plane (`JMS_BRIDGE_SSH_PASSWORD`, via `sshpass`),
  for the window before an SSH key is registered. `BatchMode` is deliberately not set on
  this path (it would suppress the very prompt sshpass exists to answer).
- **Direct-SSH hosts** (`ssh_host`/`ssh_port`/`ssh_user`) for nodes reachable without
  the gateway.
- **Persistent credential storage.** `jms-bridge login --save-password` now also stores
  the password itself, so a later login (a new shell, an expired session token) needs no
  secret. Resolution is layered `JMS_BRIDGE_PASSWORD` → OS keyring → a mode-600
  `~/.config/jms-bridge/credentials.toml` (overridable with `JMS_BRIDGE_CREDENTIALS`); a
  working keyring takes the password and deletes any plaintext copy. `login --forget`
  deletes every stored copy and `jms-bridge doctor` reports the store and keyring
  backend. The keyring is an optional `[keyring]` extra, so the default install is
  unchanged and a headless host degrades quietly to the file.

- **Domestic-accelerator platform profiles (8 vendors).** `ascend-910c` / `cuda` no
  longer cover the whole fleet: `hygon-dcu` (海光, `hy-smi`), `metax` (沐曦, `mx-smi`),
  `enflame` (燧原, `efsmi`), `kunlunxin` (昆仑芯, `xpu-smi`), `thead` (平头哥, `ppu-smi`),
  `iluvatar` (天数, `ixsmi`), `moore-threads` (摩尔线程, `mthreads-gmi`) and `tsingmicro`
  (清微, `tsm_smi`) each probe their own SMI for liveness, runtime version and device
  count, so `jms-bridge probe` and preflight report real numbers instead of falling back
  to a bare `uname`. Each also carries a **per-vendor occupancy query** — hygon keys on
  the `HCU Index` each KFD process holds, metax/enflame/moore-threads/tsingmicro on the
  device id in each process row, and kunlunxin/thead/iluvatar on the device header a
  nested `Process ID` sits under — so the busy gate applies on these hosts too. A vendor
  with a recognizable "nothing running" reply licenses a trusted zero (metax, enflame,
  thead, kunlunxin, moore-threads); one that does not stays *unknown* on an idle host
  (hygon, iluvatar, tsingmicro), which raises the flag only on a positive parse. Their
  `dist_backend` values are the vendors' commonly cited torchrun backends but are
  **unverified on this fleet**; confirm before a multi-node launch.

### Changed

- **Every tool that takes a caller-supplied command now shares one denylist and one
  override.** The two halves had drifted apart in opposite directions: `jms_dev_run`'s
  `ut_cmd` override was interpolated into a container shell with no `_looks_dangerous`
  check at all, while `jms_cluster_submit` screened its `command` and hard-failed with no
  `allow_dangerous` escape hatch, so a caller who genuinely needed an override was stuck.
  The commands an inventory carries (`ut_cmd`, `env_setup`) are spec text at the same trust
  level as the rest of `hosts.yaml`, and are deliberately still not screened.
- **The onboarding path now leads to a working dev loop.** `jms-bridge init` writes an
  inventory with a `workspaces:` block, so the documented quickstart is no longer a
  sequence that ends in `jms_dev_run` matching nothing. Its `Next:` steps are ordered
  around the one field a new user cannot fill in yet — a JumpServer **asset id** is only
  knowable from `jms_list_assets` *after* a login — and both `init` and the README point
  at `examples/`, which exercises every supported field.
- **`init` writes `hosts.yaml` mode 600 as well.** It names every host and its asset id;
  previously only `config.toml` was tightened, leaving the inventory world-readable until
  a later `login` (which only ever touched the config).
- **`doctor` labels the two credentials and checks `sshpass`.** The control plane takes
  `JMS_BRIDGE_PASSWORD` and the data plane takes `JMS_BRIDGE_SSH_PASSWORD`; a single
  "credentials" line invited the assumption that `login --save-password` covered both. It
  now reports each separately, and reports `sshpass` under the same section when
  `ssh_password` is set, so password auth fails with a diagnosis instead of an unexplained
  data-plane error.
- **All ranks of one coordinated launch now share a single job id.** Ranks live on
  different hosts so one id cannot collide, and it is what lets `jms_job_status` /
  `jms_job_wait` / `jms_job_logs` / `jms_cluster_cancel` take that one id instead of a
  rank-to-host pairing the caller had to reconstruct.
- **A zero device-occupancy count is no longer trusted on its own.** A non-empty reply
  with no recognizable device row is equally what a drifted vendor format or a stray
  warning looks like, so it now degrades to *unknown* rather than "the cards are free".
  Ascend licenses its zero with the `proc-mem` table header; CUDA answers nothing when
  free, so it needs no license and any non-empty no-UUID reply is a warning.
- **The default UT command is scoped to `tests/unit`** (was a bare `tests/`). That is
  the target repo's agnostic suite; a bare `tests/` also collects `tests/manual/**` and
  `tests/perf/**`, which are hardware-bound dev scripts — some parse `sys.argv` at
  import, which aborts collection with an argparse `SystemExit(2)` under pytest.
  `tests/integration/**` needs models and datasets, so it stays a separate explicit run.
- **Dependency versions pinned.** The `dev` extras are pinned exactly (`==`) so a test
  run means the same thing locally and on CI; the runtime dependencies gain
  major-version caps (`mcp>=2.0,<3`, `httpx>=0.27,<1`, `PyYAML>=6.0,<7`) so an upstream
  major cannot break a checkout silently.

### Fixed

- **`SSHPASS` was exported to every child the runner spawned.** A direct (`ssh_host`) host
  is key-based, so `sshpass` is not in its argv — handing it the bastion password put the
  secret where it had no purpose. It now reaches only the child that will actually
  authenticate with it, and `close()`'s `ssh -O exit` (which talks to the existing control
  socket) carries none either.
- **The SSH runner was never closed on shutdown.** Nothing called `SSHRunner.close()` when
  the server exited, so every session left one `ControlMaster` socket and one `/tmp/jmsb-*`
  directory per host it had talked to. `main()` now tears down the lazily-built clients on
  the way out — best-effort, so a cleanup failure cannot turn a clean exit into a nonzero
  one.
- **The `iluvatar` probe silently found nothing on three of four hosts.** `ixsmi`
  ships under `/usr/local/corex/bin` and is only on `PATH` on some nodes; where it
  was not, the bare `ixsmi` probe returned empty and every consumer read the host as
  having no accelerator. The probe (and the new occupancy query) now sets
  `LD_LIBRARY_PATH` to the corex lib dir and falls back to the install path, so all
  four hosts report their device count and their busy cards.
- **`jms-bridge doctor` reported green for assets KoKo refused.** A rejection (no
  permission, asset offline, bad address) comes back over the gateway with ssh **exit
  code 0** and the error text on stdout, and the data-plane check trusted the exit code
  alone — so a `403 没有资产或资产未激活`, a `No route to host` or a bad-address dial all
  read as `OK`. The probe now requires the `jms-bridge-ok` sentinel it echoes to come
  back in the reply, so only a command that actually ran on the far side counts as
  reached. Six unauthorized DGX assets and several unreachable hosts that doctor had
  been marking green now correctly fail.
- **`jms_cluster_cancel` on an empty selection reported success.** `resolve([])` returns
  `[]` without raising and `all()` over nothing is `True`, so a cancel that reached no
  host read as a green result. It now reports `ok: false` with "no hosts matched the
  selection; nothing was cancelled", the same guard the other fan-out tools carry.
- **`jms_cluster_preflight` under-reported `world_size_estimate`.** A reachable host
  whose probe reported no device count was counted as **0** processes while
  `jms_cluster_submit` gives it `nproc=1`, so the preflight predicted a narrower
  collective than the launch would actually form. Both now apply the same
  `max(1, device_count or 1)`, and a test pins the two together.
- **`jms_cluster_submit` assigned ranks to hosts that could never join.** An unreachable
  host reports no device count and the tool still gave it `nproc=1`, so every launched
  rank was told a `JMS_WORLD_SIZE` larger than the set that could actually rendezvous —
  the job hangs, or the survivors train a collective of the wrong width. Every target is
  now preflighted first (the same readiness test `jms_cluster_preflight` reports, so the
  two tools never disagree) and an unreachable host, or one holding a stale rendezvous
  `port`, fails the call and names itself instead of being ranked.
- **One rank's failure discarded every other rank's launch result.** A raise before a
  result exists (a missing local tool, a transport error) escaped the fan-out `gather`
  and surfaced the whole coordinated launch as an opaque tool error, hiding which nodes
  came up. Each rank is now isolated, so the rest of the batch reports normally. The same
  applies to `jms_dev_sync` / `jms_dev_run`: a workspace whose delivery or UT run raises
  is recorded as *that workspace's* failure instead of losing the cross-chip table.
- **A missing local tool reached the agent as an opaque message.** A fresh machine with no
  `ssh`, `rsync` or `sshpass` produced a bare `FileNotFoundError`, shown as
  `Error executing tool <name>`. Each now raises `ConfigError` naming the binary and what
  to install.
- **An incremental job-log poll could silently skip output.** `jms_job_logs` returned the
  file's *total* size while the read was clipped at `max_bytes`, and polling with it (as
  both the tool and the README instructed) skipped everything between the clip and EOF —
  routine for a training log. The result now carries `next_offset`, where the read
  actually stopped, and the docs point at it.
- **`jms_dev_logs` re-ran the drill-in under the wrong interpreter.** It ignored the
  workspace's `env_setup`, so a failing test was re-run under the container's default
  interpreter — the wrong ABI on most chips — and reported the cause of a failure the
  fleet run never hit. Activation now precedes the repo `cd`, exactly as `jms_dev_run`.
- **A `push` workspace with `local: ~/...` delivered a directory literally named `~`.**
  rsync runs through `exec` with no shell, so `~` was never expanded and the chip could
  run stale code while the run still looked green; `local` is now `expanduser`-ed, the
  same way `_local_sha` already did.
- **A job could abort before its command ever started when `env_setup` read an unset
  variable.** `jms_submit_job`'s runner script runs under `set -u`, and vendor env
  scripts (CANN's `set_env.sh`, many conda hooks) reference variables that may be unset;
  the runner now relaxes `-u` around `env_setup`, matching `wrap_command`.
- **Declare `tomli` for Python 3.10.** `config.py` falls back to `tomli` when
  `tomllib` is absent, but `tomli` was not a declared dependency, so an install on 3.10
  crashed at import. Now a conditional dependency.
- **rsync worked only with GNU rsync.** `--info=stats1` is a GNU extension that the
  macOS built-in (openrsync) rejects; the transfer now runs with no GNU-only flags and
  works with either implementation.
- **Setup errors reached the agent as an opaque message.** `ConfigError` /
  `InventoryError` raised inside a tool were wrapped by the MCP SDK and shown as
  `Error executing tool <name>`. They are now translated to `ToolError`, so the
  actionable text ("No inventory at …; run `jms-bridge init` … or set
  `JMS_BRIDGE_INVENTORY`") is what a freshly-cloned-but-unconfigured checkout actually
  sees.
- **CLI and server disagreed on the config path.** The CLI hard-coded the default path
  and ignored `JMS_BRIDGE_CONFIG`, so `login` could cache a token in one file while the
  server read another. Both now resolve it through `config.config_path()`.
- **`jms-bridge login` robustness:** validates that a config exists before trying to
  write to it, reports transport failures (DNS/refused/TLS/timeout) as a message rather
  than a traceback, and `init` writes the config mode 600 from the start.
- **Timeout detection on Python 3.10/3.11** — a timeout raised as `TimeoutError` was
  caught but `asyncio.TimeoutError` was not, so a long command could surface as an
  unhandled error instead of a `timed_out` result.
- **Control-socket directory leaked** one `/tmp/jmsb-*` dir per reload; `close()` now
  removes it and the runner re-creates it on demand.
- `jms_submit_job` accepts the `allow_dangerous=true` override the README already
  documented (and the tool schema now exposes it).

## 0.1.0 — 2026-10-09

Initial release.

- **MCP tool surface (11 tools):** `jms_list_assets`, `jms_audit_query`,
  `jms_create_ticket`, `jms_probe`, `jms_exec`, `jms_sync_push`, `jms_sync_pull`,
  `jms_submit_job`, `jms_job_status`, `jms_job_logs`, `jms_job_cancel`.
- **Control plane:** JumpServer REST client with access-key (HMAC-SHA256) and
  private-token auth, asset/account discovery, command audit, tickets.
- **Data plane:** SSH through the KoKo gateway with `ControlMaster` multiplexing and
  bounded parallel fan-out; rsync push/pull.
- **Heterogeneous platforms:** `ascend-910c`, `cuda`, `aarch64`, `generic` profiles
  (probe command, version parsing, device count, distributed backend).
- **Non-blocking jobs:** tmux-backed submit/status/logs/cancel, with incremental log
  polling by byte offset.
- **CLI:** `init`, `doctor`, `probe`, `serve`.
- Built against `mcp>=2.0` (`MCPServer`).
