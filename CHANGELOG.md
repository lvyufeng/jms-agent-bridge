# Changelog

## Unreleased

### Added

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

### Changed

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

- **Declare `tomli` for Python 3.10.** `config.py` falls back to `tomli` when
  `tomllib` is absent, but `tomli` was not a declared dependency, so an install on 3.10
  crashed at import. Now a conditional dependency.
- **rsync worked only with GNU rsync.** `--info=stats1` is a GNU extension that the
  macOS built-in (openrsync) rejects; the transfer now runs with no GNU-only flags and
  works with either implementation.
- **Setup errors reached the agent as an opaque message.** `ConfigError` /
  `InventoryError` raised inside a tool were wrapped by the MCP SDK and shown as
  `Error executing tool <name>`. They are now translated to `ToolError`, so the
  actionable text ("No inventory at …; create one or set `JMS_BRIDGE_INVENTORY`") is
  what a freshly-cloned-but-unconfigured checkout actually sees.
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
