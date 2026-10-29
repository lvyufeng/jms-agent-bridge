# Compact resident service

`jms-bridge serve --compact` exposes five tools and keeps SSH connections inside
its process. The existing `serve`, `doctor`, `probe`, and YAML-based workflows
remain available. Compact mode is opt-in and requires Python 3.10+ on a POSIX
host, Bash on target assets, and rsync on both ends for file synchronization.

```sh
pip install 'jms-agent-bridge[compact]'
jms-bridge serve --compact --transport streamable-http --port 8766
```

This serves Streamable HTTP at `http://127.0.0.1:8766/mcp`. The listener is restricted
to loopback; it is a local tool with the privileges of the invoking user, not an
unauthenticated network service to expose publicly. Run one instance under your
existing service manager. `examples/jms-bridge.service` is a systemd user-service
example; adapt `ExecStart` to the installed executable and enable it with
`systemctl --user enable --now jms-bridge`. For startup without an interactive
login, configure user lingering according to the host's administration policy.

For a single stdio client, use `jms-bridge serve --compact` without `--transport`.
One process owns a state directory. Multiple agents should share the resident HTTP
endpoint; separate stdio processes need separate `bridge.state_dir` values.
A file lock prevents two services from overwriting the same shell registry.

Use the normal `[bastion]` API credentials and SSH key/agent or SSH password
configuration. The compact transport validates host keys using OpenSSH known_hosts;
provision a trusted gateway key first. It does not invoke OpenSSH or reuse its
ControlMaster sockets. Transfers use a private local socket relay to carry rsync
traffic over the same authenticated AsyncSSH connection.

Optional settings in `config.toml`:

```toml
[bridge]
refresh_interval = 3600
# Select an account only when it is present in the asset's permitted accounts.
# Otherwise exactly one usable account must exist; ambiguity is not guessed.
preferred_account = "root"
# Defaults to the hostname in bastion.url and bastion.ssh_port.
# ssh_host = "gateway.example.com"
# known_hosts = "~/.ssh/known_hosts"
# host_key_alias = "gateway.example.com"
# REST also honors the normal HTTP(S)_PROXY environment when rest_proxy is unset.
# rest_proxy = "http://proxy.example.com:3128"
# This is an argv array, never a locally interpreted shell command:
# ssh_proxy_command = ["nc", "-X", "connect", "-x", "proxy.example.com:3128", "gateway.example.com", "2222"]
# state_dir = "~/.local/state/jms-bridge"
# Active commands are separate from authentication and idle connections.
# execution_limit = 8       # Unset: use bastion.max_parallel.
# inspection_reserved = 2   # Default 0; must be less than the effective limit.
# queue_timeout = 60        # Separate from remote execution timeout.
# result_ttl_seconds = 604800
# result_cleanup_interval_seconds = 3600
```

`JMS_BRIDGE_STATE_DIR` overrides `bridge.state_dir`; otherwise `XDG_STATE_HOME`
(or `~/.local/state`) supplies the default. Keep credentials out of examples and
version control. Restart the service after changing its connection configuration.

## Tools and progressive disclosure

| Tool | Purpose |
| --- | --- |
| `fs` | Structured remote file exploration and upload/download; optional existing Docker container. |
| `run` | Execute literal Bash or a local script file, optionally in remote Docker. |
| `read_result` | Read stored stdout/stderr/JSON by byte cursor, tail, or literal search. |
| `discover` | Search permitted assets, inspect status, or disclose operation schemas. |
| `call` | Invoke one discovered operation with its validated arguments. |

Examples use tool arguments, independent of the client's namespacing:

```json
{"topic":"assets","query":"worker"}
{"targets":["worker-1"],"script":"printf '%s\n' \"$PWD\"","cwd":"/tmp"}
{"targets":["worker-1"],"script_path":"/path/to/check.sh","container":"development"}
{"result_id":"<returned-id>","stream":"stderr","tail":true,"limit":2048}
{"query":"jobs"}
{"operation":"jms_submit_job"}
```

Then `call(operation="jms_submit_job", arguments={...})` uses the discovered schema.
All 25 existing operations remain discoverable, plus `session_list` and
`session_close`. Workspace-specific development operations still require the
legacy configured-workspace workflow; REST discovery does not infer a workspace.
`jms_inventory_reload` in compact mode refreshes REST without closing SSH sessions.
Generic platform metadata does not establish accelerator readiness.

REST pagination and permitted account/protocol details build the machine scope.
Successful refreshes replace the entire snapshot, including removals. Incomplete
or failed refreshes retain the last successful snapshot with freshness/error
metadata. Cached snapshots are bound to the deployment and credential identity.
Background refresh runs hourly even when idle; retries after failure are bounded.
An unavailable API does not prevent MCP initialization or status inspection.
No password login or credential-renewal behavior is added by compact mode.

Optional `hosts.yaml` metadata supplies only `env_setup`, `workdir`, and `platform`,
joined by canonical `asset_id` UUID. It cannot add permitted assets or override
REST login/accounts. A missing or empty file is normal. A malformed mapping or
unreadable YAML is ignored; invalid fields are skipped while other valid fields
remain usable. Shell fields must be strings without NUL (empty strings are allowed),
and `platform` must be a supported key; invalid platforms fall back to `generic`.
Repeated UUIDs with overlay fields are ambiguous, so their overlays are all ignored.

`discover(topic="status")`, asset search and `jms_inventory_reload` report up to
five `metadata_warnings`, with `metadata_warning_count` and
`metadata_warnings_truncated`. Diagnostics use one-based host entry positions and
field names, never configuration values, host keys, file paths or raw parser errors.
Fix the file and refresh to clear warnings. Metadata warnings do not mark REST
refresh as failed; failure retains the last published metadata and its warnings.

Refresh reads/parses metadata, builds selectors and writes/fsyncs the cache in a
worker thread. The new snapshot, metadata and warnings are published together only
after success; status and existing targets remain available during disk work.
Calls with an existing snapshot do not queue behind an in-progress refresh, even
when the snapshot is stale; freshness remains visible in inventory status and
execution replies. The first refresh must still finish before targets are usable.
Cancellation waits for outstanding disk work before releasing the refresh lock and
keeps the old in-memory view. A complete newer cache may already be on disk and is
eligible for normal cache recovery after restart. Cache recovery during service
construction remains synchronous. No command or request is replayed by refresh.

## Output and shell semantics

Every execution gets private `meta.json`, `stdout.log`, and `stderr.log` files;
low-frequency responses also get a `result.json`. Previews share a default 4096
byte budget across targets and reserve space for stderr. `max_output_bytes` is
256..16384. Pages are at most 16384 bytes plus up to three bytes for UTF-8 alignment.
Non-UTF-8 bytes remain intact on disk and display with replacement characters.
Literal searches return bounded snippets and a continuation cursor, scanning at
most 8 MiB per request. Completed outputs expire after 7 days by default, with a
startup and hourly sweep. Reads do not extend retention. Active writers are
protected; old orphan outputs from a prior service expire from creation time.
Cleanup never removes remote source files or job directories. Request receipts
are stored separately and retained so output expiry never enables re-execution.

Compact `jms_exec` and native transfer replies do not reload complete log files
into memory. They cache at most 16 KiB of stdout's prefix and each end of stderr;
compact replies still share their existing 4096-byte head/stdout, tail/stderr budget.
The bounded snapshots report truncation from the true on-disk byte counts. Read
`result_id` with `read_result` to access all bytes/pages; non-UTF-8 bytes are never
re-encoded back into the stored logs. Preview reads run off the event loop.

Internal `NativeSSHRunner.exec_on` / `exec_many` keep full parser input by default;
presentation-only callers explicitly use `preview_only=True`. Native `rsync`
defaults to bounded previews, with `preview_only=False` available for internal full
output consumers. Legacy short error summaries keep stderr's prefix, while compact
fan-out diagnostics retain its tail. Full parser results no longer retain a second
complete raw-byte copy. Job log paging, launcher acknowledgments, probe sections,
pytest/attestation and workspace parsing remain full-input paths; this is not a
claim that every parser is memory-bounded. The legacy OpenSSH runner is unchanged.

Set a unique `request_id` on `run` or on
`call(operation="jms_submit_job", arguments={...}, request_id="...")` before
starting important work. After a lost response, discover/call `request_status`
with the original ID. Reusing the same ID and arguments returns the receipt;
changed arguments are rejected. Caller cancellation leaves an accepted request
owned by the resident service. A restart never resumes or replays it: reconcile
saved result/job IDs when `completion_unknown` is true. A prepared job ID is a
reconciliation candidate, not proof that the remote launcher acknowledged it.

The existing destructive-pattern guard is a local typo check.
`allow_dangerous=true` acknowledges that check for a specific command already
authorized by the user; it neither grants approval nor overrides client review,
OS permissions or JumpServer policy. `script_guard_check` diagnoses the live
predicate without executing the script. A narrow literal `python3 -I -S -`
heredoc can ignore proven Python comment tokens; strings, executable text,
malformed input and unsupported shell forms retain the original checks.

Execution capacity is consumed only by active commands, not cold authentication
or idle SSH connections. General work cannot consume the configured inspection
reserve; inspection can use idle general capacity. Waiting for one named shell
does not occupy another execution slot. Queue expiry means `not_started`, while
timeout/disconnect after transmission remains uncertain and is never replayed.
Status reports active/queued work, retention and owned requests. Performance logs
separate queue, connection, channel, execution, login wait/pacing, handshake and
authentication timing without logging scripts or credentials. Login pacing stays
at its existing five seconds.

For repeatable ten-worker local measurements and the inspection-reserve tradeoff,
see [the concurrency benchmark](concurrency-benchmark.md). It uses synthetic assets
and loopback SSH, not a production fleet or LLM orchestration. Reservation applies
only to the inspection lane; ordinary coordinator `run`/`jms_exec` calls remain
in the general queue. The benchmark changes no runtime defaults.

Default runs start isolated `bash -s` processes with `set -euo pipefail`.
An explicit `session_id` retains a Bash shell's directory, exports, functions, and
options on each target, with calls serialized per shell. Named shells use normal
Bash error behavior unless the script changes it. Container/image options are
bound to that session; use a different ID to change them.
Concurrent calls are serialized when they reach the shell; await dependent calls
explicitly rather than assuming parallel requests execute in submission order.

`container` uses `docker exec -i`; `image` creates a `docker run --rm -i` container
with Bash as the entrypoint. `cwd` and `env` apply inside it. `docker_args` holds
additional Docker argv elements. No nested user-provided command quoting is needed.
With a named session, an image container lives for the shell's lifetime.

| Situation | Behavior |
| --- | --- |
| Idle SSH connection closes | Next isolated run reconnects automatically. |
| Transport fails before script transmission | Opening a channel can retry once. |
| Disconnect/timeout after transmission | Save partial output, return `completion_unknown=true`; never replay. |
| Named shell exits, disconnects, or service restarts | Mark state lost; require a new ID or explicit `session_close` reset. |

`execution_state` is `not_started`, `completed`, or `unknown`. `executed=true`
confirms receipt of the start marker; `executed=null` means start itself is
uncertain. An exit code alone cannot establish reachability: gateways can reject
an asset while returning zero. Lost shell IDs remain recorded across restarts;
saved output also survives. Idle disconnects do not generate unsolicited MCP
notifications. A timeout closes the channel but does not prove remote termination.

Avoid `exit`, `exec`, redirecting the retained shell's standard streams, or unmanaged
background commands in a named shell. Use job operations for long/background work.

## Clients

Codex TOML:

```toml
[mcp_servers.jms-bridge]
url = "http://127.0.0.1:8766/mcp"
startup_timeout_sec = 30
tool_timeout_sec = 900
```

dsh Cordis patch:

```yaml
- insert:
    - id: mcp-jms-bridge
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        transport: streamable-http
        serverName: jms-bridge
        url: http://127.0.0.1:8766/mcp
        toolCallTimeoutMs: 900000
```

Claude Code (`.mcp.json` in the project, or `mcpServers` in `~/.claude.json`):

```json
{
  "mcpServers": {
    "jms-bridge": {
      "type": "http",
      "url": "http://127.0.0.1:8766/mcp"
    }
  }
}
```

For a stdio client that should own its own process instead of sharing the resident
endpoint, drop `type`/`url` and use a command — but give each such process its own
`bridge.state_dir`, because one process owns one state directory:

```json
{
  "mcpServers": {
    "jms-bridge": {
      "command": "jms-bridge",
      "args": ["serve", "--compact"]
    }
  }
}
```

All three clients negotiate MCP `2025-11-25` and see the same four tools, so a
single resident endpoint can serve them concurrently.

### Accounts must be unambiguous

The REST snapshot builds SSH targets only from assets whose account is resolved: a
`preferred_account` that the asset actually permits, or exactly one available
account. An asset with neither is listed but is **not** a target, and `run` then
fails with `no permitted SSH assets matched the targets`. Many deployments give
every asset both `root` and a second account, so without `preferred_account` every
asset drops out and the snapshot reports `0 SSH targets`. Set it in `config.toml`:

```toml
[bridge]
preferred_account = "root"
```

`discover(topic="status")` reports `ssh_targets`; if it is `0` while `assets` is
not, this is the cause.

Reload the client's MCP connection after changing the tool surface.
An optional agent skill is in `examples/skills/jumpserver-access/SKILL.md`.

## Live accelerator inspection

Discover `jms_accelerator_status`, then call it with `targets` containing 1–8 exact
REST asset names, IDs or unambiguous aliases. Groups, wildcards, `all`, and batches
larger than eight are rejected before SSH. Timeout is 15–180 seconds per host.

The read-only collector detects installed vendor tools rather than trusting an
inventory platform label. NVIDIA, Ascend, MThreads, Kunlunxin, MetaX, Enflame,
TsingMicro, T-Head, Iluvatar and Hygon adapters expose utilization, memory,
temperature, health and process ownership where available. Follow-up PID metadata
can identify the user, command and existing Docker container. Collection starts
no GPU workload and installs nothing. Command-line credential fields are redacted;
do not treat redaction as a guarantee for arbitrary application-specific secrets.

Availability requires complete, valid metrics and a recognized successful process
query: utilization below 5%, memory at most 10%, and no occupying compute process.
Unknown, malformed, failed or shared-ownership data is never reported free.
Results include every requested host, even when one fails, with full snapshots
saved for `read_result(stream="result")`. A snapshot neither reserves devices nor
authorizes stopping another process. Recheck before submitting a workload.

Inspection uses the shared SSH connections and configured inspection execution
reserve. The collector needs Python 3.9+ on the target. Parser attribution and
license are in `src/jms_bridge/inspection_vendor/`; tests use sanitized samples.

See [structured filesystem operations](filesystem.md) for action options, paging, transfer semantics and automatic channel reuse. When the client and target share storage, prefer a verified local/remote path mapping over redundant transfers; mappings must be supplied by the deployment.

## Compact responses and timing

The five top-level tools emit minified JSON text while preserving the structured
content, error behavior, metadata and whitespace inside string values. Full saved
files and byte offsets are unchanged. Shared initialization instructions carry
only cross-tool rules; usage details live in the relevant tool description and
discovered operation schema. Clients should parse JSON rather than depend on
presentation indentation. Schemas are still published with the same input shapes.

Validators for trusted generated operation schemas are compiled once per service;
validation still reports the same best-match argument error. The `jms_fs` alias
avoids redoing complete schema validation while retaining action-specific checks.

`tool_finished` log events contain tool/operation name, handler duration, response
bytes, storage wait, storage operation count and JSON compaction duration. Handler
timing excludes the outer HTTP/client/agent path. Storage wait sums concurrent
child operations and can exceed wall time. Logs contain no scripts, file contents
or credentials. Connection timing splits login queue, existing pacing, handshake
and authentication; no login policy change is introduced.

Read-only filesystem probes reuse bounded private shell channels, with a fresh
Python probe for every request. File contents are not cached, and ordinary `run`
keeps its isolated-shell default. Cold connection setup and pool saturation still
have higher latency; `read_many` and verified shared local storage avoid needless
round trips. Measure warm and cold distributions separately.

A client-side transport disconnect or temporarily missing tool schema does not
establish that remote work failed. Reconnect and refresh tool discovery; retry
read-only checks with bounded backoff. Reconcile mutations using the original
request ID before considering any new submission.
