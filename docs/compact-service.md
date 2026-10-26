# Compact resident service

`jms-bridge serve --compact` exposes four tools and keeps SSH connections inside
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
```

`JMS_BRIDGE_STATE_DIR` overrides `bridge.state_dir`; otherwise `XDG_STATE_HOME`
(or `~/.local/state`) supplies the default. Keep credentials out of examples and
version control. Restart the service after changing its connection configuration.

## Tools and progressive disclosure

| Tool | Purpose |
| --- | --- |
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

## Output and shell semantics

Every execution gets private `meta.json`, `stdout.log`, and `stderr.log` files;
low-frequency responses also get a `result.json`. Previews share a default 4096
byte budget across targets and reserve space for stderr. `max_output_bytes` is
256..16384. Pages are at most 16384 bytes plus up to three bytes for UTF-8 alignment.
Non-UTF-8 bytes remain intact on disk and display with replacement characters.
Literal searches return bounded snippets and a continuation cursor, scanning at
most 8 MiB per request. Files are not automatically expired: operators manage
retention when no clients need the saved result IDs.

Default runs start isolated `bash -s` processes with `set -euo pipefail`.
An explicit `session_id` retains a Bash shell's directory, exports, functions, and
options on each target, with calls serialized per shell. Named shells use normal
Bash error behavior unless the script changes it. Container/image options are
bound to that session; use a different ID to change them.

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
