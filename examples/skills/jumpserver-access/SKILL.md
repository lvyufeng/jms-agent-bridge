---
name: jumpserver-access
description: Run scripts, remote Docker commands, jobs, accelerator occupancy, and file transfers on JumpServer assets through the compact jms-agent-bridge MCP service.
---

# JumpServer access

Use `fs`, `run`, `terminal`, `read_result`, `discover`, `call` and `meta`.
Client-specific prefixes may differ.

- For repeated commands on one target, open a `terminal` bound to that target,
  then use its handle for exec/read/write and close it when done. Discover
  `terminal_<action>` for options. Running is not completion; after shell loss,
  inspect saved output and request receipts before taking further action.
- Routine replies are compact; use `meta` for diagnostics or `detail="full"`
  for the previous response shape. `fs` accepts advanced fields in `options`;
  `discover(operation="jms_fs", query="read")` shows a single action's fields.

- Find authorized targets with `discover(topic="assets", query="<name or address>")`.
  Use an exact returned name or UUID. REST refreshes hourly; YAML does not grant
  target access. Do not fan out to `all` unless the task calls for it.
- Use `fs` for `list`, `stat`, `read`, `read_many`, `glob`, `search`, `tree`,
  `checksum`, `upload` and `download`. Discover `jms_fs` for its schema. Paths and
  patterns are data; no shell quoting is needed. Use `read_many` for several
  known files, byte cursors for text and row offsets for search/directory pages.
  Exploration uses Bash 4.2+ and GNU tools on the target, without target Python.
  Search prefers rg; without it, GNU grep uses POSIX ERE and reports missing ignore rules.
  An optional existing `container` changes the remote view. Upload/download write
  their destination; `local_path` refers to the MCP server machine. Reconcile
  transfers with the original `request_id` after a lost reply.
  If storage is shared, first verify the deployment's local/remote path mapping
  and use local access instead of transmitting the same files through the gateway.
- Pass literal Bash as `run(targets=[...], script="...")`, or use an absolute local
  `script_path`. Use `cwd`, `env`, `container` or `image` instead of nested quoting.
  `docker_args` is an argv list. Do not encode the script or add shell quote layers.
- Output is saved completely. Start with the compact preview, then use
  `read_result` with `stream`, byte `offset`/`limit`, `tail`, or a literal `pattern`.
  Continue from `next_offset`; clipped previews are not complete logs.
- For important runs and job submission, generate a unique `request_id` before
  calling. After a lost reply, discover/call `request_status` with that original
  ID; do not invent another ID or submit again while completion is unknown.
  Repeated IDs never replay work. Saved output expires after the configured TTL
  (default 7 days); receipts remain and reads do not renew retention.
- `allow_dangerous` acknowledges only the local pattern guard for a specific
  action already authorized by the user. It does not grant approval or override
  client review or JumpServer policy. `script_guard_check` diagnoses that guard
  without executing anything. Do not change channels to retry an external denial.
- Discover a category such as `jobs` or `files`, then request one operation's
  schema with `discover(operation="...")` before invoking `call`. Large operation
  responses are read using `read_result(stream="result")`; `execution_results`
  references retain the underlying command logs.
- Default executions have isolated shell state and reuse the connection.
  Use a task-specific `session_id` only when `cd`, exports, or functions must
  survive. Close task-owned shells through the discovered `session_close` operation.
- The service reconnects idle connections and may retry opening a channel before
  script transmission. A failure after transmission is never replayed automatically.
  If `completion_unknown=true`, inspect saved output and remote state before
  deciding whether to repeat a command. `executed=null` means even its start is uncertain.
- A temporary client transport error or missing tool schema is not proof that
  the remote operation failed. Refresh the MCP connection/discovery and retry
  read-only checks with bounded backoff. Keep tracking accepted work; reconcile
  mutations by their original request ID before making another submission.
- A lost named shell is reported explicitly, including after a service restart.
  Use a new ID and reapply setup, or explicitly close/reset the previous ID. Do not
  assume reconnecting restores shell state. Avoid changing the shell's standard
  streams, `exit`/`exec`, and unmanaged background commands in retained shells.
- To find available GPUs/NPUs, discover `jms_accelerator_status`, then call it
  with at most 8 exact REST targets. It returns utilization, memory, PID/user,
  container and task summaries; full details are saved. Unknown/incomplete data
  is never free. A snapshot is not a reservation; recheck before launching work.
  Split a larger fleet explicitly rather than requesting `all` or wildcards.
- For background or long work, discover job operations. `run` is bounded to 600
  seconds. A timeout does not prove that a remote process stopped.
- `discover(topic="status")` reports connection and inventory state. A stale REST
  snapshot is not current permission or hardware-health evidence. The gateway
  enforces permissions; a rejected operation should not be retried through another transport.

Apply the user's authorized scope to remote changes and transfers. Output may
contain secrets: retrieve relevant excerpts instead of copying entire logs into chat.
