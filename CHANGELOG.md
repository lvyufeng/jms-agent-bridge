# Changelog

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
