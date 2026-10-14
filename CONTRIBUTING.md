# Contributing

## Layout

```
src/jms_bridge/
  config.py      config loading (TOML + env overrides)
  inventory.py   hosts.yaml -> Host/Inventory, selectors, remote-command wrapping
  platforms.py   accelerator profiles (ascend-910c / cuda / aarch64 / generic)
  ssh.py         SSH data plane: ControlMaster multiplexing, parallel fan-out
  jobs.py        remote job lifecycle (tmux + pidfile + logfile)
  api.py         JumpServer REST control plane (auth, discovery, audit, tickets)
  server.py      the MCP tool surface
  cli.py         init / doctor / probe / serve
tests/           unit tests (no network)
```

## Ground rules

- **Control plane vs data plane.** Discovery/audit/tickets go through the REST API;
  anything that runs on a host goes through SSH. Keep the split.
- **Never block a tool call.** Long work becomes a job (`jms_submit_job`), never a
  synchronous `jms_exec`.
- **Argv, not shell strings.** `subprocess` always receives a list; remote scripts are
  piped to `bash -s`. Nothing user-supplied may be interpreted by the local shell.
- **No PTY/menu emulation.** Only JumpServer's documented per-asset direct login.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check src tests
pytest -q
```

Integration is verified with `jms-bridge doctor` (read-only) against a real bastion,
not from CI. CI runs lint + unit tests only.

## Verifying a fresh clone

The first-run path is scripted, so a fresh clone can be *proved* to work rather than
assumed:

```bash
JMS_BRIDGE_PASSWORD='...' scripts/verify-onboarding.sh
```

It runs `init` → `login` → `jms_list_assets` → build `hosts.yaml` → `doctor` (the gate)
→ `jms_dev_discover` → one real `jms_dev_run`, and stops at the first step that does not
work, printing that step's actual output. The config, the inventory and the credentials
file are all pinned inside a scratch directory and the OS keyring is bypassed, so it never
touches `~/.config/jms-bridge`; `$HOME` is left alone because ssh still needs the real
`~/.ssh`. Exit status is 0 only when the cross-chip table comes back green.

`scripts/verify_support.py` and `scripts/mcp_call.py` are the helpers it drives: `mcp_call.py`
speaks the MCP wire protocol (there is no CLI subcommand for the `jms_*` tools), and
`verify_support.py` turns the `jms_list_assets` result into a `hosts.yaml` (`mkhosts`) and appends
a discovered `workspaces:` block to it (`discover`).
