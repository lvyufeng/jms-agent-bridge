# jms-agent-bridge — Design

**Date:** 2026-10-09
**Status:** implemented and verified against a live JumpServer (v4.10.18-ce)
**Author:** lvyufeng

## Problem

I run heterogeneous compute hosts — Huawei Ascend 910C (openEuler, CANN), NVIDIA
4090 nodes (Ubuntu/CUDA), an aarch64 board — reached only through an open-source
**JumpServer** bastion (`https://bastion.example.com`). The bastion is the
audited, permission-gated entry point; there is no general-purpose SSH jump.

I want a Claude Code session to treat that whole cluster as one workspace:
probe liveness, push experiment code, launch **Torch-FL** jobs on the right
accelerator per node, watch them, and collect results — all without bypassing the
bastion's audit chain and without me hand-driving N SSH sessions.

Existing options are inadequate:

- `github.com/jumpserver/mcp` — **official**, but it is a thin auto-generated
  OpenAPI→MCP wrapper: no SSH data plane, no job model, no multi-host fan-out, and
  it is effectively unmaintained.
- Community MCP servers — all 0–1 star single-author projects; the ones that shell
  out do it ad hoc with no accelerator awareness and no audit discipline.
- The bastion's own AI (Kael / skills) is hard-wired to OpenAI Codex and unrelated
  to Claude Code.

## Insight

**Control plane over REST, data plane over SSH.**

- JumpServer's REST API is stable, audited, and the right place for *discovery*
  (assets, accounts, permission), *audit queries*, and *tickets*.
- Actually running commands, moving files, and launching training jobs is **not**
  expressible over the REST API — that needs the SSH transport. On this deployment
  the KoKo gateway supports **per-asset direct login**
  (`<login>#<account>#<asset_id>@<koko_host> -p 2224`) landing straight on the asset
  with no interactive menu, so it *does* behave as a jump host for a client that can
  construct those login strings.

Neither half alone is sufficient; together they cover the whole workflow.

## Architecture

```
Claude Code  ──stdio MCP──▶  jms-agent-bridge (runs on my macOS machine)
                                 │
                                 ├── control plane: JumpServer REST API  (/api/v1)
                                 │     auth · asset/account discovery · audit · tickets
                                 │
                                 └── data plane: ssh/rsync through KoKo (:2224)
                                       ControlMaster-multiplexed · parallel fan-out
                                       remote job = tmux + pidfile + logfile
```

The MCP server runs **locally** (launched by Claude Code as a stdio subprocess). It
never gets deployed to the bastion; it drives the local `ssh`/`rsync` binaries,
which inherit the user's existing key agent and bastion knowledge.

### Why not run it on the bastion

The bastion is shared, audited infrastructure. A local server means: no install
politics, credentials stay in my keychain, and the tool is useful for anyone
pointing it at any JumpServer.

## Components

| Module | Responsibility |
|---|---|
| `config.py` | Load `~/.config/jms-bridge/config.toml` (bastion URL, ssh port, auth mode, ssh key path) |
| `inventory.py` | Load `hosts.yaml`: logical host → ssh target + `platform` + `env_setup` + `workdir` |
| `platforms.py` | Accelerator profiles: `ascend-910c`, `cuda`, `aarch64`, `generic` — probe commands, env activation, distributed-launch templates |
| `ssh.py` | SSH/rsync transport: ControlMaster multiplexing, argv-safe command building, bounded parallelism, per-host result capture |
| `jobs.py` | Remote job model: `submit` (tmux + pidfile + log), `status`, `logs`, `cancel` |
| `api.py` | JumpServer REST client (auth, assets, accounts, audits, tickets) |
| `server.py` | FastMCP server exposing the tool surface |
| `cli.py` | `jms-bridge init` / `probe` / `doctor` for onboarding and debugging |

## Tool surface (v1)

| Tool | Plane | Purpose |
|---|---|---|
| `jms_list_assets` | REST | Discover assets + accounts + permission from JumpServer |
| `jms_probe` | SSH | Liveness + accelerator + driver/CANN version across a host set |
| `jms_exec` | SSH | Short synchronous command on one or many hosts (bounded timeout) |
| `jms_sync_push` / `jms_sync_pull` | rsync | Push/pull code + artifacts across hosts |
| `jms_submit_job` | SSH | Launch a long-running command (training) detached, return a job id |
| `jms_job_status` / `jms_job_logs` / `jms_job_cancel` | SSH | Non-blocking lifecycle for jobs |
| `jms_audit_query` | REST | Recent command/session records — answers "what did the agent do" |
| `jms_create_ticket` | REST | Raise an approval ticket when policy requires one |

## The job model (the crux)

A Torch-FL run lasts hours. The octopus gateway on these hosts has been measured at
**~200s TTFB** for non-streaming replies, and KoKo handshakes are slow. Therefore an
MCP tool call **must never block** on a long command.

Every long action is: write a small runner script, launch it inside a **tmux**
session (survives disconnect), record a pidfile and a logfile under a per-job
directory on the remote host, and return a `job_id` immediately. `status` greps the
pidfile + reads a status file; `logs` tails the logfile with a byte offset so the
agent can poll incrementally. This is what makes "start FL training on 5 hosts and
watch them" tractable over a slow, lossy, audited link.

## Platform adapters

Each host declares a `platform`; the adapter supplies:

- **probe**: the accelerator status command (`npu-smi info` vs `nvidia-smi`) and how
  to read identity/versions from it.
- **env**: the snippet that activates the toolchain (`source env.sh`, a conda env,
  CUDA env vars).
- **distributed**: how to launch a multi-worker run on that platform
  (`torchrun` + HCCL for Ascend, `torchrun` + NCCL for CUDA).
- **sync**: rsync include/exclude quirks (e.g. never drag a local venv onto a board).

Target hosts: `ascend-910c`, `cuda`, `aarch64`, `generic`.

## Non-goals (v1)

- **No PTY / KoKo-menu emulation.** The per-asset direct login removes the need;
  scraping the interactive menu is fragile and stays out.
- **No bypass of the audit chain.** All data-plane traffic goes through KoKo.
- **No secrets in the repo.** Auth via SSH key + JumpServer access key, both loaded
  from the local config/keychain, never committed.
- **No scheduler / orchestration engine.** We are the thin bridge; the agent does
  the planning.

## Security

- Data plane uses SSH keys (not the rotating password) and **per-asset identity** in
  the login string, so every command is attributed in JumpServer's audit log.
- The REST client stores the access key id/secret and signs requests (HMAC); secrets
  are read from env/config, never logged.
- Command construction is **argv-safe** (no shell-string interpolation from tool
  arguments into a remote shell), so an asset name can't inject a command.
- A `denylist` of obviously destructive command patterns is applied to `jms_exec`
  with an explicit override flag, and every invocation is surfaced for the agent to
  confirm — the bastion's own blacklist remains the backstop.

## Testing

- Unit tests for: inventory parsing, argv construction (injection cases), platform
  probe parsing (fixture `npu-smi`/`nvidia-smi` output), job status parsing, HMAC
  signing against a known vector.
- A `doctor` command that runs the real path against the bastion read-only
  (`jms_list_assets`, one `jms_probe`) so integration is verified without launching
  anything expensive.
- CI: lint + unit tests on every push; no network in CI.