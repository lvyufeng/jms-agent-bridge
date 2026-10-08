# jms-agent-bridge

**An MCP bridge that lets Claude Code drive a heterogeneous GPU/NPU cluster through a
[JumpServer](https://github.com/jumpserver/jumpserver) bastion — without breaking the
audit chain.**

Point it at your bastion, describe your hosts once, and a single Claude Code session can
probe the cluster, push experiment code, launch a Torch-FL run across Ascend and CUDA
nodes in parallel, watch it, and pull results back. Every byte still goes through
JumpServer's audited SSH gateway.

```
Claude Code ──stdio MCP──▶ jms-agent-bridge (local)
                              ├─ control plane: JumpServer REST API  (discovery · audit · tickets)
                              └─ data plane:    ssh/rsync via KoKo   (commands · sync · jobs)
```

## Why this exists

JumpServer's own MCP server is an auto-generated OpenAPI wrapper: no SSH data plane, no
job model, no multi-host fan-out. Community MCP servers for it are 0–1 star single-author
projects with no accelerator awareness. Meanwhile the bastion's own AI is hard-wired to a
different agent runtime.

`jms-agent-bridge` takes the two halves that actually work — **REST for the control
plane, SSH for the data plane** — and adds the three things a real cluster workflow
needs:

- **Heterogeneous hosts behind one address.** `hosts.yaml` maps a friendly name to a
  JumpServer asset plus a *platform profile* (`ascend-910c` / `cuda` / `aarch64`), so
  `npu-smi` vs `nvidia-smi` vs `uname` is the tool's problem, not the agent's.
- **Non-blocking jobs.** A training run takes hours; the gateway buffers responses for
  ~200s. Every long action is launched detached in `tmux` and polled, so no tool call
  ever blocks.
- **Parallel fan-out with audited identity.** Commands run against many hosts at once,
  each under the per-asset login string, so JumpServer attributes every one.

## Install

```bash
pip install -e .            # or: pipx install .
```

Requires `ssh` and `rsync` on PATH (the bridge shells out to them and inherits your key
agent).

## Configure

```bash
jms-bridge init             # writes ~/.config/jms-bridge/{config.toml,hosts.yaml}
$EDITOR ~/.config/jms-bridge/config.toml
$EDITOR ~/.config/jms-bridge/hosts.yaml
```

`config.toml` — the bastion and its control-plane credentials:

```toml
[bastion]
url = "https://bastion.example.com"
ssh_port = 2222                 # KoKo gateway port
login_user = "your-login"       # the login dimension of `<login>#<account>#<asset_id>`
# ssh_key = "~/.ssh/id_ed25519"
verify_tls = true
# Prefer env vars for secrets: JMS_BRIDGE_ACCESS_KEY_ID / JMS_BRIDGE_ACCESS_KEY_SECRET
# or JMS_BRIDGE_PRIVATE_TOKEN.
```

`hosts.yaml` — your cluster:

```yaml
hosts:
  ascend-910c:
    login: your-login
    account: root
    asset_id: 2b0d31bf-....          # from jms_list_assets, or the JumpServer UI
    platform: ascend-910c
    env_setup: "source /path/to/env.sh"
    workdir: /path/to/torch-fl
    tags: [gpu, ascend, training]

  node-4090-1:
    login: your-login
    account: ubuntu
    asset_id: 11111111-....
    platform: cuda
    workdir: ~/torch-fl
    tags: [gpu, cuda, training]

groups:
  cluster: [ascend-910c, node-4090-1]
```

Check it end to end:

```bash
jms-bridge doctor           # tooling → config → API → SSH, layer by layer
jms-bridge probe cluster    # accelerator + driver/CANN version per host
```

## Register with Claude Code

```bash
claude mcp add jms-bridge -- jms-bridge serve
```

## Tools

| Tool | Plane | What it does |
|---|---|---|
| `jms_list_assets` | REST | Assets + accounts this account can reach (for building the inventory) |
| `jms_audit_query` | REST | Recent command records — "what did the agent run?" |
| `jms_create_ticket` | REST | Raise an approval ticket when policy gates an action |
| `jms_probe` | SSH | Liveness + accelerator + version across a host set |
| `jms_exec` | SSH | Short command on one or many hosts, in parallel |
| `jms_sync_push` / `jms_sync_pull` | rsync | Move code and artifacts across hosts |
| `jms_submit_job` | SSH | Launch a long command detached (tmux) → job id |
| `jms_job_status` / `jms_job_logs` / `jms_job_cancel` | SSH | Non-blocking job lifecycle |
| `jms_cluster_preflight` | SSH | Check every host: reachable, accelerator visible, rendezvous port free; estimates `world_size` |
| `jms_cluster_submit` | SSH | Launch **one coordinated** job across the cluster with a shared rendezvous |

Selectors accept host names, group names, or tags — `["cluster"]`, `["gpu"]`, `["all"]`.

## A Torch-FL run, end to end

```text
jms_probe(["cluster"])                          # who is up, what accelerator, which CANN/CUDA
jms_sync_push(["cluster"], "./fl", "~/torch-fl")# same code on every node
jms_submit_job(["cluster"], "torchrun --nproc_per_node=8 train.py --rounds 100")
jms_job_status(["cluster"], "<job-id>")         # poll
jms_job_logs("ascend-910c", "<job-id>", offset=0)
jms_job_logs("ascend-910c", "<job-id>", offset=<size>)   # incremental
jms_sync_pull(["cluster"], "~/torch-fl/checkpoints", "./results")
jms_audit_query()                                # confirm the trail in JumpServer
```

### Two kinds of "parallel"

`jms_submit_job` runs the *same command independently* on each host — right for "probe
everything", "pull logs everywhere". `jms_cluster_submit` runs *one coordinated job*:
every host is a rank node given the same rendezvous, so a distributed launcher
(`torchrun`, etc.) forms a single `world_size=N` job.

```text
jms_cluster_preflight(["cluster"])              # ready? world_size estimate, port free?
jms_cluster_submit(
  ["ascend-910c", "node-4090-1", "node-4090-2"],
  "torchrun --nnodes=$JMS_WORLD_SIZE --node_rank=$JMS_RANK "
  "--nproc_per_node=$JMS_NPROC "
  "--rdzv_endpoint=$JMS_MASTER_ADDR:$JMS_MASTER_PORT train.py --rounds 100",
)
# -> {job_id per host, rendezvous, world_size, ranks:[...]}
```

Each rank gets `JMS_RANK`, `JMS_LOCAL_RANK`, `JMS_NPROC`, `JMS_WORLD_SIZE`,
`JMS_MASTER_ADDR`, `JMS_MASTER_PORT` exported before its command. The bridge supplies
the rank/world wiring (the part that must be identical across nodes); you supply the
launcher. Note: ranks talk to each other **directly** over the rendezvous, not through
the bastion — so `master_addr` must be the host's own reachable address (set `address:`
in `hosts.yaml`), and all nodes must be mutually routable.

## Design notes

- **Local, not deployed.** The server runs on your machine, launched by Claude Code. No
  install on shared bastion infrastructure; credentials stay in your key agent.
- **Control plane vs data plane.** Discovery/audit/tickets use the REST API; execution
  uses SSH. Neither alone covers the workflow — see
  [`docs/plans/2026-10-09-jms-agent-bridge-design.md`](docs/plans/2026-10-09-jms-agent-bridge-design.md).
- **Connection reuse.** `ControlMaster` means N commands cost one slow KoKo handshake.
- **Injection-safe.** Remote scripts are piped to `bash -s`; `subprocess` always gets an
  argv list, so nothing user-supplied is interpreted by the local shell.
- **No PTY menu scraping.** Only JumpServer's per-asset direct login is used — stable,
  documented behaviour, not UI simulation.

## Security

- Data plane uses SSH keys and per-asset identity, so every command is attributed in the
  audit log. The bridge never bypasses the bastion.
- **Know this about login approval:** JumpServer runs its login-review (登录复核) wait only
  for *interactive* asset sessions. A non-interactive direct connect — which is exactly
  what `jms_exec` / `jms_submit_job` use, because the remote script is piped to `bash -s`
  — **does not enter the approval queue**. Sessions and commands are still recorded in the
  audit log, but a pending login-approval ticket will not block a one-shot command. If
  policy requires approval before anything runs, gate at the account/permission layer or
  raise a `login_asset_confirm` ticket with `jms_create_ticket` and wait for it out of
  band.
- `jms_exec` / `jms_submit_job` apply a small denylist for obviously destructive commands
  (`rm -rf /`, `mkfs`, `dd if=`, fork bombs), overridable with `allow_dangerous=true`.
  This is a *typo guard*, not a security boundary — JumpServer's command blacklist and
  permission model are the real controls.
- Secrets are read from env/config and never logged.

## License

Apache-2.0.