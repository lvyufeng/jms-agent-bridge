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
    asset_id: <asset-uuid>            # from jms_list_assets, or the JumpServer UI
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

Authenticate the control plane, then check it end to end:

```bash
export JMS_BRIDGE_PASSWORD='...'   # your JumpServer password
jms-bridge login                   # caches a session token in the config (mode 600)
jms-bridge doctor                  # tooling → config → API → SSH, layer by layer
jms-bridge probe cluster           # accelerator + driver/CANN version per host
```

`login` exchanges your username/password for a token once, so the MCP server itself
never handles a password. If your deployment issues API keys instead, skip `login`
and set `JMS_BRIDGE_ACCESS_KEY_ID` / `JMS_BRIDGE_ACCESS_KEY_SECRET` or
`JMS_BRIDGE_PRIVATE_TOKEN`.

The **data plane** authenticates to the KoKo SSH gateway. Register an SSH key with
JumpServer (个人设置 → SSH 公钥) and point `ssh_key` at it — that is the recommended
setup. For a quick start before the key is registered, KoKo also accepts password
auth; set `JMS_BRIDGE_SSH_PASSWORD` and it is used via `sshpass` (installed locally,
never on the remote host):

```bash
export JMS_BRIDGE_SSH_PASSWORD='...'
jms-bridge doctor
```

## Register with Claude Code

```bash
claude mcp add jms-bridge -- jms-bridge serve
```

## Tools

18 tools, grouped by what they act on. "REST" = control plane (JumpServer API,
metadata only); "SSH" = data plane (real execution through the KoKo gateway).

| Tool | Plane | What it does |
|---|---|---|
| `jms_list_assets` | REST | Assets + accounts this account can reach (for building the inventory) |
| `jms_audit_query` | REST | Recent command records — "what did the agent run?" |
| `jms_create_ticket` | REST | Raise an approval ticket when policy gates an action |
| `jms_probe` | SSH | Liveness + accelerator + version across a host set |
| `jms_exec` | SSH | Short command on one or many hosts, in parallel |
| `jms_sync_push` / `jms_sync_pull` | rsync | Move code and artifacts across hosts |
| `jms_submit_job` | SSH | Launch a long command detached (tmux/setsid) → job id |
| `jms_job_status` / `jms_job_logs` / `jms_job_cancel` | SSH | Non-blocking job lifecycle |
| `jms_cluster_preflight` | SSH | Check every host: reachable, accelerator visible, rendezvous port free; estimates `world_size` |
| `jms_cluster_plan` | SSH | Partition targets into backend-homogeneous **islands** (HCCL/NCCL) |
| `jms_cluster_submit` | SSH | Launch **one coordinated** job across a cluster with a shared rendezvous |
| `jms_dev_discover` | SSH | Inspect each host's container and emit a paste-ready `workspaces:` block |
| `jms_dev_sync` | SSH | Deliver a git `ref` into every selected workspace's container (git / proxy-git / push) |
| `jms_dev_run` | SSH | Deliver `ref`, run the UT suite in every container, return **one cross-chip consistency table** |
| `jms_dev_logs` | SSH | Full traceback for one `test_id` on one workspace (drill into a FAIL) |

Selectors accept host names, group names, or tags — `["cluster"]`, `["gpu"]`, `["all"]`.
The `jms_dev_*` tools instead select **workspaces** by name, tag, or `all`.

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

### Heterogeneous clusters — islands and federation

Ascend (HCCL) and NVIDIA (NCCL) nodes **cannot form one collective** — different
libraries, different transports. So a mixed cluster is not one `world_size`; it is
several *islands*, one per communication backend, that **federate** (exchange model
updates between rounds) rather than share a collective.

`jms_cluster_plan` returns this layout; `jms_cluster_submit` refuses a mixed set and
points you at it:

```text
jms_cluster_plan(["cluster"])
# -> islands: [
#      {backend: "hccl", hosts: ["ascend-910c"],  world_size: 1, rendezvous: {...}},
#      {backend: "nccl", hosts: ["node-4090-1", "node-4090-2"], world_size: 2, ...},
#    ]

# then one coordinated job per island (they form their own collectives):
jms_cluster_submit(["node-4090-1", "node-4090-2"], "torchrun --nnodes=2 --node_rank=$JMS_RANK ...")
jms_cluster_submit(["ascend-910c"],             "torchrun --nnodes=1 --node_rank=0 ...")
```

The island *exchange* (all-reduce across islands, or a parameter server) is
application logic you write in the training script; the bridge gives each island its
launch and rendezvous. For a plain same-chip cluster this collapses to a single island
and a normal `jms_cluster_submit` — no federation needed.

## The flagos development loop

Training is the *slow* loop (`jms_submit_job` / `jms_cluster_submit`). Development is
the *fast* one: **write once, verify on every chip.** For a multi-backend plugin like
Torch-FL, "it works" is only true when the *same* code path passes the *same* UT suite
on every accelerator — so the loop's whole job is to make that one action and surface
any cross-chip divergence at a glance.

A **workspace** binds the four things that must agree across chips — host, container,
repo, and how to run UT — and is declared in the inventory under a `workspaces:` key:

```yaml
workspaces:
  fl-ascend:
    host: ascend-910c
    container: lvyufeng-torch-fl     # same container name on every host
    repo: /work/Torch-FL             # path *inside* the container
    source: git                      # git | proxy-git | push
    tags: [ascend]

  fl-4090:
    host: node-4090-1
    container: lvyufeng-torch-fl
    repo: /work/Torch-FL
    source: push                     # this host can't reach GitHub
    mount: /srv/torch-fl             # host dir bind-mounted to `repo`
    local: ~/Projects/Torch-FL       # local tree the bridge rsyncs from
    tags: [cuda]
```

You rarely need to write that block by hand. `jms_dev_discover` inspects a whole host
set and emits it for you — which container exists, where the checkout lives inside it,
which python/pytest is there, and whether the container can reach its git remote:

```text
jms_dev_discover(["cluster"], container="lvyufeng-torch-fl")   # -> {workspaces:{...}, yaml:"..."}
```

It returns a `workspaces:` YAML fragment to paste into `hosts.yaml` (one workspace per
host, `fl-<host>`, plus `fl-<host>-2` if a host has two checkouts), alongside the raw
per-host findings in `hosts`. The emitted `source:` is a **guess from reachability
alone**, so anything it could not confirm — a container it could not inspect, an
unreachable origin, a missing pytest — comes back as a `warnings` entry, and a host
without the container or without a matching checkout is listed in `skipped` rather than
emitted as a half-formed entry. Confirm the guess (`source`, and any `proxy` / `mount` /
`local`) before relying on it.

Once the block is in the inventory, the loop is three calls (workspaces are selected by
name, by tag, or by `all` — unlike hosts they have no group concept):

```text
jms_dev_sync(["all"], ref="feat/unify-softmax")   # deliver the ref to every workspace
# or a subset by tag:  ["ascend"]  /  ["ascend", "cuda"]
jms_dev_run(["all"],  ref="feat/unify-softmax")   # → consistency table
jms_dev_logs("fl-ascend", "tests/test_ops.py::test_layer_norm")  # full traceback on a FAIL
```

`jms_dev_run` returns one table, aligned by test id across every chip:

```text
test_id                  fl-ascend   fl-4090
test_softmax             PASS        PASS
test_layer_norm          PASS        FAIL   ← AssertionError: max diff 1e-3
test_gemm_dtype          SKIP        PASS   ← divergence: skipped on one backend
```

A row is **consistent** only when *every* workspace ran the test and they all agree; a
chip that skipped a case shows `SKIP`, not a silent omission. `ok` is true only when
nothing diverged, no chip failed or timed out, and every chip is confirmed to be at the
**same commit** (`revisions`). A FAIL carries a one-line assertion message inline — enough
to decide whether to dig; use `jms_dev_logs` for the full traceback.

Delivery is per-workspace because reachability differs (direct GitHub, behind a proxy,
or offline); all three reduce to *"make the container's repo be at `ref`"*:

| `source` | Action |
|---|---|
| `git` | in container: `git fetch origin <ref> --depth=1 && git checkout -f --detach FETCH_HEAD` |
| `proxy-git` | same, with `http_proxy`/`https_proxy` injected into the *delivery* command |
| `push` | the bridge rsyncs the local tree into the host `mount` (offline fallback) |

Two things worth knowing:

- **`ut_cmd` must emit pytest `-rA` lines.** The consistency table is parsed from
  `PASSED/FAILED/ERROR/SKIPPED … <id> - <msg>` short-summary lines; a custom `ut_cmd`
  that does not print them yields an empty table and `ok=false` (that is deliberate — an
  empty result must never read as "everything passed").
- **`push` cannot pin a revision in-container.** It has no git in the container, so its
  reported `revisions` entry is the *local* tree's HEAD; `jms_dev_run` still checks that
  every chip — push included — reported a revision, and fails the run if one did not.

The git hub is your fork: the agent pushes to `lvyufeng/torch-fl` (the `ref` the
containers fetch), and once the table is green all the way across, opens the PR to
`flagos-ai/torch-fl`. Registers/indices never cross the bastion by hand.

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