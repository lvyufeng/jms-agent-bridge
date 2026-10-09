# flagos dev workflow — Design

**Date:** 2026-10-09
**Status:** approved
**Author:** lvyufeng

## Problem

The goal is to develop **Torch-FL** (`flagos-ai/Torch-FL`) — a PyTorch plugin that
hides vendor differences behind one logical device (`flagos`) — from a **single Claude
Code session** across heterogeneous accelerator hosts (Ascend 910C, NVIDIA 4090,
MetaX, …) reached through the JumpServer bastion.

Torch-FL's first design principle is that callers write *one* code path: no
`if is_npu: ... else: ...` at the call site. So development of the plugin means one
implementation must be proven *identical in behavior* on every backend. Today that
proof is manual and fragmented: a separate Claude Code lives on each host inside a
docker container, and there is no single view of "does this operator work everywhere
the same way?"

The previous workflow looked like this:

```
per host: ssh → docker container → claude code → edit → run UT   (N independent agents)
```

The desired workflow:

```
one claude code ──▶ jms-agent-bridge ──▶ all containers ──▶ one consistency table
```

## Goal and acceptance

Make "write once, verify on every chip" a **single action**, and surface cross-chip
divergence at a glance.

Acceptance (decided during brainstorming):

1. The same code path runs on every backend (the plugin's API surface is uniform).
2. The same UT suite runs on every backend and passes.
3. *(Deferred)* Numerical equivalence across backends — not required yet; the table is
   PASS/FAIL, not a value comparison.

Non-goals for this iteration:

- Long training runs stay on the **existing** async job model (`jms_submit_job` /
  `jms_cluster_submit`); this workflow is the *development* loop, not the *training*
  loop. Keeping them separate keeps the dev loop a fast synchronous call.
- Multi-machine collective collectives for UT — UT runs per host independently.

## Current state (what already exists)

jms-agent-bridge exposes 14 MCP primitives over two planes:

- **Control plane** (JumpServer REST): `jms_list_assets`, `jms_audit_query`,
  `jms_create_ticket`.
- **Data plane** (SSH through KoKo): `jms_probe`, `jms_exec`, `jms_sync_push/pull`,
  `jms_submit_job` + `jms_job_status/logs/cancel`, `jms_cluster_preflight/plan/submit`.

Both planes are live against the real bastion (v4.10.18-ce). What is missing for this
workflow: the data plane reaches the **host**, but development happens inside a
**container** on that host; and there is no notion of "a workspace = host + container +
repo + how to run UT".

## Design

### 1. Workspace — the core abstraction

A **Workspace** is the unit the agent addresses. It binds four things that must agree
across hosts for the workflow to be uniform:

```
Workspace = host (through the bastion) + container (same name everywhere)
          + flagos repo + how to run UT
```

Inventory (`hosts.yaml`) gains a `workspaces:` section:

```yaml
workspaces:
  ascend-910c-108:
    host: ascend-910c-108          # an existing host entry (bastion asset)
    container: torchfl             # container name — uniform across hosts
    source: git                    # git | proxy-git | push
    repo: /work/Torch-FL           # repo path *inside the container*
    proxy: http://proxy:8080       # only for source=proxy-git
    ut_cmd: "python -m pytest tests/ -q --tb=line -rA"
    tags: [ascend, hccl]

  cuda-4090-1:
    host: cuda-4090-1
    container: torchfl
    source: push                   # this host has no GitHub access
    repo: /work/Torch-FL
    mount: /srv/torch-fl           # host dir bind-mounted to repo (for source=push)
    ut_cmd: "python -m pytest tests/ -q --tb=line -rA"
    tags: [cuda, nccl]
```

All commands in a workspace are wrapped as
`docker exec <container> bash -lc '<cmd>'` — so every existing capability
(`jms_exec`, `jms_submit_job`) gains a "run inside the container" mode without
changing its shape.

### 2. Delivery — three sources, one verb

The git topology (decided during brainstorming):

```
Mac agent ──push──▶ lvyufeng/torch-fl (fork)
                        │  fetch (container, no creds needed for a public fork)
                        ▼
              each container ── runs UT ──▶ all green ──▶ PR to flagos-ai/torch-fl
```

`ref` is the branch the agent pushed to the **fork**. The fork is read-only from the
clusters' perspective; the upstream repo is only ever the PR target.

Delivery is per-host configurable because network reachability differs (some hosts
reach GitHub directly, some need a proxy, some are offline). All three reduce to one
semantic — *"make the container's repo be at `ref`"*:

| `source` | Action |
|---|---|
| `git` | in container: `git fetch origin && git checkout <ref>` |
| `proxy-git` | same, with `http_proxy`/`https_proxy` injected into the command |
| `push` | bridge rsyncs from the local checkout to the host `mount` dir (offline fallback) |

The agent sees one verb:

```
jms_dev_sync(["flagos-cluster"], ref="feat/unify-softmax")
```

### 3. Run + aggregate — one suite, one table

`jms_dev_run` is **synchronous** (the dev loop must be low-friction): deliver the
current ref, run `ut_cmd` in each container, parse per-test outcomes, return one table.

Each host runs pytest in a machine-readable mode (`-rA` / `--tb=line`, optionally
`--json-report`); the bridge reports `{test_id: outcome}` per host and aligns by
`test_id` (union — a case a chip skipped is shown as such, not silently dropped).

```
test_id                  ascend-910c  cuda-4090  maca-ppu
test_softmax             PASS         PASS       PASS
test_layer_norm          PASS         FAIL       PASS   ← assertion msg inline
test_gemm_dtype          SKIP         PASS       PASS   ← divergence
```

**Failure info:** each FAIL carries a one-line summary inline (the `--tb=line` first
line / assertion message) — enough for the agent to decide whether to dig. Full
tracebacks are fetched on demand.

### 4. Drill-down

`jms_dev_logs(target, test_id)` returns the full output for one test on one workspace.
pytest leaves per-test output on disk in the container (e.g. via `--junitxml` or a
saved `-rA` report); the bridge slices out the requested test by name.

### 5. Training stays separate

Long runs keep using the existing async job model (`jms_cluster_submit` with rank /
world_size / rendezvous). `jms_dev_run` is deliberately **not** built on the job model
— mixing them would force UT through a poll loop and add friction to the 90% case.

## Tool surface (new)

| Tool | What it does |
|---|---|
| `jms_dev_sync` | Deliver a `ref` to every selected workspace (git / proxy-git / push) |
| `jms_dev_run` | Sync + run UT in every container + return the consistency table |
| `jms_dev_logs` | Full output for one `test_id` on one workspace |

Existing tools remain; `jms_exec` / `jms_submit_job` gain container-awareness through
the workspace's `container` field.

## Open questions / risks

- **UT duration.** "Synchronous" assumes the suite finishes fast enough to survive the
  ~200s KoKo TTFB on some paths. If a full suite is too slow, `jms_dev_run` needs a
  partial-result poll mode. Decide after measuring a real suite on the 910C.
- **Container reachability.** The host must be able to `docker exec` as the login user
  (root on these assets) — confirm on each chip family.
- **Runtime build per chip.** Torch-FL has per-vendor build/select paths; if the
  container's installed `torch_fl` is built from the repo, "checkout ref" must be
  followed by a rebuild. Whether UT runs against an editable install or a built
  extension is a per-host `ut_cmd` concern, not a bridge concern — but call it out.