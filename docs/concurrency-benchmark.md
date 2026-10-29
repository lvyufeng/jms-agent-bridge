# Local ten-worker concurrency benchmark

Run from the checkout with the development dependencies installed:

```sh
python scripts/benchmark_concurrency.py --rounds 3 --output /tmp/jms-concurrency-report.json
```

The output must be a **new** file: it is created exclusively with mode 0600 and is
never overwritten. Without `--output`, JSON goes to stdout. Runtime logs go to
stderr. An exception leaves a partial/empty report and exits nonzero; do not treat
such a file as valid benchmark evidence. A complete report has an `ok` gate that
checks successful workloads, unknown/not-started outcomes and resource cleanup.

## What it measures

- One shared **in-process `BridgeService`**, ten synthetic REST assets with separate
  real AsyncSSH connections, and a disposable loopback SSH server running local Bash.
- Cold login of ten assets with the **unchanged production five-second pacing**.
  The first login has no pacing wait; the remaining nine impose roughly 45 seconds,
  plus handshake/authentication and other overhead. Cold authentication does not
  consume execution slots.
- Three warm batches of ten requests with 0.3-second work and eight execution slots.
- Repeated comparisons of 0/1/2 inspection-reserved slots under general saturation.
- A 40-request burst with two reserved slots (six general slots).
- Cancelling six started requests and four queued requests, with durable terminal
  results and released execution/queue resources.
- Ten queue expirations under synthetic slot blockers: none transmits the script.
- Slow explicitly initiated REST snapshot preparation while an old stale inventory
  remains usable by status and execution calls.
- A post-transmission disconnect: partial output, unknown completion, exactly one
  local SSH command channel, and no replay.
- Closing the service and loopback server, with no remaining tracked Bash children
  or handler tasks. This local process cleanup is **not** proof that a production
  remote command always terminates when its SSH channel closes.

This is **not** an LLM agent-team experiment, an MCP HTTP/stdio transport benchmark,
a real JumpServer/KoKo benchmark, an accelerator test or distributed training.
The simulated assets all run on the same MCP host. It provides a bridge regression
baseline, not production capacity or an SLO. The bridge still does no agent
orchestration: Claude Code/dsh own the worker lifecycle and messaging.

## Metrics and controls

- Per-request `wall_seconds` ends when `service.run` returns, including ordinary
  result-store/preview work. Subsequent diagnostic metadata retrieval is separately
  recorded as `report_metadata_seconds`.
- Batch `wall_seconds` includes worker completion and report metadata collection.
- `timing` and `connection_timing` use the runner's stored execution records, with
  connection, login queue/pacing, handshake/authentication, execution queue, channel
  and execution phases kept separate. These timings are not disjoint wall-clock
  intervals and must not be subtracted to infer active-worker curves.
- `queue_sample_peak` reads internal counters every 5ms, not MCP status RPCs. It is
  approximate and sampling itself has overhead. Short peaks may be missed.
- `scheduler.peak_active` tracks granted execution slots. The independent
  `max_concurrent_children` counts currently unexited local Bash children at child
  creation, rather than trusting the scheduler's own bookkeeping. It is a local
  observed peak, not a remote hardware/process census.
- The scheduler is replaced between configurations only after active/queued work
  is empty; the local child peak resets only after the preceding children drain.
  Existing authenticated connections are retained, so warm experiments do not pay
  another cold-login round. No production setting or pacing code is changed.
- Request distributions use median p50 and nearest-rank p95. With ten requests or
  three repetitions, p95 is not a stable production percentile estimate; keep the
  per-request/round evidence rather than claiming an SLO from it.

Only the fixed local workloads are supported. `--rounds` accepts 1..20,
`--work-seconds` 0.1..2 and `--burst` 10..200. The burst's planned work must fit the
experiment's bounded execution/queue budget; invalid CLI workloads fail before
starting the listener. The report path is reserved before the cold experiment.
All private keys, host-key trust, REST asset records and state directories are
synthetic and temporary. Ambient `hosts.yaml` is isolated; no real config is loaded.

## Recorded local baseline

One completed run on macOS 26.5.2 arm64 / Python 3.14.6, using three warm/reserve
repetitions, eight slots, ten distinct assets, 0.3-second worker sleeps and a
40-request burst. Request wall ends at the `service.run` reply; batch wall includes
report collection. These numbers are observations, not acceptance thresholds.

| Case | Observed result |
| --- | --- |
| Cold ten-asset wave | 45.073s; adjacent login starts 5.005–5.007s apart |
| Warm ten-worker waves | 0.644–0.670s batch wall; observed Bash peak 8; no new logins |
| Inspection reserve 0 | Inspection wall 325–338ms; queue 302–312ms |
| Inspection reserve 1 | Inspection wall 26–63ms; queue 0ms; Bash peak 8 |
| Inspection reserve 2 | Inspection wall 16–25ms; queue 0ms; Bash peak 7 (six workers + one inspection) |
| Burst of 40 with reserve 2 | All succeeded; request p95 2.258s; queue p95 1.896s; sampled queue peak 34; Bash peak 6 |
| Cancel ten workers | Six unknown post-send + four not-started queued results; all terminal |
| Queue expiry under slot blockers | All ten not started; measured queue 51–54ms for a 50ms budget |
| Slow snapshot preparation | Status 0.041ms; one ordinary warm command 10.338ms while refresh remained pending |
| Disconnect after PARTIAL | One SSH command channel; unknown completion; no replay |
| Cleanup | Zero active/queued slots, active results, owned requests, tracked children and handlers |

Reserve 1 removed inspection queue delay in this run while leaving seven general
slots. Reserve 2 did not demonstrate a need for another reserved slot: this experiment
has only one simultaneous inspection. Worker p95 across all reserve cases ranged
roughly 652–706ms, with too few repetitions to claim a throughput improvement.
The pending wait queue is still unbounded; the burst demonstrates accumulation,
not a safe production admission limit. The raw JSON report retains per-request
records, counts and measurement scope for comparison with future runs.

## Interpreting a result

1. **Prefer one shared resident service and warm connections.** For ten distinct
   assets, cold login serialization is expected to dominate the first wave. Do not
   remove the login grace period to improve a synthetic benchmark score.
2. **Start with one inspection reserve if latency-sensitive inspections are needed.**
   With `execution_limit=8`, reserve 1 leaves seven general slots; reserve 2 leaves
   six. Compare worker throughput and actual inspection concurrency before choosing.
   Reserve capacity applies only to operations routed through the **inspection lane**
   (for example structured accelerator inspection). A main agent's ordinary `run`
   or `jms_exec` stays general work and does not get priority simply because it is
   the coordinator. This script uses a direct inspection-lane command to isolate
   scheduling from vendor probing.
3. **Keep execution and queue budgets distinct.** A queued expiry must say
   `not_started`; a post-send timeout/disconnect remains uncertain. Do not retry an
   uncertain non-idempotent command without reconciliation.
4. **Do not infer a safe queue limit from a 40-request local burst.** The current
   wait queue has no length cap. The next implementation should add explicit
   admission/backpressure with a not-started outcome, informed by controlled
   deployment measurements, rather than silently dropping work or replaying it.
5. CI uses fast loopback regressions with a shared authenticated connection to test
   the scheduler, channels and cleanup. The standalone command is the evidence for
   ten distinct paced logins; CI does not lower the production pacing or claim that
   its fast fixture measures cold fleet performance.

The checked-in code does not change `execution_limit`, `inspection_reserved`,
`queue_timeout`, disk quotas or real server configuration. Run deployment checks
only on explicitly authorized hosts and use detached job tools for long training.
