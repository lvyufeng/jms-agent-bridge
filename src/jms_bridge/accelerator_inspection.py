"""Native, bounded fleet inspection using the bridge-owned SSH transport."""

import asyncio
import csv
import io
import json
import math
import re
import time
from pathlib import Path

from .async_io import offload
from .inspection_vendor.families import FAMILIES, Proc
from .scheduling import queue_rejection

OPERATION = "jms_accelerator_status"
MAX_HOSTS = 8
POLICY = {
    "free_util_below_pct": 5,
    "free_memory_at_most_pct": 10,
    "requires_complete_metrics_and_process_query": True,
    "snapshot_only_not_a_reservation": True,
}
SCHEMA = {
    "name": OPERATION,
    "description": (
        "Read live GPU/NPU availability, utilization, memory and occupying PID/user/container/task "
        "on 1–8 explicit REST assets. Read-only; no workload, no YAML scope. Summary includes "
        'all targets; full details are saved for read_result(stream="result"). '
        "Unknown or incomplete data is never free."
    ),
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "targets": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_HOSTS,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
                "description": (
                    "Exact asset names, IDs, IPs, or unambiguous aliases. "
                    "No all/wildcards/groups; discover assets first."
                ),
            },
            "timeout": {
                "type": "integer",
                "minimum": 15,
                "maximum": 180,
                "default": 90,
                "description": "Per-host collection timeout in seconds.",
            },
        },
        "required": ["targets"],
    },
    "annotations": {
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
}


def remote_script(options=None):
    source = Path(__file__).with_name("inspection_probe.py").read_text()
    options = json.dumps(options or {})
    return (
        "python3 -I - <<'__JMS_INSPECTION_PY__'\n"
        + source
        + "\nmain(json.loads("
        + repr(options)
        + "))\n__JMS_INSPECTION_PY__\n"
    )


def device_state(card, complete, process_known, shared_processes=False):
    if card.health and card.health.upper() not in ("OK", "NORMAL", "GOOD"):
        return "unhealthy"
    if card.procs and not card.procs_shared:
        return "busy"
    if card.util is not None and math.isfinite(card.util) and card.util >= 5:
        return "busy"
    memory_valid = (
        card.mem_used is not None
        and card.mem_total is not None
        and math.isfinite(card.mem_used)
        and math.isfinite(card.mem_total)
        and 0 <= card.mem_used <= card.mem_total
        and card.mem_total > 0
    )
    if memory_valid and card.mem_used / card.mem_total > 0.10:
        return "memory_held"
    if (
        not complete
        or not process_known
        or shared_processes
        or not memory_valid
        or card.util is None
        or not math.isfinite(card.util)
        or not 0 <= card.util < 5
    ):
        return "unknown"
    return "free"


def process_table_known(family, text, raw):
    if family == "nvidia":
        return (
            len(raw.get("queries", [])) >= 2
            and not raw["queries"][1].get("error")
            and raw["queries"][1].get("rc") == 0
        )
    if family == "hygon":
        return "##HYGON-PROCS" in text and not raw.get("errors")
    return bool(
        re.search(
            r"(?i)(process(?:es)?\s*:|process\s+(?:id|name)|\bPID\b|no running processes)",
            text,
        )
    )


def normalize_bus(value):
    parts = value.lower().split(":")
    return ":".join([parts[0].lstrip("0") or "0", *parts[1:]])


def fix_nvidia_processes(cards, raw):
    """Keep a PID even when its memory is N/A; never silently drop unknown rows."""
    queries = raw.get("queries", [])
    if len(queries) < 2:
        return ["missing NVIDIA process query"]
    buses, errors = {}, []
    gpu_rows = [
        r for r in csv.reader(io.StringIO(queries[0]["stdout"])) if any(x.strip() for x in r)
    ]
    if len(gpu_rows) != len(cards) or len({c.index for c in cards}) != len(cards):
        errors.append("NVIDIA device table incomplete or malformed")
    for row in gpu_rows:
        if len(row) >= 8 and row[0].strip().isdigit():
            buses[normalize_bus(row[-1].strip())] = int(row[0])
    by_id = {c.index: c for c in cards}
    for card in cards:
        card.procs = []
    for row in csv.reader(io.StringIO(queries[1]["stdout"])):
        if not row or not any(x.strip() for x in row):
            continue
        if len(row) < 4 or not row[1].strip().isdigit():
            errors.append("unparsed NVIDIA process row")
            continue
        owner = by_id.get(buses.get(normalize_bus(row[0].strip())))
        if owner is None:
            errors.append("NVIDIA process device could not be mapped")
            continue
        try:
            mem = float(row[-1].strip().split()[0])
        except (ValueError, IndexError):
            mem = None
        owner.procs.append(
            Proc(
                pid=int(row[1]),
                user=None,
                name=",".join(row[2:-1]).strip(),
                mem_mib=mem,
            )
        )
    return errors


def parse_snapshot(snapshot):
    devices, processes, errors = [], {}, []
    for key, raw in snapshot.get("vendors", {}).items():
        if key not in FAMILIES:
            errors.append(key + ": unsupported accelerator tool")
            continue
        text = raw.get("text", "")
        problems = list(raw.get("errors", []))
        try:
            parsed = FAMILIES[key].parse(text)
        except (ValueError, IndexError, TypeError, KeyError) as exc:
            errors.append(key + ": parser error " + type(exc).__name__)
            continue
        if parsed.error or not parsed.cards:
            errors.append(key + ": " + (parsed.error or "no devices parsed"))
            continue
        if key == "nvidia":
            problems.extend(fix_nvidia_processes(parsed.cards, raw))
        known = process_table_known(key, text, raw)
        if not known:
            problems.append("process table unavailable or unrecognized")
        errors.extend(key + ": " + p for p in problems)
        shared = any(c.procs_shared and c.procs for c in parsed.cards)
        for card in parsed.cards:
            device_id = key + ":" + str(card.index)
            state = device_state(card, not problems, known, shared)
            devices.append(
                {
                    "id": device_id,
                    "family": key,
                    "index": card.index,
                    "physical_card": card.card if card.card is not None else card.index,
                    "model": card.name,
                    "state": state,
                    "util_pct": card.util,
                    "memory_used_mib": card.mem_used,
                    "memory_total_mib": card.mem_total,
                    "temperature_c": card.temp,
                    "power_w": card.power,
                    "health": card.health,
                    "process_table_known": known,
                    "process_scope": "host" if shared else "device",
                    "pids": []
                    if card.procs_shared
                    else sorted({p.pid for p in card.procs if p.pid is not None}),
                }
            )
            for p in card.procs:
                if p.pid is None:
                    continue
                row = processes.setdefault(
                    p.pid,
                    {
                        "pid": p.pid,
                        "user": p.user,
                        "name": p.name,
                        "devices": [],
                        "memory_by_device_mib": {},
                        "scope": "host" if card.procs_shared else "device",
                    },
                )
                if not card.procs_shared:
                    if device_id not in row["devices"]:
                        row["devices"].append(device_id)
                    row["memory_by_device_mib"][device_id] = p.mem_mib
    if not devices:
        state = "unknown" if snapshot.get("vendors") else "no_accelerator_detected"
    elif errors:
        state = "partial" if any(d["state"] == "free" for d in devices) else "unknown"
    elif all(d["state"] == "free" for d in devices):
        state = "free"
    elif any(d["state"] == "free" for d in devices):
        state = "partial"
    elif any(d["state"] == "busy" for d in devices):
        state = "busy"
    elif any(d["state"] == "unhealthy" for d in devices):
        state = "unhealthy"
    elif any(d["state"] == "memory_held" for d in devices):
        state = "memory_held"
    else:
        state = "unknown"
    return {
        "state": state,
        "devices": devices,
        "processes": list(processes.values()),
        "errors": errors,
        "node": snapshot.get("node", {}),
        "sampled_at": snapshot.get("finished_at"),
        "whole_node_available": bool(devices) and state == "free",
    }


def compact_host(host):
    devices = host.get("devices", [])
    groups = {}
    for p in host.get("processes", []):
        container = p.get("container", {}).get("name")
        command = p.get("command") or p.get("name") or "?"
        key = (p.get("user"), container, command)
        group = groups.setdefault(
            key,
            {
                "user": p.get("user"),
                "container": container,
                "command": command[:120],
                "pids": [],
                "devices": [],
            },
        )
        group["pids"].append(p["pid"])
        group["devices"] = sorted(set(group["devices"]) | set(p.get("devices", [])))
        if p.get("scope") == "host":
            group["scope"] = "host; device mapping unknown"
    row = {
        k: host.get(k)
        for k in (
            "host",
            "address",
            "state",
            "sampled_at",
            "collected_at",
            "whole_node_available",
            "result_id",
            "timing",
            "sample_received_at",
        )
    }
    row.update(
        models=sorted({d["model"] for d in devices}),
        free_devices=[d["id"] for d in devices if d["state"] == "free"],
        devices=[
            [
                d[k]
                for k in (
                    "id",
                    "state",
                    "util_pct",
                    "memory_used_mib",
                    "memory_total_mib",
                )
            ]
            for d in devices
        ],
        occupants=list(groups.values())[:4],
        process_count=len(host.get("processes", [])),
        more_occupant_groups=max(0, len(groups) - 4),
    )
    if host.get("errors"):
        row["errors"] = host["errors"][:4]
    if host.get("warnings"):
        row["warnings"] = host["warnings"][:3]
    if host.get("backpressure"):
        row["backpressure"] = host["backpressure"]
    return row


class AcceleratorInspector:
    def __init__(self, catalog, runner, store):
        self.catalog, self.runner, self.store = catalog, runner, store

    async def inspect(self, targets, timeout=90):
        if not isinstance(targets, list) or not 1 <= len(targets) <= MAX_HOSTS:
            raise ValueError("inspection requires 1–8 explicit targets; no automatic batching")
        if type(timeout) is not int or not 15 <= timeout <= 180:
            raise ValueError("inspection timeout must be an integer in 15..180 seconds")
        for target in targets:
            if (
                not isinstance(target, str)
                or not target
                or target.lower() in ("all", "*")
                or any(c in target for c in "*?[]")
            ):
                raise ValueError(
                    "inspection requires exact asset names, IDs or IPs; no wildcard/all selectors"
                )
        inventory = await self.catalog.ensure()
        hosts = {}
        for target in targets:
            resolved = inventory.resolve([target])
            if len(resolved) != 1:
                raise ValueError("each inspection target must resolve to exactly one machine")
            hosts[resolved[0].asset_id] = resolved[0]
        if len(hosts) > MAX_HOSTS:
            raise ValueError("inspection resolved to more than 8 machines")
        # Inspection uses reserved execution capacity and the shared authenticated connections.
        results = await asyncio.gather(*(self.one(h, timeout) for h in hosts.values()))
        value = {
            "ok": all(not r.get("errors") for r in results),
            "policy": POLICY,
            "inventory_stale": self.catalog.status()["stale"],
            "hosts": results,
        }
        result_id = await offload(self.store.save_json, value, kind="call", operation=OPERATION)
        return {
            "ok": value["ok"],
            "result_id": result_id,
            "policy": POLICY,
            "inventory_stale": value["inventory_stale"],
            "device_columns": [
                "id",
                "state",
                "util_pct",
                "memory_used_mib",
                "memory_total_mib",
            ],
            "hosts": [compact_host(r) for r in results],
            "details": (
                'read_result(result_id, stream="result"); each host has its own detail result_id'
            ),
        }

    async def execute_json(self, host, options, timeout):
        result = await self.runner.execute_to_store(
            host, remote_script(options), timeout, lane="inspection"
        )
        if not result.get("ok"):
            return None, result
        try:
            path = self.store.path(result["result_id"], "stdout")
            if (await offload(path.stat)).st_size > 8 * 1024 * 1024:
                raise ValueError("collector output too large")
            data = json.loads(await offload(path.read_text))
            if not isinstance(data, dict):
                raise TypeError("invalid collector output")
            return data, result
        except (ValueError, TypeError, OSError):
            return None, dict(result, parse_error=True)

    async def one(self, host, timeout):
        row = {
            "host": host.name,
            "address": host.address,
            "asset_id": host.asset_id,
            "state": "unknown",
            "whole_node_available": False,
            "sampled_at": None,
            "devices": [],
            "processes": [],
            "errors": [],
            "warnings": [],
            "execution_results": [],
            "timing": {},
            "sample_received_at": None,
        }

        def record(result):
            row["execution_results"].append(result["result_id"])
            if rejection := queue_rejection(result):
                row.setdefault("backpressure", []).append(
                    {"result_id": result["result_id"], **rejection}
                )
            for key, value in result.get("timing", {}).items():
                row["timing"][key] = round(row["timing"].get(key, 0) + value, 3)

        try:
            snapshot, result = await self.execute_json(host, {}, timeout)
            record(result)
            if snapshot is None:
                row["state"] = (
                    "unknown"
                    if result.get("reason") == "queue_full"
                    else "down"
                    if result.get("connection_state") != "connected"
                    else "unknown"
                )
                row["errors"].append(
                    "collection failed; inspect execution result " + result["result_id"]
                )
            else:
                row["sample_received_at"] = time.time()
                row.update(parse_snapshot(snapshot))
                pids = sorted({p["pid"] for p in row["processes"]})
                if pids:
                    details, detail_result = await self.execute_json(
                        host,
                        {
                            "mode": "processes",
                            "pids": pids[:128],
                            "sampled_at": snapshot["finished_at"],
                        },
                        min(timeout, 45),
                    )
                    record(detail_result)
                    if details:
                        by_pid = {p["pid"]: p for p in details["processes"]}
                        for p in row["processes"]:
                            p.update(by_pid.get(p["pid"], {"detail_state": "not_collected"}))
                        if details.get("docker_error"):
                            row["warnings"].append(details["docker_error"])
                        row["process_details_observed_at"] = details["observed_at"]
                    else:
                        row["warnings"].append("process enrichment failed; PID occupancy retained")
                    if len(pids) > 128:
                        row["warnings"].append(
                            "process metadata limited to 128 PIDs; full occupancy retained"
                        )
        except Exception as exc:  # noqa: BLE001 -- preserve other hosts on unexpected per-host failures
            row["errors"].append("inspection failed: " + type(exc).__name__)
        row["collected_at"] = time.time()
        row["result_id"] = await offload(
            self.store.save_json, row, kind="inspection", host=host.name
        )
        return row
