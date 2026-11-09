"""Small response views; execution, receipts and stored results stay canonical.

Only known facade envelopes are reduced. Business payloads and errors from
unrecognized operations are left intact. No shared per-client view/target state.
"""

from copy import deepcopy


def _identity(row, addresses):
    out = dict(row)
    host = out.get("host")
    if host in addresses:
        out["host"] = addresses[host]
    return out


def _execution(row, addresses):
    # Keep stable result/host/status keys used by clients. Diagnostic detail is
    # still present in result metadata, not duplicated in every successful call.
    keys = (
        "host",
        "result_id",
        "ok",
        "exit_code",
        "execution_state",
        "stdout",
        "stderr",
        "stdout_truncated",
        "stderr_truncated",
        "warning",
        "error",
        "message",
    )
    if not row.get("ok", False):
        return _identity(row, addresses)
    out = {k: row[k] for k in keys if k in row}
    if row.get("execution_state") != "completed" or row.get("completion_unknown"):
        for k in (
            "executed",
            "completion_unknown",
            "timed_out",
            "queue_timed_out",
            "failure_stage",
            "connection_state",
        ):
            if k in row:
                out[k] = row[k]
    if row.get("session_state") == "lost":
        out.update(session_id=row.get("session_id"), session_state="lost")
    if row.get("connection_recovery") or row.get("reconnected"):
        out["reconnected"] = True
    return _identity(out, addresses)


def _receipt_flags(value, out):
    if value.get("request_state") not in (None, "complete") or value.get("completion_unknown"):
        for k in (
            "request_id",
            "request_state",
            "completion_unknown",
            "error",
            "execution_results",
            "submission_receipts",
        ):
            if k in value:
                out[k] = deepcopy(value[k])
    if value.get("duplicate"):
        out.update(request_id=value["request_id"], duplicate=True, execution_replayed=False)
    for key in ("execution_results", "submission_receipts"):
        rows = value.get(key, [])
        if not isinstance(rows, list):
            continue
        if any(
            row.get("reason") == "queue_full"
            or row.get("completion_unknown")
            or row.get("execution_state") in ("unknown", "not_started")
            or row.get("ok") is False
            for row in rows
            if isinstance(row, dict)
        ):
            out[key] = deepcopy(rows)
    return out


def _file_page(value):
    # Continuation, partial lines, concurrent modification and failures remain
    # visible. stat/search/list/checksum payloads are not generic field-pruned.
    if not isinstance(value, dict):
        return value
    out = dict(value)
    if "text" in value and "next_byte_offset" in value:
        for k in ("backend", "elapsed_seconds", "encoding", "decoding_errors"):
            out.pop(k, None)
        if value.get("eof") and not value.get("changed_during_read"):
            out.pop("fingerprint", None)
        for k in ("partial_first_line", "partial_last_line", "changed_during_read"):
            if value.get(k) is False:
                out.pop(k, None)
    if isinstance(out.get("files"), list):
        out["files"] = [_file_page(v) for v in out["files"]]
    return out


def compact(tool, value, arguments, addresses=None):
    """Return a presentation copy. Never mutate a saved receipt or its fingerprint."""
    addresses = addresses or {}
    if not isinstance(value, dict):
        return value
    if tool == "terminal" or (
        tool == "call" and str(arguments.get("operation", "")).startswith("terminal_")
    ):
        out = {
            k: v
            for k, v in value.items()
            if k
            not in (
                "request_id",
                "request_state",
                "accepted_at",
                "finished_at",
                "operation",
                "duplicate",
                "execution_replayed",
                "execution_results",
                "submission_receipts",
            )
        }
        if out.get("completion_unknown") is False:
            out.pop("completion_unknown")
        return _receipt_flags(value, out)
    if tool == "run" or (tool == "call" and arguments.get("operation") == "jms_exec"):
        if "results" not in value:
            return value
        out = {"ok": value["ok"], "results": [_execution(r, addresses) for r in value["results"]]}
        if value.get("inventory_stale"):
            out["inventory_stale"] = True
        return _receipt_flags(value, out)
    if tool == "fs" or (tool == "call" and arguments.get("operation") == "jms_fs"):
        # Unknown/failed execution must retain transport diagnostics inline.
        if not value.get("ok"):
            return value
        out = {
            k: value[k]
            for k in (
                "ok",
                "host",
                "result_id",
                "action",
                "local_path",
                "path",
                "overwrite",
                "existing_files",
                "stdout",
                "stderr",
                "stdout_truncated",
                "stderr_truncated",
                "warning",
            )
            if k in value
        }
        if "data" in value:
            out["data"] = _file_page(value["data"])
        return _receipt_flags(value, _identity(out, addresses))
    if tool == "read_result":
        # Actual page start/cursor/eof are part of the result, not optional metadata.
        return {k: v for k, v in value.items() if k not in ("stream", "total_bytes")}
    if tool == "discover":
        if arguments.get("topic") == "assets" and "assets" in value:
            out = {k: deepcopy(v) for k, v in value.items() if k != "inventory"}
            for asset in out["assets"]:
                asset.pop("name", None)
                asset.pop("accounts", None)
            if value.get("inventory", {}).get("stale"):
                out["inventory_stale"] = True
            return out
        return value
    if tool != "call":
        return value
    operation = arguments.get("operation")
    if operation in ("request_status", "result_metadata", "script_guard_check"):
        return value
    if operation == "jms_accelerator_status":
        out = {k: deepcopy(v) for k, v in value.items() if k not in ("policy", "details")}
        for row in out.get("hosts", []):
            row.pop("timing", None)
            if row.get("address"):
                row["host"] = row["address"]
        return out
    # Known generic call envelope. Keep its shape and all operation-specific data.
    out = {k: v for k, v in value.items() if k not in ("execution_results", "submission_receipts")}
    if isinstance(value.get("data"), dict):
        data = {
            k: deepcopy(v)
            for k, v in value["data"].items()
            if k not in ("execution_results", "submission_receipts")
        }
        data = _receipt_flags(value["data"], data)
        for key in ("jobs", "status", "hosts"):
            if isinstance(data.get(key), list):
                data[key] = [
                    _identity(row, addresses) if isinstance(row, dict) else row for row in data[key]
                ]
        if operation == "jms_submit_job":
            receipts = value["data"].get("submission_receipts", [])
            for job in data.get("jobs", []):
                # A candidate ID and unknown launch must never disappear when
                # the launch ACK is missing; it is needed for reconciliation.
                receipt = next(
                    (
                        r
                        for r in receipts
                        if addresses.get(r.get("host"), r.get("host")) == job.get("host")
                        and (not job.get("job_id") or r.get("job_id") == job.get("job_id"))
                    ),
                    None,
                )
                if receipt and (not job.get("launched") or receipt.get("completion_unknown")):
                    job.update(
                        {
                            k: receipt[k]
                            for k in (
                                "job_id",
                                "submission_state",
                                "completion_unknown",
                                "result_id",
                            )
                            if k in receipt
                        }
                    )
        out["data"] = data
    return _receipt_flags(value, out)
