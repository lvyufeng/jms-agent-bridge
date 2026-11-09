"""Presentation must retain error and per-target backpressure evidence."""

from jms_bridge.response_views import compact


def test_mixed_fanout_retains_rejection_receipts():
    rejection = {
        "host": "busy",
        "ok": False,
        "reason": "queue_full",
        "retryable": True,
        "queue": {"lane": "general", "depth": 2, "limit": 2},
        "execution_state": "not_started",
        "executed": False,
        "completion_unknown": False,
    }
    rows = [{"host": "ok", "ok": True}, rejection]
    payload = {"ok": True, "result_id": "saved", "data": {"execution_results": rows}}
    result = compact("call", payload, {"operation": "jms_sync_push"})
    assert result["data"]["execution_results"] == rows


def test_run_failure_retains_queue_and_unknown_fields():
    failed = {
        "host": "busy",
        "ok": False,
        "reason": "queue_full",
        "retryable": True,
        "queue": {"lane": "general", "depth": 2, "limit": 2},
        "execution_state": "not_started",
        "executed": False,
        "completion_unknown": False,
    }
    result = compact("run", {"ok": False, "results": [failed]}, {})
    assert result["results"][0] == failed
