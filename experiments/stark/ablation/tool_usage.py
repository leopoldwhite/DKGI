"""Tool-call counts of one attempt: issued Pi tool calls and requests to the graph-tool service."""

from collections import Counter

import common as c


def issued(attempt, arm):
    if arm == "H":
        return c.val.harness.behavior(attempt, "codex")
    events = c.base.read_jsonl(attempt / "artifacts/events.jsonl")
    starts = {e["toolCallId"]: e for e in events if e.get("type") == "tool_execution_start"}
    ends = {e["toolCallId"]: e for e in events if e.get("type") == "tool_execution_end"}
    service_log = attempt / "tool_requests.jsonl"
    service_calls = [r for r in c.base.read_jsonl(service_log) if r.get("phase") == "agent"] if service_log.exists() else []
    return {"issued_tool_calls": len(starts), "tools_by_name": dict(Counter(e["toolName"] for e in starts.values())),
            "service_calls_by_name": dict(Counter(r["request"]["name"] for r in service_calls if r.get("request"))),
            "tool_completion_unknown": len(starts.keys() - ends.keys()),
            "completed_tool_errors": sum(bool(e.get("isError")) for e in ends.values())}
