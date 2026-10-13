"""Native, request-level Codex tracing without changing its execution settings."""

import json
from collections import Counter
from pathlib import Path
import shlex

SOURCE_ROOT = "https://github.com/openai/codex/blob/rust-v0.156.1/codex-rs/"
PROTOCOL = {
    "codex_version": "0.156.1",
    "recording": "CODEX_ROLLOUT_TRACE_ROOT=attempt/codex_trace; --ephemeral retained",
    "main_HOME_and_CODEX_HOME": "unchanged",
    "persistence_and_memory": "No new session persistence; existing disabled memories, rules and user-config settings retained",
    "model_tool_truncation": {"model": "gpt-6-luna", "mode": "tokens", "limit": 10000,
                              "meter": "Approximate UTF-8 byte count / 4, not a model tokenizer",
                              "direction": "Preserve half-budget prefix and half-budget suffix; omit middle",
                              "serialization_allowance": "History permits 20% extra for headers/serialization"},
    "code_mode": "functions.exec defaults to 10000 approximate output tokens; the model may specify its own per-call limit",
    "nested_exec": "Without max_output_tokens, the code cell receives collected raw exec output (up to the existing 1 MiB collection cap); only printed values become model input",
    "limit_changes": "none",
    "sources": [SOURCE_ROOT + path for path in (
        "rollout-trace/README.md", "rollout-trace/src/inference.rs",
        "models-manager/models.json", "core/src/tools/code_mode/mod.rs",
        "core/src/tools/context.rs", "core/src/context_manager/history.rs",
        "utils/string/src/truncate.rs", "core/src/unified_exec/mod.rs")],
}


def configure(command, env, attempt, harness="codex"):
    if harness != "codex":
        return command, env
    attempt = Path(attempt).resolve()
    trace_root = attempt / "codex_trace"
    trace_root.mkdir(parents=True, exist_ok=True)
    env = dict(env, CODEX_ROLLOUT_TRACE_ROOT=str(trace_root))
    return command, env


def _read_jsonl(path):
    with path.open() as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def _payload(bundle, reference):
    path = (bundle / reference["path"]).resolve()
    if not path.is_relative_to(bundle.resolve()):
        raise ValueError("Trace payload escapes its bundle")
    return json.loads(path.read_text())


def _text(output):
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "\n".join(item.get("text", "") for item in output if isinstance(item, dict))
    raise ValueError("Unknown Codex function output format")


def export(attempt):
    """Index exact model requests, retaining all per-request output versions."""
    attempt = Path(attempt).resolve()
    traces = sorted((attempt / "codex_trace").glob("*/trace.jsonl"))
    if len(traces) != 1:
        raise ValueError(f"Expected one native Codex trace; found {len(traces)}")
    bundle = traces[0].parent
    events = list(_read_jsonl(traces[0]))
    requests, outputs, responses, calls = [], [], {}, {}
    runtime_outputs, runtime_calls, cells = {}, {}, {}
    for event in events:
        payload = event["payload"]
        kind = payload["type"]
        if kind == "inference_completed":
            responses[payload["inference_call_id"]] = "completed"
        elif kind in {"inference_failed", "inference_cancelled"}:
            responses[payload["inference_call_id"]] = kind.removeprefix("inference_")
        elif kind == "tool_call_started":
            runtime_calls[payload["tool_call_id"]] = payload
            if payload.get("model_visible_call_id"):
                calls.setdefault(payload["model_visible_call_id"], []).append(payload["tool_call_id"])
        elif kind == "code_cell_started":
            cells[payload["runtime_cell_id"]] = payload["model_visible_call_id"]
        elif kind == "tool_call_runtime_ended":
            raw = _payload(bundle, payload["runtime_payload"])
            raw = raw.get("event", raw)
            if isinstance(raw, dict):
                raw = raw.get("msg", raw)
                aggregate = raw.get("aggregated_output")
                if isinstance(aggregate, str):
                    runtime_outputs[payload["tool_call_id"]] = aggregate
    for call_id, call in runtime_calls.items():
        cell = call.get("requester", {}).get("runtime_cell_id")
        if cell in cells:
            calls.setdefault(cells[cell], []).append(call_id)
    for event in events:
        payload = event["payload"]
        if payload["type"] != "inference_started":
            continue
        request = _payload(bundle, payload["request_payload"])
        request_id = payload["inference_call_id"]
        request_path = str((bundle / payload["request_payload"]["path"]).relative_to(attempt))
        input_items = request.get("input", [])
        if not isinstance(input_items, list):
            raise ValueError("Codex request input is not an item list")
        requests.append({"request_id": request_id, "seq": event["seq"], "path": request_path,
                         "status": responses.get(request_id, "unfinished"),
                         "model": request.get("model"), "input_items": len(input_items),
                         "previous_response_id": request.get("previous_response_id")})
        for index, item in enumerate(input_items):
            if item.get("type") not in {"function_call_output", "custom_tool_call_output"}:
                continue
            text = _text(item["output"])
            linked = calls.get(item.get("call_id"), [])
            aggregates = [runtime_outputs[call] for call in linked if call in runtime_outputs]
            outputs.append({"request_id": request_id, "request_status": responses.get(request_id, "unfinished"),
                            "request_path": request_path, "input_index": index,
                            "call_id": item.get("call_id"), "output": item["output"],
                            "characters": len(text), "bytes_utf8": len(text.encode()),
                            "truncation_notice": "truncated" in text or "bytes omitted" in text,
                            "linked_runtime_call_ids": linked,
                            "runtime_aggregate_bytes": sum(len(value.encode()) for value in aggregates) if aggregates else None,
                            "comparison_note": "One code cell may print, transform, or omit several nested tool results"})
    if not requests:
        raise ValueError("Native trace contains no model request")
    for filename, rows in (("model_requests.jsonl", requests), ("model_tool_outputs.jsonl", outputs)):
        with (attempt / filename).open("w") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
    unique = {(row["call_id"], _text(row["output"])) for row in outputs}
    stdout_completed = Counter()
    if (attempt / "stdout.jsonl").is_file():
        for event in _read_jsonl(attempt / "stdout.jsonl"):
            item = event.get("item", {})
            if event.get("type") == "item.completed" and item.get("type") == "command_execution":
                stdout_completed[tuple(shlex.split(item["command"]))] += 1
    native_completed = Counter()
    for event in events:
        payload = event["payload"]
        if payload["type"] == "tool_call_runtime_ended":
            raw = _payload(bundle, payload["runtime_payload"])
            if isinstance(raw.get("command"), list):
                native_completed[tuple(raw["command"])] += 1
    summary = {"native_trace": str(traces[0].relative_to(attempt)), "requests": len(requests),
               "completed_requests": sum(row["status"] == "completed" for row in requests),
               "tool_output_occurrences": len(outputs), "unique_tool_output_versions": len(unique),
               "unique_tool_call_ids_in_requests": len({row["call_id"] for row in outputs}),
               "native_command_end_events": sum(native_completed.values()),
               "stdout_command_end_events": sum(stdout_completed.values()),
               "native_command_ends_missing_from_stdout": sum((native_completed - stdout_completed).values()),
               "source": "Native inference_started request payload, after context construction and truncation",
               "delivery_caveat": "A failed request is recorded as sent, not proof of provider receipt"}
    (attempt / "codex_observability.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", type=Path)
    print(json.dumps(export(parser.parse_args().attempt), indent=2))
