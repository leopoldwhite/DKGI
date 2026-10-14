"""Index exact Pi provider requests already archived in events.jsonl."""

import argparse
import json
from pathlib import Path
import re

MARKER = re.compile(r"\n\[\.\.\.truncated, ([0-9]+) chars omitted\]\Z")


def js_length(text):
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def requests(path):
    with Path(path).open("rb") as stream:
        line = 0
        while True:
            offset = stream.tell()
            raw = stream.readline()
            if not raw:
                return
            line += 1
            event = json.loads(raw)
            if event.get("type") == "provider_request_context":
                yield offset, line, event


def provider_items(payload):
    """Normalize native Responses and Messages tool records for transport-independent audits."""
    if isinstance(payload.get("input"), list):
        return payload["input"]
    if not isinstance(payload.get("messages"), list):
        raise ValueError("Unsupported provider request payload")
    items = []
    if payload.get("system"):
        items.append({"role": "system", "content": payload["system"]})
    for message in payload["messages"]:
        content = message.get("content", [])
        if isinstance(content, str):
            items.append({"role": message["role"], "content": content})
            continue
        text = [block for block in content if block.get("type") == "text"]
        if text:
            items.append({"role": message["role"], "content": text})
        for block in content:
            if block.get("type") == "tool_use":
                items.append({"type": "function_call", "call_id": block["id"], "name": block["name"],
                              "arguments": json.dumps(block["input"])})
            elif block.get("type") == "tool_result":
                output = block.get("content", "")
                if isinstance(output, list):
                    output = "\n".join(b["text"] for b in output if b.get("type") == "text") or "(see attached image)"
                items.append({"type": "function_call_output", "call_id": block["tool_use_id"], "output": output})
    return items


def inspect(event):
    messages = {m["toolCallId"].split("|")[0]: m for m in event["messages"]
                if m.get("role") == "toolResult"}
    outputs = [item for item in provider_items(event["payload"]) if item.get("type") == "function_call_output"]
    assert len(outputs) == len(messages), "Request tool result coverage differs"
    assert {item["call_id"] for item in outputs} == messages.keys(), "Request tool IDs differ"
    records = []
    for item in outputs:
        message = messages[item["call_id"]]
        blocks = [b["text"] for b in message["content"] if b.get("type") == "text"]
        expected = "\n".join(blocks) or "(see attached image)"
        expected = expected.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "ignore")
        assert item["output"] == expected, "Model payload differs from recorded transformed context"
        marker = MARKER.fullmatch(blocks[-1]) if blocks else None
        content_length = sum(js_length(s) for s in (blocks[:-1] if marker else blocks))
        if not message.get("isError"):
            assert content_length <= 20000, "Successful tool output exceeds unchanged L3 limit"
            if marker:
                assert content_length == 20000 and int(marker[1]) > 0
        records.append({"call_id": item["call_id"], "tool": message.get("toolName"),
                        "text_chars": len(expected), "text_utf16_units": js_length(expected),
                        "text_utf8_bytes": len(expected.encode()), "content_utf16_units": content_length,
                        "l3_truncated": bool(marker), "is_error": bool(message.get("isError"))})
    return records


def export(attempt):
    attempt = Path(attempt)
    source = attempt / "artifacts/events.jsonl"
    records = []
    for offset, line, event in requests(source):
        assert event["requestIndex"] == len(records) + 1, "Missing or repeated Pi request index"
        records.append({"request_index": event["requestIndex"], "byte_offset": offset, "line": line,
                        "tool_results": inspect(event)})
    assert records, "No Pi request contexts archived"
    summary = {"status": "passed", "requests": len(records),
               "tool_output_occurrences": sum(len(r["tool_results"]) for r in records),
               "l3_truncated_occurrences": sum(t["l3_truncated"] for r in records for t in r["tool_results"]),
               "source": str(source), "limit_utf16_units": 20000,
               "note": "Full payload and post-context-management messages remain in each indexed source event."}
    (attempt / "pi_request_index.json").write_text(json.dumps({**summary, "index": records}, indent=2) + "\n")
    return summary


def self_check():
    text = "a" * 19998 + "\U0001f680"
    assert js_length(text) == 20000
    blocks = [{"type": "text", "text": text}, {"type": "text", "text": "\n[...truncated, 5 chars omitted]"}]
    event = {"messages": [{"role": "toolResult", "toolCallId": "call_a|fc_b", "content": blocks}],
             "payload": {"input": [{"type": "function_call_output", "call_id": "call_a",
                                     "output": "\n".join(b["text"] for b in blocks)}]}}
    assert inspect(event)[0]["l3_truncated"]
    event["payload"]["input"][0]["output"] = "lost"
    try:
        inspect(event)
    except AssertionError:
        pass
    else:
        raise AssertionError("Missing payload mismatch detection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", nargs="?", type=Path)
    parser.add_argument("--request", type=int, help="Print the exact archived request event")
    args = parser.parse_args()
    if args.request is not None:
        if args.attempt is None:
            parser.error("--request requires an attempt path")
        event = next(e for _, _, e in requests(args.attempt / "artifacts/events.jsonl")
                     if e["requestIndex"] == args.request)
        print(json.dumps(event, ensure_ascii=True, indent=2))
    elif args.attempt:
        print(json.dumps(export(args.attempt), indent=2))
    else:
        self_check()
        print("Pi request context self-check passed")
