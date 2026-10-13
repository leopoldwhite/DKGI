"""Cache public DCI-Agent-Lite traces and extract issued bash calls without execution."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO = "DCI-Agent/eval-logs"
REVISION = "b9cb2a846acbccbac9450ca13a9da0f134e75b60"
PREFIX = "dci-agent-lite/browsecomp-plus"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "behavior_710" / "aligned"


def fetch(url: str) -> tuple[bytes, str | None]:
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "dci-behavior-audit/1"}), timeout=90) as response:
                return response.read(), response.headers.get("Link")
        except HTTPError as error:
            if error.code != 429 and not 500 <= error.code <= 599:
                raise
            if attempt == 2:
                raise
            delay = min(int(error.headers.get("Retry-After", "0") or 0), 30) if str(error.headers.get("Retry-After", "")).isdigit() else 0
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            delay = 0
        time.sleep(max(delay, 2 ** attempt))
    raise AssertionError("unreachable")


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


def list_files(revision: str) -> tuple[list[dict], list[dict]]:
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/{PREFIX}?recursive=false&limit=1000"
    entries = []
    while url:
        body, link = fetch(url)
        entries.extend(json.loads(body))
        match = re.search(r'<([^>]+)>;\s*rel="next"', link or "")
        url = match.group(1) if match else ""
    files = [entry for entry in entries if entry["type"] == "file" and re.fullmatch(rf"{re.escape(PREFIX)}/[0-9]+\.jsonl", entry["path"])]
    excluded = [entry for entry in entries if entry not in files]
    if not files or len({entry["path"] for entry in files}) != len(files):
        raise ValueError("Missing or duplicate public trace paths")
    return sorted(files, key=lambda item: int(Path(item["path"]).stem)), excluded


def extract_trace(data: bytes, path: Path, expected_qid: str) -> tuple[list[dict], dict]:
    calls, metadata = [], {"query_id": expected_qid, "models": [], "providers": [], "tool_issued": {}, "tool_completed": {}}
    issued_counts, completed_counts = Counter(), Counter()
    models, providers, issued_ids, result_ids = set(), set(), {}, set()
    duplicate_calls = duplicate_results = 0
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if str(record.get("qid")) != expected_qid:
            raise ValueError(f"Query ID mismatch in {path}:{line_number}")
        messages = record["messages"]
        results = {}
        for index, message in enumerate(messages):
            if message.get("role") != "toolResult":
                continue
            identity = message.get("toolCallId")
            if not identity:
                raise ValueError(f"Missing tool result ID in {path}:{line_number}")
            if identity in result_ids:
                duplicate_results += 1
                continue
            result_ids.add(identity)
            results[identity] = (index, message)
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            if message.get("model"):
                models.add(message["model"])
            if message.get("provider"):
                providers.add(message["provider"])
            for block_index, block in enumerate(message.get("content", [])):
                if not isinstance(block, dict) or block.get("type") != "toolCall":
                    continue
                identity, name = block.get("id"), block.get("name")
                if not identity or not name:
                    raise ValueError(f"Missing tool call identity in {path}:{line_number}")
                call_text = json.dumps([name, block.get("arguments")], sort_keys=True)
                if identity in issued_ids:
                    if issued_ids[identity] != call_text:
                        raise ValueError(f"Conflicting duplicate tool call in {path}:{line_number}")
                    duplicate_calls += 1
                    continue
                issued_ids[identity] = call_text
                issued_counts[name] += 1
                result = results.get(identity)
                if result:
                    if result[1].get("toolName") != name:
                        raise ValueError(f"Tool name mismatch in {path}:{line_number}")
                    completed_counts[name] += 1
                if name != "bash":
                    continue
                command = block.get("arguments", {}).get("command")
                if not isinstance(command, str):
                    raise ValueError(f"Invalid bash command in {path}:{line_number}")
                calls.append({
                    "dataset": "browsecomp-plus", "query_id": expected_qid,
                    "step": issued_counts[name], "command": command,
                    "tool_call_id": identity, "trace_path": str(path), "trace_line": line_number,
                    "message_index": index, "content_index": block_index,
                    "result_message_index": result[0] if result else None,
                    "completed": result is not None,
                    "is_error": result[1].get("isError") if result else None,
                    "hit_at_1": None, "model": message.get("model"),
                })
    metadata.update({
        "models": sorted(models), "providers": sorted(providers),
        "tool_issued": dict(issued_counts), "tool_completed": dict(completed_counts),
        "duplicate_tool_calls": duplicate_calls, "duplicate_tool_results": duplicate_results,
        "unmatched_tool_results": len(result_ids - issued_ids.keys()),
        "hit_at_1": None,
    })
    return calls, metadata


def download_one(entry: dict, revision: str, cache: Path) -> tuple[list[dict], dict]:
    path = cache / Path(entry["path"]).name
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{entry['path']}"
    # The pinned revision fixes the source; files are checked by their listed size.
    cached = path.exists()
    data = path.read_bytes() if cached else b""
    if len(data) != entry["size"]:
        data, _ = fetch(url)
        if len(data) != entry["size"]:
            raise ValueError(f"Size mismatch for {entry['path']}")
        atomic_write(path, data)
        cached = False
    calls, metadata = extract_trace(data, path, path.stem)
    metadata.update({
        "trace_path": str(path), "source_path": entry["path"], "url": url,
        "bytes": len(data), "cached": cached,
        "model": metadata["models"][0] if len(metadata["models"]) == 1 else None,
        "bash_issued": metadata["tool_issued"].get("bash", 0),
        "bash_completed": metadata["tool_completed"].get("bash", 0),
        "read_issued": metadata["tool_issued"].get("read", 0),
    })
    return calls, metadata


def self_check() -> None:
    data = json.dumps({"qid": "7", "messages": [
        {"role": "assistant", "model": "example", "content": [
            {"type": "thinking", "thinking": "never exported"},
            {"type": "toolCall", "id": "a", "name": "bash", "arguments": {"command": "rg x | head"}},
            {"type": "toolCall", "id": "b", "name": "read", "arguments": {"path": "x"}},
            {"type": "toolCall", "id": "c", "name": "bash", "arguments": {"command": "false"}},
            {"type": "toolCall", "id": "a", "name": "bash", "arguments": {"command": "rg x | head"}},
        ]},
        {"role": "toolResult", "toolCallId": "a", "toolName": "bash", "isError": True},
        {"role": "toolResult", "toolCallId": "b", "toolName": "read", "isError": False},
    ]}).encode()
    calls, meta = extract_trace(data, Path("/example/7.jsonl"), "7")
    assert len(calls) == 2 and calls[0]["completed"] and calls[0]["is_error"]
    assert not calls[1]["completed"] and calls[1]["result_message_index"] is None
    assert meta["tool_issued"] == {"bash": 2, "read": 1}
    assert meta["tool_completed"] == {"bash": 1, "read": 1}
    assert meta["duplicate_tool_calls"] == 1 and "never exported" not in json.dumps(calls)
    print("original trace extraction checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    cache = args.output.resolve() / "original_logs"
    cache.mkdir(parents=True, exist_ok=True)
    files, excluded = list_files(args.revision)
    print(f"Public cohort: {len(files)} traces, {sum(entry['size'] for entry in files):,} bytes, revision {args.revision}", flush=True)
    extracted, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(download_one, entry, args.revision, cache): entry for entry in files}
        for done, task in enumerate(as_completed(tasks), 1):
            try:
                extracted.append(task.result())
            except Exception as error:
                errors.append({"path": tasks[task]["path"], "error": str(error)})
            if done % 50 == 0 or done == len(files):
                print(f"Processed {done}/{len(files)}; errors={len(errors)}", flush=True)
    if errors:
        atomic_write(args.output / "original_download_errors.json", json.dumps(errors, indent=2).encode() + b"\n")
        raise RuntimeError(f"Failed {len(errors)} traces; aggregate outputs were not written")
    extracted.sort(key=lambda item: int(item[1]["query_id"]))
    calls = [call for group, _ in extracted for call in group]
    per_query = [metadata for _, metadata in extracted]
    issued, completed = Counter(), Counter()
    for record in per_query:
        issued.update(record["tool_issued"])
        completed.update(record["tool_completed"])
    manifest = {
        "repo_id": REPO, "revision": args.revision, "source_prefix": PREFIX,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "cohort": "All numeric query traces publicly present at the pinned revision; not verified as Figure 6's 100 cases.",
        "outcome_policy": "No explicit correctness labels present; outcomes remain unknown.",
        "query_count": len(per_query), "source_bytes": sum(record["bytes"] for record in per_query),
        "tool_issued": dict(issued), "tool_completed": dict(completed),
        "bash_without_result": issued["bash"] - completed["bash"],
        "duplicate_tool_calls": sum(record["duplicate_tool_calls"] for record in per_query),
        "duplicate_tool_results": sum(record["duplicate_tool_results"] for record in per_query),
        "unmatched_tool_results": sum(record["unmatched_tool_results"] for record in per_query),
        "excluded_entries": excluded, "queries": per_query,
    }
    manifest["totals"] = {key: manifest[key] for key in (
        "query_count", "source_bytes", "tool_issued", "tool_completed", "bash_without_result",
        "duplicate_tool_calls", "duplicate_tool_results", "unmatched_tool_results",
    )}
    command_data = b"".join(json.dumps(call, ensure_ascii=False).encode() + b"\n" for call in calls)
    manifest["commands_bytes"] = len(command_data)
    atomic_write(args.output / "original_commands.jsonl", command_data)
    atomic_write(args.output / "original_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False).encode() + b"\n")
    print(json.dumps({key: manifest[key] for key in ("query_count", "tool_issued", "tool_completed", "bash_without_result", "duplicate_tool_calls", "duplicate_tool_results", "unmatched_tool_results")}), flush=True)


if __name__ == "__main__":
    main()
