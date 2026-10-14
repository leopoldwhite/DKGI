"""STaRK adapters for the Codex and Pi harnesses."""

from collections import Counter
import json
from pathlib import Path
import subprocess

from variants import stark
import align_intent_distribution as aligned
from download_original_behavior import extract_trace

base = stark.base
run_process = base.run_process
ARMS = {"codex": stark.ARM, "pi": {"harness": "pi", "context_level": "level3"}}
LUNA_MODEL = {
    "id": "gpt-6-luna", "name": "GPT-6 Luna", "api": "openai-responses",
    "reasoning": True, "input": ["text"], "contextWindow": 272000,
    "maxTokens": 128000,
    "cost": {"input": 0.10, "output": 0.50, "cacheRead": 0.01, "cacheWrite": 0.125},
}


def prepare_attempt(attempt, row, protocol, harness, prompt_text):
    attempt = Path(attempt)
    arm = ARMS[harness]
    if row.get("split") != "val":
        raise ValueError("Prompt validation accepts only val tasks")
    protocol = base.configure_protocol(protocol, harness=harness)
    if harness == "pi" and protocol.get("pi_max_turns") != 300:
        raise ValueError("Pi requires pi_max_turns=300")
    # Prevent the RPC helper from falling back to an automatic build.
    if harness == "pi" and not (base.REPO / "pi-mono/packages/coding-agent/dist/cli.js").is_file():
        raise RuntimeError("Pi CLI is missing; rebuilding is forbidden")
    for folder in (attempt, attempt / "tmp", attempt / "logs", attempt / "state"):
        folder.mkdir(parents=True, exist_ok=True)
    (attempt / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    command, env = base.build_command(attempt, row, arm, protocol)
    base.preflight(attempt, row, arm, env)
    denied = [stark.ROOT / "private", stark.ROOT / "public",
              stark.prepare.SOURCE / "data/qa", stark.prepare.SOURCE / "data/skb"]
    probe = "import os\nfor path in " + repr([str(path) for path in denied]) + ":\n"
    probe += " try:\n  fd=os.open(path,os.O_RDONLY)\n except (PermissionError, FileNotFoundError):\n  continue\n else:\n  os.close(fd)\n  raise AssertionError('label/source path readable: '+path)\n"
    subprocess.run(base.isolation(attempt, Path(row["corpus_dir"]), harness) +
                   ["/usr/bin/python3", "-S", "-c", probe], cwd=row["corpus_dir"], env=env,
                   check=True, capture_output=True, timeout=30)
    checks = base.read_json(attempt / "preflight.json")
    checks["stark_labels_and_source"] = "denied by actual open probes"
    if harness == "codex":
        help_result = subprocess.run(command[:-1] + ["--help"], cwd=row["corpus_dir"], env=env,
                                     check=True, capture_output=True, text=True, timeout=30)
        (attempt / "cli-help.txt").write_text(help_result.stdout, encoding="utf-8")
        checks["codex_cli_arguments"] = "accepted by --help"
    base.save(attempt / "preflight.json", checks)
    base.save(attempt / "command.json", command)
    return command, env


def collect(attempt, harness, require_completed=True):
    result = base.collect(Path(attempt), harness, require_completed=require_completed)
    usage = result["usage_raw"]
    keys = ("input", "output", "cacheRead", "cacheWrite") if harness == "pi" else (
        "input_tokens", "output_tokens", "cached_input_tokens")
    complete = all(type(usage.get(key)) is int and usage[key] >= 0 for key in keys)
    if harness == "pi" and require_completed:
        state = base.read_json(Path(attempt) / "artifacts/state.json")
        messages = [entry["message"] for entry in state["messages"]
                    if entry.get("event") == "message_end" and entry.get("message", {}).get("role") == "assistant"]
        complete = complete and bool(messages) and all(
            all(type(message.get("usage", {}).get(key)) is int and message["usage"][key] >= 0 for key in keys)
            for message in messages)
    if require_completed and not complete:
        raise RuntimeError("Incomplete token usage in completed trajectory")
    result.update(cached_input_tokens=usage.get("cacheRead" if harness == "pi" else "cached_input_tokens"),
                  usage_complete=complete, usage_notes=result["usage_scope"])
    return result


def behavior(attempt, harness):
    attempt = Path(attempt)
    if harness == "codex":
        path = attempt / "stdout.jsonl"
        calls = list(aligned.codex_calls(base.read_jsonl(path)))
        extra = {"read_calls": 0}
    else:
        path = attempt / "artifacts/events.jsonl"
        events = base.read_jsonl(path)
        messages = [event["message"] for event in events if event.get("type") == "message_end"]
        data = json.dumps({"qid": attempt.parent.name, "messages": messages}).encode()
        calls, metadata = extract_trace(data, path, attempt.parent.name)
        extra = {"read_calls": metadata["tool_issued"].get("read", 0),
                 "duplicate_tool_calls": metadata["duplicate_tool_calls"],
                 "unmatched_tool_results": metadata["unmatched_tool_results"]}
    classifications = [aligned.classify(call["command"]) for call in calls]
    counts = Counter(item["category"] for item in classifications)
    return {"bash_calls": len(calls), "python_calls": counts["python"],
            "python_share": counts["python"] / len(calls) if calls else None,
            "bash_calls_completed": sum(call["completed"] for call in calls),
            "intent_counts": {name: counts[name] for name in aligned.ORDER},
            "shell_parse_errors": sum(item["shell_parse_errors"] for item in classifications),
            "python_parse_errors": sum(item["python_parse_errors"] for item in classifications),
            "denominator": "unique issued bash calls; independent read calls excluded", **extra}
