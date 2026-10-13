"""No-tool Pi variant of the harness for the closed-book probes."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parent / "harness_fixes"))
import adapter

pv, base = adapter.runner, adapter.runner.base
ARM = {"harness": "pi", "context_level": "level3"}


def prepare(attempt, prompt, protocol):
    if not (base.REPO / "pi-mono/packages/coding-agent/dist/cli.js").is_file():
        raise ValueError("Pi CLI missing; automatic rebuild forbidden")
    for name in ("", "cwd", "tmp", "logs", "state"):
        (attempt / name).mkdir(parents=True, exist_ok=True)
    (attempt / "prompt.txt").write_text(prompt)
    row = {"corpus_dir": str(attempt / "cwd")}
    command, env = base.build_command(attempt, row, ARM, protocol)
    models = base.read_json(attempt / "agent/models.json")
    provider = models["providers"]["openai"]
    provider.pop("modelOverrides", None)
    provider["models"] = [pv.harness.LUNA_MODEL]
    base.save(attempt / "agent/models.json", models)
    command[command.index("--tools") + 1] = ""
    command.append("--extra-arg=--no-tools")
    for name in ("blocking_stdout.cjs", "pi_rpc_retry.py"):
        shutil.copyfile(ROOT.parent / "prompt_val/transport" / name, attempt / name)
    offset = command.index("dci.benchmark.pi_rpc_runner")
    command[offset - 1:offset + 1] = [str(attempt / "pi_rpc_retry.py")]
    env["NODE_OPTIONS"] = "--require=" + str(attempt / "blocking_stdout.cjs")
    command, env = adapter.launcher.configure(command, env, attempt, "pi")
    settings = base.read_json(attempt / "agent/settings.json")
    settings["retry"] = {"enabled": False}
    base.save(attempt / "agent/settings.json", settings)
    # The fixed shell is the only configured wrapper; the old read grant is unnecessary.
    old = str(base.ROOT / "sandbox-shell.sh")
    offset = command.index(old)
    del command[offset - 1:offset + 1]
    denied = [ROOT.parent / name for name in ("corpus", "private", "public")]
    denied += [ROOT.parents[2] / "STaRK/data", attempt.parent, ROOT / "index",
               Path.home() / ".codex/auth.json", "/proc/self/environ"]
    probe = "import os\nassert os.listdir('.') == []\n"
    probe += "for path in " + repr([str(p) for p in denied]) + ":\n"
    probe += " try:\n  fd=os.open(path,os.O_RDONLY)\n except (PermissionError, FileNotFoundError):\n  continue\n else:\n  os.close(fd)\n  raise AssertionError('unexpected readable path: '+path)\n"
    cli = str(base.REPO / "pi-mono/packages/coding-agent/dist/cli.js")
    probe += "try:\n fd=os.open(" + repr(cli) + ",os.O_WRONLY)\nexcept PermissionError:\n pass\nelse:\n os.close(fd)\n raise AssertionError('runtime writable')\n"
    outer = command[:command.index("--") + 1]
    subprocess.run(outer + ["/usr/bin/python3", "-S", "-c", probe], cwd=attempt / "cwd",
                   env=env, capture_output=True, check=True, timeout=30)
    base.save(attempt / "preflight.json", {"status": "passed", "checked_at": base.now(),
              "empty_cwd": True, "runtime_readonly": True, "denied": [str(p) for p in denied]})
    base.save(attempt / "command.json", command)
    return command, env


def collect(attempt, require_completed=True):
    events_path = attempt / "artifacts/events.jsonl"
    requests = list(adapter.pi_context.requests(events_path))
    for _, _, event in requests:
        if event["payload"].get("tools", []) or event.get("tools", []):
            raise AssertionError("Provider request contained tools")
        if any(m.get("role") in {"assistant", "toolResult"} for m in event.get("messages", [])):
            raise AssertionError("Provider request contained earlier assistant/tool content")
        if any(item.get("type") in {"function_call", "function_call_output"} for item in event["payload"].get("input", [])):
            raise AssertionError("Provider request contained earlier assistant/tool content")
    events = base.read_jsonl(events_path)
    assistants = [e["message"] for e in events if e.get("type") == "message_end" and e.get("message", {}).get("role") == "assistant"]
    if any(e.get("type", "").startswith("tool_execution_") for e in events):
        raise AssertionError("Tool execution occurred")
    if any(b.get("type") == "toolCall" for m in assistants for b in m.get("content", [])):
        raise AssertionError("Assistant issued a tool call")
    if require_completed and (len(requests) != 1 or len(assistants) != 1):
        raise AssertionError("Expected exactly one provider request and assistant reply")
    result = adapter.collect(attempt, "pi", require_completed=require_completed)
    result["no_tools_check"] = {"status": "passed", "requests": len(requests), "assistant_replies": len(assistants), "tool_calls": 0}
    return result
