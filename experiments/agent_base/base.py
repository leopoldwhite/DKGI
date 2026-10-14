"""Agent process runner shared by every configuration: Landlock isolation, Codex and Pi command lines,
sandbox preflight, a subreaper that owns all tool processes, and result collection."""

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
_provider_spec = importlib.util.spec_from_file_location("dkgi_provider", ROOT / "provider.py")
provider = importlib.util.module_from_spec(_provider_spec)
_provider_spec.loader.exec_module(provider)
configure_protocol = provider.configure_protocol
redact = provider.redact
require_api_key = provider.require_api_key
total_cost = provider.total_cost

REPO = ROOT.parent.parent / "DCI-Agent-Lite"
LANDLOCK = ROOT.parent / "codex_probe/landlock_exec.py"
CODEX = Path(shutil.which("codex") or "codex-not-found").resolve()
CODEX_ROOT = CODEX.parent.parent
NODE_ROOT = Path(shutil.which("node") or "node-not-found").resolve().parent.parent
PYTHON = REPO / ".venv/bin/python"
PYTHON_ROOT = PYTHON.resolve().parent.parent
MODEL_CONFIGURATION_KEYS = ("provider", "model", "api", "reasoning_effort", "context_window", "max_tokens")


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def isolation(attempt, corpus, harness):
    readable = ["/usr", "/bin", "/lib", "/lib64", "/etc/ld.so.cache", "/etc/ssl/certs",
                "/etc/ssl/openssl.cnf", "/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf",
                "/etc/passwd", "/etc/localtime", CODEX_ROOT, corpus, LANDLOCK, ROOT / "sandbox-shell.sh"]
    if harness == "pi":
        readable += [NODE_ROOT, PYTHON_ROOT, REPO / ".venv", REPO / "src",
                     REPO / "pi-mono/package.json", REPO / "pi-mono/node_modules"]
        readable += [REPO / "pi-mono/packages" / name for name in ("ai", "agent", "tui", "coding-agent")]
    command = ["/usr/bin/python3", "-S", str(LANDLOCK)]
    for path in readable:
        command += ["--read", str(path)]
    writable = [attempt, Path("/dev/null"), Path("/dev/urandom"), Path("/dev/random")]
    if harness == "codex":
        writable.append(Path.home() / ".codex/installation_id")
    for path in writable:
        command += ["--write", str(path)]
    return command + ["--"]


def build_command(attempt, row, arm, protocol):
    protocol = configure_protocol(protocol, arm["harness"])
    save(attempt / "model_configuration.json", {key: protocol[key] for key in MODEL_CONFIGURATION_KEYS})
    corpus = Path(row["corpus_dir"])
    env = provider.process_env(protocol)
    env.update(TMPDIR=str(attempt / "tmp"), PATH=f"{NODE_ROOT}/bin:{CODEX_ROOT}/codex-path:/usr/bin:/bin",
               PI_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", DCI_CORPUS=str(corpus), DCI_ATTEMPT=str(attempt),
               DCI_CODEX_ROOT=str(CODEX_ROOT), DCI_LANDLOCK=str(LANDLOCK))
    if arm["harness"] == "pi":
        agent = attempt / "agent"
        agent.mkdir(exist_ok=True)
        save(agent / "models.json", provider.pi_provider_config(protocol))
        save(agent / "settings.json", {"shellPath": str(ROOT / "sandbox-shell.sh"), "quietStartup": True})
        cmd = [str(PYTHON), "-m", "dci.benchmark.pi_rpc_runner", "--provider", protocol["provider"], "--model", protocol["model"],
               "--tools", "read,bash", "--package-dir", str(REPO / "pi-mono/packages/coding-agent"),
               "--agent-dir", str(agent), "--cwd", str(corpus), "--question-file", str(attempt / "prompt.txt"),
               "--output-dir", str(attempt / "artifacts"), "--max-turns", str(protocol["pi_max_turns"]),
               "--compact-tool-updates", f"--extra-arg=--thinking {protocol['reasoning_effort']}",
               f"--extra-arg=--context-management-level {arm['context_level']}"]
        cmd += [f"--extra-arg={flag}" for flag in ("--no-skills", "--no-extensions", "--no-prompt-templates", "--no-themes", "--offline")]
    else:
        config = {
            "permissions.dci.filesystem": '{":root"="write"}', "permissions.dci.network.enabled": "false",
            "default_permissions": '"dci"', "web_search": '"disabled"', "project_doc_max_bytes": "0",
            "shell_environment_policy.inherit": '"core"',
            "shell_environment_policy.exclude": '["*KEY*","*TOKEN*","*SECRET*"]',
            "log_dir": json.dumps(str(attempt / "logs")), "sqlite_home": json.dumps(str(attempt / "state")),
            "model_provider": '"dkgi_openai"',
            "model_context_window": str(protocol["context_window"]), "features.skip_host_skill_discovery": "true",
            "model_providers.dkgi_openai": '{name="OpenAI",base_url=' + json.dumps(protocol["base_url"]) + ',env_key=' +
            json.dumps(protocol["api_key_env"]) + ',wire_api="responses",requires_openai_auth=false}',
        }
        if protocol["reasoning_effort"] != "off":
            config["model_reasoning_effort"] = json.dumps(protocol["reasoning_effort"])
        for feature in ("shell_snapshot", "plugins", "remote_plugin", "apps", "hooks", "multi_agent", "memories", "browser_use", "computer_use", "image_generation"):
            config[f"features.{feature}"] = "false"
        cmd = [str(CODEX), "-a", "never"]
        for key, value in config.items():
            cmd += ["-c", f"{key}={value}"]
        cmd += ["exec", "--ignore-user-config", "--ignore-rules", "--ephemeral", "--skip-git-repo-check",
                "--model", protocol["model"], "--json", "--color", "never", "-o", str(attempt / "final.txt"), "-"]
    return isolation(attempt, corpus, arm["harness"]) + cmd, env


def preflight(attempt, row, arm, env):
    corpus = Path(row["corpus_dir"])
    probe = """import os, socket
assert os.listdir(CORPUS)
for path in DENIED:
    try:
        fd = os.open(path, os.O_RDONLY)
    except (PermissionError, FileNotFoundError):
        continue
    else:
        os.close(fd)
        raise AssertionError('unexpected readable path: ' + path)
try:
    fd = os.open(CORPUS + '/.dci-write-check', os.O_CREAT | os.O_WRONLY, 0o600)
except PermissionError:
    pass
else:
    os.close(fd)
    raise AssertionError('corpus writable')
"""
    probe = "CORPUS=" + repr(str(corpus)) + "\nDENIED=" + repr([str(ROOT / "private"), str(REPO / "data"), str(attempt.parent), str(Path.home() / ".codex/auth.json"), "/proc/self/environ"]) + "\n" + probe
    outer = isolation(attempt, corpus, arm["harness"])
    subprocess.run(outer + ["/usr/bin/python3", "-S", "-c", probe], cwd=corpus, env=env, check=True, capture_output=True)
    network_probe = "import os,socket; assert not any(k in os.environ for k in ['MODEL_API_KEY','OPENAI_API_KEY','ANTHROPIC_API_KEY'])\ntry:\n socket.socket()\nexcept PermissionError:\n pass\nelse:\n raise AssertionError('network available')"
    shell = [str(ROOT / "sandbox-shell.sh"), "-c", "python3 -S -c " + shlex.quote(network_probe)]
    subprocess.run(outer + shell, cwd=corpus, env=env, check=True, capture_output=True)
    save(attempt / "preflight.json", {"filesystem": "passed", "shell_wrapper_network_and_credentials": "passed",
         "scope": "Outer filesystem and Pi shell wrapper; does not launch a Codex model/tool turn", "checked_at": now()})


def run_process(command, env, attempt, corpus, timeout):
    # A private subreaper also owns tools that call setsid() or outlive Pi.
    reaper = """import ctypes, os, signal, subprocess, sys, time
from pathlib import Path
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
    raise OSError(ctypes.get_errno(), 'Cannot enable child subreaper')
def terminate(signum, frame):
    raise SystemExit(128 + signum)
signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)
try:
    code = subprocess.Popen(sys.argv[1:]).wait()
finally:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    deadline = time.monotonic() + 10
    signalled = {}
    while True:
        try:
            while True:
                reaped, _ = os.waitpid(-1, os.WNOHANG)
                if not reaped:
                    break
                signalled.pop(reaped, None)
        except ChildProcessError:
            break
        sig = signal.SIGTERM if time.monotonic() < deadline else signal.SIGKILL
        children = Path(f'/proc/self/task/{os.getpid()}/children').read_text().split()
        for child in children:
            pid = int(child)
            if signalled.get(pid) != sig:
                # These are our unreaped children, so their PIDs cannot be reused.
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
                signalled[pid] = sig
        time.sleep(0.02)
if code < 0:
    if -code != signal.SIGKILL:
        signal.signal(-code, signal.SIG_DFL)
    os.kill(os.getpid(), -code)
sys.exit(code)
"""
    with (attempt / "prompt.txt").open("rb") as source, (attempt / "stdout.jsonl").open("wb") as output, (attempt / "stderr.txt").open("wb") as errors:
        process = subprocess.Popen([sys.executable, "-S", "-c", reaper, *command], cwd=corpus, env=env, stdin=source, stdout=output, stderr=errors, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            process.terminate()
            process.wait()
            if isinstance(exc, subprocess.TimeoutExpired):
                exc.cmd = command
            raise


def collect(attempt, harness, require_completed=True):
    if harness == "pi":
        state = read_json(attempt / "artifacts/state.json")
        if require_completed and state["status"] != "completed":
            raise RuntimeError("Pi status: " + state["status"])
        messages = [entry["message"] for entry in state["messages"] if entry.get("event") == "message_end" and entry.get("message", {}).get("role") == "assistant"]
        usage = {key: sum(message.get("usage", {}).get(key, 0) for message in messages) for key in ("input", "output", "cacheRead", "cacheWrite")}
        tools = [entry for entry in state["tool_calls"] if entry["event"] == "tool_execution_end"]
        with (attempt / "artifacts/events.jsonl").open() as source:
            event_counts = Counter(json.loads(line).get("type") for line in source if line.strip())
        compactions = event_counts["compaction_start"]
        result = {"usage_raw": usage, "input_tokens": usage["input"] + usage["cacheRead"] + usage["cacheWrite"],
                  "output_tokens": usage["output"], "tool_calls": len(tools), "tool_errors": sum(bool(t.get("isError")) for t in tools),
                  "model_requests": None if compactions else len(messages), "agent_loop_requests": len(messages),
                  "compaction_starts": compactions, "compaction_ends": event_counts["compaction_end"],
                  "usage_scope": "Recorded agent-loop usage; summary calls and unreported retries are excluded",
                  "summary_usage": None}
        final_path = attempt / "artifacts/final.txt"
    else:
        events = read_jsonl(attempt / "stdout.jsonl")
        turns = [event for event in events if event["type"] == "turn.completed"]
        terminal_events = [event for event in events if event["type"] in ("turn.completed", "turn.failed", "error")]
        if require_completed and (not terminal_events or terminal_events[-1]["type"] != "turn.completed"):
            raise RuntimeError("Codex did not complete successfully")
        usage = turns[-1]["usage"] if turns else {}
        tools = [event["item"] for event in events if event["type"] == "item.completed" and event["item"]["type"] == "command_execution"]
        result = {"usage_raw": usage, "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                  "tool_calls": len(tools), "tool_errors": sum(t.get("exit_code") != 0 for t in tools), "model_requests": None,
                  "error_events": sum(event["type"] == "error" for event in events),
                  "usage_scope": "Native turn.completed usage; per-request accounting is not exposed"}
        final_path = attempt / "final.txt"
    final = final_path.read_text().strip() if final_path.exists() else ""
    if require_completed and not final:
        raise RuntimeError("No final answer")
    result["prediction"] = final
    configuration_path = attempt / "model_configuration.json"
    if configuration_path.exists():
        saved = read_json(configuration_path)
        identity = {key: saved[key] for key in MODEL_CONFIGURATION_KEYS}
        result.update(model_configuration=identity, provider=identity["provider"], model=identity["model"])
    return result


@contextmanager
def job_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
