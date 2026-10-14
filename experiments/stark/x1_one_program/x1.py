"""One program (X1): Full DKGI with the Shell only allowlist plus python3 started once.

Library: prompt, prepare (Full DKGI preparation + X1 shell behind H6), per-attempt counter, preflight probes,
one attempt / one item with retries and resume, and audits. Reuses ../ablation_final/final.py (imported, never
edited) for data loading, scheduling, H6 audit, path-leak count, finalization and result layout. Runs are
identified only by arm / dataset / split / query_id; an item is finished when its result.json is terminal.
"""
import ast
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STARK = HERE.parent
sys.path.insert(0, str(STARK / "ablation_final"))
import final as f  # noqa: E402
sys.path.insert(0, str(HERE))
import gate_service  # noqa: E402
import precheck_interpreters as commands  # noqa: E402  (lexical command parser, reused for audits)

c, base, stark, variants, launcher = f.c, f.base, f.stark, f.variants, f.launcher
pi_transport = f.pi_transport

ARM = "O1"
DATASETS = f.DATASETS
SPLITS = f.SPLITS
RUNS = STARK / "trials"            # neutral path visible to the model
VISIBLE_FORBIDDEN = "ablation"
AGENT_TIMEOUT_SECONDS = 1800
MAX_TECHNICAL_ATTEMPTS = f.MAX_TECHNICAL_ATTEMPTS
MAX_ATTEMPT_DIRS = f.MAX_ATTEMPT_DIRS
STOP = threading.Event()
HARD_STOP = threading.Event()

# Fixed before the run: appended to the Full DKGI tool paragraph as its last line.
SENTENCE = ("Python 3 can be started only once in this task; after that, python3 is unavailable and only shell "
            "text tools and bash remain.")
REFUSAL = "python3: unavailable: Python 3 can be started only once in this task.\n"   # printed by the shim
ALLOWLIST = "rg head tail cat sed wc cut sort comm ls find pwd xargs echo bash".split()
WRAPPER = HERE / "h6/capped-x1.sh"
BASE_SHELL = HERE / "sandbox/x1_shell.py"
TRUSTED_READS = [HERE / "h6", HERE / "sandbox", f.CAP_FILTER]
PI_RETRY = {"enabled": True, "maxRetries": 2, "baseDelayMs": 15000, "maxDelayMs": 120000}   # as ablation_final


def now():
    return base.now()


def save(path, value):
    c.save(path, value)


def read(path):
    return c.read(path)


read_optional = f.read_optional
redact = f.redact
key = f.key


def jobkey(job):
    return f"{job['arm']}:{job['dataset']}:{job['query_id']}"


def item_dir(dataset, split, query_id):
    return RUNS / ARM / dataset / SPLITS[split] / str(query_id)


def result_path(job):
    return item_dir(job["dataset"], job["split"], job["query_id"]) / "result.json"


def is_terminal(job):
    value = read_optional(result_path(job))
    return bool(value and value.get("terminal") is True)


def schedule(questions, split):
    return f.schedule(questions, [ARM], split)


# ---------------------------------------------------------------- prompt

def full_dkgi_prompt(row, attempt):
    """The Full DKGI prompt (P1-R1: P0 with the output sentence replaced once), corpus experiments/stark/corpus/<ds>."""
    original = dict(row, corpus_dir=str(STARK / "corpus" / row["dataset"]))
    return variants.replace_once(f.p0_prompt(original, attempt), variants.OLD_OUTPUT, f.R1_OUTPUT)


def prompt(row, attempt):
    """Full DKGI prompt plus the One program sentence as the last line of the tool paragraph."""
    full = full_dkgi_prompt(row, attempt)
    if not full.endswith("\n") or full.count("Corpus directory:") != 1:
        raise ValueError("Unexpected Full DKGI prompt layout")
    return full + SENTENCE + "\n"


# ---------------------------------------------------------------- preparation

def prepare(attempt, row, protocol):
    """Full DKGI preparation (harness, Landlock, preflight) with Pi's shellPath set to the X1 H6 wrapper."""
    attempt = Path(attempt)
    corpus = STARK / "corpus" / row["dataset"]
    # The reused val harness accepts only rows labelled val; the real split is recorded in arm_x1.json.
    harness_row = dict(row, split="val", corpus_dir=str(corpus))
    text = prompt(row, attempt)
    if any(VISIBLE_FORBIDDEN in v.lower() for v in (text, str(corpus), str(attempt))):
        raise ValueError("A model-visible path or the prompt contains a forbidden substring")
    command, env = c.adapter.prepare_attempt(attempt, harness_row, dict(protocol, pi_max_turns=300), "pi", text)
    settings_path = attempt / "agent/settings.json"
    settings = read(settings_path)
    if settings.get("shellPath") != str(launcher.SHELL):
        raise ValueError(f"Unexpected Pi shellPath before X1: {settings.get('shellPath')}")
    settings["shellPath"] = str(WRAPPER)
    settings["retry"] = dict(PI_RETRY)
    save(settings_path, settings)
    command = list(command)
    boundary = command.index("--")
    command[boundary:boundary] = [part for path in TRUSTED_READS for part in ("--read", str(path))]
    launcher.preflight(command, env, attempt, harness_row)
    save(attempt / "command.json", command)
    record = read(attempt / "harness_fixes.json")
    record["h6"] = {**f.H6_PARAMETERS, "wrapper": str(WRAPPER), "base_shell": str(BASE_SHELL)}
    save(attempt / "harness_fixes.json", record)
    save(attempt / "arm_x1.json", {"arm": ARM, "dataset": row["dataset"], "query_id": row["query_id"],
                                   "split": row["split"], "prompt_base": "P1-R1 + X1 sentence", "cwd": str(corpus),
                                   "shellPath": str(WRAPPER), "h6_base_shell": str(BASE_SHELL),
                                   "allowlist": ALLOWLIST + ["python3 (one start)"]})
    if (attempt / "prompt.txt").read_text() != text:
        raise ValueError("prompt.txt differs from the X1 prompt")
    return command, env, corpus


def shell(command, env, cwd, script, timeout=120):
    """Run one command through the configured outer sandbox and the X1 H6 wrapper, as Pi's bash tool does."""
    outer = command[:command.index("--") + 1]
    return subprocess.run(outer + [str(WRAPPER), "-c", script], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=timeout)


PYTHON_PROBE = r"""
import os, socket, subprocess, sys
out = {"executable": sys.executable, "ld_preload": os.environ.get("LD_PRELOAD"), "lock_env": os.environ.get("X1_LOCK_SPEC")}
def denied(action):
    try:
        action()
    except (PermissionError, FileNotFoundError, subprocess.CalledProcessError):
        return True
    return False
out["exec_python_binary"] = not denied(lambda: subprocess.run(["/usr/bin/python3", "-c", "1"]))
out["exec_python_via_loader"] = not denied(lambda: subprocess.run(["/lib64/ld-linux-x86-64.so.2", "/usr/bin/python3.12", "-c", "1"], check=True))
out["read_python_binary"] = not denied(lambda: open("/usr/bin/python3.12", "rb").close())
out["exec_awk"] = not denied(lambda: subprocess.run(["/usr/bin/awk", "1"], stdin=subprocess.DEVNULL))
out["exec_perl"] = not denied(lambda: subprocess.run(["/usr/bin/perl", "-e", "1"]))
out["network"] = not denied(lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM).close())
out["labels"] = not denied(lambda: open(os.environ["X1P_LABELS"]).close())
out["corpus"] = os.path.exists("DATA_FORMAT.txt") and len(open("DATA_FORMAT.txt").read()) > 0
shim = subprocess.run(["python3", "-c", "print('nested-ran')"], capture_output=True, text=True)
out["nested_python3"] = {"returncode": shim.returncode, "stdout": shim.stdout, "stderr": shim.stderr}
out["rg"] = subprocess.run(["rg", "--version"], capture_output=True).returncode == 0
print("x1-python-ok " + repr(out))
"""


def preflight(attempt, row, command, env, cwd, gate):
    """Isolation probes, H6 liveness and a live test of the counter (preflight phase), then a fresh start."""
    attempt = Path(attempt)
    directory = attempt.parent
    outer = command[:command.index("--") + 1]
    ds = row["dataset"]
    denied = [RUNS, HERE, f.ABLATION, f.HERE, STARK / "pi_full_test", STARK / "dci_run", STARK / "private",
              STARK / "public", STARK.parents[1] / "STaRK/data", STARK / "corpus2", STARK / "kb", STARK / "corpus_views",
              directory / "result.json", gate.log_path, *[STARK / "corpus" / d for d in DATASETS if d != ds]]
    code = "import os\nfor path in " + repr([str(p) for p in denied]) + ":\n"
    code += (" try:\n  fd=os.open(path,os.O_RDONLY)\n except (PermissionError, FileNotFoundError):\n  continue\n"
             " else:\n  os.close(fd)\n  raise AssertionError('Readable denied path: '+path)\n")
    probe = subprocess.run(outer + ["/usr/bin/python3", "-S", "-c", code], cwd=cwd, env=env,
                           capture_output=True, text=True, timeout=60)
    if probe.returncode:
        raise RuntimeError("X1 outer isolation probe failed: " + probe.stderr[-2000:])
    checks = {"outer_denied_paths": len(denied)}
    live = shell(command, env, cwd, "echo h6-ok; exit 5", 60)
    if live.returncode != 5 or live.stdout.strip() != "h6-ok":
        raise RuntimeError("X1 H6 wrapper probe failed: " + live.stdout[-500:] + live.stderr[-500:])
    labels = STARK / "private" / ds / "test-0.1.jsonl"
    inner = shell(command, env, cwd, " ; ".join([
        f"cat {shlex.quote(str(labels))} >/dev/null 2>&1 && echo LEAK-labels",
        f"ls {shlex.quote(str(RUNS))} >/dev/null 2>&1 && echo LEAK-runs",
        f"cat {shlex.quote(str(directory / 'result.json'))} >/dev/null 2>&1 && echo LEAK-result",
        f"cat {shlex.quote(str(gate.log_path))} >/dev/null 2>&1 && echo LEAK-gatelog",
        "head -c 1 DATA_FORMAT.txt >/dev/null && echo corpus-ok"]), 60)
    if "LEAK" in inner.stdout or "corpus-ok" not in inner.stdout:
        raise RuntimeError("X1 inner isolation probe failed: " + inner.stdout[-500:] + inner.stderr[-500:])
    # The label path goes in through the environment, so the counter log (argv only) never contains it.
    first = shell(command, env, cwd, f"X1P_LABELS={shlex.quote(str(labels))} python3 -c " + shlex.quote(PYTHON_PROBE), 120)
    marker = re.search(r"x1-python-ok (\{.*\})", first.stdout)
    if first.returncode or not marker:
        raise RuntimeError("X1 first python3 start failed: " + first.stdout[-1000:] + first.stderr[-1000:])
    inside = ast.literal_eval(marker.group(1))
    expected = {"exec_python_binary": False, "exec_python_via_loader": False, "read_python_binary": False,
                "exec_awk": False, "exec_perl": False, "network": False, "labels": False, "corpus": True,
                "ld_preload": None, "lock_env": None, "executable": "/usr/bin/python3", "rg": True}
    wrong = {k: inside.get(k) for k, v in expected.items() if inside.get(k) != v}
    nested = inside.get("nested_python3") or {}
    if wrong or nested.get("returncode") != 126 or nested.get("stdout") or nested.get("stderr") != REFUSAL:
        raise RuntimeError(f"X1 interpreter lock probe failed: {wrong} {nested}")
    # H6 (cap_filter) merges stderr into stdout, as the model sees it.
    second = shell(command, env, cwd, "python3 -c 'print(\"x1-second-ran\")'; echo rc=$?", 60)
    if "x1-second-ran" in second.stdout or second.stdout != REFUSAL + "rc=126\n":
        raise RuntimeError("X1 second python3 start was not refused: " + second.stdout[-500:] + second.stderr[-500:])
    others = shell(command, env, cwd, "awk 1 </dev/null; perl -e 1; node -e 1; /usr/bin/python3 -c 1; "
                   "python3.12 -c 1; /lib64/ld-linux-x86-64.so.2 /usr/bin/python3.12 -c 1; echo end", 60)
    if not others.stdout.endswith("\nend\n") or others.stdout.count("\n") < 7:
        raise RuntimeError("X1 interpreter denial probe failed: " + others.stdout[-500:] + others.stderr[-500:])
    summary = gate.summary("preflight")
    # The nested start inside Python never reaches the gate (subprocess closes the request descriptor); the shim
    # refuses it by itself. The gate sees the first start (granted) and the second command (refused).
    if (summary["granted"], summary["refused"], summary["started"]) != (1, 1, 1) or \
            [e.get("exit_code") for e in summary["exit"]] != [0]:
        raise RuntimeError(f"X1 counter log differs from the preflight probes: {summary}")
    gate.begin_agent_phase()
    checks.update(labels_and_other_runs="denied", h6_wrapper="passed", inner_isolation="passed",
                  first_python3="started under the interpreter lock", nested_python3="refused",
                  second_python3="refused", other_interpreters="denied", counter_preflight=summary,
                  counter_reset_for_agent=True)
    return checks


# ---------------------------------------------------------------- audits

def validate_tools(attempt):
    events = list(c.adapter.pi_context.requests(Path(attempt) / "artifacts/events.jsonl"))
    if not events:
        raise AssertionError("No provider requests")
    for _, _, event in events:
        names = {tool.get("name", tool.get("function", {}).get("name")) for tool in event["payload"].get("tools", [])}
        if names != {"read", "bash"}:
            raise AssertionError(f"X1 provider tools differ: {sorted(names)}")
    return {"requests": len(events), "tools": ["bash", "read"], "status": "passed"}


def trajectory_audit(attempt):
    """Bash commands that mention python or other interpreters, and outputs with the fixed refusal."""
    path = Path(attempt) / "artifacts/conversation.json"
    if not path.exists():
        return None
    conversation = json.loads(path.read_text())
    calls, outputs = {}, {}
    for message in conversation.get("messages", []):
        content = message.get("content") if isinstance(message.get("content"), list) else []
        if message.get("role") == "assistant":
            for item in content:
                if item.get("type") == "toolCall" and item.get("name") == "bash":
                    calls[item.get("id")] = (item.get("arguments") or {}).get("command") or ""
        elif message.get("role") == "toolResult":
            outputs[message.get("toolCallId")] = "".join(i.get("text", "") for i in content if isinstance(i, dict))
    python_calls, other_calls = [], 0
    for call_id, command in calls.items():
        programs = commands.programs(command)
        if any(commands.PYTHON.match(p) for p in programs) or "python" in command:
            python_calls.append(call_id)
        other_calls += any(p in commands.INTERPRETERS for p in programs)
    first = next(iter(python_calls), None)
    return {"bash_calls": len(calls), "commands_mentioning_python": len(python_calls),
            "commands_running_other_interpreters": other_calls,
            "outputs_with_refusal": sum(REFUSAL.strip() in outputs.get(k, "") for k in calls),
            "first_python_command_chars": len(calls[first]) if first else None,
            "first_python_command_lines": calls[first].count("\n") + 1 if first else None}


# ---------------------------------------------------------------- one attempt / one item

def run_attempt(job, row, gold, candidates, protocol, attempt):
    result = {**job, "harness": "pi", "prompt_base": "P1-R1 + X1 sentence", "status": "running", "terminal": False,
              "attempt_path": str(attempt), "started_at": now(), "metrics": {}}
    save(attempt / "result.json", result)
    gate, phase, started = None, "preflight", time.monotonic()
    try:
        gate = gate_service.Gate(attempt, attempt.parent / f"{attempt.name}.python_gate.jsonl")
        command, env, cwd = prepare(attempt, row, protocol)
        result["preflight"] = preflight(attempt, row, command, env, cwd, gate)
        if STOP.is_set():
            raise InterruptedError("Queue stopped before model call")
        phase = "agent"
        result["returncode"] = base.run_process(command, env, attempt, cwd, AGENT_TIMEOUT_SECONDS)
        if result["returncode"]:
            raise RuntimeError("Agent exit " + str(result["returncode"]))
        result.update(pi_transport.collect(attempt, "pi"))
        phase = "provider_tools"
        result["provider_tools"] = validate_tools(attempt)
        phase = "scoring"
        result["metrics"] = stark.score(result["prediction"], gold, candidates)
        result["returned_id_count"] = result["metrics"]["returned_count"]
        result["status"] = "scored"
    except Exception as error:
        result.update(status="parse_failed" if phase == "scoring" else "failed", error_phase=phase,
                      error_type=type(error).__name__, error=redact(error))
        state = attempt / "artifacts/state.json"
        if state.exists():
            try:
                result["error"] += " " + redact(read(state).get("error", ""))
            except Exception:
                pass
        if phase in ("agent", "provider_tools", "scoring"):
            try:
                result.update({k: v for k, v in pi_transport.collect(attempt, "pi", require_completed=False).items()
                               if k not in result or phase == "agent"})
            except Exception:
                pass
        # A stop signal can reach the agent directly (exit 143) while only STOP is set; such an attempt is
        # interrupted, not a failure.
        if (HARD_STOP.is_set() or STOP.is_set()) and phase in ("agent", "preflight") or isinstance(error, InterruptedError):
            result["status"] = "interrupted"
        result["retry_reason"] = None if result["status"] == "interrupted" else stark.retry_reason(result, attempt)
        if result.get("error_phase") == "preflight" and result.get("error_type") in ("TimeoutExpired", "InterruptedError"):
            result["retry_reason"] = "preflight_timeout"
            result["uncounted_attempt"] = True
        result["provider_refusal"] = f.refusal(result)
        result["rate_limit_exhausted"] = any(w in result["error"].lower() for w in ("429", "rate limit", "concurrency limit"))
    finally:
        if gate:
            result["python_gate"] = gate.summary("agent")
            gate.close()
    audits = (("tool_usage", lambda: f.ablation_runner.issued(attempt, ARM)), ("h6_audit", lambda: f.h6_audit(attempt)),
              ("python_trajectory", lambda: trajectory_audit(attempt)))
    for name, method in audits:
        try:
            if (attempt / "artifacts/events.jsonl").exists():
                result[name] = method()
        except Exception as error:
            result[name + "_error"] = redact(error)
    result["attempt_cost_usd"] = result.get("observed_cost_usd_lower_bound")
    result.update(finished_at=now(), wall_time_seconds=time.monotonic() - started)
    save(attempt / "result.json", result)
    return result


def try_recover(job, directory, previous, gold, candidates):
    """A crashed attempt whose agent already finished is scored without a new model call."""
    attempt = Path(previous.get("attempt_path", ""))
    if not attempt.is_dir() or not f.has_final(attempt):
        return None
    try:
        recovered = pi_transport.collect(attempt, "pi")
        tools = validate_tools(attempt)
        metrics = stark.score(recovered["prediction"], gold, candidates)
    except Exception:
        return None
    result = {**previous, **recovered, "provider_tools": tools, "metrics": metrics,
              "returned_id_count": metrics["returned_count"], "status": "scored",
              "recovered_without_new_model_call": True}
    log = attempt.parent / f"{attempt.name}.python_gate.jsonl"
    if log.exists() and "python_gate" not in result:
        result["python_gate"] = gate_service.summarize_log(log, "agent")
    for name, method in (("tool_usage", lambda: f.ablation_runner.issued(attempt, ARM)), ("h6_audit", lambda: f.h6_audit(attempt)),
                         ("python_trajectory", lambda: trajectory_audit(attempt))):
        try:
            result[name] = method()
        except Exception as error:
            result[name + "_error"] = redact(error)
    result["attempt_cost_usd"] = result.get("observed_cost_usd_lower_bound")
    save(attempt / "result.json", result)
    return f.finalize(job, directory, result)


def execute(job, row, gold, candidates, protocol, on_rate_limit=None):
    """Run one item to a terminal state (or leave it non-terminal only if the queue is stopping)."""
    directory = result_path(job).parent
    with base.job_lock(directory):
        previous = read_optional(directory / "result.json")
        if previous and previous.get("terminal") is True:
            return previous
        if previous and previous.get("status") == "running":
            recovered = try_recover(job, directory, previous, gold, candidates)
            if recovered:
                return recovered
        while True:
            dirs = f.attempts(directory)
            last = read_optional(dirs[-1] / "result.json") if dirs else None
            if dirs and (f.counted_attempts(directory) >= MAX_TECHNICAL_ATTEMPTS or len(dirs) >= MAX_ATTEMPT_DIRS):
                return f.finalize(job, directory, last or {**job, "status": "failed", "error": "attempt crashed"},
                                  "attempts_exhausted")
            if STOP.is_set():
                if previous or dirs:
                    save(directory / "result.json", {**(last or previous or job), "terminal": False, "status": "interrupted"})
                return None
            attempt = directory / f"attempt_{len(dirs) + 1:03d}"
            attempt.mkdir(parents=True)
            save(directory / "result.json", {**job, "status": "running", "terminal": False,
                                            "attempt_path": str(attempt), "started_at": now()})
            result = run_attempt(job, row, gold, candidates, protocol, attempt)
            if result["status"] == "scored":
                return f.finalize(job, directory, result)
            if result["status"] == "interrupted":
                save(directory / "result.json", {**result, "terminal": False})
                return None
            if result.get("provider_refusal"):
                return f.finalize(job, directory, result, "provider_refusal")
            if result.get("retry_reason") is None:
                return f.finalize(job, directory, result, "non_retryable_" + str(result.get("error_phase")))
            if f.counted_attempts(directory) >= MAX_TECHNICAL_ATTEMPTS:
                return f.finalize(job, directory, result, "attempts_exhausted")
            if len(f.attempts(directory)) >= 10:
                return f.finalize(job, directory, result, "preflight_timeouts_exhausted")
            if result.get("rate_limit_exhausted") and on_rate_limit:
                on_rate_limit(job)
            delay = min(300, 60 * 2 ** (f.counted_attempts(directory) - 1))
            print(f"RETRY {jobkey(job)} reason={result['retry_reason']} in {delay}s", flush=True)
            if STOP.wait(delay):
                save(directory / "result.json", {**result, "terminal": False})
                return None


kill_active_agents = f.kill_active_agents
