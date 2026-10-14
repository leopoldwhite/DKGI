"""Pi transport used by every Pi run: lossless RPC backpressure (blocking stdout) and native request retries."""

from datetime import datetime
from pathlib import Path
import random
import shutil

import runner

PARENT = Path(__file__).resolve().parent
TRANSPORT = PARENT / "transport"

RETRY = {"enabled": True, "maxRetries": 6, "baseDelayMs": 15000, "maxDelayMs": 120000}
original_prepare = runner.harness.prepare_attempt
original_finalize = runner.finalize


def prepare_attempt(attempt, row, protocol, harness, prompt_text):
    command, env = original_prepare(attempt, row, protocol, harness, prompt_text)
    if harness != "pi":
        raise ValueError("This transport accepts only Pi")
    for name in ("blocking_stdout.cjs", "pi_rpc_retry.py"):
        shutil.copyfile(TRANSPORT / name, attempt / name)
    index = command.index("dci.benchmark.pi_rpc_runner")
    if command[index - 1] != "-m":
        raise ValueError("Unexpected Pi launch command")
    command[index - 1:index + 1] = [str(attempt / "pi_rpc_retry.py")]
    env["NODE_OPTIONS"] = "--require=" + str(attempt / "blocking_stdout.cjs")
    settings = runner.base.read_json(attempt / "agent/settings.json")
    settings["retry"] = {**RETRY, "baseDelayMs": random.randint(10000, 15000)}
    runner.base.save(attempt / "agent/settings.json", settings)
    runner.base.save(attempt / "command.json", command)
    runner.base.save(attempt / "transport.json", {"node_options": env["NODE_OPTIONS"], "retry": settings["retry"]})
    return command, env


def collect(attempt, harness, require_completed=True):
    result = runner.base.collect(attempt, harness, require_completed=require_completed)
    state = runner.base.read_json(attempt / "artifacts/state.json")
    messages = [e["message"] for e in state.get("messages", [])
                if e.get("event") == "message_end" and e.get("message", {}).get("role") == "assistant"]
    good = [m for m in messages if m.get("stopReason") not in {"error", "aborted"}]
    complete = state.get("status") == "completed" and bool(good) and all(
        all(type(m.get("usage", {}).get(k)) is int and m["usage"][k] >= 0
            for k in ("input", "output", "cacheRead", "cacheWrite")) for m in good)
    if require_completed and not complete:
        raise RuntimeError("Missing usage in successful Pi responses")
    result.update(usage_complete=complete, retry_usage_incomplete=len(good) != len(messages),
                  failed_model_responses=len(messages) - len(good),
                  cached_input_tokens=result["usage_raw"]["cacheRead"],
                  usage_notes="Successful response usage validated; failed/retried response usage may be unreported.")
    result["observed_cost_usd_lower_bound"] = runner.cost({**result, "usage_complete": True}, "pi")
    return result


def finalize(result, directory):
    attempt = Path(result["attempt_path"])
    state_path = attempt / "artifacts/state.json"
    if state_path.exists():
        state = runner.base.read_json(state_path)
        if state.get("finished_at") and state.get("started_at"):
            duration = (datetime.fromisoformat(state["finished_at"]) - datetime.fromisoformat(state["started_at"])).total_seconds()
            if result.get("wall_time_seconds", 0) + 1 < duration:
                result.update(wall_time_seconds=duration, started_at=state["started_at"], timing_recovered_from_artifacts=True)
                runner.base.save(attempt / "result.json", result)
    result = original_finalize(result, directory)
    history = [runner.base.read_json(p) for p in directory.glob("attempt_*/result.json")]
    result["usage_incomplete"] |= any(r.get("retry_usage_incomplete", False) for r in history)
    costs = [{"attempt_cost_usd": r.get("observed_cost_usd_lower_bound")
              if r.get("observed_cost_usd_lower_bound") is not None else r.get("attempt_cost_usd")} for r in history]
    result["cost_usd"] = runner.base.total_cost(costs) if len(costs) == result["attempt_count"] else None
    result["unknown_cost_attempts"] = sum(r["attempt_cost_usd"] is None for r in costs) + result["attempt_count"] - len(costs)
    runner.base.save(directory / "result.json", result)
    return result
