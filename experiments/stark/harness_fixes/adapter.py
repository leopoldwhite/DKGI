"""Harness of every Pi and Codex run: Pi transport, Landlock launcher, preflight and observability around the STaRK adapter."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "prompt_val"))
import runner
import pi_transport
import launcher
import codex_observability
import pi_context

sys.path.insert(0, str(ROOT))
original_collect = runner.harness.collect


def prepare_attempt(attempt, row, protocol, harness, prompt_text):
    prepare = pi_transport.prepare_attempt if harness == "pi" else pi_transport.original_prepare
    command, env = prepare(attempt, row, protocol, harness, prompt_text)
    command, env = launcher.configure(command, env, attempt, harness)
    command, env = codex_observability.configure(command, env, attempt, harness)
    launcher.preflight(command, env, attempt, row)
    runner.base.save(attempt / "command.json", command)
    runner.base.save(attempt / "harness_fixes.json", {
        "tool_home": str(attempt / "tool_home"), "system_readonly": launcher.SYSTEM_READS,
        "codex_trace_root": env.get("CODEX_ROLLOUT_TRACE_ROOT"),
        "main_home_unchanged": True, "prompt_and_output_limits_unchanged": True})
    return command, env


def collect(attempt, harness, require_completed=True):
    method = pi_transport.collect if harness == "pi" else original_collect
    result = method(attempt, harness, require_completed=require_completed)
    if require_completed:
        result["observability"] = (pi_context.export(attempt) if harness == "pi"
                                   else codex_observability.export(attempt))
    return result


def install():
    runner.harness.prepare_attempt = prepare_attempt
    runner.harness.collect = collect
    runner.finalize = pi_transport.finalize
