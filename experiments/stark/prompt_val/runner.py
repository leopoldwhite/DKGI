"""Run protocol, cost and result finalization shared by the STaRK runners."""

from concurrent.futures import wait
import os
from pathlib import Path
import signal
import subprocess
import threading

# Codex CLI 0.156.1 was used for every reported run; CODEX_BIN points to that binary if it is installed elsewhere.
PINNED_CODEX = Path(os.environ.get("CODEX_BIN") or
                    Path.home() / ".codex/packages/standalone/releases/0.156.1-x86_64-unknown-linux-musl/bin/codex")
os.environ["PATH"] = str(PINNED_CODEX.parent) + os.pathsep + os.environ["PATH"]
CODEX_VERSION = "codex-cli 0.156.1"

import variants  # noqa: E402
import harness  # noqa: E402

stark, base = variants.stark, variants.stark.base
ROOT = Path(__file__).resolve().parent
STOP = threading.Event()


def protocol(harness=None):
    """Official provider settings, preserving paper defaults unless explicitly overridden."""
    configured = base.configure_protocol(base.read_json(base.ROOT / "protocol.json"), harness=harness)
    configured.update(seed=stark.prepare.SEED, max_returned_ids=100,
                      scoring="Hit/Recall at k; truncated MRR@100; invalid and duplicate IDs occupy rank with no gain")
    if harness == "codex":
        configured["codex_version"] = subprocess.check_output([str(base.CODEX), "--version"], text=True).strip()
        if configured["codex_version"] != CODEX_VERSION:
            raise ValueError(f"Expected {CODEX_VERSION}, found {configured['codex_version']} at {base.CODEX}")
    return configured


def guard_run_configuration(path, protocols, results_directory=None):
    """Do not reuse terminal results across provider/model changes."""
    path = Path(path)
    identity = {h: {k: p[k] for k in ("provider", "model", "api", "reasoning_effort", "context_window", "max_tokens")} for h, p in protocols.items()}
    if path.exists():
        if base.read_json(path) != identity:
            raise ValueError("Run inference configuration changed; use a new output directory")
    else:
        changed = os.environ.get("MODEL_NAME") or any(p["provider"] != "openai" or p["model"] not in ("gpt-6-luna", "gpt-6-astra") for p in protocols.values())
        if changed and any(Path(results_directory or path.parent).glob("**/result.json")):
            raise ValueError("Existing results have no provider identity; use a new output directory")
        base.save(path, identity)


def key(row):
    return f"{row['dataset']}:{row['query_id']}"


def job_key(job):
    return f"{job['group']}:{job['dataset']}:{job['query_id']}"


def load_data():
    rows, labels, candidates, sources = {}, {}, {}, []
    for dataset in stark.prepare.DATASETS:
        public = stark.ROOT / "public" / dataset / "val.jsonl"
        private = stark.ROOT / "private" / dataset / "val.jsonl"
        for row in base.read_jsonl(public):
            if row["dataset"] != dataset or row["split"] != "val" or key(row) in rows:
                raise ValueError("Invalid or duplicate val row")
            rows[key(row)] = row
        for row in base.read_jsonl(private):
            labels[f"{dataset}:{row['query_id']}"] = row["answer_ids"]
        candidate_file = stark.ROOT / "corpus" / dataset / "candidate_ids.txt"
        candidates[dataset] = {int(v) for v in candidate_file.read_text().split()}
        sources.extend((public, private, candidate_file))
    if rows.keys() != labels.keys():
        raise ValueError("Val questions and labels differ")
    return rows, labels, candidates, sources


def cost(result, harness_name):
    provider = os.environ.get("MODEL_PROVIDER", "openai")
    model = os.environ.get("MODEL_NAME", os.environ.get("DCI_MODEL", ""))
    if provider != "openai" or model != "gpt-6-luna":
        return None
    if not result.get("usage_complete"):
        return None
    raw = result["usage_raw"]
    if harness_name == "pi":
        return (raw["input"] * .10 + raw["cacheRead"] * .01 + raw["cacheWrite"] * .125 + raw["output"] * .50) / 1e6
    return ((raw["input_tokens"] - raw["cached_input_tokens"]) * .10 +
            raw["cached_input_tokens"] * .01 + raw["output_tokens"] * .50) / 1e6


def format_cost(value):
    return "unknown" if value is None else f"${value:.4f}"


def finalize(result, directory):
    history = [base.read_json(p) for p in sorted(directory.glob("attempt_*/result.json"))]
    result.update(terminal=True, attempt_count=len(list(directory.glob("attempt_*"))),
                  total_wall_seconds=sum(r.get("wall_time_seconds", 0) for r in history),
                  total_input_tokens=sum(r.get("input_tokens") or 0 for r in history),
                  total_output_tokens=sum(r.get("output_tokens") or 0 for r in history),
                  cost_usd=base.total_cost(history) if len(history) == len(list(directory.glob("attempt_*"))) else None,
                  unknown_cost_attempts=sum(r.get("attempt_cost_usd") is None for r in history) + len(list(directory.glob("attempt_*"))) - len(history),
                  usage_incomplete=len(history) < len(list(directory.glob("attempt_*"))) or
                  any(not r.get("usage_complete") for r in history))
    stark.prepare.save(directory / "result.json", result)
    return result


def cancel_workers(active):
    STOP.set()
    for future in active:
        future.cancel()
    signalled = set()
    while any(not future.done() for future in active):
        # Each run_process child is a subreaper that cleans up its own tools.
        for task in Path("/proc/self/task").iterdir():
            try:
                children = (task / "children").read_text().split()
            except FileNotFoundError:
                continue
            for value in children:
                pid = int(value)
                if pid in signalled:
                    continue
                try:
                    if b"PR_SET_CHILD_SUBREAPER" not in Path(f"/proc/{pid}/cmdline").read_bytes():
                        continue
                    os.kill(pid, signal.SIGTERM)
                    signalled.add(pid)
                except ProcessLookupError:
                    pass
                except FileNotFoundError:
                    pass
        wait(active, timeout=.1)
