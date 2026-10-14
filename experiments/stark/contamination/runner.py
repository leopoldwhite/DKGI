"""Closed-book ID and name probes (A0-ID, A0-Name): smoke gate and the full no-tools probe queue on test."""

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time

from runtime import ROOT, base, pv, prepare, collect
from prompts import prompt_for
from score_names import parse, score as score_names

COUNTS = {"amazon": 1642, "mag": 2665, "prime": 2801}
GROUPS = [(probe, dataset) for probe in ("A0-ID", "A0-Name") for dataset in COUNTS
          if probe == "A0-ID" or dataset != "amazon"]
STOP = threading.Event()
FATAL = re.compile(r"\b(?:400|401|402|403|404|405|422)\b|invalid.?api.?key|unauthori[sz]ed|authentication|insufficient.?quota|model.not.found|unknown model|unsupported|unexpected argument|unrecognized argument", re.I)
RATE = re.compile(r"rate.?limit|concurrency limit|too many requests|\b429\b", re.I)
RUN_NAME = "contamination-20260926"
# Plain experiment parameters compared on resume.
PARAMETERS = ("protocol", "schedule", "max_workers", "max_attempts", "native_retry", "pricing_source")


class Halt(RuntimeError):
    """Known fatal condition; the run stops without an automatic retry."""


def key(job):
    return f"{job['probe']}:{job['dataset']}:{job['query_id']}"


def result_path(job):
    return ROOT / "runs" / job["probe"] / job["dataset"] / "test" / job["query_id"] / "result.json"


def issue(message):
    with (ROOT / "issues.log").open("a") as out:
        out.write(f"\n- {base.now()}: {message}\n")


def load_data():
    rows, gold, candidates, indexes = {}, {}, {}, {}
    for dataset, count in COUNTS.items():
        public = base.read_jsonl(ROOT.parent / "public" / dataset / "test.jsonl")
        labels = base.read_jsonl(ROOT.parent / "private" / dataset / "test.jsonl")
        assert len(public) == len(labels) == count
        for row in public:
            assert row["dataset"] == dataset and row["split"] == "test" and row["query_id"].isdigit()
            assert (dataset, row["query_id"]) not in rows
            rows[dataset, row["query_id"]] = row
        for row in labels:
            gold[dataset, row["query_id"]] = row["answer_ids"]
        candidates[dataset] = {int(v) for v in (ROOT.parent / "corpus" / dataset / "candidate_ids.txt").read_text().split()}
        if dataset != "amazon":
            path = ROOT / "index" / f"{dataset}.json"
            indexes[dataset] = base.read_json(path)["names"]
            assert isinstance(indexes[dataset], dict) and indexes[dataset]
    assert rows.keys() == gold.keys()
    return rows, gold, candidates, indexes


def schedule(rows):
    smoke, rest = [], []
    for probe, dataset in GROUPS:
        questions = sorted((r for r in rows.values() if r["dataset"] == dataset), key=lambda r: int(r["query_id"]))
        random.Random(20260926).shuffle(questions)
        for i, row in enumerate(questions):
            (smoke if i < 3 else rest).append({"probe": probe, "dataset": dataset, "query_id": row["query_id"]})
    random.Random(20260926).shuffle(smoke)
    random.Random(20260926).shuffle(rest)
    assert len(smoke) == 15 and len(smoke + rest) == 12574
    return smoke, rest


def record_configuration(protocol, jobs):
    """Record the probe configuration as readable parameters and refuse a changed configuration.

    The inference files (harness adapter, Pi build, and this directory's runner, prompts, scoring and name
    indexes) are listed by path."""
    paths = [ROOT / name for name in ("runner.py", "runtime.py", "prompts.py", "score_names.py", "name_index.py")]
    paths += sorted((ROOT / "index").glob("*.json"))
    identity = {"experiment": RUN_NAME, "protocol": protocol, "schedule": jobs, "max_workers": 8, "max_attempts": 6,
                "native_retry": False, "files": [str(p) for p in paths],
                "pricing_source": ("archived gpt-6-luna list rates: input 0.10, cached input 0.01, output 0.50 USD/million"
                                   if protocol["provider"] == "openai" and protocol["model"] == "gpt-6-luna" else "unknown; cost not estimated")}
    manifest = ROOT / "manifest.json"
    if manifest.exists():
        saved = base.read_json(manifest)
        if any(saved.get(name) != identity[name] for name in PARAMETERS):
            raise Halt("Probe configuration changed")
    else:
        base.save(manifest, identity)
    return RUN_NAME


def score_result(result, job, gold, candidates, indexes):
    parsed = parse(result["prediction"], job["probe"])
    result["parsed"] = parsed
    if job["probe"] == "A0-Name":
        result["metrics"] = score_names(parsed, gold, indexes[job["dataset"]])
    else:
        metrics = pv.stark.score(result["prediction"], gold, candidates[job["dataset"]])
        metrics["legal_candidate_count"] = sum(n in candidates[job["dataset"]] for n in parsed)
        result["metrics"] = metrics
    result["status"] = "scored"


def finalize(result, directory):
    history = [base.read_json(p) for p in sorted(directory.glob("attempt_*/result.json"))]
    result.update(terminal=True, attempt_count=len(history),
                  total_wall_seconds=sum(r.get("wall_time_seconds", 0) for r in history),
                  total_input_tokens=sum(r.get("input_tokens") or 0 for r in history),
                  total_output_tokens=sum(r.get("output_tokens") or 0 for r in history),
                  cost_usd=base.total_cost(history) if len(history) == len(list(directory.glob("attempt_*"))) else None,
                  unknown_cost_attempts=sum(r.get("attempt_cost_usd") is None for r in history) + len(list(directory.glob("attempt_*"))) - len(history),
                  usage_incomplete=any(not r.get("usage_complete") for r in history))
    base.save(directory / "result.json", result)
    return result


def execute(job, row, gold, candidates, indexes, protocol, run_name):
    directory = result_path(job).parent
    with base.job_lock(directory):
        config = {"job": job, "prompt": prompt_for(row, job["probe"])}
        if (directory / "config.json").exists():
            saved = base.read_json(directory / "config.json")
            if {name: saved.get(name) for name in config} != config:
                raise Halt("Attempt configuration changed: " + str(directory))
        base.save(directory / "config.json", config)
        previous = base.read_json(directory / "result.json") if (directory / "result.json").exists() else {}
        if previous.get("terminal"):
            return previous
        attempts = sorted(directory.glob("attempt_*"))
        # Recover a completed provider response after an interrupted run, without calling again.
        if attempts and not (attempts[-1] / "result.json").exists():
            attempt = attempts[-1]
            recovered = {**job, "attempt_path": str(attempt), "runtime_revision": run_name, "recovered": True, "fatal": False}
            try:
                recovered.update(collect(attempt))
                score_result(recovered, job, gold, candidates, indexes)
                recovered["attempt_cost_usd"] = pv.cost(recovered, "pi")
                state = base.read_json(attempt / "artifacts/state.json")
                from datetime import datetime
                recovered["wall_time_seconds"] = (datetime.fromisoformat(state["finished_at"]) - datetime.fromisoformat(state["started_at"])).total_seconds()
                base.save(attempt / "result.json", recovered)
                return finalize(recovered, directory)
            except AssertionError as exc:
                raise Halt(str(exc)) from exc
            except (OSError, ValueError, KeyError, RuntimeError):
                recovered.update(status="interrupted", usage_complete=False, error="Interrupted attempt could not be recovered")
                base.save(attempt / "result.json", recovered)
        if attempts and (attempts[-1] / "result.json").exists():
            prior = base.read_json(attempts[-1] / "result.json")
            if prior.get("status") == "scored" or prior.get("fatal") or len(attempts) >= 6:
                return finalize(prior, directory)
        for number in range(len(attempts) + 1, 7):
            if STOP.is_set():
                return {**job, "status": "interrupted", "terminal": False}
            attempt = directory / f"attempt_{number:03d}"
            attempt.mkdir()
            started = time.monotonic()
            result = {**job, "attempt_path": str(attempt), "runtime_revision": run_name, "status": "running",
                      "started_at": base.now(), "fatal": False, "terminal": False, "metrics": {}}
            base.save(directory / "result.json", result)
            phase = "preflight"
            try:
                command, env = prepare(attempt, config["prompt"], protocol)
                phase = "agent"
                result["returncode"] = base.run_process(command, env, attempt, attempt / "cwd", protocol["timeout_seconds"])
                if result["returncode"]:
                    raise RuntimeError(f"Agent exit {result['returncode']}")
                result.update(collect(attempt))
                phase = "scoring"
                score_result(result, job, gold, candidates, indexes)
            except Exception as exc:
                result.update(status="parse_failed" if phase == "scoring" else "failed", error_phase=phase,
                              error_type=type(exc).__name__, error=str(exc))
                state_path = attempt / "artifacts/state.json"
                if phase == "agent" and state_path.exists():
                    try:
                        result.update(collect(attempt, require_completed=False))
                    except AssertionError as violation:
                        result.update(fatal=True, isolation_violation=str(violation))
                    except (OSError, ValueError, RuntimeError, KeyError):
                        pass
                    result["error"] += "\n" + str(base.read_json(state_path).get("error", ""))
                error_log = attempt / "stderr.txt"
                if error_log.exists():
                    result["error"] += "\n" + error_log.read_text(errors="replace")[-12000:]
                result["error"] = base.redact(result["error"])
                result["retry_reason"] = pv.stark.retry_reason(result, attempt)
                if RATE.search(result["error"]):
                    result["retry_reason"] = "rate_limit"
                result["fatal"] = result.get("fatal") or phase == "preflight" or isinstance(exc, AssertionError) or bool(FATAL.search(result["error"])) or result.get("retry_reason") is None
            result.update(finished_at=base.now(), wall_time_seconds=time.monotonic() - started,
                          attempt_cost_usd=pv.cost(result, "pi"))
            base.save(attempt / "result.json", result)
            if result["status"] == "scored" or result["fatal"] or number == 6:
                return finalize(result, directory)
            base.save(directory / "result.json", result)
            delay = random.uniform(.5, 1) * min(120, 10 * 2 ** (number - 1))
            print(f"RETRY {key(job)} attempt={number} delay={delay:.1f} reason={result['retry_reason']}", flush=True)
            STOP.wait(delay)
        raise Halt("Unexpected exhausted attempt state")


def snapshot(state, results, active):
    groups = {f"{p}:{d}": {"planned": COUNTS[d], "completed": 0, "scored": 0, "failed": 0} for p, d in GROUPS}
    for result in results.values():
        group = groups[f"{result['probe']}:{result['dataset']}"]
        group["completed"] += 1
        group["scored" if result["status"] == "scored" else "failed"] += 1
    state.update(updated_at=base.now(), heartbeat_at=base.now(), completed=len(results), running=len(active),
                 groups=groups, known_cost_usd=sum(r["cost_usd"] for r in results.values() if r.get("cost_usd") is not None),
                 unknown_cost_items=sum(r.get("cost_usd") is None for r in results.values()),
                 input_tokens=sum(r.get("total_input_tokens", 0) for r in results.values()),
                 output_tokens=sum(r.get("total_output_tokens", 0) for r in results.values()),
                 incomplete_usage_runs=sum(r.get("usage_incomplete", True) for r in results.values()),
                 active=[key(job) for job in active.values()])
    base.save(ROOT / "state.json", state)


def run_queue(jobs, rows, gold, candidates, indexes, protocol, run_name, results, state):
    pending = deque(j for j in jobs if key(j) not in results)
    active = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        try:
            while pending or active:
                while pending and len(active) < state["workers"] and not STOP.is_set():
                    job = pending.popleft()
                    identity = job["dataset"], job["query_id"]
                    active[pool.submit(execute, job, rows[identity], gold[identity], candidates, indexes, protocol, run_name)] = job
                snapshot(state, results, active)
                if STOP.is_set():
                    raise InterruptedError("Termination requested")
                done, _ = wait(active, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    job = active.pop(future)
                    result = future.result()
                    if not result.get("terminal"):
                        raise InterruptedError("Attempt interrupted")
                    results[key(job)] = result
                    print(f"DONE {len(results)}/12574 {key(job)} {result['status']}", flush=True)
                    snapshot(state, results, active)
                    if result.get("fatal"):
                        raise Halt(f"Fatal {key(job)}: {result.get('error', '')}")
                    if result["status"] != "scored":
                        before = state["workers"]
                        state["workers"] = max(1, before // 2)
                        issue(f"Retries exhausted for {key(job)}; concurrency reduced {before} -> {state['workers']}. Failure remains zero-scored.")
                        group = state["groups"][f"{job['probe']}:{job['dataset']}"]
                        if group["failed"] / group["completed"] > .05:
                            raise Halt(f"Group failure rate exceeds 5%: {job['probe']}:{job['dataset']}")
        finally:
            if active:
                STOP.set()
                pv.cancel_workers(active)
                for future, job in active.items():
                    if not future.cancelled() and future.done():
                        try:
                            result = future.result()
                            if result.get("terminal"):
                                results[key(job)] = result
                        except BaseException:
                            pass
                snapshot(state, results, {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print((ROOT / "state.json").read_text())
        return
    with base.job_lock(ROOT):
        protocol = {**pv.protocol(harness="pi"), "pi_max_turns": 300}
        rows, gold, candidates, indexes = load_data()
        smoke, rest = schedule(rows)
        run_name = record_configuration(protocol, smoke + rest)
        base.save(ROOT / "schedule.json", {"smoke": smoke, "formal": rest})
        if not args.execute:
            print(json.dumps({"planned": 12574, "smoke": 15, "experiment": run_name}))
            return
        state_path = ROOT / "state.json"
        state = base.read_json(state_path) if state_path.exists() else {"started_at": base.now(), "workers": 8}
        if state.get("status") == "blocked":
            raise Halt("Persisted blocked state requires review; refusing automatic resume")
        state.update(status="running", planned=12574, pid=os.getpid(), runtime_revision=run_name)
        results = {}
        for job in smoke + rest:
            path = result_path(job)
            if path.exists():
                result = base.read_json(path)
                if any(result.get(name) != value for name, value in job.items()):
                    raise Halt("Cached job identity mismatch")
                if result.get("terminal"):
                    results[key(job)] = result
        snapshot(state, results, {})
        try:
            base.require_api_key(protocol)
        except ValueError as error:
            state.update(status="blocked", error=str(error))
            snapshot(state, results, {})
            raise Halt(state["error"])
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: STOP.set())
        try:
            state["phase"] = "smoke"
            run_queue(smoke, rows, gold, candidates, indexes, protocol, run_name, results, state)
            if any(results[key(j)]["status"] != "scored" for j in smoke):
                raise Halt("Smoke did not pass all 15 runs")
            base.save(ROOT / "smoke.json", {"status": "passed", "checked_at": base.now(), "jobs": [results[key(j)] for j in smoke]})
            state.update(phase="formal", smoke="passed")
            run_queue(rest, rows, gold, candidates, indexes, protocol, run_name, results, state)
            state.update(status="completed", phase="analysis", finished_at=base.now())
            snapshot(state, results, {})
            subprocess.run([sys.executable, str(ROOT / "analyze.py")], check=True)
            state["phase"] = "complete"
        except Halt as exc:
            state.update(status="blocked", error=str(exc), stopped_at=base.now())
            issue(str(exc))
            raise
        except InterruptedError as exc:
            state.update(status="interrupted", error=str(exc))
            raise
        except BaseException as exc:
            state.update(status="crashed", error=type(exc).__name__ + ": " + str(exc))
            raise
        finally:
            snapshot(state, results, {})


if __name__ == "__main__":
    try:
        main()
    except Halt as exc:
        print("BLOCKED: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(78)
