"""One program runner (X1, arm O1): Pi (L3 + H6), final prompt + one sentence, Shell only allowlist + python3 started once.

  run.py --check                       offline checks only (no model calls) -> checks/offline.json
  run.py --smoke                       val smoke: 1 unused val question per dataset -> checks/smoke.json
  run.py --split test-0.1 --workers N  formal run (710 questions); resumable

Items are identified by arm/dataset/split/query_id; an item whose result.json is terminal is never run again.
SIGTERM/SIGINT once: stop admitting, let active runs finish. Twice: terminate active agents (their
attempts are marked interrupted and do not use up the attempt budget). Exit 3 = shared fault stop.
"""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import os
from pathlib import Path
import signal
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import x1 as x  # noqa: E402

SHARED_FAULT_STREAK = 3


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=("val", "test-0.1"), default="val")
    p.add_argument("--datasets", default=",".join(x.DATASETS))
    p.add_argument("--limit", type=int, help="at most this many questions in total (round-robin over datasets)")
    p.add_argument("--per-dataset", type=int, help="at most this many questions per dataset")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--smoke", action="store_true", help="val smoke: 1 unused val question per dataset")
    p.add_argument("--check", action="store_true", help="offline checks only; no model calls")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    a.datasets = [d for d in a.datasets.split(",") if d]
    if set(a.datasets) - set(x.DATASETS) or not a.datasets:
        p.error("unknown or empty --datasets")
    if a.smoke:
        if a.split != "val":
            p.error("--smoke runs on val only")
        a.per_dataset = 1
        a.workers = min(a.workers, 3) if a.workers > 1 else 3
    if a.workers < 1:
        p.error("--workers must be >= 1")
    return a


def questions(args, rows):
    if args.split == "val":
        chosen = x.f.val_selection(rows)   # the val questions of ../ablation_final/selection_val.json
        if args.smoke:
            chosen = [q for q in chosen if q["smoke"]]
    else:
        chosen = x.f.test_questions()
    chosen = [q for q in chosen if q["dataset"] in args.datasets]
    if args.per_dataset:
        chosen = [q for d in args.datasets for q in [v for v in chosen if v["dataset"] == d][:args.per_dataset]]
    if args.limit:
        by = {d: [q for q in chosen if q["dataset"] == d] for d in args.datasets}
        chosen, i = [], 0
        while len(chosen) < args.limit and any(by.values()):
            d = args.datasets[i % len(args.datasets)]
            if by[d]:
                chosen.append(by[d].pop(0))
            i += 1
    return [{"dataset": q["dataset"], "query_id": str(q["query_id"])} for q in chosen]


def gate_test():
    for name in ("offline.json", "smoke.json"):
        value = x.read_optional(HERE / "checks" / name)
        if not value or value.get("status") != "passed":
            raise SystemExit(f"test-0.1 refused: checks/{name} is missing or not passed")


def persist_failure(job, error):
    path = x.result_path(job)
    previous = x.read_optional(path) or {}
    x.save(path, {**previous, **job, "status": "failed", "terminal": True, "error_phase": "queue_boundary",
                  "error_type": type(error).__name__, "error": x.redact(repr(error)), "metrics": {},
                  "failure_kind": "queue_boundary_exception", "finished_at": x.now(),
                  "cost_usd": previous.get("cost_usd", 0.0), "total_input_tokens": previous.get("total_input_tokens", 0),
                  "total_output_tokens": previous.get("total_output_tokens", 0), "tool_calls": previous.get("tool_calls", 0)})


def shared_fault_suspect(result):
    if not result or result.get("status") == "scored" or result.get("provider_refusal"):
        return False
    if result.get("error_phase") in ("preflight", "provider_tools"):
        return result.get("error_type") not in ("TimeoutExpired", "InterruptedError")
    text = (result.get("error") or "").lower()
    return result.get("error_phase") == "agent" and any(
        w in text for w in ("invalid_api_key", "unauthorized", "unauthorised", "401", "insufficient_quota"))


def smoke_report(jobs):
    runs, ok = [], True
    for job in jobs:
        r = x.read_optional(x.result_path(job)) or {}
        try:
            audit = x.f.h6_audit(r["attempt_path"])
        except Exception:
            audit = r.get("h6_audit") or {}
        gate = r.get("python_gate") or {}
        pre = (r.get("preflight") or {}).get("counter_preflight") or {}
        entry = {"arm": job["arm"], "dataset": job["dataset"], "query_id": job["query_id"], "status": r.get("status"),
                 "hit_at_1": (r.get("metrics") or {}).get("hit_at_1"), "tool_calls": r.get("tool_calls"),
                 "attempts": r.get("attempt_count"), "cost_usd": r.get("cost_usd"),
                 **{f"h6_{k}": v for k, v in audit.items()},
                 "provider_tools": (r.get("provider_tools") or {}).get("tools"),
                 "preflight_first_python3": (r.get("preflight") or {}).get("first_python3"),
                 "preflight_second_python3": (r.get("preflight") or {}).get("second_python3"),
                 "preflight_counter": {k: pre.get(k) for k in ("granted", "refused", "started")},
                 "agent_counter": {k: gate.get(k) for k in ("launch_requests", "granted", "refused", "started", "exit", "notes")},
                 "python_trajectory": r.get("python_trajectory"), "error": r.get("error")}
        good = (r.get("status") == "scored" and bool(audit) and audit.get("l3_markers_bash") == 0
                and audit.get("bash_max_utf16_units", 0) <= x.f.L3_LIMIT_UNITS and audit.get("bash_over_limit") == 0
                and (r.get("tool_calls") or 0) > 0 and (gate.get("granted") or 0) <= 1 and not gate.get("notes")
                and entry["preflight_counter"] == {"granted": 1, "refused": 1, "started": 1})
        try:
            entry["visible_ablation_path_occurrences"] = x.f.visible_path_leaks(r["attempt_path"])
        except Exception:
            entry["visible_ablation_path_occurrences"] = None
        good = good and entry["visible_ablation_path_occurrences"] == 0
        entry["passed"] = good
        ok &= good
        runs.append(entry)
    return {"status": "passed" if ok else "failed", "runs": runs, "count": len(runs), "finished_at": x.now()}


def main():
    args = parse()
    if args.check:
        import offline_check
        offline_check.run()
        print("offline checks passed:", HERE / "checks/offline.json", flush=True)
        return 0
    rows, labels, candidates = x.f.load_split(args.split)
    chosen = questions(args, rows)
    jobs = x.schedule(chosen, args.split)
    pending = [j for j in jobs if not x.is_terminal(j)]
    print(f"{args.split}: {len(chosen)} questions, arm {x.ARM}; {len(jobs) - len(pending)} already terminal; "
          f"workers {args.workers}", flush=True)
    if args.dry_run:
        for job in pending[:30]:
            print(" ", x.jobkey(job))
        return 0
    if args.split == "test-0.1":
        gate_test()
    protocol = x.c.val.protocol(harness="pi")
    x.base.require_api_key(protocol)
    x.c.val.guard_run_configuration(HERE / "provider-protocol.json", {"pi": protocol}, x.RUNS)
    lock = (HERE / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another X1 runner holds run.lock")
    state_path = HERE / f"state_{'smoke' if args.smoke else args.split}.json"
    state = {"status": "running", "split": args.split, "arm": x.ARM, "planned": len(jobs), "workers": args.workers,
             "started_at": x.now(), "pid": os.getpid(), "active": [], "stop_reason": None}
    signals = Counter()

    def on_signal(signum, _frame):
        signals["n"] += 1
        x.STOP.set()
        state["stop_reason"] = f"signal {signum}"
        if signals["n"] >= 2:
            x.HARD_STOP.set()
            x.kill_active_agents()

    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, on_signal)

    def reduce_workers(job):
        old = state["workers"]
        state["workers"] = max(1, old // 2)
        print(f"RATE-LIMIT {x.jobkey(job)}: admission workers {old} -> {state['workers']}", flush=True)

    def snapshot():
        done = [x.read_optional(x.result_path(j)) for j in jobs]
        terminal = [r for r in done if r and r.get("terminal")]
        failed = sum(r.get("status") != "scored" for r in terminal)
        scored = [r for r in terminal if r.get("status") == "scored"]
        state.update(terminal=len(terminal), scored=len(scored), failed=failed,
                     failure_rate=failed / len(terminal) if terminal else None,
                     alert_over_2pct=bool(terminal) and failed / len(terminal) > 0.02,
                     hit_at_1=sum((r.get("metrics") or {}).get("hit_at_1") or 0 for r in scored) / len(scored) if scored else None,
                     python_started=sum(bool((r.get("python_gate") or {}).get("started")) for r in terminal),
                     python_refused_requests=sum((r.get("python_gate") or {}).get("refused") or 0 for r in terminal),
                     cost_usd=sum(r.get("cost_usd") or 0 for r in terminal), heartbeat=x.now())
        x.save(state_path, state)

    streak, running, queue = 0, {}, list(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while queue or running:
            while queue and not x.STOP.is_set() and len(running) < state["workers"]:
                job = queue.pop(0)
                key = x.key(job)
                running[pool.submit(x.execute, job, rows[key], labels[key], candidates[job["dataset"]], protocol,
                                    reduce_workers)] = job
            if not running:
                break
            state["active"] = [x.jobkey(j) for j in running.values()]
            snapshot()
            done, _ = wait(running, timeout=15, return_when=FIRST_COMPLETED)
            for future in done:
                job = running.pop(future)
                try:
                    result = future.result()
                except BlockingIOError:
                    print(f"SKIP {x.jobkey(job)}: locked by another process", flush=True)
                    continue
                except Exception as error:
                    try:
                        persist_failure(job, error)
                    except Exception as persist_error:
                        state["stop_reason"] = "cannot persist results: " + x.redact(persist_error)
                        x.STOP.set()
                    result = x.read_optional(x.result_path(job))
                if result is None:
                    print(f"STOPPED {x.jobkey(job)} (non-terminal, resumes later)", flush=True)
                    continue
                m, g = result.get("metrics") or {}, result.get("python_gate") or {}
                print(f"{x.jobkey(job)} {result.get('status')} H@1={m.get('hit_at_1')} attempts={result.get('attempt_count')} "
                      f"python started={g.get('started')} refused={g.get('refused')} ${result.get('cost_usd') or 0:.4f} "
                      f"{result.get('failure_kind') or ''}", flush=True)
                streak = streak + 1 if shared_fault_suspect(result) else 0
                if streak >= SHARED_FAULT_STREAK and not x.STOP.is_set():
                    state["stop_reason"] = (f"shared fault suspected: {streak} consecutive items failed at "
                                            f"{result.get('error_phase')}: {x.redact(result.get('error', ''))[:500]}")
                    state["shared_fault"] = True
                    x.STOP.set()
    state["active"] = []
    snapshot()
    remaining = [j for j in jobs if not x.is_terminal(j)]
    state.update(status="completed" if not remaining else ("fault_stop" if state.get("shared_fault") else "stopped"),
                 remaining=len(remaining), finished_at=x.now())
    x.save(state_path, state)
    if args.smoke and not remaining:
        report = smoke_report(jobs)
        x.save(HERE / "checks/smoke.json", report)
        print("smoke:", report["status"], HERE / "checks/smoke.json", flush=True)
    print(f"finished: {state['status']}; terminal {state['terminal']}/{len(jobs)}; ${state['cost_usd']:.4f}", flush=True)
    return 3 if state.get("shared_fault") else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as error:
        print(x.redact(repr(error)), file=sys.stderr)
        sys.exit(3)
