"""Tool-variant runner: arms A1 A2 A3 A5 A6 on Pi (final prompt P1-R1, H6 output cap).

  run.py --check                       offline checks only (no model calls)
  run.py --smoke                       5 arms x 3 datasets x 1 unused val question, 1 worker
  run.py --split test-0.1 --workers N  formal run (710 questions); resumable, interleaved by question

Items are identified by arm/dataset/query_id; an item whose result.json is terminal is never run again.
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
import threading
import time

HERE = Path(__file__).resolve().parent
VENV_PYTHON = HERE.parents[2] / "baselines/ark/.venv/bin/python"
# The ARK tool service (A1/A2/A5) needs the ARK virtualenv (polars etc.); re-exec under it if needed.
if Path(sys.prefix).resolve() != VENV_PYTHON.parents[1].resolve() and VENV_PYTHON.exists() and not os.environ.get("ABLATION_FINAL_NO_REEXEC"):
    os.environ["ABLATION_FINAL_NO_REEXEC"] = "1"
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), "-B", str(Path(__file__).resolve()), *sys.argv[1:]])

sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import final as f  # noqa: E402

SHARED_FAULT_STREAK = 3     # consecutive preflight / provider-tool / credential failures on different items


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", default=",".join(f.ARMS), help="comma list from " + ",".join(f.ARMS))
    p.add_argument("--split", choices=("val", "test-0.1"), default="val")
    p.add_argument("--datasets", default=",".join(f.DATASETS))
    p.add_argument("--limit", type=int, help="at most this many questions in total")
    p.add_argument("--per-dataset", type=int, help="at most this many questions per dataset")
    p.add_argument("--workers", type=int, default=1, help="concurrent items (each makes one model call at a time)")
    p.add_argument("--smoke", action="store_true", help="val smoke: 1 unused val question per dataset, 1 worker")
    p.add_argument("--check", action="store_true", help="offline checks only; no model calls")
    p.add_argument("--dry-run", action="store_true", help="print the schedule and exit")
    a = p.parse_args()
    a.arms = [x for x in a.arms.split(",") if x]
    a.datasets = [x for x in a.datasets.split(",") if x]
    if set(a.arms) - set(f.ARMS) or set(a.datasets) - set(f.DATASETS) or not a.arms or not a.datasets:
        p.error("unknown or empty --arms/--datasets")
    if a.smoke:
        if a.split != "val":
            p.error("--smoke runs on val only")
        a.per_dataset, a.workers = 1, 1
    if a.workers < 1:
        p.error("--workers must be >= 1")
    return a


def questions(args, rows):
    if args.split == "val":
        chosen = f.val_selection(rows)
        if args.smoke:
            chosen = [q for q in chosen if q["smoke"]]
    else:
        chosen = f.test_questions()
    chosen = [q for q in chosen if q["dataset"] in args.datasets]
    if args.per_dataset:
        chosen = [q for d in args.datasets for q in [x for x in chosen if x["dataset"] == d][:args.per_dataset]]
    if args.limit:
        # Keep datasets balanced under a total limit: round-robin across datasets.
        by = {d: [q for q in chosen if q["dataset"] == d] for d in args.datasets}
        chosen, i = [], 0
        while len(chosen) < args.limit and any(by.values()):
            d = args.datasets[i % len(args.datasets)]
            if by[d]:
                chosen.append(by[d].pop(0))
            i += 1
    return [{"dataset": q["dataset"], "query_id": str(q["query_id"])} for q in chosen]


def gate_test():
    """test-0.1 needs the offline check and the val smoke to have passed (status fields, plain JSON)."""
    for name in ("offline.json", "smoke.json"):
        path = f.HERE / "checks" / name
        value = f.read_optional(path)
        if not value or value.get("status") != "passed":
            raise SystemExit(f"test-0.1 refused: {path} is missing or not passed")


def persist_failure(job, error):
    """Queue boundary: an unexpected item exception becomes that item's terminal failure."""
    path = f.result_path(job)
    previous = f.read_optional(path) or {}
    f.save(path, {**previous, **job, "status": "failed", "terminal": True, "error_phase": "queue_boundary",
                  "error_type": type(error).__name__, "error": f.redact(repr(error)), "metrics": {},
                  "failure_kind": "queue_boundary_exception", "finished_at": f.now(),
                  "cost_usd": previous.get("cost_usd", 0.0), "total_input_tokens": previous.get("total_input_tokens", 0),
                  "total_output_tokens": previous.get("total_output_tokens", 0), "tool_calls": previous.get("tool_calls", 0)})


def shared_fault_suspect(result):
    if not result or result.get("status") == "scored" or result.get("provider_refusal"):
        return False
    if result.get("error_phase") in ("preflight", "provider_tools"):
        # a preflight timeout under server load is not an isolation fault
        return result.get("error_type") not in ("TimeoutExpired", "InterruptedError")
    text = (result.get("error") or "").lower()
    return result.get("error_phase") == "agent" and any(
        w in text for w in ("invalid_api_key", "unauthorized", "unauthorised", "401", "insufficient_quota"))


def smoke_report(jobs):
    runs, ok = [], True
    for job in jobs:
        r = f.read_optional(f.result_path(job)) or {}
        try:  # recomputed from the archived provider requests (no model calls)
            audit = f.h6_audit(r["attempt_path"])
        except Exception:
            audit = r.get("h6_audit") or {}
        entry = {"arm": job["arm"], "dataset": job["dataset"], "query_id": job["query_id"], "status": r.get("status"),
                 "hit_at_1": (r.get("metrics") or {}).get("hit_at_1"), "tool_calls": r.get("tool_calls"),
                 "attempts": r.get("attempt_count"), "cost_usd": r.get("cost_usd"), **{f"h6_{k}": v for k, v in audit.items()},
                 "provider_tools": (r.get("provider_tools") or {}).get("tools"), "error": r.get("error")}
        # H6 covers bash only; L3 markers on Pi's read tool are reported (l3_markers_other_tools), not gated.
        good = (r.get("status") == "scored" and bool(audit) and audit.get("l3_markers_bash") == 0
                and audit.get("bash_max_utf16_units", 0) <= f.L3_LIMIT_UNITS and audit.get("bash_over_limit") == 0
                and (r.get("tool_calls") or 0) > 0)
        try:
            entry["visible_ablation_path_occurrences"] = f.visible_path_leaks(r["attempt_path"])
        except Exception:
            entry["visible_ablation_path_occurrences"] = None
        good = good and entry["visible_ablation_path_occurrences"] == 0
        entry["passed"] = good
        ok &= good
        runs.append(entry)
    return {"status": "passed" if ok else "failed", "runs": runs, "count": len(runs), "finished_at": f.now()}


def main():
    args = parse()
    if args.check:
        import offline_check
        offline_check.run()
        print("offline checks passed:", offline_check.CHECKS / "offline.json", flush=True)
        return 0
    rows, labels, candidates = f.load_split(args.split)
    chosen = questions(args, rows)
    jobs = f.schedule(chosen, args.arms, args.split)
    pending = [j for j in jobs if not f.is_terminal(j)]
    print(f"{args.split}: {len(chosen)} questions x {len(args.arms)} arms = {len(jobs)} items; "
          f"{len(jobs) - len(pending)} already terminal; workers {args.workers}", flush=True)
    if args.dry_run:
        for job in pending[:30]:
            print(" ", f.jobkey(job))
        return 0
    if args.split == "test-0.1":
        gate_test()
    if not os.environ.get("MODEL_API_KEY"):
        raise SystemExit("MODEL_API_KEY is not set")
    lock = (f.HERE / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another ablation_final runner holds run.lock")
    protocol = f.c.val.protocol()
    state_path = f.HERE / f"state_{'smoke' if args.smoke else args.split}.json"
    state = {"status": "running", "split": args.split, "arms": args.arms, "planned": len(jobs), "workers": args.workers,
             "started_at": f.now(), "pid": os.getpid(), "active": [], "stop_reason": None}
    signals = Counter()

    def on_signal(signum, _frame):
        signals["n"] += 1
        f.STOP.set()
        state["stop_reason"] = f"signal {signum}"
        if signals["n"] >= 2:
            f.HARD_STOP.set()
            f.kill_active_agents()

    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, on_signal)

    def reduce_workers(job):
        old = state["workers"]
        state["workers"] = max(1, old // 2)
        print(f"RATE-LIMIT {f.jobkey(job)}: admission workers {old} -> {state['workers']}", flush=True)

    def snapshot():
        done = [f.read_optional(f.result_path(j)) for j in jobs]
        terminal = [r for r in done if r and r.get("terminal")]
        per_arm = {}
        for arm in args.arms:
            values = [r for r in terminal if r.get("arm") == arm]
            failed = sum(r.get("status") != "scored" for r in values)
            per_arm[arm] = {"terminal": len(values), "scored": len(values) - failed,
                            "failure_rate": failed / len(values) if values else None,
                            "alert_over_2pct": bool(values) and failed / len(values) > 0.02}
        state.update(terminal=len(terminal), per_arm=per_arm, heartbeat=f.now(),
                     cost_usd=sum(r.get("cost_usd") or 0 for r in terminal))
        f.save(state_path, state)

    streak = 0
    running = {}
    queue = list(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while queue or running:
            while queue and not f.STOP.is_set() and len(running) < state["workers"]:
                job = queue.pop(0)
                key = f.key(job)
                running[pool.submit(f.execute, job, rows[key], labels[key], candidates[job["dataset"]], protocol,
                                    reduce_workers)] = job
            if not running:
                break
            state["active"] = [f.jobkey(j) for j in running.values()]
            snapshot()
            done, _ = wait(running, timeout=15, return_when=FIRST_COMPLETED)
            for future in done:
                job = running.pop(future)
                try:
                    result = future.result()
                except BlockingIOError:
                    print(f"SKIP {f.jobkey(job)}: locked by another process", flush=True)
                    continue
                except Exception as error:
                    try:
                        persist_failure(job, error)
                    except Exception as persist_error:
                        state["stop_reason"] = "cannot persist results: " + f.redact(persist_error)
                        f.STOP.set()
                    result = f.read_optional(f.result_path(job))
                if result is None:
                    print(f"STOPPED {f.jobkey(job)} (non-terminal, resumes later)", flush=True)
                    continue
                m = result.get("metrics") or {}
                print(f"{f.jobkey(job)} {result.get('status')} H@1={m.get('hit_at_1')} attempts={result.get('attempt_count')} "
                      f"${result.get('cost_usd') or 0:.4f} {result.get('failure_kind') or ''}", flush=True)
                streak = streak + 1 if shared_fault_suspect(result) else 0
                if streak >= SHARED_FAULT_STREAK and not f.STOP.is_set():
                    state["stop_reason"] = (f"shared fault suspected: {streak} consecutive items failed at "
                                            f"{result.get('error_phase')}: {f.redact(result.get('error', ''))[:500]}")
                    state["shared_fault"] = True
                    f.STOP.set()
    state["active"] = []
    snapshot()
    remaining = [j for j in jobs if not f.is_terminal(j)]
    state.update(status="completed" if not remaining else ("fault_stop" if state.get("shared_fault") else "stopped"),
                 remaining=len(remaining), finished_at=f.now())
    f.save(state_path, state)
    smoke_jobs = f.schedule(chosen, f.ARMS, "val") if args.smoke else []
    if args.smoke and all(f.is_terminal(j) for j in smoke_jobs):  # the gate always covers all five arms
        report = smoke_report(smoke_jobs)
        f.save(f.HERE / "checks/smoke.json", report)
        print("smoke:", report["status"], f.HERE / "checks/smoke.json", flush=True)
    print(f"finished: {state['status']}; terminal {state['terminal']}/{len(jobs)}; ${state['cost_usd']:.4f}", flush=True)
    return 3 if state.get("shared_fault") else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as error:
        print(f.redact(repr(error)), file=sys.stderr)
        sys.exit(3)
