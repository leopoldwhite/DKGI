"""One program (X1): descriptive statistics, point estimates only.

    python3 analysis_final/x1_stats.py

X1 = Full DKGI's prompt plus one sentence, Shell only's allowlist plus python3, which can be started once per query
(test subset, 710 queries; run tree trials/O1/<ds>/test/<qid>). Reported here:
  * share of queries that used the one start (result.json python_gate.started), refused further starts (gate log
    summary python_gate.refused, and tool outputs with the fixed refusal from python_trajectory);
  * length of the single program (lines, characters) next to Full DKGI's first program and all programs of a query;
    a program is the inline Python source of one shell call (python3 -c or a heredoc), parsed exactly as in
    coevolution.py; X1's program is the shell call whose gate connection was granted the start;
  * tool calls, input/output tokens, wall time and cost per query (median and mean; tab:cost prints the mean cost and
    the other medians);
  * H@1, R@20 and MRR per dataset and pooled (final_numbers.py's rule: recorded metrics, failed runs score 0).

Sources. X1: its result.json files, the gate log of the scored attempt (<item>/<attempt>.python_gate.jsonl) and the
attempt's artifacts/events.jsonl. Full DKGI and Shell only: existing analysis outputs only (out/coevolution/cache/
luna_full.jsonl for Full DKGI's programs, out/coding_effect.json for their effort and cost) plus their result.json
files for the per-dataset metrics. Read-only, no model calls. Writes out/x1_one_program.md and .json.
"""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import coevolution as CO  # noqa: E402
import final_numbers as FN  # noqa: E402

OUT = C.OUT
NAMES = {"X1": "One program", "A4": "Full DKGI", "A3": "Shell only"}
METRICS = FN.METRICS  # (key, label): H@1, R@20, MRR


# --------------------------------------------------------------------------- X1: one query

def gate_grant(log_path):
    """command_chars of the shell call whose gate connection was granted the start (agent phase of the gate log)."""
    hello, granted = {}, None
    if not log_path or not Path(log_path).exists():
        return None
    for e in C.read_jsonl(log_path):
        if e.get("phase") != "agent":
            continue
        if e.get("op") == "hello":
            hello[e.get("connection")] = e.get("command_chars")
        elif e.get("op") == "take" and e.get("granted") and granted is None:
            granted = e.get("connection")
    return hello.get(granted) if granted is not None else None


def x1_query(args):
    result, src = args
    ds, qid = Path(src).parts[-4], Path(src).parts[-2]
    g = result.get("python_gate") or {}
    pt = result.get("python_trajectory") or {}
    exits = g.get("exit") or []
    rec = {"dataset": ds, "query_id": qid, "status": result.get("status"),
           "started": (g.get("started") or 0) >= 1, "starts": g.get("started") or 0,
           "refused_gate": g.get("refused") or 0, "launch_requests": g.get("launch_requests") or 0,
           "refusal_outputs": pt.get("outputs_with_refusal") or 0,
           "python_mentions": pt.get("commands_mentioning_python") or 0,
           "other_interpreter_commands": pt.get("commands_running_other_interpreters") or 0,
           "exit_nonzero": bool(exits) and any(e.get("exit_code") not in (0, None) or e.get("signal") for e in exits),
           "program_seconds": sum(e.get("seconds") or 0 for e in exits) if exits else None,
           "argv_mode": None, "match": None, "chars": None, "lines": None, "edges": None,
           "calls_total": None, "calls_before": None, "py_commands": None}
    a = g.get("first_grant_argv") or []
    if a:
        rec["argv_mode"] = "-c" if "-c" in a else ("stdin" if "-" in a[1:] or len(a) == 1 else "other")
    attempt = result.get("attempt_path")
    tr = CO.pi_trace(attempt, union=bool(result.get("compaction_starts"))) if attempt else None
    if tr is None:
        return rec
    calls = tr["calls"]
    rec["calls_total"] = len(calls)
    infos = [CO.parse_shell(c["cmd"] or "") if c["name"] == "bash" else None for c in calls]
    py_idx = [i for i, info in enumerate(infos) if info and info["py"]]
    rec["py_commands"] = len(py_idx)
    if not rec["started"]:
        return rec
    chars = gate_grant(g.get("log"))
    idx = None
    if chars is not None:
        same = [i for i in py_idx if len(calls[i]["cmd"] or "") == chars]
        if same:
            idx, rec["match"] = same[0], "gate command length"
    if idx is None and py_idx:
        idx, rec["match"] = py_idx[0], "first Python command"
    if idx is None:
        rec["match"] = "none"
        return rec
    info, cmd = infos[idx], calls[idx]["cmd"] or ""
    code = "\n".join(info["sources"]) if info["sources"] else cmd  # as coevolution.py's program text
    rec.update(chars=len(code), lines=code.count("\n") + 1, edges=bool(CO.EDGE_RE.search(cmd)), calls_before=idx)
    return rec


def seconds_per_request(result):
    """Wall-clock seconds per model request of a query (to compare model latency across runs)."""
    w = C._num(result.get("total_wall_seconds", result.get("wall_time_seconds")))
    m = C._num(result.get("model_requests", result.get("agent_loop_requests")))
    return w / m if w and m else None


def effort(result):
    """Effort fields with the precedence of common.features (tool_calls = the recorded count, as tab:cost)."""
    cost = result.get("cost_usd", result.get("attempt_cost_usd"))
    return {"tool_calls": C._num(result.get("tool_calls")),
            "input_tokens": C._num(result.get("total_input_tokens", result.get("input_tokens"))),
            "output_tokens": C._num(result.get("total_output_tokens", result.get("output_tokens"))),
            "wall_seconds": C._num(result.get("total_wall_seconds", result.get("total_agent_seconds",
                                                                                 result.get("wall_time_seconds")))),
            "cost_usd": C._num(cost) if cost is not None else C.price_equivalent(result)}


# --------------------------------------------------------------------------- Full DKGI programs (coevolution cache)

def full_dkgi_programs():
    """{(ds, qid): [programs that ran]} for Full DKGI on the test subset, from the coevolution cache (no re-extraction)."""
    cfg = next(c for c in CO.configurations() if c["name"] == "A4")
    path = CO.CACHE / f"{cfg['source']}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}: run coevolution.py first")
    cached = {r["src"]: r for r in C.read_jsonl(path)}
    out, stale = {}, []
    for result, src in CO.iter_results(cfg):
        r = cached.get(src)
        if r is None or r.get("version") != CO.VERSION or r.get("status") != result.get("status") \
                or r.get("finished_at") != result.get("finished_at"):
            stale.append(src)
            continue
        progs = [{**CO.PROG_DEFAULTS, **p} for p in r.get("progs") or []]
        out[(r["dataset"], str(r["query_id"]))] = [p for p in progs if not p["nr"]]
    if stale:
        raise SystemExit(f"{len(stale)} Full DKGI rows missing or stale in {path}: run coevolution.py first")
    return out


# --------------------------------------------------------------------------- helpers

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct(k, n):
    return 100 * k / n if n else None


def stat(xs):
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "median": C.median(xs), "mean": mean(xs)}


def num(x, d=0):
    if x is None:
        return "n/a"
    return f"{x:,.{d}f}"


# --------------------------------------------------------------------------- main

def main():
    cfg = C.config_by_name()["X1"]
    items = [(r, s) for r, s in C.iter_results(cfg) if r.get("terminal")]
    sel = C.load_selection(C.SEL_710)
    if len(items) != len(sel):
        print(f"warning: {len(items)} terminal X1 results for {len(sel)} selected queries", flush=True)
    from multiprocessing import Pool
    with Pool(min(32, os.cpu_count() or 4)) as pool:
        recs = pool.map(x1_query, items, chunksize=4)
    for rec, (result, _) in zip(recs, items):
        rec.update(effort(result))
    n = len(recs)
    by_key = {(r["dataset"], r["query_id"]): r for r in recs}

    # 1. Starts and refusals
    started = [r for r in recs if r["started"]]
    starts = {
        "n_queries": n, "used_start": len(started), "used_start_pct": pct(len(started), n),
        "used_start_by_dataset": {ds: {"n": sum(r["dataset"] == ds for r in recs),
                                       "used": sum(r["dataset"] == ds and r["started"] for r in recs),
                                       "pct": pct(sum(r["dataset"] == ds and r["started"] for r in recs),
                                                  sum(r["dataset"] == ds for r in recs))} for ds in C.DATASETS},
        "max_starts_per_query": max(r["starts"] for r in recs),
        "refused_starts_gate": sum(r["refused_gate"] for r in recs),
        "queries_with_refused_start_gate": sum(r["refused_gate"] > 0 for r in recs),
        "refusal_message_outputs": sum(r["refusal_outputs"] for r in recs),
        "queries_with_refusal_message": sum(r["refusal_outputs"] > 0 for r in recs),
        "queries_with_more_than_one_python_command": sum((r["py_commands"] or 0) > 1 for r in recs),
        "other_interpreter_commands": sum(r["other_interpreter_commands"] for r in recs),
        "queries_with_other_interpreter_command": sum(r["other_interpreter_commands"] > 0 for r in recs),
        "program_nonzero_exit": sum(r["exit_nonzero"] for r in started),
        "program_nonzero_exit_pct_of_started": pct(sum(r["exit_nonzero"] for r in started), len(started)),
        "program_via": {m: sum(r["argv_mode"] == m for r in started) for m in ("stdin", "-c", "other")},
        "program_matched_by": {m: sum(r["match"] == m for r in started)
                               for m in ("gate command length", "first Python command", "none")},
        "program_seconds": stat([r["program_seconds"] for r in started]),
        "tool_calls_before_start": stat([r["calls_before"] for r in started]),
        "share_of_calls_before_start_pct": stat([100 * r["calls_before"] / r["calls_total"] for r in started
                                                 if r["calls_before"] is not None and r["calls_total"]]),
    }

    # 2. Program length: X1's single program vs Full DKGI's first program and all programs of a query
    full = full_dkgi_programs()
    x1p = [r for r in started if r["chars"] is not None]
    fq = {k: v for k, v in full.items() if v}
    both = [k for k, r in by_key.items() if r["chars"] is not None and k in fq]
    length = {
        "X1 single program": {"n": len(x1p), "lines": stat([r["lines"] for r in x1p]),
                              "chars": stat([r["chars"] for r in x1p]),
                              "reads_edge_files_pct": pct(sum(r["edges"] for r in x1p), len(x1p))},
        "Full DKGI first program": {"n": len(fq), "lines": stat([v[0]["lines"] for v in fq.values()]),
                                    "chars": stat([v[0]["len"] for v in fq.values()]),
                                    "reads_edge_files_pct": pct(sum(v[0]["ed"] for v in fq.values()), len(fq))},
        "Full DKGI all programs of a query (sum)": {
            "n": len(fq), "lines": stat([sum(p["lines"] for p in v) for v in fq.values()]),
            "chars": stat([sum(p["len"] for p in v) for v in fq.values()]),
            "programs_per_query": stat([len(v) for v in fq.values()]),
            "any_program_reads_edge_files_pct": pct(sum(any(p["ed"] for p in v) for v in fq.values()), len(fq))},
        "Full DKGI any program (per program)": {
            "n": sum(len(v) for v in fq.values()), "lines": stat([p["lines"] for v in fq.values() for p in v]),
            "chars": stat([p["len"] for v in fq.values() for p in v])},
        "queries_full_dkgi_with_program": len(fq),
        "same_queries": {
            "n": len(both),
            "X1 single program": {"lines": C.median([by_key[k]["lines"] for k in both]),
                                  "chars": C.median([by_key[k]["chars"] for k in both])},
            "Full DKGI first program": {"lines": C.median([fq[k][0]["lines"] for k in both]),
                                        "chars": C.median([fq[k][0]["len"] for k in both])},
            "Full DKGI all programs (sum)": {"lines": C.median([sum(p["lines"] for p in fq[k]) for k in both]),
                                             "chars": C.median([sum(p["len"] for p in fq[k]) for k in both])}},
    }

    # 3. Effort and cost (tab:cost: mean $ per query; medians of the rest). Full DKGI and Shell only: coding_effect.json.
    ce = C.read_json(OUT / "coding_effect.json")["cost"]["A4-A3"]["all"]
    keys = ("tool_calls", "input_tokens", "output_tokens", "wall_seconds", "cost_usd")
    eff = {"X1": {k: stat([r[k] for r in recs]) for k in keys},
           "A4": {k: {"n": ce[k]["n"], "median": ce[k]["median_hi"], "mean": ce[k]["mean_hi"]} for k in keys},
           "A3": {k: {"n": ce[k]["n"], "median": ce[k]["median_lo"], "mean": ce[k]["mean_lo"]} for k in keys}}
    eff["X1_total_cost_usd"] = sum(r["cost_usd"] or 0 for r in recs)
    # Wall-clock time also depends on model latency at run time: median seconds per model request per variant.
    eff["seconds_per_model_request_median"] = {a: C.median([seconds_per_request(r) for r in FN.per_query(a).values()])
                                               for a in ("X1", "A4", "A3")}

    # 4. Metrics per dataset (final_numbers.py rule)
    metrics = {}
    for arm in ("X1", "A3", "A4"):
        res = FN.per_query(arm)
        metrics[arm] = {}
        for key, label in METRICS:
            vals = FN.values(res, key)
            row = {ds: FN.level(vals, ds)[0] for ds in C.DATASETS}
            row["pooled"] = 100 * sum(vals.values()) / len(vals)
            row["n"] = len(vals)
            metrics[arm][label] = row
    h = {a: metrics[a]["H@1"]["pooled"] for a in ("X1", "A3", "A4")}
    reading = {"share_of_code_gain": (h["X1"] - h["A3"]) / (h["A4"] - h["A3"]),
               "X1_minus_shell": h["X1"] - h["A3"], "X1_minus_full": h["X1"] - h["A4"],
               "R@20_X1_minus_full": metrics["X1"]["R@20"]["pooled"] - metrics["A4"]["R@20"]["pooled"],
               "MRR_X1_minus_full": metrics["X1"]["MRR"]["pooled"] - metrics["A4"]["MRR"]["pooled"],
               "R@20_X1_minus_shell": metrics["X1"]["R@20"]["pooled"] - metrics["A3"]["R@20"]["pooled"],
               "MRR_X1_minus_shell": metrics["X1"]["MRR"]["pooled"] - metrics["A3"]["MRR"]["pooled"]}

    # H@1 by whether X1 used its start (descriptive; the groups differ in difficulty)
    h1 = {a: FN.values(FN.per_query(a), "hit_at_1") for a in ("X1", "A3", "A4")}
    groups = {}
    for name, cond in (("used the start", True), ("did not use it", False)):
        ks = [k for k, r in by_key.items() if r["started"] is cond]
        groups[name] = {"n": len(ks), **{a: 100 * sum(h1[a].get(k, 0.0) for k in ks) / len(ks) if ks else None
                                         for a in ("X1", "A3", "A4")}}

    res = {"starts": starts, "length": length, "effort": eff, "metrics": metrics, "reading": reading,
           "h1_by_start_use": groups,
           "sources": {"X1": str(Path(cfg["source"][1])), "full_dkgi_programs": str(CO.CACHE / "luna_full.jsonl"),
                       "effort_full_shell": str(OUT / "coding_effect.json")}}
    C.write_text(OUT / "x1_one_program.json", json.dumps(res, indent=1))
    C.write_text(OUT / "x1_one_program.md", report(res))
    print(f"wrote x1_one_program.md/.json: used start {starts['used_start']}/{n}, "
          f"H@1 {h['X1']:.2f}, share of code gain {reading['share_of_code_gain']:.3f}")


def report(res):
    s, L, E, M, R, G = (res[k] for k in ("starts", "length", "effort", "metrics", "reading", "h1_by_start_use"))
    n = s["n_queries"]
    lines = ["# X1 \"One program\": descriptive statistics (test subset)", "",
             "Generated by `analysis_final/x1_stats.py`. Point estimates only. Paper name: One program (`\\vone`).", "",
             "## Metrics", "",
             "Recorded metrics of the scored run, failed runs score 0 (as `final_numbers.py`).", ""]
    rows = []
    for arm in ("A4", "X1", "A3"):
        for _, m in METRICS:
            r = M[arm][m]
            rows.append([NAMES[arm], m] + [f"{r[ds]:.1f}" for ds in C.DATASETS] + [f"{r['pooled']:.2f}", r["n"]])
    lines += [C.md_table(["Variant", "Metric", "Amazon", "MAG", "Prime", "Pooled", "n"], rows), "",
              f"Pooled H@1: One program − Shell only = {R['X1_minus_shell']:+.2f}; One program − Full DKGI = "
              f"{R['X1_minus_full']:+.2f}; R@20 {R['R@20_X1_minus_full']:+.2f} and MRR {R['MRR_X1_minus_full']:+.2f} "
              f"vs Full DKGI ({R['R@20_X1_minus_shell']:+.2f} / {R['MRR_X1_minus_shell']:+.2f} vs Shell only).", "",
              f"One program recovers "
              f"(One program − Shell only) / (Full DKGI − Shell only) = {100 * R['share_of_code_gain']:.1f}% of the "
              "code gain.", "",
              "## The one start", "",
              f"- Queries that used the start (python_gate.started): {s['used_start']} / {n} "
              f"({s['used_start_pct']:.1f}%); " + ", ".join(
                  f"{C.DS_LABEL[ds]} {v['used']}/{v['n']} ({v['pct']:.1f}%)" for ds, v in s["used_start_by_dataset"].items())
              + f". At most {s['max_starts_per_query']} start per query.",
              f"- Refused further starts logged by the gate: {s['refused_starts_gate']} in "
              f"{s['queries_with_refused_start_gate']} queries; tool outputs with the fixed refusal message: "
              f"{s['refusal_message_outputs']} in {s['queries_with_refusal_message']} queries.",
              f"- Queries with more than one shell call that contains a Python program (granted, refused, or not "
              f"started at all): {s['queries_with_more_than_one_python_command']}.",
              f"- Shell calls that tried another interpreter (awk, perl, node, ...; all blocked), from "
              f"python_trajectory: {s['other_interpreter_commands']} in {s['queries_with_other_interpreter_command']} "
              "queries.",
              f"- The one program exited non-zero in {s['program_nonzero_exit']} of {s['used_start']} queries "
              f"({s['program_nonzero_exit_pct_of_started']:.1f}%); it ran for a median of "
              f"{num(s['program_seconds']['median'], 1)} s (mean {num(s['program_seconds']['mean'], 1)} s). "
              "Program passed on stdin (heredoc): "
              f"{s['program_via']['stdin']}; with -c: {s['program_via']['-c']}; other: {s['program_via']['other']}.",
              f"- Tool calls before the start: median {num(s['tool_calls_before_start']['median'], 1)}, mean "
              f"{num(s['tool_calls_before_start']['mean'], 1)}; share of the query's tool calls before it: median "
              f"{num(s['share_of_calls_before_start_pct']['median'], 1)}%.",
              f"- Program located by the gate's command length: {s['program_matched_by']['gate command length']}; "
              f"by first Python command: {s['program_matched_by']['first Python command']}; not found: "
              f"{s['program_matched_by']['none']}.", "",
              "## Program length", "",
              "A program is the inline Python source of one shell call (`python3 -c` or a heredoc), parsed as in "
              "`coevolution.py`; Full DKGI counts programs that ran (not `python: command not found`). "
              "Medians (means in brackets).", ""]
    rows = []
    for name in ("X1 single program", "Full DKGI first program", "Full DKGI all programs of a query (sum)",
                 "Full DKGI any program (per program)"):
        x = L[name]
        rows.append([name.replace("X1 single program", "One program: its single program"), x["n"], f"{num(x['lines']['median'], 1)} ({num(x['lines']['mean'], 1)})",
                     f"{num(x['chars']['median'], 0)} ({num(x['chars']['mean'], 0)})",
                     num(x.get("reads_edge_files_pct", x.get("any_program_reads_edge_files_pct")), 1)
                     if name != "Full DKGI any program (per program)" else ""])
    lines += [C.md_table(["Program", "n", "Lines", "Characters", "Touches edge files %"], rows), "",
              f"Full DKGI: {num(L['Full DKGI all programs of a query (sum)']['programs_per_query']['mean'], 2)} programs "
              f"per query with a program (median "
              f"{num(L['Full DKGI all programs of a query (sum)']['programs_per_query']['median'], 1)}); "
              f"{L['queries_full_dkgi_with_program']} of 710 queries have one.", ""]
    sq = L["same_queries"]
    lines += [f"Same queries (One program used its start and Full DKGI wrote a program, n = {sq['n']}), medians: "
              f"One program {num(sq['X1 single program']['lines'], 1)} lines / {num(sq['X1 single program']['chars'])} "
              f"characters; Full DKGI first program {num(sq['Full DKGI first program']['lines'], 1)} / "
              f"{num(sq['Full DKGI first program']['chars'])}; all Full DKGI programs of the query "
              f"{num(sq['Full DKGI all programs (sum)']['lines'], 1)} / {num(sq['Full DKGI all programs (sum)']['chars'])}.", "",
              "## Effort and cost per query", "",
              "tab:cost convention: $ per query is the mean; the other columns are medians. Means in brackets. "
              "Full DKGI and Shell only from `coding_effect.json` (same numbers as tab:cost).", ""]
    rows = []
    for arm in ("A4", "X1", "A3"):
        e = E[arm]
        rows.append([NAMES[arm], f"{e['cost_usd']['mean']:.4f} (median {e['cost_usd']['median']:.4f})",
                     f"{num(e['tool_calls']['median'], 1)} ({num(e['tool_calls']['mean'], 1)})",
                     f"{num(e['input_tokens']['median'])} ({num(e['input_tokens']['mean'])})",
                     f"{num(e['output_tokens']['median'])} ({num(e['output_tokens']['mean'])})",
                     f"{num(e['wall_seconds']['median'])} ({num(e['wall_seconds']['mean'])})"])
    lines += [C.md_table(["Variant", "$ per query (mean)", "Tool calls", "Input tokens", "Output tokens", "Time (s)"], rows),
              "", f"One program total cost: ${E['X1_total_cost_usd']:.2f} for {n} queries.", "",
              "Wall-clock time also reflects model latency during each run: median seconds per model request "
              + ", ".join(f"{NAMES[a]} {E['seconds_per_model_request_median'][a]:.1f}" for a in ("A4", "X1", "A3"))
              + ". One program's longer time comes from slower model responses during its run, not from the "
              "program (median run time above).", "",
              "## H@1 by whether One program used its start (descriptive)", "",
              "The two groups differ in difficulty; the columns compare variants on the same queries.", ""]
    rows = [[name, g["n"], f"{g['A4']:.1f}", f"{g['X1']:.1f}", f"{g['A3']:.1f}"] for name, g in G.items()]
    lines += [C.md_table(["Queries where One program", "n", "Full DKGI", "One program", "Shell only"], rows), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
