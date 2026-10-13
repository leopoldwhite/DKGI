"""Reclassify published DCI logs and STaRK with one shared nine-category rule.

The original Figure 6 classifier and 100-case selection of Li et al. (2026) are unavailable.
This is a new common-protocol comparison, not a reproduction of that figure.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shlex

import analyze_behavior as base
import plot_intent_distribution as previous


ROOT = Path(__file__).resolve().parent
ORDER = ("search_limit", "chain", "count", "search", "list", "python", "locate", "other", "read")
TITLES = {"search_limit": "Search + Limit", "chain": "Chain Search", "count": "Aggregation / Counting",
          "search": "Direct Keyword Search", "list": "List Directory", "python": "Python Scripting",
          "locate": "Locate File", "other": "Other Commands", "read": "File Read (cat)"}
PAPER_PERCENT = dict(zip(ORDER, (56.2, 20.6, 7.8, 6.3, 4.3, 2.4, 2.1, .2, .1)))
COHORTS = ("original_dci", "stark")


def classify(command):
    try:
        lexer = shlex.shlex(command, posix=True)
        lexer.whitespace_split = True
        first = next(lexer, "")
        normalized = base.shell_body(command) if Path(first).name in {"bash", "sh", "dash", "zsh"} else command
    except ValueError:
        return {"category": "other", "normalized_command": command,
                "shell_parse_errors": 1, "python_parse_errors": 0}
    # Normalize both harness formats before parsing, including quotes in heredocs.
    wrapped = "/usr/bin/bash -lc " + shlex.quote(normalized)
    parsed = base.classify(wrapped)
    try:
        category = previous.intent({"command": wrapped, **parsed})
    except ValueError:
        category = "other"
        parsed["shell_parse_errors"] = 1
    return {"category": "other" if category == "partial" else category,
            "normalized_command": normalized, "shell_parse_errors": parsed["shell_parse_errors"],
            "python_parse_errors": parsed["python_parse_errors"]}


def codex_calls(events):
    calls, ends = {}, {}
    for line, event in enumerate(events, 1):
        item = event.get("item", {})
        if item.get("type") != "command_execution":
            continue
        key = item["id"]
        if event["type"] == "item.started":
            if key in calls:
                assert calls[key]["command"] == item["command"], "Conflicting repeated start"
            else:
                calls[key] = {"command": item["command"], "tool_call_id": key, "trace_line": line}
        elif event["type"] == "item.completed":
            if key in ends:
                assert ends[key] == item, "Conflicting repeated completion"
            ends[key] = item
    assert ends.keys() <= calls.keys(), "Completion without recorded start"
    for step, (key, call) in enumerate(calls.items(), 1):
        end = ends.get(key)
        if end:
            assert call["command"] == end["command"], "Command changed between start and completion"
        yield {**call, "step": step, "completed": end is not None,
               "is_error": end.get("exit_code") not in (None, 0) if end else None}


def load_stark():
    path = ROOT / "behavior_710/per_query.csv"
    with path.open() as source:
        queries = list(csv.DictReader(source))
    commands, provenance = [], []
    for query in queries:
        trace = Path(query["attempt_path"]) / "stdout.jsonl"
        with trace.open() as source:
            calls = list(codex_calls(map(json.loads, source)))
        assert sum(c["completed"] for c in calls) == int(query["calls"])
        for call in calls:
            commands.append({**call, "cohort": "stark", "dataset": query["dataset"],
                             "query_id": query["query_id"], "hit_at_1": int(query["hit_at_1"]),
                             "trace_path": str(trace), "model": "gpt-6-luna"})
        provenance.append({"dataset": query["dataset"], "query_id": query["query_id"],
                           "trace_path": str(trace), "bytes": trace.stat().st_size})
    return commands, queries, provenance


def self_check():
    examples = [("rg x nodes | head", "search_limit"), ("rg -m20 x nodes", "search_limit"),
                ("rg x nodes | rg y | head", "chain"), ("rg x nodes && rg y nodes", "search"),
                ("rg --files | head", "locate"), ("ls | rg txt", "locate"),
                ("rg x | wc -l", "count"), ("rg -c x nodes", "count"),
                ("python3 -c 'print(1)'", "python"),
                ("python3 <<'PY'\nprint('rg x | head')\nPY", "python"),
                ("python3 <<'PY'\n# we'll search next\nprint(1)\nPY", "python"),
                ("rg x nodes; python3 -c 'print(1)'", "python"),
                ("cat schema.json", "read"), ("cat x | head", "other"),
                ("sed -n '1,20p' x", "other"), ("ls", "list"),
                ("rg 'python3 -c' nodes", "search")]
    for command, expected in examples:
        for raw in (command, "/usr/bin/bash -lc " + shlex.quote(command)):
            assert classify(raw)["category"] == expected, (raw, expected)
    def event(kind, key, command):
        return {"type": kind, "item": {"type": "command_execution", "id": key, "command": command, "exit_code": 0}}
    events = [event("item.started", "a", "ls"), event("item.started", "a", "ls"),
              event("item.completed", "a", "ls"), event("item.started", "b", "pwd")]
    calls = list(codex_calls(events))
    assert len(calls) == 2 and sum(c["completed"] for c in calls) == 1
    assert {classify(c)["category"] for c, _ in examples} <= set(ORDER)
    print(f"Passed {len(examples) * 2} classifier checks and issued/completed deduplication check")


def summarize(commands, queries):
    rows = []
    for cohort in COHORTS:
        nq = queries[cohort]
        for scope in ("issued", "completed"):
            selected = [r for r in commands if r["cohort"] == cohort and (scope == "issued" or r["completed"])]
            counts = Counter(r["category"] for r in selected)
            for category in ORDER:
                rows.append({"cohort": cohort, "scope": scope, "category": category, "label": TITLES[category],
                             "queries": nq, "total_calls": len(selected), "calls": counts[category],
                             "percent": 100 * counts[category] / len(selected),
                             "calls_per_query": counts[category] / nq,
                             "queries_with_behavior": len({(r["dataset"], r["query_id"]) for r in selected if r["category"] == category})})
            assert sum(counts.values()) == len(selected)
    return rows


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import PercentFormatter

    lookup = {(r["cohort"], r["category"]): r for r in rows if r["scope"] == "issued"}
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42, "font.family": "DejaVu Sans"}):
        fig, (ax, legend) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.05, 1.65]})
        fig.subplots_adjust(left=.07, right=.98, bottom=.23, top=.82, wspace=.13)
        fig.suptitle("Bash Intent Distribution: Shared Classification", fontsize=21, weight="bold", y=.96)
        fig.text(.5, .902, "Published DCI trajectories and DKGI trajectories on STaRK, reclassified by the same code", ha="center", fontsize=12, color="#555555")
        for x, cohort in enumerate(COHORTS):
            bottom = 0
            for category in ORDER:
                p = lookup[cohort, category]["percent"]
                color = previous.CATEGORIES[category][2]
                ax.bar(x, p, bottom=bottom, width=.58, color=color, edgecolor="white", linewidth=1)
                if p >= 4:
                    ax.text(x, bottom + p / 2, f"{p:.1f}%", ha="center", va="center", fontsize=12, weight="bold", color="#202020")
                bottom += p
        labels = ("Original DCI / BC+\nGPT-5.4-nano", "DKGI / STaRK\ngpt-6-luna")
        ax.set_xticks(range(2), [f"{label}\n{lookup[cohort, 'python']['queries']:,} queries; {lookup[cohort, 'python']['total_calls']:,} calls" for cohort, label in zip(COHORTS, labels)], fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_xlim(-.6, 1.6)
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_ylabel("Percentage of issued shell calls")
        ax.grid(axis="y", color="#E4E7EB")
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        legend.set_axis_off()
        legend.text(.80, 1.025, "Original DCI", ha="right", fontsize=10, color="#555555")
        legend.text(.99, 1.025, "STaRK", ha="right", fontsize=10, color="#555555")
        for i, category in enumerate(ORDER):
            y = .95 - i * .112
            legend.add_patch(Rectangle((.01, y - .014), .042, .048, transform=legend.transAxes, color=previous.CATEGORIES[category][2]))
            legend.text(.075, y + .01, TITLES[category], va="center", fontsize=12, weight="bold")
            for x, cohort in ((.80, "original_dci"), (.99, "stark")):
                legend.text(x, y + .01, f"{lookup[cohort, category]['percent']:.2f}%", ha="right", va="center", fontsize=11)
        fig.text(.07, .045, "One issued bash/shell tool call = one observation. Same nine categories and priority; read tools excluded.\nNew analysis of public logs, not a reproduction of the 100-case Figure 6 of Li et al. (2026). Tasks and models still differ.", color="#555555", fontsize=10)
        previous.save(fig, output, "aligned_distribution")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "behavior_710/aligned")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        self_check()
        return
    manifest_path = args.output / "original_manifest.json"
    manifest = base.read_json(manifest_path)
    original_path = args.output / "original_commands.jsonl"
    with original_path.open() as source:
        original = [dict(json.loads(line), cohort="original_dci") for line in source]
    assert {q["model"] for q in manifest["queries"]} == {"gpt-5.4-nano"}
    for query in manifest["queries"]:
        # Stale-input guard by recorded size; the issued/completed totals below check the extraction.
        assert Path(query["trace_path"]).stat().st_size == query["bytes"], f"Modified trace: {query['trace_path']}"
    stark, stark_queries, stark_provenance = load_stark()
    commands = original + stark
    seen = set()
    for command in commands:
        identity = tuple(command[k] for k in ("cohort", "dataset", "query_id", "tool_call_id"))
        assert identity not in seen, identity
        seen.add(identity)
        command.update(classify(command["command"]))
    nq = {"original_dci": len(manifest["queries"]), "stark": len(stark_queries)}
    assert sum(q["bash_issued"] for q in manifest["queries"]) == len(original)
    assert sum(q["bash_completed"] for q in manifest["queries"]) == sum(c["completed"] for c in original)
    rows = summarize(commands, nq)
    previous.write_csv(args.output / "comparison.csv", rows)
    with (args.output / "command_labels.jsonl").open("w") as destination:
        for command in commands:
            destination.write(json.dumps(command, ensure_ascii=False) + "\n")
    checks = {cohort: {"queries": nq[cohort], "issued": sum(c["cohort"] == cohort for c in commands),
                      "completed": sum(c["cohort"] == cohort and c["completed"] for c in commands),
                      "shell_parse_errors": sum(c["shell_parse_errors"] for c in commands if c["cohort"] == cohort)} for cohort in COHORTS}
    paths = (Path(__file__), Path(base.__file__), Path(previous.__file__), original_path, manifest_path, ROOT / "behavior_710/per_query.csv")
    provenance = {"sources": [str(p.resolve()) for p in paths], "stark_traces": stark_provenance,
                  "checks": checks, "unit": "unique issued bash/shell tool call", "independent_read_tools": "excluded",
                  "classifier": "Previous shared shell parser; partial/text-processing mapped to other; nine categories",
                  "matching_priority": list(previous.CATEGORIES), "category_mapping": {"partial": "other"},
                  "paper_reference": {"url": "https://arxiv.org/html/2605.05242v1#A1.F6", "queries": 100,
                                      "calls": 3168, "percent": PAPER_PERCENT, "recomputed": False,
                                      "limitation": "Original classifier and Figure 6 sample IDs were not found"}}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    lines = ["# Shared-classifier comparison", "", "Paper percentages are published references, not recomputed. Both local columns use the same classifier.", ""]
    for scope in ("issued", "completed"):
        lookup = {(r["cohort"], r["category"]): r for r in rows if r["scope"] == scope}
        lines += [f"## {scope.title()} Calls", "", "| Category | Li et al. (2026) Fig. 6 reference | Original public DCI | STaRK | Original calls/query | STaRK calls/query |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for category in ORDER:
            a, b = [lookup[c, category] for c in COHORTS]
            lines.append(f"| {TITLES[category]} | {PAPER_PERCENT[category]:.1f}% | {a['calls']} ({a['percent']:.2f}%) | {b['calls']} ({b['percent']:.2f}%) | {a['calls_per_query']:.3f} | {b['calls_per_query']:.3f} |")
        lines.append("")
    (args.output / "tables.md").write_text("\n".join(lines) + "\n")
    plot(rows, args.output)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
