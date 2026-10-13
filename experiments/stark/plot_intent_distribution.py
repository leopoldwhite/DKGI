"""Plot command-level intent distributions from the 710-query analysis.

No recorded command is executed. Shell categories are mutually exclusive;
Python graph-operation labels overlap and therefore use separate horizontal bars.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import shlex

import analyze_behavior as base


# Priority is explicit: the first matching category owns the entire shell call.
CATEGORIES = {
    "python": ("Python Scripting", "python -c / heredoc", "#7BC8A4"),
    "count": ("Aggregation / Counting", "wc / rg --count", "#F3CA7A"),
    "chain": ("Chained Content Search", "rg ... | rg ...", "#F0AE80"),
    "search_limit": ("Search + Limit", "rg ... | head / rg -m", "#E99B91"),
    "search": ("Direct Content Search", "rg pattern; no detected cap", "#D0CA7D"),
    "locate": ("Locate File", "rg --files / find / ls | rg", "#7DC1B4"),
    "read": ("Whole-file Read", "cat; includes schema and format", "#9AA6D1"),
    "partial": ("Partial Read / Text Processing", "head / sed / awk / cut", "#AFCB8E"),
    "list": ("List Directory", "ls / pwd", "#88B5CA"),
    "other": ("Other / Unparsed", "unmatched or invalid shell", "#BDBDBD"),
}
OUTCOMES = (("all", None), ("correct", 1), ("wrong", 0))
# Keep quoted operators inside arguments; shlex handles each resulting command.
CHUNKS = re.compile(r"'[^']*'|\"(?:\\.|[^\"\\])*\"|\\.|[;&|()\n]+|[^'\"\\;&|()\n]+")


def pipelines(command):
    body, _ = base.separate_heredocs(base.shell_body(command))
    groups, group, segment = [], [], []
    # Lexical shell patterns only; a full Bash parser would be needed for dynamic expansion.
    for chunk in [m.group() for m in CHUNKS.finditer(body)] + [";"]:
        if chunk and all(c in ";&|()\n" for c in chunk):
            if segment:
                program, args = base.program_args(shlex.split("".join(segment)))
                if program:
                    group.append((program, args))
            segment = []
            if chunk not in {"|", "|&"}:
                if group:
                    groups.append(group)
                group = []
        else:
            segment.append(chunk)
    return groups


def option(args, long_names, short_letters):
    for arg in args:
        if arg == "--":
            break
        if arg.split("=", 1)[0] in long_names:
            return True
        if arg.startswith("-") and not arg.startswith("--"):
            if any(letter in arg[1:] for letter in short_letters):
                return True
    return False


def intent(row):
    if "python" in row["tools"]:
        return "python"
    if row["shell_parse_errors"]:
        return "other"
    groups = pipelines(row["command"])
    content, chain, limited, locate = [], False, False, False
    programs = [p for group in groups for p, _ in group]
    for group in groups:
        file_stream, searches = False, []
        for i, (program, args) in enumerate(group):
            if program in {"ls", "find"} or (program in base.SEARCH and "--files" in args):
                file_stream = True
                locate |= program == "find" or (program in base.SEARCH and "--files" in args)
            is_search = program in base.SEARCH and "--files" not in args
            if program == "xargs" and any(a in base.SEARCH for a in args):
                is_search = True
                # xargs passes matched filenames as search inputs, not as stdin text.
                file_stream = False
            if is_search:
                if file_stream:
                    locate = True
                else:
                    searches.append(args)
                    limited |= option(args, {"--max-count"}, "m") or any(p in {"head", "tail"} for p, _ in group[i + 1:])
        content.extend(searches)
        chain |= len(searches) >= 2
    if "wc" in programs or any(option(a, {"--count", "--count-matches"}, "c") for a in content):
        return "count"
    if chain:
        return "chain"
    if content:
        return "search_limit" if limited else "search"
    if locate:
        return "locate"
    if any(len(g) == 1 and g[0][0] == "cat" for g in groups):
        return "read"
    if set(programs) & {"head", "tail", "sed", "awk", "cut", "sort", "uniq", "jq", "cat"}:
        return "partial"
    if set(programs) & {"ls", "pwd"}:
        return "list"
    return "other"


def self_check():
    examples = [
        ("python3 -c 'print(1)' | head", "python"),
        ("python3 <<'PY'\nprint('rg | head')\nPY", "python"),
        ("rg term nodes | head -5", "search_limit"),
        ("rg -nim20 term nodes", "search_limit"),
        ("rg --max-count=20 term nodes", "search_limit"),
        ("rg x nodes | rg y | head", "chain"),
        ("rg -l x nodes | xargs -r rg y", "chain"),
        ("rg x nodes && rg y nodes", "search"),
        ("rg '|' nodes", "search"),
        ("rg 'python -c / head' nodes", "search"),
        ("rg x nodes && head schema.json", "search"),
        ("rg x nodes | wc -l", "count"),
        ("rg -ic x nodes", "count"),
        ("pwd && rg --files | head -30", "locate"),
        ("ls nodes | rg jsonl | head", "locate"),
        ("find nodes -type f", "locate"),
        ("cat schema.json && cat DATA_FORMAT.txt", "read"),
        ("cat candidate_ids.txt | head", "partial"),
        ("sed -n '1,10p' schema.json", "partial"),
        ("pwd && ls", "list"),
        ("echo 'rg | head'", "other"),
        ("rg 'unclosed", "other"),
    ]
    for command, expected in examples:
        actual = intent({"command": command, **base.classify(command)})
        assert actual == expected, (command, actual, expected)
    print(f"Passed {len(examples)} classification checks")


def table_rows(commands, queries):
    rows = []
    for dataset in ["ALL", *sorted({q["dataset"] for q in queries})]:
        for outcome, hit in OUTCOMES:
            selected = [r for r in commands if (dataset == "ALL" or r["dataset"] == dataset) and (hit is None or r["hit_at_1"] == hit)]
            nq = sum((dataset == "ALL" or q["dataset"] == dataset) and (hit is None or int(q["hit_at_1"]) == hit) for q in queries)
            counts = Counter(r["intent"] for r in selected)
            for key, (title, _, _) in CATEGORIES.items():
                rows.append(dict(dataset=dataset, outcome=outcome, category=key, label=title,
                                 queries=nq, total_calls=len(selected), calls=counts[key],
                                 percent=100 * counts[key] / len(selected) if selected else None))
            assert sum(counts.values()) == len(selected)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=240, facecolor="white")


def plot_shell(rows, output, outcomes, name):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import PercentFormatter

    selected = {(r["outcome"], r["category"]): r for r in rows if r["dataset"] == "ALL"}
    order = sorted(CATEGORIES, key=lambda k: -selected["all", k]["calls"])
    fig, (ax, legend) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.15, 1.8]})
    fig.subplots_adjust(left=.08, right=.98, top=.84, bottom=.20, wspace=.12)
    fig.suptitle("Bash Intent Distribution", fontsize=22, weight="bold", y=.96)
    overall = selected["all", order[0]]
    fig.text(.5, .902, f"STaRK test-0.1 | gpt-6-luna | {overall['queries']:,} queries | {overall['total_calls']:,} shell calls", ha="center", fontsize=12, color="#555555")
    for x, outcome in enumerate(outcomes):
        bottom = 0
        for rank, key in enumerate(order, 1):
            percent = selected[outcome, key]["percent"]
            ax.bar(x, percent, bottom=bottom, width=.58, color=CATEGORIES[key][2], edgecolor="white", linewidth=1)
            if percent >= 3.5:
                ax.text(x, bottom + percent / 2, f"{rank}. {percent:.1f}%", ha="center", va="center", fontsize=11, weight="bold", color="#202020")
            bottom += percent
    ax.set_ylim(0, 100)
    ax.set_xlim(-.65, len(outcomes) - .35)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylabel("Percentage of shell calls", fontsize=12)
    ax.set_xticks(range(len(outcomes)), [f"{o.title()}\n{selected[o, order[0]]['queries']:,} queries\n{selected[o, order[0]]['total_calls']:,} calls" for o in outcomes])
    ax.grid(axis="y", color="#E4E7EB")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    legend.set_axis_off()
    legend.text(.99, 1.02, "Overall share", ha="right", fontsize=11, color="#555555")
    for rank, key in enumerate(order, 1):
        title, example, color = CATEGORIES[key]
        y = .94 - (rank - 1) * .097
        legend.add_patch(Rectangle((.01, y - .016), .047, .044, color=color, transform=legend.transAxes))
        legend.text(.08, y + .014, f"{rank}. {title}", va="center", fontsize=12, weight="bold")
        legend.text(.08, y - .027, example, va="center", fontsize=10, color="#626262")
        legend.text(.99, y + .014, f"{selected['all', key]['percent']:.2f}%", ha="right", va="center", fontsize=12, weight="bold")
    fig.text(.08, .045, "One category per completed shell call; fixed rule priority; final scored attempts only.\nEach bar totals 100%. Correct / wrong means query Hit@1. Nonzero command exits are included.", fontsize=10, color="#555555")
    save(fig, output, name)
    plt.close(fig)


def plot_graph(stats, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    names = {"both_directions": "Check both endpoints", "neighbor_collect": "Collect graph results",
             "set_seed": "Filter by ID collection", "node_enrichment": "Look up result nodes",
             "scalar_seed": "Filter by single ID", "relation_sweep": "Loop over relation files",
             "count_neighbors": "Count graph results", "sort_results": "Sort graph results",
             "node_type_filter": "Filter node types", "adjacency_index": "Build adjacency buckets",
             "attribute_filter": "Filter node attributes", "multiple_relations": "Read multiple named relations",
             "chained_expansion": "Chain edge lookups", "union": "Union result sets",
             "intersection": "Intersect result sets", "candidate_filter": "Filter answer eligibility",
             "difference": "Subtract result sets", "subprocess_search": "Search via subprocess",
             "seed_loop": "Repeat lookup over seed IDs"}
    group = stats["groups"]["ALL"]["all"]["labels"]
    total = group["edge_read"]["calls"]
    keys = sorted(names, key=lambda k: -group[k]["calls"])
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.subplots_adjust(left=.29, right=.97, top=.84, bottom=.12)
    fig.suptitle("Python Graph Operations", fontsize=22, weight="bold", y=.965)
    fig.text(.5, .909, f"{total:,} graph-processing shell calls | {group['edge_read']['queries']:,} queries | overlapping labels", ha="center", fontsize=12, color="#555555")
    for y, key in enumerate(keys):
        n = group[key]["calls"]
        percent = 100 * n / total
        ax.barh(y, percent, height=.65, color="#7DC1B4", edgecolor="#386F65", linewidth=.5)
        ax.text(percent + 1, y, f"{percent:.1f}% ({n:,})", va="center", fontsize=10)
    ax.set_yticks(range(len(keys)), [names[k] for k in keys], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.set_xlabel("Percentage of Python graph-processing shell calls", fontsize=12)
    ax.grid(axis="x", color="#E4E7EB")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.text(.04, .035, "Static code patterns, not measured execution counts. One call may have several labels; bars do not sum to 100%.\nReading and iterating edge data are omitted here as basic operations; all 21 labels are included in the CSV.", fontsize=10, color="#555555")
    save(fig, output, "python_graph_operations")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=base.ROOT / "behavior_710")
    parser.add_argument("--output", type=Path, default=base.ROOT / "behavior_710/intent")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        self_check()
        return
    with (args.input / "per_query.csv").open() as source:
        queries = list(csv.DictReader(source))
    commands = [json.loads(line) for line in (args.input / "commands.jsonl").open()]
    for row in commands:
        row["intent"] = intent(row)
    identities = [(r["dataset"], r["query_id"], r["step"]) for r in commands]
    assert len(set(identities)) == len(commands) == sum(int(q["calls"]) for q in queries)
    by_query = Counter((r["dataset"], r["query_id"]) for r in commands)
    outcomes = {(q["dataset"], q["query_id"]): int(q["hit_at_1"]) for q in queries}
    assert len(outcomes) == len(queries)
    assert all(by_query[q["dataset"], q["query_id"]] == int(q["calls"]) for q in queries)
    assert all(outcomes[r["dataset"], r["query_id"]] == r["hit_at_1"] for r in commands)
    rows = table_rows(commands, queries)
    graph = base.read_json(args.input / "graph/statistics.json")
    # Stale-input guard: the graph statistics must count the same shell and Python calls.
    assert graph["checks"]["all_shell_calls"] == len(commands)
    assert graph["checks"]["python_shell_calls"] == sum(r["intent"] == "python" for r in commands)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "shell_intents.csv", rows)
    fields = ("dataset", "query_id", "hit_at_1", "step", "intent", "exit_code", "trace_path", "line")
    write_csv(args.output / "command_labels.csv", [{k: r[k] for k in fields} for r in commands])
    graph_rows = []
    for dataset, groups in graph["groups"].items():
        for outcome, group in groups.items():
            denominator = group["labels"]["edge_read"]["calls"]
            for key, values in group["labels"].items():
                graph_rows.append(dict(dataset=dataset, outcome=outcome, behavior=key,
                                       graph_calls=denominator, calls=values["calls"],
                                       percent=100 * values["calls"] / denominator if denominator else None))
    write_csv(args.output / "python_graph_operations.csv", graph_rows)
    provenance = [str(p.resolve()) for p in [Path(__file__), Path(base.__file__), args.input / "commands.jsonl", args.input / "per_query.csv", args.input / "graph/statistics.json"]]
    (args.output / "provenance.json").write_text(json.dumps({"sources": provenance, "category_priority": list(CATEGORIES), "scope": "Final scored attempts; completed shell calls; nonzero exits included; static classification"}, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42}):
        plot_shell(rows, args.output, ["all"], "bash_intent_distribution")
        plot_shell(rows, args.output, ["all", "correct", "wrong"], "bash_intent_by_outcome")
        plot_graph(graph, args.output)
    lines = ["# Command-level intent statistics", "", "Final scored attempts; one category per shell call; columns use their own call denominator.", ""]
    for dataset in ["ALL", "amazon", "mag", "prime"]:
        selected = {(r["outcome"], r["category"]): r for r in rows if r["dataset"] == dataset}
        lines += [f"## {dataset}", "", "| Intent | All calls | All % | Correct calls | Correct % | Wrong calls | Wrong % |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for key in sorted(CATEGORIES, key=lambda k: -selected["all", k]["calls"]):
            a, c, w = [selected[o, key] for o, _ in OUTCOMES]
            lines.append(f"| {CATEGORIES[key][0]} | {a['calls']} | {a['percent']:.2f}% | {c['calls']} | {c['percent']:.2f}% | {w['calls']} | {w['percent']:.2f}% |")
        lines.append("")
    (args.output / "tables.md").write_text("\n".join(lines) + "\n")
    print(json.dumps([r for r in rows if r["dataset"] == "ALL" and r["outcome"] == "all"], indent=2))


if __name__ == "__main__":
    main()
