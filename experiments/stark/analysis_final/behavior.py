"""Behavior metrics per configuration.

All arms: H@1, returned IDs, tool calls, tokens, wall time, cost.
File arms: aligned nine-category command distribution (align_intent_distribution.classify, the classifier
used for the BC+ comparison), Python share, graph operations inside Python (analyze_graph_behavior),
model-visible output volume and truncation counts.
Tool arms: calls per tool, share of `neighbors` calls that hit the k cap, A5 service calls per bash call.

Unit for the command distribution: one issued shell command. Pi `read` calls and custom tools are excluded
from the shell denominator (reported separately), as in the existing analyses. Codex commands come from
stdout.jsonl command_execution items (nested commands of a code cell count individually).
Truncation: Pi L3 marker, H6 marker, Codex "tokens truncated" marker in model-visible text. Codex runs without
recorded model input have no visible text; their count is an estimate = raw command outputs above 40 KB (the Codex
limit), labeled "est.".
"""

import argparse
from collections import Counter

import common as C

GRAPH_SHOW = ("edge_read", "neighbor_collect", "chained_expansion", "intersection", "adjacency_index", "seed_loop",
              "node_enrichment", "candidate_filter")
INTENT_COLORS = dict(zip([k for k in C.INTENTS if k != "other"], C.PALETTE))
INTENT_COLORS["other"] = C.GRAY


def summarize(cfg, rows):
    out = []
    for ds in (*C.DATASETS, "all"):
        sel = [r for r in rows if r["terminal"] and (ds == "all" or r["dataset"] == ds)]
        n = len(sel)
        if not n:
            out.append({"dataset": ds, "n": 0})
            continue
        intents = Counter()
        gops = Counter()
        tools = Counter()
        for r in sel:
            intents.update(r.get("intents") or {})
            gops.update({k: 1 for k, v in (r.get("graph_ops") or {}).items() if v})
            tools.update(r.get("tools_by_name") or {})
        bash = sum(intents.values())
        py = sum(r.get("python_calls", 0) for r in sel)
        trunc_keys = ("trunc_l3", "trunc_h6", "trunc_codex") if cfg["harness"] != "codex_raw" else ("trunc_est_large",)
        trunc_per_q = [sum(r.get(k, 0) for k in trunc_keys) for r in sel]
        nb = sum(r.get("neighbors_calls", 0) for r in sel)
        row = {"dataset": ds, "n": n, "h1": sum(r["hit1"] for r in sel) / n,
               "median_returned": C.median([r["returned"] for r in sel]),
               "median_calls": C.median([r.get("tool_calls_result") if r.get("tool_calls_result") is not None else r.get("calls") for r in sel]),
               "bash_calls": bash, "bash_per_query": bash / n, "read_calls": sum(r.get("read_calls", 0) for r in sel),
               "python_calls": py, "python_share": py / bash if bash else None,
               "queries_with_python": sum(r.get("python_calls", 0) > 0 for r in sel) / n,
               "python_edge_share": (sum(r.get("py_edges_calls", 0) for r in sel) / py) if py else None,
               "python_node_share": (sum(r.get("py_nodes_calls", 0) for r in sel) / py) if py else None,
               "median_visible_chars": C.median([r.get("visible_chars") for r in sel if r.get("visibility") != "missing"]),
               "visibility": "upper_bound (raw)" if cfg["harness"] == "codex_raw" else "recorded",
               "truncated_outputs": sum(trunc_per_q), "truncated_per_query": sum(trunc_per_q) / n,
               "queries_truncated": sum(t > 0 for t in trunc_per_q) / n,
               "truncation_basis": "est. raw > 40 KB" if cfg["harness"] == "codex_raw" else "markers",
               "trunc_l3": sum(r.get("trunc_l3", 0) for r in sel), "trunc_h6": sum(r.get("trunc_h6", 0) for r in sel),
               "trunc_codex": sum(r.get("trunc_codex", 0) for r in sel),
               "median_input_tokens": C.median([r.get("input_tokens") for r in sel]),
               "median_output_tokens": C.median([r.get("output_tokens") for r in sel]),
               "total_input_tokens": sum(r.get("input_tokens") or 0 for r in sel),
               "total_output_tokens": sum(r.get("output_tokens") or 0 for r in sel),
               "median_wall_seconds": C.median([r.get("wall_seconds") for r in sel]),
               "known_cost_usd": sum(r.get("cost_usd") or 0 for r in sel),
               "service_calls": sum(r.get("service_calls", 0) for r in sel),
               "neighbors_calls": nb, "neighbors_cap_rate": (sum(r.get("neighbors_capped", 0) for r in sel) / nb) if nb else None,
               "service_per_bash": (sum(r.get("service_calls", 0) for r in sel) / bash) if (cfg["kind"] == "tool_code" and bash) else None}
        for k in C.INTENTS:
            row[f"intent_{k}"] = intents[k] / bash if bash else None
        for k in C.GRAPH_LABELS:
            row[f"graph_{k}_queries"] = gops[k] / n
        row["tools_by_name"] = dict(tools)
        out.append(row)
    return out


def plot_intents(cfgs, tables, name, title, keep_all=False):
    """keep_all: keep every configuration as a column (tool-only arms and not-yet-run arms are annotated)."""
    present = [c for c in cfgs if keep_all or next(r for r in tables[c["name"]] if r["dataset"] == "all").get("bash_calls")]
    if not present or not any(next(r for r in tables[c["name"]] if r["dataset"] == "all").get("bash_calls") for c in present):
        C.placeholder_fig(name, title, "No shell commands recorded yet for these configurations.")
        return
    plt = C.mpl()
    fig, ax = plt.subplots(figsize=(max(3.3, 0.55 * len(present) + 2.4), 3.0))
    order = [k for k in C.INTENTS if k != "other"] + ["other"]
    first_drawn = next(c for c in present if next(r for r in tables[c["name"]] if r["dataset"] == "all").get("bash_calls"))
    for i, c in enumerate(present):
        row = next(r for r in tables[c["name"]] if r["dataset"] == "all")
        if not row.get("bash_calls"):
            note = "not run yet" if not row.get("n") else ("tool calls only" if c["kind"] == "tool" else "no shell calls")
            ax.text(i, 50, note, ha="center", va="center", rotation=90, fontsize=6, color=C.INK2)
            ax.text(i, 101.5, f"{row.get('n', 0)}q", ha="center", fontsize=5.5, color=C.INK2)
            continue
        bottom = 0.0
        for k in order:
            v = 100 * (row.get(f"intent_{k}") or 0)
            ax.bar(i, v, bottom=bottom, width=0.66, color=INTENT_COLORS[k], edgecolor="white", linewidth=0.8,
                   label=C.INTENT_LABEL[k] if c is first_drawn else None)
            if v >= 7:
                ax.text(i, bottom + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=6, color="white" if k in ("search_limit", "python", "count") else C.INK)
            bottom += v
        ax.text(i, 101.5, f"{row['n']}q", ha="center", fontsize=5.5, color=C.INK2)
    ax.set_xticks(range(len(present)), [c["label"] for c in present], rotation=35, ha="right", fontsize=6.5)
    ax.set_ylim(0, 106)
    ax.set_ylabel("% of issued shell commands")
    ax.grid(axis="x", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.5)
    ax.set_title(title, fontsize=8.5)
    fig.tight_layout()
    C.save_fig(fig, name)
    plt.close(fig)


def main():
    ap = C.common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    print("behavior.py: loading")
    cfgs, data, cov = C.load_all(args.configs, use_cache=not args.no_cache)
    tables, long_rows = {}, []
    for c in cfgs:
        tables[c["name"]] = summarize(c, data[c["name"]])
        for r in tables[c["name"]]:
            long_rows.append({"config": c["name"], "label": c["label"], "tier": c["tier"], "kind": c["kind"], **r})
    C.write_csv(C.OUT / "behavior_by_config.csv", long_rows,
                fields=sorted({k for r in long_rows for k in r}, key=lambda k: (k not in ("config", "label", "tier", "kind", "dataset", "n"), k)))

    def main_table(tier):
        rows = []
        for c in [c for c in cfgs if c["tier"] == tier]:
            for r in tables[c["name"]]:
                if not r["n"]:
                    continue
                rows.append([c["label"], C.DS_LABEL[r["dataset"]], r["n"], C.fmt(r["h1"], True), C.fmt(r["median_returned"]),
                             C.fmt(r["median_calls"]), C.fmt(r["bash_per_query"]), C.fmt(r["python_share"], True),
                             C.fmt(r["python_edge_share"], True), C.fmt(r["median_visible_chars"], digits=0),
                             f"{C.fmt(r['truncated_per_query'], digits=2)}{' est.' if r['truncation_basis'] != 'markers' else ''}",
                             C.fmt(r["queries_truncated"], True), C.fmt(r["median_input_tokens"], digits=0),
                             C.fmt(r["median_output_tokens"], digits=0), C.fmt(r["median_wall_seconds"], digits=0),
                             f"{r['known_cost_usd']:.2f}"])
        return C.md_table(["Configuration", "Dataset", "N", "H@1", "Med. IDs", "Med. tool calls", "Shell/q", "Python %",
                           "Py reads edges %", "Med. visible chars", "Trunc./q", "Queries trunc. %", "Med. input tok",
                           "Med. output tok", "Med. s", "Known USD"], rows, [False, False] + [True] * 14) if rows else "_No terminal results yet._"

    def intent_table(tier):
        rows = []
        for c in [c for c in cfgs if c["tier"] == tier]:
            r = next(x for x in tables[c["name"]] if x["dataset"] == "all")
            if r.get("bash_calls"):
                rows.append([c["label"], r["bash_calls"], *[C.fmt(r[f"intent_{k}"], True) for k in C.INTENTS]])
        return C.md_table(["Configuration", "Shell calls", *[C.INTENT_LABEL[k] for k in C.INTENTS]], rows) if rows else "_n/a_"

    def graph_table(tier):
        rows = []
        for c in [c for c in cfgs if c["tier"] == tier]:
            for r in tables[c["name"]]:
                if r["n"] and r.get("python_calls"):
                    rows.append([c["label"], C.DS_LABEL[r["dataset"]], r["n"], C.fmt(r["queries_with_python"], True),
                                 *[C.fmt(r[f"graph_{k}_queries"], True) for k in GRAPH_SHOW]])
        return C.md_table(["Configuration", "Dataset", "N", "Queries w/ Python %", *[C.graph.LABELS[k] + " %q" for k in GRAPH_SHOW]],
                          rows, [False, False] + [True] * (2 + len(GRAPH_SHOW))) if rows else "_n/a_"

    def tool_table():
        rows = []
        for c in [c for c in cfgs if c["kind"] in ("tool", "tool_code")]:
            for r in tables[c["name"]]:
                if r["n"]:
                    tb = r.get("tools_by_name") or {}
                    rows.append([c["label"], C.DS_LABEL[r["dataset"]], r["n"],
                                 ", ".join(f"{k}:{v / r['n']:.1f}" for k, v in sorted(tb.items())),
                                 C.fmt(r["service_calls"] / r["n"]), r["neighbors_calls"], C.fmt(r["neighbors_cap_rate"], True),
                                 C.fmt(r["service_per_bash"])])
        return C.md_table(["Configuration", "Dataset", "N", "Model tool calls per query", "Service calls/q", "neighbors calls",
                           "neighbors hit k cap %", "A5 service calls per bash"], rows, [False, False, True, False, True, True, True, True]) if rows else "_No tool-arm results yet._"

    md = ["# Behavior", "", "Generated by `behavior.py`. Medians over terminal queries; percentages of issued shell "
          "commands unless marked %q (share of queries). `Python %` = shell commands running Python. `Py reads edges %` = "
          "Python commands whose code or arguments reference edge files. Truncation counts are model-visible markers "
          "(Pi L3 `[...truncated`, H6 `harness output limit`, Codex `tokens truncated`); Codex without recorded model input uses the estimate "
          "`raw output > 40 KB`.", "",
          "## Coverage", "", C.coverage_md(cov, cfgs), "",
          "## Paper configurations: effectiveness, cost, output volume", "", main_table("paper"), "",
          "## Paper configurations: command distribution (aligned nine-category classifier)", "", intent_table("paper"), "",
          "## Paper configurations: graph operations inside Python", "", graph_table("paper"), "",
          "## Tool arms", "", tool_table(), "",
          "## Development configurations (val)", "", main_table("dev"), "", intent_table("dev"), "", graph_table("dev"), "",
          "Figures: main `figures/behavior_intents_710.pdf`; appendix `figures/appendix/behavior_intents_full.pdf`, "
          "`figures/appendix/behavior_intents_dev.pdf`. "
          "LaTeX: `tex/behavior_table.tex`."]
    C.write_text(C.OUT / "behavior.md", "\n".join(md))

    tex_rows = []
    for c in [c for c in cfgs if c["tier"] == "paper" and c["name"] != "pi_final_human"]:
        r = next(x for x in tables[c["name"]] if x["dataset"] == "all")
        if not r["n"]:
            tex_rows.append([c["label"], *["--"] * 8])
            continue
        tex_rows.append([c["label"], r["n"], C.fmt(r["median_calls"]), C.fmt(r["python_share"], True),
                         C.fmt(r["python_edge_share"], True), C.fmt(r["truncated_per_query"], digits=2) + ("*" if r["truncation_basis"] != "markers" else ""),
                         C.fmt((r["median_input_tokens"] or 0) / 1000, digits=0) + "k", C.fmt(r["median_wall_seconds"], digits=0),
                         f"{r['known_cost_usd']:.2f}"])
    C.write_text(C.OUT / "tex/behavior_table.tex", C.latex_table(
        ["Config", "N", "Calls", "Py %", "Py+edges %", "Trunc./q", "In tok", "Sec.", "USD"], tex_rows,
        "Behavior per configuration (all datasets pooled; medians per query). Py+edges: share of Python commands that "
        "read edge files. *Estimated from raw output size (model input not recorded).", "tab:behavior"))
    by = {c["name"]: c for c in cfgs}
    plot_intents([by[n] for n in C.MAIN_710 if n in by], tables, "behavior_intents_710",
                 "Shell command intents (test-0.1, 710 queries)", keep_all=True)
    plot_intents([by[n] for n in C.FULL_SET if n in by], tables, "appendix/behavior_intents_full", "Shell command intents (full test)")
    plot_intents([c for c in cfgs if c["tier"] == "dev"], tables, "appendix/behavior_intents_dev", "Shell command intents (val)")
    print("behavior.py: wrote behavior.md, behavior_by_config.csv, tex/behavior_table.tex, figures/behavior_intents_710.pdf, figures/appendix/behavior_intents_*.pdf")


if __name__ == "__main__":
    main()
