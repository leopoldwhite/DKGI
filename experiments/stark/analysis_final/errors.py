"""Error taxonomy per configuration x dataset, from model-visible tool outputs.

Categories (every terminal query gets exactly one, checked in this order):
  correct            H@1 = 1 (recomputed from prediction and the split's gold file)
  empty_format       failed run or provider refusal, unparseable prediction, or empty ID list
  ranking            a gold ID is in the returned list, but not at rank 1
  seen_not_selected  no gold in the list, but a gold ID appeared in some model-visible tool output
  retrieval_miss     no gold in the list and no gold ID in any model-visible tool output
  unknown_visibility wrong, gold not in list, and model-visible outputs were not recorded

"Appeared" (default, --seen record; main text): a gold node record in a visible tool output,
i.e. "node_id": ID in JSONL text or a tool-output header "[ID]".
Appendix sensitivity (--seen id): the gold ID as any standalone integer token (also counts ID lists in edge rows).
Codex runs without a recorded model input: the "visible" text is the raw command output (upper bound on
seen, lower bound on retrieval_miss); a head+tail truncation estimate is reported as a further sensitivity.
"""

import argparse
from collections import Counter

import common as C

CATS = ("ranking", "seen_not_selected", "retrieval_miss", "empty_format", "unknown_visibility")
SHORT = {"ranking": "Rank", "seen_not_selected": "Seen", "retrieval_miss": "Miss", "empty_format": "Empty",
         "unknown_visibility": "Unk."}
COLORS = {"ranking": C.PALETTE[0], "seen_not_selected": C.PALETTE[3], "retrieval_miss": C.PALETTE[1],
          "empty_format": C.PALETTE[6], "unknown_visibility": C.GRAY}


def summarize(rows, seen_key):
    out = []
    for ds in (*C.DATASETS, "all"):
        sel = [r for r in rows if r["terminal"] and (ds == "all" or r["dataset"] == ds)]
        n = len(sel)
        cats = Counter(C.error_category(r, seen_key) for r in sel)
        wrong = n - cats["correct"]
        row = {"dataset": ds, "n": n, "h1": cats["correct"] / n if n else None, "wrong": wrong}
        for c in CATS:
            row[f"{c}_n"] = cats[c]
            row[f"{c}_of_all"] = cats[c] / n if n else None
            row[f"{c}_of_errors"] = cats[c] / wrong if wrong else None
        out.append(row)
    return out


def seen_variants(cfg):
    keys = [("record", "gold_seen_record"), ("id", "gold_seen_id")]
    if cfg["harness"] == "codex_raw":
        keys += [("record_head_tail_estimate", "gold_seen_sim"), ("id_head_tail_estimate", "gold_seen_sim_id")]
    return keys


def plot(cfgs, tables, name, title):
    present = [c for c in cfgs if any(r["n"] for r in tables[c["name"]] if r["dataset"] != "all")]
    if not present:
        C.placeholder_fig(name, title, "No terminal results yet for these configurations.")
        return
    plt = C.mpl()
    height = 0.3 * len(present) + 1.5
    fig, axes = plt.subplots(1, 3, figsize=(7.0, height), sharey=True)
    labels = [c["label"] for c in present]
    for ax, ds in zip(axes, C.DATASETS):
        left = [0.0] * len(present)
        for cat in CATS:
            vals = []
            for c in present:
                row = next(r for r in tables[c["name"]] if r["dataset"] == ds)
                vals.append(100 * (row[f"{cat}_of_all"] or 0))
            ax.barh(range(len(present)), vals, left=left, color=COLORS[cat], height=0.62,
                    edgecolor="white", linewidth=0.8, label=C.ERROR_LABEL[cat])
            left = [a + b for a, b in zip(left, vals)]
        for i, c in enumerate(present):
            n = next(r for r in tables[c["name"]] if r["dataset"] == ds)["n"]
            ax.text(left[i] + 1, i, f"n={n}", va="center", fontsize=6, color=C.INK2)
        ax.set_title(C.DS_LABEL[ds], fontsize=8.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of all queries (bar = 1 - H@1)" if ds == "mag" else "")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(present)), labels)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0.42 / height, 1, 1))
    C.save_fig(fig, name)
    plt.close(fig)


def main():
    ap = C.common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--seen", choices=("id", "record"), default="record")
    args = ap.parse_args()
    seen_key = {"id": "gold_seen_id", "record": "gold_seen_record"}[args.seen]
    print("errors.py: loading")
    cfgs, data, cov = C.load_all(args.configs, use_cache=not args.no_cache)
    tables, long_rows, per_query, sens = {}, [], [], []
    for c in cfgs:
        tables[c["name"]] = summarize(data[c["name"]], seen_key)
        for r in tables[c["name"]]:
            long_rows.append({"config": c["name"], "label": c["label"], "tier": c["tier"], "seen": args.seen, **r})
        for variant, key in seen_variants(c):
            for r in summarize(data[c["name"]], key):
                sens.append({"config": c["name"], "label": c["label"], "variant": variant, **r})
        for r in data[c["name"]]:
            if r["terminal"]:
                per_query.append({"config": c["name"], "dataset": r["dataset"], "query_id": r["query_id"],
                                  "category": C.error_category(r, seen_key),
                                  "category_id": C.error_category(r, "gold_seen_id"),
                                  "gold_rank": r["gold_rank"], "returned": r["returned"], "visibility": r["visibility"],
                                  "gold_first_call": r["gold_first_call"], "top1_seen": r["top1_seen"]})
    out = C.OUT
    C.write_csv(out / "errors_by_config.csv", long_rows)
    C.write_csv(out / "errors_seen_sensitivity.csv", sens)
    C.write_csv(out / "errors_per_query.csv", per_query)

    def table(tier):
        rows = []
        for c in [c for c in cfgs if c["tier"] == tier]:
            for r in tables[c["name"]]:
                if not r["n"]:
                    continue
                rows.append([c["label"], C.DS_LABEL[r["dataset"]], r["n"], C.fmt(r["h1"], True),
                             *[f"{C.fmt(r[f'{k}_of_all'], True)} ({C.fmt(r[f'{k}_of_errors'], True, 0)})" for k in CATS]])
        return C.md_table(["Configuration", "Dataset", "N", "H@1", *[SHORT[k] + " % all (% err)" for k in CATS]], rows,
                          [False, False] + [True] * 7) if rows else "_No terminal results yet._"

    sens_rows = [[s["label"], C.DS_LABEL[s["dataset"]], s["variant"], s["n"],
                  *[C.fmt(s[f"{k}_of_errors"], True) for k in ("ranking", "seen_not_selected", "retrieval_miss", "empty_format")]]
                 for s in sens if s["n"] and s["dataset"] in ("amazon", "all") and s["config"] in
                 ("pi_final_full", "H", "A1", "A2", "A3", "A5", "A6", "t11_pi_p1", "r1_pi_p1r1", "r1_codex_p1r1")]
    md = ["# Error taxonomy", "",
          "Generated by `errors.py`. One category per terminal query; percentages are of all terminal queries, "
          "in parentheses of wrong queries. Pending queries are excluded.", "",
          "Seen definition: `record` (main text): a gold node record (`\"node_id\": ID`) or tool-output header (`[ID]`) "
          "in a model-visible tool output." if args.seen == "record" else
          "Seen definition: `id` (appendix sensitivity): the gold ID as a standalone integer in a model-visible tool output.", "",
          "Order: correct, empty/format, ranking, seen-not-selected, retrieval-miss. Failed runs and provider refusals "
          "are counted as empty/format.", "",
          "**Codex without recorded model input**: `Seen` uses the raw command output "
          "(stdout aggregated_output), an *upper bound* on what the model saw; its `Miss` is a *lower bound*.", "",
          "## Coverage", "", C.coverage_md(cov, cfgs), "",
          "## Paper configurations (test)", "", table("paper"), "",
          "## Development configurations (val; for script testing and appendix)", "", table("dev"), "",
          "## Sensitivity to the definition of \"seen\" (share of wrong queries)", "",
          C.md_table(["Configuration", "Dataset", "Seen variant", "N", "Rank", "Seen", "Miss", "Empty"], sens_rows,
                     [False, False, False] + [True] * 5) if sens_rows else "_n/a_", "",
          "Figures: main `figures/errors_710.pdf`; appendix `figures/appendix/errors_full.pdf`, `figures/appendix/errors_dev.pdf`. "
          "LaTeX: `tex/errors_table.tex` (record criterion), `tex/appendix/errors_seen_id_table.tex` (integer criterion)."]
    C.write_text(out / "errors.md", "\n".join(md))

    by = {c["name"]: c for c in cfgs}

    def tex_for(key, variant_label, tables_, label, star=True):
        rows = []
        for group in (C.FULL_SET, C.MAIN_710):
            names = [n for n in group if n in by]
            if rows and names:
                rows.append("midrule")
            for name in names:
                c = by[name]
                cells = []
                for ds in C.DATASETS:
                    r = next(x for x in tables_[name] if x["dataset"] == ds)
                    cells += ([C.fmt(r["h1"], True), C.fmt(r["ranking_of_all"], True), C.fmt(r["seen_not_selected_of_all"], True),
                               C.fmt(r["retrieval_miss_of_all"], True), C.fmt(r["empty_format_of_all"], True)] if r["n"] else ["--"] * 5)
                rows.append([C.Raw(C.tex_escape(c["label"]) + ("$^\\dagger$" if c["harness"] == "codex_raw" else "")), *cells])
        header = ["Config", *[f"{C.DS_LABEL[d][:3]} {h}" for d in C.DATASETS for h in ("H@1", "Rk", "Seen", "Miss", "Emp")]]
        return C.latex_table(header, rows,
                             f"Error taxonomy (percent of all queries; {variant_label}). Categories are assigned in the order "
                             "correct, Emp, Rk, Seen, Miss. Emp: empty list, unparseable output, failed run or provider refusal; "
                             "Rk: gold returned but not first; Seen: gold appeared in a model-visible tool output but was not "
                             "returned; Miss: gold never appeared. Top block: full test; bottom block: test-0.1 (710). "
                             "Dagger: model-visible output not recorded (raw command output used), so Seen is an upper bound "
                             "and Miss a lower bound.", label, colspec="l" + "rrrrr" * 3, star=star)

    C.write_text(out / "tex/errors_table.tex", tex_for("record", "seen = gold node record in visible output", tables, "tab:errors"))
    id_tables = {c["name"]: summarize(data[c["name"]], "gold_seen_id") for c in cfgs}
    C.write_text(out / "tex/appendix/errors_seen_id_table.tex",
                 tex_for("id", "sensitivity: seen = gold ID as any standalone integer", id_tables, "tab:errors-seen-id"))
    plot([by[n] for n in C.MAIN_710 if n in by], tables, "errors_710", "Where the wrong answers go (test-0.1, 710 queries)")
    plot([by[n] for n in C.FULL_SET if n in by], tables, "appendix/errors_full", "Error taxonomy (full test)")
    plot([c for c in cfgs if c["tier"] == "dev"], tables, "appendix/errors_dev", "Error taxonomy on val (development runs)")
    print("errors.py: wrote errors.md, errors_*.csv, tex/errors_table.tex, tex/appendix/errors_seen_id_table.tex, figures/errors_710.pdf, figures/appendix/errors_*.pdf")


if __name__ == "__main__":
    main()
