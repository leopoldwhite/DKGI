"""H@1 stratified by text constraint (per-query tags) and by the closed-book name probe A0.

Text-constraint strata: the model annotator's per-query tags in
data/query_tags/tags_effective.jsonl of this package (fields dataset, split, query_id,
has_text_constraint; --text-constraint-tags names another file). Every MAG query counts as text-constrained and
MAG is not stratified; Amazon and Prime use the annotator's labels; unresolved tags are excluded and counted.

A0 strata (MAG, Prime; A0-Name): a query is "known without the corpus" when the A0-Name probe has H@1 = 1
(sensitivity: H@20 = 1, i.e. any of the 20 names maps to a gold node). H@1 on the "unknown" group is the
prespecified net gain from reading the corpus. A0 terminal failures score 0 (probe failure policy); pending
A0 queries are excluded and the paired coverage is reported. Amazon has no A0-Name probe (A0-ID only).
"""

import argparse
from collections import Counter
from pathlib import Path

import common as C

TAGS = C.STARK.parents[1] / "data/query_tags/tags_effective.jsonl"  # --text-constraint-tags
A0 = C.STARK / "contamination/runs"


def load_tags():
    tags, unresolved = {}, Counter()
    if not TAGS.exists():
        return tags, unresolved
    for r in C.read_jsonl(TAGS):
        split = "test" if r["split"] == "test-0.1" else r["split"]
        if r.get("has_text_constraint") is None:
            unresolved[(r["dataset"], split)] += 1
            continue
        tags[(r["dataset"], split, str(r["query_id"]))] = bool(r["has_text_constraint"])
    return tags, unresolved


def load_a0():
    """{(probe, ds, qid): {'h1':..., 'h20':..., 'terminal': bool}} from contamination run trees."""
    out = {}
    for probe in ("A0-ID", "A0-Name"):
        for ds in C.DATASETS:
            d = A0 / probe / ds / "test"
            if not d.is_dir():
                continue
            for q in d.iterdir():
                p = q / "result.json"
                if not p.exists():
                    continue
                try:
                    r = C.read_json(p)
                except ValueError:
                    continue
                terminal = bool(r.get("terminal", r.get("status") in ("scored", "failed")))
                if not terminal:
                    continue
                m = C.parse_obj(r.get("metrics")) if r.get("status") == "scored" else {}
                out[(probe, ds, q.name)] = {"h1": float(m.get("hit_at_1", 0.0)), "h20": float(m.get("hit_at_20", 0.0)),
                                            "status": r.get("status")}
    return out


def a0_summary(a0):
    rows = []
    for probe in ("A0-ID", "A0-Name"):
        for ds in C.DATASETS:
            sel = [v for (p, d, _), v in a0.items() if p == probe and d == ds]
            if not sel and not (probe == "A0-Name" and ds == "amazon"):
                rows.append({"probe": probe, "dataset": ds, "n": 0})
            if not sel:
                continue
            n = len(sel)
            h1 = sum(v["h1"] for v in sel)
            rows.append({"probe": probe, "dataset": ds, "n": n, "h1": h1 / n,
                         "h20": sum(v["h20"] for v in sel) / n, "failed": sum(v["status"] != "scored" for v in sel),
                         "contamination_signal": (h1 / n >= 0.01) if probe == "A0-ID" else None})
    return rows


def stratum(rows, label):
    n = len(rows)
    k = sum(r["hit1"] for r in rows)
    cats = Counter(C.error_category(r) for r in rows)
    return {"stratum": label, "n": n, "h1": k / n if n else None,
            **{f"cat_{c}": cats[c] / n if n else None for c in C.ERROR_CATS}}


def main():
    global TAGS
    ap = C.common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--text-constraint-tags", type=Path, default=TAGS,
                    help="JSONL of per-query text-constraint tags (default: the package's data/query_tags/tags_effective.jsonl)")
    args = ap.parse_args()
    TAGS = args.text_constraint_tags
    print("strata.py: loading")
    cfgs, data, cov = C.load_all(args.configs, use_cache=not args.no_cache)
    tags, unresolved = load_tags()
    a0 = load_a0()
    text_rows, a0_rows = [], []
    for c in cfgs:
        term = [r for r in data[c["name"]] if r["terminal"]]
        split = "test" if c["split"] in ("test", "test-0.1") else c["split"]
        for ds in C.DATASETS:
            sel = [r for r in term if r["dataset"] == ds]
            if not sel:
                continue
            if ds == "mag":
                text_rows.append({"config": c["name"], "label": c["label"], "tier": c["tier"], "dataset": ds,
                                  **stratum(sel, "all (text-constrained)"), "untagged": 0})
                continue
            known = [r for r in sel if (ds, split, r["query_id"]) in tags]
            for flag, name in ((True, "text constraint"), (False, "relational only")):
                text_rows.append({"config": c["name"], "label": c["label"], "tier": c["tier"], "dataset": ds,
                                  **stratum([r for r in known if tags[(ds, split, r["query_id"])] == flag], name),
                                  "untagged": len(sel) - len(known)})
        if split != "test":
            continue
        for ds in ("mag", "prime"):
            sel = [r for r in term if r["dataset"] == ds]
            if not sel:
                continue
            paired = [r for r in sel if ("A0-Name", ds, r["query_id"]) in a0]
            for crit in ("h1", "h20"):
                known = [r for r in paired if a0[("A0-Name", ds, r["query_id"])][crit] >= 1]
                unknown = [r for r in paired if a0[("A0-Name", ds, r["query_id"])][crit] < 1]
                a0_known_dci_wrong = sum(1 for r in known if not r["hit1"])
                ku, uu = stratum(known, "A0 correct"), stratum(unknown, "A0 incorrect")
                a0_rows.append({"config": c["name"], "label": c["label"], "tier": c["tier"], "dataset": ds, "criterion": crit,
                                "dci_n": len(sel), "paired": len(paired), "known_n": ku["n"], "unknown_n": uu["n"],
                                "dci_h1_all_paired": (sum(r["hit1"] for r in paired) / len(paired)) if paired else None,
                                "dci_h1_known": ku["h1"], "dci_h1_unknown": uu["h1"],
                                "a0_h1_on_paired": (len(known) / len(paired)) if paired else None,
                                "cells": {"a0_wrong_dci_wrong": sum(1 for r in unknown if not r["hit1"]),
                                          "a0_wrong_dci_right": sum(1 for r in unknown if r["hit1"]),
                                          "a0_right_dci_wrong": a0_known_dci_wrong,
                                          "a0_right_dci_right": len(known) - a0_known_dci_wrong}})
    a0s = a0_summary(a0)
    C.write_csv(C.OUT / "strata_text_constraint.csv", text_rows)
    C.write_csv(C.OUT / "strata_a0.csv", a0_rows)
    C.write_csv(C.OUT / "a0_probe_summary.csv", a0s)

    def text_table(tier):
        rows = [[r["label"], C.DS_LABEL[r["dataset"]], r["stratum"], r["n"], C.fmt(r["h1"], True),
                 C.fmt(r["cat_ranking"], True),
                 C.fmt(r["cat_seen_not_selected"], True), C.fmt(r["cat_retrieval_miss"], True), C.fmt(r["cat_empty_format"], True),
                 r["untagged"]] for r in text_rows if r["tier"] == tier and r["n"]]
        return C.md_table(["Configuration", "Dataset", "Stratum", "N", "H@1", "Rank %", "Seen %", "Miss %",
                           "Empty %", "Untagged"], rows, [False, False, False] + [True] * 7) if rows else "_No terminal results yet._"

    def a0_table(tier):
        rows = [[r["label"], C.DS_LABEL[r["dataset"]], "H@1" if r["criterion"] == "h1" else "H@20 (sens.)",
                 f"{r['paired']}/{r['dci_n']}", r["known_n"], r["unknown_n"], C.fmt(r["dci_h1_known"], True),
                 f"**{C.fmt(r['dci_h1_unknown'], True)}**",
                 C.fmt(r["dci_h1_all_paired"], True), C.fmt(r["a0_h1_on_paired"], True),
                 "{a0_wrong_dci_wrong}/{a0_wrong_dci_right}/{a0_right_dci_wrong}/{a0_right_dci_right}".format(**r["cells"])]
                for r in a0_rows if r["tier"] == tier and r["paired"]]
        return C.md_table(["Configuration", "Dataset", "A0 'known' criterion", "Paired/n", "Known n", "Unknown n",
                           "H@1 known", "H@1 unknown (net gain)", "H@1 paired", "A0-Name H@1",
                           "A0w-Dw / A0w-Dr / A0r-Dw / A0r-Dr"], rows, [False, False, False] + [True] * 8) if rows else "_No paired rows yet._"

    probe_rows = [[r["probe"], C.DS_LABEL[r["dataset"]], r["n"], C.fmt(r.get("h1"), True, 2),
                   C.fmt(r.get("h20"), True, 2), r.get("failed", 0),
                   {True: "yes (>=1%)", False: "no", None: "-"}[r.get("contamination_signal")]] for r in a0s]
    unres = ", ".join(f"{d}/{s}: {n}" for (d, s), n in sorted(unresolved.items())) or "none"
    md = ["# Stratified H@1", "", "Generated by `strata.py`.", "",
          "## A0 probe status (contamination/runs, terminal results only)", "",
          C.md_table(["Probe", "Dataset", "Terminal", "H@1 %", "H@20 %", "Failed", "Contamination signal"], probe_rows,
                     [False, False] + [True] * 5), "",
          "A0 is incomplete until the probe run reports completion.", "",
          "## Net gain from reading the corpus (A0-Name strata; MAG, Prime; test)", "",
          "Bold = H@1 on queries the corpus-free probe could not answer (prespecified net gain).", "",
          "### Paper configurations", "", a0_table("paper"), "",
          "## H@1 by text constraint", "",
          "Tags: `data/query_tags/tags_effective.jsonl` unless `--text-constraint-tags` names another file. "
          f"Unresolved tags (excluded): {unres}. MAG is not stratified.", "",
          "### Paper configurations", "", text_table("paper"), "",
          "### Development configurations (val)", "", text_table("dev"), "",
          "## Coverage", "", C.coverage_md(cov, cfgs), "",
          "LaTeX: `tex/strata_text_table.tex`, `tex/strata_a0_table.tex`, `tex/appendix/strata_a0_h20_table.tex`."]
    C.write_text(C.OUT / "strata.md", "\n".join(md))

    by = {c["name"]: c for c in cfgs}
    groups = [[n for n in grp if n in by] for grp in (C.FULL_SET, C.MAIN_710)]

    def blocks(make_row):
        rows = []
        for names in groups:
            if rows and names:
                rows.append("midrule")
            rows += [make_row(by[n]) for n in names]
        return rows

    def text_row(c):
        cells = []
        for ds in ("amazon", "prime"):
            for name in ("text constraint", "relational only"):
                r = next((x for x in text_rows if x["config"] == c["name"] and x["dataset"] == ds and x["stratum"] == name), None)
                cells += [C.fmt(r["h1"], True) if r and r["n"] else "--", r["n"] if r and r["n"] else "--"]
        return [c["label"], *cells]

    C.write_text(C.OUT / "tex/strata_text_table.tex", C.latex_table(
        ["Config", "Ama text", "n", "Ama rel", "n", "Pri text", "n", "Pri rel", "n"], blocks(text_row),
        "H@1 by whether the query adds a text constraint to its relational constraints (per-query tags). MAG is "
        "treated as text-constrained throughout and is not stratified. Top block: full test; bottom: test-0.1 (710).",
        "tab:strata-text", star=True))

    def a0_row(crit):
        def row(c):
            cells = []
            for ds in ("mag", "prime"):
                r = next((x for x in a0_rows if x["config"] == c["name"] and x["dataset"] == ds and x["criterion"] == crit), None)
                if not r or not r["paired"]:
                    cells += ["--"] * 4
                    continue
                cells += [r["paired"], C.fmt(r["a0_h1_on_paired"], True),
                          f"{C.fmt(r['dci_h1_known'], True)} ({r['known_n']})",
                          f"{C.fmt(r['dci_h1_unknown'], True)} ({r['unknown_n']})"]
            return [c["label"], *cells]
        return row

    head = ["Config", "MAG n", "A0 H@1", "H@1 on A0+", "H@1 on A0- (net gain)", "Pri n", "A0 H@1", "H@1 on A0+", "H@1 on A0- (net gain)"]
    cap = ("H@1 stratified by the corpus-free probe A0-Name (no corpus, no tools). A0+: the probe's {k} maps to a gold "
           "node; A0-: it does not. H@1 on A0- (n in parentheses) is the net gain from reading the corpus. "
           "n: questions with a terminal result of both the configuration and A0. Amazon has no A0-Name probe; A0-ID H@1: {id}.")
    id_note = ", ".join(f"{C.DS_LABEL[r['dataset']]} {C.fmt(r.get('h1'), True, 2)}% (n={r['n']})" for r in a0s if r["probe"] == "A0-ID" and r["n"])
    C.write_text(C.OUT / "tex/strata_a0_table.tex", C.latex_table(
        head, blocks(a0_row("h1")), cap.format(k="top-1 name", id=id_note or "n/a"), "tab:strata-a0", star=True))
    C.write_text(C.OUT / "tex/appendix/strata_a0_h20_table.tex", C.latex_table(
        head, blocks(a0_row("h20")), "Sensitivity: " + cap.format(k="any of its 20 names", id=id_note or "n/a"),
        "tab:strata-a0-h20", star=True))
    print("strata.py: wrote strata.md, strata_*.csv, a0_probe_summary.csv, tex/strata_text_table.tex, tex/strata_a0_table.tex, tex/appendix/strata_a0_h20_table.tex")


if __name__ == "__main__":
    main()
