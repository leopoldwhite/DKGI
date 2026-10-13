"""Final numbers for the main table and the ablation table (test split).

Main table: full test (7,108 queries), Pi + GPT-6-Luna, Pi + GPT-6-Astra and Codex + GPT-6-Luna; the human-generated
set (Luna; not reported in the paper). Ablation (test-0.1, 710): levels and paired differences vs A4 (point estimates,
common.paired_difference). Metrics: H@1, R@20, MRR. Failed terminal results score 0.
Ablation H@1 is also pooled over MAG and Prime together (the relation-heavy datasets; level only, point estimate;
paper: Full DKGI 80.0 vs Graph tools + code 78.4).
Reads result.json files only.

Writes out/final_numbers.md, out/final_numbers.json, out/tex/final_main_rows.tex, out/tex/final_ablation_table.tex.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
METRICS = (("hit_at_1", "H@1"), ("recall_at_20", "R@20"), ("mrr", "MRR"))


def metric(result, key):
    m = result.get("metrics") or {}
    if key == "mrr":
        return m.get("mrr_at_100", m.get("mrr"))
    return m.get(key)


def per_query(name):
    """{(ds, qid): result} for the terminal results of a configuration (failed terminal results count as 0)."""
    cfg = C.config_by_name()[name]
    out = {}
    for r, path in C.iter_results(cfg):
        if not r.get("terminal"):
            continue
        ds, qid = Path(path).parts[-4], Path(path).parts[-2]
        out[(ds, qid)] = r
    return out


def values(results, key):
    return {k: float(metric(r, key) or 0.0) for k, r in results.items()}


def level(vals, ds):
    v = [x for (d, _), x in vals.items() if d == ds]
    return (100 * sum(v) / len(v), len(v)) if v else (None, 0)


MAG_PRIME = ("mag", "prime")


def level_pooled(vals, datasets):
    """Mean over all queries of the given datasets together (each query counts once)."""
    v = [x for (d, _), x in vals.items() if d in datasets]
    return (100 * sum(v) / len(v), len(v)) if v else (None, 0)


def main_rows():
    rows = {}
    sources = {"luna": per_query("pi_final_full"), "astra": per_query("astra_pi_full"),
               "codex_luna": per_query("codex_luna_full"), "luna_human": per_query("pi_final_human")}
    for name, results in sources.items():
        for key, label in METRICS:
            rows.setdefault(name, {})[label] = {ds: level(values(results, key), ds) for ds in C.DATASETS}
    return rows


ARM_CONFIGS = (("A4", "A4"), ("A1", "A1"), ("A2", "A2"), ("A3", "A3"), ("A5", "A5"), ("A6", "A6"), ("H", "H"),
               ("X1", "X1"), ("Astra-Pi", "astra_pi_710"), ("Astra-Codex", "astra_codex_710"))


def ablation_metric(key):
    """Levels per dataset and pooled, and paired differences vs A4, for one metric."""
    arms = {arm: values(per_query(cfg), key) for arm, cfg in ARM_CONFIGS}
    a4 = arms["A4"]
    out = {}
    for arm, vals in arms.items():
        row = {ds: level(vals, ds) for ds in C.DATASETS}
        allv = list(vals.values())
        row["all"] = (100 * sum(allv) / len(allv) if allv else None, len(allv))
        row["mag+prime"] = level_pooled(vals, MAG_PRIME)
        if arm != "A4":
            pairs = {ds: [(a4[k], vals[k]) for k in vals if k[0] == ds and k in a4] for ds in C.DATASETS}
            row["delta_vs_A4"] = {ds: C.paired_difference({ds: pairs[ds]}) for ds in C.DATASETS}
            row["delta_vs_A4"]["all"] = C.paired_difference(pairs)
        out[arm] = row
    return out, arms


def ablation():
    """H@1 table (as before) plus pooled R@20 and MRR levels and paired differences for every arm."""
    out, arms = ablation_metric("hit_at_1")
    for key, label in METRICS[1:]:
        extra, _ = ablation_metric(key)
        for arm, row in extra.items():
            out[arm][label] = {"all": row["all"], "delta_vs_A4": row.get("delta_vs_A4", {}).get("all")}
    astra_pairs = {ds: [(arms["Astra-Pi"][k], arms["Astra-Codex"][k]) for k in arms["Astra-Codex"]
                        if k[0] == ds and k in arms["Astra-Pi"]] for ds in C.DATASETS}
    out["Astra-Codex"]["delta_vs_Astra-Pi"] = C.paired_difference(astra_pairs)
    return out


def f(x, d=1):
    return "n/a" if x is None else f"{x:.{d}f}"


def diff(t):
    return "n/a" if t is None or t[0] is None else f"{100 * t[0]:+.1f}"


def main():
    main = main_rows()
    abl = ablation()
    C.write_text(OUT / "final_numbers.json", json.dumps({"main": main, "ablation": abl}, indent=1) + "\n")
    lines = ["# Final numbers (test)", "",
             "Main table: full test. Ablation: test-0.1 (710), paired differences vs A4. Failed terminal results "
             "score 0. Point estimates.", "", "## Main table rows", ""]
    header = ["Row", "Metric"] + [f"{ds} (n)" for ds in C.DATASETS]
    rows = []
    for name, label in (("luna", "Full DKGI (Pi) + GPT-6-Luna"), ("astra", "Full DKGI (Pi) + GPT-6-Astra"),
                        ("codex_luna", "Full DKGI (Codex) + GPT-6-Luna"), ("luna_human", "Full DKGI (Pi) + GPT-6-Luna, human set")):
        for _, m in METRICS:
            rows.append([label, m] + [f"{f(main[name][m][ds][0])} ({main[name][m][ds][1]})" for ds in C.DATASETS])
    lines += [C.md_table(header, rows), "", "## Ablation (test-0.1, H@1)", ""]
    header = ["Arm", "amazon", "mag", "prime", "all", "Δ vs A4 amazon", "Δ mag", "Δ prime", "Δ all"]
    rows = []
    for arm, r in abl.items():
        d = r.get("delta_vs_A4", {})
        rows.append([arm] + [f(r[k][0]) for k in ("amazon", "mag", "prime", "all")] +
                    [diff(d.get(k)) for k in ("amazon", "mag", "prime", "all")])
    lines += [C.md_table(header, rows), "", "## Ablation (test-0.1, pooled R@20 and MRR)", ""]
    header = ["Arm"] + [x for _, m in METRICS[1:] for x in (f"{m} all", f"Δ {m} vs A4")]
    rows = [[arm] + [x for _, m in METRICS[1:] for x in (f(r[m]["all"][0]), diff(r[m]["delta_vs_A4"]))]
            for arm, r in abl.items()]
    lines += [C.md_table(header, rows), "",
              f"Astra Codex − Astra Pi (pooled H@1): {diff(abl['Astra-Codex']['delta_vs_Astra-Pi'])}.", "",
              "## Ablation (test-0.1, H@1 pooled over MAG and Prime)", "",
              "Mean over the MAG and Prime queries together (each query counts once); point estimates. Paper "
              "(ablation section): Full DKGI = A4, Graph tools + code = A5.", ""]
    rows = [[arm, f(r["mag+prime"][0]), r["mag+prime"][1]] for arm, r in abl.items()]
    lines += [C.md_table(["Arm", "H@1 MAG+Prime", "n"], rows), ""]
    C.write_text(OUT / "final_numbers.md", "\n".join(lines))
    tex = ["% generated by analysis_final/final_numbers.py: H@1 / R@20 / MRR per dataset"]
    for name, label in (("luna", r"\dkgi{} (Pi) & GPT-6-Luna"), ("codex_luna", r"\dkgi{} (Codex) & GPT-6-Luna"),
                        ("astra", r"\dkgi{} (Pi) & GPT-6-Astra")):
        cells = [f(main[name][m][ds][0]) for ds in C.DATASETS for _, m in METRICS]
        tex.append(label + " & " + " & ".join(cells) + r" \\")
    C.write_text(OUT / "tex/final_main_rows.tex", "\n".join(tex) + "\n")
    tex = [r"\begin{tabular}{lrrrrl}", r"\toprule", r"Arm & Amazon & MAG & Prime & All & $\Delta$ vs A4 (all) \\",
           r"\midrule"]
    for arm, r in abl.items():
        tex.append(f"{arm} & " + " & ".join(f(r[k][0]) for k in ("amazon", "mag", "prime", "all")) + " & " +
                   (diff(r["delta_vs_A4"]["all"]) if "delta_vs_A4" in r else "--") + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    C.write_text(OUT / "tex/final_ablation_table.tex", "\n".join(tex) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
