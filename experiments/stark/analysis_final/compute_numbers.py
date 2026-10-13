"""Recompute the appendix numbers from the experiment files and the other analysis outputs (read-only).

Usage (from anywhere):
    python3 compute_numbers.py            # all sections except the symmetrised edge counts
    python3 compute_numbers.py --edges    # also count distinct ordered edge pairs (about 2 minutes, needs pandas)

The script only reads experiment files and prints plain text. It makes no model or API calls,
writes nothing. Failed runs score zero, as in the paper.
"""

import argparse
import csv
import json
from pathlib import Path

STARK = Path(__file__).resolve().parents[1]
DATASETS = ("amazon", "mag", "prime")


def pct(x):
    return f"{100 * x:.1f}"


def corpus():
    print("== Corpus export (corpus.tex)")
    total = {"nodes": 0, "edges": 0, "bytes": 0}
    for ds in DATASETS:
        root = STARK / "corpus" / ds
        schema = json.loads((root / "schema.json").read_text())
        prepared = json.loads((STARK / "prepared" / f"{ds}.json").read_text())
        node_files = sorted((root / "nodes").glob("*.jsonl"))
        edge_files = sorted((root / "edges").glob("*.tsv"))
        node_bytes = sum(p.stat().st_size for p in node_files)
        edge_bytes = sum(p.stat().st_size for p in edge_files)
        with (root / "candidate_ids.txt").open() as f:
            candidates = sum(1 for _ in f)
        files = node_files + edge_files + [root / n for n in ("schema.json", "DATA_FORMAT.txt", "candidate_ids.txt")]
        all_bytes = sum(p.stat().st_size for p in files)
        edges = sum(schema["stored_edges_by_type"].values())
        assert all_bytes == prepared["corpus_bytes"], (ds, all_bytes, prepared["corpus_bytes"])
        assert sum(prepared["node_type_counts"].values()) == schema["nodes"]
        print(f"{ds}: nodes {schema['nodes']:,}; candidates {schema['candidate_count']:,} "
              f"(candidate_ids.txt lines {candidates:,}; types {schema['candidate_types']})")
        print(f"  node types: {prepared['node_type_counts']}")
        print(f"  relations ({len(schema['relation_types'])}): " + ", ".join(
            f"{schema['relation_types'][k]}={v:,}" for k, v in schema["stored_edges_by_type"].items()))
        print(f"  stored edge rows {edges:,}; node shards {len(node_files)} ({node_bytes:,} B); "
              f"edge files {len(edge_files)} ({edge_bytes:,} B); export total {all_bytes:,} B; "
              f"largest node shard {max(p.stat().st_size for p in node_files):,} B; "
              f"node files per node {node_bytes / schema['nodes']:,.0f} B ({node_bytes / schema['nodes'] / 1000:.1f} KB)")
        print(f"  queries per split: {prepared['qa_counts']}; "
              f"queries with gold outside candidates: {prepared['gold_outside_candidates_queries']}")
        total["nodes"] += schema["nodes"]
        total["edges"] += edges
        total["bytes"] += all_bytes
    print(f"all: nodes {total['nodes']:,}; stored edge rows {total['edges']:,}; bytes {total['bytes']:,}")
    print("DATA_FORMAT.txt identical across datasets:",
          len({(STARK / "corpus" / ds / "DATA_FORMAT.txt").read_text() for ds in DATASETS}) == 1)


def symmetrised_edges():
    import numpy as np
    import pandas as pd
    print("== Distinct ordered pairs after adding reverse rows (corpus.tex footnote on STaRK counts)")
    for ds in DATASETS:
        parts = [pd.read_csv(p, sep="\t", dtype=np.int64).to_numpy()
                 for p in sorted((STARK / "corpus" / ds / "edges").glob("*.tsv"))]
        e = np.concatenate(parts)
        key = e[:, 0] * (1 << 21) + e[:, 1]
        rkey = e[:, 1] * (1 << 21) + e[:, 0]
        distinct = np.unique(key)
        # Distinct ordered pairs (a, b) whose reverse (b, a) is also stored (self-loops count once).
        both = int(np.isin(np.unique(rkey), distinct, assume_unique=True).sum())
        loops = int((e[:, 0] == e[:, 1]).sum())
        print(f"{ds}: rows {len(e):,}; distinct ordered pairs {len(distinct):,} "
              f"(duplicate rows {len(e) - len(distinct):,}; self-loop rows {loops:,}); "
              f"distinct pairs whose reverse is also stored {both:,}; "
              f"distinct ordered pairs with reverses added {len(np.unique(np.concatenate([key, rkey]))):,}")


def costs():
    print("== Cost and effort per configuration (harness.tex), from analysis_final/out/behavior_by_config.csv")
    rows = {(r["config"], r["dataset"].lower()): r for r in
            csv.DictReader((STARK / "analysis_final/out/behavior_by_config.csv").open())}
    # Table order: full test (Full DKGI Luna, Full DKGI (Codex) Luna, Full DKGI Astra), then the test subset with
    # GPT-6-Luna, then the test subset with GPT-6-Astra (Full DKGI: the 710 subset queries of the full-test run
    # astra_pi_full; Full DKGI (Codex): its own test-subset run). One program is not in behavior_by_config.csv.
    for cfg in ("pi_final_full", "codex_luna_full", "astra_pi_full", "A4", "A5", "A2", "A3", "A1", "A6", "H",
                "astra_pi_710", "astra_codex_710"):
        r = rows[(cfg, "all")]
        n = int(r["n"])
        usd = float(r["known_cost_usd"])
        svc = f"; service calls/q {int(r['service_calls']) / n:.1f}" if r["service_calls"] not in ("", "0") else ""
        print(f"{cfg:14s} n={n} USD/q {usd / n:.4f} (total {usd:.2f}); median tool calls {float(r['median_calls']):g}; "
              f"median input tok {float(r['median_input_tokens']):,.0f}; median output tok "
              f"{float(r['median_output_tokens']):,.0f}; median s {float(r['median_wall_seconds']):.0f}{svc}")


def human():
    print("== Human-generated queries (human.tex), from analysis_final/out/final_numbers.json")
    main = json.loads((STARK / "analysis_final/out/final_numbers.json").read_text())["main"]
    for row in ("luna_human", "luna"):
        for metric in ("H@1", "R@20", "MRR"):
            cells = main[row][metric]
            n = sum(cells[d][1] for d in DATASETS)
            pooled = sum(cells[d][0] * cells[d][1] for d in DATASETS) / n
            print(f"{row:10s} {metric:4s} " + " / ".join(f"{cells[d][0]:.1f} (n={cells[d][1]})" for d in DATASETS)
                  + f"; pooled {pooled:.1f} (n={n})")


def load_rows(path):
    return [json.loads(line) for line in path.open()]


def metric(row, key):
    if row.get("status") != "scored":
        return 0.0
    return float((row.get("metrics") or {}).get(key, 0.0))


def table(rows, groups, subsets):
    keys = (("H@1", "hit_at_1"), ("R@20", "recall_at_20"), ("MRR", "mrr_at_100"))
    for group in groups:
        for label, key in keys:
            cells = []
            for name, pick in subsets:
                vals = [metric(r, key) for r in rows if r["group"] == group and pick(r)]
                cells.append(f"{name} {pct(sum(vals) / len(vals))} (n={len(vals)})" if vals else f"{name} -")
            print(f"{group:12s} {label:4s} " + "; ".join(cells))


def validation(base_dir, confirmatory):
    print(f"== Prompt-selection round (validation.tex), {base_dir}/results_raw.jsonl")
    rows = load_rows(base_dir / "results_raw.jsonl")
    base = {(r["dataset"], str(r["query_id"])) for r in json.loads((base_dir / "selection.json").read_text())}
    sup = {(r["dataset"], str(r["query_id"])) for r in json.loads((base_dir / "supplement_amazon.json").read_text())}
    assert not base & sup
    in_base = lambda r: (r["dataset"], str(r["query_id"])) in base
    subsets = [("amazon300", lambda r: r["dataset"] == "amazon"),
               ("amazon150", lambda r: r["dataset"] == "amazon" and in_base(r)),
               ("mag", lambda r: r["dataset"] == "mag"), ("prime", lambda r: r["dataset"] == "prime"),
               ("all450", in_base)]
    groups = ["pi_p0", "pi_p1", "pi_p2", "codex_p0", "codex_p1", "codex_p2"]
    table(rows, groups, subsets)
    print("failed runs:", sorted((r["group"], r["dataset"], r["query_id"]) for r in rows if r.get("status") != "scored"
                                 and r["group"] in groups))
    # Paired H@1 differences (plain mean differences on shared queries).
    h1 = {(r["group"], r["dataset"], str(r["query_id"])): metric(r, "hit_at_1") for r in rows}
    for h in ("pi", "codex"):
        for a, b, scope in (("p1", "p0", "all"), ("p2", "p1", "base")):
            parts, pooled = [], []
            for ds in DATASETS:
                qs = [q for (g, d, q) in h1 if g == f"{h}_{b}" and d == ds and (scope == "all" or (d, q) in base)]
                qs = [q for q in qs if (f"{h}_{a}", ds, q) in h1]
                diffs = [h1[(f"{h}_{a}", ds, q)] - h1[(f"{h}_{b}", ds, q)] for q in qs]
                pooled += diffs
                parts.append(f"{ds} {100 * sum(diffs) / len(diffs):+.2f} (n={len(diffs)})")
            # Pooled over every query of the comparison: 600 for p1-p0 (Amazon 300), 450 for p2-p1.
            parts.append(f"pooled {100 * sum(pooled) / len(pooled):+.2f} (n={len(pooled)})")
            print(f"{h} {a}-{b}: " + "; ".join(parts))
    p2 = [h1[(f"{h}_p2", d, q)] - h1[(f"{h}_p1", d, q)] for (g, d, q) in h1 for h in ("pi", "codex")
          if g == f"{h}_p2"]
    print(f"p2-p1 pooled over both harnesses: {100 * sum(p2) / len(p2):+.2f} (n={len(p2)} pairs)")

    print(f"== Confirmatory round (validation.tex), {confirmatory}")
    rows = load_rows(confirmatory)
    subsets = [(ds, (lambda d: lambda r: r["dataset"] == d)(ds)) for ds in DATASETS] + [("all", lambda r: True)]
    table(rows, ["pi_p0", "pi_p1r1", "codex_p0", "codex_p1r1"], subsets)
    print("failed runs:", sum(r.get("status") != "scored" for r in rows))
    h1 = {(r["group"], r["dataset"], str(r["query_id"])): metric(r, "hit_at_1") for r in rows}
    for h in ("pi", "codex"):
        parts, pooled = [], []
        for ds in DATASETS:
            qs = [q for (g, d, q) in h1 if g == f"{h}_p0" and d == ds and (f"{h}_p1r1", ds, q) in h1]
            diffs = [h1[(f"{h}_p1r1", ds, q)] - h1[(f"{h}_p0", ds, q)] for q in qs]
            pooled += diffs
            parts.append(f"{ds} {100 * sum(diffs) / len(diffs):+.2f} (n={len(diffs)})")
        parts.append(f"pooled {100 * sum(pooled) / len(pooled):+.2f} (n={len(pooled)})")
        print(f"{h} p1r1-p0: " + "; ".join(parts))
    for g in ("pi_p0", "pi_p1r1", "codex_p0", "codex_p1r1"):
        ids = sorted(r["returned_id_count"] for r in rows if r["group"] == g)
        print(f"{g}: median returned ids {ids[len(ids) // 2]}")


def probes():
    print("== Closed-book probes (probes.tex), contamination/probe_strata.json")
    s = json.loads((STARK / "contamination/probe_strata.json").read_text())
    for key, v in s["a0_rates"].items():
        if key.startswith("full/"):
            print(f"{key}: {v['k']}/{v['n']} = {v['h1'][0]:.2f}%; hit@20 {v['h20']:.2f}%")
    for key in ("same_name_h1/full/A4/mag", "same_name_h1/full/A4/prime",
                "same_name_h20/full/A4/mag", "same_name_h20/full/A4/prime",
                "exact_h1/full/A4/mag", "exact_h1/full/A4/prime", "fuzzy_h1/full/A4/mag", "fuzzy_h1/full/A4/prime"):
        v = s["strata"].get(key)
        if v:
            known = f"{v['known'][0]:.1f}" if v["known"] else "n/a"
            print(f"{key}: probe-correct n={v['n_known']} H@1 {known}; probe-wrong n={v['n_unknown']} H@1 {v['unknown'][0]:.1f}")
    for arm in ("A4", "A5", "A2", "A3", "A1", "A6"):
        cells = []
        for ds in ("mag", "prime"):
            v = s["strata"][f"same_name_h1/test-0.1/{arm}/{ds}"]
            cells.append(f"{ds} probe-wrong n={v['n_unknown']} H@1 {v['unknown'][0]:.1f}; "
                         f"probe-correct n={v['n_known']} H@1 {v['known'][0]:.1f}")
        print(f"test subset {arm}: " + " | ".join(cells))
    print("not scored (count as wrong):", {k: v["not_scored"] for k, v in s["quality"].items() if v["not_scored"]})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edges", action="store_true")
    ap.add_argument("--selection-round", type=Path, default=STARK / "prompt_val",
                    help="directory with results_raw.jsonl, selection.json and supplement_amazon.json of the prompt-selection round")
    ap.add_argument("--confirmatory-round", type=Path, default=STARK / "prompt_val_r1/results_raw.jsonl",
                    help="results_raw.jsonl of the confirmatory round")
    args = ap.parse_args()
    corpus()
    if args.edges:
        symmetrised_edges()
    costs()
    human()
    if (args.selection_round / "results_raw.jsonl").exists() and args.confirmatory_round.exists():
        validation(args.selection_round, args.confirmatory_round)
    else:
        print("== Validation rounds: result files not found; skipped")
    probes()


if __name__ == "__main__":
    main()
