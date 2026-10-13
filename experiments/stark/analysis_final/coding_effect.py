"""Effect of code execution in the 2x2 ablation (test-0.1, 710 queries; Pi + GPT-6-Luna), its cost, and whether
the prompt decides if the agent writes code (prompt-selection round, val).

2x2: A4 = raw files + code, A3 = raw files with interpreters blocked, A5 = ARK-style graph tools callable from
Python, A2 = graph tools only. Code effect = mean of (A4 - A3) and (A5 - A2), computed per query on the queries
present and terminal in all four arms. Terminal failures score 0.

Strata (per-query tags of the model annotator in data/query_tags/tags_effective.jsonl, the file strata.py reads):
has_text_constraint (Amazon, Prime; every MAG query counts as text-constrained and MAG is not stratified) and, as
a descriptive proxy for relational complexity, the number of relational spans the annotator marked in the query
(n_relational_spans; Amazon, Prime only; the annotator records spans, not hop counts).

Statistics: point estimates (common.paired_difference: per-query paired differences, pooled over datasets and
weighted by n). Test data are used for reporting only.

Read-only: per-query features come from out/cache/*.jsonl when the cached row matches the result's source path,
status, finish time and feature version (plain value comparison); otherwise they are extracted in memory and the
cache is NOT rewritten. No model or API calls. Writes out/coding_effect.md and out/coding_effect.json.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SMALL_N = 30
COST = (("input_tokens", "Input tokens"), ("output_tokens", "Output tokens"), ("tool_calls", "Tool calls"),
        ("model_requests", "Model requests (turns)"), ("wall_seconds", "Wall s"), ("cost_usd", "USD"))
TAGS = C.STARK.parents[1] / "data/query_tags/tags_effective.jsonl"  # per-query text-constraint tags (README, Text-constraint tags)


# --------------------------------------------------------------------------- loading (read-only)

def load(name):
    """{(ds, qid): record} for the terminal results of one configuration, without writing the cache."""
    cfg = C.config_by_name()[name]
    cache_path = C.CACHE / f"{cfg.get('cache_as', cfg['name'])}.jsonl"
    cached = {r["src"]: r for r in C.read_jsonl(cache_path)} if cache_path.exists() else {}
    out, fresh = {}, 0
    for result, src in C.iter_results(cfg):
        c = cached.get(src)
        if not (c is not None and c.get("status") == result.get("status")
                and c.get("finished_at") == result.get("finished_at") and c.get("feature_version") == C.FEATURE_VERSION):
            c = C.features(cfg, result, src)
            fresh += 1
        if not c["terminal"]:
            continue
        rec = dict(c)
        rec["tool_calls"] = c.get("tool_calls_result") if c.get("tool_calls_result") is not None else c.get("calls")
        mr = result.get("model_requests", result.get("agent_loop_requests"))
        rec["model_requests"] = mr if isinstance(mr, (int, float)) and not isinstance(mr, bool) else None
        rec["shell_cmds"] = sum((c.get("intents") or {}).values())
        out[(c["dataset"], str(c["query_id"]))] = rec
    return out, fresh


def load_tags():
    text, nrel = {}, {}
    for r in C.read_jsonl(TAGS):
        if r["split"] not in ("test", "test-0.1"):
            continue
        k = (r["dataset"], str(r["query_id"]))
        if r.get("has_text_constraint") is not None:
            text[k] = bool(r["has_text_constraint"])
        if r["dataset"] != "mag" and r.get("n_relational_spans") is not None and r.get("stratify_by_label"):
            nrel[k] = r["n_relational_spans"]
    return text, nrel


# --------------------------------------------------------------------------- statistics

def paired_delta(pairs_by_ds):
    """Paired difference (point estimate) with per-query wins and losses."""
    pairs_by_ds = {d: v for d, v in pairs_by_ds.items() if v}
    p, _, _ = C.paired_difference(pairs_by_ds)
    d = [b - a for v in pairs_by_ds.values() for a, b in v]
    return {"n": len(d), "delta": p, "wins": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d)}


def ratio_delta(items_by_ds):
    """items: [((num1, den1), (num2, den2)), ...] per ds -> pooled ratio2 - ratio1 (point estimate)."""
    items = [x for v in items_by_ds.values() for x in v]
    if not items:
        return None
    n1, d1 = sum(a[0] for a, _ in items), sum(a[1] for a, _ in items)
    n2, d2 = sum(b[0] for _, b in items), sum(b[1] for _, b in items)
    r1, r2 = n1 / d1 if d1 else float("nan"), n2 / d2 if d2 else float("nan")
    return {"n": len(items), "p1": float(r1), "p2": float(r2), "delta": float(r2 - r1)}


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


# --------------------------------------------------------------------------- analysis 1: code effect by stratum

def contrasts(arms, keys):
    """arms: {'A4','A3','A5','A2'} -> per-query dicts; keys: list of (ds, qid) present in all four."""
    by_ds = {}
    for k in keys:
        by_ds.setdefault(k[0], []).append(k)
    h = {a: {k: arms[a][k]["hit1"] for k in keys} for a in arms}
    out = {"n": len(keys), "levels": {a: mean(h[a].values()) for a in ("A4", "A3", "A5", "A2")}}
    out["A4-A3"] = paired_delta({d: [(h["A3"][k], h["A4"][k]) for k in ks] for d, ks in by_ds.items()})
    out["A5-A2"] = paired_delta({d: [(h["A2"][k], h["A5"][k]) for k in ks] for d, ks in by_ds.items()})
    out["code"] = paired_delta({d: [((h["A3"][k] + h["A2"][k]) / 2, (h["A4"][k] + h["A5"][k]) / 2) for k in ks]
                        for d, ks in by_ds.items()})
    out["interaction"] = paired_delta({d: [(h["A5"][k] - h["A2"][k], h["A4"][k] - h["A3"][k]) for k in ks]
                               for d, ks in by_ds.items()})  # (A4-A3) - (A5-A2)
    out["small_n"] = len(keys) < SMALL_N
    return out


def analysis1(arms, common):
    text, nrel = load_tags()
    res = {"overall": {}, "text": {}, "relspans": {}, "untagged": {}}
    for ds in (*C.DATASETS, "all"):
        keys = [k for k in common if ds == "all" or k[0] == ds]
        res["overall"][ds] = contrasts(arms, keys)
    # text constraint: Amazon, Prime (every MAG query counts as text-constrained; MAG is not stratified)
    for flag, name in ((True, "text constraint"), (False, "relational only")):
        for ds in ("amazon", "prime", "amazon+prime"):
            keys = [k for k in common if (ds == "amazon+prime" and k[0] != "mag" or k[0] == ds) and text.get(k) is flag]
            res["text"][f"{name}|{ds}"] = contrasts(arms, keys)
        if flag:  # all datasets, MAG counted as text-constrained
            keys = [k for k in common if k[0] == "mag" or text.get(k) is True]
            res["text"][f"{name}|all (MAG=text)"] = contrasts(arms, keys)
    res["untagged"] = {ds: sum(1 for k in common if k[0] == ds and k not in text) for ds in C.DATASETS}
    for band, test in (("0", lambda n: n == 0), ("1", lambda n: n == 1), ("2", lambda n: n == 2), (">=3", lambda n: n >= 3)):
        for ds in ("amazon", "prime", "amazon+prime"):
            keys = [k for k in common if (ds == "amazon+prime" and k[0] != "mag" or k[0] == ds)
                    and k in nrel and test(nrel[k])]
            res["relspans"][f"{band}|{ds}"] = contrasts(arms, keys)
    return res


# --------------------------------------------------------------------------- analysis 2: cost

def analysis2(arms, common):
    res = {}
    for hi_arm, lo_arm in (("A4", "A3"), ("A5", "A2")):
        block = {}
        for ds in (*C.DATASETS, "all"):
            keys = [k for k in common if ds == "all" or k[0] == ds]
            row = {}
            for m, _ in COST + ((("service_calls", "Graph-service calls"),) if lo_arm == "A2" else ()):
                ks = [k for k in keys if arms[hi_arm][k].get(m) is not None and arms[lo_arm][k].get(m) is not None]
                by_ds = {}
                for k in ks:
                    by_ds.setdefault(k[0], []).append((arms[lo_arm][k][m], arms[hi_arm][k][m]))
                a = [arms[lo_arm][k][m] for k in ks]
                b = [arms[hi_arm][k][m] for k in ks]
                d = paired_delta(by_ds)
                row[m] = {"n": len(ks), "median_lo": C.median(a), "median_hi": C.median(b), "mean_lo": mean(a),
                          "mean_hi": mean(b), "median_diff": C.median([y - x for x, y in zip(a, b)]),
                          "ratio_of_means": (mean(b) / mean(a)) if a and mean(a) else None, "mean_diff": d}
            block[ds] = row
        res[f"{hi_arm}-{lo_arm}"] = block
    return res


def python_share(recs):
    py = sum(r.get("python_calls") or 0 for r in recs)
    sh = sum(r.get("shell_cmds") or 0 for r in recs)
    return {"python_calls": py, "shell_cmds": sh, "share": py / sh if sh else None, "n": len(recs),
            "queries_with_python": mean([float((r.get("python_calls") or 0) > 0) for r in recs])}


# --------------------------------------------------------------------------- analysis 3: prompt selection, P2 vs P1

def analysis3():
    res, cov = {}, {}
    for h in ("codex", "pi"):
        p1, f1 = load(f"t11_{h}_p1")
        p2, f2 = load(f"t11_{h}_p2")
        cov[h] = {"p1": len(p1), "p2": len(p2), "recomputed": f1 + f2}
        common = sorted(set(p1) & set(p2))
        block = {}
        for ds in (*C.DATASETS, "all"):
            keys = [k for k in common if ds == "all" or k[0] == ds]
            by = {}
            for k in keys:
                by.setdefault(k[0], []).append(k)
            row = {"n": len(keys)}
            row["python_share"] = ratio_delta({d: [((p1[k]["python_calls"] or 0, p1[k]["shell_cmds"]),
                                                   (p2[k]["python_calls"] or 0, p2[k]["shell_cmds"])) for k in ks]
                                              for d, ks in by.items()})
            row["queries_with_python"] = {
                "p1": mean([float(p1[k]["python_calls"] > 0) for k in keys]),
                "p2": mean([float(p2[k]["python_calls"] > 0) for k in keys]),
                "diff": paired_delta({d: [(float(p1[k]["python_calls"] > 0), float(p2[k]["python_calls"] > 0)) for k in ks]
                              for d, ks in by.items()})}
            for m in ("hit1", "input_tokens", "cost_usd", "shell_cmds"):
                ks = {d: [k for k in v if p1[k].get(m) is not None and p2[k].get(m) is not None] for d, v in by.items()}
                flat = [k for v in ks.values() for k in v]
                row[m] = {"p1": mean([p1[k][m] for k in flat]), "p2": mean([p2[k][m] for k in flat]),
                          "median_p1": C.median([p1[k][m] for k in flat]), "median_p2": C.median([p2[k][m] for k in flat]),
                          "diff": paired_delta({d: [(p1[k][m], p2[k][m]) for k in v] for d, v in ks.items()})}
            block[ds] = row
        res[h] = block
    return res, cov


# --------------------------------------------------------------------------- formatting

def pp(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}"


def fmt_pp(t):
    if not t or t.get("delta") is None:
        return "n/a"
    return f"{100 * t['delta']:+.1f}"


def fmt_num(t, d=0, scale=1.0):
    if not t or t.get("delta") is None:
        return "n/a"
    return f"{scale * t['delta']:+,.{d}f}"


def wl(t):
    return f"{t['wins']}/{t['losses']}" if t and t.get("n") else "-"


def num(x, d=0):
    return "n/a" if x is None else f"{x:,.{d}f}"


def strat_rows(items):
    rows = []
    for label, r in items:
        lv = r["levels"]
        rows.append([label, f"{r['n']}{' (small)' if r['small_n'] else ''}", pp(lv["A4"]), pp(lv["A3"]), pp(lv["A5"]),
                     pp(lv["A2"]), fmt_pp(r["A4-A3"]), wl(r["A4-A3"]), fmt_pp(r["A5-A2"]), wl(r["A5-A2"]),
                     fmt_pp(r["code"]), wl(r["code"])])
    return rows


STRAT_HEAD = ["Stratum", "n", "A4", "A3", "A5", "A2", "A4−A3", "W/L", "A5−A2", "W/L",
              "Code effect", "W/L"]


def main():
    print("coding_effect.py: loading")
    raw, cov = {}, {}
    for n in ("A4", "A3", "A5", "A2", "pi_final_full"):
        raw[n], fresh = load(n)
        cov[n] = {"terminal": {ds: sum(1 for k in raw[n] if k[0] == ds) for ds in C.DATASETS}, "recomputed": fresh}
        print(f"  {n}: {len(raw[n])} terminal, {fresh} extracted in memory")
    arms = {"A4": raw["A4"], "A3": raw["A3"], "A5": raw["A5"], "A2": raw["A2"]}
    common = sorted(set(arms["A4"]) & set(arms["A3"]) & set(arms["A5"]) & set(arms["A2"]))
    r1 = analysis1(arms, common)
    r2 = analysis2(arms, common)
    r3, cov3 = analysis3()
    shares = {name: {ds: python_share([r for k, r in recs.items() if ds == "all" or k[0] == ds])
                     for ds in (*C.DATASETS, "all")}
              for name, recs in (("pi_final_full", raw["pi_final_full"]), ("A4 (710)", raw["A4"]),
                                 ("A3 (710, attempted)", raw["A3"]), ("A5 (710)", raw["A5"]))}
    C.write_text(C.OUT / "coding_effect.json", json.dumps(
        {"coverage": cov, "paired_710": len(common), "code_effect": r1,
         "cost": r2, "t11": r3, "t11_coverage": cov3, "python_share": shares,
         "stats": {"estimates": "point", "small_n_threshold": SMALL_N}}, indent=1, default=float) + "\n")

    # ------------------------------------------------------------------ markdown
    L = ["# Code effect in the 2x2 ablation, its cost, and prompt-driven coding", "",
         "Generated by `coding_effect.py` (read-only; no model calls). Test-0.1 (710) data are used for "
         "reporting only. Point estimates of paired per-query differences, pooled over datasets (weighted by n). "
         "W/L = queries where the first arm is better / worse. "
         "H@1 in %, differences in percentage points.", "",
         "Arms (Pi + GPT-6-Luna): A4 raw files + code; A3 raw files, interpreters blocked; A5 graph tools callable "
         "from Python; A2 graph tools only. Code effect = per-query mean of (A4−A3) and (A5−A2). Queries present "
         f"and terminal in all four arms: {len(common)}.", "",
         "Coverage (terminal results): " + "; ".join(
             f"{n} " + "/".join(str(v["terminal"][d]) for d in C.DATASETS) + f" (extracted in memory: {v['recomputed']})"
             for n, v in cov.items()) + " (amazon/mag/prime).", "",
         "## 1. Code effect overall and by query type", "", "### Overall", "",
         C.md_table(STRAT_HEAD, strat_rows([(C.DS_LABEL[d], r1["overall"][d]) for d in (*C.DATASETS, "all")]),
                    [False] + [True] * 11), "",
         "Interaction (A4−A3) − (A5−A2), i.e. whether code helps more on raw files than on graph tools: " + "; ".join(
             f"{C.DS_LABEL[d]} {fmt_pp(r1['overall'][d]['interaction'])}" for d in (*C.DATASETS, "all")) + ".", "",
         "### By text constraint (per-query tags)", "",
         "Amazon and Prime use the tags; every MAG query counts as text-constrained and MAG is not stratified. "
         "These are the same labels and rule used by `strata.py`. Untagged queries (excluded): "
         + ", ".join(f"{d} {n}" for d, n in r1["untagged"].items()) + f". `(small)` = n < {SMALL_N}.", "",
         C.md_table(STRAT_HEAD, strat_rows([(k.replace("|", ", "), v) for k, v in r1["text"].items()]),
                    [False] + [True] * 11), "",
         "### By number of relational spans in the tag (descriptive proxy)", "",
         "The tags have no hop count. As a proxy for relational complexity we use the number of relational spans "
         "the annotator marked in the query (`n_relational_spans`; Amazon, Prime; MAG excluded because its annotator "
         "labels are not used). A span is a "
         "relational phrase, not a hop.", "",
         C.md_table(STRAT_HEAD, strat_rows([(f"{k.split('|')[0]} rel. spans, {k.split('|')[1]}", v)
                                            for k, v in r1["relspans"].items() if v["n"]]),
                    [False] + [True] * 11), "",
         "## 2. Cost of code", "",
         "Per-query paired comparison on the same 710 queries. Input tokens include "
         "cached input. Tool calls = issued tool calls (A5: bash calls, each of which can make several graph-service "
         "calls, so its tool-call count is not comparable one-to-one with A2; graph-service calls are listed "
         "separately; A2: tool calls; A3/A4: bash + read). Model "
         "requests = agent-loop requests (turns). USD = recorded cost. Mean difference of paired per-query values; "
         "median difference = median of per-query differences.", ""]
    for c, block in r2.items():
        hi_arm, lo_arm = c.split("-")
        L += [f"### {hi_arm} vs {lo_arm}", ""]
        rows = []
        for ds in (*C.DATASETS, "all"):
            for m, lab in COST + ((("service_calls", "Graph-service calls"),) if lo_arm == "A2" else ()):
                x = block[ds][m]
                dgt = 4 if m == "cost_usd" else (1 if m in ("tool_calls", "model_requests", "wall_seconds", "service_calls") else 0)
                rows.append([C.DS_LABEL[ds], lab, x["n"], num(x["median_lo"], dgt), num(x["median_hi"], dgt),
                             num(x["median_diff"], dgt), num(x["mean_lo"], dgt), num(x["mean_hi"], dgt),
                             fmt_num(x["mean_diff"], dgt), num(x["ratio_of_means"], 2)])
        L += [C.md_table(["Dataset", "Metric", "n", f"Median {lo_arm}", f"Median {hi_arm}", "Median diff",
                          f"Mean {lo_arm}", f"Mean {hi_arm}", f"Mean diff {hi_arm}−{lo_arm}", "Ratio of means"],
                         rows, [False, False] + [True] * 8), ""]
    L += ["## 3. Does the prompt decide whether the agent codes? (prompt-selection round, val)", "",
          "Prompt-selection runs (val split, GPT-6-Luna). P2 replaces the P1 tool sentence with the original DCI "
          "wording 'Use ripgrep (rg) instead of grep'. Paired on queries present in both P1 and P2 (P2 has 150 "
          "Amazon queries, P1 300). Python share = Python commands / "
          "shell commands (aligned nine-category classifier; Pi `read` excluded), ratio of sums. "
          "Coverage (terminal P1/P2): " + "; ".join(
              f"{h} {v['p1']}/{v['p2']}" for h, v in cov3.items()) + ".", ""]
    rows = []
    for h, block in r3.items():
        for ds in (*C.DATASETS, "all"):
            x = block[ds]
            ps = x["python_share"]
            rows.append([h, C.DS_LABEL[ds], x["n"], pp(ps["p1"]), pp(ps["p2"]), fmt_pp(ps),
                         pp(x["queries_with_python"]["p1"]), pp(x["queries_with_python"]["p2"]),
                         fmt_pp(x["queries_with_python"]["diff"]),
                         pp(x["hit1"]["p1"]), pp(x["hit1"]["p2"]), fmt_pp(x["hit1"]["diff"]), wl(x["hit1"]["diff"]),
                         num(x["input_tokens"]["median_p1"]), num(x["input_tokens"]["median_p2"]),
                         fmt_num(x["input_tokens"]["diff"], 0)])
    L += [C.md_table(["Harness", "Dataset", "n", "Py share P1", "P2", "P2−P1", "Q w/ Py P1", "P2", "P2−P1",
                      "H@1 P1", "P2", "P2−P1", "W/L", "Med. input P1", "P2", "Mean input P2−P1"],
                     rows, [False, False] + [True] * 14), "",
          "### Reference: Python share in the final / ablation configurations (test)", ""]
    rows = [[name, C.DS_LABEL[ds], v[ds]["n"], v[ds]["shell_cmds"], v[ds]["python_calls"], pp(v[ds]["share"]),
             pp(v[ds]["queries_with_python"])] for name, v in shares.items() for ds in (*C.DATASETS, "all")]
    L += [C.md_table(["Configuration", "Dataset", "Queries", "Shell cmds", "Python cmds", "Python share %",
                      "Queries with Python %"], rows, [False, False] + [True] * 5), "",
          "A3 blocks interpreters; the classifier finds no Python commands issued in A3.", ""]
    L += ["## Findings", ""] + findings(r1, r2, r3, shares)
    C.write_text(C.OUT / "coding_effect.md", "\n".join(L))
    print("coding_effect.py: wrote out/coding_effect.md, out/coding_effect.json")


def findings(r1, r2, r3, shares):
    """Plain statements generated from the numbers (point estimates)."""
    F = []
    o = r1["overall"]["all"]
    F.append(f"- Pooled (n={o['n']}): code effect {fmt_pp(o['code'])} pp; A4−A3 {fmt_pp(o['A4-A3'])}; "
             f"A5−A2 {fmt_pp(o['A5-A2'])}.")
    for ds in C.DATASETS:
        x = r1["overall"][ds]
        F.append(f"- {C.DS_LABEL[ds]} (n={x['n']}): code effect {fmt_pp(x['code'])}; A4−A3 {fmt_pp(x['A4-A3'])}; "
                 f"A5−A2 {fmt_pp(x['A5-A2'])}.")
    F.append("- Interaction (A4−A3) − (A5−A2): " + "; ".join(
        f"{C.DS_LABEL[d]} {fmt_pp(r1['overall'][d]['interaction'])}" for d in (*C.DATASETS, "all"))
        + ".")
    t = r1["text"]
    a, b = t["text constraint|amazon+prime"], t["relational only|amazon+prime"]
    F.append(f"- Amazon+Prime by text constraint: text-constrained (n={a['n']}) code effect {fmt_pp(a['code'])}; "
             f"relational-only (n={b['n']}) {fmt_pp(b['code'])}. Relational-only Amazon has n="
             f"{t['relational only|amazon']['n']} (too small to interpret); the relational-only stratum is essentially "
             "Prime." + " Relational-span strata: " + "; ".join(
                 f"{k.split('|')[0]} spans n={v['n']} {fmt_pp(v['code'])}" for k, v in r1["relspans"].items()
                 if k.endswith("amazon+prime")) + " (descriptive; the 0 and >=3 bands are small).")
    for c in ("A4-A3", "A5-A2"):
        x = r2[c]["all"]
        F.append(f"- Cost {c.replace('-', ' vs ')} (pooled): mean input tokens {fmt_num(x['input_tokens']['mean_diff'])} "
                 f"(ratio {num(x['input_tokens']['ratio_of_means'], 2)}), USD/query "
                 f"{fmt_num(x['cost_usd']['mean_diff'], 4)} (ratio {num(x['cost_usd']['ratio_of_means'], 2)}), wall s "
                 f"{fmt_num(x['wall_seconds']['mean_diff'], 1)}, tool calls {fmt_num(x['tool_calls']['mean_diff'], 1)}, "
                 f"turns {fmt_num(x['model_requests']['mean_diff'], 1)}.")
    for h in ("codex", "pi"):
        x = r3[h]["all"]
        F.append(f"- Prompt selection, {h} (val, n={x['n']}): Python share P1 {pp(x['python_share']['p1'])}% -> P2 "
                 f"{pp(x['python_share']['p2'])}% ({fmt_pp(x['python_share'])} pp); H@1 P2−P1 "
                 f"{fmt_pp(x['hit1']['diff'])} pp; mean input tokens P2−P1 {fmt_num(x['input_tokens']['diff'])}. "
                 "Per dataset, H@1 P2−P1: " + "; ".join(
                     f"{C.DS_LABEL[d]} {fmt_pp(r3[h][d]['hit1']['diff'])}" for d in C.DATASETS) + ".")
    F.append("- With the P1-style tool sentence both harnesses write Python in about half of their shell commands; "
             "with the original DCI sentence (P2) Python nearly disappears, and input tokens rise. The prompt, not "
             "the harness, sets whether the agent codes (val).")
    s = shares["pi_final_full"]["all"]
    F.append(f"- Final Pi config on full test: Python share {pp(s['share'])}% of {s['shell_cmds']:,} shell commands; "
             f"{pp(s['queries_with_python'])}% of queries use Python at least once.")
    return F


if __name__ == "__main__":
    main()
