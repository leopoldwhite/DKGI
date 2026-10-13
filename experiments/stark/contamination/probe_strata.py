"""Closed-book probe strata (offline, read-only): H@1 on the questions the probe answered or missed.

Inputs (read only, never modified):
  contamination/runs/A0-ID|A0-Name/<ds>/test/<qid>/result.json   closed-book probes (gpt-6-luna)
  contamination/index/<ds>.json                                   exact name index (all homonyms, Prime aliases)
  private/<ds>/test.jsonl                                         gold labels
  public/<ds>/test-0.1.jsonl                                      710-question subset
  pi_full_test/runs/pi_final/<ds>/test/<qid>/result.json          A4 = Pi final, full test
  trials/<arm>/<ds>/test/<qid>/result.json                        A1 A2 A3 A5 A6, test-0.1

Outputs (this folder only): probe_strata_cache.json (per-question rows; reused unless --refresh),
probe_strata.json, probe_strata.md.

Name-to-node mappings for A0-Name (applied to the first 20 names of each answer):
  exact      a name counts only if its normalized form maps to exactly one node, and that node is gold
  same_name  official rule (contamination/score_names.py): all nodes sharing the normalized name
             (homonyms, and Prime aliases) are expanded; hit if any is gold
  fuzzy      same_name, or difflib ratio >= 0.9 between the normalized name and any normalized
             name of a gold node
A0-ID is a node-ID probe; mapping does not apply, so its rate is the same under all three.

Point estimates: H@1 per stratum, arm minus A4 on the same questions, known minus unknown.
Failures count as 0 for every config. No model/API calls. Stdlib only.
"""

import argparse
import difflib
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
STARK = HERE.parent
CONT = STARK / "contamination"
DS = ("amazon", "mag", "prime")
NAME_DS = ("mag", "prime")
ARMS = ("A1", "A2", "A3", "A5", "A6")
MAPPINGS = ("exact", "same_name", "fuzzy")
FUZZY = 0.9


def norm(value):  # same rule as contamination/name_index.py
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(c if c.isalnum() else " " for c in value).split())


def jl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def scored_h1(r):
    if not r or r.get("status") != "scored":
        return 0.0
    return float((r.get("metrics") or {}).get("hit_at_1") or 0.0)


def status(r):
    return "missing" if r is None else str(r.get("status"))


# ---------------------------------------------------------------- extraction

def extract():
    gold = {d: {str(r["query_id"]): r["answer_ids"] for r in jl(STARK / f"private/{d}/test.jsonl")} for d in DS}
    t01 = {d: sorted(str(r["query_id"]) for r in jl(STARK / f"public/{d}/test-0.1.jsonl")) for d in DS}
    cache = {"extracted_at": datetime.now(timezone.utc).isoformat(), "t01": t01,
             "a0_id": {}, "a0_name": {}, "a4": {}, "arms": {a: {} for a in ARMS}}
    for d in DS:
        rows = {}
        for q in gold[d]:
            r = read(CONT / f"runs/A0-ID/{d}/test/{q}/result.json")
            m = (r or {}).get("metrics") or {}
            ok = r is not None and r.get("status") == "scored"
            rows[q] = {"status": status(r), "h1": float(m.get("hit_at_1") or 0.0) if ok else 0.0,
                       "h20": float(m.get("hit_at_20") or 0.0) if ok else 0.0}
        cache["a0_id"][d] = rows
        print("A0-ID", d, len(rows), file=sys.stderr, flush=True)
    for d in NAME_DS:
        raw = {}
        for q in gold[d]:
            r = read(CONT / f"runs/A0-Name/{d}/test/{q}/result.json")
            names = []
            if r is not None and r.get("status") == "scored":
                try:
                    names = [x for x in json.loads(r["prediction"])["names"][:20] if isinstance(x, str)]
                except Exception:
                    names = []
            raw[q] = (status(r), names, (r or {}).get("metrics") or {})
        index = json.loads((CONT / f"index/{d}.json").read_text())["names"]
        need = {i for v in gold[d].values() for i in v}
        gold_names = {}
        for nm, ids in index.items():
            for i in ids:
                if i in need:
                    gold_names.setdefault(i, []).append(nm)
        rows = {}
        for q, (st, names, m) in raw.items():
            g = set(gold[d][q])
            gn = sorted({x for i in g for x in gold_names.get(i, [])})
            vec = {k: [] for k in MAPPINGS}
            for name in names:
                key = norm(name)
                ids = set(index.get(key, [])) if key else set()
                same = bool(ids & g)
                exact = len(ids) == 1 and same
                fuzzy = same or (bool(key) and any(difflib.SequenceMatcher(None, key, x).ratio() >= FUZZY for x in gn))
                vec["exact"].append(exact)
                vec["same_name"].append(same)
                vec["fuzzy"].append(fuzzy)
            rows[q] = {"status": st, "n_names": len(names),
                       "file_h1": float(m.get("hit_at_1") or 0.0) if st == "scored" else 0.0,
                       "file_h20": float(m.get("hit_at_20") or 0.0) if st == "scored" else 0.0}
            for k in MAPPINGS:
                rows[q][f"{k}_h1"] = float(bool(vec[k][:1] and vec[k][0]))
                rows[q][f"{k}_h20"] = float(any(vec[k]))
        del index
        cache["a0_name"][d] = rows
        print("A0-Name", d, len(rows), file=sys.stderr, flush=True)
    for d in DS:
        rows = {}
        for q in gold[d]:
            r = read(STARK / f"pi_full_test/runs/pi_final/{d}/test/{q}/result.json")
            rows[q] = {"status": status(r), "terminal": (r or {}).get("terminal"), "h1": scored_h1(r)}
        cache["a4"][d] = rows
        print("A4", d, len(rows), file=sys.stderr, flush=True)
    for a in ARMS:
        for d in DS:
            rows = {}
            for q in t01[d]:
                r = read(STARK / f"trials/{a}/{d}/test/{q}/result.json")
                rows[q] = {"status": status(r), "terminal": (r or {}).get("terminal"), "h1": scored_h1(r)}
            cache["arms"][a][d] = rows
        print("arm", a, file=sys.stderr, flush=True)
    (HERE / "probe_strata_cache.json").write_text(json.dumps(cache))
    return cache


# ---------------------------------------------------------------- statistics

def mean(x):
    return sum(x) / len(x) if x else float("nan")


def rate(vals):
    """H@1 in percent (point estimate) as a one-element list. vals: list of 0/1."""
    return [100 * mean(vals)] if vals else None


def paired_diff(x, y):
    """mean(x) - mean(y) on the same questions, in points."""
    return [100 * (mean(x) - mean(y))] if x else None


def gap(a, b):
    """mean(a) - mean(b) of two strata, in points."""
    return [100 * (mean(a) - mean(b))] if a and b else None


# ---------------------------------------------------------------- analysis

def known_flag(cache, d, q, crit):
    """crit: (probe-mapping, cutoff). Amazon has only A0-ID."""
    mapping, cut = crit
    if d == "amazon":
        return cache["a0_id"][d][q]["h1" if cut == "h1" else "h20"] == 1.0
    return cache["a0_name"][d][q][f"{mapping}_{cut}"] == 1.0


def analyze(cache):
    t01 = cache["t01"]
    full = {d: sorted(cache["a4"][d]) for d in DS}
    res = {"extracted_at": cache["extracted_at"], "generated_at": datetime.now(timezone.utc).isoformat(),
           "quality": {}, "a0_rates": {}, "strata": {}, "cross": {}, "sensitivity": {}}

    # quality / completeness
    q = res["quality"]
    for d in DS:
        q[f"A0-ID/{d}"] = {"n": len(cache["a0_id"][d]), "not_scored": sorted(k for k, v in cache["a0_id"][d].items() if v["status"] != "scored")}
        q[f"A4/{d}"] = {"n": len(cache["a4"][d]), "not_scored": sorted(k for k, v in cache["a4"][d].items() if v["status"] != "scored")}
        for a in ARMS:
            q[f"{a}/{d}"] = {"n": len(cache["arms"][a][d]), "not_scored": sorted(k for k, v in cache["arms"][a][d].items() if v["status"] != "scored")}
    for d in NAME_DS:
        rows = cache["a0_name"][d]
        q[f"A0-Name/{d}"] = {"n": len(rows), "not_scored": sorted(k for k, v in rows.items() if v["status"] != "scored"),
                             "same_name_vs_official_h1_mismatch": sorted(k for k, v in rows.items() if v["same_name_h1"] != v["file_h1"]),
                             "same_name_vs_official_h20_mismatch": sorted(k for k, v in rows.items() if v["same_name_h20"] != v["file_h20"])}

    # A0 hit rates under each mapping
    for scope, qs in (("full", full), ("test-0.1", t01)):
        for d in DS:
            rows = cache["a0_id"][d]
            k = int(sum(rows[x]["h1"] for x in qs[d]))
            v = [rows[x]["h1"] for x in qs[d]]
            ent = {"n": len(v), "k": k, "h1": rate(v),
                   "h20": 100 * mean([rows[x]["h20"] for x in qs[d]])}
            res["a0_rates"][f"{scope}/A0-ID/{d}"] = ent
        for d in NAME_DS:
            rows = cache["a0_name"][d]
            for m in MAPPINGS:
                v = [rows[x][f"{m}_h1"] for x in qs[d]]
                res["a0_rates"][f"{scope}/A0-Name/{d}/{m}"] = {
                    "n": len(v), "k": int(sum(v)), "h1": rate(v),
                    "h20": 100 * mean([rows[x][f"{m}_h20"] for x in qs[d]]),
                    "k20": int(sum(rows[x][f"{m}_h20"] for x in qs[d]))}

    # stratification
    def config_vals(cfg, d):
        return cache["a4"][d] if cfg == "A4" else cache["arms"][cfg][d]

    crits = {f"{m}_{c}": (m, c) for m in MAPPINGS for c in ("h1", "h20")}
    for cname, crit in crits.items():
        for scope, qs, cfgs in (("full", full, ("A4",)), ("test-0.1", t01, ("A4",) + ARMS)):
            for d in DS:
                known = [x for x in qs[d] if known_flag(cache, d, x, crit)]
                unknown = [x for x in qs[d] if not known_flag(cache, d, x, crit)]
                a4 = cache["a4"][d]
                for cfg in cfgs:
                    rows = config_vals(cfg, d)
                    kv = [rows[x]["h1"] for x in known]
                    uv = [rows[x]["h1"] for x in unknown]
                    ent = {"n_known": len(kv), "n_unknown": len(uv), "known": rate(kv), "unknown": rate(uv),
                           "all": rate(kv + uv), "gap_known_minus_unknown": gap(kv, uv)}
                    if cfg != "A4":
                        ent["minus_A4_known"] = paired_diff(kv, [a4[x]["h1"] for x in known])
                        ent["minus_A4_unknown"] = paired_diff(uv, [a4[x]["h1"] for x in unknown])
                        ent["minus_A4_all"] = paired_diff(kv + uv, [a4[x]["h1"] for x in known + unknown])
                    res["strata"][f"{cname}/{scope}/{cfg}/{d}"] = ent
                    # 2x2: A0 correct/incorrect x config correct/incorrect
                    res["cross"][f"{cname}/{scope}/{cfg}/{d}"] = {
                        "A0c_Cc": sum(1 for x in known if rows[x]["h1"] == 1.0),
                        "A0c_Cw": sum(1 for x in known if rows[x]["h1"] != 1.0),
                        "A0w_Cc": sum(1 for x in unknown if rows[x]["h1"] == 1.0),
                        "A0w_Cw": sum(1 for x in unknown if rows[x]["h1"] != 1.0)}
    return res


# ---------------------------------------------------------------- report

def f(est, d=2):
    if est is None:
        return "n/a"
    return f"{est[0]:.{d}f}"


def p(est, d=1):
    return "n/a" if est is None else f"{est[0]:.{d}f}"


def report(res):
    L = []
    w = L.append
    w("# Closed-book probe strata")
    w("")
    w(f"Generated {res['generated_at']} by `probe_strata.py` from per-question data extracted {res['extracted_at']} "
      "(`probe_strata_cache.json`). Offline; no model calls.")
    w("")
    w("- **Configs.** A0-ID / A0-Name = closed-book probes (gpt-6-luna, `contamination/runs`). "
      "A4 = Pi final (gpt-6-luna, H6, P1-R1), `pi_full_test/runs/pi_final`, on the same questions. "
      "A1, A2, A3, A5, A6 = `trials/<arm>`, test-0.1 (710 questions); A4 on test-0.1 is the pi_final subset.")
    w("- **Stratum.** \"A0 correct\" = A0-Name H@1 = 1 on MAG and Prime. Amazon has no A0-Name by design, "
      "so Amazon is stratified by A0-ID and every Amazon question is in the A0-incorrect stratum. Primary criterion: same-name mapping (the official rule), H@1.")
    w("- **Statistics.** Point estimates; \"arm - A4\" compares the same questions. Failures count as 0.")
    w("- **Mappings (A0-Name).** *exact* = the normalized name maps to exactly one node and it is gold (homonyms/aliases shared by several nodes do not count); "
      "*same-name* = official `score_names.py` rule, every node sharing the normalized name (and Prime aliases) is expanded; "
      "*fuzzy* = same-name, or difflib ratio >= 0.9 between the normalized name and any normalized name of a gold node. "
      "A0-ID is an ID probe, so its rate does not depend on the mapping.")
    w("")

    w("## 0. Data completeness")
    w("")
    w("| Group | n | not scored (counted as 0) |")
    w("|---|---:|---|")
    for k, v in res["quality"].items():
        ns = v["not_scored"]
        w(f"| {k} | {v['n']} | {len(ns)}{(' (' + ', '.join(ns[:6]) + ')') if ns else ''} |")
    for d in NAME_DS:
        v = res["quality"][f"A0-Name/{d}"]
        w("")
        w(f"Check: recomputed same-name H@1 / hit@20 vs the official per-question metrics on {d}: "
          f"{len(v['same_name_vs_official_h1_mismatch'])} / {len(v['same_name_vs_official_h20_mismatch'])} mismatches.")
    w("")

    w("## 1. Closed-book hit rates by mapping (H@1 %; hit@20 %)")
    w("")
    for scope in ("full", "test-0.1"):
        w(f"### {scope}")
        w("")
        w("| Probe | Dataset | n | Mapping | H@1 hits | H@1 | hit@20 | vs same-name (H@1 pts) |")
        w("|---|---|---:|---|---:|---|---:|---:|")
        for d in DS:
            e = res["a0_rates"][f"{scope}/A0-ID/{d}"]
            w(f"| A0-ID | {d} | {e['n']} | (IDs; mapping n/a) | {e['k']} | {f(e['h1'])} | {e['h20']:.2f} | - |")
        for d in NAME_DS:
            base = res["a0_rates"][f"{scope}/A0-Name/{d}/same_name"]["h1"][0]
            for m in MAPPINGS:
                e = res["a0_rates"][f"{scope}/A0-Name/{d}/{m}"]
                w(f"| A0-Name | {d} | {e['n']} | {m} | {e['k']} | {f(e['h1'])} | {e['h20']:.2f} | {e['h1'][0] - base:+.2f} |")
        w("")
    w("Reading: the official same-name rule is neither a lower nor an upper bound. Relative to strict one-node exact matching it adds the homonym/alias hits; "
      "relative to fuzzy matching it misses near-verbatim titles. Do not call it an \"upper bound\" of parametric knowledge.")
    w("")

    def strata_table(cname, scope, cfgs, title):
        w(title)
        w("")
        w("| Config | Dataset | A0-correct n | A0-incorrect n | H@1 A0-correct | H@1 A0-incorrect | H@1 all | known - unknown | arm - A4, A0-incorrect | arm - A4, A0-correct |")
        w("|---|---|---:|---:|---|---|---:|---|---|---|")
        for cfg in cfgs:
            for d in DS:
                e = res["strata"].get(f"{cname}/{scope}/{cfg}/{d}")
                if e is None:
                    continue
                w(f"| {cfg} | {d} | {e['n_known']} | {e['n_unknown']} | {f(e['known'], 1)} | {f(e['unknown'], 1)} | {p(e['all'], 2)} | "
                  f"{f(e['gap_known_minus_unknown'], 1)} | {f(e.get('minus_A4_unknown'), 1) if cfg != 'A4' else '-'} | "
                  f"{f(e.get('minus_A4_known'), 1) if cfg != 'A4' else '-'} |")
        w("")

    w("## 2. Stratified H@1 by A0 outcome (primary: same-name mapping, A0 H@1)")
    w("")
    strata_table("same_name_h1", "full", ("A4",), "### Full test, A4")
    strata_table("same_name_h1", "test-0.1", ("A4",) + ARMS, "### test-0.1, A4 and ablation arms")
    w("On Amazon every question is A0-incorrect (A0-ID has 0 H@1 hits), so the A0-incorrect column equals the overall H@1. "
      "On MAG test-0.1 the A0-correct stratum has only a handful of questions; do not interpret it.")
    w("")

    w("## 3. 2x2 cross tables: A0-Name H@1 (same-name) x config H@1 (question counts)")
    w("")
    w("Cells: A0 correct & config correct / A0 correct & config wrong / A0 wrong & config correct / A0 wrong & config wrong. Amazon uses A0-ID.")
    w("")
    for scope, cfgs in (("full", ("A4",)), ("test-0.1", ("A4",) + ARMS)):
        w(f"### {scope}")
        w("")
        w("| Config | Dataset | A0 ok, C ok | A0 ok, C wrong | A0 wrong, C ok | A0 wrong, C wrong | total |")
        w("|---|---|---:|---:|---:|---:|---:|")
        for cfg in cfgs:
            for d in DS:
                c = res["cross"][f"same_name_h1/{scope}/{cfg}/{d}"]
                w(f"| {cfg} | {d} | {c['A0c_Cc']} | {c['A0c_Cw']} | {c['A0w_Cc']} | {c['A0w_Cw']} | {sum(c.values())} |")
        w("")
    w("### Full test, A4, by mapping and cutoff")
    w("")
    w("| Criterion | Dataset | A0 ok, A4 ok | A0 ok, A4 wrong | A0 wrong, A4 ok | A0 wrong, A4 wrong |")
    w("|---|---|---:|---:|---:|---:|")
    for m in MAPPINGS:
        for cut in ("h1", "h20"):
            for d in NAME_DS:
                c = res["cross"][f"{m}_{cut}/full/A4/{d}"]
                w(f"| {m}, A0 {'H@1' if cut == 'h1' else 'hit@20'} | {d} | {c['A0c_Cc']} | {c['A0c_Cw']} | {c['A0w_Cc']} | {c['A0w_Cw']} |")
    w("")

    w("## 4. Sensitivity: H@1 on the A0-incorrect stratum under each criterion")
    w("")
    w("Entries: A0-incorrect n, H@1. Amazon is unaffected (A0-ID; hit@20 criterion moves 1 question).")
    w("")
    hdr = [f"{m} {'H@1' if c == 'h1' else 'hit@20'}" for m in MAPPINGS for c in ("h1", "h20")]
    w("| Scope | Config | Dataset | " + " | ".join(hdr) + " |")
    w("|---|---|---|" + "---|" * len(hdr))
    for scope, cfgs in (("full", ("A4",)), ("test-0.1", ("A4",) + ARMS)):
        for cfg in cfgs:
            for d in NAME_DS:
                cells = []
                for m in MAPPINGS:
                    for c in ("h1", "h20"):
                        e = res["strata"][f"{m}_{c}/{scope}/{cfg}/{d}"]
                        cells.append(f"{e['n_unknown']}: {f(e['unknown'], 1)}")
                w(f"| {scope} | {cfg} | {d} | " + " | ".join(cells) + " |")
    w("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-read raw result files instead of cache.json")
    args = ap.parse_args()
    cp = HERE / "probe_strata_cache.json"
    cache = extract() if args.refresh or not cp.exists() else json.loads(cp.read_text())
    res = analyze(cache)
    (HERE / "probe_strata.json").write_text(json.dumps(res, indent=1))
    (HERE / "probe_strata.md").write_text(report(res))
    print("wrote", HERE / "probe_strata.md")


if __name__ == "__main__":
    main()
